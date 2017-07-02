#include <ctype.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "reader.h"
#include "pattern.h"
#include "sequence.h"
#include "tools.h"
#include "features.h"


char *concat(const char *s1, const char *s2) {
    char *result = xmalloc((strlen(s1) + strlen(s2) + 1) * sizeof(char));//+1 for the zero-terminator
    //in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    // info("concat s1 = %s, s2 = %s, res = %s.", s1, s2, result);
    return result;
}

void updateMaxMemory(tok_t *tok, rdr_t *reader) {
    uint32_t segStart = 0;
    uint64_t id;
    if (reader->doSemi == true) {
        while (segStart < tok->len) {
            // segEnd = tok->sege[segStart];
            char *label = tok->lbl[segStart];
            id = qrk_str2id(reader->lbl, label);

            if ((reader->maxMemory[id]) < (int) (tok->segl[segStart] + 1)) {
                reader->maxMemory[id] = tok->segl[segStart] + 1;
                reader->maxSegment = max(reader->maxSegment, reader->maxMemory[id]);
            }
            segStart = tok->sege[segStart] + 1;
        }
    } else {
        for (uint32_t i = 0; i < tok->len; ++i) {
            tok->segs[i] = tok->sege[i] = i;
            tok->segl[i] = 1;
            id = qrk_str2id(reader->lbl, tok->lbl[i]);
            reader->maxMemory[id] = 1;
        }
    }
}

void putIntoDatabase(char *newObs, char *labelPat, rdr_t *database) {
    // char *newObs = concat(head, tail);
    qrk_str2id(database->obs, newObs);
    qrk_str2id(database->pats, labelPat);
    char *newFeats = concat(concat(labelPat, "_"), newObs);
    qrk_str2id(database->featList, newFeats);
    free(newFeats);
    return;
}

feature_t *constructFeature(char *obs, char *labelPat) {
    feature_t *feat = xmalloc(sizeof(feature_t));
    feat->pats = labelPat;
    feat->obs = obs;
    return feat;
}

/* pat_exec:
 *   Execute a compiled pattern at position 'at' in the given tokens sequences
 *   in order to produce an observation string. The string is returned as a
 *   newly allocated memory block and the caller is responsible to free it when
 *   not needed anymore.
 */
char *pat_exec(const pat_t *pat, const tok_t *tok, uint32_t at) {
    static char *bval[] = {"_x-1", "_x-2", "_x-3", "_x-4", "_x-#"};
    static char *eval[] = {"_x+1", "_x+2", "_x+3", "_x+4", "_x+#"};
    const uint32_t T = tok->len;
    // Prepare the buffer who will hold the result
    uint32_t size = 16, pos = 0;
    char *buffer = xmalloc(sizeof(char) * size);
    // And loop over the compiled items
    for (uint32_t it = 0; it < pat->nitems; it++) {
        const pat_item_t *item = &(pat->items[it]);
        char *value = NULL;
        uint32_t len = 0;
        // First, if needed, we retrieve the token at the referenced
        // position in the sequence. We store it in value and let the
        // command handler do what it need with it.
        if (item->type != 's') {
            int pos = item->offset;
            if (item->absolute) {
                if (item->offset < 0)
                    pos += T;
                else
                    pos--;
            } else {
                pos += at;
            }
            uint32_t col = item->column;
            if (pos < 0)
                value = bval[min(-pos - 1, 4)];
            else if (pos >= (int32_t)T)
                value = eval[min( pos - (int32_t)T, 4)];
            else if (col >= tok->cnts[pos])
                fatal("missing tokens, cannot apply pattern");
            else
                value = tok->toks[pos][col];
        }
        // Next, we handle the command, 's' and 'x' are very simple but
        // 't' and 'm' require us to call the regexp matcher.
        if (item->type == 's') {
            value = item->value;
            len = strlen(value);
        } else if (item->type == 'x') {
            len = strlen(value);
        }
        // And we add it to the buffer, growing it if needed. If the
        // user requested it, we also remove caps from the string.
        if (pos + len >= size - 1) {
            while (pos + len >= size - 1)
                size = size * 1.4;
            buffer = xrealloc(buffer, sizeof(char) * size);
        }
        memcpy(buffer + pos, value, len);
        if (item->caps)
            for (uint32_t i = pos; i < pos + len; i++)
                buffer[i] = tolower(buffer[i]);
        pos += len;
    }
    // Adjust the result and return it.
    buffer[pos++] = '\0';
    buffer = xrealloc(buffer, sizeof(char) * pos);
    return buffer;
}

feature_dat_t *generateFeaturesAt(rdr_t *rdr, tok_t *tok, uint32_t segStart, uint32_t segEnd, char *labelPat, bool doSemi, int i) {
    uint32_t size = 500; // I consider that this is the maximum feature number.
    feature_dat_t *allFeats = xmalloc(sizeof(feature_dat_t));
    allFeats->len = 0;
    allFeats->features = xmalloc(sizeof(feature_t *) * size);
    for (uint32_t x = 0; (x < rdr->ntpl[i]) && (i <= (int) segStart); x++) {
        // Get the observation and map it to an identifier
        char *obs = pat_exec(rdr->tpl[i][x], tok, segStart); // read one pattern inside rdr->pats and return it. ex. "*1:_x-1"
        allFeats->features[allFeats->len++] = constructFeature(obs, labelPat);
        if (allFeats->len == size) {
            size = (uint32_t) (size * 1.4);
            allFeats = xrealloc(allFeats, sizeof(feature_dat_t) * size);
        }
    }
    return allFeats;
}

