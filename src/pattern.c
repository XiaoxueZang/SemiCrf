#include <ctype.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pattern.h"
#include "tools.h"

// I assume that label is only one char.
// Unit Tested
char *generateLabelPattern(tok_t *toks, uint32_t segStart) {
    char *labelPat = xmalloc(sizeof(char) * (toks->maxOrder) * (toks->maxLabelLen + 1));
    uint32_t index = 0;
    uint32_t pos = segStart;

    for (uint32_t i = 0; i <= toks->maxOrder; ++i) {
        size_t labelLen = strlen(toks->lbl[pos]); // sizeof(toks->lbl[pos]) / sizeof(toks->lbl[pos][0]);
        strcpy(labelPat + index, toks->lbl[pos]);
        *(labelPat + index + 1) = '|';
        index += (uint32_t)labelLen + 1;
        if (pos == 0) {
            break;
        } else {
            pos = toks->segs[pos - 1];
        }
    }
    *(labelPat + index - 1) = '\0';
    // info("inside generateLabelPattern\n");
    // info("pos is %u, %s, labelPat is %s\n", pos, toks->lbl[segStart], labelPat);
    return labelPat;
}

// Unit tested.
labelPat_t *generateLabelPatStruct(char *labelPat) {
    labelPat_t *labelPatStruct = malloc(sizeof(labelPat_t));
    uint32_t size = 20;
    uint32_t segNum = 0;
    uint32_t len = 0;
    for (uint32_t i = 0; i < size; ++i) {
        ++len;
        char a = *(labelPat + i);
        if (a == '\0') {
            break;
        } else if (a == '|') {
            ++segNum;
        }
    }
    labelPatStruct->length = len;
    labelPatStruct->labelPat = labelPat;
    labelPatStruct->order = segNum;
    labelPatStruct->segNum = segNum + 1;
    labelPatStruct->suffixes = malloc(sizeof(char *) * (segNum + 1));
    labelPatStruct->prefixes = malloc(sizeof(char *) * (segNum));
    // char **suffixes = labelPatStruct->suffixes;
    for (uint32_t i = 0; i < segNum; ++i) {
        labelPatStruct->suffixes[i] = malloc(sizeof(char) * len);
        labelPatStruct->prefixes[i] = malloc(sizeof(char) * len);
    }
    labelPatStruct->suffixes[segNum] = malloc(sizeof(char) * len);
    for (uint32_t i = 0; i < len; ++i) {
        char a = *(labelPat + i);
        if ((a == '|') | (a == '\0')) {
            memcpy(labelPatStruct->suffixes[(labelPatStruct->order) - segNum], labelPat, i);
            *(labelPatStruct->suffixes[(labelPatStruct->order) - segNum] + i) = '\0';
            // info("%u suffix is %s.\n", labelPatStruct->order-segNum, labelPatStruct->suffixes[labelPatStruct->order-segNum]);
            --segNum;
        }
    }
    uint32_t i = 0;
    uint32_t index = 0;
    while (labelPat[i] != '\0') {
        if (labelPat[i] == '|') {
            memcpy(labelPatStruct->prefixes[index++], labelPat + i + 1, len - i - 1);
        }
        ++i;
    }
    return labelPatStruct;
}

feature_dat_t *generateObs(rdr_t *rdr, tok_t *tok, uint32_t segStart, uint32_t segEnd, char *labelPat, bool doSemi) {
    uint32_t size = 5000;
    feature_dat_t *featurePack = xmalloc(sizeof(feature_dat_t));
    featurePack->len = 0;
    featurePack->features = xmalloc(sizeof(feature_t *) * size);
    labelPat_t *labelPatStruct = generateLabelPatStruct(labelPat);
    feature_dat_t *partFeat = xmalloc(sizeof(feature_dat_t) * size);
    for (int i = labelPatStruct->order; i >= 0; --i) {
        partFeat = generateFeaturesAt(rdr, tok, segStart, segEnd, labelPatStruct->suffixes[i], doSemi, i);
        if (partFeat == NULL) continue;
        uint32_t arraySize = partFeat->len;
        for (uint32_t j = 0; j < arraySize; ++j) {
            featurePack->features[featurePack->len++] = partFeat->features[j];
            if (featurePack->len == size) {
                size *= 1.4;
                featurePack->features = realloc(featurePack->features, sizeof(feature_t *) * size);
            }
        }
    }
    free(labelPatStruct);
    /*
    for (uint32_t t = 0; t < partFeat->len; ++t) {
        free(partFeat->features[t]);
    }
    free(partFeat->features);
    free(partFeat);
     */
    return featurePack;
}

// Unit tested.
uint64_t getLastLabelId(rdr_t *rdr, const char *p) {
    uint32_t i;
    for (i = 0; i < strlen(p); ++i) {
        if ((p[i] == '|') || (p[i] == '\0')) break;
    }
    char *lastLabel = xmalloc(sizeof(char)*i);
    memcpy(lastLabel, p, i);
    lastLabel[i] = '\0';
    uint64_t id = qrk_str2id(rdr->lbl, lastLabel);
    free(lastLabel);
    return id;
}

uint64_t getLongestIndexId(char *labelPat, qrk_t *qrk) {
    labelPat_t *labelPatStruct = generateLabelPatStruct(labelPat);
    qrk_lock(qrk, true);
    uint64_t id;
    for (int i = labelPatStruct->order; i >= 0; --i) {
        id = qrk_str2id(qrk, labelPatStruct->suffixes[i]);
        if (id != none) return id;
    }
    fatal("No longest suffix index.\n");
    return 0;
}

char *getLongestSuffix(char* labelPat, qrk_t *qrk) {
    uint64_t id;
    labelPat_t *labelPatStruct = generateLabelPatStruct(labelPat);
    for (int i = labelPatStruct->order; i >= 0; --i) {
        id = qrk_str2id(qrk, labelPatStruct->suffixes[i]);
        if (id != none) return labelPatStruct->suffixes[i];
    }
    info("getLongestSuffix. pattern is %s.\n", labelPat);
    fatal("No longest suffix.\n");
    return "";
};

/* pat_comp:
 *   Compile the pattern to a form more suitable to easily apply it on tokens
 *   list during data reading. The given pattern string is interned in the
 *   compiled pattern and will be freed with it, so you don't have to take care
 *   of it and must not modify it after the compilation.
 */
pat_t *pat_comp(char *p) {
    pat_t *pat = NULL;
    // Allocate memory for the compiled pattern, the allocation is based
    // on an over-estimation of the number of required item. As compiled
    // pattern take a neglectible amount of memory, this waste is not
    // important.
    uint32_t mitems = 0;
    for (uint32_t pos = 0; p[pos] != '\0'; pos++)
        if (p[pos] == '%')
            mitems++;
    mitems = mitems * 2 + 1;
    pat = xmalloc(sizeof(pat_t) + sizeof(pat->items[0]) * mitems);
    pat->src = p;
    // Next, we go through the pattern compiling the items as they are
    // found. Commands are parsed and put in a corresponding item, and
    // segment of char not in a command are put in a 's' item.
    uint32_t nitems = 0;
    uint32_t ntoks = 0;
    uint32_t pos = 0;
    while (p[pos] != '\0') {
        pat_item_t *item = &(pat->items[nitems++]);
        item->value = NULL;
        if (p[pos] == '%') {
            // This is a command, so first parse its type and check
            // its a valid one. Next prepare the item.
            const char type = (const char) tolower(p[pos + 1]);
            if (type != 'x' && type != 't' && type != 'm')
                fatal("unknown command type: '%c'", type);
            item->type = type;
            item->caps = (p[pos + 1] != type);
            pos += 2;
            // Next we parse the offset and column and store them in
            // the item.
            const char *at = p + pos;
            uint32_t col;
            int32_t off;
            int nch;
            item->absolute = false;
            if (sscanf(at, "[@%"SCNi32",%"SCNu32"%n", &off, &col, &nch) == 2)
                item->absolute = true;
            else if (sscanf(at, "[%"SCNi32",%"SCNu32"%n", &off, &col, &nch) != 2)
                fatal("invalid pattern: %s", p);
            item->offset = off;
            item->column = col;
            ntoks = max(ntoks, col);
            pos += nch;
            // And parse the end of the argument list, for 'x' there
            // is nothing to read but for 't' and 'm' we have to get
            // read the regexp.
            if (type == 't' || type == 'm') {
                if (p[pos] != ',' && p[pos + 1] != '"')
                    fatal("missing arg in pattern: %s", p);
                const int32_t start = (pos += 2);
                while (p[pos] != '\0') {
                    if (p[pos] == '"')
                        break;
                    if (p[pos] == '\\' && p[pos+1] != '\0')
                        pos++;
                    pos++;
                }
                if (p[pos] != '"')
                    fatal("unended argument: %s", p);
                const int32_t len = pos - start;
                item->value = xmalloc(sizeof(char) * (len + 1));
                memcpy(item->value, p + start, len);
                item->value[len] = '\0';
                pos++;
            }
            // Just check the end of the arg list and loop.
            if (p[pos] != ']')
                fatal("missing end of pattern: %s", p);
            pos++;
        } else {
            // No command here, so build an 's' item with the chars
            // until end of pattern or next command and put it in
            // the list.
            const int32_t start = pos;
            while (p[pos] != '\0' && p[pos] != '%')
                pos++;
            const int32_t len = pos - start;
            item->type  = 's';
            item->caps  = false;
            item->value = xmalloc(sizeof(char) * (len + 1));
            memcpy(item->value, p + start, len);
            item->value[len] = '\0';
        }
    }
    pat->ntoks = ntoks;
    pat->nitems = nitems;
    return pat;
}

/* pat_free:
 *   Free all memory used by a compiled pattern object. Note that this will free
 *   the pointer to the source string given to pat_comp so you must be sure to
 *   not use this pointer again.
 */
void pat_free(pat_t *pat) {
    for (uint32_t it = 0; it < pat->nitems; it++)
        free(pat->items[it].value);
    free(pat->src);
    free(pat);
}
