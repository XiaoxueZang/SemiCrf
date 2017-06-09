#include <ctype.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "wapiti.h"
#include "pattern.h"
#include "quark.h"
#include "reader.h"
#include "sequence.h"
#include "tools.h"
#include "features.h"
#include "vmath.h"

#define MAX_LABEL_COUNT 10

/*******************************************************************************
 * Datafile reader
 *
 *   And now come the data file reader which use the previous module to parse
 *   the input data in order to produce seq_t objects representing interned
 *   sequences.
 *
 *   This is where the sequence will go through the tree steps to build seq_t
 *   objects used internally. There is two way do do this. First the simpler is
 *   to use the rdr_readseq function which directly read a sequence from a file
 *   and convert it to a seq_t object transparently. This is how the training
 *   and development data are loaded.
 *   The second way consist of read a raw sequence with rdr_readraw and next
 *   converting it to a seq_t object with rdr_raw2seq. This allow the caller to
 *   keep the raw sequence and is used by the tagger to produce a clean output.
 *
 *   There is no public interface to the tok_t object as it is intended only for
 *   internal use in the reader as an intermediate step to apply patterns.
 ******************************************************************************/

/* rdr_new:
 *   Create a new empty reader object. If no patterns are loaded before you
 *   start using the reader the input data are assumed to be already prepared
 *   list of features. They must either start with a prefix 'u', 'b', or '*', or
 *   you must set autouni to true in order to automatically add a 'u' prefix.*
*/
rdr_t *rdr_new(bool doSemi) {
    rdr_t *rdr = xmalloc(sizeof(rdr_t));
    rdr->doSemi = doSemi;
    if (doSemi == true)
        rdr->maxSegment = 1;
    else
        rdr->maxSegment = -1;
    rdr->npats = rdr->nfeats = rdr->npats = 0;
    rdr->nlbl = rdr->nforwardStateMap = rdr->nbackwardStateMap = 0;
    rdr->lbl = qrk_new();
    rdr->obs = qrk_new();
    rdr->pats = qrk_new();
    rdr->featList = qrk_new();
    rdr->backwardStateMap = qrk_new();
    rdr->forwardStateMap = qrk_new();
    rdr->maxMemory = xmalloc(sizeof(uint32_t) * MAX_LABEL_COUNT);
    for (int i = 0; i < MAX_LABEL_COUNT; ++i) {
        rdr->maxMemory[i] = 0;
    }
    return rdr;
}

/* rdr_free:
 *   Free all memory used by a reader object including the quark database, so
 *   any string returned by them must not be used after this call.
 */
void rdr_free(rdr_t *rdr) {
    free(rdr->maxMemory);
    qrk_free(rdr->pats);
    qrk_free(rdr->backwardStateMap);
    qrk_free(rdr->forwardStateMap);
    qrk_free(rdr->lbl);
    qrk_free(rdr->obs);
    free(rdr);
}

/* rdr_freeraw:
 *   Free all memory used by a raw_t object.
 */
void rdr_freeraw(raw_t *raw) {
    for (uint32_t t = 0; t < raw->len; t++)
        free(raw->lines[t]);
    free(raw);
}

void idmap_free(id_map_t *id) {
    free(id->ids);
    free(id);
}

/* rdr_freeseq:
 *   Free all memory used by a seq_t object.
 */
void rdr_freetok(tok_t *tok, bool lbl) {
    // free(tok->cur_t[0]);
    free(tok->cur_t);
    // free(tok->lbl[0]);
    for (uint32_t t = 0; t < tok->len; t++) {
        if (tok->cnts[t] == 0)
            continue;
        free(tok->toks[t][0]);
        free(tok->toks[t]);
    }
    free(tok->cnts);
    if (lbl == true) {
        // free(tok->lbl[0]);
        free(tok->lbl);
    }
    free(tok->sege);
    free(tok->segs);
    free(tok->segl);
    idmap_free(tok->observationMapjp);
    xvm_free(tok->empiricalScore);
    free(tok);
}


/* rdr_freedat:
 *   Free all memory used by a dat_t object.
 */
void rdr_freedat(dat_t *dat) {
    for (uint32_t i = 0; i < dat->nseq; i++)
        rdr_freetok(dat->tok[i], dat->lbl);
        free(dat->tok);
    free(dat);
}

/* rdr_readline:
 *   Read an input line from <file>. The line can be of any size limited only by
 *   available memory, a buffer large enough is allocated and returned. The
 *   caller is responsible to free it. On end-of-file, NULL is returned.
 */
char *rdr_readline(FILE *file) {
    if (feof(file))
        return NULL;
    // Initialize the buffer
    uint32_t len = 0, size = 16;
    char *buffer = xmalloc(size);
    // We read the line chunk by chunk until end of line, file or error
    while (!feof(file)) {
        if (fgets(buffer + len, size - len, file) == NULL) {
            // On NULL return there is two possible cases, either an
            // error or the end of file
            if (ferror(file))
                pfatal("cannot read from file");
            // On end of file, we must check if we have already read
            // some data or not
            if (len == 0) {
                free(buffer);
                return NULL;
            }
            break;
        }
        // Check for end of line, if this is not the case enlarge the
        // buffer and go read more data
        len += strlen(buffer + len);
        if (len == size - 1 && buffer[len - 1] != '\n') {
            size = (uint32_t) size * 1.4;
            buffer = xrealloc(buffer, size);
            continue;
        }
        break;
    }
    // At this point empty line should have already catched so we just
    // remove the end of line if present and resize the buffer to fit the
    // data
    if (buffer[len - 1] == '\n')
        buffer[--len] = '\0';
    return xrealloc(buffer, len + 1);
}

/* rdr_loadpat:
 *   Load and compile patterns from given file and store them in the reader. As
 *   we compile patterns, syntax errors in them will be raised at this time.
 */

/* rdr_readraw:
 *   Read a raw sequence from given file: a set of lines terminated by end of
 *   file or by an empty line. Return NULL if file end was reached before any
 *   sequence was read.
 */
raw_t *rdr_readraw(rdr_t *rdr, FILE *file) {
    if (feof(file))
        return NULL;
    // Prepare the raw sequence object
    uint32_t size = 32, cnt = 0;
    raw_t *raw = xmalloc(sizeof(raw_t) + sizeof(char *) * size);
    // And read the next sequence in the file, this will skip any blank line
    // before reading the sequence stoping at end of file or on a new blank
    // line.
    while (!feof(file)) {
        char *line = rdr_readline(file);
        if (line == NULL)
            break;
        // Check for empty line marking the end of the current sequence
        size_t len = strlen(line);
        while (len != 0 && isspace(line[len - 1]))
            len--;
        if (len == 0) {
            free(line);
            // Special case when no line was already read, we try
            // again. This allow multiple blank lines beetwen
            // sequences.
            if (cnt == 0)
                continue;
            break;
        }
        // Next, grow the buffer if needed and add the new line in it
        if (size == cnt) {
            size *= 1.4;
            raw = xrealloc(raw, sizeof(raw_t)
                                + sizeof(char *) * size);
        }
        raw->lines[cnt++] = line; // raw -> lines is ["a 1", "b 0", ...] is one sequence.
    }
    // If no lines was read, we just free allocated memory and return NULL
    // to signal the end of file to the caller. Else, we adjust the object
    // size and return it.
    if (cnt == 0) {
        free(raw);
        return NULL;
    }
    raw = xrealloc(raw, sizeof(raw_t) + sizeof(char *) * cnt);
    raw->len = cnt;
    return raw;
}

/* rdr_mapobs:
 *   Map an observation to its identifier, automatically adding a 'u' prefix in
 *   'autouni' mode.

static uint64_t rdr_mapobs(rdr_t *rdr, const char *str) {
	if (!rdr->autouni)
		return qrk_str2id(rdr->obs, str);
	char tmp[strlen(str) + 2];
	tmp[0] = 'u';
	strcpy(tmp + 1, str);
	return qrk_str2id(rdr->obs, tmp);
}
*/

/* rdr_raw2seq:
 *   Convert a raw sequence to a tok_t object suitable for training or
 *   labelling. If lbl is true, the last column is assumed to be a label and
 *   interned also.
 */
// sequence ex:
// angle 14 1
// is    13 0
// ...
// (one sequence of the input file)
tok_t *rdr_raw2tok(rdr_t *rdr, const raw_t *raw, bool lbl) {
    const uint32_t T = raw->len;
    // Allocate the tok_t object, the label array is allocated only if they
    // are requested by the user.
    tok_t *tok = xmalloc(sizeof(tok_t) + T * sizeof(char **));
    tok->cnts = xmalloc(sizeof(uint32_t) * T);
    tok->lbl = NULL;
    tok->segs = xmalloc(sizeof(uint32_t) * T);
    tok->sege = xmalloc(sizeof(uint32_t) * T);
    tok->segl = xmalloc(sizeof(uint32_t) * T);
    tok->cur_t = xmalloc(sizeof(char *) * T);
    tok->maxOrder = 3;
    tok->maxLabelLen = 0;
    tok->observationMapjp = NULL;
    tok->empiricalScore = NULL;
    if (lbl == true)
        tok->lbl = xmalloc(sizeof(char *) * T);
    // We now take the raw sequence line by line and split them in list of
    // tokens. To reduce memory fragmentation, the raw line is copied and
    // his reference is kept by the first tokens, next tokens are pointer to
    // this copy.
    for (uint32_t t = 0; t < T; t++) {
        // Get a copy of the raw line skiping leading space characters
        const char *src = raw->lines[t]; // the t line of a sequence ex: a b d g 1
        while (isspace(*src))
            src++;
        char *line = xstrdup(src);
        // Split one line in tokens. Structure info into toks (char **) and then put the info in toks into tok.
        // The final image of toks:
        // [['a', '1'], ['b', '0'], ..., ]
        char *toks[strlen(line) / 2 + 1];
        uint32_t cnt = 0;
        while (*line != '\0') {
            toks[cnt++] = line;   // toks[0] = &(a), the address of that token (a)
            while (*line != '\0' && !isspace(*line))
                line++;
            if (*line == '\0')
                break;
            *line++ = '\0';
            while (*line != '\0' && isspace(*line))
                line++;
        }

        // If user specified that data are labelled, move the last token
        // to the label array.
        if (lbl == true) {
            tok->lbl[t] = toks[cnt - 1];
            tok->maxLabelLen = max(tok->maxLabelLen, strlen(toks[cnt-1]));
            // put the label into label database;
            qrk_str2id(rdr->lbl, tok->lbl[t]);
            cnt--;
        }
        tok->cur_t[t] = toks[4]; // the current word;
        // And put the remaining tokens in the tok_t object

        // uint32_t i = t;
        if (t > 0 && (strcmp(tok->lbl[t], tok->lbl[t - 1]) == 0)) {
            tok->segs[t] = tok->segs[t - 1];
        } else {
            tok->segs[t] = t;
        }
        tok->cnts[t] = cnt;
        tok->toks[t] = xmalloc(sizeof(char *) * cnt);
        memcpy(tok->toks[t], toks, sizeof(char *) * cnt);
    }
    tok->len = T;
    // set segmentend and segment length.
    for (int i = T - 1; i >= 0; --i) {
        if (i < T - 1 && (strcmp(tok->lbl[i], tok->lbl[i + 1]) == 0)) {
            tok->sege[i] = tok->sege[i + 1];
        } else {
            tok->sege[i] = (uint32_t)i;
        }
        tok->segl[i] = tok->sege[i] - tok->segs[i];
    }
    if (lbl == true) {
        updateReader(tok, rdr);
    }
    /*
    seq_t *seq = NULL;
    // the following part should be removed and be executed after reading all the sequences.
    // Convert the tok_t to a seq_t
    seq = rdr_rawtok2seq(rdr, tok);

    // Before returning the sequence, we have to free the tok_t
    for (uint32_t t = 0; t < T; t++) {
        if (tok->cnts[t] == 0)
            continue;
        free(tok->toks[t][0]);
        free(tok->toks[t]);
    }
    free(tok->cnts);
    if (lbl == true)
        free(tok->lbl);
    free(tok);
     */
    return tok;
}

/* rdr_readseq:
 *   Simple wrapper around rdr_readraw and rdr_raw2seq to directly read a
 *   sequence as a seq_t object from file. This take care of all the process
 *   and correctly free temporary data. If lbl is true the sequence is assumed
 *   to be labeled.
 *   Return NULL if end of file occure before anything as been read.
 */
tok_t *rdr_readtok(rdr_t *rdr, FILE *file, bool lbl) {
    raw_t *raw = rdr_readraw(rdr, file); // return one seq (one paragraph).lines[T] is the contents of line T of seq
    // info("inside rdr_readseq: raw_t length is %d\n", raw->len);
    if (raw == NULL)
        return NULL;
    tok_t *tok = rdr_raw2tok(rdr, raw, lbl);
    rdr_freeraw(raw);
    return tok;
}

/* rdr_readdat:
 *   Read a full dataset at once and return it as a dat_t object. This function
 *   take and interpret his parameters like the single sequence reading
 *   function.
 */
dat_t *rdr_readdat(rdr_t *rdr, FILE *file, bool lbl) {
    uint32_t size = 2000;
    dat_t *dat = xmalloc(sizeof(dat_t));
    dat->nseq = 0;
    dat->mlen = 0;
    dat->lbl = lbl;
    dat->tok = xmalloc(sizeof(tok_t *) * size);
    // Load sequences
    while (!feof(file)) {
        // Read the next sequence
        tok_t *seq = rdr_readtok(rdr, file, lbl);
        if (seq == NULL)
            break;
        // Grow the buffer if needed
        if (dat->nseq == size) {
            size *= 1.4;
            dat->tok = xrealloc(dat->tok, sizeof(seq_t *) * size);
        }
        // And store the sequence
        dat->tok[dat->nseq++] = seq;
        dat->mlen = max(dat->mlen, seq->len);
        if (dat->nseq % 1000 == 0)
            info("%7"PRIu32" sequences loaded\n", dat->nseq);
    }
    // If no sequence readed, cleanup and repport
    if (dat->nseq == 0) {
        free(dat->tok);
        free(dat);
        return NULL;
    }
    // Adjust the dataset size and return
    if (size > dat->nseq)
        dat->tok = xrealloc(dat->tok, sizeof(seq_t *) * dat->nseq);
    if (lbl == true) {
        rdr->nlbl = qrk_count(rdr->lbl);
        rdr->npats = qrk_count(rdr->pats);
        generateForwardStateMap(rdr);
        generateBackwardStateMap(rdr);
    }
    return dat;
}

void updateReader(tok_t *tok, rdr_t *rdr) {
    updateMaxMemory(tok, rdr);
    uint32_t segStart = 0;
    uint32_t segEnd = 0;
    // put an empty feature inside feature database. It is for the following gradient computatinon.
    qrk_str2id(rdr->featList, "");
    while (segStart < tok->len) {
        segEnd = tok->sege[segStart];
        char *labelPat = generateLabelPattern(tok, segStart, segEnd);

        feature_dat_t *features = generateObs(tok, rdr, segStart, segEnd, labelPat);
        for (uint32_t id = 0; id < features->len; ++id) {
            putIntoDatabase(features->features[id]->obs, features->features[id]->pats, rdr);
        }
        segStart = segEnd + 1;
    }
}


void generateForwardStateMap(rdr_t *reader) {
    qrk_t *forwardStateMap = reader->forwardStateMap;
    qrk_str2id(forwardStateMap, "");
    char *str = xmalloc(sizeof(char) * (2 * reader->maxSegment - 1));
    uint64_t nlbl = reader->nlbl;
    for (uint64_t id = 0; id < nlbl; ++id) {
        strcpy(str, qrk_id2str(reader->lbl, id));
        qrk_str2id(forwardStateMap, str);
        // info("the forwardstatemap %s\n", str);
    }
    for (uint64_t id = 0; id < reader->npats; ++id) {
        strcpy(str, qrk_id2str(reader->pats, id));
        labelPat_t *strStruct = generateLabelPatStruct(str);
        uint32_t size = strStruct->segNum - 1;
        for (int i = 0; i < size; ++i) {
            // info("the forwardstatemap %s\n", strStruct->prefixes[i]);
            qrk_str2id(forwardStateMap, strStruct->prefixes[i]);
        }
    }
    reader->nforwardStateMap = qrk_count(forwardStateMap);
    qrk_lock(forwardStateMap, true);
    return;
}

void generateBackwardStateMap(rdr_t *reader) {
    uint32_t size = 50;
    int lastLabel;
    char *p = xmalloc(sizeof(char) * size);
    for (uint64_t id = 0; id < reader->nforwardStateMap; ++id) {
        strcpy(p, qrk_id2str(reader->forwardStateMap, id));
        lastLabel = (strcmp(p, "") == 0) ? -1 : (int) getLastLabelId(reader, p);

        for (uint64_t yid = 0; yid < reader->nlbl; ++yid) {
            if ((yid != lastLabel) || (reader->maxSegment == 1)) {
                char *head = concat(qrk_id2str(reader->lbl, yid), "|");
                const char *py = (strcmp(p, "") == 0) ? qrk_id2str(reader->lbl, yid) : concat(head, p);
                qrk_str2id(reader->backwardStateMap, py);
                // info("the backwardstatemap %s\n", py);
            }
        }
    }
    qrk_lock(reader->backwardStateMap, true);
    reader->nbackwardStateMap = qrk_count(reader->backwardStateMap);
    return;
}

/* rdr_load:
 *   Read from the given file a reader saved previously with rdr_save. The given
 *   reader must be empty, comming fresh from rdr_new. Be carefull that this
 *   function performs almost no checks on the input data, so if you modify the
 *   reader and make a mistake, it will probably result in a crash.
*/
void rdr_load(rdr_t *rdr, FILE *file) {
    const char *err = "broken file, invalid reader format";
    // int autouni = rdr->autouni;
    fpos_t pos;
    fgetpos(file, &pos);

    if (fscanf(file, "#rdr#%d\n", &rdr->maxSegment) != 1)
        fatal(err);
    qrk_load(rdr->lbl, file);
    qrk_load(rdr->obs, file);
    qrk_load(rdr->pats, file);
    qrk_load(rdr->featList, file);
    qrk_load(rdr->forwardStateMap, file);
    qrk_load(rdr->backwardStateMap, file);
}


/* rdr_save:
 *   Save the reader to the given file so it can be loaded back. The save format
 *   is plain text and portable accros computers.
*/
void rdr_save(const rdr_t *rdr, FILE *file) {
    if (fprintf(file, "#rdr#%d\n", rdr->maxSegment) < 0)
        pfatal("cannot write to file");
    qrk_save(rdr->lbl, file);
    qrk_save(rdr->obs, file);
    qrk_save(rdr->pats, file);
    qrk_save(rdr->featList, file);
    qrk_save(rdr->forwardStateMap, file);
    qrk_save(rdr->backwardStateMap, file);
}

