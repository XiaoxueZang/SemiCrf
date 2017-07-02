#ifndef reader_h
#define reader_h

#include <stdbool.h>
#include <stdio.h>
#include <stddef.h>

#include "quark.h"
#include "sequence.h"

#define MAX_DEPTH 4

typedef struct pat_s pat_t;
typedef struct pat_item_s pat_item_t;
struct pat_item_s {
    char      type;
    bool      caps;
    char     *value;
    bool      absolute;
    int32_t   offset;
    uint32_t  column;
};

struct pat_s {
    char     *src;
    uint32_t  ntoks;
    uint32_t  nitems;
    pat_item_t items[];
};

typedef struct rdr_s rdr_t;
/* rdr_t:
 *   The reader object who hold all informations needed to parse the input file:
 *   the patterns and quark for labels and observations. We keep separate count
 *   for unigrams and bigrams pattern for simpler allocation of sequences. We
 *   also store the expected number of column in the input data to check that
 *   pattern are appliables.
 */
struct rdr_s {
    bool doSemi;
    int32_t maxSegment;
    uint64_t nlbl;     // Y Total number of labels
    uint64_t npats;      //  P   Total number of patterns
    uint64_t nforwardStateMap;
    uint32_t ntpls;      //  P   Total number of patterns
    uint32_t ntpl[MAX_DEPTH];
    // uint32_t nz, no, nt, nth;
    qrk_t *lbl;        //      Labels database
    qrk_t *obs;        //      Observation database
    qrk_t *pats;       //      patterns database
    qrk_t *featList;   //      featureList
    qrk_t *forwardStateMap;
    qrk_t *backwardStateMap;
    int32_t *maxMemory;  //      maxMemory: maximum length of continuous labels. the index is the id of (qrk_t *)lbl.
    pat_t    **tpl[];       // [P]  List of precompiled patterns
};

rdr_t *rdr_new(bool doSemi);

void updateReader(tok_t *tok, rdr_t *rdr);

void generateForwardStateMap(rdr_t *reader);

void generateBackwardStateMap(rdr_t *reader);

void rdr_free(rdr_t *rdr);

void rdr_freeraw(raw_t *raw);

void rdr_freedat(dat_t *dat);

void idmap_free(id_map_t *id);

void rdr_freetok(tok_t *tok, bool lbl);

raw_t *rdr_readraw(FILE *file);

tok_t *rdr_raw2tok(rdr_t *rdr, const raw_t *raw, bool lbl, bool doTrain);

tok_t *rdr_readtok(rdr_t *rdr, FILE *file, bool lbl, bool doTrain);

dat_t *rdr_readdat(rdr_t *rdr, FILE *file, bool lbl, bool doTrain);

void rdr_loadpat(rdr_t *rdr, FILE *file);

void rdr_load(rdr_t *rdr, FILE *file);

void rdr_save(const rdr_t *rdr, FILE *file);

char *rdr_readline(FILE *file);

#endif

