#ifndef reader_h
#define reader_h

#include <stdbool.h>
#include <stdio.h>
#include <stddef.h>

#include "wapiti.h"
#include "pattern.h"
#include "quark.h"
#include "sequence.h"
#include "features.h"

/* rdr_t:
 *   The reader object who hold all informations needed to parse the input file:
 *   the patterns and quark for labels and observations. We keep separate count
 *   for unigrams and bigrams pattern for simpler allocation of sequences. We
 *   also store the expected number of column in the input data to check that
 *   pattern are appliables.
 */
typedef struct list_s list_t;
struct list_s {
    uint32_t key;
    uint32_t val;
    list_t *next;
};


typedef struct rdr_s rdr_t;
struct rdr_s {
    int32_t maxSegment;
    uint64_t nlbl;     // Y Total number of labels
    uint64_t npats;      //  P   Total number of patterns
    uint64_t nfeats;  //      Number of unigram and bigram patterns
    // uint64_t ntoks;      //      Expected number of tokens in input
    uint64_t nforwardStateMap;
    uint64_t nbackwardStateMap;
    // pat_t    **pats;       // [P]  List of precompiled patterns
    qrk_t *lbl;        //      Labels database
    qrk_t *obs;        //      Observation database
    qrk_t *pats;       //      patterns database
    qrk_t *featList;   //      featureList
    qrk_t *forwardStateMap;
    qrk_t *backwardStateMap;
    // uint64_t **featMap;     //      featureMap
    int32_t *maxMemory;  //      maxMemory: maximum length of continuous labels. the index is the id of (qrk_t *)lbl.
};

rdr_t *rdr_new(uint32_t maxSegment);

void updateReader(tok_t *tok, rdr_t *rdr);

void generateForwardStateMap(rdr_t *reader);

void generateBackwardStateMap(rdr_t *reader);

void rdr_free(rdr_t *rdr);

void rdr_freeraw(raw_t *raw);

void rdr_freeseq(seq_t *seq);

void rdr_freedat(dat_t *dat);

// void rdr_loadpat(rdr_t *rdr, FILE *file);

raw_t *rdr_readraw(rdr_t *rdr, FILE *file);

tok_t *rdr_raw2tok(rdr_t *rdr, const raw_t *raw, bool lbl);

tok_t *rdr_readtok(rdr_t *rdr, FILE *file, bool lbl);

dat_t *rdr_readdat(rdr_t *rdr, FILE *file, bool lbl);

void rdr_load(rdr_t *rdr, FILE *file);

void rdr_save(const rdr_t *rdr, FILE *file);

char *rdr_readline(FILE *file);

#endif

