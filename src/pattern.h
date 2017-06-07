#ifndef pattern_h
#define pattern_h

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "sequence.h"
#include "reader.h"
#include "model.h"

typedef struct qrk_s qrk_t;
typedef struct labelPat_s labelPat_t;
typedef struct feature_dat_s feature_dat_t;
typedef struct rdr_s rdr_t;
struct labelPat_s {
    char *labelPat;
    uint32_t order;
    uint32_t length;
    uint32_t segNum;
    char **suffixes;
    char **prefixes;
};

// labelPat_t *findStructOrBuild(rdr_t *reader, char *labelPat);
labelPat_t *generateLabelPatStruct(char *labelPat);

char *generateLabelPattern(tok_t *toks, uint32_t segStart, uint32_t segEnd);

feature_dat_t *generateObs(tok_t *tok, rdr_t *reader, uint32_t segStart, uint32_t segEnd, char *labelPat);

uint64_t getLastLabelId(rdr_t *rdr, const char *p);

uint64_t getLongestIndexId(char *labelPat, qrk_t *qrk);

char *getLongestSuffix(char* labelPat, qrk_t *qrk);

#endif

