#ifndef pattern_h
#define pattern_h

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "sequence.h"
#include "reader.h"
#include "model.h"
#include "quark.h"
#include "features.h"
#include "reader.h"

typedef struct labelPat_s labelPat_t;
struct labelPat_s {
    char *labelPat;
    uint32_t order;
    uint32_t length;
    uint32_t segNum;
    char **suffixes;
    char **prefixes;
};

labelPat_t *generateLabelPatStruct(char *labelPat);

char *generateLabelPattern(tok_t *toks, uint32_t segStart);

feature_dat_t *generateObs(rdr_t *rdr, tok_t *tok, uint32_t segStart, uint32_t segEnd, char *labelPat, bool doSemi);

uint64_t getLastLabelId(rdr_t *rdr, const char *p);

uint64_t getLongestIndexId(char *labelPat, qrk_t *qrk);

char *getLongestSuffix(char* labelPat, qrk_t *qrk);

pat_t *pat_comp(char *p);

void pat_free(pat_t *pat);

#endif

