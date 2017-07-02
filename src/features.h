#ifndef WAPITI_FEATURES_H
#define WAPITI_FEATURES_H

#include <ctype.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <stddef.h>

#include "reader.h"
#include "sequence.h"

// typedef struct rdr_s rdr_t;
typedef struct feature_s feature_t;
typedef struct feature_dat_s feature_dat_t;
struct feature_s {
    char *obs;
    char *pats;
    uint32_t val;
};

struct feature_dat_s {
    uint32_t len;
    feature_t **features;
};

char *concat(const char *s1, const char *s2);

void updateMaxMemory(tok_t *tok, rdr_t *reader);

void putIntoDatabase(char *obs, char *labelPat, rdr_t *database);

feature_dat_t *generateFeaturesAt(rdr_t *rdr, tok_t *tok, uint32_t segStart, uint32_t segEnd, char *labelPat, bool doSemi, int level);

char *pat_exec(const pat_t *pat, const tok_t *tok, uint32_t at);

#endif //WAPITI_FEATURES_H
