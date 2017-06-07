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
    return result;
}

void updateMaxMemory(tok_t *tok, rdr_t *reader) {
    uint32_t segStart = 0;
    uint64_t id;
    if (reader->maxSegment == 0) {
        while (segStart < tok->len) {
            // segEnd = tok->sege[segStart];
            char *label = tok->lbl[segStart];
            id = qrk_str2id(reader->lbl, label);

            if ((reader->maxMemory[id]) < (tok->segl[segStart] + 1)) {
                reader->maxMemory[id] = tok->segl[segStart] + 1;
                reader->maxSegment = max(reader->maxSegment, reader->maxMemory[id]);
            }
            segStart = tok->sege[segStart] + 1;
        }
    } else if (reader->maxSegment == 1) {
        for (uint32_t i = 0; i < tok->len; ++i) {
            tok->segs[i] = tok->sege[i] = 0;
            tok->segl[i] = 1;
            id = qrk_str2id(reader->lbl, tok->lbl[i]);
            reader->maxMemory[id] = 1;
        }
    } else {
        fatal("Set maxSegment = -1 for semi-CRF and maxSegment = 1 for CRF.");
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

feature_dat_t* generateCrfFeaturesAt(tok_t *tok, uint32_t segStart, uint32_t segEnd, char *labelPat) {
    uint32_t size = 500; // I consider that this is the maximum feature number.
    feature_dat_t* allFeats = xmalloc(sizeof(feature_dat_t));
    allFeats->len = 0;
    allFeats->features = xmalloc(sizeof(feature_t *) * size);
    // the length of featureHead is 39.
    static char *featHead[] = {"SENLEN.", "SENWORDCNT.", "LEFTDIST.", "RIGHTDIST.", "CURWORD.", "PREWORD.", "PPREWORD.",
                        "NEXTWORD.", "NNEXTWORD.", "CURPOS.", "PREPOS.", "PPREPOS.", "NEXTPOS.", "NNEXTPOS.",
                        "CURWLEN.", "PREWLEN.", "PPREWLEN.", "NEXTWLEN.", "NNEXTWLEN.", "0FLOOR.", "0FLOORDIST.",
                        "0CEIL.", "0CEILDIST.", "1FLOOR.", "1FLOORDIST.", "1CEIL.", "1CEILDIST.", "2FLOOR.",
                        "2FLOORDIST.", "2CEIL.", "2CEILDIST.", "3FLOOR.", "3FLOORDIST.", "3CEIL.", "3CEILDIST.",
                        "4FLOOR.", "4FLOORDIST.", "4CEIL.", "4CEILDIST."};
    for (uint32_t i = 0; i <= 2; ++i) {
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[i], tok->toks[segStart][i]), labelPat);
    }
    allFeats->features[allFeats->len++] = constructFeature(concat(featHead[3], tok->toks[segEnd][3]), labelPat);
    for (uint32_t i = segStart; i <= segEnd; ++i) {
        for (uint32_t j = 4; j <= 18; ++j) {
            allFeats->features[allFeats->len++] = constructFeature(concat(featHead[j], tok->toks[i][j]), labelPat);
        }
    }
    if (strcmp(tok->toks[segStart][19], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[19], tok->toks[segStart][19]), labelPat);
    if (strcmp(tok->toks[segStart][20], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[20], tok->toks[segStart][20]), labelPat);
    if (strcmp(tok->toks[segEnd][21], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[21], tok->toks[segEnd][21]), labelPat);
    if (strcmp(tok->toks[segEnd][22], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[22], tok->toks[segEnd][22]), labelPat);

    if (strcmp(tok->toks[segStart][23], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[23], tok->toks[segStart][23]), labelPat);
    if (strcmp(tok->toks[segStart][24], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[24], tok->toks[segStart][24]), labelPat);
    if (strcmp(tok->toks[segEnd][25], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[25], tok->toks[segEnd][25]), labelPat);
    if (strcmp(tok->toks[segEnd][26], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[26], tok->toks[segEnd][26]), labelPat);

    if (strcmp(tok->toks[segStart][27], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[27], tok->toks[segStart][27]), labelPat);
    if (strcmp(tok->toks[segStart][28], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[28], tok->toks[segStart][28]), labelPat);
    if (strcmp(tok->toks[segEnd][29], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[29], tok->toks[segEnd][29]), labelPat);
    if (strcmp(tok->toks[segEnd][30], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[30], tok->toks[segEnd][30]), labelPat);
    if (strcmp(tok->toks[segStart][31], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[23], tok->toks[segStart][23]), labelPat);
    if (strcmp(tok->toks[segStart][32], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[24], tok->toks[segStart][24]), labelPat);
    if (strcmp(tok->toks[segEnd][33], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[25], tok->toks[segEnd][25]), labelPat);
    if (strcmp(tok->toks[segEnd][34], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[26], tok->toks[segEnd][26]), labelPat);

    if (strcmp(tok->toks[segStart][35], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[35], tok->toks[segStart][35]), labelPat);
    if (strcmp(tok->toks[segStart][36], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[36], tok->toks[segStart][36]), labelPat);
    if (strcmp(tok->toks[segEnd][37], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[37], tok->toks[segEnd][37]), labelPat);
    if (strcmp(tok->toks[segEnd][38], "-1"))
        allFeats->features[allFeats->len++] = constructFeature(concat(featHead[38], tok->toks[segEnd][38]), labelPat);
    // free(featHead);
    return allFeats;
}

feature_dat_t *generateFirstOrderFeaturesAt(tok_t *tok, uint32_t segStart, uint32_t segEnd, char *labelPat) {
    uint32_t size = 100; // I consider that this is the maximum feature number.
    feature_dat_t* allFeats = xmalloc(sizeof(feature_dat_t));
    allFeats->len = 0;
    allFeats->features = xmalloc(sizeof(feature_t *) * size);
    if (segStart > 0) {
        allFeats->features[allFeats->len++] = constructFeature("1E", labelPat);
        for (uint32_t i = segStart; i <= segEnd; ++i) {
            allFeats->features[allFeats->len++] = constructFeature(concat("E1W.", tok->toks[i][4]), labelPat);
        }
    }
    return allFeats;
}

feature_dat_t *generateSecondOrderFeaturesAt(tok_t *tok, uint32_t segStart, uint32_t segEnd, char *labelPat) {
    uint32_t size = 100; // I consider that this is the maximum feature number.
    feature_dat_t* allFeats = xmalloc(sizeof(feature_dat_t));
    allFeats->len = 0;
    allFeats->features = xmalloc(sizeof(feature_t *) * size);
    if (segStart > 1) {
        allFeats->features[allFeats->len++] = constructFeature("2E", labelPat);
        for (uint32_t i = segStart; i <= segEnd; ++i) {
            allFeats->features[allFeats->len++] = constructFeature(concat("E2W.", tok->toks[i][4]), labelPat);
        }
    }
    return allFeats;
}

feature_dat_t *generateThirdOrderFeaturesAt(tok_t *tok, uint32_t segStart, uint32_t segEnd, char *labelPat) {
    uint32_t size = 100; // I consider that this is the maximum feature number.
    feature_dat_t* allFeats = xmalloc(sizeof(feature_dat_t));
    allFeats->len = 0;
    allFeats->features = xmalloc(sizeof(feature_t *) * size);
    if (segStart > 2) {
        allFeats->features[allFeats->len++] = constructFeature("3E", labelPat);
        for (uint32_t i = segStart; i <= segEnd; ++i) {
            allFeats->features[allFeats->len++] = constructFeature(concat("E3W.", tok->toks[i][4]), labelPat);
        }
    }
    return allFeats;
}
