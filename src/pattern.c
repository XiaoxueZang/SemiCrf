#include <ctype.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pattern.h"
#include "sequence.h"
#include "reader.h"
#include "tools.h"
#include "features.h"


// I assume that label is only one char.
// Unit Tested
char *generateLabelPattern(tok_t *toks, uint32_t segStart, uint32_t segEnd) {
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
    uint32_t size = 200;
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
        if (a == '|' | a == '\0') {
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

feature_dat_t *generateObs(tok_t *tok, rdr_t *reader, uint32_t segStart, uint32_t segEnd, char *labelPat) {
    uint32_t size = 5000;
    feature_dat_t *featurePack = xmalloc(sizeof(feature_dat_t));
    featurePack->len = 0;
    featurePack->features = xmalloc(sizeof(feature_t *) * size);
    labelPat_t *labelPatStruct = generateLabelPatStruct(labelPat);
    feature_dat_t *partFeat = xmalloc(sizeof(feature_dat_t) * size);
    for (int i = labelPatStruct->order; i >= 0; --i) {
        if (i == 0) partFeat = generateCrfFeaturesAt(tok, segStart, segEnd, labelPatStruct->suffixes[i]);
        if (i == 1) partFeat = generateFirstOrderFeaturesAt(tok, segStart, segEnd, labelPatStruct->suffixes[i]);
        if (i == 2) partFeat = generateSecondOrderFeaturesAt(tok, segStart, segEnd, labelPatStruct->suffixes[i]);
        if (i == 3) partFeat = generateThirdOrderFeaturesAt(tok, segStart, segEnd, labelPatStruct->suffixes[i]);
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

// some little problem.
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