#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "model.h"
#include "options.h"
#include "quark.h"
#include "reader.h"
#include "pattern.h"
#include "tools.h"
#include "vmath.h"
#include "features.h"

/* mdl_new:
 *   Allocate a new empty model object linked with the given reader. The model
 *   have to be synchronized before starting training or labelling. If you not
 *   provide a reader (as it will loaded from file for example) you must be sure
 *   to set one in the model before any attempts to synchronize it.
 */
mdl_t *mdl_new(rdr_t *rdr) {
    mdl_t *mdl = xmalloc(sizeof(mdl_t));
    mdl->nlbl = mdl->nobs = mdl->nftr = mdl->npats = 0;
    mdl->featureMap = NULL;
    mdl->theta = NULL;
    // mdl->empiricalScore = NULL;
    mdl->train = mdl->devel = NULL;
    mdl->reader = rdr;
    mdl->werr = NULL;
    mdl->total = 0.0;
    mdl->forwardTransition = NULL;
    mdl->allSuffixes = NULL;
    mdl->backwardTransition = NULL;
    mdl->patternTransition = NULL;
    return mdl;
}

/* mdl_free:
 *   Free all memory used by a model object inculding the reader and datasets
 *   loaded in the model.
 */
void mdl_free(mdl_t *mdl) {
    free(mdl->lastForwardStateLabel);
    free(mdl->backwardTransition);
    free(mdl->lastPatternLabel);
    free(mdl->patternBackwardId);
    free(mdl->featureMap);
    transition_free(mdl->patternTransition);
    transition_free(mdl->forwardTransition);
    if (mdl->theta != NULL)
        xvm_free(mdl->theta);
    if (mdl->train != NULL)
        rdr_freedat(mdl->train);
    if (mdl->devel != NULL)
        rdr_freedat(mdl->devel);
    if (mdl->reader != NULL)
        rdr_free(mdl->reader);
    if (mdl->werr != NULL)
        free(mdl->werr);
    free(mdl);
}

/* mdl_sync:
 *   Synchronize the model with its reader. As the model is just a placeholder
 *   for features weights and interned sequences, it know very few about the
 *   labels and observations, all the informations are kept in the reader. A
 *   sync will get the labels and observations count as well as the observation
 *   kind from the reader and build internal structures representing the model.
 *
 *   If the model was already synchronized before, there is an existing model
 *   incompatible with the new one to be created. In this case there is two
 *   possibility :
 *     - If only new observations was added, the weights of the old ones remain
 *       valid and are kept as they form a probably good starting point for
 *       training the new model, the new observation get a 0 weight ;
 *     - If new labels was added, the old model are trully meaningless so we
 *       have to fully discard them and build a new empty model.
 *   In any case, you must never change existing labels or observations, if this
 *   happen, you need to create a new model and destroy this one.
 *
 *   After synchronization, the labels and observations databases are locked to
 *   prevent new one to be created. You must unlock them explicitly if needed.
 *   This reduce the risk of mistakes.
 */
void mdl_sync(mdl_t *mdl, bool doTrain) {
    const uint64_t Y = qrk_count(mdl->reader->lbl);
    const uint64_t O = qrk_count(mdl->reader->obs);
    const uint64_t F = qrk_count(mdl->reader->featList);
    const uint64_t P = qrk_count(mdl->reader->pats);
    const uint64_t ForS = qrk_count(mdl->reader->forwardStateMap);
    const uint64_t BackS = qrk_count(mdl->reader->backwardStateMap);
    qrk_lock(mdl->reader->lbl, true);
    qrk_lock(mdl->reader->obs, true);
    qrk_lock(mdl->reader->pats, true);
    qrk_lock(mdl->reader->featList, true);
    qrk_lock(mdl->reader->forwardStateMap, true);
    qrk_lock(mdl->reader->backwardStateMap, true);
    // If model is already synchronized, do nothing and just return
    if (mdl->nlbl == Y && mdl->nobs == O)
        return;
    if (Y == 0 || O == 0)
        fatal("cannot synchronize an empty model");

    mdl->npats = P;
    mdl->nlbl = Y;
    mdl->nobs = O;
    mdl->nftr = F;
    mdl->nfws = ForS;
    mdl->nbws = BackS;
    mdl->forwardTransition = xmalloc(sizeof(transition_map_t) * mdl->nfws);
    mdl->backwardTransition = xmalloc(sizeof(int) * (mdl->nbws) * (Y));
    mdl->allSuffixes = xmalloc(sizeof(id_map_t) * mdl->nbws);
    mdl->lastForwardStateLabel = xmalloc(sizeof(int) * mdl->nfws);
    mdl->lastPatternLabel = xmalloc(sizeof(int) * P);
    mdl->patternBackwardId = xmalloc(sizeof(int) * P);
    mdl->patternTransition = xmalloc(sizeof(transition_map_t) * P);
    mdl->theta = xvm_new(F);

    if (doTrain) {
        generateSentenceObs(mdl);
        generateFeatureMap(mdl);
        buildForwardTransition(mdl);
        buildBackwardTransition(mdl);
        buildPatternTransition(mdl);
        generateEmpiricalFeatureScore(mdl);
    }
    else {
        generateFeatureMap(mdl);
        buildForwardTransition(mdl);
        buildBackwardTransition(mdl);
        buildPatternTransition(mdl);
    }
    info("finish mdl_sync\n");
    return;
}

void buildForwardTransition(mdl_t *mdl) {
    info("inside buildForwardTransition\n");
    rdr_t *reader = mdl->reader;
    uint32_t curLen;
    uint64_t size = mdl->nfws;
    uint64_t index, backIndex;
    transition_map_t (*forwardTransition)[size] = (void *) mdl->forwardTransition;
    int *lastForwardStateLabel = mdl->lastForwardStateLabel;
    for (uint32_t i = 0; i < size; ++i) {
        (*forwardTransition)[i].idsOne = xmalloc(sizeof(uint64_t) * size);
        (*forwardTransition)[i].idsTwo = xmalloc(sizeof(uint64_t) * size);
        (*forwardTransition)[i].len = 0;
    }
    for (uint32_t fsid = 0; fsid < size; ++fsid) {
        const char *fs = qrk_id2str(reader->forwardStateMap, fsid);
        lastForwardStateLabel[fsid] = (strcmp(fs, "") == 0) ? -1 : (int) getLastLabelId(reader, fs);
        for (uint32_t lbid = 0; lbid < mdl->nlbl; ++lbid) {
            if ((lbid != (uint32_t) lastForwardStateLabel[fsid]) || (reader->maxSegment == 1)) {
                const char *pky = (strcmp(fs, "") == 0) ? qrk_id2str(reader->lbl, lbid) : concat(
                        concat(qrk_id2str(reader->lbl, lbid), "|"), fs);
                index = getLongestIndexId((char *) pky, reader->forwardStateMap);
                curLen = (*forwardTransition)[index].len;
                (*forwardTransition)[index].idsOne[curLen] = fsid;
                backIndex = qrk_str2id(reader->backwardStateMap, pky);
                (*forwardTransition)[index].idsTwo[curLen] = backIndex;
                ++(*forwardTransition)[index].len;
            }
        }
    }
}

void buildBackwardTransition(mdl_t *mdl) {
    info("inside buildBackwardTransition\n");
    int lastLabelId;
    qrk_t *lbQrk = mdl->reader->lbl;
    uint64_t patId;
    char *sk;
    int (*poi)[mdl->nbws][mdl->nlbl] = (void *) mdl->backwardTransition;
    char *siy = xmalloc(sizeof(char) * (mdl->reader->maxSegment * 2 - 1));
    for (uint64_t siId = 0; siId < mdl->nbws; ++siId) {
        const char *si = qrk_id2str(mdl->reader->backwardStateMap, siId);
        lastLabelId = strcmp(si, "") == 0 ? -1 : (int) getLastLabelId(mdl->reader, si);
        for (uint32_t lbId = 0; lbId < mdl->nlbl; ++lbId) {
            if ((lbId != (uint32_t) lastLabelId) || (mdl->reader->maxSegment == 1)) {
                siy = concat(concat(qrk_id2str(lbQrk, lbId), "|"), si);
                sk = getLongestSuffix(siy, mdl->reader->backwardStateMap);
                (*poi)[siId][lbId] = (int) qrk_str2id(mdl->reader->backwardStateMap, sk);
            } else {
                (*poi)[siId][lbId] = -1;
            }
        }
        labelPat_t *suffixes = generateLabelPatStruct((char *) si);
        mdl->allSuffixes[siId].ids = xmalloc(sizeof(uint64_t) * suffixes->segNum);
        mdl->allSuffixes[siId].len = 0;
        for (uint32_t i = 0; i < suffixes->segNum; ++i) {
            patId = qrk_str2id(mdl->reader->pats, suffixes->suffixes[i]);
            if (patId != none) {
                mdl->allSuffixes[siId].ids[mdl->allSuffixes[siId].len++] = patId;
            }
        }
    }
    free(siy);
}

void buildPatternTransition(mdl_t *mdl) {
    info("inside buildPatternTransition\n");
    qrk_t *patDat = mdl->reader->pats;
    qrk_t *backDat = mdl->reader->backwardStateMap;
    qrk_t *forDat = mdl->reader->forwardStateMap;
    qrk_t *lbDat = mdl->reader->lbl;
    uint64_t lastY, lastLb, piyId, ziIndex;
    transition_map_t *map = mdl->patternTransition;
    char *piy; // = xmalloc(sizeof(char) * (mdl->reader->maxSegment * 2 - 1));
    for (uint64_t pId = 0; pId < mdl->npats; ++pId) {
        const char *p = qrk_id2str(patDat, pId);
        mdl->patternBackwardId[pId] = (int) qrk_str2id(backDat, p);
        lastY = getLastLabelId(mdl->reader, p);
        mdl->lastPatternLabel[pId] = lastY == none ? -1 : (int) lastY;
        mdl->patternTransition[pId].len = 0;
        mdl->patternTransition[pId].idsOne = xmalloc(sizeof(uint64_t) * mdl->nfws);
        mdl->patternTransition[pId].idsTwo = xmalloc(sizeof(uint64_t) * mdl->nbws);
    }
    for (uint32_t piId = 0; piId < mdl->nfws; ++piId) {
        const char *pi = qrk_id2str(forDat, piId);
        lastLb = strcmp(pi, "") == 0 ? none : getLastLabelId(mdl->reader, pi);
        for (uint64_t y = 0; y < mdl->nlbl; ++y) {
            if ((y != lastLb) || (mdl->reader->maxSegment == 1)) {
                piy = strcmp(pi, "") == 0 ? (char *) qrk_id2str(lbDat, y) : concat(concat(qrk_id2str(lbDat, y), "|"),
                                                                                   pi);
                piyId = qrk_str2id(backDat, piy);
                labelPat_t *patStruct = generateLabelPatStruct(piy);
                for (uint64_t ziId = 0; ziId < patStruct->segNum; ++ziId) {
                    ziIndex = qrk_str2id(patDat, patStruct->suffixes[ziId]);
                    if (ziIndex != none) {
                        map[ziIndex].idsOne[map[ziIndex].len] = piId;
                        map[ziIndex].idsTwo[map[ziIndex].len] = piyId;
                        ++map[ziIndex].len;
                    }
                }
            }
        }
    }
}

void generateFeatureMap(mdl_t *mdl) {
    uint16_t size = 50;
    char *f = xmalloc(sizeof(char) * size);
    uint64_t i, j;
    mdl->featureMap = xmalloc(sizeof(uint64_t) * mdl->npats * mdl->nobs);


    for (i = 0; i < mdl->npats; ++i) {
        for (j = 0; j < mdl->nobs; ++j) {
            mdl->featureMap[i * (mdl->nobs) + j] = 0;
        }
    }

    for (i = 0; i < mdl->npats; ++i) {
        for (j = 0; j < mdl->nobs; ++j) {
            f = concat(concat(qrk_id2str(mdl->reader->pats, i), "_"), qrk_id2str(mdl->reader->obs, j));
            uint64_t id = qrk_str2id(mdl->reader->featList, f);
            if (id != none)
                mdl->featureMap[i * (mdl->nobs) + j] = id;
        }
    }
    free(f);
}

void generateSentenceObs(mdl_t *mdl) {
    uint64_t obId;
    dat_t *dat = mdl->train;
    int32_t S = mdl->reader->maxSegment;
    for (uint32_t n = 0; n < dat->nseq; ++n) {
        tok_t *tok = dat->tok[n];
        uint32_t T = tok->len;
        tok->observationMapjp = xmalloc(sizeof(id_map_t) * T * S);
        id_map_t (*poi)[T][S] = (void *) tok->observationMapjp;
        for (uint32_t i = 0; i < T; ++i) {
            for (uint32_t j = i; j < T && j - i < (uint32_t) S; ++j) {
                (*poi)[i][j - i].len = 0;
                (*poi)[i][j - i].ids = xmalloc(sizeof(uint64_t) * mdl->nobs);
                char *labelPat = generateLabelPattern(tok, i);
                feature_dat_t *features = generateObs(tok, i, j, labelPat);
                qrk_t *set = qrk_new();
                qrk_lock(set, true);
                for (uint32_t id = 0; id < features->len; ++id) {
                    char *obs = features->features[id]->obs;
                    obId = qrk_str2id(mdl->reader->obs, obs);
                    if ((obId != none) && (qrk_str2id(set, obs) == none)) {
                        qrk_lock(set, false);
                        qrk_str2id(set, obs);
                        qrk_lock(set, true);
                        (*poi)[i][j - i].ids[(*poi)[i][j - i].len++] = obId;
                    }
                }
                qrk_free(set);
                // (*poi)[i][j - i].ids = xrealloc((*poi)[i][j - i].ids, (*poi)[i][j - i].len);
            }
        }
    }
}

void generateEmpiricalFeatureScore(mdl_t *mdl) {
    uint64_t sId, patId, patIndex, obIndex, featId;
    dat_t *dat = mdl->train;
    int32_t S = mdl->reader->maxSegment;
    uint32_t segStart, segEnd;
    id_map_t *observationMapjd;
    const uint32_t L = dat->nseq;
    const uint64_t F = mdl->nftr;
    uint64_t (*featureMap)[mdl->npats][mdl->nobs] = (void *) mdl->featureMap;
    // double (*emScore)[L][F] = (void *)mdl->empiricalScore;

    for (uint32_t n = 0; n < L; ++n) {
        tok_t *tok = dat->tok[n];
        uint32_t T = tok->len;
        tok->empiricalScore = xvm_new(F);
        double *emScore = tok->empiricalScore;
        for (uint64_t i = 0; i < F; ++i) {
            emScore[i] = 0;
        }
        id_map_t (*obsMap)[T][S] = (void *) tok->observationMapjp;
        segStart = 0;
        while (segStart < T) {
            segEnd = tok->sege[segStart];
            observationMapjd = &((*obsMap)[segStart][segEnd - segStart]);
            char *labelPat = generateLabelPattern(tok, segStart);
            sId = qrk_str2id(mdl->reader->backwardStateMap, labelPat);
            for (patIndex = 0; patIndex < mdl->allSuffixes[sId].len; ++patIndex) {
                for (obIndex = 0; obIndex < observationMapjd->len; ++obIndex) {
                    patId = mdl->allSuffixes[sId].ids[patIndex];
                    featId = (*featureMap)[patId][observationMapjd->ids[obIndex]];
                    emScore[featId] += featId != 0 ? 1 : 0;
                }
            }
            segStart = segEnd + 1;
        }
    }
}


/* mdl_save:
 *   Save a model to be restored later in a platform independant way.
*/
void mdl_save(mdl_t *mdl, FILE *file) {
    uint64_t nact = 0;
    for (uint64_t f = 0; f < mdl->nftr; f++)
        if (mdl->theta[f] != 0.0)
            nact++;
    fprintf(file, "#mdl#%d#%"PRIu64"\n", mdl->type, nact);
    rdr_save(mdl->reader, file);
    for (uint64_t f = 0; f < mdl->nftr; f++)
        if (mdl->theta[f] != 0.0)
            fprintf(file, "%"PRIu64"=%f\n", f, mdl->theta[f]);
}

/* mdl_load:
 *   Read back a previously saved model to continue training or start labeling.
 *   The returned model is synced and the quarks are locked. You must give to
 *   this function an empty model fresh from mdl_new.
*/
void mdl_load(mdl_t *mdl, FILE *file) {
    const char *err = "invalid model format";
    uint64_t nact = 0;
    int type;
    if (fscanf(file, "#mdl#%d#%"SCNu64"\n", &type, &nact) == 2) {
        mdl->type = type;
    } else {
        rewind(file);
        if (fscanf(file, "#mdl#%"SCNu64"\n", &nact) == 1)
            mdl->type = 0;
        else
            fatal(err);
    }
    rdr_load(mdl->reader, file);
    mdl_sync(mdl, false);
    for (uint64_t i = 0; i < nact; i++) {
        uint64_t f;
        double v;
        if (fscanf(file, "%"SCNu64"=%la\n", &f, &v) != 2)
            fatal(err);
        mdl->theta[f] = v;
    }
}

void transition_free(transition_map_t *a) {
    free(a->idsOne);
    free(a->idsTwo);
    free(a);
}