#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "wapiti.h"
#include "model.h"
#include "options.h"
#include "quark.h"
#include "reader.h"
#include "tools.h"
#include "vmath.h"
#include "pattern.h"

/*******************************************************************************
 * Linear chain CRF model
 *
 *   There is three concept that must be well understand here, the labels,
 *   observations, and features. The labels are the values predicted by the
 *   model at each point of the sequence and denoted by Y. The observations are
 *   the values, at each point of the sequence, given to the model in order to
 *   predict the label and denoted by O. A feature is a test on both labels and
 *   observations, denoted by F. In linear chain CRF there is two kinds of
 *   features :
 *     - unigram feature who represent a test on the observations at the current
 *       point and the label at current point.
 *     - bigram feature who represent a test on the observation at the current
 *       point and two labels : the current one and the previous one.
 *   So for each observation, there Y possible unigram features and Y*Y possible
 *   bigram features. The kind of features used by the model for a given
 *   observation depend on the pattern who generated it.
 ******************************************************************************/

/* mdl_new:
 *   Allocate a new empty model object linked with the given reader. The model
 *   have to be synchronized before starting training or labelling. If you not
 *   provide a reader (as it will loaded from file for example) you must be sure
 *   to set one in the model before any attempts to synchronize it.
 */
mdl_t *mdl_new(rdr_t *rdr) {
    mdl_t *mdl = xmalloc(sizeof(mdl_t));
    mdl->nlbl = mdl->nobs = mdl->nftr = 0;
    mdl->kind = NULL;
    mdl->uoff = mdl->boff = NULL;
    mdl->theta = NULL;
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
    free(mdl->kind);
    free(mdl->uoff);
    free(mdl->boff);
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
void mdl_sync(mdl_t *mdl) {
    const uint32_t Y = qrk_count(mdl->reader->lbl);
    const uint64_t O = qrk_count(mdl->reader->obs);
    const uint64_t F = qrk_count(mdl->reader->featList);
    const uint32_t P = qrk_count(mdl->reader->pats);
    qrk_lock(mdl->reader->lbl, true);
    qrk_lock(mdl->reader->obs, true);
    qrk_lock(mdl->reader->pats, true);
    // If model is already synchronized, do nothing and just return
    if (mdl->nlbl == Y && mdl->nobs == O)
        return;
    if (Y == 0 || O == 0)
        fatal("cannot synchronize an empty model");
    // If new labels was added, we have to discard all the model. In this
    // case we also display a warning as this is probably not expected by
    // the user. If only new observations was added, we will try to expand
    // the model.
    uint64_t oldF = mdl->nftr;
    uint64_t oldO = mdl->nobs;
    if (mdl->nlbl != Y && mdl->nlbl != 0) {
        warning("labels count changed, discarding the model");
        free(mdl->kind);
        mdl->kind = NULL;
        free(mdl->uoff);
        mdl->uoff = NULL;
        free(mdl->boff);
        mdl->boff = NULL;
        if (mdl->theta != NULL) {
            xvm_free(mdl->theta);
            mdl->theta = NULL;
        }
        oldF = oldO = 0;
    }
    mdl->npats = P;
    mdl->nlbl = Y;
    mdl->nobs = O;
    mdl->nftr = F;

    generateForwardStateMap(mdl);
    generateBackwardStateMap(mdl);

    // initializeTransitions(mdl);
    mdl->nfws = mdl->reader->nforwardStateMap;
    mdl->nbws = mdl->reader->nbackwardStateMap;
    mdl->forwardTransition = xmalloc(sizeof(transition_map_t) * mdl->nfws);
    mdl->backwardTransition = xmalloc(sizeof(int) * (mdl->nbws) * (Y));
    mdl->allSuffixes = xmalloc(sizeof(id_map_t) * mdl->nbws);
    mdl->lastForwardStateLabel = xmalloc(sizeof(int) * mdl->nfws);
    mdl->lastPatternLabel = xmalloc(sizeof(int) * P);
    mdl->patternBackwardId = xmalloc(sizeof(int) * P);
    mdl->patternTransition = xmalloc(sizeof(transition_map_t) * P);

    generateSentenceObs(mdl);
    buildForwardTransition(mdl);
    buildBackwardTransition(mdl);
    buildPatternTransition(mdl);
    return;
}

/*
    // Allocate the observations datastructure. If the model is empty or
    // discarded, a new one iscreated, else the old one is expanded.
    char *kind = xrealloc(mdl->kind, sizeof(char) * O);
    uint64_t *uoff = xrealloc(mdl->uoff, sizeof(uint64_t) * O);
    uint64_t *boff = xrealloc(mdl->boff, sizeof(uint64_t) * O);
    mdl->kind = kind;
    mdl->uoff = uoff;
    mdl->boff = boff;
    // Now, we can setup the features. For each new observations we fill the
    // kind and offsets arrays and count total number of features as well.
    // uint64_t F = oldF;
    for (uint64_t o = oldO; o < O; o++) {
        const char *obs = qrk_id2str(mdl->reader->obs, o);
        switch (obs[0]) {
            case 'u':
                kind[o] = 1;
                break;
            case 'b':
                kind[o] = 2;
                break;
            case '*':
                kind[o] = 3;
                break;
        }
        if (kind[o] & 1)
            uoff[o] = F, F += Y;
        if (kind[o] & 2)
            boff[o] = F, F += Y * Y;
    }
    mdl->nftr = F;
    // We can finally grow the features weights vector itself. We set all
    // the new features to 0.0 but don't touch the old ones.
    // This is a bit tricky as aligned malloc cannot be simply grown so we
    // have to allocate a new vector and copy old values ourself.
    if (oldF != 0) {
        double *new = xvm_new(F);
        for (uint64_t f = 0; f < oldF; f++)
            new[f] = mdl->theta[f];
        xvm_free(mdl->theta);
        mdl->theta = new;
    } else {
        mdl->theta = xvm_new(F);
    }
    for (uint64_t f = oldF; f < F; f++)
        mdl->theta[f] = 0.0;
    // And lock the databases
    qrk_lock(mdl->reader->lbl, true);
    qrk_lock(mdl->reader->obs, true);
}
*/

void generateForwardStateMap(mdl_t *mdl) {
    rdr_t *reader = mdl->reader;
    qrk_t *forwardStateMap = reader->forwardStateMap;
    qrk_str2id(forwardStateMap, "");
    char *str = xmalloc(sizeof(char) * (2 * reader->maxSegment - 1));
    for (uint64_t id = 0; id < mdl->nlbl; ++id) {
        strcpy(str, qrk_id2str(reader->lbl, id));
        qrk_str2id(forwardStateMap, str);
        // info("the forwardstatemap %s\n", str);
    }
    for (uint64_t id = 0; id < mdl->npats; ++id) {
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

void generateBackwardStateMap(mdl_t *mdl) {
    uint32_t size = 50;
    rdr_t *reader = mdl->reader;
    int lastLabel;
    char *p = xmalloc(sizeof(char) * size);
    for (uint64_t id = 0; id < reader->nforwardStateMap; ++id) {
        strcpy(p, qrk_id2str(reader->forwardStateMap, id));
        lastLabel = (strcmp(p, "") == 0) ? -1 : (int) getLastLabelId(reader, p);

        for (uint64_t yid = 0; yid < mdl->nlbl; ++yid) {
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

void buildForwardTransition(mdl_t *mdl) {
    rdr_t *reader = mdl->reader;
    uint32_t curLen;
    uint32_t size = reader->nforwardStateMap;
    uint64_t index, backIndex;
    // uint32_t forwardTransitionOne[reader->nforwardStateMap][reader->nforwardStateMap] = mdl->reader->forwardTransitionOne;
    // uint32_t forwardTransitionTwo[reader->nforwardStateMap][reader->nbackwardStateMap] = m;
    transition_map_t (*forwardTransition)[size] = (void *) mdl->forwardTransition;
    int *lastForwardStateLabel = mdl->lastForwardStateLabel;
    for (uint32_t i = 0; i < size; ++i) {
        (*forwardTransition)[i].idsOne = xmalloc(sizeof(uint64_t) * size);
        (*forwardTransition)[i].idsTwo = xmalloc(sizeof(uint64_t) * size);
        (*forwardTransition)[i].len = 0;
    }
    for (uint32_t fsid = 0; fsid < size; ++fsid) {
        // char *fs = xmalloc(sizeof(char) * size);
        const char *fs = qrk_id2str(reader->forwardStateMap, fsid);
        lastForwardStateLabel[fsid] = (strcmp(fs, "") == 0) ? -1 : (int) getLastLabelId(reader, fs);
        for (uint32_t lbid = 0; lbid < mdl->nlbl; ++lbid) {
            if ((lbid != lastForwardStateLabel[fsid]) || (reader->maxSegment == 1)) {
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
            if ((lbId != lastLabelId) || (mdl->reader->maxSegment == 1)) {
                siy = concat(concat(qrk_id2str(lbQrk, lbId), "|"), si);
                sk = getLongestSuffix(siy, mdl->reader->backwardStateMap);
                (*poi)[siId][lbId] = (int) qrk_str2id(mdl->reader->backwardStateMap, sk);
            } else {
                (*poi)[siId][lbId] = -1;
            }
        }
        labelPat_t *suffixes = generateLabelPatStruct((char *) si);
        mdl->allSuffixes[siId].ids = xmalloc(sizeof(uint64_t) * suffixes->segNum);
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
    qrk_t *patDat = mdl->reader->pats;
    qrk_t *backDat = mdl->reader->backwardStateMap;
    qrk_t *forDat = mdl->reader->forwardStateMap;
    qrk_t *lbDat = mdl->reader->lbl;
    uint64_t lastY, lastLb, piyId, ziIndex;
    transition_map_t *map = mdl->patternTransition;
    char *piy = xmalloc(sizeof(char) * (mdl->reader->maxSegment * 2 - 1));
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

void generateSentenceObs(mdl_t *mdl) {
    uint64_t obId;
    dat_t *dat = mdl->train;
    uint32_t S = mdl->reader->maxSegment;
    for (uint32_t n = 0; n < dat->nseq; ++n) {
        tok_t *tok = dat->tok[n];
        uint32_t T = tok->len;
        tok->observationMapjp = xmalloc(sizeof(id_map_t) * T * S);
        id_map_t (*poi)[T][S] = (void *)tok->observationMapjp;
        for (uint32_t i = 0; i < T; ++i) {
            for (uint32_t j = i; j < T && j - i < S; ++j) {
                (*poi)[i][j - i].len = 0;
                (*poi)[i][j - i].ids = xmalloc(sizeof(uint64_t) * mdl->nobs);
                char *labelPat = generateLabelPattern(tok, i, j);
                feature_dat_t *features = generateObs(tok, mdl->reader, i, j, labelPat);
                for (uint32_t id = 0; id < features->len; ++id) {
                    char *obs = features->features[id]->obs;
                    obId = qrk_str2id(mdl->reader->obs, obs);
                    if (obId != none) {
                        (*poi)[i][j - i].ids[(*poi)[i][j - i].len++] = obId;
                    }
                }
                // (*poi)[i][j - i].ids = xrealloc((*poi)[i][j - i].ids, (*poi)[i][j - i].len);
            }
        }
    }
}
/* mdl_save:
 *   Save a model to be restored later in a platform independant way.

void mdl_save(mdl_t *mdl, FILE *file) {
	uint64_t nact = 0;
	for (uint64_t f = 0; f < mdl->nftr; f++)
		if (mdl->theta[f] != 0.0)
			nact++;
	fprintf(file, "#mdl#%d#%"PRIu64"\n", mdl->type, nact);
	rdr_save(mdl->reader, file);
	for (uint64_t f = 0; f < mdl->nftr; f++)
		if (mdl->theta[f] != 0.0)
			fprintf(file, "%"PRIu64"=%la\n", f, mdl->theta[f]);
}
*/
/* mdl_load:
 *   Read back a previously saved model to continue training or start labeling.
 *   The returned model is synced and the quarks are locked. You must give to
 *   this function an empty model fresh from mdl_new.

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
	mdl_sync(mdl);
	for (uint64_t i = 0; i < nact; i++) {
		uint64_t f;
		double v;
		if (fscanf(file, "%"SCNu64"=%la\n", &f, &v) != 2)
			fatal(err);
		mdl->theta[f] = v;
	}
}
*/
