#ifndef model_h
#define model_h

#include <stddef.h>
#include <stdint.h>
#include <sys/time.h>

#include "wapiti.h"
#include "options.h"
#include "sequence.h"
#include "reader.h"

typedef struct timeval tms_t;
typedef struct rdr_s rdr_t;
typedef struct qrk_s qrk_t;
typedef struct id_map_s id_map_t;


typedef struct transition_map_s transition_map_t;
struct transition_map_s {
    uint32_t len;
    uint64_t *idsOne;
    uint64_t *idsTwo;
};

/* mdl_t:
 *   Represent a linear-chain CRF model. The model contain both unigram and
 *   bigram features. It is caracterized by <nlbl> the number of labels, <nobs>
 *   the number of observations, and <nftr> the number of features.
 *
 *   Each observations have a corresponding entry in <kind> whose first bit is
 *   set if the observation is unigram and second one if it is bigram. Note that
 *   an observation can be both. An unigram observation produce Y features and a
 *   bigram one produce Y * Y features.
 *   The <theta> array keep all features weights. The <*off> array give for each
 *   observations the offset in the <theta> array where the features of the
 *   observation are stored.
 *
 *   The <*off> and <theta> array are initialized only when the model is
 *   synchronized. As you can add new labels and observations after a sync, we
 *   keep track of the old counts in <olbl> and <oblk> to detect inconsistency
 *   and resynchronize the model if needed. In this case, if the number of
 *   labels have not changed, the previously trained weights are kept, else they
 *   are now meaningless so discarded.
 */
typedef struct mdl_s mdl_t;
struct mdl_s {
    opt_t *opt;     //       options for training
    int type;    //       model type

    // Size of various model parameters
    uint64_t nlbl;    //   Y   number of labels
    uint64_t nobs;    //   O   number of observations
    uint64_t nftr;    //   F   number of features
    uint64_t npats;   //   P   number of patterns
    uint64_t nfws;    //   H
    uint64_t nbws;    //   B

    transition_map_t *forwardTransition;
    int *lastForwardStateLabel;
    int *backwardTransition; // [B][Y]
    transition_map_t *patternTransition;
    id_map_t *allSuffixes;
    int *lastPatternLabel;
    int *patternBackwardId;

    // The model itself
    double *theta;   //  [F]  features weights
    // double *empiricalScore; // [N*F] empirical feature score
    uint64_t *featureMap;

    // Datasets
    dat_t *train;   //       training dataset
    dat_t *devel;   //       development dataset
    rdr_t *reader;

    // Stoping criterion
    double *werr;    //       Window of error rate of last iters
    uint32_t wcnt;    //       Number of iters in the window
    uint32_t wpos;    //       Position for the next iter

    // Timing
    tms_t timer;   //       start time of last iter
    double total;   //       total training time

};

mdl_t *mdl_new(rdr_t *rdr);

void mdl_free(mdl_t *mdl);

void mdl_sync(mdl_t *mdl);

void buildForwardTransition(mdl_t *mdl);

void buildBackwardTransition(mdl_t *mdl);

void buildPatternTransition(mdl_t *mdl);

void generateSentenceObs(mdl_t *mdl);

void generateFeatureMap(mdl_t *mdl);

void generateEmpiricalFeatureScore(mdl_t *mdl);

// void mdl_compact(mdl_t *mdl);
void mdl_save(mdl_t *mdl, FILE *file);

void mdl_load(mdl_t *mdl, FILE *file);

#endif
