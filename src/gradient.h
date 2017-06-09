#ifndef gradient_h
#define gradient_h

#include "wapiti.h"
#include "model.h"
#include "sequence.h"

/* grd_st_t:
 *   State tracker for the gradient computation. To compute the gradient we need
 *   to perform several steps and communicate between them a lot of intermediate
 *   values, all these temporary are stored in this object.
 *   A tracker can be used to compute sequence of length <len> at most, before
 *   using it you must call grd_stcheck to ensure that the tracker is big enough
 *   for your sequence.
 *   This tracker is used to perform single sample gradient computations or
 *   partial gradient computation in online algorithms and for decoding with
 *   posteriors.
 */
typedef struct grd_st_s grd_st_t;
struct grd_st_s {
    mdl_t *mdl;
    uint32_t len;     // =T        max length of sequence
    double *g;       // [F]       vector where to put gradient updates
    double lloss;   //           loss value for the sequence
    double logZx;
    // double *psi;     // [F]       the transitions scores
    // double   *psiuni;  // [T][Y]    | Same as psi in sparse format
    // uint32_t *psiyp;   // [T][Y][Y] |
    // uint32_t *psiidx;  // [T][Y]    |
    // uint32_t *psioff;  // [T]
    double *alpha;   // [T][Y]    forward scores
    double *beta;    // [T][Y]    backward scores
    double *marginal; // [P][T][S]
    double *expec;   // [F]
    // double *scale;   // [T]       scaling factors of forward scores
    // double *unorm;   // [T]       normalization factors for unigrams
    // double *bnorm;   // [T]       normalization factors for bigrams
    uint32_t first;   //           first position where gradient is needed
    uint32_t last;    //           last position where gradient is needed
};

grd_st_t *grd_stnew(mdl_t *mdl, double *g);

void grd_stfree(grd_st_t *grd_st);

void grd_stcheck(grd_st_t *grd_st, uint32_t len);

void grd_dospl(grd_st_t *grd_st, const tok_t *seq);

/* grd_t:
 *   Multi-threaded full dataset gradient computer. This is used to compute the
 *   gradient by algorithm working on the full dataset at each iterations. It
 *   efficiently compute it using the fact it is additive to use as many threads
 *   as allowed.
 */
typedef struct grd_s grd_t;
struct grd_s {
    mdl_t *mdl;
    grd_st_t **grd_st;
};

grd_t *grd_new(mdl_t *mdl, double *g);

void grd_free(grd_t *grd);

double grd_gradient(grd_t *grd);

#endif

