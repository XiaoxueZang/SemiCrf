#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wapiti.h"
#include "gradient.h"
#include "model.h"
#include "options.h"
#include "progress.h"
#include "sequence.h"
#include "tools.h"
#include "thread.h"
#include "vmath.h"

/* atm_inc:
 *   Atomically increment the value pointed by [ptr] by [inc]. If ATM_ANSI is
 *   defined this NOT atomic at all so caller must have to deal with this.
 */
#ifdef ATM_ANSI
static inline
void atm_inc(double *value, double inc) {
    *value += inc;
}
#else

static inline
void atm_inc(volatile double *value, double inc) {
    while (1) {
        volatile union {
            double d;
            uint64_t u;
        } old, new;
        old.d = *value;
        new.d = old.d + inc;
        uint64_t *ptr = (uint64_t *) value;
        if (__sync_bool_compare_and_swap(ptr, old.u, new.u))
            break;
    }
}

#endif

/******************************************************************************
 * Linear-chain CRF gradient computation
 *
 *   This section is responsible for computing the gradient of the
 *   log-likelihood function to optimize over a single sequence.
 *
 *   There is two version of this code, one using dense matrix and one with
 *   sparse matrix. The sparse version use the fact that for L1 regularized
 *   trainers, the bigrams scores will be very sparse so there is a way to
 *   reduce the amount of computation needed in the forward backward at the
 *   price of a more complex implementation. Due to the fact that using a sparse
 *   matrix have a cost, this implementation is slower on L2 regularized models
 *   and on lighty L1-regularized models, this is why there is also a classical
 *   dense version of the algorithm used for example by the L-BFGS trainer.
 *
 *   The sparse matrix implementation is a bit tricky because we need to store
 *   all values in sequences in order to use the vector exponential who gives
 *   also a lot of performance improvement on vector able machine.
 *   We need four arrays noted <val>, <off>, <idx>, and <yp>. For each positions
 *   t, <off>[t] value indicate where the non-zero values for t starts in <val>.
 *   The other arrays gives the y and yp indices of these values. The easier one
 *   to retrieve is yp, the yp indice for value at <val>[<off>[t] + n] is stored
 *   at the same position in <yp>.
 *   The y are more difficult: the indice y are stored with n between <idx>[y-1]
 *   and <idx>[y]. It may seems inefective but the matrix is indexed in the
 *   other way, we go through the idx array, and for each y we get the yp and
 *   values, so in practice it's very efficient.
 *
 *   This can seem too complex but we have to keep in mind that Y are generally
 *   very low and any sparse-matrix have overhead so we have to reduce it to the
 *   minimum in order to get a real improvment. Dedicated library are optimized
 *   for bigger matrix where the overhead is not a so important problem.
 *   Another problem here is cache size. The optimization process will last most
 *   of his time in this function so it have to be well optimized and we already
 *   need a lot of memory for other data so we have to be carefull here if we
 *   don't want to flush the cache all the time. Sparse matrix require less
 *   memory than dense one only if we now in advance the number of non-zero
 *   entries, which is not the case here, so we have to use a scheme which in
 *   the worst case use as less as possible memory.
 ******************************************************************************/

/* grd_fldopsi:
 *   We first have to compute the Ψ_t(y',y,x) weights defined as
 *       Ψ_t(y',y,x) = \exp( ∑_k θ_k f_k(y',y,x_t) )
 *   So at position 't' in the sequence, for each couple (y',y) we have to sum
 *   weights of all features. Only the observations present at this position
 *   will have a non-nul weight so we can sum only on thoses. As we use only two
 *   kind of features: unigram and bigram, we can rewrite this as
 *       \exp (  ∑_k μ_k(y, x_t)     f_k(y, x_t)
 *             + ∑_k λ_k(y', y, x_t) f_k(y', y, x_t) )
 *   Where the first sum is over the unigrams features and the second is over
 *   bigrams ones.
 *   This allow us to compute Ψ efficiently in three steps
 *     1/ we sum the unigrams features weights by looping over actives
 *          unigrams observations. (we compute this sum once and use it
 *          for each value of y')
 *     2/ we add the bigrams features weights by looping over actives
 *          bigrams observations (we don't have to do this for t=0 since
 *          there is no bigrams here)
 *     3/ we take the component-wise exponential of the resulting matrix
 *          (this can be done efficiently with vector maths)
 */
/*
void grd_fldopsi(grd_st_t *grd_st, const tok_t *tok) {
    const mdl_t *mdl = grd_st->mdl;
    const uint64_t F = mdl->nftr;
    const double *x = mdl->theta;
    const uint32_t Y = mdl->nlbl;
    const uint32_t T = tok->len;
    double (*psi)[F] = (void *) grd_st->psi;
    for (uint32_t t = 0; t < T; t++) {
        const pos_t *pos = &(tok[t]);
        for (uint32_t y = 0; y < Y; y++) {
            double sum = 0.0;
            for (uint32_t n = 0; n < pos->ucnt; n++) {
                const uint64_t o = pos->uobs[n];
                sum += x[mdl->uoff[o] + y];
            }
            for (uint32_t yp = 0; yp < Y; yp++)
                (*psi)[t][yp][y] = sum;
        }
    }
    for (uint32_t t = 1; t < T; t++) {
        const pos_t *pos = &(tok->pos[t]);
        for (uint32_t yp = 0, d = 0; yp < Y; yp++) {
            for (uint32_t y = 0; y < Y; y++, d++) {
                double sum = 0.0;
                for (uint32_t n = 0; n < pos->bcnt; n++) {
                    const uint64_t o = pos->bobs[n];
                    sum += x[mdl->boff[o] + d];
                    if (sum > 0.0)
                        info("sum of {*psi}[%u][%u][%u] is %f", t, yp, y, sum);
                }
                (*psi)[t][yp][y] += sum;
            }
        }
    }
    xvm_expma((double *) psi, (double *) psi, 0.0, (uint64_t) T * Y * Y);
}
*/

/* grd_spdopsi:
 *   For the sparse version, we keep the two sum separate so we will have
 *   separate Ψ_t(y,x) and Ψ_t(y',y,x). The first one define a vector for
 *   unigram at each position, and the second one a matrix for bigrams.  This is
 *   where the trick is as we will store Ψ_t(y',y,x) - 1. If the sum is nul, his
 *   exponential will be 1.0 and so we have to store 0.0.  As most of the sum
 *   are expected to be nul the resulting matrix will be very sparse and we will
 *   save computation in the forward-backward.
 *
 *   So we compute Ψ differently here
 *     1/ we sum the unigrams features weights by looping over actives
 *          unigrams observations and store them in |psiuni|.
 *     2/ we sum the bigrams features weights by looping over actives
 *          bigrams observations (we don't have to do this for t=0 since
 *          there is no bigrams here) and we store the non-nul one in the
 *          sparse matrix.
 *     3/ we take the component-wise exponential of the unigrams vectors,
 *          and the component-wise exponential of the sparse matrix minus
 *          one. (here also this can be done efficiently with vector
 *          maths)
 */

/* grd_flfwdbwd:
 *   Now, we go to the forward-backward algorithm. As this part of the code rely
 *   on a lot of recursive sums and products of exponentials, we have to take
 *   care of numerical problems.
 *   First the forward recursion
 *       | α_1(y) = Ψ_1(y,x)
 *       | α_t(y) = ∑_{y'} α_{t-1}(y') * Ψ_t(y',y,x)
 *   Next come the backward recursion which is very similar
 *       | β_T(y') = 1
 *       | β_t(y') = ∑_y β_{t+1}(y) * Ψ_{t+1}(y',y,x)
 *   The numerical problems can appear here. To solve them we will scale the α_t
 *   and β_t vectors so they sum to 1 but we have to keep the scaling coeficient
 *   as we will need them later.
 *   Now, we have to compute the nomalization factor. But, due to the scaling
 *   performed during the forward-backward recursions, we have to compute it at
 *   each positions and separately for unigrams and bigrams using
 *       for unigrams: Z_θ(t) = ∑_y α_t(y) β_t(y)
 *       for bigrams:  Z_θ(t) = ∑_y α_t(y) β_t(y) / α-scale_t
 *   with α-scale_t the scaling factor used for the α vector at position t
 *   in the forward recursion.
 */
void grd_fwd(grd_st_t *grd_st, const tok_t *tok) {
    const mdl_t *mdl = grd_st->mdl;
    const uint32_t T = tok->len;
    const uint64_t A = mdl->nfws;
    const int32_t S = mdl->reader->maxSegment;
    // const double (*psi)[F] = (void *) grd_st->psi;
    double (*alpha)[T + 1][A] = (void *) grd_st->alpha;
    // double (*beta )[T + 1][B] = (void *) grd_st->beta;
    uint64_t (*featureMap)[mdl->npats][mdl->nobs] = (void *)mdl->featureMap;

    uint32_t i, j;
    uint64_t featId;
    id_map_t (*obMap)[T][S] = (void *) tok->observationMapjp;
    for (i = 1; i < A; ++i)
        // for (j = 0; j <= T; ++j)
        (*alpha)[0][i] = -INFINITY;
    (*alpha)[0][0] = 0.0;
    // scale[0] = xvm_unit((*alpha)[0], (*alpha)[0], Y);
    for (j = 0; j < T; ++j) {
        for (i = 0; i < A; i++) {
            (*alpha)[j + 1][i] = -INFINITY;
            int y = mdl->lastForwardStateLabel[i];
            int maxmem = y == -1 ? 0 : mdl->reader->maxMemory[y];
            transition_map_t prevState = mdl->forwardTransition[i];
            int d = 0;
            while (d < maxmem && d <= j) {
                id_map_t observationMapjd = (*obMap)[j - d][d];
                for (int l = 0; l < prevState.len; ++l) {
                    uint64_t pkId = prevState.idsOne[l];
                    uint64_t pkyId = prevState.idsTwo[l];
                    double featureScore = 0.0;
                    for (uint32_t patIndex = 0; patIndex < mdl->allSuffixes[pkyId].len; ++patIndex) {
                        for (uint32_t obId = 0; obId < observationMapjd.len; ++obId) {
                            uint64_t patId = mdl->allSuffixes[pkyId].ids[patIndex];
                            featId = (*featureMap)[patId][observationMapjd.ids[obId]];
                            featureScore += featId > 0 ? mdl->theta[featId] : 0;
                        }
                    }
                    (*alpha)[j + 1][i] = logSumExp((*alpha)[j + 1][i], (*alpha)[j - d][pkId] + featureScore);
                }
                ++d;
            }

        }
        // scale[j] = xvm_unit((*alpha)[j], (*alpha)[j], A);
    }
    return;
}

void grd_bwd(grd_st_t *grd_st, const tok_t *tok) {
    const mdl_t *mdl = grd_st->mdl;
    const uint64_t Y = mdl->nlbl;
    const uint32_t T = tok->len;
    const uint64_t B = mdl->nbws;
    const int32_t S = mdl->reader->maxSegment;
    double (*beta )[T + 1][B] = (void *) grd_st->beta;
    // double *scale = grd_st->scale;
    // double *unorm = grd_st->unorm;
    // double *bnorm = grd_st->bnorm;
    id_map_t (*obMap)[T][S] = (void *) tok->observationMapjp;
    uint64_t (*featureMap)[mdl->npats][mdl->nobs] = (void *)mdl->featureMap;
    int i, j, y, skId, minseq, d;
    double featureScore;
    uint64_t patId, featId;
    // id_map_t *observationMapjd = NULL;
    // uint64_t *skIds;

    for (j = T; j >= 0; --j) {
        for (i = 0; i < B; ++i) {
            (*beta)[j][i] = 0;
        }
    }

    for (j = T - 1; j > 0; --j) {
        for (i = 0; i < B; ++i) {
            (*beta)[j][i] = -INFINITY;
            for (y = 0; y < Y; ++y) {
                skId = mdl->backwardTransition[i * Y + y];
                if (skId != -1) {
                    minseq = min((uint32_t) mdl->reader->maxMemory[y], T - j);
                    for (d = 0; d < minseq; ++d) {
                        id_map_t *observationMapjd = &((*obMap)[j][d]);
                        featureScore = 0.0;
                        for (uint32_t patIndex = 0; patIndex < mdl->allSuffixes[skId].len; ++patIndex) {
                            for (uint32_t obId = 0; obId < observationMapjd->len; ++obId) {
                                patId = mdl->allSuffixes[skId].ids[patIndex];
                                featId = (*featureMap)[patId][observationMapjd->ids[obId]];
                                featureScore += featId != 0 ? mdl->theta[featId] : 0;
                            }
                        }
                        (*beta)[j][i] = logSumExp((*beta)[j][i], (*beta)[j + d + 1][skId] + featureScore);
                    }
                }
            }
        }
    }
    return;
}

void grd_logZx(grd_st_t *grd_st, const tok_t *tok) {
    const mdl_t *mdl = grd_st->mdl;
    const uint64_t A = mdl->nfws;
    const uint32_t T = tok->len;
    const double (*alpha)[T+1][A] = (void *)grd_st->alpha;
    double logZx = -INFINITY;
    for (uint32_t i = 0; i < A; ++i) {
        logZx = logSumExp(logZx, (*alpha)[T][i]);
    }
    grd_st->logZx = logZx;
    return;
}

void grd_marginal_exp(grd_st_t *grd_st, const tok_t *tok) {
    const mdl_t *mdl = grd_st->mdl;
    const uint64_t F = mdl->nftr;
    const uint32_t T = tok->len;
    const uint64_t A = mdl->nfws;
    const uint64_t B = mdl->nbws;
    const uint64_t P = mdl->npats;
    const int32_t S = mdl->reader->maxSegment;

    double (*beta)[T + 1][B] = (void *) grd_st->beta;
    double *expec = grd_st->expec;
    double (*alpha)[T + 1][A] = (void *) grd_st->alpha;
    uint64_t (*featureMap)[mdl->npats][mdl->nobs] = (void *)mdl->featureMap;
    id_map_t (*obMap)[T][S] = (void *) tok->observationMapjp;
    id_map_t *observationMapjd;
    uint32_t obId;

    int i, j, y, d, maxmem, maxLen, patIndex;
    double featureScore;
    uint64_t piId, piyId, zId, patId, featId;
    double (*marginal)[P][T][S] = (void *)grd_st->marginal;
    (*marginal)[0][0][0] = 1;
    for (i = 0; i < F; ++i) {
        expec[i] = 0;
    }
    for (zId = 0; zId < P; ++zId) {
        y = mdl->lastPatternLabel[zId];
        maxmem = y == -1 ? 0 : mdl->reader->maxMemory[y];
        for (i = 0; i < T; ++i) {
            maxLen = min(maxmem, (int)T-i);
            for (d = 0; d < S; ++d) {
                (*marginal)[zId][i][d] = 0;
            }
            for (d = 0; d < maxLen; ++d) {
                observationMapjd = &((*obMap)[i][d]);
                (*marginal)[zId][i][d] = -INFINITY;
                for (j = 0; j < mdl->patternTransition[zId].len; ++j) {
                    piId = mdl->patternTransition[zId].idsOne[j];
                    piyId = mdl->patternTransition[zId].idsTwo[j];
                    featureScore = 0.0;
                    for (patIndex = 0; patIndex < mdl->allSuffixes[piyId].len; ++patIndex) {
                        for (obId = 0; obId < observationMapjd->len; ++obId) {
                            patId = mdl->allSuffixes[piyId].ids[patIndex];
                            featId = (*featureMap)[patId][observationMapjd->ids[obId]];
                            featureScore += (featId != 0) ? mdl->theta[featId] : 0;
                        }
                    }
                    (*marginal)[zId][i][d] = logSumExp((*marginal)[zId][i][d], (*alpha)[i][piId] + (*beta)[i + d + 1][piyId] + featureScore);
                }
                (*marginal)[zId][i][d] = exp((*marginal)[zId][i][d] - grd_st->logZx);
                for (obId = 0; obId < observationMapjd->len; ++obId) {
                    featId = (*featureMap)[zId][observationMapjd->ids[obId]];
                    expec[featId] +=  featId != 0 ? (*marginal)[zId][i][d] : 0;
                }
            }
        }
    }
    // free(observationMapjd);
    return;
}

void grd_loss(grd_st_t *grd_st, const tok_t *tok) {
    const mdl_t *mdl = grd_st->mdl;
    const uint64_t F = mdl->nftr;
    for (uint64_t i = 0; i < F; ++i) {
        grd_st->g[i] = grd_st->expec[i] - tok->empiricalScore[i];
        grd_st->lloss -= mdl->theta[i] * tok->empiricalScore[i];
    }
    grd_st->lloss += grd_st->logZx;
}

void grd_docrf(grd_st_t *grd_st, const tok_t *tok) {
    // const mdl_t *mdl = grd_st->mdl;
    grd_st->first = 0;
    grd_st->last = tok->len - 1;

    grd_fwd(grd_st, tok);
    grd_bwd(grd_st, tok);
    grd_logZx(grd_st, tok);

    grd_marginal_exp(grd_st, tok);
    grd_loss(grd_st, tok);
}

/******************************************************************************
 * Dataset gradient computation
 *
 *   This section is responsible for computing the gradient of the
 *   log-likelihood function to optimize over the full training set.
 *
 *   The gradient computation is multi-threaded, you first have to call the
 *   function 'grd_setup' to prepare the workers pool, and next you can use
 *   'grd_gradient' to ask for the full gradient as many time as you want. Each
 *   time the gradient is computed over the full training set, using the curent
 *   value of the parameters and applying the regularization. If need the
 *   pseudo-gradient can also be computed. When you have done, you have to call
 *   'grd_cleanup' to free the allocated memory.
 *
 *   This require an additional vector of size <nftr> per thread after the
 *   first, so it can take a lot of memory to compute big models on a lot of
 *   threads. It is strongly discouraged to ask for more threads than you have
 *   cores, or to more thread than you have memory to hold vectors.
 ******************************************************************************/

/* grd_stcheck:
 *   Check that enough memory is allocated in the gradient object so that the
 *   linear-chain codepath can be computed for a sequence of the given length.
 */
void grd_stcheck(grd_st_t *grd_st, uint32_t len) {
    // Check if user ask for clearing the state tracker or if he requested a
    // bigger tracker. In this case we have to free the previous allocated
    // memory.
    if (len == 0 || (len > grd_st->len && grd_st->len != 0)) {
        xvm_free(grd_st->marginal);
        grd_st->marginal = NULL;
        xvm_free(grd_st->alpha);
        grd_st->alpha = NULL;
        xvm_free(grd_st->beta);
        grd_st->beta = NULL;
        xvm_free(grd_st->unorm);
        grd_st->unorm = NULL;
        xvm_free(grd_st->bnorm);
        grd_st->bnorm = NULL;
        xvm_free(grd_st->scale);
        grd_st->scale = NULL;
        grd_st->len = 0;
    }
    if (len == 0 || len <= grd_st->len)
        return;
    // If we are here, we have to allocate a new state. This is simple, we
    // just have to take care of the special case for sparse mode.
    const uint64_t F = grd_st->mdl->nftr;
    const uint32_t T = len;
    const uint64_t A = grd_st->mdl->nfws;
    const uint64_t B = grd_st->mdl->nbws;
    const int32_t S = grd_st->mdl->reader->maxSegment;
    const uint64_t P = grd_st->mdl->npats;
    grd_st->marginal = xvm_new(P * T * S);
    grd_st->alpha = xvm_new((T+1) * A);
    grd_st->beta = xvm_new((T+1) * B);
    grd_st->expec = xvm_new(F);
    grd_st->len = len;
}

/* grd_stnew:
 *   Allocation memory for gradient computation state. This allocate memory for
 *   the longest sequence present in the data set.
 */
grd_st_t *grd_stnew(mdl_t *mdl, double *g) {
    grd_st_t *grd_st = xmalloc(sizeof(grd_st_t));
    grd_st->mdl = mdl;
    grd_st->len = 0;
    grd_st->g = g;
    grd_st->marginal = NULL;
    grd_st->alpha = NULL;
    grd_st->beta = NULL;
    grd_st->unorm = NULL;
    grd_st->bnorm = NULL;
    grd_st->scale = NULL;
    return grd_st;
}

/* grd_stfree:
 *   Free all memory used by gradient computation.
 */
void grd_stfree(grd_st_t *grd_st) {
    grd_stcheck(grd_st, 0);
    free(grd_st);
}

/* grd_dospl:
 *   Compute the gradient of a single sample choosing between the maxent
 *   optimised codepath and classical one depending of the sample.
 */
void grd_dospl(grd_st_t *grd_st, const tok_t *tok) {
    grd_stcheck(grd_st, tok->len);
    // rdr_t *rdr = grd_st->mdl->reader;
    grd_docrf(grd_st, tok);
}

/* grd_new:
 *   Allocate a new parallel gradient computer. Return a grd_t object who can
 *   compute gradient over the full data set and store it in the vector <g>.
 */
grd_t *grd_new(mdl_t *mdl, double *g) {
    const uint32_t W = mdl->opt->nthread;
    grd_t *grd = xmalloc(sizeof(grd_t));
    grd->mdl = mdl;
    grd->grd_st = xmalloc(sizeof(grd_st_t *) * W);
#ifdef ATM_ANSI
    grd->grd_st[0] = grd_stnew(mdl, g);
    for (uint32_t w = 1; w < W; w++)
        grd->grd_st[w] = grd_stnew(mdl, xvm_new(mdl->nftr));
#else
    for (uint32_t w = 0; w < W; w++)
        grd->grd_st[w] = grd_stnew(mdl, g);
#endif
    return grd;
}

/* grd_free:
 *   Free all memory allocated for the given gradient computer object.
 */
void grd_free(grd_t *grd) {
    const uint32_t W = grd->mdl->opt->nthread;
#ifdef ATM_ANSI
    for (uint32_t w = 1; w < W; w++)
        xvm_free(grd->grd_st[w]->g);
#endif
    for (uint32_t w = 0; w < W; w++)
        grd_stfree(grd->grd_st[w]);
    free(grd->grd_st);
    free(grd);
}

/* grd_worker:
 *   This is a simple function who compute the gradient over a subset of the
 *   training set. It is mean to be called by the thread spawner in order to
 *   compute the gradient over the full training set.
 */
static
void grd_worker(job_t *job, uint32_t id, uint32_t cnt, grd_st_t *grd_st) {
    unused(id && cnt);
    mdl_t *mdl = grd_st->mdl;
    const dat_t *dat = mdl->train;
    // We first cleanup the gradient and value as our parent don't do it (it
    // is better to do this also in parallel)
    grd_st->lloss = 0.0;
#ifdef ATM_ANSI
    const uint64_t F = mdl->nftr;
    for (uint64_t f = 0; f < F; f++)
        grd_st->g[f] = 0.0;
#endif
    // Now all is ready, we can process our sequences and accumulate the
    // gradient and inverse log-likelihood
    uint32_t count, pos;
    while (mth_getjob(job, &count, &pos)) {
        for (uint32_t s = pos; !uit_stop && s < pos + count; s++)
            grd_dospl(grd_st, dat->tok[s]);
        if (uit_stop)
            break;
    }
}

/* grd_gradient:
 *   Compute the gradient and value of the negative log-likelihood of the model
 *   at current point. The computation is done in parallel taking profit of
 *   the fact that the gradient over the full training set is just the sum of
 *   the gradient of each sequence.
 */
double grd_gradient(grd_t *grd) {
    mdl_t *mdl = grd->mdl;
    const double *x = mdl->theta;
    const uint64_t F = mdl->nftr;
    const uint32_t W = mdl->opt->nthread;
    double *g = grd->grd_st[0]->g;
#ifndef ATM_ANSI
    for (uint64_t f = 0; f < F; f++)
        g[f] = 0.0;
#endif
    // All is ready to compute the gradient, we spawn the threads of
    // workers, each one working on a part of the data. As the gradient and
    // log-likelihood are additive, computing the final values will be
    // trivial.
    mth_spawn((func_t *) grd_worker, W, (void **) grd->grd_st,
              mdl->train->nseq, mdl->opt->jobsize);
    if (uit_stop)
        return -1.0;
    // All computations are done, it just remain to add all the gradients
    // and negative log-likelihood from all the workers.
    double fx = grd->grd_st[0]->lloss;
    for (uint32_t w = 1; w < W; w++)
        fx += grd->grd_st[w]->lloss;
    info("loss is %f.\n", fx);
#ifdef ATM_ANSI
    for (uint32_t w = 1; w < W; w++)
        for (uint64_t f = 0; f < F; f++)
            g[f] += grd->grd_st[w]->g[f];
#endif
    // If needed we clip the gradient: setting to 0.0 all coordinates where
    // the function is 0.0.
    if (mdl->opt->lbfgs.clip == true)
        for (uint64_t f = 0; f < F; f++)
            if (x[f] == 0.0)
                g[f] = 0.0;
    // Now we can apply the elastic-net penalty. Depending of the values of
    // rho1 and rho2, this can in fact be a classical L1 or L2 penalty.
    const double rho1 = mdl->opt->rho1;
    const double rho2 = mdl->opt->rho2;
    double nl1 = 0.0, nl2 = 0.0;
    for (uint64_t f = 0; f < F; f++) {
        const double v = x[f];
        g[f] += rho2 * v;
        nl1 += fabs(v);
        nl2 += v * v;
    }
    fx += nl1 * rho1 + nl2 * rho2 / 2.0;
    return fx;
}

