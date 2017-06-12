#include <inttypes.h>
#include <float.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "wapiti.h"
#include "gradient.h"
#include "model.h"
#include "quark.h"
#include "reader.h"
#include "sequence.h"
#include "thread.h"
#include "tools.h"
#include "decoder.h"
#include "vmath.h"

/******************************************************************************
 * Sequence tagging
 *
 *   This module implement sequence tagging using a trained model and model
 *   evaluation on devlopment set.
 *
 *   The viterbi can be quite intensive on the stack if you push in it long
 *   sequence and use large labels set. It's less a problem than in gradient
 *   computations but it can show up in particular cases. The fix is to call it
 *   through the mth_spawn function and request enough stack space, this will be
 *   fixed in next version.
 ******************************************************************************/

/* tag_expsc:
 *   Compute the score lattice for classical Viterbi decoding. This is the same
 *   as for the first step of the gradient computation with the exception that
 *   we don't need to take the exponential of the scores as the Viterbi decoding
 *   works in log-space.
 */

typedef struct trace {
    uint32_t prev_pos;
    uint64_t pkId;
    int label;
} trace_t;

/* tag_viterbi:
 *   This function implement the Viterbi algorithm in order to decode the most
 *   probable sequence of labels according to the model. Some part of this code
 *   is very similar to the computation of the gradient as expected.
 *
 *   And like for the gradient, the caller is responsible to ensure there is
 *   enough stack space.
 */
void tag_viterbi(mdl_t *mdl, const tok_t *seq,
                 uint32_t out[], double *sc, double psc[]) {
    const uint32_t T = seq->len;
    const uint64_t A = mdl->nfws;
    uint32_t j, k, l;
    uint64_t i, pkId, pkyId;
    int maxmem, d;
    char *pky;
    double *maxScore = xvm_new((T + 1) * A);
    double (*maxScoreAr)[T + 1][A] = (void *) maxScore;
    trace_t *trace = xmalloc(sizeof(trace_t) * T * A);
    trace_t (*traceAr)[T][A] = (void *) trace;
    for (i = 0; i < A; ++i)
        (*maxScoreAr)[0][i] = -INFINITY;
    (*maxScoreAr)[0][0] = 0;
    for (j = 0; j < T; ++j) {
        for (i = 0; i < A; ++i)
            (*maxScoreAr)[j + 1][i] = -INFINITY;
        for (i = 0; i < A; ++i) {
            int y = mdl->lastForwardStateLabel[i];
            maxmem = y == -1 ? 0 : mdl->reader->maxMemory[y];
            d = 0;
            while ((d < maxmem) && ((int) (j - d) >= 0)) {
                for (k = 0; k < mdl->forwardTransition[i].len; ++k) {
                    pkId = mdl->forwardTransition[i].idsOne[k];
                    pkyId = mdl->forwardTransition[i].idsTwo[k];
                    pky = (char *) qrk_id2str(mdl->reader->backwardStateMap, pkyId);
                    feature_dat_t *features = generateObs((tok_t *) seq, mdl->reader, j - d, j, pky);
                    double featureScore = 0;
                    for (l = 0; l < features->len; ++l) {
                        char *f = concat(concat(features->features[l]->pats, "_"), features->features[l]->obs);
                        uint64_t id = qrk_str2id(mdl->reader->featList, f);
                        if (id != none)
                            featureScore += mdl->theta[id];
                    }
                    if (featureScore + (*maxScoreAr)[j - d][pkId] > (*maxScoreAr)[j + 1][i]) {
                        (*maxScoreAr)[j + 1][i] = featureScore + (*maxScoreAr)[j - d][pkId];
                        (*traceAr)[j][i].prev_pos = j - d - 1;
                        (*traceAr)[j][i].pkId = pkId;
                        (*traceAr)[j][i].label = y;
                    }
                }
                ++d;
            }
        }
    }
    double max = -INFINITY;
    trace_t *traceMax = NULL;
    for (i = 0; i < A; ++i) {
        if (max < (*maxScoreAr)[T][i]) {
            max = (*maxScoreAr)[T][i];
            traceMax = &(*traceAr)[T - 1][i];
        }
    }
    int32_t curPos = T - 1;
    while (curPos >= 0) {
        uint32_t prevPos = traceMax->prev_pos;
        uint64_t prevPat = traceMax->pkId;
        for (int32_t n = prevPos; n < curPos; ++n) {
            out[n] = (uint32_t) traceMax->label;
        }
        curPos = prevPos;
        if (curPos >= 0)
            traceMax = &(*traceAr)[prevPos][prevPat];
    }
    return;
}

/* tag_nbviterbi:
 *   This function implement the Viterbi algorithm in order to decode the N-most
 *   probable sequences of labels according to the model. It can be used to
 *   compute only the best one and will return the same sequence than the
 *   previous function but will be slower to do it.
 */ /*
void tag_nbviterbi(mdl_t *mdl, const seq_t *seq, uint32_t N,
                   uint32_t out[][N], double sc[], double psc[][N]) {
	const uint32_t Y = mdl->nlbl;
	const uint32_t T = seq->len;
	double   *vpsi  = xvm_new(T * Y * Y);
	uint32_t *vback = xmalloc(sizeof(uint32_t) * T * Y * N);
	double   (*psi) [T][Y    ][Y] = (void *)vpsi;
	uint32_t (*back)[T][Y * N]    = (void *)vback;
	double *cur = xmalloc(sizeof(double) * Y * N);
	double *old = xmalloc(sizeof(double) * Y * N);
	// We first compute the scores for each transitions in the lattice of
	// labels.
	int op;
	if (mdl->type == 1)
		op = tag_memmsc(mdl, seq, vpsi);
	else if (mdl->opt->lblpost)
		op = tag_postsc(mdl, seq, (double *)psi);
	else
		op = tag_expsc(mdl, seq, (double *)psi);
	if (mdl->opt->force)
		tag_forced(mdl, seq, vpsi, op);
	// Here also, it's classical but we have to keep the N best paths
	// leading to each nodes of the lattice instead of only the best one.
	// This mean that code is less trivial and the current implementation is
	// not the most efficient way to do this but it works well and is good
	// enough for the moment.
	// We first build the list of all incoming arcs from all paths from all
	// N-best nodes and next select the N-best one. There is a lot of room
	// here for later optimisations if needed.
	for (uint32_t y = 0, d = 0; y < Y; y++) {
		cur[d++] = (*psi)[0][0][y];
		for (uint32_t n = 1; n < N; n++)
			cur[d++] = -DBL_MAX;
	}
	for (uint32_t t = 1; t < T; t++) {
		for (uint32_t d = 0; d < Y * N; d++)
			old[d] = cur[d];
		for (uint32_t y = 0; y < Y; y++) {
			// 1st, build the list of all incoming
			double lst[Y * N];
			for (uint32_t yp = 0, d = 0; yp < Y; yp++) {
				for (uint32_t n = 0; n < N; n++, d++) {
					lst[d] = old[d];
					if (op)
						lst[d] *= (*psi)[t][yp][y];
					else
						lst[d] += (*psi)[t][yp][y];
				}
			}
			// 2nd, init the back with the N first
			uint32_t *bk = &(*back)[t][y * N];
			for (uint32_t n = 0; n < N; n++)
				bk[n] = n;
			// 3rd, search the N highest values
			for (uint32_t i = N; i < N * Y; i++) {
				// Search the smallest current value
				uint32_t idx = 0;
				for (uint32_t n = 1; n < N; n++)
					if (lst[bk[n]] < lst[bk[idx]])
						idx = n;
				// And replace it if needed
				if (lst[i] > lst[bk[idx]])
					bk[idx] = i;
			}
			// 4th, get the new scores
			for (uint32_t n = 0; n < N; n++)
				cur[y * N + n] = lst[bk[n]];
		}
	}
	// Retrieving the best paths is similar to classical Viterbi except that
	// we have to search for the N bet ones and there is N time more
	// possibles starts.
	for (uint32_t n = 0; n < N; n++) {
		uint32_t bst = 0;
		for (uint32_t d = 1; d < Y * N; d++)
			if (cur[d] > cur[bst])
				bst = d;
		if (sc != NULL)
			sc[n] = cur[bst];
		cur[bst] = -DBL_MAX;
		for (uint32_t t = T; t > 0; t--) {
			const uint32_t yp = (t != 1) ? (*back)[t - 1][bst] / N: 0;
			const uint32_t y  = bst / N;
			out[t - 1][n] = y;
			if (psc != NULL)
				psc[t - 1][n] = (*psi)[t - 1][yp][y];
			bst = (*back)[t - 1][bst];
		}
	}
	free(old);
	free(cur);
	free(vback);
	xvm_free(vpsi);
}
*/
/* tag_label:
 *   Label a data file using the current model. This output an almost exact copy
 *   of the input file with an additional column with the predicted label. If
 *   the check option is specified, the input file must be labelled and the
 *   predicted labels will be checked against the provided ones. This will
 *   output error rates during the labelling and detailed statistics per label
 *   at the end.
 */
void tag_label(mdl_t *mdl, FILE *fin, FILE *fout) {
    qrk_t *lbls = mdl->reader->lbl;
    const uint32_t Y = mdl->nlbl;
    const uint32_t N = mdl->opt->nbest;  // only output the best solution. N = 1.
    // We start by preparing the statistic collection to be ready if check
    // option is used. The stat array hold the following for each label
    //   [0] # of reference with this label
    //   [1] # of token we have taged with this label
    //   [2] # of match of the two preceding
    uint64_t tcnt = 0, terr = 0;
    uint64_t scnt = 0, serr = 0;
    uint64_t stat[3][Y];
    for (uint32_t y = 0; y < Y; y++)
        stat[0][y] = stat[1][y] = stat[2][y] = 0;
    // Next read the input file sequence by sequence and label them, we have
    // to take care of not discarding the raw input as we want to send it
    // back to the output with the additional predicted labels.
    while (!feof(fin)) {
        // So, first read an input sequence keeping the raw_t object
        // available, and label it with Viterbi.
        raw_t *raw = rdr_readraw(mdl->reader, fin);
        if (raw == NULL)
            break;
        tok_t *toks = rdr_raw2tok(mdl->reader, raw,
                                  mdl->opt->check | mdl->opt->force, false);
        const uint32_t T = toks->len;
        uint32_t *out = xmalloc(sizeof(uint32_t) * T * N);
        double *psc = xmalloc(sizeof(double) * T * N);
        double *scs = xmalloc(sizeof(double) * N);
        if (N == 1)
            tag_viterbi(mdl, toks, (uint32_t *) out, scs, (double *) psc);
        else
            fatal("Do not support the N-most probable labelling.");
        // Next we output the raw sequence with an aditional column for
        // the predicted labels
        for (uint32_t n = 0; n < N; n++) {
            if (mdl->opt->outsc)
                fprintf(fout, "# %d %f\n", (int) n, scs[n]);
            for (uint32_t t = 0; t < T; t++) {
                if (!mdl->opt->label) {
                    fprintf(fout, "%s\t", toks->toks[t][4]);
                    fprintf(fout, "%s\t", toks->toks[t][toks->len - 1]);
                }
                uint32_t lbl = out[t * N + n];
                const char *lblstr = qrk_id2str(lbls, lbl);
                fprintf(fout, "%s", lblstr);
                if (mdl->opt->outsc) {
                    fprintf(fout, "\t%s", lblstr);
                    fprintf(fout, "/%f", psc[t * N + n]);
                }
                fprintf(fout, "\n");
            }
            fprintf(fout, "\n");
        }
        fflush(fout);
        // If user provided reference labels, use them to collect
        // statistics about how well we have performed here. Labels
        // unseen at training time are discarded.
        if (mdl->opt->check) {
            bool err = false;
            uint64_t labelId;
            for (uint32_t t = 0; t < T; t++) {
                labelId = qrk_str2id(mdl->reader->lbl, toks->lbl[t]);
                if (labelId == none)
                    continue;
                stat[0][labelId]++;
                stat[1][out[t * N]]++;
                if (labelId != out[t * N])
                    terr++, err = true;
                else
                    stat[2][out[t * N]]++;
            }
            tcnt += T;
            serr += err;
        }
        // Cleanup memory used for this sequence
        free(scs);
        free(psc);
        free(out);
        rdr_freetok(toks, mdl->opt->check);
        rdr_freeraw(raw);
        // And report our progress, at regular interval we display how
        // much sequence are labelled and if possible the current tokens
        // and sequence error rates.
        if (++scnt % 1000 == 0) {
            info("%10"PRIu64" sequences labeled", scnt);
            if (mdl->opt->check) {
                const double te = (double) terr / tcnt * 100.0;
                const double se = (double) serr / scnt * 100.0;
                info("\t%5.2f%%/%5.2f%%", te, se);
            }
            info("\n");
        }
    }
    // If user have provided reference labels, we have collected a lot of
    // statistics and we can repport global token and sequence error rate as
    // well as precision recall and f-measure for each labels.
    if (mdl->opt->check) {
        const double te = (double) terr / tcnt * 100.0;
        const double se = (double) serr / scnt * 100.0;
        info("    Nb sequences  : %"PRIu64"\n", scnt);
        info("    Token error   : %5.2f%%\n", te);
        info("    Sequence error: %5.2f%%\n", se);
        info("* Per label statistics\n");
        for (uint32_t y = 0; y < Y; y++) {
            const char *lbl = qrk_id2str(lbls, y);
            const double Rc = (double) stat[2][y] / stat[0][y];
            const double Pr = (double) stat[2][y] / stat[1][y];
            const double F1 = 2.0 * (Pr * Rc) / (Pr + Rc);
            info("    %-6s", lbl);
            info("  Pr=%.2f", Pr);
            info("  Rc=%.2f", Rc);
            info("  F1=%.2f\n", F1);
        }
    }
}

/* eval_t:
 *   This a state tracker used to communicate between the main eval function and
 *   its workers threads, the <mdl> and <dat> fields are used to transmit to the
 *   workers informations needed to make the computation, the other fields are
 *   for returning the partial results.
 */
typedef struct eval_s eval_t;
struct eval_s {
    mdl_t *mdl;
    dat_t *dat;
    uint64_t tcnt;  // Processed tokens count
    uint64_t terr;  // Tokens error found
    uint64_t scnt;  // Processes sequences count
    uint64_t serr;  // Sequence error found
};

/* tag_evalsub:
 *   This is where the real evaluation is done by the workers, we process data
 *   by batch and for each batch do a simple Viterbi and scan the result to find
 *   errors.
 */
static void tag_evalsub(job_t *job, uint32_t id, uint32_t cnt, eval_t *eval) {
    unused(id && cnt);
    mdl_t *mdl = eval->mdl;
    dat_t *dat = eval->dat;
    eval->tcnt = 0;
    eval->terr = 0;
    eval->scnt = 0;
    eval->serr = 0;
    // We just get a job a process all the squence in it.
    uint32_t count, pos;
    while (mth_getjob(job, &count, &pos)) {
        for (uint32_t s = pos; s < pos + count; s++) {
            // Tag the sequence with the viterbi
            const tok_t *seq = dat->tok[s];
            const uint32_t T = seq->len;
            uint32_t *out = xmalloc(sizeof(uint32_t) * T);
            tag_viterbi(mdl, seq, out, NULL, NULL);
            // And check for eventual (probable ?) errors
            bool err = false;
            uint64_t labelId;
            for (uint32_t t = 0; t < T; t++) {
                labelId = qrk_str2id(mdl->reader->lbl, seq->lbl[t]);
                if (labelId != out[t])
                    eval->terr++, err = true;
            }
            eval->tcnt += T;
            eval->scnt += 1;
            eval->serr += err;
            free(out);
        }
    }
}

/* tag_eval:
 *   Compute the token error rate and sequence error rate over the devel set (or
 *   taining set if not available).
 */
void tag_eval(mdl_t *mdl, double *te, double *se) {
    const uint32_t W = mdl->opt->nthread;
    dat_t *dat = (mdl->devel == NULL) ? mdl->train : mdl->devel;
    // First we prepare the eval state for all the workers threads, we just
    // have to give them the model and dataset to use. This state will be
    // used to retrieve partial result they computed.
    eval_t *eval[W];
    for (uint32_t w = 0; w < W; w++) {
        eval[w] = xmalloc(sizeof(eval_t));
        eval[w]->mdl = mdl;
        eval[w]->dat = dat;
    }
    // And next, we call the workers to do the job and reduce the partial
    // result by summing them and computing the final error rates.
    mth_spawn((func_t *) tag_evalsub, W, (void *) eval, dat->nseq,
              mdl->opt->jobsize);
    uint64_t tcnt = 0, terr = 0;
    uint64_t scnt = 0, serr = 0;
    for (uint32_t w = 0; w < W; w++) {
        tcnt += eval[w]->tcnt;
        terr += eval[w]->terr;
        scnt += eval[w]->scnt;
        serr += eval[w]->serr;
        free(eval[w]);
    }
    *te = (double) terr / tcnt * 100.0;
    *se = (double) serr / scnt * 100.0;
}

