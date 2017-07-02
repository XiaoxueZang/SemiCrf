#ifndef decoder_h
#define decoder_h

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "wapiti.h"
#include "model.h"
#include "sequence.h"

void tag_viterbi(mdl_t *mdl, const tok_t *seq,
                 uint32_t out[]); // , double *sc, double psc[]);

void tag_label(mdl_t *mdl, FILE *fin, FILE *fout);

void tag_eval(mdl_t *mdl, double *te, double *se);

#endif

