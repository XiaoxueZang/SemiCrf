/*
 *      Wapiti - A linear-chain CRF tool
 *
 * Copyright (c) 2009-2013  CNRS
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef reader_h
#define reader_h

#include <stdbool.h>
#include <stdio.h>
#include <stddef.h>

#include "wapiti.h"
#include "pattern.h"
#include "quark.h"
#include "sequence.h"
#include "features.h"

/* rdr_t:
 *   The reader object who hold all informations needed to parse the input file:
 *   the patterns and quark for labels and observations. We keep separate count
 *   for unigrams and bigrams pattern for simpler allocation of sequences. We
 *   also store the expected number of column in the input data to check that
 *   pattern are appliables.
 */
typedef struct list_s list_t;
struct list_s {
    uint32_t key;
    uint32_t val;
    list_t *next;
};


typedef struct rdr_s rdr_t;
struct rdr_s {
    int32_t maxSegment;
    // bool       autouni;    //      Automatically add 'u' prefix
    uint64_t npats;      //  P   Total number of patterns
    uint64_t nfeats;  //      Number of unigram and bigram patterns
    uint64_t ntoks;      //      Expected number of tokens in input
    uint64_t nforwardStateMap;
    uint64_t nbackwardStateMap;
    // pat_t    **pats;       // [P]  List of precompiled patterns
    qrk_t *lbl;        //      Labels database
    qrk_t *obs;        //      Observation database
    qrk_t *pats;       //      patterns database
    qrk_t *featList;   //      featureList
    qrk_t *forwardStateMap;
    qrk_t *backwardStateMap;
    // qrk_t *labelPatBase;
    // uint32_t **featMap;     //      featureMap
    int32_t *maxMemory;  //      maxMemory: maximum length of continuous labels. the index is the id of (qrk_t *)lbl.
    // uint32_t  *lastForwardStateLabel;
    list_t *backwardTransition;
    uint32_t backwardTransitionLen;
    list_t *allSuffixes;
    uint32_t allSuffixesLen;
    list_t lastForwardStateLabel;
    list_t lastPatternLabel;
    list_t backwardStateList;
};

rdr_t *rdr_new(uint32_t maxSegment);

void rdr_free(rdr_t *rdr);

void rdr_freeraw(raw_t *raw);

void rdr_freeseq(seq_t *seq);

void rdr_freedat(dat_t *dat);

void rdr_loadpat(rdr_t *rdr, FILE *file);

raw_t *rdr_readraw(rdr_t *rdr, FILE *file);

tok_t *rdr_raw2tok(rdr_t *rdr, const raw_t *raw, bool lbl);

tok_t *rdr_readtok(rdr_t *rdr, FILE *file, bool lbl);

dat_t *rdr_readdat(rdr_t *rdr, FILE *file, bool lbl);

// void rdr_load(rdr_t *rdr, FILE *file);

// void rdr_save(const rdr_t *rdr, FILE *file);

char *rdr_readline(FILE *file);

#endif

