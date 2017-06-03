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

#ifndef pattern_h
#define pattern_h

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "sequence.h"
#include "reader.h"
#include "model.h"

typedef struct qrk_s qrk_t;
typedef struct labelPat_s labelPat_t;
typedef struct feature_dat_s feature_dat_t;
typedef struct rdr_s rdr_t;
struct labelPat_s {
    char *labelPat;
    uint32_t order;
    uint32_t length;
    uint32_t segNum;
    char **suffixes;
    char **prefixes;
};

// labelPat_t *findStructOrBuild(rdr_t *reader, char *labelPat);
labelPat_t *generateLabelPatStruct(char *labelPat);

char *generateLabelPattern(tok_t *toks, uint32_t segStart, uint32_t segEnd);

feature_dat_t *generateObs(tok_t *tok, rdr_t *reader, uint32_t segStart, uint32_t segEnd, char *labelPat);

uint64_t getLastLabelId(rdr_t *rdr, const char *p);

uint64_t getLongestIndexId(char *labelPat, qrk_t *qrk);

char *getLongestSuffix(char* labelPat, qrk_t *qrk);

#endif

