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
#include <ctype.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "wapiti.h"
#include "pattern.h"
#include "quark.h"
#include "reader.h"
#include "sequence.h"
#include "tools.h"
#include "features.h"

#define MAX_LABEL_COUNT 10

/*******************************************************************************
 * Datafile reader
 *
 *   And now come the data file reader which use the previous module to parse
 *   the input data in order to produce seq_t objects representing interned
 *   sequences.
 *
 *   This is where the sequence will go through the tree steps to build seq_t
 *   objects used internally. There is two way do do this. First the simpler is
 *   to use the rdr_readseq function which directly read a sequence from a file
 *   and convert it to a seq_t object transparently. This is how the training
 *   and development data are loaded.
 *   The second way consist of read a raw sequence with rdr_readraw and next
 *   converting it to a seq_t object with rdr_raw2seq. This allow the caller to
 *   keep the raw sequence and is used by the tagger to produce a clean output.
 *
 *   There is no public interface to the tok_t object as it is intended only for
 *   internal use in the reader as an intermediate step to apply patterns.
 ******************************************************************************/

/* rdr_new:
 *   Create a new empty reader object. If no patterns are loaded before you
 *   start using the reader the input data are assumed to be already prepared
 *   list of features. They must either start with a prefix 'u', 'b', or '*', or
 *   you must set autouni to true in order to automatically add a 'u' prefix.*
*/
rdr_t *rdr_new(uint32_t maxSegment) {
	rdr_t *rdr = xmalloc(sizeof(rdr_t));
	rdr->maxSegment = maxSegment;
	rdr->npats = rdr->nfeats = rdr->npats = 0;
	rdr->ntoks = 0;
	rdr->lbl = qrk_new();
	rdr->obs = qrk_new();
	rdr->pats = qrk_new();
	rdr->featList = qrk_new();
	rdr->backwardStateMap = qrk_new();
	rdr->forwardStateMap = qrk_new();
	rdr->maxMemory = xmalloc(sizeof(uint32_t) * MAX_LABEL_COUNT);
	return rdr;
}

/* rdr_free:
 *   Free all memory used by a reader object including the quark database, so
 *   any string returned by them must not be used after this call.
 */
void rdr_free(rdr_t *rdr) {
	for (uint32_t i = 0; i < rdr->npats; i++)
		// pat_free(rdr->pats[i]);
	free(rdr->pats);
	qrk_free(rdr->lbl);
	qrk_free(rdr->obs);
	free(rdr);
}

/* rdr_freeraw:
 *   Free all memory used by a raw_t object.
 */
void rdr_freeraw(raw_t *raw) {
	for (uint32_t t = 0; t < raw->len; t++)
		free(raw->lines[t]);
	free(raw);
}

/* rdr_freeseq:
 *   Free all memory used by a seq_t object.
 */
void rdr_freeseq(seq_t *seq) {
	free(seq->raw);
	free(seq);
}

/* rdr_freedat:
 *   Free all memory used by a dat_t object.
 */
void rdr_freedat(dat_t *dat) {
	for (uint32_t i = 0; i < dat->nseq; i++)
		// rdr_freeseq(dat->tok[i]);
	free(dat->tok);
	free(dat);
}

/* rdr_readline:
 *   Read an input line from <file>. The line can be of any size limited only by
 *   available memory, a buffer large enough is allocated and returned. The
 *   caller is responsible to free it. On end-of-file, NULL is returned.
 */
char *rdr_readline(FILE *file) {
	if (feof(file))
		return NULL;
	// Initialize the buffer
	uint32_t len = 0, size = 16;
	char *buffer = xmalloc(size);
	// We read the line chunk by chunk until end of line, file or error
	while (!feof(file)) {
		if (fgets(buffer + len, size - len, file) == NULL) {
			// On NULL return there is two possible cases, either an
			// error or the end of file
			if (ferror(file))
				pfatal("cannot read from file");
			// On end of file, we must check if we have already read
			// some data or not
			if (len == 0) {
				free(buffer);
				return NULL;
			}
			break;
		}
		// Check for end of line, if this is not the case enlarge the
		// buffer and go read more data
		len += strlen(buffer + len);
		if (len == size - 1 && buffer[len - 1] != '\n') {
			size = size * 1.4;
			buffer = xrealloc(buffer, size);
			continue;
		}
		break;
	}
	// At this point empty line should have already catched so we just
	// remove the end of line if present and resize the buffer to fit the
	// data
	if (buffer[len - 1] == '\n')
		buffer[--len] = '\0';
	return xrealloc(buffer, len + 1);
}

/* rdr_loadpat:
 *   Load and compile patterns from given file and store them in the reader. As
 *   we compile patterns, syntax errors in them will be raised at this time.
 */

/* rdr_readraw:
 *   Read a raw sequence from given file: a set of lines terminated by end of
 *   file or by an empty line. Return NULL if file end was reached before any
 *   sequence was read.
 */
raw_t *rdr_readraw(rdr_t *rdr, FILE *file) {
	if (feof(file))
		return NULL;
	// Prepare the raw sequence object
	uint32_t size = 32, cnt = 0;
	raw_t *raw = xmalloc(sizeof(raw_t) + sizeof(char *) * size);
	// And read the next sequence in the file, this will skip any blank line
	// before reading the sequence stoping at end of file or on a new blank
	// line.
	while (!feof(file)) {
		char *line = rdr_readline(file);
		if (line == NULL)
			break;
		// Check for empty line marking the end of the current sequence
		int len = strlen(line);
		while (len != 0 && isspace(line[len - 1]))
			len--;
		if (len == 0) {
			free(line);
			// Special case when no line was already read, we try
			// again. This allow multiple blank lines beetwen
			// sequences.
			if (cnt == 0)
				continue;
			break;
		}
		// Next, grow the buffer if needed and add the new line in it
		if (size == cnt) {
			size *= 1.4;
			raw = xrealloc(raw, sizeof(raw_t)
			                + sizeof(char *) * size);
		}
		raw->lines[cnt++] = line; // raw -> lines is ["a 1", "b 0", ...] is one sequence.
	}
	// If no lines was read, we just free allocated memory and return NULL
	// to signal the end of file to the caller. Else, we adjust the object
	// size and return it.
	if (cnt == 0) {
		free(raw);
		return NULL;
	}
	raw = xrealloc(raw, sizeof(raw_t) + sizeof(char *) * cnt);
	raw->len = cnt;
	return raw;
}

/* rdr_mapobs:
 *   Map an observation to its identifier, automatically adding a 'u' prefix in
 *   'autouni' mode.

static uint64_t rdr_mapobs(rdr_t *rdr, const char *str) {
	if (!rdr->autouni)
		return qrk_str2id(rdr->obs, str);
	char tmp[strlen(str) + 2];
	tmp[0] = 'u';
	strcpy(tmp + 1, str);
	return qrk_str2id(rdr->obs, tmp);
}
*/

/* rdr_rawtok2seq:
 *   Convert a tok_t to a seq_t object taking each tokens as a feature without
 *   applying patterns.

static seq_t *rdr_rawtok2seq(rdr_t *rdr, const tok_t *tok) {
	const uint32_t T = tok->len;
	uint32_t size = rdr->obs[]; // the size of unifeatures for one line (one token);

	seq_t *seq = xmalloc(sizeof(seq_t) + sizeof(pos_t) * T);
	seq->raw = xmalloc(sizeof(uint64_t) * size);
	seq->len = T;
	uint64_t *raw = seq->raw;
	for (uint32_t t = 0; t < T; t++) {
		seq->pos[t].lbl = (uint32_t)-1;
		seq->pos[t].ucnt = 0;
		seq->pos[t].uobs = raw;
		for (uint32_t n = 0; n < tok->cnts[t]; n++) {
			if (!rdr->autouni && tok->toks[t][n][0] == 'b')
				continue;
			uint64_t id = rdr_mapobs(rdr, tok->toks[t][n]);
			if (id != none) {
				(*raw++) = id;
				seq->pos[t].ucnt++;
			}
		}
		seq->pos[t].bcnt = 0;
		if (rdr->autouni)
			continue;
		seq->pos[t].bobs = raw;
		for (uint32_t n = 0; n < tok->cnts[t]; n++) {
			if (tok->toks[t][n][0] == 'u')
				continue;
			uint64_t id = rdr_mapobs(rdr, tok->toks[t][n]);
			if (id != none) {
				(*raw++) = id;
				seq->pos[t].bcnt++;
			}
		}
	}
	// And finally, if the user specified it, populate the labels
	if (tok->lbl != NULL) {
		for (uint32_t t = 0; t < T; t++) {
			const char *lbl = tok->lbl[t];
			uint64_t id = qrk_str2id(rdr->lbl, lbl);
			seq->pos[t].lbl = id;
		}
	}
	return seq;
}

*/
/* rdr_raw2seq:
 *   Convert a raw sequence to a seq_t object suitable for training or
 *   labelling. If lbl is true, the last column is assumed to be a label and
 *   interned also.
 */
// sequence ex:
// angry 14 1
// is 13 0
// ...
// (one sequence of the input file)
tok_t *rdr_raw2tok(rdr_t *rdr, const raw_t *raw, bool lbl) {
	const uint32_t T = raw->len;
	// Allocate the tok_t object, the label array is allocated only if they
	// are requested by the user.
	tok_t *tok = xmalloc(sizeof(tok_t) + T * sizeof(char **));
	tok->cnts = xmalloc(sizeof(uint32_t) * T);
	tok->lbl = NULL;
	tok->segs = xmalloc(sizeof(uint32_t) * T);
	tok->sege = xmalloc(sizeof(uint32_t) * T);
	tok->segl = xmalloc(sizeof(uint32_t) * T);
	tok->cur_t = xmalloc(sizeof(char *) * T);
	tok->maxOrder = 3;
	tok->observationMapjp = NULL;
	if (lbl == true)
		tok->lbl = xmalloc(sizeof(char *) * T);
	// We now take the raw sequence line by line and split them in list of
	// tokens. To reduce memory fragmentation, the raw line is copied and
	// his reference is kept by the first tokens, next tokens are pointer to
	// this copy.
	for (uint32_t t = 0; t < T; t++) {
		// Get a copy of the raw line skiping leading space characters
		const char *src = raw->lines[t]; // the t line of a sequence ex: a b d g 1
		while (isspace(*src))
			src++;
		char *line = xstrdup(src);
		// Split one line in tokens. Structure info into toks (char **) and then put the info in toks into tok.
		// The final image of toks:
		// [['a', '1'], ['b', '0'], ..., ]
		char *toks[strlen(line) / 2 + 1];
		uint32_t cnt = 0;
		while (*line != '\0') {
			toks[cnt++] = line;   // toks[0] = &(a), the address of that token (a)
			while (*line != '\0' && !isspace(*line))
				line++;
			if (*line == '\0')
				break;
			*line++ = '\0';
			while (*line != '\0' && isspace(*line))
				line++;
		}

		// If user specified that data are labelled, move the last token
		// to the label array.
		if (lbl == true) {
			tok->lbl[t] = toks[cnt - 1];
			// put the label inside the labels;
			qrk_str2id(rdr->lbl, tok->lbl[t]);
			cnt--;
		}
		tok->cur_t[t] = toks[4]; // the current word;
		// And put the remaining tokens in the tok_t object

		// uint32_t i = t;
		if (t >0 && (strcmp(tok->lbl[t], tok->lbl[t-1]) == 0)) {
			tok->segs[t] = tok->segs[t-1];
		} else {
			tok->segs[t] = t;
		}
		tok->cnts[t] = cnt;
		tok->toks[t] = xmalloc(sizeof(char *) * cnt);
		memcpy(tok->toks[t], toks, sizeof(char *) * cnt);
	}
	tok->len = T;
	// set segmentend and segment length.
	for (int i = T-1; i >= 0; --i) {
		if (i < T-1 && (strcmp(tok->lbl[i], tok->lbl[i+1]) == 0)) {
			tok->sege[i] = tok->sege[i+1];
		} else {
			tok->sege[i] = i;
		}
		tok->segl[i] = tok->sege[i] - tok->segs[i];
	}
	updateMaxMemory(tok, rdr);
	uint32_t segStart = 0;
	uint32_t segEnd = 0;
	while (segStart < T) {
		segEnd = tok->sege[segStart];
		char *labelPat = generateLabelPattern(tok, segStart, segEnd);
		feature_dat_t *features = generateObs(tok, rdr, segStart, segEnd, labelPat);
		for (uint32_t id = 0; id < features->len; ++id) {
			putIntoDatabase(features->features[id]->obs, features->features[id]->pats, rdr);
		}
		segStart = segEnd + 1;
	}
	/*
	seq_t *seq = NULL;
	// the following part should be removed and be executed after reading all the sequences.
	// Convert the tok_t to a seq_t
	seq = rdr_rawtok2seq(rdr, tok);

	// Before returning the sequence, we have to free the tok_t
	for (uint32_t t = 0; t < T; t++) {
		if (tok->cnts[t] == 0)
			continue;
		free(tok->toks[t][0]);
		free(tok->toks[t]);
	}
	free(tok->cnts);
	if (lbl == true)
		free(tok->lbl);
	free(tok);
	 */
	return tok;
}

/* rdr_readseq:
 *   Simple wrapper around rdr_readraw and rdr_raw2seq to directly read a
 *   sequence as a seq_t object from file. This take care of all the process
 *   and correctly free temporary data. If lbl is true the sequence is assumed
 *   to be labeled.
 *   Return NULL if end of file occure before anything as been read.
 */
tok_t *rdr_readtok(rdr_t *rdr, FILE *file, bool lbl) {
	raw_t *raw = rdr_readraw(rdr, file); // return one seq (one paragraph).lines[T] is the contents of line T of seq
	// info("inside rdr_readseq: raw_t length is %d\n", raw->len);
	if (raw == NULL)
		return NULL;
	tok_t *tok = rdr_raw2tok(rdr, raw, lbl);
	rdr_freeraw(raw);
	return tok;
}

/* rdr_readdat:
 *   Read a full dataset at once and return it as a dat_t object. This function
 *   take and interpret his parameters like the single sequence reading
 *   function.
 */
dat_t *rdr_readdat(rdr_t *rdr, FILE *file, bool lbl) {
	uint32_t size = 2000;
	dat_t *dat = xmalloc(sizeof(dat_t));
	dat->nseq = 0;
	dat->mlen = 0;
	dat->lbl = lbl;
	dat->tok = xmalloc(sizeof(tok_t *) * size);
	// Load sequences
	while (!feof(file)) {
		// Read the next sequence
		tok_t *seq = rdr_readtok(rdr, file, lbl);
		if (seq == NULL)
			break;
		// Grow the buffer if needed
		if (dat->nseq == size) {
			size *= 1.4;
			dat->tok = xrealloc(dat->tok, sizeof(seq_t *) * size);
		}
		// And store the sequence
		dat->tok[dat->nseq++] = seq;
		dat->mlen = max(dat->mlen, seq->len);
		if (dat->nseq % 1000 == 0)
			info("%7"PRIu32" sequences loaded\n", dat->nseq);
	}
	// If no sequence readed, cleanup and repport
	if (dat->nseq == 0) {
		free(dat->tok);
		free(dat);
		return NULL;
	}
	// Adjust the dataset size and return
	if (size > dat->nseq)
		dat->tok = xrealloc(dat->tok, sizeof(seq_t *) * dat->nseq);
	return dat;
}


/* rdr_load:
 *   Read from the given file a reader saved previously with rdr_save. The given
 *   reader must be empty, comming fresh from rdr_new. Be carefull that this
 *   function performs almost no checks on the input data, so if you modify the
 *   reader and make a mistake, it will probably result in a crash.

void rdr_load(rdr_t *rdr, FILE *file) {
	const char *err = "broken file, invalid reader format";
	int autouni = rdr->autouni;
	fpos_t pos;
	fgetpos(file, &pos);
	if (fscanf(file, "#rdr#%"PRIu32"/%"PRIu32"/%d\n",
			&rdr->npats, &rdr->ntoks, &autouni) != 3) {
		// This for compatibility with previous file format
		fsetpos(file, &pos);
		if (fscanf(file, "#rdr#%"PRIu32"/%"PRIu32"\n",
				&rdr->npats, &rdr->ntoks) != 2)
			fatal(err);
	}
	rdr->autouni = autouni;
	rdr->nuni = rdr->nbi = 0;
	if (rdr->npats != 0) {
		rdr->pats = xmalloc(sizeof(pat_t *) * rdr->npats);
		for (uint32_t p = 0; p < rdr->npats; p++) {
			char *pat = ns_readstr(file);
			rdr->pats[p] = pat_comp(pat);
			switch (tolower(pat[0])) {
				case 'u': rdr->nuni++; break;
				case 'b': rdr->nbi++;  break;
				case '*': rdr->nuni++;
				          rdr->nbi++;  break;
			}
		}
	}
	qrk_load(rdr->lbl, file);
	qrk_load(rdr->obs, file);
}

/* rdr_save:
 *   Save the reader to the given file so it can be loaded back. The save format
 *   is plain text and portable accros computers.

void rdr_save(const rdr_t *rdr, FILE *file) {
	if (fprintf(file, "#rdr#%"PRIu32"/%"PRIu32"/%d\n",
			rdr->npats, rdr->ntoks, rdr->autouni) < 0)
		pfatal("cannot write to file");
	for (uint32_t p = 0; p < rdr->npats; p++)
		ns_writestr(file, rdr->pats[p]->src);
	qrk_save(rdr->lbl, file);
	qrk_save(rdr->obs, file);
}
*/