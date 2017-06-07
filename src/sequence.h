#ifndef sequence_h
#define sequence_h

#include <stddef.h>
#include <stdint.h>

// #include "model.h"

/*******************************************************************************
 * Sequences and Dataset objects
 *
 *   Sequences represent the input data feeded by the user in Wapiti either for
 *   training or labelling. The internal form used here is very different from
 *   the data read from files and the convertion process is done in three steps
 *   illustrated here:
 *         +------+     +-------+     +-------+     +-------+
 *         | FILE | --> | raw_t | --> | tok_t | --> | seq_t |
 *         +------+     +-------+     +-------+     +-------+
 *   First the sequence is read as a set of lines from the input file, this
 *   give a raw_t object. Next this set of lines is split in tokens and
 *   eventually the last one is separated as it will become a label, this result
 *   in a tok_t object.
 *   The last step consist in applying all the patterns givens by the user to
 *   extract from these tokens the observations made on the sequence in order to
 *   build the seq_t object which can be used by the trainer and tagger.
 *
 *   A dataset object is just a container for a list of sequences in internal
 *   form used to store either training or development set.
 *
 *   All the convertion process is driven by the reader object and, as it is
 *   responsible for creating the objects with a quite special allocation
 *   scheme, we just have to implement function for freeing these objects here.
 ******************************************************************************/

/* raw_t:
 *   Data-structure representing a raw sequence as a set of lines read from the
 *   input file. This is the result of the first step of the interning process.
 *   We keep this form separate from the tokenized one as we want to be able to
 *   output the sequence as it was read in the labelling mode.
 *
 *   This represent a sequence of lengths <len> and for each position 't' you
 *   find the corresponding line at <lines>[t].
 *
 *   The <lines> array is allocated with data structure, and the different lines
 *   are allocated separatly.
 */
typedef struct raw_s raw_t;
struct raw_s {
	uint32_t  len;      //   T     Sequence length
	char     *lines[];  //  [T]    Raw lines directly from file
};



typedef struct id_map_s id_map_t;
struct id_map_s {
	uint32_t len;
	uint64_t *ids;
};

/* tok_t:
 *   Data-structure representing a tokenized sequence. This is the result of the
 *   second step of the interning process after the raw sequence have been split
 *   in tokens and eventual labels separated from the observations.
 *
 *   For each position 't' in the sequence of length <len>, you find at <lbl>[t]
 *   the eventual label provided in input file, and at <toks>[t] a list of
 *   string tokens of length <cnts>[t].
 *
 *   Memory allocation here is a bit special as the first token at each position
 *   point to a memory block who hold a copy of the raw line. Each other tokens
 *   and the label are pointer in this block. This reduce memory fragmentation.
 */
// this stores a sequence: lbl is a list of labels and toks is a list of words/tokens.
typedef struct tok_s tok_t;
struct tok_s {
	uint32_t maxOrder;
	uint32_t   len;     //   T     Sequence length
	char     **lbl;     //  [T]    List of labels strings
	uint32_t  *cnts;    //  [T]    Length of tokens lists
	uint32_t *segs; // [T] start position
	uint32_t *sege; // [T] end position
	uint32_t *segl; // [t] segment length
	char     **cur_t; // current token
	size_t maxLabelLen;
	id_map_t *observationMapjp;   // [S][maxSegment]
	double *empiricalScore; // [F]
	char     **toks[];  //  [T][]  Tokens lists
};

/* seq_t:
 *   Data-structure representing a sequence of length <len> in the internal form
 *   used by the trainers and the tagger. For each position 't' in the sequence
 *   (0 <= t < <len>) there is some observations made on the data and an
 *   eventual label if provided in the input file.
 *
 *   There is two kind of features: unigrams and bigrams one, build by combining
 *   one observation and one or two labels. At position 't', the unigrams
 *   features are built using the list of observations from <uobs>[t] which
 *   contains <ucnt>[t] items, and the observation at <lbl>[t]. The bigrams
 *   features are obtained in the same way using <bobs> and <bcnt>, and have to
 *   be combined also with <lbl>[t-1].
 *
 *   If the sequence is read from a file without label, as it is the case in
 *   labelling mode, the <lbl> field will be NULL and so, the sequence cannot be
 *   used for training.
 *
 *   The raw field is private and used internaly for efficient memory
 *   allocation. This allow to allocate <lbl>, <*cnt>, and all the list in
 *   <*obs> with the datastructure itself.
 */
typedef struct pos_s pos_t;
typedef struct seq_s seq_t;
struct seq_s {
	uint32_t  len;
	uint64_t *raw;

	struct pos_s {
		uint32_t  lbl;
		uint32_t  ucnt,  bcnt; // number of unigram features and bigram features;
		uint64_t *uobs, *bobs; // pointers to unigram features and bigram features;
	} pos[];
	// tok_t *tok;
};

/* dat_t:
 *   Data-structure representing a full dataset: a collection of sequences ready
 *   to be used for training or to be labelled. It keep tracks of the maximum
 *   sequence length as the trainer need this for memory allocation. The dataset
 *   contains <nseq> sequence stored in <seq>. These sequences are labeled only
 *   if <lbl> is true.
 */

typedef struct dat_s dat_t;
struct dat_s {
	uint32_t   max_seg;  // max segment, -1 if not semi; else the maximum length of segment
	bool       lbl;   //         True iff sequences are labelled
	uint32_t   mlen;  //         Length of the longest sequence in the set
	uint32_t   nseq;  //   S     Number of sequences in the
	tok_t    **tok;   //  [S]    List of sequences
};

#endif
