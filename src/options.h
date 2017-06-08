#ifndef options_h
#define options_h

#include <stdint.h>
#include <stdbool.h>

#include "wapiti.h"

/* opt_t:
 *   This structure hold all user configurable parameter for Wapiti and is
 *   filled with parameters from command line.
 */
typedef struct opt_s opt_t;
struct opt_s {
	int       mode;
	char     *input,  *output;
	bool     doSemi;
	// Options for training
	char     *type;
	char     *algo,   *pattern;
	char     *model,  *devel;
	char     *rstate, *sstate;
	bool      compact, sparse;
	uint32_t  nthread;
	uint32_t  jobsize;
	uint32_t  maxiter;
	double    rho1,    rho2;
	// Window size criterion
	uint32_t  objwin;
	uint32_t  stopwin;
	double    stopeps;
	// Options specific to L-BFGS
	struct {
		bool     clip;
		uint32_t histsz;
		uint32_t maxls;
	} lbfgs;
	// Options specific to SGD-L1
	struct {
		double   eta0;
		double   alpha;
	} sgdl1;
	// Options specific to BCD
	struct {
		double   kappa;
	} bcd;
	// Options specific to RPROP
	struct {
		double   stpmin;
		double   stpmax;
		double   stpinc;
		double   stpdec;
		bool     cutoff;
	} rprop;
	// Options for labelling
	bool      label;
	bool      check;
	bool      outsc;
	bool      lblpost;
	uint32_t  nbest;
	bool      force;
	// Options for model dump
	int       prec;
	bool      all;
};

extern const opt_t opt_defaults;

void opt_parse(int argc, char *argv[argc], opt_t *opt);

#endif

