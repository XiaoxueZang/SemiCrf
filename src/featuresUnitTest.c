#include <ctype.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "decoder.h"
#include "model.h"
#include "options.h"
#include "progress.h"
#include "quark.h"
#include "reader.h"
#include "sequence.h"
#include "tools.h"
#include "trainers.h"
#include "wapiti.h"
#include "pattern.h"

/*******************************************************************************
 * Training
 ******************************************************************************/
static void dotrain(mdl_t *mdl) {
    // Load the pattern file. This will unlock the database if previously
    // locked by loading a model.
    if (mdl->opt->pattern != NULL) {
        info("* Load patterns\n");
        FILE *file = fopen(mdl->opt->pattern, "r");
        if (file == NULL)
            pfatal("cannot open pattern file");
        rdr_loadpat(mdl->reader, file);
        info("Finish loading the pattern file\n");
        fclose(file);
        qrk_lock(mdl->reader->obs, false);
    }
    info("* Load training data\n");
    FILE *file = stdin;
    if (mdl->opt->input != NULL) {
        file = fopen(mdl->opt->input, "r");
        if (file == NULL)
            pfatal("cannot open input data file");
    }
    mdl->train = rdr_readdat(mdl->reader, file, true, true);
    if (mdl->opt->input != NULL)
        fclose(file);

    if (mdl->train == NULL || mdl->train->nseq == 0)
        fatal("no train data loaded");

    // If present, load the development set in the model. If not specified,
    // the training dataset will be used instead.
    if (mdl->opt->devel != NULL) {
        info("* Load development data\n");
        FILE *file = fopen(mdl->opt->devel, "r");
        if (file == NULL)
            pfatal("cannot open development file");
        mdl->devel = rdr_readdat(mdl->reader, file, true, false);
        fclose(file);
    }
    if (mdl->theta == NULL)
        info("* Initialize the model\n");
    else
        info("* Resync the model\n");

    mdl_sync(mdl, true);
    // Display some statistics as we all love this.
    info("* Summary\n");
    info("    nb train:    %"PRIu32"\n", mdl->train->nseq);
    if (mdl->devel != NULL)
        info("    nb devel:    %"PRIu32"\n", mdl->devel->nseq);
    info("    nb labels:   %"PRIu32"\n", mdl->nlbl);
    info("    nb blocks:   %"PRIu64"\n", mdl->nobs);
    info("    nb features: %"PRIu64"\n", mdl->nftr);
    // And train the model...
    info("* Train the model with %s\n", mdl->opt->algo);
    uit_setup(mdl);
    trn_lbfgs(mdl);
    uit_cleanup(mdl);
    info("* Save the model\n");
    file = stdout;
    if (mdl->opt->output != NULL) {
        file = fopen(mdl->opt->output, "w");
        if (file == NULL)
            pfatal("cannot open output model");
    }
    mdl_save(mdl, file);
    if (mdl->opt->output != NULL)
        fclose(file);
    info("* Done\n");
}

/*******************************************************************************
 * Labeling
 ******************************************************************************/
static void dolabel(mdl_t *mdl) {
    // First, load the model provided by the user. This is mandatory to
    // label new datas ;-)
    if (mdl->opt->model == NULL)
        fatal("you must specify a model");
    info("* Load model\n");
    FILE *file = fopen(mdl->opt->model, "r");
    if (file == NULL)
        pfatal("cannot open input model file");
    mdl_load(mdl, file);
    // Open input and output files
    FILE *fin = stdin, *fout = stdout;
    if (mdl->opt->input != NULL) {
        fin = fopen(mdl->opt->input, "r");
        if (fin == NULL)
            pfatal("cannot open input data file");
    }
    if (mdl->opt->output != NULL) {
        fout = fopen(mdl->opt->output, "w");
        if (fout == NULL)
            pfatal("cannot open output data file");
    }
    // Do the labelling
    info("* Label sequences\n");
    tag_label(mdl, fin, fout);
    info("* Done\n");
    // And close files
    if (mdl->opt->input != NULL)
        fclose(fin);
    if (mdl->opt->output != NULL)
        fclose(fout);
}

int main(int argc, char *argv[argc]) {
    // We first parse command line switchs
    opt_t opt = opt_defaults;
    opt_parse(argc, argv, &opt);
    // Next we prepare the model
    mdl_t *mdl = mdl_new(rdr_new(opt.doSemi));
    mdl->opt = &opt;
    switch (opt.mode) {
        case 0:
            dotrain(mdl);
            break;
        case 1:
            dolabel(mdl);
            break;
    }
    // dotrain(mdl);
    mdl_free(mdl);
    return EXIT_SUCCESS;
}
