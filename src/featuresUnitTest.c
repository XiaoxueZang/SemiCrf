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


static void dotrain(mdl_t *mdl) {
    // Check if the user requested the type or trainer list. If this is not
    // the case, search them in the lists.

    // 05/27 I deleted load_pat part and want to add it inside reading the training file.
    // rdr_loadpat(mdl->reader, file);
    // info("wapiti: load the pattern file");
    // qrk_lock(mdl->reader->obs, false); // Maybe I should do this function after loading the reader??

    // Load the training data. When this is done we lock the quarks as we
    // don't want to put in the model, informations present only in the
    // devlopment set.
    info("* Load training data\n");
    FILE *file = stdin;
    if (mdl->opt->input != NULL) {
        file = fopen(mdl->opt->input, "r");
        if (file == NULL)
            pfatal("cannot open input data file");
    }
    mdl->train = rdr_readdat(mdl->reader, file, true);
    // intializeForwardBackwardPattens(mdl);
    if (mdl->opt->input != NULL)
        fclose(file);

    // if (mdl->train == NULL || mdl->train->nseq == 0)
    //     fatal("no train data loaded");

    if (mdl->theta == NULL)
        info("* Initialize the model\n");
    else
        info("* Resync the model\n");
    mdl_sync(mdl);
    return;
}

int main(int argc, char *argv[argc]) {
    // We first parse command line switchs
    opt_t opt = opt_defaults;
    opt_parse(argc, argv, &opt);
    // Next we prepare the model
    mdl_t *mdl = mdl_new(rdr_new(opt.maxSegment));
    mdl->opt = &opt;
    // And switch to requested mode
    dotrain(mdl);
    mdl_free(mdl);
    return EXIT_SUCCESS;
}
