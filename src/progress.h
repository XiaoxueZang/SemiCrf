#ifndef progress_h
#define progress_h

#include <stdbool.h>
#include <stdint.h>

#include "wapiti.h"
#include "model.h"

extern bool uit_stop;

void uit_setup(mdl_t *mdl);
void uit_cleanup(mdl_t *mdl);
// bool uit_progress(mdl_t *mdl, uint32_t it, double obj);

#endif

