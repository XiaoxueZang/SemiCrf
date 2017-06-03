#ifndef thread_h
#define thread_h

#include <stdint.h>
#include <pthread.h>

#include "model.h"

typedef struct job_s job_t;

typedef void (func_t)(job_t *job, uint32_t id, uint32_t cnt, void *ud);

bool mth_getjob(job_t *job, uint32_t *cnt, uint32_t *pos);
void mth_spawn(func_t *f, uint32_t W, void *ud[W], uint32_t size, uint32_t batch);

#endif
