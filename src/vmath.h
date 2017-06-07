#ifndef vmath_h
#define vmath_h

#include <stdint.h>

const char *xvm_mode(void);

double *xvm_new(uint64_t N);
void    xvm_free(double x[]);

void xvm_neg(double r[], const double x[], uint64_t N);
void xvm_sub(double r[], const double x[], const double y[], uint64_t N);
void xvm_scale(double r[], const double x[], double a, uint64_t N);
double xvm_unit(double r[], const double x[], uint64_t N);

double xvm_norm(const double x[], uint64_t N);
double xvm_dot(const double x[], const double y[], uint64_t N);

void xvm_axpy(double r[], double a, const double x[], const double y[],
		uint64_t N);

void xvm_expma(double r[], const double x[], double a, uint64_t N);

double logSumExp(double a, double b);

#endif

