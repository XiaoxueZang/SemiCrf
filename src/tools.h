#ifndef tools_h
#define tools_h

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define unused(v) ((void)(v))
#define none ((uint64_t)-1)

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) < (b) ? (b) : (a))

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof((x)) / sizeof((x)[0]))
#endif

void fatal(const char *msg, ...);
void pfatal(const char *msg, ...);
void warning(const char *msg, ...);
void info(const char *msg, ...);

void *xmalloc(size_t size);
void *xrealloc(void *ptr, size_t size);
char *xstrdup(const char *str);

char *ns_readstr(FILE *file);
void ns_writestr(FILE *file, const char *str);

// char *convertUint64ToChar(uint64_t num);

#endif
