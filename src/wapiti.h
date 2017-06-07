#ifndef wapiti_h
#define wapiti_h

#define VERSION "1.5.0"

/* XVM_ANSI:
 *   By uncomenting the following define, you can force wapiti to not use SSE2
 *   even if available.
 */
//#define XVM_ANSI

/* MTH_ANSI:
 *   By uncomenting the following define, you can disable the use of POSIX
 *   threads in the multi-threading part of Wapiti, for non-POSIX systems.
 */
//#define MTH_ANSI

/* ATM_ANSI:
 *   By uncomenting the following define, you can disable the use of atomic
 *   operation to update the gradient. This imply that multi-threaded gradient
 *   computation will require more memory but is more portable.
 */
//#define ATM_ANSI

/* Without multi-threading we disable atomic updates as they are not needed and
 * can only decrease performances in this case.
 */
#ifdef MTH_ANSI
#define ATM_ANSI
#endif

#endif

