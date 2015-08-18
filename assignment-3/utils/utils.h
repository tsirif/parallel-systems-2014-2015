#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>

#ifdef DOUBLE
#define CONF_HEIGHT 8
#define CONF_WIDTH 8
#define ONE 1lu
#define ZERO 0lu
#define BIT "64"
typedef uint64_t pint;
#else
/**
* @brief The height of a tile assigned to a thread.
*/
#define CONF_HEIGHT 4
/**
* @brief The width of a tile assigned to a thread.
*/
#define CONF_WIDTH 8
#define ONE 1u
#define ZERO 0u
#define BIT "32"
typedef uint32_t pint;
#endif

/**
 * @brief The number of iterations (life generations) over the GOL matrix.
 */
#define DFL_RUNS 10

#define THRESHOLD 0.4

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define POS(i, j) (i*N + j)

/* swap 2 int* pointers */
inline void swap(int **a, int **b)
{
  int *t;
  t = *a;
  *a = *b;
  *b = t;
}

inline void swap_p(pint **a, pint **b)
{
  pint *t;
  t = *a;
  *a = *b;
  *b = t;
}

extern void read_from_file(int* X, const char* filename, int M, int N);
extern void save_table(int* X, int M, int N, const char* filename);
extern void generate_table(int* X, int M, int N);
extern void print_table(int* A, int M, int N);

#endif
