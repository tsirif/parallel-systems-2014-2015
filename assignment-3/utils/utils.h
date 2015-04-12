#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#ifdef DOUBLE
#define CONF_HEIGHT 8
#define CONF_WIDTH 8
  typedef uint64_t uint;
#else
/**
* @brief The height of a tile assigned to a thread.
*/
#define CONF_HEIGHT 4
/**
* @brief The width of a tile assigned to a thread.
*/
#define CONF_WIDTH 8
  typedef uint32_t uint;
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
static inline void swap(int **a, int **b)
{
  int *t;
  t = *a;
  *a = *b;
  *b = t;
}

static inline void swap_uint(uint **a, uint **b){
  uint *t;
  t = *a;
  *a = *b;
  *b = t;
}

/* Position of i-th row j-th element using our current data arrangement. */

static void read_from_file(int* X, const char* filename, int M, int N)
{
  FILE *fp = fopen(filename, "r+");
  int size = fread(X, sizeof(int), M * N, fp);
  printf("elements: %d\n", size);
  fclose(fp);
}

static void save_table(int *X, int M, int N, const char *filename)
{
  FILE *fp;
  printf("Saving table in file %s\n", filename);
  fp = fopen(filename, "w+");
  fwrite(X, sizeof(int), M*N, fp);
  fclose(fp);
}

static void generate_table(int *X, int M, int N)
{
  srand(time(NULL));
  int counter = 0;

  for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
          X[i * N + j] = ( (float)rand() / (float)RAND_MAX ) < THRESHOLD;
          counter += X[i * N + j];
      }
  }

  printf("Number of non zerow elements: %d\n", counter);
  printf("Perncent: %f\n", (float)counter / (float)(M * N));
}

static void print_table(int *A, int M, int N)
{
  for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j)
          printf("%s%d "ANSI_COLOR_RESET, A[i * N + j] ? ANSI_COLOR_BLUE : ANSI_COLOR_RED, A[i * N + j]);

      printf("\n");
  }

  printf("\n");
}

#endif
