#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>

#define THRESHOLD 0.4

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

/**
 * @brief Gets last cuda error and if it's not a cudaSuccess
 * prints debug information on stderr and aborts.
 */
static inline void cudaCheckErrors(char const* msg, char const* filename, int line)
{
  do {
    cudaError_t __err = cudaGetLastError();
    if (__err != cudaSuccess) {
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",
          msg, cudaGetErrorString(__err),
          filename, line);
      exit(1);
    }
  } while (0);
}

/* swap 2 int* pointers */
static inline void swap(int **a, int **b)
{
  int *t;
  t = *a;
  *a = *b;
  *b = t;
}

/* Position of i-th row j-th element using our current data arrangement. */

static void read_from_file(int *X, char *filename, int N)
{
  FILE *fp = fopen(filename, "r+");
  int size = fread(X, sizeof(int), N, fp);
  printf("elements: %d\n", size);
  fclose(fp);
}

static void save_table(int *X, int N, const char *filename)
{
  FILE *fp;
  printf("Saving table in file %s\n", filename);
  fp = fopen(filename, "w+");
  fwrite(X, sizeof(int), N, fp);
  fclose(fp);
}

static void generate_table(int *X, int N)
{
  srand(time(NULL));
  int counter = 0;

  for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
          X[i * N + j] = ( (float)rand() / (float)RAND_MAX ) < THRESHOLD;
          counter += X[i * N + j];
      }
  }

  printf("Number of non zerow elements: %d\n", counter);
  printf("Perncent: %f\n", (float)counter / (float)(N * N));
}

static void print_table(int *A, int N)
{
  for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j)
          printf("%s%d "ANSI_COLOR_RESET, A[i * N + j] ? ANSI_COLOR_BLUE : ANSI_COLOR_RED, A[i * N + j]);

      printf("\n");
  }

  printf("\n");
}
