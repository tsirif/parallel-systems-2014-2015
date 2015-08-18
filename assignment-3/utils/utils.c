#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Position of i-th row j-th element using our current data arrangement. */

void read_from_file(int* X, const char* filename, int M, int N)
{
  FILE *fp = fopen(filename, "r+");
  int size = fread(X, sizeof(int), M * N, fp);
  printf("elements: %d\n", size);
  fclose(fp);
}

void save_table(int *X, int M, int N, const char *filename)
{
  FILE *fp;
  printf("Saving table in file %s\n", filename);
  fp = fopen(filename, "w+");
  fwrite(X, sizeof(int), M*N, fp);
  fclose(fp);
}

void generate_table(int *X, int M, int N)
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

void print_table(int *A, int M, int N)
{
  for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j)
          printf("%s%d "ANSI_COLOR_RESET, A[i * N + j] ? ANSI_COLOR_BLUE : ANSI_COLOR_RED, A[i * N + j]);

      printf("\n");
  }

  printf("\n");
}
