#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <stdio.h>
#include <stdint.h>
#include <stdint.h>

static inline void swap(uint32_t** x, uint32_t** y)
{
  uint32_t *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;
}

static int cmpfunc(const void* a, const void* b)
{
  if (*(const uint32_t*)a < *(const uint32_t*)b) return -1;
  if (*(const uint32_t*)a == *(const uint32_t*)b) return 0;
  return 1;
}

static inline void print_array(uint32_t *array, int N)
{
  for (int i = 0; i < N; ++i) {
    printf("%u\n", array[i]);
  }
}

static inline void output_array(uint32_t* array, int N, int rank)
{
  char filename[10];
  FILE* f;
  if (rank == -1) {
    sprintf(filename, "output.txt");
  }
  else {
    sprintf(filename, "output_%d.txt", rank);
  }
  f = fopen(filename, "w");
  for (int i = 0; i < N; ++i) {
    fprintf(f, "%u\n", array[i]);
  }
  fclose(f);
}

void grama_quicksort(uint32_t* array, uint32_t* sorted_array, unsigned int N,
                     unsigned int p, int lv);

#endif  // MPI_UTILS_H
