#include <stdlib.h>

#include "utils.h"

uint* capacity = NULL;

void append(uint** L, uint* C, uint index, uint value)
{
  if (C[index] == capacity[index])
  {
    capacity[index] += DFL_CAPACITY;
    L[index] = (uint*) realloc(L[index], capacity[index] * sizeof(uint));
  }
  L[index][C[index]] = value;
  C[index] += 1;
}

void reverse(uint** L, uint* LC, uint N, uint** R, uint* RC)
{
  if (capacity) free((void*)capacity);
  capacity = NULL;
  capacity = (uint*) malloc((*N) * sizeof(uint));
  R = (uint**) malloc((*N) * sizeof(uint*));
  RC = (uint*) calloc(*N * sizeof(uint));
  for (int i = 0; i < N; i++)
  {
    capacity[i] = DFL_CAPACITY;
    R[i] = (uint*) malloc(DFL_CAPACITY * sizeof(uint));
  }

  int i, j;
  for (i = 0; i < N; ++i)
  {
    for (j = 0; j < LC[i]; ++j)
    {
      append(R, RC, L[i][j], i);
    }
  }
}
