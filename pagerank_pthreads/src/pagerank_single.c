#include <stdlib.h>
#include <math.h>

#include "pagerank_pthreads/defines.h"
#include "pagerank_pthreads/pagerank_single.h"

FLOAT max(FLOAT const * x, uint N)
{
  FLOAT max_val = 0.0;
  for (uint i = 0; i < N; ++i)
    max_val = (max_val < x[i]) ? x[i] : max_val;
  return max_val;
}

void abs_diff(FLOAT const * x, FLOAT const * y, FLOAT* res, uint N)
{
  for (uint i = 0; i < N; ++i)
    res[i] = fabs(x[i] - y[i]);
}

void swap(FLOAT** x, FLOAT** y)
{
  FLOAT* tmp = *x;
  *x = *y;
  *y = tmp;
}

void fill(FLOAT* x, FLOAT value, uint N)
{
  for (uint i = 0; i < N; ++i)
    x[i] = value;
}

void multiply(FLOAT* x, FLOAT value, uint N)
{
  for (uint i = 0; i < N; ++i)
    x[i] *= value;
}

void add(FLOAT* x, FLOAT value, uint N)
{
  for (uint i = 0; i < N; ++i)
    x[i] += value;
}

/**
 * @note ERR is a preprocessor definition which determines convergence
 */
int pagerank_power(uint * const * L, uint const * C, FLOAT** x, uint N)
{
  const FLOAT p = 0.85;
  const FLOAT delta = (1 - p) / N;
  *x = (FLOAT*) calloc(1 / (FLOAT) N,  N * sizeof(FLOAT));
  FLOAT *z, *tmp;
  z = (FLOAT*) malloc(N * sizeof(FLOAT));
  if (z == NULL) exit(2);
  tmp = (FLOAT*) malloc(N * sizeof(FLOAT));
  if (tmp == NULL) exit(2);
  int cnt = 0;
  do
  {
    swap(x, &z);
    fill(*x, 0.0, N);
    for (uint i = 0; i < N; ++i)
    {
      if (C[i] == 0)
      {
        add(*x, z[i] / N, N);
      }
      else
      {
        FLOAT weight = z[i] / C[i];
        for (uint j = 0; j < C[i]; ++j)
        {
          x[0][L[i][j]] += weight;
        }
      }
    }
    multiply(*x, p, N);
    add(*x, delta, N);
    ++cnt;
    abs_diff(*x, z, tmp, N);
  } while (max(tmp, N) >= ERR);

  free((void*) tmp);
  free((void*) z);
  return cnt;
}
