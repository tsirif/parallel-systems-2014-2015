#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "pagerank_pthreads/defines.h"
#include "pagerank_pthreads/pagerank_single.h"

inline FLOAT max(FLOAT const * x, uint N)
{
  FLOAT max_val = 0;
  for (uint i = 0; i < N; ++i)
    max_val = (max_val < x[i]) ? x[i] : max_val;
  return max_val;
}

inline void abs_diff(FLOAT const * x, FLOAT const * y, FLOAT* res, uint N)
{
  for (uint i = 0; i < N; ++i)
    res[i] = fabs(x[i] - y[i]);
}

inline FLOAT max_abs_diff(FLOAT const * x, FLOAT const * y, uint N)
{
  FLOAT max_val = 0, val = 0;
  for (uint i = 0; i < N; ++i)
  {
    val = fabs(x[i] - y[i]);
    max_val = (max_val < val) ? val : max_val;
  }
  return max_val;
}

inline void swap(FLOAT** x, FLOAT** y)
{
  FLOAT* tmp = *x;
  *x = *y;
  *y = tmp;
}

inline void fill(FLOAT* x, FLOAT value, uint N)
{
  for (uint i = 0; i < N; ++i)
    x[i] = value;
}

inline void multiply(FLOAT* x, FLOAT value, uint N)
{
  for (uint i = 0; i < N; ++i)
    x[i] *= value;
}

inline void add(FLOAT* x, FLOAT value, uint N)
{
  for (uint i = 0; i < N; ++i)
    x[i] += value;
}

/**
 * @note ERR is a preprocessor definition which determines convergence
 */
int pagerank_power(uint * const * L, uint const * C, FLOAT** x, uint N)
{
  FLOAT *z;
  *x = (FLOAT*) malloc(N * sizeof(FLOAT));
  if (*x == NULL) exit(-2);
  z = (FLOAT*) malloc(N * sizeof(FLOAT));
  if (z == NULL) exit(-2);

  struct timeval startwtime, endwtime;
  gettimeofday(&startwtime, NULL);

  const FLOAT p = 0.85;
  const FLOAT delta = (1 - p) / N;
  fill(*x, 1 / (FLOAT) N, N);
  int cnt = 0;
  FLOAT well_prob;
  do
  {
    well_prob = 0;
    swap(x, &z);
    fill(*x, 0, N);
    for (uint i = 0; i < N; ++i)
    {
      if (C[i] == 0)
      {
        well_prob += z[i];
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
    add(*x, well_prob / N, N);
    multiply(*x, p, N);
    add(*x, delta, N);
    ++cnt;
  } while (max_abs_diff(*x, z, N) >= ERR);

  gettimeofday(&endwtime, NULL);
  double pagerank_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
      /1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
  printf("Time to compute pagerank vector: %f\n", pagerank_time);

  free((void*) z);
  return cnt;
}
