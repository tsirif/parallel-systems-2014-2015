// TODO organize headers and libs of single threaded operations
// TODO make defines.h
// TODO test single threaded (float and double)
// TODO implement output functions and move to utils.h

#ifdef DOUBLE
#define FLOAT double
#else
#define FLOAT float
#endif  // DOUBLE

#define E 0.0001

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "pagerank_pthreads/utils.h"

/**
 * @brief function that calculates pagerank vector with power method
 * @param L [uint const **] sparse transition matrix
 * @param C [uint const *] num of edges which exit a node
 * @param N [uint] num of nodes
 * @param x [FLOAT**] pointer to pagerank vector, will have final result
 * @return num of iterations needed for the pagerank vector to converge
 */
int pagerank_power(uint const ** L, uint const * C, uint N, FLOAT** x);

inline FLOAT max(FLOAT const * x, uint N);
inline void abs_diff(FLOAT const * x, FLOAT const * y, FLOAT* res, uint N);
inline void swap(FLOAT** x, FLOAT** y);
inline void fill(FLOAT* x, FLOAT value, uint N);
inline void multiply(FLOAT* x, FLOAT value, uint N);
inline void add(FLOAT* x, FLOAT value, uint N);

// TODO
void output_pagerank_vector(char const * filename,
    int cnt, FLOAT const * x, uint N, uint E);
// TODO
/* void output_ranked_nodes(char const * filename, */
/*     int cnt, FLOAT const * x, uint N, uint E);  */

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    printf("Not enough arguments: single-pr filename [out-filename(dfl out.txt) s/v(dfl v)]\n");
    exit(1);
  }
  char *input= argv[1];
  char *output = "out.txt";
  char *opt_s = "v";
  if (argc == 3)
    output = argv[2];
  if (argc == 4)
    opt_s = argv[3];
  int opt;
  if (!strcmp(opt_s, "v"))
    opt = 0;
  else if (!strcmp(opt_s, "s"))
    opt = 1;
  else
  {
    printf("Non valid option parameter. Using default: pagerank vector\n");
    opt = 0;
  }

  uint **L, *C, N, E;
  // read transition matrix from input file
  read_graph(input, &L, &C, &N, &E);

  FLOAT *x;
  int cnt;
  cnt = pagerank_power(L, C, &x, N);

  if (opt)
  {
    // output_ranked_nodes(output, cnt, x, N, E);
  }
  else
  {
    output_pagerank_vector(output, cnt, x, N, E);
  }

  for (uint i = 0; i < N; ++i)
    free((void*) L[i]);
  free((void*) L);
  free((void*) C);
  free((void*) x);
  free((void*) capacity);
  return 0;
}

int pagerank_power(uint const ** L, uint const * C, FLOAT** x, uint N)
{
  const FLOAT p = 0.85;
  const FLOAT delta = (1 - p) / N;
  *x = (FLOAT*) calloc(1 / (FLOAT) N,  N * sizeof(FLOAT));
  uint *z, *tmp;
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
  } while (max(tmp, N) >= E);

  free((void*) tmp);
  free((void*) z);
  return cnt;
}

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
