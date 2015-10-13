#include <stdlib.h>
#include <stdio.h>

#include "pagerank_pthreads/defines.h"
#include "pagerank_pthreads/utils.h"

uint* capacity = NULL;
static FLOAT const * pagerank_vector = NULL;

void append(uint** L, uint* C, uint* cap, uint index, uint value)
{
  if (C[index] == cap[index])
  {
    cap[index] += DFL_CAPACITY;
    uint* tmp;
    tmp = (uint*) realloc(L[index], cap[index] * sizeof(uint));
    if (tmp == NULL)
    {
      printf("Error resizing L[%u] to capacity %u\n", index, cap[index]);
      exit(1);
    }
    L[index] = tmp;
  }
  L[index][C[index]] = value;
  C[index] += 1;
}

void reverse(uint** L, uint* LC, uint N, uint*** R, uint** RC)
{
  if (capacity) free((void*)capacity);
  capacity = NULL;
  capacity = (uint*) malloc(N * sizeof(uint));
  if (capacity == NULL) exit(-1);
  *R = (uint**) malloc(N * sizeof(uint*));
  if (*R == NULL) exit(-1);
  *RC = (uint*) malloc(N * sizeof(uint));
  if (*RC == NULL) exit(-1);
  for (uint i = 0; i < N; i++)
  {
    RC[0][i] = 0.0;
    capacity[i] = DFL_CAPACITY;
    R[0][i] = (uint*) malloc(DFL_CAPACITY * sizeof(uint));
    if (R[0][i] == NULL) exit(-1);
  }

  uint i, j;
  for (i = 0; i < N; ++i)
  {
    for (j = 0; j < LC[i]; ++j)
    {
      append(*R, *RC, capacity, L[i][j], i);
    }
  }
}

void print_sparse_matrix(uint** L, uint* C, uint N)
{
  printf("======= Printing sparse matrix =======\n");
  printf("This matrix has %u nodes.\n", N);
  for (uint i = 0; i < N; ++i)
  {
    printf("from node: %u (goes to %u nodes)\n", i, C[i]);
    printf("to: ");
    for (uint j = 0; j < C[i]; ++j)
    {
      printf("%u ", L[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

/**
 * @note to be called only when pagerank_vector points to an allocated array
 */
int rank(const void * p1, const void * p2)
{
  uint a = *(uint*)p1;
  uint b = *(uint*)p2;
  if (pagerank_vector[a] > pagerank_vector[b]) return -1;
  if (pagerank_vector[a] == pagerank_vector[b]) return 0;
  return 1;
}

/**
 * @note ERR is a preprocessor definition which determines convergence
 */
void output_pagerank_vector(char const * output, char const * input,
    int cnt, FLOAT const * x, uint N, uint E)
{
  FILE *fout;
  fout = fopen(output, "w");
  if (fout == NULL)
  {
    printf("Error opening output file: %s\n", output);
    exit(1);
  }

  fprintf(fout, "# Output of power pagerank algorithm on graph of %s\n", input);
  fprintf(fout, "# Nodes: %u Edges: %u\n", N, E);
  fprintf(fout, "# Converged in %d iterations with %f convergence error\n",
      cnt, ERR);
  fprintf(fout, "# pagerank vector: node - pagerank-probability (permyriad)\n");
  for (uint i = 0; i < N - 1; ++i)
  {
    fprintf(fout, "%u %.1f\n", i, x[i] * 10000);
  }
  fprintf(fout, "%u %.1f", N - 1, x[N - 1] * 10000);

  fclose(fout);
}

/**
 * @note ERR is a preprocessor definition which determines convergence
 */
void output_ranked_nodes(char const * output, char const * input,
    int cnt, FLOAT const * x, uint N, uint E)
{
  FILE *fout;
  fout = fopen(output, "w");
  if (fout == NULL)
  {
    printf("Error opening output file: %s\n", output);
    exit(1);
  }

  fprintf(fout, "# Output of power pagerank algorithm on graph of %s\n", input);
  fprintf(fout, "# Nodes: %u Edges: %u\n", N, E);
  fprintf(fout, "# Converged in %d iterations with %f convergence error\n",
      cnt, ERR);
  fprintf(fout, "# ranked nodes: node - pagerank-probability (permyriad)\n");

  uint* nums;
  uint i;
  nums = (uint*) malloc(N * sizeof(uint));
  for (i = 0; i < N; ++i)
    nums[i] = i;

  pagerank_vector = x;
  qsort(nums, N, sizeof(uint), rank);

  for (uint i = 0; i < N - 1; ++i)
  {
    fprintf(fout, "%u %.1f\n", nums[i], x[nums[i]] * 10000);
  }
  fprintf(fout, "%u %.1f", nums[N - 1], x[nums[N - 1]] * 10000);

  fclose(fout);
}
