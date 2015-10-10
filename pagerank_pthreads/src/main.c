#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pagerank_pthreads/defines.h"
#include "pagerank_pthreads/utils.h"
#include "pagerank_pthreads/pagerank_single.h"

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
    output_ranked_nodes(output, input, cnt, x, N, E);
  }
  else
  {
    output_pagerank_vector(output, input, cnt, x, N, E);
  }

  for (uint i = 0; i < N; ++i)
    free((void*) L[i]);
  free((void*) L);
  free((void*) C);
  free((void*) x);
  free((void*) capacity);
  return 0;
}
