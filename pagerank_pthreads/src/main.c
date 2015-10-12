#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "pagerank_pthreads/defines.h"
#include "pagerank_pthreads/utils.h"
#ifndef PTHREADS
#include "pagerank_pthreads/pagerank_single.h"
#else
#include "pagerank_pthreads/pagerank_pthreads.h"
#endif  // PTHREADS

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
  if (argc > 2)
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

  struct timeval startwtime, endwtime;

  printf("Reading graph and save it to a matrix\n");
#ifndef PTHREADS
  uint **L, *C, N, E;
  read_graph(input, &L, &C, &N, &E);
#else
  uint **R, *RC, *LC, N, E;
  read_graph_reverse(input, &R, &RC, &LC, &N, &E);
#endif  // PTHREADS

  FLOAT *x;
  int cnt;

  printf("Execute power pagerank algorithm\n");
  gettimeofday(&startwtime, NULL);
#ifndef PTHREADS
  cnt = pagerank_power(L, C, &x, N);
#else
  cnt = pagerank_power(R, RC, LC, &x, N);
#endif  // PTHREADS
  gettimeofday(&endwtime, NULL);
  double pagerank_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
      /1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
  printf("Time to compute pagerank vector: %f\n", pagerank_time);

  printf("Output result to file\n");
  if (opt)
  {
    output_ranked_nodes(output, input, cnt, x, N, E);
  }
  else
  {
    output_pagerank_vector(output, input, cnt, x, N, E);
  }

#ifndef PTHREADS
  for (uint i = 0; i < N; ++i)
    free((void*) L[i]);
  free((void*) L);
  free((void*) C);
#else
  for (uint i = 0; i < N; ++i)
    free((void*) R[i]);
  free((void*) R);
  free((void*) RC);
  free((void*) LC);
#endif  // PTHREADS
  free((void*) x);
  free((void*) capacity);
  return 0;
}
