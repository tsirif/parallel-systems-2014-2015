#include <stdio.h>
#define NDEBUG
#include <assert.h>

#define TEST(x) {\
  printf("Testing %s at %d: ", "x", __LINE__);\
  if (x) {\
    printf("SUCCESS!\n");\
  }\
  else {\
    printf("FAILURE...\n");\
    assert(0);\
  }\
}

#include "pagerank-pthreads/utils.h"

int main()
{
  printf("Commencing test for parser.c\n");

  uint **L, *C, N, E;
  printf("Testing read_graph...\n");
  read_graph("example-matrix.txt", L, C, &N, &E);
  print_sparse_matrix(L, C, N);

  TEST(N == 3);
  TEST(E == 4);
  TEST(C[0] == 1);
  TEST(C[1] == 1);
  TEST(C[2] == 2);
  TEST(L[0][0] == 1);
  TEST(L[1][0] == 2);
  TEST(L[2][0] == 0);
  TEST(L[2][1] == 1);

  for (uint i = 0; i < N; ++i)
    free((void*) L[i]);
  free((void*) L);
  free((void*) C);

  uint *LC;
  printf("Testing read_graph...\n");
  read_graph_reverse("example-matrix.txt", L, C, LC, &N, &E);
  print_sparse_matrix(L, C, N);

  TEST(N == 3);
  TEST(E == 4);
  TEST(LC[0] == 1);
  TEST(LC[1] == 1);
  TEST(LC[2] == 2);
  TEST(C[0] == 1);
  TEST(C[1] == 2);
  TEST(C[2] == 1);
  TEST(L[0][0] == 2);
  TEST(L[1][0] == 0);
  TEST(L[1][1] == 2);
  TEST(L[2][0] == 1);

  for (uint i = 0; i < N; ++i)
    free((void*) L[i]);
  free((void*) L);
  free((void*) C);
  free((void*) LC);

  free((void*) capacity);

  return 0;
}
