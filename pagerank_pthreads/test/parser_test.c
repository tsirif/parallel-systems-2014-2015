#include <stdio.h>
#include <stdlib.h>
#ifndef NDEBUG
#define NDEBUG
#endif
#include <assert.h>

#define S(x) #x
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define TEST_EQ_UINT(x, y) {\
  printf("Testing " S(x) " == " S(y) " at line %d: ", __LINE__);\
  if ((x) == (y)) {\
    printf(ANSI_COLOR_GREEN "SUCCESS!" ANSI_COLOR_RESET "\n");\
  }\
  else {\
    printf(ANSI_COLOR_RED "FAILURE..." ANSI_COLOR_RESET "\n");\
    printf(S(x) " is: %u\n", (x));\
    assert(0);\
  }\
}

#include "pagerank_pthreads/utils.h"

int main()
{
  printf("Commencing test for parser.c\n");

  uint **L, *C, N, E;
  printf("\nTesting read_graph . . .\n");
  read_graph("example-matrix.txt", &L, &C, &N, &E);
  print_sparse_matrix(L, C, N);

  TEST_EQ_UINT(N, 3);
  TEST_EQ_UINT(E, 4);
  TEST_EQ_UINT(C[0], 1);
  TEST_EQ_UINT(C[1], 1);
  TEST_EQ_UINT(C[2], 2);
  TEST_EQ_UINT(L[0][0], 1);
  TEST_EQ_UINT(L[1][0], 2);
  TEST_EQ_UINT(L[2][0], 0);
  TEST_EQ_UINT(L[2][1], 1);

  for (uint i = 0; i < N; ++i)
    free((void*) L[i]);
  free((void*) L);
  free((void*) C);

  uint *LC;
  printf("\nTesting read_graph_reverse . . .\n");
  read_graph_reverse("example-matrix.txt", &L, &C, &LC, &N, &E);
  print_sparse_matrix(L, C, N);

  TEST_EQ_UINT(N, 3);
  TEST_EQ_UINT(E, 4);
  TEST_EQ_UINT(LC[0], 1);
  TEST_EQ_UINT(LC[1], 1);
  TEST_EQ_UINT(LC[2], 2);
  TEST_EQ_UINT(C[0], 1);
  TEST_EQ_UINT(C[1], 2);
  TEST_EQ_UINT(C[2], 1);
  TEST_EQ_UINT(L[0][0], 2);
  TEST_EQ_UINT(L[1][0], 0);
  TEST_EQ_UINT(L[1][1], 2);
  TEST_EQ_UINT(L[2][0], 1);

  for (uint i = 0; i < N; ++i)
    free((void*) L[i]);
  free((void*) L);
  free((void*) C);
  free((void*) LC);

  free((void*) capacity);

  return 0;
}
