#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/* #ifndef NDEBUG */
/* #define NDEBUG */
/* #endif         */
#include <assert.h>

#include "pagerank_pthreads/pagerank_single.h"
#include "pagerank_pthreads/utils.h"

#define _S(x) #x
#define S(x) _S(x)
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define TEST_EQ_INT(x, y) {\
  printf("Testing " S(x) " == " S(y) " at line %d: ", __LINE__);\
  if ((x) == (y)) {\
    printf(ANSI_COLOR_GREEN "SUCCESS!" ANSI_COLOR_RESET "\n");\
  }\
  else {\
    printf(ANSI_COLOR_RED "FAILURE..." ANSI_COLOR_RESET "\n");\
    printf(S(x) " is: %d\n", (x));\
    assert(0);\
  }\
}
#define TEST_EQ_FLOAT(x, y) {\
  printf("Testing " S(x) " == " S(y) " at line %d: ", __LINE__);\
  if (fabs((x) - (y)) < ERR) {\
    printf(ANSI_COLOR_GREEN "SUCCESS!" ANSI_COLOR_RESET "\n");\
  }\
  else {\
    printf(ANSI_COLOR_RED "FAILURE..." ANSI_COLOR_RESET " (epsilon is " S(ERR) ")\n");\
    printf(S(x) " is: %f\n", (x));\
    assert(0);\
  }\
}

int main()
{
  printf("Commencing test for pagerank_single.c\n");

  FLOAT x[4] = {0.0, 0.1, 0.2, 0.3};
  FLOAT y[4] = {5, 9, 2, 6};
  FLOAT res[4], tmp;

  printf("\nTesting max . . .\n");
  tmp = max(x, 4);
  TEST_EQ_FLOAT(tmp, 0.3);
  tmp = max(y, 4);
  TEST_EQ_FLOAT(tmp, 9);

  printf("\nTesting abs_diff . . .\n");
  abs_diff(x, y, res, 4);
  TEST_EQ_FLOAT(res[0], 5);
  TEST_EQ_FLOAT(res[1], 8.9);
  TEST_EQ_FLOAT(res[2], 1.8);
  TEST_EQ_FLOAT(res[3], 5.7);

  printf("\nTesting fill . . .\n");
  fill(x, 8.1, 4);
  TEST_EQ_FLOAT(x[0], 8.1);
  TEST_EQ_FLOAT(x[1], 8.1);
  TEST_EQ_FLOAT(x[2], 8.1);
  TEST_EQ_FLOAT(x[3], 8.1);

  printf("\nTesting multiply . . .\n");
  multiply(y, 0.2, 4);
  TEST_EQ_FLOAT(y[0], 1);
  TEST_EQ_FLOAT(y[1], 1.8);
  TEST_EQ_FLOAT(y[2], 0.4);
  TEST_EQ_FLOAT(y[3], 1.2);

  printf("\nTesting add . . .\n");
  add(y, 3.4, 4);
  TEST_EQ_FLOAT(y[0], 4.4);
  TEST_EQ_FLOAT(y[1], 5.2);
  TEST_EQ_FLOAT(y[2], 3.8);
  TEST_EQ_FLOAT(y[3], 4.6);

  uint **L, *C, N, E;
  FLOAT* vector;
  int cnt;
  printf("\nTesting pagerank_power . . .\n");
  printf("Input: example-matrix.txt\n");
  read_graph("example-matrix.txt", &L, &C, &N, &E);
  print_sparse_matrix(L, C, N);
  cnt = pagerank_power(L, C, &vector, N);
  TEST_EQ_INT(cnt, 25);
  TEST_EQ_FLOAT(vector[0], 0.21481051);
  TEST_EQ_FLOAT(vector[1], 0.39740004);
  TEST_EQ_FLOAT(vector[2], 0.38778944);
  free((void*)vector);
  free((void*)C);
  for (uint i = 0; i < N; ++i)
    free((void*)L[i]);
  free((void*)L);

  printf("Input: example-matrix2.txt\n");
  read_graph("example-matrix2.txt", &L, &C, &N, &E);
  print_sparse_matrix(L, C, N);
  cnt = pagerank_power(L, C, &vector, N);
  TEST_EQ_INT(cnt, 17);
  TEST_EQ_FLOAT(vector[0], 0.21087014);
  TEST_EQ_FLOAT(vector[1], 0.30048969);
  TEST_EQ_FLOAT(vector[2], 0.32721841);
  TEST_EQ_FLOAT(vector[3], 0.16142176);
  free((void*)vector);
  free((void*)C);
  for (uint i = 0; i < N; ++i)
    free((void*)L[i]);
  free((void*)L);

  return 0;
}
