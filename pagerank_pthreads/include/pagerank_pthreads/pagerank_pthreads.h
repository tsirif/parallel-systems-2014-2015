#ifndef PAGERANK_PTHREADS_PAGERANK_PTHREADS_H
#define PAGERANK_PTHREADS_PAGERANK_PTHREADS_H

#include "pagerank_pthreads/defines.h"

#define NTHREADS 8

typedef struct _thread_data_t
{
  int tid;
  uint * const * R;
  uint const * RC;
  uint const * C;
  FLOAT** xPtr;
  FLOAT** zPtr;
  FLOAT* tmp;
  uint N;
  int* cnt;
  pthread_barrier_t* barrierPtr;
  uint** wells;
  uint* wells_c;
  uint* wells_cap;
  FLOAT* maximum;
} thread_data_t;

inline FLOAT max(FLOAT const * x, uint start, uint finish);
inline void abs_diff(FLOAT const * x, FLOAT const * y, FLOAT* res, uint start, uint finish);
inline void swap(FLOAT** x, FLOAT** y);
inline void fill(FLOAT* x, FLOAT value, uint start, uint finish);
inline void multiply(FLOAT* x, FLOAT value, uint start, uint finish);
inline void add(FLOAT* x, FLOAT value, uint start, uint finish);

void* thread_pagerank_power(void* arg);

/**
 * @brief function that calculates pagerank vector with power method
 * @param R [uint const **] reverse sparse transition matrix
 * @param RC [uint const *] num of edges which enter a node
 * @param C [uint const *] num of edges which exit a node
 * @param x [FLOAT**] pointer to pagerank vector, will have final result
 * @param N [uint] num of nodes
 * @note FLOAT is a preprocessor definition which is resolved to float or double
 * type
 * @return num of iterations needed for the pagerank vector to converge
 */
int pagerank_power(uint * const * R, uint const * RC, uint const * C, FLOAT** x, uint N);

#endif  // PAGERANK_PTHREADS_PAGERANK_PTHREADS_H
