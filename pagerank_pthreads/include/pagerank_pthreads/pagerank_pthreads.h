#ifndef PAGERANK_PTHREADS_PAGERANK_PTHREADS_H
#define PAGERANK_PTHREADS_PAGERANK_PTHREADS_H

#include "pagerank_pthreads/defines.h"

#define NTHREADS 2

FLOAT max(FLOAT const * x, uint start, uint finish);
void abs_diff(FLOAT const * x, FLOAT const * y, FLOAT* res, uint start, uint finish);
FLOAT max_abs_diff(FLOAT const * x, FLOAT const * y, uint start, uint finish);
void swap(FLOAT** x, FLOAT** y);
void fill(FLOAT* x, FLOAT value, uint start, uint finish);
void multiply(FLOAT* x, FLOAT value, uint start, uint finish);
void add(FLOAT* x, FLOAT value, uint start, uint finish);

void* thread_pagerank_power(void* arg);

/**
 * @brief function that calculates pagerank vector with power method
 * @param Rarg [uint const **] reverse sparse transition matrix
 * @param RCarg [uint const *] num of edges which enter a node
 * @param Carg [uint const *] num of edges which exit a node
 * @param x [FLOAT**] pointer to pagerank vector, will have final result
 * @param Narg [uint] num of nodes
 * @note FLOAT is a preprocessor definition which is resolved to float or double
 * type
 * @return num of iterations needed for the pagerank vector to converge
 */
int pagerank_power(uint * const * Rarg, uint const * RCarg, uint const * Carg,
    FLOAT** x, uint Narg);

#endif  // PAGERANK_PTHREADS_PAGERANK_PTHREADS_H
