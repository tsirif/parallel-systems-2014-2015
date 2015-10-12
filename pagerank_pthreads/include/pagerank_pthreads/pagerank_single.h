#ifndef PAGERANK_PTHREADS_PAGERANK_SINGLE_H
#define PAGERANK_PTHREADS_PAGERANK_SINGLE_H

#include "pagerank_pthreads/defines.h"

FLOAT max(FLOAT const * x, uint N);
void abs_diff(FLOAT const * x, FLOAT const * y, FLOAT* res, uint N);
void swap(FLOAT** x, FLOAT** y);
void fill(FLOAT* x, FLOAT value, uint N);
void multiply(FLOAT* x, FLOAT value, uint N);
void add(FLOAT* x, FLOAT value, uint N);

/**
 * @brief function that calculates pagerank vector with power method
 * @param L [uint const **] sparse transition matrix
 * @param C [uint const *] num of edges which exit a node
 * @param x [FLOAT**] pointer to pagerank vector, will have final result
 * @param N [uint] num of nodes
 * @note FLOAT is a preprocessor definition which is resolved to float or double
 * type
 * @return num of iterations needed for the pagerank vector to converge
 */
int pagerank_power(uint * const * L, uint const * C, FLOAT** x, uint N);

#endif  // PAGERANK_PTHREADS_PAGERANK_SINGLE_H
