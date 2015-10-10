#ifndef PAGERANK_PTHREADS_UTILS_H
#define PAGERANK_PTHREADS_UTILS_H

#include "pagerank_pthreads/defines.h"

extern uint* capacity;

void append(uint** L, uint* C, uint index, uint value);
void reverse(uint** L, uint* LC, uint N, uint** R, uint* RC);

int read_graph(char const * filename, uint*** L, uint** C, uint* N, uint* E);
int read_graph_reverse(char const * filename, uint*** R, uint** RC, uint** LC, uint* N, uint* E);

void print_sparse_matrix(uint** L, uint* C, uint N);

int rank(const void * p1, const void * p2);

/**
 * @brief given resulting pagerank probability vector, save it in a file
 * @param output [char const *] name of output file
 * @param input [char const *] name of input file
 * @param cnt [int] number of iterations till convergence was achieved
 * @param x [FLOAT const *] pagerank array
 * @param N [uint] dimension of pagerank array
 * @param E [uint] number of edges in graph
 * @note FLOAT is a preprocessor definition which is resolved to float or double
 * type
 */
void output_pagerank_vector(char const * output, char const * input,
    int cnt, FLOAT const * x, uint N, uint E);
/**
 * @brief save nodes sorted by their own pagerank probability
 * @param output [char const *] name of output file
 * @param input [char const *] name of input file
 * @param cnt [int] number of iterations till convergence was achieved
 * @param x [FLOAT const *] pagerank array
 * @param N [uint] dimension of pagerank array
 * @param E [uint] number of edges in graph
 * @note FLOAT is a preprocessor definition which is resolved to float or double
 * type
 */
void output_ranked_nodes(char const * output, char const * input,
    int cnt, FLOAT const * x, uint N, uint E);

#endif  // PAGERANK_PTHREADS_UTILS_H
