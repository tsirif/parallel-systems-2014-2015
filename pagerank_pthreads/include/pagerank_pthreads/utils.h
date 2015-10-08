#ifndef PAGERANK_PTHREADS_UTILS_H
#define PAGERANK_PTHREADS_UTILS_H

#include <stdint.h>

typedef uint32_t uint;

#define DFL_CAPACITY 10
extern uint* capacity;

void append(uint** L, uint* C, uint index, uint value);
void reverse(uint** L, uint* LC, uint N, uint** R, uint* RC);

int read_graph(char const * filename, uint*** L, uint** C, uint* N, uint* E);
int read_graph_reverse(char const * filename, uint*** R, uint** RC, uint** LC, uint* N, uint* E);

void print_sparse_matrix(uint** L, uint* C, uint N);

#endif  // PAGERANK_PTHREADS_UTILS_H
