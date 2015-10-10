#ifndef PAGERANK_PTHREADS_DEFINES_H
#define PAGERANK_PTHREADS_DEFINES_H

#include <stdint.h>

#ifdef DOUBLE
#define FLOAT double
#else
#define FLOAT float
#endif  // DOUBLE

#define ERR 0.0001

typedef uint32_t uint;

#define DFL_CAPACITY 10

#endif  // PAGERANK_PTHREADS_DEFINES_H
