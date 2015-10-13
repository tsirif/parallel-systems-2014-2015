#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#include "pagerank_pthreads/defines.h"
#include "pagerank_pthreads/utils.h"
#include "pagerank_pthreads/pagerank_pthreads.h"

inline FLOAT max(FLOAT const * x, uint start, uint finish)
{
  FLOAT max_val = 0.0;
  for (uint i = start; i < finish; ++i)
    max_val = (max_val < x[i]) ? x[i] : max_val;
  return max_val;
}

inline void abs_diff(FLOAT const * x, FLOAT const * y, FLOAT* res, uint start, uint finish)
{
  for (uint i = start; i < finish; ++i)
    res[i] = fabs(x[i] - y[i]);
}

inline void swap(FLOAT** x, FLOAT** y)
{
  FLOAT* tmp = *x;
  *x = *y;
  *y = tmp;
}

inline void fill(FLOAT* x, FLOAT value, uint start, uint finish)
{
  for (uint i = start; i < finish; ++i)
    x[i] = value;
}

inline void multiply(FLOAT* x, FLOAT value, uint start, uint finish)
{
  for (uint i = start; i < finish; ++i)
    x[i] *= value;
}

inline void add(FLOAT* x, FLOAT value, uint start, uint finish)
{
  for (uint i = start; i < finish; ++i)
    x[i] += value;
}

/**
 * @note ERR is a preprocessor definition which determines convergence
 */
void* thread_pagerank_power(void* arg)
{
  thread_data_t* data = (thread_data_t*)arg;
  uint chunk = data->N / NTHREADS;
  if (chunk == 0) chunk = 1;
  uint start = data->tid * chunk;
  uint end;
  if (data->tid != NTHREADS - 1)
    end = (data->tid + 1) * chunk;
  else
    end = data->N;

  int cnt = 0;
  const FLOAT p = 0.85;
  const FLOAT delta = (1 - p) / data->N;

  fill(*(data->zPtr), 1 / (FLOAT) data->N, start, end);
  fill(*(data->xPtr), 0.0, start, end);

  data->wells_c[data->tid] = 0;
  data->wells_cap[data->tid] = DFL_CAPACITY;
  data->wells[data->tid] = (uint*) malloc(DFL_CAPACITY * sizeof(uint));
  if (data->wells[data->tid] == NULL) exit(-3);
  for (uint i = start; i < end; ++i)
  {
    if (data->C[i] == 0)
      append(data->wells, data->wells_c, data->wells_cap, data->tid, i);
  }
  pthread_barrier_wait(data->barrierPtr);

  do
  {
    if (cnt != 0)
    {
      fill(*(data->xPtr), 0.0, start, end);
    }
    for (uint i = start; i < end; ++i)
    {
      for (uint j = 0; j < data->RC[i]; ++j)
      {
        uint coming = data->R[i][j];
        data->xPtr[0][i] += data->zPtr[0][coming] / data->C[coming];
      }
    }
    FLOAT well_prob = 0.0;
    for (int ki = 0; ki < NTHREADS; ++ki)
    {
      for (int kj = 0; kj < data->wells_c[ki]; ++kj)
      {
        well_prob += data->zPtr[0][data->wells[ki][kj]];
      }
    }
    add(*(data->xPtr), well_prob / data->N, start, end);
    multiply(*(data->xPtr), p, start, end);
    add(*(data->xPtr), delta, start, end);
    abs_diff(*(data->xPtr), *(data->zPtr), data->tmp, start, end);
    data->maximum[data->tid] = max(data->tmp, start, end);
    ++cnt;
    // find global maximum and share knowledge across threads
    pthread_barrier_wait(data->barrierPtr);
    if (data->tid == 0)
    {
      FLOAT maximum = 0.0;
      for (int i = 0; i < NTHREADS; ++i)
        maximum = (maximum < data->maximum[i]) ? data->maximum[i] : maximum;
      for (int i = 0; i < NTHREADS; ++i)
        data->maximum[i] = maximum;
      swap(data->xPtr, data->zPtr);
    }
    pthread_barrier_wait(data->barrierPtr);
  } while (data->maximum[data->tid] >= ERR);

  data->cnt[data->tid] = cnt;

  free((void*)data->wells[data->tid]);
  pthread_exit(NULL);
}

int pagerank_power(uint * const * R, uint const * RC, uint const * C, FLOAT** x, uint N)
{
  FLOAT *z, *tmp;
  *x = (FLOAT*) malloc(N * sizeof(FLOAT));
  if (*x == NULL) exit(-2);
  z = (FLOAT*) malloc(N * sizeof(FLOAT));
  if (z == NULL) exit(-2);
  tmp = (FLOAT*) malloc(N * sizeof(FLOAT));
  if (tmp == NULL) exit(-2);

  pthread_t thr[NTHREADS];
  uint* wells[NTHREADS];
  uint wells_c[NTHREADS], wells_cap[NTHREADS];
  FLOAT maximum[NTHREADS];
  int cnt[NTHREADS];
  thread_data_t data[NTHREADS];

  pthread_barrier_t barrier;
  int i, rc;

  if((rc = pthread_barrier_init(&barrier, NULL, NTHREADS)))
  {
    fprintf(stderr, "Error: pthread_barrier_init, rc: %d\n", rc);
    exit(-4);
  }

  for (i = 0; i < NTHREADS; ++i)
  {
    data[i].tid = i;
    data[i].R = R;
    data[i].RC = RC;
    data[i].C = C;
    data[i].xPtr = x;
    data[i].zPtr = &z;
    data[i].tmp = tmp;
    data[i].N = N;
    data[i].cnt = cnt;
    data[i].barrierPtr = &barrier;
    data[i].wells = wells;
    data[i].wells_c = wells_c;
    data[i].wells_cap = wells_cap;
    data[i].maximum = maximum;
  }

  for (i = 0; i < NTHREADS; ++i)
  {
    if ((rc = pthread_create(&thr[i], NULL, thread_pagerank_power, &data[i])))
    {
      fprintf(stderr, "Error: pthread_create, rc: %d\n", rc);
      exit(-4);
    }
  }

  for (i = 0; i < NTHREADS; ++i)
  {
    pthread_join(thr[i], NULL);
  }

  swap(x, &z);

  int counter = cnt[0];
  for (i = 1; i < NTHREADS; ++i)
  {
    if (cnt[i] != counter)
    {
      fprintf(stderr, "Error: counters returned from threads are not equal!\n");
      exit(-5);
    }
  }

  pthread_barrier_destroy(&barrier);
  free((void*) tmp);
  free((void*) z);
  return counter;
}
