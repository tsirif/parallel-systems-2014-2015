#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

#include "pagerank_pthreads/defines.h"
#include "pagerank_pthreads/utils.h"
#include "pagerank_pthreads/pagerank_pthreads.h"

inline FLOAT max(FLOAT const * x, uint start, uint finish)
{
  FLOAT max_val = 0;
  for (uint i = start; i < finish; ++i)
    max_val = (max_val < x[i]) ? x[i] : max_val;
  return max_val;
}

inline void abs_diff(FLOAT const * x, FLOAT const * y, FLOAT* res, uint start, uint finish)
{
  for (uint i = start; i < finish; ++i)
    res[i] = fabs(x[i] - y[i]);
}

inline FLOAT max_abs_diff(FLOAT const * x, FLOAT const * y, uint start, uint finish)
{
  FLOAT max_val = 0, val = 0;
  for (uint i = start; i < finish; ++i)
  {
    val = fabs(x[i] - y[i]);
    max_val = (max_val < val) ? val : max_val;
  }
  return max_val;
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
  fill(*(data->xPtr), 0, start, end);

  uint well_c, i, j;
  for (i = well_c = start; i < end; ++i)
  {
    if (data->C[i] == 0)
      data->wells[well_c++] = i;
  }

  FLOAT total_wells_prob, maximum;
  do
  {
    if (cnt != 0)
      fill(*(data->xPtr), 0, start, end);
    for (i = start; i < end; ++i)
    {
      for (j = 0; j < data->RC[i]; ++j)
      {
        uint coming = data->R[i][j];
        data->xPtr[0][i] += data->zPtr[0][coming] / data->C[coming];
      }
    }
    total_wells_prob = 0;
    for (i = start; i < well_c; ++i)
      total_wells_prob += data->zPtr[0][data->wells[i]];
    data->wells_prob[data->tid] = total_wells_prob;
    pthread_barrier_wait(data->barrierPtr);
    total_wells_prob = 0;
    for (j = 0; j < NTHREADS; ++j)
      total_wells_prob += data->wells_prob[j];
    add(*(data->xPtr), total_wells_prob / data->N, start, end);
    multiply(*(data->xPtr), p, start, end);
    add(*(data->xPtr), delta, start, end);
    data->maximum[data->tid] = max_abs_diff(*(data->xPtr), *(data->zPtr), start, end);
    ++cnt;
    // find global maximum and share knowledge across threads
    pthread_barrier_wait(data->barrierPtr);
    maximum = 0;
    for (j = 0; j < NTHREADS; ++j)
      maximum = (maximum < data->maximum[j]) ? data->maximum[j] : maximum;
    if (data->tid == 0)
      swap(data->xPtr, data->zPtr);
    pthread_barrier_wait(data->barrierPtr);
  } while (maximum >= ERR);

  data->cnt[data->tid] = cnt;

  pthread_exit(NULL);
}

int pagerank_power(uint * const * R, uint const * RC, uint const * C, FLOAT** x, uint N)
{
  FLOAT *z;
  uint* wells;
  *x = (FLOAT*) malloc(N * sizeof(FLOAT));
  if (*x == NULL) exit(-2);
  z = (FLOAT*) malloc(N * sizeof(FLOAT));
  if (z == NULL) exit(-2);
  wells = (uint*) malloc(N * sizeof(uint));
  if (wells == NULL) exit(-2);

  struct timeval startwtime, endwtime;
  gettimeofday(&startwtime, NULL);

  pthread_t thr[NTHREADS];
  FLOAT maximum[NTHREADS], wells_prob[NTHREADS];
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
    data[i].N = N;
    data[i].cnt = cnt;
    data[i].barrierPtr = &barrier;
    data[i].wells = wells;
    data[i].wells_prob = wells_prob;
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

  pthread_barrier_destroy(&barrier);

  gettimeofday(&endwtime, NULL);
  double pagerank_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
      /1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
  printf("Time to compute pagerank vector: %f\n", pagerank_time);

  int counter = cnt[0];
  for (i = 1; i < NTHREADS; ++i)
  {
    if (cnt[i] != counter)
    {
      fprintf(stderr, "Error: counters returned from threads are not equal!\n");
      exit(-5);
    }
  }

  free((void*) wells);
  free((void*) z);
  return counter;
}
