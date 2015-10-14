#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

#include "pagerank_pthreads/defines.h"
#include "pagerank_pthreads/utils.h"
#include "pagerank_pthreads/pagerank_pthreads.h"

static uint * const * R = NULL;
static uint const * RC = NULL;
static uint const * C = NULL;
static FLOAT* xPtr = NULL;
static FLOAT* zPtr = NULL;
static uint N = 0;
static int cnt = 0;
static pthread_barrier_t* barrierPtr = NULL;
static uint* wells = NULL;
static FLOAT* wells_prob = NULL;
static FLOAT* maximum = NULL;

inline FLOAT max(FLOAT const * x, uint start, uint finish)
{
  FLOAT max_val = 0;
  uint i;
  for (i = start; i < finish; ++i)
    max_val = (max_val < x[i]) ? x[i] : max_val;
  return max_val;
}

inline void abs_diff(FLOAT const * x, FLOAT const * y, FLOAT* res, uint start, uint finish)
{
  uint i;
  for (i = start; i < finish; ++i)
    res[i] = fabs(x[i] - y[i]);
}

inline FLOAT max_abs_diff(FLOAT const * x, FLOAT const * y, uint start, uint finish)
{
  FLOAT max_val = 0, val = 0;
  uint i;
  for (i = start; i < finish; ++i)
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
  uint i;
  for (i = start; i < finish; ++i)
    x[i] = value;
}

inline void multiply(FLOAT* x, FLOAT value, uint start, uint finish)
{
  uint i;
  for (i = start; i < finish; ++i)
    x[i] *= value;
}

inline void add(FLOAT* x, FLOAT value, uint start, uint finish)
{
  uint i;
  for (i = start; i < finish; ++i)
    x[i] += value;
}

/**
 * @note ERR is a preprocessor definition which determines convergence
 */
void* thread_pagerank_power(void* arg)
{
  const int tid = *(int*)arg;
  uint chunk = N / NTHREADS;
  if (chunk == 0) chunk = 1;
  uint start = tid * chunk;
  uint end;
  if (tid != NTHREADS - 1)
    end = (tid + 1) * chunk;
  else
    end = N;

  const FLOAT p = 0.85;
  const FLOAT delta = (1 - p) / N;

  fill(zPtr, 1 / (FLOAT)N, start, end);
  fill(xPtr, 0, start, end);

  /* printf("%d: C\n", tid); */
  uint well_c, i, j;
  for (i = well_c = start; i < end; ++i)
  {
    if (C[i] == 0) wells[well_c++] = i;
  }

  /* printf("%d: begin iters\n", tid); */
  FLOAT total_wells_prob, total_maximum;
  do
  {
    /* printf("%d: filling normals\n", tid); */
    if (cnt != 0)
      fill(xPtr, 0, start, end);
    for (i = start; i < end; ++i)
    {
      for (j = 0; j < RC[i]; ++j)
      {
        uint coming = R[i][j];
        xPtr[i] += zPtr[coming] / C[coming];
      }
    }
    /* printf("%d: adding local wells\n", tid); */
    total_wells_prob = 0;
    for (i = start; i < well_c; ++i)
      total_wells_prob += zPtr[wells[i]];
    wells_prob[tid] = total_wells_prob;
    /* printf("%d: finding global wells\n", tid); */
    pthread_barrier_wait(barrierPtr);
    total_wells_prob = 0;
    for (j = 0; j < NTHREADS; ++j)
      total_wells_prob += wells_prob[j];
    add(xPtr, total_wells_prob / N, start, end);
    multiply(xPtr, p, start, end);
    add(xPtr, delta, start, end);
    /* printf("%d: finding local maximum\n", tid); */
    maximum[tid] = max_abs_diff(xPtr, zPtr, start, end);
    // find global maximum and share knowledge across threads
    pthread_barrier_wait(barrierPtr);
    /* printf("%d: finding global maximum\n", tid); */
    total_maximum = 0;
    for (j = 0; j < NTHREADS; ++j)
      total_maximum = (total_maximum < maximum[j]) ? maximum[j] : total_maximum;
    if (tid == 0)
    {
      ++cnt;
      swap(&xPtr, &zPtr);
    }
    pthread_barrier_wait(barrierPtr);
  } while (total_maximum >= ERR);

  pthread_exit(NULL);
}

int pagerank_power(uint * const * Rarg, uint const * RCarg, uint const * Carg, FLOAT** x, uint Narg)
{
  R = Rarg;
  RC = RCarg;
  C = Carg;
  N = Narg;
  cnt = 0;
  xPtr = (FLOAT*) malloc(N * sizeof(FLOAT));
  if (xPtr == NULL) exit(-2);
  zPtr = (FLOAT*) malloc(N * sizeof(FLOAT));
  if (zPtr == NULL) exit(-2);
  wells = (uint*) malloc(N * sizeof(uint));
  if (wells == NULL) exit(-2);
  wells_prob = (FLOAT*) malloc(NTHREADS * sizeof(FLOAT));
  if (wells_prob == NULL) exit(-2);
  maximum = (FLOAT*) malloc(NTHREADS * sizeof(FLOAT));
  if (maximum == NULL) exit(-2);
  pthread_t thr[NTHREADS];
  int tid[NTHREADS];

  struct timeval startwtime, endwtime;
  gettimeofday(&startwtime, NULL);

  pthread_barrier_t barrier;
  int i, rc;

  if((rc = pthread_barrier_init(&barrier, NULL, NTHREADS)))
  {
    fprintf(stderr, "Error: pthread_barrier_init, rc: %d\n", rc);
    exit(-4);
  }
  barrierPtr = &barrier;

  for (i = 0; i < NTHREADS; ++i)
  {
    tid[i] = i;
    if ((rc = pthread_create(&thr[i], NULL, thread_pagerank_power, &tid[i])))
    {
      fprintf(stderr, "Error: pthread_create, rc: %d\n", rc);
      exit(-4);
    }
  }

  for (i = 0; i < NTHREADS; ++i)
  {
    pthread_join(thr[i], NULL);
  }

  pthread_barrier_destroy(&barrier);

  swap(&xPtr, &zPtr);
  *x = xPtr;

  gettimeofday(&endwtime, NULL);
  double pagerank_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
      /1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
  printf("Time to compute pagerank vector: %f\n", pagerank_time);

  free((void*) zPtr);
  free((void*) wells);
  free((void*) wells_prob);
  free((void*) maximum);
  return cnt;
}
