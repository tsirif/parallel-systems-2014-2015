#include "omp.h"
#include <stdio.h>
#define NUM_THREADS 8

static long num_steps = 10000000;
double step;

void serial();
void exercise2();
void exercise3();
void exercise4();

void main ()
{
  omp_set_dynamic(0);
  omp_set_num_threads(NUM_THREADS);
  printf("Maximum number of threads: %d\n", omp_get_max_threads());
  omp_lock_t lock;
  omp_init_lock(&lock);
  #pragma omp parallel
  {
    #pragma omp single
    {
      printf("Number of processors: %d\n", omp_get_num_procs());
      printf("Do we work in parallel? ");
      if (omp_in_parallel()) printf("YES\n");
      else printf("NO\n");
      printf("#Dynamic threads: %d\n", omp_get_dynamic());
      printf("#Threads: %d\n", omp_get_num_threads());
    }
    int ID = omp_get_thread_num();
    omp_set_lock(&lock);
    printf("hello(%d)", ID);
    printf(" world(%d) \n", ID);
    omp_unset_lock(&lock);
  }
  omp_destroy_lock(&lock);

  // Serial implementation of an integral
  serial();
  // One should care not to write code with race conditions.
  // In most occasions, a thread should has exclusive write permissions to a place
  // in memory.
  // Avoid line caching, which leads to false sharing because local memory is
  // copied across all threads.
  // False sharing makes this parallel code slower than the serial.
  omp_set_num_threads(NUM_THREADS);
  exercise2();
  // Eliminating false sharing reveals the benefits of parallel coding.
  omp_set_num_threads(4);
  exercise3();
  omp_set_num_threads(8);
  exercise3();
  omp_set_num_threads(16);
  exercise3();
  // Spawning 32 threads is not as beneficial as 16. Parallelization is hindered
  // by data transfer/copying durations and pc counter arrangements.
  omp_set_num_threads(32);
  exercise3();
  // Using 'parallel for reduction' ensures minimal changes to serial code.
  // A small speed-up is achieved.
  omp_set_num_threads(NUM_THREADS);
  exercise4();
}

void serial()
{
  double duration = omp_get_wtime();
  int i;
  double pi = 0.0, x, sum = 0.0;
  step = 1.0/(double) num_steps;
  for (i = 0; i < num_steps; i += 1)
  {
    x = (i + 0.5) * step;
    sum += 4.0/(1.0 + x * x);
  }
  pi = sum * step;
  printf("Yo! Result is: %f \n", pi);
  duration = omp_get_wtime() - duration;
  printf("Duration: %f \n", duration);
}

void exercise2()
{
  double duration = omp_get_wtime();
  int nthreads, i;
  double pi, sum[NUM_THREADS];
  step = 1.0/(double) num_steps;
  #pragma omp parallel
  {
    int id = omp_get_thread_num(), i;
    int nthreads_par = omp_get_num_threads();
    if (id == 0) nthreads = nthreads_par; // get info to seq scope
    double x;
    // Each thread writes to its own region.
    sum[id] = 0.0;
    for (i = id; i < num_steps; i += nthreads_par)
    {
      x = (i + 0.5) * step;
      sum[id] += 4.0/(1.0 + x * x);
    }
  }
  // A single thread sequentially combines results.
  for (i = 0, pi = 0.0; i < nthreads; i++)
  {
    pi += sum[i] * step;
  }
  printf("Yo 2! Result is: %f \n", pi);
  duration = omp_get_wtime() - duration;
  printf("Duration: %f \n", duration);
}

void exercise3()
{
  double duration = omp_get_wtime();
  double pi = 0.0;
  step = 1.0/(double) num_steps;
  #pragma omp parallel
  {
    #pragma omp single
    printf("Exercise 3, spawned threads: %d\n", omp_get_num_threads());
    int i;
    int id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    // Work with scalar local in order to avoid cache line sharing (false sharing)
    double x, sum = 0.0;
    // Each thread writes to its own region.
    for (i = id; i < num_steps; i += nthreads)
    {
      x = (i + 0.5) * step;
      sum += 4.0/(1.0 + x * x);
    }
    // Each thread combines its results. Sync must be forced.
    #pragma omp critical
    pi += sum * step;
  }
  printf("Yo 3! Result is: %f \n", pi);
  duration = omp_get_wtime() - duration;
  // exercice 3 will be faster than exercise 2
  printf("Duration: %f \n", duration);
}

void exercise4()
{
  double duration = omp_get_wtime();
  int i;
  double pi = 0.0, sum = 0.0;
  step = 1.0/(double) num_steps;
  #pragma omp parallel for reduction(+: sum)
  for (i = 0; i < num_steps; i += 1)
  {
    double x = (i + 0.5) * step;
    sum += 4.0/(1.0 + x * x);
  }
  pi = sum * step;
  printf("Yo 4! Result is: %f \n", pi);
  duration = omp_get_wtime() - duration;
  printf("Duration: %f \n", duration);
}
