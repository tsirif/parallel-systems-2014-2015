#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "mpi.h"

#ifdef SFMT
#include "SFMT/SFMT.h"
#endif


inline void swap(uint64_t** x, uint64_t** y);
inline void print_array(uint64_t* array, int N);
int cmpfunc(const void* a, const void* b);
void swap_process_data(uint64_t** array_in, uint64_t** array_out,
                       int N, int fellow, int tag, int dir);
void compare_and_keep(uint64_t** array_in, uint64_t* array_out,
                      int N, int dir);
inline void test_validity(uint64_t* array, int N, int numTasks, int rank);
uint64_t* tmp_array = NULL;

int main(int argc, char *argv[])
{
/******************************************************************************
 *                              Handle Arguments                              *
 ******************************************************************************/

  int num_tasks, N;
  long int seed;
  if (argc < 3) {
    printf("Invalid command line argument option!\n");
    printf("Usage : %s p q where p is the number of MPI processes to "
    "be spawned and q the number of elements in each process.\n", argv[0]);
    exit(1);
  }
  else {
    num_tasks = 1 << atoi(argv[1]);
    N = 1 << atoi(argv[2]);
    if (argc >= 4) {
      seed = strtol(argv[3], NULL, 10);
    }
    else {
      seed = 123456;
    }
  }

/******************************************************************************
 *                           Initialize Processors                            *
 ******************************************************************************/

  int num_proc, rank, rc;

  rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
    printf("Error starting MPI program.\nTerminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /*
   * The master process checks that the correct number of processes
   * has started working.
   */
  if (rank == 0 && num_proc != num_tasks) {
      printf("The number of tasks is not equal to the one passed to "
             "the master process and thus the sorting procedure will stop!\n");
      printf("Terminating.\n");
      MPI_Abort(MPI_COMM_WORLD, 2);
  }

  uint64_t* in_array = (uint64_t*) malloc(N * sizeof(uint64_t));
  uint64_t* out_array = (uint64_t*) malloc(N * sizeof(uint64_t));
  tmp_array = (uint64_t*) malloc(N * sizeof(uint64_t));
  if (in_array == NULL || out_array == NULL || tmp_array == NULL) {
    printf("Memory allocation error: couldn't allocate enough memory.\n");
    printf("Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, 3);
  }

/******************************************************************************
 *                          Generate Random Dataset                           *
 ******************************************************************************/

  int i;
#ifdef SFMT  // if with SIMD fast mersenne twister prg
  if (rank == 0)
    printf("Generating random with SFMT.\n");
  sfmt_t sfmt;
  sfmt_init_gen_rand(&sfmt, seed+rank+time(NULL));

  for (i = 0; i < N; ++i) {
    in_array[i] = sfmt_genrand_uint64(&sfmt);
  }
#else
  if (rank == 0) {
    printf("Generating random by default rand generator.\n");
    srand(time(NULL));
  }
  for (i = 0; i < N; ++i) {
    in_array[i] = rand();
  }
#endif  // SFMT

  /* if (rank == 0) { */
    /* printf("YOLO!\n"); */
    /* print_array(in_array, N); */
  /* } */

#ifdef COMPARE
  int final_size = N * num_proc;
  uint64_t* final;
  char comparison = 1;
  if (rank == 0) {
    final = (uint64_t*) malloc(final_size * sizeof(uint64_t));
    if (final == NULL) {
      printf("Could not allocate memory for the buffer "
             "so as to receive all the data.\n");
      printf("Comparison with  will not be performed!\n");
      comparison = 0;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (comparison) {
    MPI_Gather(in_array, N, MPI_UNSIGNED_LONG,
               final, N, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  }
#endif  // COMPARE

/******************************************************************************
 *                           Parallel Bitonic Sort                            *
 ******************************************************************************/

#if defined(TIME) || defined(COMPARE)
  struct timeval startwtime, endwtime;
  double seq_time;
  if (rank == 0) {
	  gettimeofday(&startwtime, NULL);
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif  // TIME or COMPARE

  // Initially each processor sort serially its own data.
  qsort(in_array, N, sizeof(uint64_t), cmpfunc);

  /* if (rank == 0) { */
    /* printf("YO!\n"); */
    /* print_array(in_array, N); */
  /* } */

  int k, j, dir;

  for (k = 2; k <= num_proc; k = k<<1) {
    for (j = k>>1; j > 0; j = j>>1) {
      MPI_Barrier(MPI_COMM_WORLD);
      dir = rank&k;
      if ((rank^j) > rank) {
        dir = !dir;
        swap_process_data(&in_array, &out_array, N, rank+j, rank+2*j+k, dir);
      }
      else {
        swap_process_data(&in_array, &out_array, N, rank-j, rank+j+k, dir);
      }
      compare_and_keep(&in_array, out_array, N, dir);
    }
  }

#if defined(TIME) || defined(COMPARE)
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    gettimeofday(&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
            + endwtime.tv_sec - startwtime.tv_sec);
    printf("parallel bitonic clock time = %f\n", seq_time);
  }
#endif  // TIME or COMPARE

  /* if (rank == 0) { */
    /* printf("YOLOR!\n"); */
    /* print_array(in_array, N); */
  /* } */

#ifdef FILEOUT
  char filename[10];
  FILE* f;
  sprintf(filename, "output_%d.txt", rank);
  f = fopen(filename, "w");
  for (int i = 0; i < N; ++i) {
    fprintf(f, "%lu\n", in_array[i]);
  }
  fclose(f);
#endif  // FILE_OUT

/******************************************************************************
 *                       Test Validity of Parallel Sort                       *
 ******************************************************************************/

#ifdef TEST
  test_validity(in_array, N, num_proc, rank);
#endif  // TEST

  free(in_array);
  free(out_array);
  free(tmp_array);

  MPI_Finalize();

/******************************************************************************
 *                       Compare with Serial Quicksort                        *
 ******************************************************************************/

#ifdef COMPARE
  if (rank == 0) {
    if (comparison) {
      gettimeofday(&startwtime, NULL);
      qsort(final, final_size, sizeof(uint64_t), cmpfunc);
      gettimeofday(&endwtime, NULL);
      seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
      + endwtime.tv_sec - startwtime.tv_sec);
      printf("serial quicksort clock time = %f\n\n", seq_time);
      free(final);
    }
  }
#endif  // COMPARE

  return 0;
}

inline void swap(uint64_t **x, uint64_t **y)
{
  uint64_t *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;
}

inline void print_array(uint64_t *array, int N)
{
  for (int i = 0; i < N; ++i) {
    printf("%lu\n", array[i]);
  }
}

int cmpfunc(const void* a, const void* b)
{
   return ( *(const uint64_t*)a - *(const uint64_t*)b );
}

void swap_process_data(uint64_t** array_in, uint64_t** array_out,
                       int N, int fellow, int tag, int dir)
{
  MPI_Status status;
  if (dir == 0) {
    MPI_Send(*array_in, N, MPI_UNSIGNED_LONG, fellow, tag, MPI_COMM_WORLD);
    MPI_Recv(*array_out, N, MPI_UNSIGNED_LONG, fellow, tag, MPI_COMM_WORLD, &status);
  }
  else {
    MPI_Recv(*array_out, N, MPI_UNSIGNED_LONG, fellow, tag, MPI_COMM_WORLD, &status);
    MPI_Send(*array_in, N, MPI_UNSIGNED_LONG, fellow, tag, MPI_COMM_WORLD);
  }
}

void compare_and_keep(uint64_t** array_in, uint64_t* array_out, int N, int dir)
{
  int i, in_flag, out_flag;
  out_flag = in_flag = dir == 0 ? N-1 : 0;
  for (i = 0; i < N; ++i) {
    if (dir == 0) {
      if ((*array_in)[in_flag] > array_out[out_flag])
        tmp_array[N-i-1] = (*array_in)[in_flag--];
      else
        tmp_array[N-i-1] = array_out[out_flag--];
    }
    else {
      if ((*array_in)[in_flag] > array_out[out_flag])
        tmp_array[i] = array_out[out_flag++];
      else
        tmp_array[i] = (*array_in)[in_flag++];
    }
  }
  swap(array_in, &tmp_array);
}

inline void test_validity(uint64_t* array, int N, int numTasks, int rank)
{
  int i;
  int final_size = N * numTasks;
  uint64_t *final;
  if (rank == 0) {
    final = (uint64_t*) malloc(final_size * sizeof(uint64_t));
    if (final == NULL) {
      printf("Could not allocate memory for the buffer so as to "
          "receive all the data. The test will not be performed! \n ");
      return;
    }
  }
  MPI_Gather(array, N, MPI_UNSIGNED_LONG,
             final, N, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    int fail = 0;
    for (i = 1; i < final_size; ++i) {
      fail = fail || (final[i] < final[i-1]);
      if (fail) break;
    }
    printf("Parallel bitonic sort - validity test: ");
    if (fail) printf("FAIL\n");
    else printf("PASS\n");
    free(final);
  }
}
