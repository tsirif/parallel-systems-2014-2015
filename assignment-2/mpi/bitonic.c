#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "mpi.h"
#include "omp.h"
#include "utils.h"

#ifdef DCMT
#include "dcmt/include/dc.h"
#endif


void swap_process_data(uint32_t** array_in, uint32_t** array_out,
                       int N, int fellow, int tag, int dir);
void compare_and_keep(uint32_t** array_in, uint32_t* array_out,
                      int N, int dir);
inline void test_validity(uint32_t* array, int N, int numTasks, int rank);

uint32_t* tmp_array = NULL;

int main(int argc, char *argv[])
{
/******************************************************************************
 *                              Handle Arguments                              *
 ******************************************************************************/

  unsigned int num_tasks, N;
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

  uint32_t* in_array = (uint32_t*) malloc(N * sizeof(uint32_t));
  uint32_t* out_array = (uint32_t*) malloc(N * sizeof(uint32_t));
  tmp_array = (uint32_t*) malloc(N * sizeof(uint32_t));
  if (in_array == NULL || out_array == NULL || tmp_array == NULL) {
    printf("Memory allocation error: couldn't allocate enough memory.\n");
    printf("Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, 3);
  }

#ifdef GRAMA
  omp_set_nested(1);
  omp_set_dynamic(0);
#endif

/******************************************************************************
 *                          Generate Random Dataset                           *
 ******************************************************************************/

  unsigned int i;
#if defined(DCMT)  // if with dynamic creator of mersenne twisters prg
  if (rank == 0)
    printf("Generating random with DCMT.\n");
  mt_struct* mtst;
  mtst = get_mt_parameter_id_st(32, 607, rank, seed);
  if (mtst == NULL) {
    printf("Error finding an independent set of parameters for dcmt prg.\n");
    printf("Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, 4);
  }
  sgenrand_mt(time(NULL), mtst);

  for (i = 0; i < N; ++i) {
    in_array[i] = genrand_mt(mtst);
  }

  free_mt_struct(mtst);
#else
  if (rank == 0) {
    printf("Generating random by default rand generator.\n");
    srand(time(NULL));
  }
  for (i = 0; i < N; ++i) {
    in_array[i] = rand();
  }
#endif  // SCMT or else

  /* if (rank == 1) { */
    /* printf("YOLO!\n"); */
    /* print_array(in_array, N); */
  /* } */

#ifdef COMPARE
  int final_size = N * num_proc;
  uint32_t* final;
  char comparison = 1;
  if (rank == 0) {
    final = (uint32_t*) malloc(final_size * sizeof(uint32_t));
    if (final == NULL) {
      printf("Could not allocate memory for the buffer "
             "so as to receive all the data.\n");
      printf("Comparison with  will not be performed!\n");
      comparison = 0;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (comparison) {
    MPI_Gather(in_array, N, MPI_UNSIGNED,
               final, N, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
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
  struct timeval stwt_qs, edwt_qs;
  double seq_time_qs;
  if (rank == 0) {
    gettimeofday(&stwt_qs, NULL);
  }
#ifdef GRAMA
  grama_quicksort(in_array, tmp_array, N, 4, 0);
  if (rank == 0)
    printf("parallel ");
#else
  qsort(in_array, N, sizeof(uint32_t), cmpfunc);
  if (rank == 0)
    printf("serial ");
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    gettimeofday(&edwt_qs, NULL);
    seq_time_qs = (double)((edwt_qs.tv_usec - stwt_qs.tv_usec)/1.0e6
            + edwt_qs.tv_sec - stwt_qs.tv_sec);
    printf("partial quicksort clock time = %f\n", seq_time_qs);
  }

  /* if (rank == 1) { */
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

  MPI_Barrier(MPI_COMM_WORLD);
#if defined(TIME) || defined(COMPARE)
  if (rank == 0) {
    gettimeofday(&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
            + endwtime.tv_sec - startwtime.tv_sec);
    printf("parallel bitonic clock time = %f\n", seq_time);
  }
#endif  // TIME or COMPARE

  /* if (rank == 1) { */
    /* printf("YOLOR!\n"); */
    /* print_array(in_array, N); */
  /* } */

#if defined(FILEOUT) && !defined(TEST)
  output_array(in_array, N, rank);
#endif  // FILE_OUT and not TEST

/******************************************************************************
 *                       Test Validity of Parallel Sort                       *
 ******************************************************************************/

#ifdef TEST
  test_validity(in_array, N, num_tasks, rank);
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
      qsort(final, final_size, sizeof(uint32_t), cmpfunc);
      gettimeofday(&endwtime, NULL);
      seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
      + endwtime.tv_sec - startwtime.tv_sec);
      printf("serial quicksort clock time = %f\n\n", seq_time);
      free(final);
    }
  }
#endif  // COMPARE

  printf("Rank-%d finished successfully!\n", rank);
  return 0;
}

void swap_process_data(uint32_t** array_in, uint32_t** array_out,
                       int N, int fellow, int tag, int dir)
{
  MPI_Request request[2];
  MPI_Status status[2];
  if (dir == 0) {
    MPI_Isend(*array_in, N, MPI_UNSIGNED, fellow,
              tag, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(*array_out, N, MPI_UNSIGNED, fellow,
              tag, MPI_COMM_WORLD, &request[1]);
  }
  else {
    MPI_Irecv(*array_out, N, MPI_UNSIGNED, fellow,
              tag, MPI_COMM_WORLD, &request[0]);
    MPI_Isend(*array_in, N, MPI_UNSIGNED, fellow,
              tag, MPI_COMM_WORLD, &request[1]);
  }
  MPI_Waitall(2, request, status);
}

void compare_and_keep(uint32_t** array_in, uint32_t* array_out, int N, int dir)
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

inline void test_validity(uint32_t* array, int N, int num_procs, int rank)
{
  int i;
  int final_size = N * num_procs;
  uint32_t *final;
  if (rank == 0) {
    final = (uint32_t*) malloc(final_size * sizeof(uint32_t));
    if (final == NULL) {
      printf("Could not allocate memory for the buffer so as to "
          "receive all the data. The test will not be performed! \n ");
      return;
    }
  }
  MPI_Gather(array, N, MPI_UNSIGNED,
             final, N, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    /* printf("\nfinal array:\n"); */
    /* print_array(final, final_size); */
    /* printf("\n"); */
#ifdef FILEOUT
    output_array(final, final_size, -1);
#endif
    int fail = 0;
    #pragma omp parallel for reduction(||: fail)
    for (i = 1; i < final_size; ++i) {
      fail = fail || (final[i] < final[i-1]);
    }
    printf("parallel bitonic sort - validity test: ");
    if (fail) printf("FAIL\n");
    else printf("PASS\n");
    free(final);
  }
}
