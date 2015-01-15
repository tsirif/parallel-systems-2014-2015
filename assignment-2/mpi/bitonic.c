#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "mpi.h"


inline void swap(int **x, int **y);
void swap_process_data(int** array_in, int** array_out, int N, int fellow, int tag, int dir);
void compare_and_keep(int** array_in, int* array_out, int N, int dir);
void test_validity(int* array, int N, int numTasks, int rank);
int* tmp_array = NULL;

int cmpfunc(const void* a, const void* b)
{
   return ( *(const int*)a - *(const int*)b );
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    /* If not print a warning message with the correct way to use */
    /* the program and terminate the execution. */
    printf("Invalid command line argument option! \n");
    printf("Usage : %s p q where p is the number of MPI processes to "
    "be spawned and q the number of elements in each process.\n ", argv[0]);
    exit(1);
  }

/******************************************************************************
 *                           Initialize Processors                            *
 ******************************************************************************/

  int numTasks, rank, rc, N, i;

  N = 1<<atoi(argv[2]);

  rc = MPI_Init(&argc,&argv);
  if (rc != MPI_SUCCESS) {
    printf("Error starting MPI program.\nTerminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /*
   * The master process checks that the correct number of processes
   * has started working.
   */
  if (rank == 0 && numTasks != (1<<atoi(argv[1]))) {
      printf("The number of tasks is not equal to the one passed to "
              "the master process and thus the sorting procedure will stop! \n");

      // Terminate the MPI processes.
      MPI_Abort(MPI_COMM_WORLD, 2);
      exit(2);
  }

  int* in_array = (int*) malloc(N * sizeof(int));
  int* out_array = (int*) malloc(N * sizeof(int));
  tmp_array = (int*) malloc(N * sizeof(int));
  if (in_array == NULL || out_array == NULL || tmp_array == NULL) {
    printf("Memory allocation error: couldn't allocate enough memory.\nTerminating.\n");
    MPI_Abort(MPI_COMM_WORLD, 3);
    exit(3);
  }

/******************************************************************************
 *                          Generate random dataset                           *
 ******************************************************************************/

  if (rank == 0) {
    srand(time(NULL));
  }

  for (i = 0; i < N; ++i) {
    in_array[i] = rand();
  }

/******************************************************************************
 *                           Parallel Bitonic Sort                            *
 ******************************************************************************/
  // Initially each processor sort serially its own data.
  qsort(in_array, N, sizeof(int), cmpfunc);

  int k, j, dir;

  for (k = 2; k <= numTasks; k = k<<1) {
    for (j = k>>1; j > 0; j = j>>1) {
      MPI_Barrier(MPI_COMM_WORLD);
      dir = rank&k;
      /* if (rank == 1) { */
        /* printf("pre\n"); */
        /* printf("in_array\n"); */
        /* for (int i = 0; i < N; ++i) { */
          /* printf("%d\n", in_array[i]); */
        /* } */
        /* printf("out_array\n"); */
        /* for (int i = 0; i < N; ++i) { */
          /* printf("%d\n", out_array[i]); */
        /* } */
      /* } */
      if ((rank^j) > rank) {
        dir = !dir;
        swap_process_data(&in_array, &out_array, N, rank+j, rank+2*j+k, dir);
      }
      else {
        swap_process_data(&in_array, &out_array, N, rank-j, rank+j+k, dir);
      }
      /* if (rank == 1) { */
        /* printf("ppre\n"); */
        /* printf("in_array\n"); */
        /* for (int i = 0; i < N; ++i) { */
          /* printf("%d\n", in_array[i]); */
        /* } */
        /* printf("out_array\n"); */
        /* for (int i = 0; i < N; ++i) { */
          /* printf("%d\n", out_array[i]); */
        /* } */
      /* } */
      compare_and_keep(&in_array, out_array, N, dir);
      /* if (rank == 1) { */
        /* printf("post\n"); */
        /* printf("in_array\n"); */
        /* for (int i = 0; i < N; ++i) { */
          /* printf("%d\n", in_array[i]); */
        /* } */
      /* } */
    }
  }

  /* char filename[10]; */
  /* FILE* f; */
  /* sprintf(filename, "output_%d.txt", rank); */
  /* f = fopen(filename, "w"); */
  /* for (int i = 0; i < N; ++i) { */
    /* fprintf(f, "%d\n", in_array[i]); */
  /* } */
  /* fclose(f); */

/******************************************************************************
 *                         Test validity of algorithm                         *
 ******************************************************************************/

  test_validity(in_array, N, numTasks, rank);

  free(in_array);
  free(out_array);
  free(tmp_array);

  MPI_Finalize();

  return 0;
}

inline void swap(int **x, int **y)
{
  int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;
}

void swap_process_data(int** array_in, int** array_out, int N, int fellow, int tag, int dir)
{
  MPI_Status status;
  if (dir == 0) {
    MPI_Send(*array_in, N, MPI_INT, fellow, tag, MPI_COMM_WORLD);
    MPI_Recv(*array_out, N, MPI_INT, fellow, tag, MPI_COMM_WORLD, &status);
  }
  else {
    MPI_Recv(*array_out, N, MPI_INT, fellow, tag, MPI_COMM_WORLD, &status);
    MPI_Send(*array_in, N, MPI_INT, fellow, tag, MPI_COMM_WORLD);
  }
}

void compare_and_keep(int** array_in, int* array_out, int N, int dir)
{
  int i, in_flag, out_flag;
  out_flag = in_flag = dir == 0 ? N-1 : 0;
  /* if (dir == 0) { */
    /* printf("%d\n", N); */
    /* printf("of: %d\n", out_flag); */
    /* printf("if: %d\n", in_flag); */
  /* } */
  for (i = 0; i < N; ++i) {
    if (dir == 0) {
      if ((*array_in)[in_flag] > array_out[out_flag]) {
        /* printf("%d: %d\n", N-i-1, (*array_in)[in_flag]); */
        tmp_array[N-i-1] = (*array_in)[in_flag--];
      }
      else {
        /* printf("%d: %d\n", N-i-1, (array_out)[out_flag]); */
        tmp_array[N-i-1] = array_out[out_flag--];
      }
    }
    else {
      if ((*array_in)[in_flag] > array_out[out_flag])
        tmp_array[i] = array_out[out_flag++];
      else
        tmp_array[i] = (*array_in)[in_flag++];
    }
  }
  /* if (dir == 0) { */
    /* printf("tmp_array\n"); */
    /* for (int i = 0; i < N; ++i) { */
      /* printf("%d\n", tmp_array[i]); */
    /* } */
  /* } */
  swap(array_in, &tmp_array);
}

void test_validity(int* array, int N, int numTasks, int rank)
{
  int i;
  int final_size = N * numTasks;
  int *final;
  if (rank == 0) {
    final = (int*) malloc(final_size * sizeof(int));
    if (final == NULL) {
      printf("Could not allocate memory for the buffer so as to "
          "receive all the data. The test will not be performed! \n ");
      return;
    }
  }
  MPI_Gather(array, N, MPI_INT, final, N, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    int fail = 0;
    for (i = 1; i < final_size; ++i) {
      fail = (fail || (final[i] < final[i-1]));
      if (fail) break;
    }
    printf("Parallel bitonic sort - validity test: ");
    if (fail) printf("FAIL\n");
    else printf("PASS\n");
  }
}
