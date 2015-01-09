#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "mpi.h"


inline void exchange(int* a, int i, int* b, int j);
void send(int tag, int receiver, int* in_array, int N);
void receive_and_compare(int tag, int sender,
                         int* in_array, int* out_array, int N, int dir);

int cmpfunc(const void* a, const void* b)
{
   return ( *(const int*)a - *(const int*)b );
}

int main(int argc, char *argv[])
{
  printf("A");
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

  int numTasks, rank, rc, N;

  N = 1<<atoi(argv[2]);

  rc = MPI_Init(&argc,&argv);
  if (rc != MPI_SUCCESS) {
    printf ("Error starting MPI program. Terminating.\n");
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
  printf("B");

  int* array = (int*) malloc(N * sizeof(int));
  int* array_tmp = (int*) malloc(N * sizeof(int));

  if (rank == 0) {
    srand(time(NULL));
  }

  for (int i = 0; i < N; ++i) {
    array[i] = rand();
  }
  printf("C");

  qsort(array, N, sizeof(int), cmpfunc);

  int k, j;

  printf("\nD%d", numTasks);
  for (k = 2; k <= numTasks; k = k<<1) {
    for (j = k>>1; j > 0; j = j>>1) {
      MPI_Barrier(MPI_COMM_WORLD);
      if ((rank ^ j) > rank) {
        send(rank+2*j+k, rank+j, array, N);
      }
      else {
        receive_and_compare(rank+j+k, rank-j, array, array_tmp, N, rank&k);
      }
      qsort(array, N, sizeof(int), cmpfunc);
    }
  }

  char filename[10];
  FILE* f;
  sprintf(filename, "output_%d.txt", rank);
  f = fopen(filename, "w");
  for (int i = 0; i < N; ++i) {
    fprintf(f, "%d\n", in_array[i]);
  }
  fclose(f);

  free(array);
  free(array_tmp);
  MPI_Finalize();

  return 0;
}

/** INLINE procedure exchange() : pair swap **/
inline void exchange(int* a, int i, int* b, int j)
{
  int t;
  t = a[i];
  a[i] = b[j];
  b[j] = t;
}

void send(int tag, int receiver, int* in_array, int N)
{
  /* for (int i = 0; i < N; ++i) {                                          */
  /*   MPI_Isend(&array[i], 1, MPI_INT, receiver, 1, MPI_COMM_WORLD, &req); */
  /* }                                                                      */
  MPI_Status stat;
  /* int* incoming = (int*) malloc(N * sizeof(int)); */
  MPI_Send(in_array, N, MPI_INT, receiver, tag, MPI_COMM_WORLD);
  MPI_Recv(in_array, N, MPI_INT, receiver, tag, MPI_COMM_WORLD, &stat);
}

void receive_and_compare(int tag, int sender,
                         int* in_array, int* out_array, int N, int dir)
{
  MPI_Status stat;
  MPI_Recv(out_array, N, MPI_INT, sender, tag, MPI_COMM_WORLD, &stat);
  for (int i = 0; i < N; ++i) {
    if (dir == 0 && out_array[i] > in_array[N-i-1])
      exchange(out_array, i, in_array, N-i-1);
    if (dir != 0 && out_array[i] < in_array[N-i-1])
      exchange(out_array, i, in_array, N-i-1);
  }
  MPI_Send(out_array, N, MPI_INT, sender, tag, MPI_COMM_WORLD);
}
