#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../utils/utils.h"
//TODO: debloat

#define DFL_RUNS 10   //!< the number of iterations if not else specified
#define POS(i, j) (j + dim * i)



//TODO: change it with nvidia's function
/* Determines the number of threads per block.
 * Returns a power of 2 number that evenly divides the total number of elements*/
int find_thread_count(const int dim)
{
  if (dim == 0) return 0;
  int result = 2;
  while ((dim % result == 0) && (result < 1024)) result *= 2;
  return result >> 1;
}

__global__ void cuda_compute(int* d_result, int* d_table, int dim)
{
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  GROUP_WIDTH = TILE_WIDTH * sqrt(num_of_threads_per_block);

  int row = by * GROUP_WIDTH + ty * TILE_WIDTH;
  int col = bx * GROUP_WIDTH + tx * TILE_WIDTH;

  int tile[TILE_WIDTH+2][TILE_WIDTH+2];  // this tile includes ghost cells
  int result_tile[TILE_WIDTH][TILE_WIDTH];

  int i, j, d_i, d_j;
  // copy from global memory to local memory the tile
  for (i = 0; i < TILE_WIDTH + 2; ++i) {
    for (j = 0; i < TILE_WIDTH + 2; ++j) {
      d_i = (row - 1 + i + dim) % dim;
      d_j = (col - 1 + j + dim) % dim;
      tile[i][j] = d_table[d_j + dim * d_i];
    }
  }

  int left, right, up, down, alive_neighbors;

  for (i = 1; i < TILE_WIDTH + 1; ++i) {
    for (j = 1; j < TILE_WIDTH + 1; ++j) {
      alive_neighbors =
        tile[i - 1][j - 1] + // top-left
        tile[i - 1][j] + // top
        tile[i - 1][j + 1] + // top-right
        tile[i][j - 1] + // left
        tile[i][j + 1] + // right
        tile[i + 1][j - 1] + // bot-left
        tile[i + 1][j] + // bot
        tile[i + 1][j + 1]; // bot-right
      d_i = (row - 1 + i + dim) % dim;
      d_j = (col - 1 + j + dim) % dim;
      result_tile[i - 1][j - 1] = (alive_neighbors == 3) ||
        (alive_neighbors == 2 && tile[i][j]) ? 1 : 0;
      d_result[d_j + dim * d_i] = result_tile[i - 1][j - 1];
    }
  }
}

int main(int argc, char **argv)
{
/******************************************************************************
 *                    Initialization of program variables                     *
 ******************************************************************************/

  if (argc < 3) {
    printf("usage: %s FILE dimension\n", argv[0]);
    exit(1);
  }
  // get number of GOF generations if available else use the default
  int n_runs;
  if (argc == 4) n_runs = atoi(argv[3]);
  else n_runs = DFL_RUNS;
  // get size of GOF matrix
  const int dim = atoi(argv[2]);
  // total cells in the squared sized matrix
  const int total_elements = dim * dim;
  // size of GOF matrix
  const int mem_size = total_elements * sizeof(int);
  // get name of file which contains the initial GOF matrix
  char* filename = argv[1];
  // initialize and parse the matrix out of the file
  int *table;
  printf("Reading %dx%d table from file %s\n", dim, dim, filename);
  table = (int*) malloc(mem_size);
  read_from_file(table, filename, dim);
  printf("Finished reading table\n");
#ifdef PRINT
  print_table(table, dim);
#endif

/******************************************************************************
 *                           Device initialization                            *
 ******************************************************************************/

  // calculate number of blocks and number of threads per block
  int t_count = find_thread_count(total_elements);
  dim3 thread_count(t_count);
  //TODO: fix error with blocks count when the input array is big
  dim3 blocks_count(total_elements / t_count);
  // allocate memory on device
  int *d_table, *d_result;
  cudaMalloc((void **) &d_table,  mem_size);
  cudaCheckErrors("device allocation of GOF matrix failed");
  cudaMalloc((void **) &d_result,  mem_size);
  cudaCheckErrors("device allocation of GOF helper matrix failed");
  // transfer memory from initial matrix from host to device
  cudaMemcpy(d_table, table, mem_size, cudaMemcpyHostToDevice);
  cudaCheckErrors("host to device memory copy failed");

/******************************************************************************
 *                           Calculation execution                            *
 ******************************************************************************/

  // start timewatch
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start) ;
  cudaEventCreate(&stop) ;
  cudaEventRecord(start, 0) ;

  for (int i = 0; i < n_runs; ++i) {
    // compute with gpu
    cuda_compute <<< blocks_count, thread_count >>>(d_result, d_table, dim);
    cudaCheckErrors("computation with gpu failed");
    swap(&d_table, &d_result);
#ifdef PRINT
    cudaMemcpy(table, d_table, mem_size, cudaMemcpyDeviceToHost);
    print_table(table, N);
#endif
  }

  // end timewatch
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("CUDA time to run:  %f s \n", time/1000);

/******************************************************************************
 *                  Device finalization and Result printing                   *
 ******************************************************************************/

  // transfer memory from resulting matrix from device to host
  cudaMemcpy(table, d_table, mem_size, cudaMemcpyDeviceToHost);
  print_table(table, dim);
  cudaCheckErrors("device to host memory copy failed");
  // reset gpu
  cudaDeviceReset();
  // save results to a file
  save_table(table, dim, "cuda-results.bin");
}
