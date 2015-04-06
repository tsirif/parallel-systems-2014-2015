#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint32_t uint;

// TODO: a conversion between the two grid of life representations
// int per cell <-> uint per 32 cells in the following configuration
//  0  1  2  3  4  5  6  7
//  8  9 10 11 12 13 14 15
// 16 17 18 19 20 21 22 23
// 24 25 26 27 28 29 30 31
// width of this tile: 8
// height of this tile: 4

// TODO: Test if implementation is correct.

__global__ void calculate_next_generation(
    uint const* d_table, uint* d_result, int dim,
    int block_width, int block_height)
{
  // TODO mby put as arguments only the necessary: m_width, m_height
  const int thread_width = 8;
  const int thread_height = 4;
  const int m_width = dim / thread_width;
  const int m_height = dim / thread_height;
  const int m_size = m_width * m_height;

  int row = (__mul24(blockIdx.x, blockDim.x) + threadIdx.x) * m_width;
  int col = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

  int t_row = (row - m_width + m_size) % m_size;
  int b_row = (row + m_width) % m_size;
  int l_col = (col - 1 + m_width) % m_width;
  int r_col = (col + 1) % m_width;

  // bring information to local memory (global/cache/registers)
  uint this_tile = d_table[row + col];
  uint tl_tile = d_table[t_row + l_col];
  uint t_tile = d_table[t_row + col];
  uint tr_tile = d_table[t_row + r_col];
  uint l_tile = d_table[row + l_col];
  uint r_tile = d_table[row + r_col];
  uint bl_tile = d_table[b_row + l_col];
  uint b_tile = d_table[b_row + col];
  uint br_tile = d_table[b_row + r_col];

  // build resulting tile in local memory (register)
  uint result_tile = 0;
  uint i, alive_cells = 0;
  uint first_cells, second_cells, this_cell;

  // Update vertical edge 1 - 6
  first_cells = (this_tile & 1u) + (this_tile >> 8 & 1u) + (t_tile >> 24 & 1u);
  second_cells = (this_tile >> 1 & 1u) + (this_tile >> 9 & 1u) + (t_tile >> 25 & 1u);
  for (i = 1; i < 7; ++i) {
    this_cell = this_tile >> i & 1u;
    if (i & 1u) {
      alive_cells = first_cells;
      first_cells = (this_tile >> i+1 & 1u) + (this_tile >> i+9 & 1u) +
        (t_tile >> i+25 & 1u);
      alive_cells += first_cells;
      alive_cells += second_cells - this_cell;
    }
    else {
      alive_cells = second_cells;
      second_cells = (this_tile >> i+1 & 1u) + (this_tile >> i+9 & 1u) +
        (t_tile >> i+25 & 1u);
      alive_cells += second_cells;
      alive_cells += first_cells - this_cell;
    }
    result_tile |= alive_cells == 3 || (alive_cells == 2
        && this_cell) ? 1u << i : 0u;
  }

  // Update 9 - 14
  first_cells = (this_tile & 1u) + (this_tile >> 8 & 1u) + (this_tile >> 16 & 1u);
  second_cells = (this_tile >> 1 & 1u) + (this_tile >> 9 & 1u) + (this_tile >> 17 & 1u);
  for (i = 9; i < 15; ++i) {
    this_cell = this_tile >> i & 1u;
    if (i & 1u) {
      alive_cells = first_cells;
      first_cells = (this_tile >> i+1 & 1u) + (this_tile >> i+9 & 1u) +
        (this_tile >> i-7 & 1u);
      alive_cells += first_cells;
      alive_cells += second_cells - this_cell;
    }
    else {
      alive_cells = second_cells;
      second_cells = (this_tile >> i+1 & 1u) + (this_tile >> i+9 & 1u) +
        (this_tile >> i-7 & 1u);
      alive_cells += second_cells;
      alive_cells += first_cells - this_cell;
    }
    result_tile |= alive_cells == 3 || (alive_cells == 2
        && this_cell) ? 1u << i : 0u;
  }

  // Update 17 - 22
  first_cells = (this_tile >> 24 & 1u) + (this_tile >> 8 & 1u) + (this_tile >> 16 & 1u);
  second_cells = (this_tile >> 25 & 1u) + (this_tile >> 9 & 1u) + (this_tile >> 17 & 1u);
  for (i = 17; i < 23; ++i) {
    this_cell = this_tile >> i & 1u;
    if (i & 1u) {
      alive_cells = first_cells;
      first_cells = (this_tile >> i+1 & 1u) + (this_tile >> i+9 & 1u) +
        (this_tile >> i-7 & 1u);
      alive_cells += first_cells;
      alive_cells += second_cells - this_cell;
    }
    else {
      alive_cells = second_cells;
      second_cells = (this_tile >> i+1 & 1u) + (this_tile >> i+9 & 1u) +
        (this_tile >> i-7 & 1u);
      alive_cells += second_cells;
      alive_cells += first_cells - this_cell;
    }
    result_tile |= alive_cells == 3 || (alive_cells == 2
        && this_cell) ? 1u << i : 0u;
  }

  // Update vertical edge 25 - 30
  first_cells = (this_tile >> 24 & 1u) + (b_tile & 1u) + (this_tile >> 16 & 1u);
  second_cells = (this_tile >> 25 & 1u) + (b_tile >> 1 & 1u) + (this_tile >> 17 & 1u);
  for (i = 25; i < 31; ++i) {
    this_cell = this_tile >> i & 1u;
    if (i & 1u) {
      alive_cells = first_cells;
      first_cells = (this_tile >> i-7 & 1u) + (this_tile >> i+1 & 1u) +
        (b_tile >> i-23 & 1u);
      alive_cells += first_cells;
      alive_cells += second_cells - this_cell;
    }
    else {
      alive_cells = second_cells;
      second_cells = (this_tile >> i-7 & 1u) + (this_tile >> i+1 & 1u) +
        (b_tile >> i-23 & 1u);
      alive_cells += second_cells;
      alive_cells += first_cells - this_cell;
    }
    result_tile |= alive_cells == 3 || (alive_cells == 2
        && this_cell) ? 1u << i : 0u;
  }

  // Update corners 0, 7, 24, 31
  alive_cells =
    (tl_tile >> 31) +
    (t_tile >> 24 & 1u) + (t_tile >> 25 & 1u) +
    (this_tile >> 1 & 1u) + (this_tile >> 9 & 1u) + (this_tile >> 8 & 1u) +
    (l_tile >> 7 & 1u) + (l_tile >> 15 & 1u);
  result_tile |= alive_cells == 3 || (alive_cells == 2
      && (this_tile & 1u)) ? 1u : 0u;
  alive_cells =
    (tr_tile >> 24 & 1u) +
    (t_tile >> 30 & 1u) + (t_tile >> 31) +
    (this_tile >> 6 & 1u) + (this_tile >> 14 & 1u) + (this_tile >> 15 & 1u) +
    (r_tile & 1u) + (r_tile >> 8 & 1u);
  result_tile |= alive_cells == 3 || (alive_cells == 2
      && (this_tile >> 7 & 1u)) ? 1u : 0u;
  alive_cells =
    (bl_tile >> 7 & 1u) +
    (b_tile >> 1u) + (b_tile >> 1 & 1u) +
    (this_tile >> 16 & 1u) + (this_tile >> 17 & 1u) + (this_tile >> 25 & 1u) +
    (l_tile >> 23 & 1u) + (l_tile >> 31);
  result_tile |= alive_cells == 3 || (alive_cells == 2
      && (this_tile >> 24 & 1u)) ? 1u : 0u;
  alive_cells =
    (br_tile & 1u) +
    (b_tile >> 6 & 1u) + (b_tile >> 7 & 1u) +
    (this_tile >> 22 & 1u) + (this_tile >> 23 & 1u) + (this_tile >> 30 & 1u) +
    (r_tile >> 16 & 1u) + (r_tile >> 24 & 1u);
  result_tile |= alive_cells == 3 || (alive_cells == 2
      && (this_tile >> 31)) ? 1u : 0u;

  // Update horizontal edges 8, 16, 15, 23
  alive_cells =
    (this_tile & 1u) + (this_tile >> 16 & 1u) +
    (this_tile >> 1 & 1u) + (this_tile >> 9 & 1u) + (this_tile >> 17 & 1u) +
    (l_tile >> 7 & 1u) + (l_tile >> 15 & 1u) + (l_tile >> 23 & 1u);
  result_tile |= alive_cells == 3 || (alive_cells == 2
      && (this_tile >> 8 & 1u)) ? 1u : 0u;
  alive_cells =
    (this_tile >> 8 & 1u) + (this_tile >> 24 & 1u) +
    (this_tile >> 9 & 1u) + (this_tile >> 17 & 1u) + (this_tile >> 25 & 1u) +
    (l_tile >> 31) + (l_tile >> 15 & 1u) + (l_tile >> 23 & 1u);
  result_tile |= alive_cells == 3 || (alive_cells == 2
      && (this_tile >> 16 & 1u)) ? 1u : 0u;
  alive_cells =
    (this_tile >> 7 & 1u) + (this_tile >> 23 & 1u) +
    (this_tile >> 6 & 1u) + (this_tile >> 14 & 1u) + (this_tile >> 22 & 1u) +
    (r_tile & 1u) + (r_tile >> 8 & 1u) + (l_tile >> 16 & 1u);
  result_tile |= alive_cells == 3 || (alive_cells == 2
      && (this_tile >> 15 & 1u)) ? 1u : 0u;
  alive_cells =
    (this_tile >> 15 & 1u) + (this_tile >> 31) +
    (this_tile >> 30 & 1u) + (this_tile >> 14 & 1u) + (this_tile >> 22 & 1u) +
    (r_tile >> 24 & 1u) + (r_tile >> 8 & 1u) + (l_tile >> 16 & 1u);
  result_tile |= alive_cells == 3 || (alive_cells == 2
      && (this_tile >> 23 & 1u)) ? 1u : 0u;

  // send result but to global memory
  d_result[row + col] = result_tile;
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

