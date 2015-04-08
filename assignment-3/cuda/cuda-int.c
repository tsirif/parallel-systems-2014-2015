#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../utils/utils.h"

/**
 * @brief Gets last cuda error and if it's not a cudaSuccess
 * prints debug information on stderr and aborts.
 */
#define cudaCheckErrors(msg, yolo, yolo2) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
              msg, cudaGetErrorString(__err), \
              __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0)

/**
 * @brief The number of iterations (life generations) over the GOL matrix.
 */
#define DFL_RUNS 1
/**
 * @brief The width of a tile assigned to a thread.
 */
#define CONF_WIDTH 8
/**
 * @brief The height of a tile assigned to a thread.
 */
#define CONF_HEIGHT 4

// TODO: a conversion between the two grid of life representations
// int per cell <-> uint per 32 cells in the following configuration
//  0  1  2  3  4  5  6  7
//  8  9 10 11 12 13 14 15
// 16 17 18 19 20 21 22 23
// 24 25 26 27 28 29 30 31
// width of this tile: 8
// height of this tile: 4

// TODO: Test if implementation is correct.

/**
 * @brief Kernel that advances the GOL by one generation.
 * @returns Nothing.
 * @param d_table The GOL matrix of the current generation.
 * @param d_result The resulting GOL matrix of the next generation.
 * @param m_width
 * @param m_height
 * @param m_size
 */
__global__ void calculate_next_generation(
  uint const *d_table, uint *d_result,
  uint m_width, uint m_height, uint m_size)
{
  const int row = (__mul24(blockIdx.x, blockDim.x) + threadIdx.x) * m_width;
  const int col = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;



  const int t_row = (row - m_width + m_size) % m_size;
  const int b_row = (row + m_width) % m_size;
  const int l_col = (col - 1 + m_width) % m_width;
  const int r_col = (col + 1) % m_width;

  if ((row + col    ) >= m_size) return;

  if ((t_row + l_col) >= m_size) return;

  if ((t_row + col  ) >= m_size) return;

  if ((t_row + r_col) >= m_size) return;

  if ((row + l_col  ) >= m_size) return;

  if ((row + r_col  ) >= m_size) return;

  if ((b_row + l_col) >= m_size) return;

  if ((b_row + col  ) >= m_size) return;

  if ((b_row + r_col) >= m_size) return;

  //TODO: write only own tile to shared memory, write some edges of the block in shared memory and then sync and read neighbors from shared memory
  // bring information to local memory (global/cache/registers)
  const uint this_tile = d_table[row + col    ];
  const uint tl_tile = d_table  [t_row + l_col];
  const uint t_tile = d_table   [t_row + col  ];
  const uint tr_tile = d_table  [t_row + r_col];
  const uint l_tile = d_table   [row + l_col  ];
  const uint r_tile = d_table   [row + r_col  ];
  const uint bl_tile = d_table  [b_row + l_col];
  const uint b_tile = d_table   [b_row + col  ];
  const uint br_tile = d_table  [b_row + r_col];

  // build resulting tile in local memory (register)
  uint result_tile = 0;
  uint alive_cells = 0;
  uint first_cells, second_cells, this_cell;

  // Update vertical edge 1 - 6
  first_cells = (this_tile & 1u) +
                (this_tile >> 8 & 1u) +
                (t_tile >> 24 & 1u);
  second_cells = (this_tile >> 1 & 1u) +
                 (this_tile >> 9 & 1u) +
                 (t_tile >> 25 & 1u);

  for (int i = 1; i < 7; ++i) {
    // (x & 1u) == (x % 2) , x >= 0 but mod operator is relatively slow in cuda so we avoid it.
    this_cell = this_tile >> i & 1u;

    if (i & 1u) {
      alive_cells = first_cells;
      first_cells = ((this_tile >> (i + 1)) & 1u) + ((this_tile >> (i + 9)) & 1u) +
                    ((t_tile >> (i + 25)) & 1u);
      alive_cells += first_cells;
      alive_cells += second_cells - this_cell;
    } else {
      alive_cells = second_cells;
      second_cells = ((this_tile >> (i + 1)) & 1u) + ((this_tile >> (i + 9)) & 1u) +
                     ((t_tile >> (i + 25)) & 1u);
      alive_cells += second_cells;
      alive_cells += first_cells - this_cell;
    }

    result_tile |= (alive_cells == 3) || (alive_cells == 2
                                          && this_cell) ? (1u << i) : 0u;
  }

  // Update 9 - 14
  first_cells = (this_tile & 1u) +
                ((this_tile >> 8) & 1u) +
                ((this_tile >> 16) & 1u);
  second_cells = ((this_tile >> 1) & 1u) +
                 ((this_tile >> 9) & 1u) +
                 ((this_tile >> 17) & 1u);

  for (int i = 9; i < 15; ++i) {
    this_cell = (this_tile >> i) & 1u;

    if (i & 1u) {
      alive_cells = first_cells;
      first_cells = ((this_tile >> (i + 1)) & 1u) + ((this_tile >> (i + 9)) & 1u) +
                    ((this_tile >> (i - 7)) & 1u);
      alive_cells += first_cells;
      alive_cells += second_cells - this_cell;
    } else {
      alive_cells = second_cells;
      second_cells = ((this_tile >> (i + 1)) & 1u) + ((this_tile >> (i + 9)) & 1u) +
                     ((this_tile >> (i - 7)) & 1u);
      alive_cells += second_cells;
      alive_cells += first_cells - this_cell;
    }

    result_tile |= (alive_cells == 3) || (alive_cells == 2
                                          && this_cell) ? (1u << i) : 0u;
  }

  // Update 17 - 22
  first_cells = ((this_tile >> 24) & 1u) +
                ((this_tile >> 8) & 1u) +
                ((this_tile >> 16) & 1u);
  second_cells = ((this_tile >> 25) & 1u) +
                 ((this_tile >> 9) & 1u) +
                 ((this_tile >> 17) & 1u);

  for (int i = 17; i < 23; ++i) {
    this_cell = (this_tile >> i) & 1u;

    if (i & 1u) {
      alive_cells = first_cells;
      first_cells = ((this_tile >> (i + 1)) & 1u) + ((this_tile >> (i + 9)) & 1u) +
                    ((this_tile >> (i - 7)) & 1u);
      alive_cells += first_cells;
      alive_cells += second_cells - this_cell;
    } else {
      alive_cells = second_cells;
      second_cells = ((this_tile >> (i + 1)) & 1u) + ((this_tile >> (i + 9)) & 1u) +
                     ((this_tile >> (i - 7)) & 1u);
      alive_cells += second_cells;
      alive_cells += first_cells - this_cell;
    }

    result_tile |= (alive_cells == 3) || (alive_cells == 2
                                          && this_cell) ? (1u << i) : 0u;
  }

  // Update vertical edge 25 - 30
  first_cells = ((this_tile >> 24) & 1u) +
                (b_tile & 1u) +
                ((this_tile >> 16) & 1u);
  second_cells = ((this_tile >> 25) & 1u) +
                 ((b_tile >> 1) & 1u) +
                 ((this_tile >> 17) & 1u);

  for (int i = 25; i < 31; ++i) {
    this_cell = (this_tile >> i) & 1u;

    if (i & 1u) {
      alive_cells = first_cells;
      first_cells = ((this_tile >> (i - 7)) & 1u) + ((this_tile >> (i + 1)) & 1u) +
                    ((b_tile >> (i - 23)) & 1u);
      alive_cells += first_cells;
      alive_cells += second_cells - this_cell;
    } else {
      alive_cells = second_cells;
      second_cells = ((this_tile >> (i - 7)) & 1u) + ((this_tile >> (i + 1)) & 1u) +
                     ((b_tile >> (i - 23)) & 1u);
      alive_cells += second_cells;
      alive_cells += first_cells - this_cell;
    }

    result_tile |= (alive_cells == 3) || (alive_cells == 2
                                          && this_cell) ? (1u << i) : 0u;
  }

  // Update corners 0, 7, 24, 31
  alive_cells =
    (tl_tile >> 31) +
    ((t_tile >> 24) & 1u) +
    ((t_tile >> 25) & 1u) +
    ((this_tile >> 1) & 1u) +
    ((this_tile >> 9) & 1u) +
    ((this_tile >> 8) & 1u) +
    ((l_tile >> 7) & 1u) +
    ((l_tile >> 15) & 1u);
  result_tile |= (alive_cells == 3) || (alive_cells == 2
                                        && (this_tile & 1u)) ? 1u : 0u;
  alive_cells =
    ((tr_tile >> 24) & 1u) +
    ((t_tile >> 30) & 1u) +
    (t_tile >> 31) +
    ((this_tile >> 6) & 1u) +
    ((this_tile >> 14) & 1u) +
    ((this_tile >> 15) & 1u) +
    (r_tile & 1u) +
    ((r_tile >> 8) & 1u);
  result_tile |= (alive_cells == 3) || (alive_cells == 2
                                        && (this_tile >> 7 & 1u)) ? 1u : 0u;
  alive_cells =
    ((bl_tile >> 7) & 1u) +
    (b_tile >> 1u) +
    ((b_tile >> 1) & 1u) +
    ((this_tile >> 16) & 1u) +
    ((this_tile >> 17) & 1u) +
    ((this_tile >> 25) & 1u) +
    ((l_tile >> 23) & 1u) +
    (l_tile >> 31);
  result_tile |= (alive_cells == 3) || (alive_cells == 2
                                        && (this_tile >> 24 & 1u)) ? 1u : 0u;
  alive_cells =
    (br_tile & 1u) +
    ((b_tile >> 6) & 1u) + ((b_tile >> 7) & 1u) +
    ((this_tile >> 22) & 1u) + ((this_tile >> 23) & 1u) + ((this_tile >> 30) & 1u) +
    ((r_tile >> 16) & 1u) + ((r_tile >> 24) & 1u);
  result_tile |= (alive_cells == 3) || (alive_cells == 2
                                        && (this_tile >> 31)) ? 1u : 0u;

  // Update horizontal edges 8, 16, 15, 23
  alive_cells =
    (this_tile & 1u) +
    ((this_tile >> 16) & 1u) +
    ((this_tile >> 1) & 1u) +
    ((this_tile >> 9) & 1u) +
    ((this_tile >> 17) & 1u) +
    ((l_tile >> 7) & 1u) +
    ((l_tile >> 15) & 1u) +
    ((l_tile >> 23) & 1u);
  result_tile |= (alive_cells == 3) || (alive_cells == 2
                                        && (this_tile >> 8 & 1u)) ? 1u : 0u;
  alive_cells =
    ((this_tile >> 8) & 1u) +
    ((this_tile >> 24) & 1u) +
    ((this_tile >> 9) & 1u) +
    ((this_tile >> 17) & 1u) +
    ((this_tile >> 25) & 1u) +
    (l_tile >> 31) +
    ((l_tile >> 15) & 1u) +
    ((l_tile >> 23) & 1u);
  result_tile |= (alive_cells == 3) || (alive_cells == 2
                                        && (this_tile >> 16 & 1u)) ? 1u : 0u;
  alive_cells =
    ((this_tile >> 7) & 1u) +
    ((this_tile >> 23) & 1u) +
    ((this_tile >> 6) & 1u) +
    ((this_tile >> 14) & 1u) +
    ((this_tile >> 22) & 1u) +
    (r_tile & 1u) +
    ((r_tile >> 8) & 1u) +
    ((l_tile >> 16) & 1u);
  result_tile |= (alive_cells == 3) || (alive_cells == 2
                                        && (this_tile >> 15 & 1u)) ? 1u : 0u;
  alive_cells =
    ((this_tile >> 15) & 1u) +
    (this_tile >> 31) +
    ((this_tile >> 30) & 1u) +
    ((this_tile >> 14) & 1u) +
    ((this_tile >> 22) & 1u) +
    ((r_tile >> 24) & 1u) +
    ((r_tile >> 8) & 1u) + (l_tile >> 16 & 1u);
  result_tile |= (alive_cells == 3) || (alive_cells == 2
                                        && (this_tile >> 23 & 1u)) ? 1u : 0u;

  // send result but to global memory
  d_result[row + col] = result_tile;
}

/**
 * @brief Creates a tiled GOL matrix from a normal GOL matrix.
 * @returns Nothing.
 * @param d_table The normal GOL matrix.
 * @param d_utable The output tiled GOL matrix
 * @param m_width
 * @param m_height
 * @param m_size
 */
__global__ void convert_to_tiled(
  int const *d_table, uint *d_utable,
  uint m_width, uint m_height, uint m_size)
{
  int row = (__mul24(blockIdx.x, blockDim.x) + threadIdx.x) * m_width;
  int col = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
  int start_i = row * CONF_WIDTH * CONF_HEIGHT;
  int start_j = col * CONF_WIDTH;
  uint place = 1u;
  uint tile = 0u;

  if ((row + col) >= m_size) return; //del

  const int step_i = m_width * CONF_WIDTH;
  const int end_i = start_i + CONF_HEIGHT * step_i;
  const int end_j = start_j + CONF_WIDTH;
  int i, j;

  for (i = start_i; i < end_i; i += step_i) {
    for (j = start_j; j < end_j; ++j) {
      if (d_table[j + i])
        tile |= place;

      place <<= 1;
    }
  }

  d_utable[col + row] = tile;
}

/**
 * @brief Creates a normal GOL matrix from a tiled GOL matrix.
 * @returns Nothing.
 * @param d_table The output normal GOL matrix.
 * @param d_utable The tiled GOL matrix
 * @param m_width
 * @param m_height
 * @param m_size
 */
__global__ void convert_from_tiled(
  int *d_table, uint const *d_utable,
  uint m_width, uint m_height, uint m_size)
{
  int row = (__mul24(blockIdx.x, blockDim.x) + threadIdx.x) * m_width;
  int col = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
  int start_i = row * CONF_WIDTH * CONF_HEIGHT;
  int start_j = col * CONF_WIDTH;
  int place = 0;

  if ((row + col) >= m_size) return; //del

  const uint tile = d_utable[col + row];

  const int step_i = m_width * CONF_WIDTH;
  const int end_i = start_i + CONF_HEIGHT * step_i;
  const int end_j = start_j + CONF_WIDTH;
  int i, j;

  for (i = start_i; i < end_i; i += step_i) {
    for (j = start_j; j < end_j; ++j) {
      d_table[j + i] = (int) (tile >> place++ & 1u);
  }
}


/**
 * @brief Main function
 * */
int main(int argc, char **argv)
{
  /******************************************************************************
   *                    Initialization of program variables                     *
   ******************************************************************************/

  if (argc < 3) {
    printf("usage: %s filename dimension (iterations)\n", argv[0]);
    exit(1);
  }

  int n_runs;

  // get number of GOL generations if available else use the default
  if (argc == 4) n_runs = atoi(argv[3]);
  else n_runs = DFL_RUNS;

  /* The dimension of one side of the GOL square matrix. */
  const uint dim = atoi(argv[2]);
  /* Total cells in the GOL square matrix. */
  const uint total_elements = dim * dim;
  /* Size of the GOL matrix in bytes. */
  const uint mem_size = total_elements * sizeof(int);
  /* The width of a tile assigned to a thread. */
  const uint thread_width = CONF_WIDTH;
  /* The height of a tile assigned to a thread. */
  const uint thread_height = CONF_HEIGHT;
  /* Example for a 8x4 tile:
   * cuda-int implementation:
   * 32 cells in sizeof(int) bytes = 4 bytes = 32 bits => 1 cell : 1 bit
   * simplistic implementation:
   * 1 : 4 bytes = 32 bits
   *
   * cuda-int implementation is 32 times smaller in memory!
   * */
  /* Total cells in the tiled GOL matrix. */
  const uint total_elements_tiled = total_elements / (thread_height * thread_width);
  /* The total size of the tiled GOL matrix in bytes. */
  const uint mem_size_tiled = mem_size / (thread_height * thread_width);
  /* Number of tiles in width. */
  const uint m_width = dim / thread_width;
  /* Number of tiles in height. */
  const uint m_height = dim / thread_height;
  // get name of file which contains the initial GOL matrix

  const dim3 block(5, 10);
  const dim3 grid(25, 25);

  char *filename = argv[1];
  // initialize and parse the matrix out of the file
  int *table;
  printf("Reading %dx%d table from file %s\n", dim, dim, filename);
  table = (int *) malloc(mem_size);
  read_from_file(table, filename, dim);
  printf("Finished reading table\n");

  /******************************************************************************
   *                              Table Conversion                              *
   ******************************************************************************/

  /******************************************************************************
   *                           Device initialization                            *
   ******************************************************************************/

  // Allocate memory on device.
  uint *d_tiled_table; /* Tiled matrix in device memory. */
  int *d_table; /* Original GOl matrix in device memory. */
  cudaMalloc((void **) &d_table,  mem_size);
  cudaCheckErrors("device allocation of GOL matrix failed", __FILE__, __LINE__);
  cudaMalloc((void **) &d_tiled_table, mem_size_tiled);
  cudaCheckErrors("device allocation of GOL uint tiled matrix failed", __FILE__, __LINE__);

  // Transfer memory from initial matrix from host to device.
  cudaMemcpy(d_table, table, mem_size, cudaMemcpyHostToDevice);
  cudaCheckErrors("copy from host to device memory failed", __FILE__, __LINE__);

  convert_to_tiled <<< grid, block >>>(d_table, d_tiled_table,
                                       m_width, m_height, total_elements_tiled);
  cudaCheckErrors("failed to convert normal repr to uint tiled repr", __FILE__, __LINE__);

  // wtf????
  // ~ cudaFree((void *) d_table);
  // ~ cudaCheckErrors("device freeing of GOL matrix failed", __FILE__, __LINE__);

  /******************************************************************************
   *                           Calculation execution                            *
   ******************************************************************************/

  // start timewatch
  /* float time;                 */
  /* cudaEvent_t start, stop;    */
  /* cudaEventCreate(&start) ;   */
  /* cudaEventCreate(&stop) ;    */
  /* cudaEventRecord(start, 0) ; */

  // calculate iterations of game of life with GPU
  uint *d_tiled_help; /* Tiled help matrix in device memory. */
  cudaMalloc((void **) &d_tiled_help, total_elements_tiled);
  cudaCheckErrors("device allocation of help matrix failed", __FILE__, __LINE__);

  //TODO: synchronize here?
  for (int i = 0; i < n_runs; ++i) {
    calculate_next_generation <<< grid, block >>>(
      d_tiled_table, d_tiled_help, m_width, m_height, total_elements_tiled);
    cudaCheckErrors("calculating next generation failed", __FILE__, __LINE__);
    swap(&d_tiled_table, &d_tiled_help);
  }

  // ~ cudaFree((void *) d_tiled_help);
  // ~ cudaCheckErrors("device freeing of help matrix failed", __FILE__, __LINE__);

  // end timewatch
  /* cudaEventRecord(stop, 0);                        */
  /* cudaEventSynchronize(stop);                      */
  /* cudaEventElapsedTime(&time, start, stop);        */
  /* printf("CUDA time to run:  %f s \n", time/1000); */

  /******************************************************************************
   *                  Device finalization and Result printing                   *
   ******************************************************************************/

  // allocation again of a matrix that holds the normal representation
  // ~ cudaMalloc((void **) &d_table,  mem_size);
  // ~ cudaCheckErrors("device allocation of GOL matrix failed", __FILE__, __LINE__);

  // convert back to normal representation of the matrix
  convert_from_tiled <<< grid, block >>>(d_table, d_tiled_table,
                                         m_width, m_height, total_elements_tiled);
  cudaCheckErrors("failed to convert to normal repr from uint tiled repr", __FILE__, __LINE__);

  // transfer memory from resulting matrix from device to host
  cudaMemcpy(table, d_table, mem_size, cudaMemcpyDeviceToHost);
  cudaCheckErrors("copy from device to host memory failed", __FILE__, __LINE__);

  // reset gpu
  cudaDeviceReset();

  // save results to a file
  save_table(table, dim, "cuda-2-results.bin");
}
