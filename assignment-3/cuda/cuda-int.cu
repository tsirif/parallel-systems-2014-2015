#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "../utils/utils.h"

// For double precision GPUs, compile with -DDOUBLE
// e.g. for our Makefile, you should invoke "make <target> PRECISION=-DDOUBLE"

/* __device__ void PRINT_TILE(pint tile)                                                                             */
/* {                                                                                                                 */
/*   for(int xxx = 0; xxx < CONF_HEIGHT; xxx++) {                                                                    */
/*     for(int yyy = 0; yyy < CONF_WIDTH; yyy++) {                                                                   */
/*       printf("%s "ANSI_COLOR_RESET, (tile >> (yyy + CONF_WIDTH * xxx) & ONE) ? ANSI_COLOR_BLUE : ANSI_COLOR_RED); */
/*     }                                                                                                             */
/*     printf("\n");                                                                                                 */
/*   }                                                                                                               */
/* }                                                                                                                 */

/**
 * @brief Gets last cuda error and if it's not a cudaSuccess
 * prints debug information on stderr and aborts.
 */
inline void cudaCheckErrors(const char msg[], const char file[], int line)
{
  do {
    cudaError_t __err = cudaGetLastError();
    if (__err != cudaSuccess) {
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",
              msg, cudaGetErrorString(__err),
              file, line);
      cudaDeviceReset();
      exit(1);
    }
  } while (0);
}

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
 * @param m_height
 * @param m_width
 * @param m_size
 */
__global__ void calculate_next_generation(
  pint const *d_table, pint *d_result,
  uint m_height, uint m_width, uint m_size)
{
  const int row = (__mul24(blockIdx.x, blockDim.x) + threadIdx.x) * m_width;
  const int col = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

#ifdef PRINT
  printf("Thread %d-%d: %u x %u == %u\n", row, col, m_height, m_width, m_size);
#endif  // PRINT

  if (row >= m_size) return;
  if (col >= m_width) return;

  const int t_row = (row - m_width + m_size) % m_size;
  const int b_row = (row + m_width) % m_size;
  const int l_col = (col - 1 + m_width) % m_width;
  const int r_col = (col + 1) % m_width;

#ifdef PRINT
  printf("Thread %d-%d:\n"
  "top row: %d, bottom row: %d\n"
  "left col: %d, right col: %d\n", row, col, t_row, b_row, l_col, r_col);
#endif  // PRINT

  //TODO: write only own tile to shared memory, write some edges of the block in shared memory and then sync and read neighbors from shared memory
  // bring information to local memory (global/cache/registers)
  const pint this_tile = d_table[row + col    ];
  const pint tl_tile = d_table  [t_row + l_col];
  const pint t_tile = d_table   [t_row + col  ];
  const pint tr_tile = d_table  [t_row + r_col];
  const pint l_tile = d_table   [row + l_col  ];
  const pint r_tile = d_table   [row + r_col  ];
  const pint bl_tile = d_table  [b_row + l_col];
  const pint b_tile = d_table   [b_row + col  ];
  const pint br_tile = d_table  [b_row + r_col];

#ifdef PRINT
  printf("Thread %d-%d:\n"
  "tl: %lu, t: %lu, tr: %lu\n"
  "l: %lu, this: %lu, r: %lu\n"
  "bl: %lu, b: %lu, br: %lu\n", row, col,
  tl_tile, t_tile, tr_tile,
  l_tile, this_tile, r_tile,
  bl_tile, b_tile, br_tile);
#endif  // PRINT


  // build resulting tile in local memory (register)
  int tr = CONF_WIDTH - 1;
  int bl = CONF_WIDTH * (CONF_HEIGHT - 1), br = CONF_WIDTH * CONF_HEIGHT - 1;
  int i = 0, j = 0;
  pint result_tile = 0;
  uint alive_cells;
  uint first_cells, second_cells;

  // if we represent a 8x4 array A as int then we can access position [i,j] of the array like this:
  //(A >> (i + 8 * j) & ONE)

  // Update first horizontal row
  first_cells = (this_tile & ONE) +
                (this_tile >> CONF_WIDTH & ONE) +
                (t_tile >> bl & ONE);
  second_cells = (this_tile >> 1 & ONE) +
                 (this_tile >> (CONF_WIDTH + 1) & ONE) +
                 (t_tile >> (bl + 1) & ONE);
  //TODO: IDEA: replace (x >> p & 1) with ((x & (2**p)) != 0)
  //TODO: pragma unroll probably doesn't cause any problems. reenable it after code works correctly.
  //TODO: IDEA: instead of having an if statement inside the loop have first_cells and seconds_cells in an array[2]
  // ~ #pragma unroll
  for (i = 1; i < CONF_WIDTH - 1; ++i) {
    uint this_cell = this_tile >> i & ONE;

    // (x & ONE) == (x % 2) , x >= 0 but mod operator is relatively slow in cuda so we avoid it.
    if (i & ONE) {
      alive_cells = first_cells;
      first_cells = (this_tile >> (i + 1) & ONE) +
                    (this_tile >> (i + CONF_WIDTH + 1) & ONE) +
                    (t_tile >> (i + bl + 1) & ONE);
      alive_cells += first_cells;
      alive_cells += second_cells - this_cell;
    } else {
      alive_cells = second_cells;
      second_cells = (this_tile >> (i + 1) & ONE) +
                     (this_tile >> (i + CONF_WIDTH + 1) & ONE) +
                     (t_tile >> (i + bl + 1) & ONE);
      alive_cells += second_cells;
      alive_cells += first_cells - this_cell;
    }

    //TODO: replace and profile with:
    // ~ result_tile |= ((alive_cells == 3) || (alive_cells == 2 && this_cell)) * (ONE << i);
    result_tile |= (alive_cells == 3) || (alive_cells == 2
                                          && this_cell) ? (ONE << i) : ZERO;
  }
/* #if defined(PRINT) and defined(DOUBLE)           */
/*   printf("Thread %d-%d: Result-tile is now:\n"); */
/*   PRINT_TILE(result_tile);                       */
/* #endif  // PRINT and DOUBLE                      */

  int start_i, end_i;
  // Update cells in the middle
  for (j = CONF_WIDTH; j < (CONF_HEIGHT - 1) * CONF_WIDTH; j += CONF_WIDTH) {
#if defined(PRINT) and defined(DOUBLE)
    printf("Thread %d-%d: Checking tile-row with j: %d\n", row, col, j);
#endif  // PRINT and DOUBLE
    start_i = j + 1;
    end_i = j + CONF_WIDTH - 1;
    first_cells = (this_tile >> (j - CONF_WIDTH) & ONE) +
                  (this_tile >> j & ONE) +
                  (this_tile >> (j + CONF_WIDTH) & ONE);
    second_cells = (this_tile >> (j + 1 - CONF_WIDTH) & ONE) +
                  (this_tile >> (j + 1) & ONE) +
                  (this_tile >> (j + 1 + CONF_WIDTH) & ONE);
    // ~ #pragma unroll
    for (i = start_i; i < end_i; ++i) {
      uint this_cell = (this_tile >> i) & ONE;
      if (i & ONE) {
        alive_cells = first_cells;
        first_cells = (this_tile >> (i + 1) & ONE) +
                      (this_tile >> (i + 1 + CONF_WIDTH) & ONE) +
                      (this_tile >> (i + 1 - CONF_WIDTH) & ONE);
        alive_cells += first_cells;
        alive_cells += second_cells - this_cell;
      } else {
        alive_cells = second_cells;
        second_cells = (this_tile >> (i + 1) & ONE) +
                      (this_tile >> (i + 1 + CONF_WIDTH) & ONE) +
                      (this_tile >> (i + 1 - CONF_WIDTH) & ONE);
        alive_cells += second_cells;
        alive_cells += first_cells - this_cell;
      }

      result_tile |= (alive_cells == 3) || (alive_cells == 2
                                            && this_cell) ? (ONE << i) : ZERO;
    }
/* #if defined(PRINT) and defined(DOUBLE)           */
/*   printf("Thread %d-%d: Result-tile is now:\n"); */
/*   PRINT_TILE(result_tile);                       */
/* #endif  // PRINT and DOUBLE                      */
  }

#ifdef PRINT
  printf("Thread %d-%d: Last horizontal begins with j: %d\n", row, col, j);
#endif  // PRINT

  // Update last horizontal row
  start_i = j + 1;
  end_i = j + CONF_WIDTH - 1;
  first_cells = (this_tile >> (j - CONF_WIDTH) & ONE) +
                (this_tile >> j & ONE) +
                (b_tile & ONE);
  second_cells = (this_tile >> (j - CONF_WIDTH + 1) & ONE) +
                (this_tile >> (j + 1) & ONE) +
                (b_tile >> 1 & ONE);
  // ~ #pragma unroll
  for (i = start_i; i < end_i; ++i) {
    uint this_cell = (this_tile >> i) & ONE;
    if (i & ONE) {
      alive_cells = first_cells;
      first_cells = (this_tile >> (i - CONF_WIDTH + 1) & ONE) +
                    (this_tile >> (i + 1) & ONE) +
                    (b_tile >> (i - j + 1) & ONE);
      alive_cells += first_cells;
      alive_cells += second_cells - this_cell;
    } else {
      alive_cells = second_cells;
      second_cells = (this_tile >> (i - CONF_WIDTH + 1) & ONE) +
                    (this_tile >> (i + 1) & ONE) +
                    (b_tile >> (i - j + 1) & ONE);
      alive_cells += second_cells;
      alive_cells += first_cells - this_cell;
    }
    result_tile |= (alive_cells == 3) || (alive_cells == 2
                                          && this_cell) ? (ONE << i) : ZERO;
  }
/* #if defined(PRINT) and defined(DOUBLE)           */
/*   printf("Thread %d-%d: Result-tile is now:\n"); */
/*   PRINT_TILE(result_tile);                       */
/* #endif  // PRINT and DOUBLE                      */

  // Update vertical col to the left
  first_cells = (l_tile >> tr & ONE) +
                (this_tile & ONE) +
                (this_tile >> 1 & ONE);
  second_cells = (l_tile >> (tr + CONF_WIDTH) & ONE) +
                 (this_tile >> CONF_WIDTH & ONE) +
                 (this_tile >> (CONF_WIDTH + 1) & ONE);
  for (i = CONF_WIDTH; i < (CONF_HEIGHT - 1) * CONF_WIDTH; i += CONF_WIDTH) {
    uint this_cell = this_tile >> i & ONE;
    if ((i/CONF_WIDTH) & ONE) {
      alive_cells = first_cells;
      first_cells = (l_tile >> (i + 2 * CONF_WIDTH - 1) & ONE) +
                    (this_tile >> (i + CONF_WIDTH) & ONE) +
                    (this_tile >> (i + CONF_WIDTH + 1) & ONE);
      alive_cells += first_cells;
      alive_cells += second_cells - this_cell;
    } else {
      alive_cells = second_cells;
      second_cells = (l_tile >> (i + 2 * CONF_WIDTH - 1) & ONE) +
                    (this_tile >> (i + CONF_WIDTH) & ONE) +
                    (this_tile >> (i + CONF_WIDTH + 1) & ONE);
      alive_cells += second_cells;
      alive_cells += first_cells - this_cell;
    }
    result_tile |= (alive_cells == 3) || (alive_cells == 2
                                          && this_cell) ? (ONE << i) : ZERO;
  }
/* #if defined(PRINT) and defined(DOUBLE)           */
/*   printf("Thread %d-%d: Result-tile is now:\n"); */
/*   PRINT_TILE(result_tile);                       */
/* #endif  // PRINT and DOUBLE                      */

  // Update vertical col to the right
  first_cells = (this_tile >> (tr - 1) & ONE) +
                (this_tile >> tr & ONE) +
                (r_tile & ONE);
  second_cells = (this_tile >> (tr + CONF_WIDTH - 1) & ONE) +
                 (this_tile >> (tr + CONF_WIDTH) & ONE) +
                 (r_tile >> (tr + 1) & ONE);
  for (i = 2 * CONF_WIDTH - 1 ; i < CONF_HEIGHT * CONF_WIDTH - 1; i += CONF_WIDTH) {
    uint this_cell = this_tile >> i & ONE;
    if (((i+1)/CONF_WIDTH) & ONE) {
      alive_cells = second_cells;
      second_cells = (this_tile >> (i + CONF_WIDTH - 1) & ONE) +
                    (this_tile >> (i + CONF_WIDTH) & ONE) +
                    (r_tile >> (i + 1) & ONE);
      alive_cells += second_cells;
      alive_cells += first_cells - this_cell;
    } else {
      alive_cells = first_cells;
      first_cells = (this_tile >> (i + CONF_WIDTH - 1) & ONE) +
                    (this_tile >> (i + CONF_WIDTH) & ONE) +
                    (r_tile >> (i + 1) & ONE);
      alive_cells += first_cells;
      alive_cells += second_cells - this_cell;
    }
    result_tile |= (alive_cells == 3) || (alive_cells == 2
                                          && this_cell) ? (ONE << i) : ZERO;
  }
/* #if defined(PRINT) and defined(DOUBLE)           */
/*   printf("Thread %d-%d: Result-tile is now:\n"); */
/*   PRINT_TILE(result_tile);                       */
/* #endif  // PRINT and DOUBLE                      */

  // Update corners 0, 7, 24, 31
  // Update 0. Needs t, tl, l.
  //TODO: use ILP? http://en.wikipedia.org/wiki/Instruction-level_parallelism
  alive_cells =
    (tl_tile >> br) +
    (t_tile >> bl & ONE) +
    (t_tile >> (bl + 1) & ONE) +
    (this_tile >> 1 & ONE) +
    (this_tile >> (CONF_WIDTH + 1) & ONE) +
    (this_tile >> CONF_WIDTH & ONE) +
    (l_tile >> tr & ONE) +
    (l_tile >> (tr + CONF_WIDTH) & ONE);
  result_tile |= (alive_cells == 3) || (alive_cells == 2
                                        && (this_tile & ONE)) ? ONE : ZERO;
  alive_cells =
    (tr_tile >> bl & ONE) +
    (t_tile >> (br - 1) & ONE) +
    (t_tile >> br) +
    (this_tile >> (tr - 1) & ONE) +
    (this_tile >> (tr + CONF_WIDTH - 1) & ONE) +
    (this_tile >> (tr + CONF_WIDTH) & ONE) +
    (r_tile & ONE) +
    (r_tile >> CONF_WIDTH & ONE);
  result_tile |= (alive_cells == 3) || (alive_cells == 2
                                        && (this_tile >> tr & ONE)) ? (ONE << tr) : ZERO;
  alive_cells =
    (bl_tile >> tr & ONE) +
    (b_tile & ONE) +
    (b_tile >> 1 & ONE) +
    (this_tile >> (bl - CONF_WIDTH) & ONE) +
    (this_tile >> (bl - CONF_WIDTH + 1) & ONE) +
    (this_tile >> (bl + 1) & ONE) +
    (l_tile >> (br - CONF_WIDTH) & ONE) +
    (l_tile >> br);
  result_tile |= (alive_cells == 3) || (alive_cells == 2
                                        && (this_tile >> bl & ONE)) ? (ONE << bl) : ZERO;
  alive_cells =
    (br_tile & ONE) +
    (b_tile >> (tr - 1) & ONE) +
    (b_tile >> tr & ONE) +
    (this_tile >> (br - CONF_WIDTH - 1) & ONE) +
    (this_tile >> (br - CONF_WIDTH) & ONE) +
    (this_tile >> (br - 1) & ONE) +
    (r_tile >> (bl - CONF_WIDTH) & ONE) +
    (r_tile >> bl & ONE);
  result_tile |= (alive_cells == 3) || (alive_cells == 2
                                        && (this_tile >> br)) ? (ONE << br) : ZERO;
/* #if defined(PRINT) and defined(DOUBLE)                                                                                  */
/*   printf("Thread %d-%d: Result-tile is now:\n");                                                                        */
/*   for(int xxx = 0; xxx < CONF_HEIGHT; xxx++) {                                                                          */
/*     for(int yyy = 0; yyy < CONF_WIDTH; yyy++) {                                                                         */
/*       printf("%s "ANSI_COLOR_RESET, (result_tile >> (yyy + CONF_WIDTH * xxx) & ONE) ? ANSI_COLOR_BLUE : ANSI_COLOR_RED); */
/*     }                                                                                                                   */
/*     printf("\n");                                                                                                       */
/*   }                                                                                                                     */
/* #endif  // PRINT and DOUBLE                                                                                             */

  // send result but to global memory
  d_result[row + col] = result_tile;
}

/**
 * @brief Creates a tiled GOL matrix from a normal GOL matrix.
 * @returns Nothing.
 * @param d_table The normal GOL matrix.
 * @param d_utable The output tiled GOL matrix
 * @param m_height
 * @param m_width
 * @param m_size
 */
__global__ void convert_to_tiled(
  int const *d_table, pint *d_utable,
  uint m_height, uint m_width, uint m_size)
{
  const int row = (__mul24(blockIdx.x, blockDim.x) + threadIdx.x) * m_width;
  const int col = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

  if (row >= m_size) return;
  if (col >= m_width) return;

  const int start_i = row * CONF_WIDTH * CONF_HEIGHT;
  const int start_j = col * CONF_WIDTH;
  pint place = ONE;
  pint tile = ZERO;

  const int step_i = m_width * CONF_WIDTH;
  const int end_i = start_i + CONF_HEIGHT * step_i;
  const int end_j = start_j + CONF_WIDTH;
  int i, j;

#pragma unroll
  for (i = start_i; i < end_i; i += step_i) {
#pragma unroll
    for (j = start_j; j < end_j; ++j) {
      tile |= place * (pint) d_table[j + i];
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
 * @param m_height
 * @param m_width
 * @param m_size
 */
__global__ void convert_from_tiled(
  int *d_table, pint const *d_utable,
  uint m_height, uint m_width, uint m_size)
{
  const int row = (__mul24(blockIdx.x, blockDim.x) + threadIdx.x) * m_width;
  const int col = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

  if (row >= m_size) return;
  if (col >= m_width) return;

  const int start_i = row * CONF_WIDTH * CONF_HEIGHT;
  const int start_j = col * CONF_WIDTH;
  int place = 0;

  const pint tile = d_utable[col + row];

  const int step_i = m_width * CONF_WIDTH;
  const int end_i = start_i + CONF_HEIGHT * step_i;
  const int end_j = start_j + CONF_WIDTH;
  int i, j;

#pragma unroll
  for (i = start_i; i < end_i; i += step_i) {
#pragma unroll
    for (j = start_j; j < end_j; ++j)
      d_table[j + i] = (int) (tile >> place++ & ONE);
  }
}

#ifndef TESTING

/**
 * @brief Main function
 * */
int main(int argc, char **argv)
{
  /******************************************************************************
   *                    Initialization of program variables                     *
   ******************************************************************************/
#ifdef PRINT
  printf("Size of pint is: %lu\n", sizeof(pint));
#endif  // PRINT

  if (argc < 3) {
    printf("usage: %s fname dim (iter blockx blocky gridx gridy)\n", argv[0]);
    exit(1);
  }

  int n_runs;

  // get number of GOL generations if available else use the default
  if (argc >= 4) n_runs = atoi(argv[3]);
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
   * speed-up is also achieved because of less movement across global, cache,
   * register and shared memories and because registers and operations in
   * single-precision GPUs are using 32 bits.
   **/
  /* Total cells in the tiled GOL matrix. */
  const uint total_elements_tiled = total_elements / (thread_height * thread_width);
  /* The total size of the tiled GOL matrix in bytes. */
  const uint mem_size_tiled = mem_size / (thread_height * thread_width);
  /* Number of tiles in width. */
  const uint m_width = dim / thread_width;
  /* Number of tiles in height. */
  const uint m_height = dim / thread_height;
  // get name of file which contains the initial GOL matrix


  // ~ int blockSize;
  // ~ int minGridSize;
  // ~ cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, calculate_next_generation);
  // ~ printf("%d %d\n", blockSize, minGridSize);

  // Warning! grid and block sizes that correspond to a bigger array will cause leaks
  // these leaks in convert_to and convert_from are currently harmless (no failure)
  // dim == 1000 == 8 * 5 * 25
  // dim == 1000 == 4 * 10 * 25
  // x > y (?)

  printf("%s: Normal grid repr: %u x %u\n"
      "%s: Tiled grid repr: %u x %u\n", argv[0], dim, dim, argv[0], m_height, m_width);

  dim3 block(1, 1);
  dim3 grid(1, 1);

  if (argc >= 8) {
    block = dim3(atoi(argv[4]), atoi(argv[5]));
    grid = dim3(atoi(argv[6]), atoi(argv[7]));
  }
  else {
    unsigned int x = 32u;
    unsigned int y = 32u;
    do {
      x >>= 1;
    } while(x >= m_height);
    do {
      y >>= 1;
    } while(y >= m_width);
    block = dim3(x, y);
    grid = dim3((int)(ceil(m_height/(float)x)), (int)(ceil(m_width/(float)y)));
  }

  char *filename = argv[1];
  // initialize and parse the matrix out of the file
  int *table;
  printf("%s: Reading %dx%d table from file %s\n", argv[0], dim, dim, filename);
  table = (int *) malloc(mem_size);
  read_from_file(table, filename, dim, dim);
  printf("%s: Finished reading table\n", argv[0]);
#ifdef PRINT
  print_table(table, dim, dim);
#endif  // PRINT

  printf("%s: Running on a grid(%d, %d) with a block(%d, %d):\nFilename: %s with dim %d for %d iterations\n",
      argv[0], grid.x, grid.y, block.x, block.y, filename, dim, n_runs);

  /******************************************************************************
   *                              Table Conversion                              *
   ******************************************************************************/

  /******************************************************************************
   *                           Device initialization                            *
   ******************************************************************************/

  // Allocate memory on device.
  pint *d_tiled_table; /* Tiled matrix in device memory. */
  int *d_table; /* Original GOl matrix in device memory. */
  cudaMalloc((void **) &d_table,  mem_size);
  cudaCheckErrors("device allocation of GOL matrix failed", __FILE__, __LINE__);
  cudaMalloc((void **) &d_tiled_table, mem_size_tiled);
  cudaCheckErrors("device allocation of GOL uint tiled matrix failed", __FILE__, __LINE__);

  // Transfer memory from initial matrix from host to device.
  cudaMemcpy(d_table, table, mem_size, cudaMemcpyHostToDevice);
  cudaCheckErrors("copy from host to device memory failed", __FILE__, __LINE__);

  convert_to_tiled <<< grid, block >>>(d_table, d_tiled_table,
                                       m_height, m_width, total_elements_tiled);
  cudaCheckErrors("failed to convert normal repr to uint tiled repr", __FILE__, __LINE__);

  cudaFree((void *) d_table);
  cudaCheckErrors("device freeing of GOL matrix failed", __FILE__, __LINE__);

  /******************************************************************************
   *                           Calculation execution                            *
   ******************************************************************************/

  // calculate iterations of game of life with GPU
  pint *d_tiled_help; /* Tiled help matrix in device memory. */
  cudaMalloc((void **) &d_tiled_help, mem_size_tiled);
  cudaCheckErrors("device allocation of help matrix failed", __FILE__, __LINE__);

  // start timewatch
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //TODO: synchronize here?
  for (int i = 0; i < n_runs; ++i) {
    calculate_next_generation <<< grid, block >>>(
      d_tiled_table, d_tiled_help, m_height, m_width, total_elements_tiled);
    cudaCheckErrors("calculating next generation failed", __FILE__, __LINE__);
    swap_p(&d_tiled_table, &d_tiled_help);
  }
  cudaStreamSynchronize(0);
  // end timewatch
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf(ANSI_COLOR_RED"CUDA-"BIT ANSI_COLOR_RESET" time to run: "
      ANSI_COLOR_RED"%f"ANSI_COLOR_RESET" ms\n", time);

  cudaFree((void *) d_tiled_help);
  cudaCheckErrors("device freeing of help matrix failed", __FILE__, __LINE__);

  /******************************************************************************
   *                  Device finalization and Result printing                   *
   ******************************************************************************/

  // allocation again of a matrix that holds the normal representation
  cudaMalloc((void **) &d_table,  mem_size);
  cudaCheckErrors("device allocation of GOL matrix failed", __FILE__, __LINE__);

  // convert back to normal representation of the matrix
  convert_from_tiled <<< grid, block >>>(d_table, d_tiled_table,
                                         m_height, m_width, total_elements_tiled);
  cudaCheckErrors("failed to convert to normal repr from uint tiled repr", __FILE__, __LINE__);

  // transfer memory from resulting matrix from device to host
  cudaMemcpy(table, d_table, mem_size, cudaMemcpyDeviceToHost);
  cudaCheckErrors("copy from device to host memory failed", __FILE__, __LINE__);

  // reset gpu
  cudaDeviceReset();

  // save results to a file
  save_table(table, dim, dim, "cuda_2_results_"BIT".bin");
#ifdef PRINT
  print_table(table, dim, dim);
#endif  // PRINT
  free((void *) table);
}
#endif  // TESTING
