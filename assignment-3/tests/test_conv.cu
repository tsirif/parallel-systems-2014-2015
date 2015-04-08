#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef uint32_t uint;

#define CONF_WIDTH 8
#define CONF_HEIGHT 4

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

__global__ void convert_from_tiled(
    int *d_table, uint const *d_utable,
    uint m_width, uint m_height, uint m_size)
{
    int row = (__mul24(blockIdx.x, blockDim.x) + threadIdx.x) * m_width;
    int col = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
    int start_i = row * CONF_WIDTH * CONF_HEIGHT;
    int start_j = col * CONF_WIDTH;
    int place = 0;

    const uint tile = d_utable[col + row];

    const int step_i = m_width * CONF_WIDTH;
    const int end_i = start_i + CONF_HEIGHT * step_i;
    const int end_j = start_j + CONF_WIDTH;
    int i, j;

    for (i = start_i; i < end_i; i += step_i) {
        for (j = start_j; j < end_j; ++j)
            d_table[j + i] = (int) (tile >> place++ & 1u);
    }
}

void print_table_tiled(uint *A, int N)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            printf("%u ", A[i * N + j]);

        printf("\n");
    }

    printf("\n");
}

void print_table(int *A, int N)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            printf("%d ", A[i * N + j]);

        printf("\n");
    }

    printf("\n");
}


int main()
{
    const uint mem_size = 32 * sizeof(int);
    const uint mem_size_tiled = sizeof(uint);

    const dim3 grid(1, 1);
    const dim3 block(1, 1);

    int *table;
    uint *table_tiled;
    uint *d_table_tiled;
    int *d_table;

    table = (int *) malloc(mem_size);

    srand(time(NULL));

    for (int i = 0; i < 32; i++) table[i] = ( (float)rand() / (float)RAND_MAX ) < 0.4;

    cudaMalloc((void **) &d_table,  mem_size);
    cudaCheckErrors("device allocation of GOL matrix failed", __FILE__, __LINE__);

    cudaMalloc((void **) &d_table_tiled, mem_size_tiled);
    cudaCheckErrors("device allocation of GOL uint tiled matrix failed", __FILE__, __LINE__);

    cudaMemcpy(d_table, table, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("copy from host to device memory failed", __FILE__, __LINE__);

    convert_to_tiled <<< grid, block >>>(d_table, d_table_tiled, 1, 1, 1);
    cudaCheckErrors("failed to convert normal repr to uint tiled repr", __FILE__, __LINE__);


    table_tiled = (uint *)malloc(mem_size_tiled);
    cudaMemcpy(table_tiled, d_table_tiled, mem_size_tiled, cudaMemcpyDeviceToHost);
    cudaCheckErrors("memcpy1", __FILE__, __LINE__);
    // ~ print_table_tiled(table_tiled, 1);
    printf("%u\n%x\n", table_tiled[0], table_tiled[0]);

    convert_from_tiled <<< grid, block >>>(d_table, d_table_tiled, 1, 1, 1);
    cudaCheckErrors("failed to convert to normal repr from uint tiled repr", __FILE__, __LINE__);

    int *table_final;
    table_final = (int *) malloc(mem_size);

    cudaMemcpy(table_final, d_table, mem_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("memcpy1", __FILE__, __LINE__);

    for (int i = 0; i < 32; i++) printf("%d ", table_final[i]);

    printf("\n");

    int result = 1;

    for (int i = 0; i < 32; i++) result &= (table_final[i] == table[i]);

    printf("equality result: %d\n", result);
}
