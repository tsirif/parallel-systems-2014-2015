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

int main()
{
    uint mem_size = 32 * sizeof(int);
    uint mem_size_tiled = sizeof(uint);

    dim3 grid(1, 1);
    dim3 block(1, 1);

    int *test_table;
    uint *test_table_tiled;
    uint *d_test_table_tiled;
    int *d_test_table;

    test_table = (int *) malloc(mem_size);

    for (int i = 0; i < 31; i++) test_table[i] = 1;
    test_table[31]=0;


    cudaMalloc((void **) &d_test_table,  mem_size);
    cudaCheckErrors("device allocation of GOL matrix failed", __FILE__, __LINE__);

    cudaMalloc((void **) &d_test_table_tiled, mem_size_tiled);
    cudaCheckErrors("device allocation of GOL uint tiled matrix failed", __FILE__, __LINE__);

    cudaMemcpy(d_test_table, test_table, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("copy from host to device memory failed", __FILE__, __LINE__);

    convert_to_tiled <<< grid, block >>>(d_test_table, d_test_table_tiled, 1, 1, 1);
    cudaCheckErrors("failed to convert normal repr to uint tiled repr", __FILE__, __LINE__);


    test_table_tiled = (uint *)malloc(mem_size_tiled);
    cudaMemcpy(test_table_tiled, d_test_table_tiled, mem_size_tiled, cudaMemcpyDeviceToHost);
    // ~ print_table_tiled(test_table_tiled, 1);
    printf("%u\n%x\n", test_table_tiled[0], test_table_tiled[0]);
}
