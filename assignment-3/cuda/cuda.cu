#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include "../utils/utils.h"
//TODO: debloat

/* gets last cuda error and if it's not a cudaSuccess
 * prints debug information on stderr and aborts */
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

/* read a table from a file */
//TODO: move in generic functions file.
void read_from_file(int *X, char *filename, int N)
{
    FILE *fp = fopen(filename, "r+");
    int size = fread(X, sizeof(int), N * N, fp);
#ifdef TEST
    printf("total elements: %d\n", size);
#endif  // TEST
    fclose(fp);
}

//TODO: move in generic functions file.
void save_table(int *X, int N)
{
    FILE *fp;
    char filename[20];
    sprintf(filename, "results.bin");
#ifdef TEST
    printf("Saving table in file %s\n", filename);
#endif  // TEST
    fp = fopen(filename, "w+");
    fwrite(X, sizeof(int), N * N, fp);
    fclose(fp);
}

//TODO: move in generic functions file.
void pre_calc(int* prev_of, int* next_of, int N)
{
    prev_of[0] = N - 1;
    next_of[N - 1] = 0;
    for (int i = 1; i < N; ++i) prev_of[i] = i - 1;
    for (int i = 0; i < N - 1; ++i) next_of[i] = i + 1;
}

/* Determines the number of threads per block.
 * Returns a power of 2 number that evenly divides the total number of elements*/
int find_thread_count(const int dim)
{
    if (dim == 0) return 0;
    int result = 2;
    while ((dim % result == 0) && (result < 1024)) result *= 2;
    return result >> 1;
}

__global__ void cuda_compute(int *d_help, const int *d_table, const int *prev, const int *next, size_t N)
{
    const size_t cell_id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = cell_id / N;
    const size_t j = cell_id % N;

    const size_t left = prev[i];
    const size_t right = next[i];
    const size_t up = prev[j];
    const size_t down = next[j];

    const int alive_neighbors = d_table[POS(left , up  )] +
                                d_table[POS(left , j   )] +
                                d_table[POS(left , down)] +
                                d_table[POS(i    , up  )] +
                                d_table[POS(i    , down)] +
                                d_table[POS(right, up  )] +
                                d_table[POS(right, j   )] +
                                d_table[POS(right, down)] ;
    if (cell_id < N * N)
        d_help[cell_id] = (alive_neighbors == 3) || (alive_neighbors == 2 && d_table[cell_id]) ? 1 : 0;
}

//TODO: move in generic functions file.
void print_table(int* A, size_t N)
{
    for (int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            printf("%s%d "ANSI_COLOR_RESET, A[i * N + j] ? ANSI_COLOR_BLUE : ANSI_COLOR_RED, A[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("usage: %s FILE dimension\n", argv[0]);
        exit(1);
    }

    int n_runs;
    if (argc == 4) n_runs = atoi(argv[3]);
    else n_runs = DFL_RUNS;

    const size_t N = atoi(argv[2]);
    const size_t total_elements = N * N;
    const size_t mem_size = total_elements * sizeof(int);
    const size_t row_mem_size = N * sizeof(int);

    char* filename = argv[1];
    int *table;
    printf("Reading %zux%zu table from file %s\n", N, N, filename);
    table = (int*) malloc(mem_size);
    read_from_file(table, filename, N);
    printf("Finished reading table\n");

#ifdef PRINT
    print_table(table, N);
#endif

    const unsigned int thread_count = find_thread_count(total_elements);
    const unsigned int blocks_count = total_elements / thread_count;

    int *prev_of;
    int *next_of;
    prev_of = (int*) malloc(N * sizeof(size_t));
    next_of = (int*) malloc(N * sizeof(size_t));
    pre_calc(prev_of, next_of, N);

    int *d_help, *d_table, *prev, *next;
    cudaMalloc((void **) &d_help,  mem_size);
    cudaCheckErrors("malloc fail");

    cudaMalloc((void **) &d_table, mem_size);
    cudaCheckErrors("malloc fail");

    cudaMalloc((void **) &prev, row_mem_size);
    cudaCheckErrors("malloc fail");

    cudaMalloc((void **) &next, row_mem_size);
    cudaCheckErrors("malloc fail");

    cudaMemcpy(d_table, table, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("memcpy fail");

    cudaMemcpy(prev, prev_of, row_mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("memcpy fail");

    cudaMemcpy(next, next_of, row_mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("memcpy fail");
    
    free(prev_of);
    free(next_of);

    struct timeval startwtime, endwtime;
    gettimeofday (&startwtime, NULL);

    for (int i = 0; i < n_runs; ++i) {
        cuda_compute <<< blocks_count, thread_count >>>(d_help, d_table, prev, next, N);
        cudaCheckErrors("compute fail");
        swap(&d_table, &d_help);

#ifdef PRINT
        cudaMemcpy(table, d_table, mem_size, cudaMemcpyDeviceToHost);
        print_table(table, N);
#endif
    }

    gettimeofday (&endwtime, NULL);
    double exec_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
                                / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("clock: %fs\n", exec_time);

    cudaMemcpy(table, d_table, total_elements * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceReset();
    save_table(table, N);
}
