#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
//TODO: debloat

void swap(int** a, int** b);

/* define colors */
//TODO: move them somewhere better
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

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


/* table is where we store the actual data,
 * help_table is used for the calculation of a new generation */
int *table;
int *help_table;
int N;

#define DFL_RUNS 10

#define BLOCK_SIZE 4
#define N_BLOCKS N/BLOCK_SIZE + N%BLOCK_SIZE

/* swap 2 int* pointers */
//TODO: move somewhere better, make it a #define(?)
inline void swap(int** a, int** b)
{
    int *t;
    t = *a;
    *a = *b;
    *b = t;
}

/* read a table from a file */
//TODO: move it somewhere better, seperate file
void read_from_file(int *X, char *filename, int N)
{
    FILE *fp = fopen(filename, "r+");
    int size = fread(X, sizeof(int), N * N, fp);
#ifdef TEST
    printf("total elements: %d\n", size);
#endif  // TEST
    fclose(fp);
}

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

#define POS(i, j) (i*N + j)
#define cPOS(i, j) (i*Nc + j)

int* prev_of;
int* next_of;

void pre_calc()
{
    prev_of = (int*) malloc(N * sizeof(size_t));
    next_of = (int*) malloc(N * sizeof(size_t));

    prev_of[0] = N - 1;
    next_of[N - 1] = 0;
    for (int i = 1; i < N; ++i) prev_of[i] = i - 1;
    for (int i = 0; i < N - 1; ++i) next_of[i] = i + 1;
}

__global__ void cuda_compute(int *d_help, const int *d_table, const int *prev, const int *next, const int Nc)
{
    //const int i = blockIdx.x * blockDim.x + threadIdx.x;
    //const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = blockIdx.x;
	const int j = blockIdx.y;

    if (i<Nc && j<Nc)
    {
        const int left = prev[i];
        const int right = next[i];
    
        const int up = prev[j];
        const int down = next[j];
    
        const int alive_neighbors = d_table[cPOS(left , up  )] +
                                    d_table[cPOS(left , j   )] +
                                    d_table[cPOS(left , down)] +
                                    d_table[cPOS(i    , up  )] +
                                    d_table[cPOS(i    , down)] +
                                    d_table[cPOS(right, up  )] +
                                    d_table[cPOS(right, j   )] +
                                    d_table[cPOS(right, down)] ;
        const int idx = cPOS(i, j);
        if (idx < Nc * Nc)
            d_help[idx] = (alive_neighbors == 3) || (alive_neighbors == 2 && d_table[idx]) ? 1 : 0;
    }
}

__global__ void cuda_copy(const int *d_help, int *d_table, const int tot, const int Nc)
{
    //const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idx = blockIdx.x * Nc + blockIdx.y;

    if (idx < tot)
        d_table[idx] = d_help[idx];

}

void print_table(int* A)
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

    int N_RUNS = DFL_RUNS;
    char* filename = argv[1];
    N = atoi(argv[2]);
    const int total_size = N * N;
    const int mem_size = total_size * sizeof(int);
    const int row_mem_size = N * sizeof(int);
    if (argc == 4) {
        N_RUNS = atoi(argv[3]);
    }

#ifdef TEST
    printf("Reading %dx%d table from file %s\n", N, N, filename);
#endif  // TEST
    table = (int*) malloc(mem_size);
    help_table = (int*) malloc(mem_size);
    read_from_file(table, filename, N);
#ifdef TEST
    printf("Finished reading table\n");
    print_table(table);
#endif  // TEST

    int *d_help, *d_table, *prev, *next;

    dim3 threadsPerBlock(N,N);
    dim3 numBlocks(1, 1);
    dim3 grid(N,N);

    cudaMalloc((void **) &d_help,  mem_size);
    cudaCheckErrors("malloc fail");

    cudaMalloc((void **) &d_table, mem_size);
    cudaCheckErrors("malloc fail");

    cudaMalloc((void **) &prev, row_mem_size);
    cudaCheckErrors("malloc fail");

    cudaMalloc((void **) &next, row_mem_size);
    cudaCheckErrors("malloc fail");

    struct timeval startwtime, endwtime;
    gettimeofday (&startwtime, NULL);

    pre_calc();

    cudaMemcpy(d_help, help_table, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("memcpy fail");

    cudaMemcpy(d_table, table, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("memcpy fail");

    cudaMemcpy(prev, prev_of, row_mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("memcpy fail");

    cudaMemcpy(next, next_of, row_mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("memcpy fail");

    for (int i = 0; i < N_RUNS; ++i) {
        cuda_compute <<< grid, 1 >>>(d_help, d_table, prev, next, N);
        cudaCheckErrors("compute fail");
        cuda_copy <<< grid, 1>>> (d_help, d_table, total_size, N);
        cudaCheckErrors("memcpy fail");
        cudaMemcpy(table, d_table, mem_size, cudaMemcpyDeviceToHost);
        cudaCheckErrors("memcpy fail");
        print_table(table);
    }

    gettimeofday (&endwtime, NULL);
    double exec_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
                                / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("clock: %fs\n", exec_time);

    cudaMemcpy(table, d_table, total_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceReset();
    save_table(table, N);

    free(table);
    free(help_table);
}
