#define TESTING
#include "../cuda/cuda-int.c"
#include <time.h>

#define mu_assert(message, test) do { if (!(test)) return message; } while (0)
#define mu_run_test(test) do { const char *message = test(); tests_run++; \
        if (message) return message; } while (0)

#define mu_run_test_wargs(test, ...) do { const char *message = test(__VA_ARGS__); tests_run++; \
        if (message) return message; } while (0)

int tests_run;

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

uint mem_size;
uint mem_size_tiled;

/* Host pointers */
int *table;
uint *table_tiled;
int *table_final;

/* Cuda device pointers */
uint *d_table_tiled;
int *d_table;

/* grid and block dimensions */
dim3 grid;
dim3 block;

/* Initializes global variables.*/
static void init()
{
    table = NULL;
    table_tiled = NULL;
    d_table_tiled = NULL;
    d_table = NULL;
    table_final = NULL;
    srand(time(NULL));
}

/* Allocates size for an array of 32 elements. */
static void init_32()
{
    mem_size = 32 * sizeof(int);
    mem_size_tiled = sizeof(uint);

    grid = dim3(1, 1);
    block = dim3(1, 1);

    table = (int *) realloc(table, mem_size);
    table_tiled = (uint *) realloc(table_tiled, mem_size_tiled);
    table_final = (int *) realloc(table_final, mem_size);

    /* Free cuda pointers just in case. */
    cudaFree(d_table);
    cudaFree(d_table_tiled);

    cudaMalloc((void **) &d_table,  mem_size);
    cudaCheckErrors("device allocation of GOL matrix failed", __FILE__, __LINE__);

    cudaMalloc((void **) &d_table_tiled, mem_size_tiled);
    cudaCheckErrors("device allocation of GOL uint tiled matrix failed", __FILE__, __LINE__);
}

inline int zero_gen(){
    return 0;
}

inline int one_gen(){
    return 1;
}

inline int rand_gen(){
    return ( (float)rand() / (float)RAND_MAX ) < 0.4;
}

/* Run test using a generator function. */
static const char *test_wgen(int (*generator)())
{

    uint expected_result = 0;

    for (int i = 0; i < 32; i++) {
        table[i] = generator();
        expected_result += table[i] << i;
    }

    cudaMemcpy(d_table, table, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("copy from host to device memory failed", __FILE__, __LINE__);

    convert_to_tiled <<< grid, block >>>(d_table, d_table_tiled, 1, 1, 1);
    cudaCheckErrors("failed to convert normal repr to uint tiled repr", __FILE__, __LINE__);

    cudaMemcpy(table_tiled, d_table_tiled, mem_size_tiled, cudaMemcpyDeviceToHost);
    cudaCheckErrors("memcpy to tiled", __FILE__, __LINE__);

    mu_assert("Unexpected tiled value!", table_tiled[0] == expected_result);

    convert_from_tiled <<< grid, block >>>(d_table, d_table_tiled, 1, 1, 1);
    cudaCheckErrors("failed to convert to normal repr from uint tiled repr", __FILE__, __LINE__);

    cudaMemcpy(table_final, d_table, mem_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("memcpy to final", __FILE__, __LINE__);

    int result = 1;
    for (int i = 0; i < 32; i++) result &= (table_final[i] == table[i]);

    mu_assert("Initial and final array differ!", result);

    return 0;
}

/* Run all tests. Also times the execution. */
static const char *all_tests()
{
    init_32();

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    mu_run_test_wargs(test_wgen, one_gen);
    mu_run_test_wargs(test_wgen, zero_gen);
    mu_run_test_wargs(test_wgen, rand_gen);
    mu_run_test_wargs(test_wgen, rand_gen);
    mu_run_test_wargs(test_wgen, rand_gen);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("CUDA tests time to run:  %f s \n", time / 1000);

    return 0;
}

int main()
{
    init();
    const char *result = all_tests();

    if (result != 0)
        printf("%s\n", result);
    else
        printf("ALL TESTS PASSED\n");

    printf("Tests run: %d\n", tests_run);

    return result != 0;
}
