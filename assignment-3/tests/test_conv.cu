#define TESTING
#include "../cuda/cuda-int.cu"
#include <time.h>
#include <stdio.h>

#define mu_assert(message, test) do { if (!(test)) return message; } while (0)
#define mu_run_test(test) do { const char *message = test(); tests_run++; \
        if (message) return message; } while (0)

#define mu_run_test_wargs(test, ...) do { const char *message = test(__VA_ARGS__); tests_run++; \
        if (message) return message; } while (0)

int tests_run;

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

static void memory_allocations()
{
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

/* Allocates memory for an array of 32 elements. */
static void init_32()
{
    mem_size = 32 * sizeof(int);
    mem_size_tiled = sizeof(uint);

    grid = dim3(1, 1);
    block = dim3(1, 1);

    memory_allocations();
}

static void init_N()
{
    mem_size = 4 * 32 * sizeof(int);
    mem_size_tiled = 4 * sizeof(uint);

    grid = dim3(1, 1);
    block = dim3(2, 2);

    memory_allocations();
}

inline int zero_gen()
{
    return 0;
}

inline int one_gen()
{
    return 1;
}

inline int rand_gen()
{
    return ( (float)rand() / (float)RAND_MAX ) < 0.4;
}

/* Run test using a generator function. */
static const char *test_32_wgen(int (*generator)())
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

//TODO: merge test_32 with test_N ?
static const char *test_N_wgen(int (*generator)())
{
    uint expected_result[4] = {0};

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 32; i++) {
            int idx = i + j * 32;
            table[idx] = generator();
            expected_result[i/8] += table[idx] << (i % 8 + j * 8);
            printf("%d ", table[idx]);
        }
        printf("\n");
    }
    for(int i=0; i<4; i++) printf("%u ", expected_result[i]);
    printf("\n");

    cudaMemcpy(d_table, table, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("copy from host to device memory failed", __FILE__, __LINE__);

    convert_to_tiled <<< grid, block >>>(d_table, d_table_tiled, 4, 1, 4);
    cudaCheckErrors("failed to convert normal repr to uint tiled repr", __FILE__, __LINE__);

    cudaMemcpy(table_tiled, d_table_tiled, mem_size_tiled, cudaMemcpyDeviceToHost);
    cudaCheckErrors("memcpy to tiled", __FILE__, __LINE__);

    for (int i = 0; i < 4; i++) printf("%u\n", table_tiled[i]);

    for (int i = 0; i < 4; i++) {
        char *msg;
        size_t msg_max_size = 100 * sizeof(char);
        msg = (char *)malloc(msg_max_size);
        snprintf(msg, msg_max_size, "Unexpected tiled value! %u != %u at i=%d", table_tiled[i], expected_result[i], i);
        mu_assert(msg, table_tiled[i] == expected_result[i]);
        free(msg);
    }

    convert_from_tiled <<< grid, block >>>(d_table, d_table_tiled, 4, 1, 4);
    cudaCheckErrors("failed to convert to normal repr from uint tiled repr", __FILE__, __LINE__);

    cudaMemcpy(table_final, d_table, mem_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("memcpy to final", __FILE__, __LINE__);

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 32; i++) {
            int idx = i + j * 32;
            mu_assert("final and initial table differ", table_final[idx] == table[idx]);
        }
    }

    return 0;
}

/* Run all tests. Also times the execution. */
static const char *all_tests()
{
    float time;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    init_32();
    mu_run_test_wargs(test_32_wgen, one_gen);
    mu_run_test_wargs(test_32_wgen, zero_gen);
    mu_run_test_wargs(test_32_wgen, rand_gen);
    mu_run_test_wargs(test_32_wgen, rand_gen);
    mu_run_test_wargs(test_32_wgen, rand_gen);

    init_N();
    mu_run_test_wargs(test_N_wgen, one_gen);
    mu_run_test_wargs(test_N_wgen, zero_gen);
    mu_run_test_wargs(test_N_wgen, rand_gen);

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
