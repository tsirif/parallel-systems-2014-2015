#define TESTING
#include "../cuda/cuda-int.c"

#define mu_assert(message, test) do { if (!(test)) return message; } while (0)
#define mu_run_test(test) do { const char *message = test(); tests_run++; \
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
int *table;
uint *table_tiled;
uint *d_table_tiled;
int *d_table;
int *table_final;

static void init()
{
    table = NULL;
    table_tiled = NULL;
    d_table_tiled = NULL;
    d_table = NULL;
    table_final = NULL;
}


static const char *test_zeros_to()
{

    mem_size = 32 * sizeof(int);
    mem_size_tiled = sizeof(uint);

    const dim3 grid(1, 1);
    const dim3 block(1, 1);

    table = (int *) realloc(table, mem_size);
    table_tiled = (uint *) realloc(table_tiled, mem_size_tiled);
    table_final = (int *) realloc(table_final, mem_size);

    for (int i = 0; i < 32; i++) table[i] = 0;

    cudaFree(d_table);
    cudaFree(d_table_tiled);

    cudaMalloc((void **) &d_table,  mem_size);
    cudaCheckErrors("device allocation of GOL matrix failed", __FILE__, __LINE__);

    cudaMalloc((void **) &d_table_tiled, mem_size_tiled);
    cudaCheckErrors("device allocation of GOL uint tiled matrix failed", __FILE__, __LINE__);

    cudaMemcpy(d_table, table, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("copy from host to device memory failed", __FILE__, __LINE__);

    convert_to_tiled <<< grid, block >>>(d_table, d_table_tiled, 1, 1, 1);
    cudaCheckErrors("failed to convert normal repr to uint tiled repr", __FILE__, __LINE__);

    cudaMemcpy(table_tiled, d_table_tiled, mem_size_tiled, cudaMemcpyDeviceToHost);
    cudaCheckErrors("memcpy to tiled", __FILE__, __LINE__);

    mu_assert("Unexpected tiled value!", table_tiled[0] == 0);

    convert_from_tiled <<< grid, block >>>(d_table, d_table_tiled, 1, 1, 1);
    cudaCheckErrors("failed to convert to normal repr from uint tiled repr", __FILE__, __LINE__);

    cudaMemcpy(table_final, d_table, mem_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("memcpy to final", __FILE__, __LINE__);

    int result = 1;

    for (int i = 0; i < 32; i++) result &= (table_final[i] == table[i]);

    mu_assert("Initial and final array differ!", result);

    return 0;
}

static const char *all_tests(){
    mu_run_test(test_zeros_to);
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


    // ~ srand(time(NULL));
    // ~ for (int i = 0; i < 32; i++) table[i] = ( (float)rand() / (float)RAND_MAX ) < 0.4;
    // ~ printf("%u\n%x\n", table_tiled[0], table_tiled[0]);
    // ~ for (int i = 0; i < 32; i++) printf("%d ", table_final[i]);
    // ~ printf("\n");

}
