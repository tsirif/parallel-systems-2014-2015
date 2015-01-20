#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

#define NTHREADS 6

void swap(int** a, int** b);
int count_neighbors(size_t x0, size_t x1, size_t x2, size_t y0, size_t y1, size_t y2);

/* define colors */
//TODO: move them somewhere better
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

/* table is where we store the actual data,
 * help_table is used for the calculation of a new generation */
int *table;
int *help_table;
size_t N;

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
    printf("total elements: %d\n", size);
    fclose(fp);
}

#define POS(i, j) (i*N + j)

inline int count_neighbors(size_t left, size_t owni, size_t right, size_t up, size_t ownj, size_t down)
{
    return 
    table[POS(left , up  )] + 
    table[POS(left , ownj)] + 
    table[POS(left , down)] + 
    table[POS(owni , up  )] + 
    table[POS(owni , down)] + 
    table[POS(right, up  )] + 
    table[POS(right, ownj)] + 
    table[POS(right, down)] ; 
}

//TODO: IDEA: precalculate y0,1,2 and x0,2
void serial_compute()
{
    size_t i,j,left,right,up,down;
    unsigned int alive_neighbors;
    #pragma omp parallel for private(left, right, up, down, alive_neighbors, j)
    for (i = 0; i < N; ++i) {

        left = (i != 0) ? i - 1 : N - 1;
        right = (i != N - 1) ? i + 1 : 0;

        for (j = 0; j < N; ++j) {

            up = (j != 0) ? j - 1 : N - 1;
            down = (j != N - 1) ? j + 1 : 0;

            alive_neighbors = count_neighbors(left, i, right, up, j, down);
            help_table[POS(i, j)] = (alive_neighbors == 3) || (alive_neighbors == 2 && table[POS(i, j)]) ? 1 : 0;
        }
    }
    swap(&table, &help_table);
}

void print_table(int* table)
{
    for (size_t i = 0; i < N; ++i) {
        for(size_t j = 0; j < N; ++j) {
            printf("%s%d "ANSI_COLOR_RESET, table[i * N + j] ? ANSI_COLOR_BLUE : ANSI_COLOR_RED, table[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

#define N_RUNS 10

int main(int argc, char **argv)
{

    if (argc != 3) {
        printf("usage: %s FILE dimension\n", argv[0]);
        exit(1);
    }

    char *filename = argv[1];
    N = atoi(argv[2]);
    size_t total_size = N * N;

    printf("Reading %lux%lu table from file %s\n", N, N, filename);
    table = malloc(total_size * sizeof(int));
    help_table = malloc(total_size * sizeof(int));
    read_from_file(table, filename, N);
    printf("Finished reading table\n");

    struct timeval startwtime, endwtime;
    gettimeofday (&startwtime, NULL); 
    for (int i = 0; i < N_RUNS; ++i) {
        memcpy(help_table, table, total_size);
        serial_compute();
        //~ print_table(table);
    }
    gettimeofday (&endwtime, NULL);
    double hash_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
                /1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("clock: %fs\n", hash_time);

    free(table);
    free(help_table);
}
