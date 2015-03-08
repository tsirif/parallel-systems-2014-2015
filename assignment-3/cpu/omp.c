#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

#define NTHREADS 6

void swap(int** a, int** b);
int count_neighbors(int x0, int x1, int x2, int y0, int y1, int y2);

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
int N;

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
#ifndef TEST
    printf("total elements: %d\n", size);
#endif  // TEST
    fclose(fp);
}

void save_table(int *X, int N)
{
    FILE *fp;
    char filename[20];
    sprintf(filename, "omp-results.bin");
#ifndef TEST
    printf("Saving table in file %s\n", filename);
#endif  // TEST
    fp = fopen(filename, "w+");
    fwrite(X, sizeof(int), N * N, fp);
    fclose(fp);
}

#define POS(i, j) (i*N + j)

inline int count_neighbors(int up, int owni, int down, int left, int ownj, int right)
{
    return
        table[POS(up   , left)] +
        table[POS(up   , ownj)] +
        table[POS(up   , right)] +
        table[POS(owni , left  )] +
        table[POS(owni , right)] +
        table[POS(down, left  )] +
        table[POS(down, ownj)] +
        table[POS(down, right)] ;
}

int* prev_of;
int* next_of;

void pre_calc()
{
    prev_of = (int*) malloc(N * sizeof(int));
    next_of = (int*) malloc(N * sizeof(int));

    prev_of[0] = N - 1;
    next_of[N - 1] = 0;
    for (int i = 1; i < N; ++i) prev_of[i] = i - 1;
    for (int i = 0; i < N - 1; ++i) next_of[i] = i + 1;
}

void serial_compute()
{
    int i, j, left, right, up, down;
    unsigned int alive_neighbors;
    #pragma omp parallel for private(left, right, up, down, alive_neighbors, j)
    for (i = 0; i < N; ++i) {

        up = prev_of[i];
        down = next_of[i];

        for (j = 0; j < N; ++j) {

            left = prev_of[j];
            right = next_of[j];

            //~ printf("(i=%lu, j=%lu) left=%lu right=%lu up=%lu down=%lu\n", i, j, left, right, up, down);

            alive_neighbors = count_neighbors(up, i, down, left, j, right);
            help_table[POS(i, j)] = (alive_neighbors == 3) || (alive_neighbors == 2 && table[POS(i, j)]) ? 1 : 0;
        }
    }
    swap(&table, &help_table);
}

void print_table(int* table)
{
    for (int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            printf("%s%d "ANSI_COLOR_RESET, table[i * N + j] ? ANSI_COLOR_BLUE : ANSI_COLOR_RED, table[i * N + j]);
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

    int N_RUNS = 10;
    char* filename = argv[1];
    N = atoi(argv[2]);
    int total_size = N * N;
    if (argc == 4) {
        N_RUNS = atoi(argv[3]);
    }

#ifndef TEST
    printf("Reading %dx%d table from file %s\n", N, N, filename);
#endif  // TEST

    table = (int*) malloc(total_size * sizeof(int));
    help_table = (int*) malloc(total_size * sizeof(int));
    read_from_file(table, filename, N);
    
#ifndef TEST
    printf("Finished reading table\n");
#endif  // TEST

#ifdef PRINT
    print_table(table);
#endif  // PRINT

    struct timeval startwtime, endwtime;
    gettimeofday (&startwtime, NULL);
    pre_calc();
    for (int i = 0; i < N_RUNS; ++i) {
        serial_compute();
        
#ifdef PRINT
        print_table(table);
#endif  // PRINT
    }
    gettimeofday (&endwtime, NULL);
    double hash_time = (double)((endwtime.tv_usec - startwtime.tv_usec)
                                / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("clock: %fs\n", hash_time);

    save_table(table, N);

    free(table);
    free(help_table);
}
