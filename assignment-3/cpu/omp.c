#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include "../utils/utils.h"

#define NTHREADS 6

int count_neighbors(int x0, int x1, int x2, int y0, int y1, int y2);

/* table is where we store the actual data,
 * help_table is used for the calculation of a new generation */
int *table;
int *help_table;
int N;

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

int *prev_of;
int *next_of;

void pre_calc(void)
{
    prev_of = (int *) malloc(N * sizeof(int));
    next_of = (int *) malloc(N * sizeof(int));
    prev_of[0] = N - 1;
    next_of[N - 1] = 0;

    for (int i = 1; i < N; ++i) prev_of[i] = i - 1;

    for (int i = 0; i < N - 1; ++i) next_of[i] = i + 1;
}

void serial_compute(void)
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

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("usage: %s FILE dimension\n", argv[0]);
        exit(1);
    }

    int N_RUNS = 10;
    char *filename = argv[1];
    N = atoi(argv[2]);
    int total_size = N * N;

    if (argc == 4)
        N_RUNS = atoi(argv[3]);

#ifndef TEST
    printf("Reading %dx%d table from file %s\n", N, N, filename);
#endif  // TEST
    table = (int *) malloc(total_size * sizeof(int));
    help_table = (int *) malloc(total_size * sizeof(int));
    read_from_file(table, filename, N, N);
#ifndef TEST
    printf("Finished reading table\n");
#endif  // TEST
#ifdef PRINT
    print_table(table, N, N);
#endif  // PRINT
    struct timeval startwtime, endwtime;
    gettimeofday (&startwtime, NULL);
    pre_calc();

    for (int i = 0; i < N_RUNS; ++i) {
        serial_compute();
#ifdef PRINT
        print_table(table, N, N);
#endif  // PRINT
    }

    gettimeofday (&endwtime, NULL);
    double time = (double)((endwtime.tv_usec - startwtime.tv_usec)
                           / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf(ANSI_COLOR_RED"OMP"ANSI_COLOR_RESET" time to run: "
        ANSI_COLOR_RED"%f"ANSI_COLOR_RESET" ms\n", time * 1000);
    save_table(table, N, N, "omp-results.bin");
    free(table);
    free(help_table);
}
