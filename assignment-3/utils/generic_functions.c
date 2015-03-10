#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define THRESHOLD 0.4
/* define colors */
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

void read_from_file(int *X, char *filename, int N)
{

    FILE *fp = fopen(filename, "r+");

    int size = fread(X, sizeof(int), N * N, fp);

    printf("elements: %d\n", size);

    fclose(fp);

}

void generate_table(int *X, int N)
{

    srand(time(NULL));
    int counter = 0;

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            X[i * N + j] = ( (float)rand() / (float)RAND_MAX ) < THRESHOLD;
            counter += X[i * N + j];
        }
    }


    printf("Number of non zerow elements: %d\n", counter);
    printf("Perncent: %f\n", (float)counter / (float)(N * N));
}

void save_table(int *X, int N, const char *filename)
{

    FILE *fp;

    printf("Saving table in file %s\n", filename);

    fp = fopen(filename, "w+");

    fwrite(X, sizeof(int), N * N, fp);

    fclose(fp);

}

void print_table(int* A, int N)
{
    for (int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            printf("%s%d "ANSI_COLOR_RESET, A[i * N + j] ? ANSI_COLOR_BLUE : ANSI_COLOR_RED, A[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}
