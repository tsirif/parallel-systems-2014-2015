#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("usage: %s [dimension]\n", argv[0]);
        exit(1);
    }

    int N = atoi(argv[1]);
    printf("Generating an %d x %d table\n", N, N);
    int *table = (int *)calloc(N, N * sizeof(int));
    generate_table(table, N);
    char filename[20];
    sprintf(filename, "table%dx%d.bin", N, N);
    save_table(table, N, filename);
    free(table);
    return 0;
}
