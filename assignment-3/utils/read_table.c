#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

int main(int argc, char **argv)
{
    char *filename = argv[1];
    int N = atoi(argv[2]);
    printf("Reading %dx%d table from file %s\n", N, N, filename);
    int *table = (int *)malloc(N * N * sizeof(int));
    read_from_file(table, filename, N, N);
    free(table);
}
