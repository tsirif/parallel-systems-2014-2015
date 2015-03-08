#include <stdio.h>
#include <stdlib.h>



void read_from_file(int* X, const char* filename, int N)
{
    FILE *fp = fopen(filename, "r+");
    fread(X, sizeof(int), N * N, fp);
    fclose(fp);
}

void save_table(const int *X, const char* filename, int N)
{
    FILE *fp;
    fp = fopen(filename, "w+");
    fwrite(X, sizeof(int), N * N, fp);
    fclose(fp);
}

#define N 6
#define TOT 36
const int block_formation[TOT] = {
    0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 0, 0,
    0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0
};

const int blinker_formation[TOT] = {
    0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0,
    0, 0, 1, 0, 0, 0,
    0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0
};

const int blinker_and_block_formation[4 * TOT] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

const int four_in_a_row[TOT] = {
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0
};

int main()
{
    save_table(block_formation, "block.bin", N);
    save_table(blinker_formation, "blinker.bin", N);
    save_table(four_in_a_row, "4row.bin", N);
    save_table(blinker_and_block_formation, "blinker_block.bin", 2*N);
}
