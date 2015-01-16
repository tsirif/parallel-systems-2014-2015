#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void read_from_file(int *X, char *filename, int N){

  FILE *fp = fopen(filename, "r+");

  int size = fread(X, sizeof(int), N*N, fp);

  printf("elements: %d\n", size);

  fclose(fp);

}

int main(int argc, char **argv){


  char *filename = argv[1];

  int N = atoi(argv[2]);

  printf("Reading %dx%d table from file %s\n", N, N, filename);

  int *table = (int *)malloc(N*N*sizeof(int));
  
  read_from_file(table, filename, N);

  free(table);

}
