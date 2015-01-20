#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define THRESHOLD 0.4

void generate_table(int *X, int N){

  srand(time(NULL));
  int counter = 0;

  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      X[i*N + j] = ( (float)rand() / (float)RAND_MAX ) < THRESHOLD; 
      counter += X[i*N + j];
    }
  }


  printf("Number of non zerow elements: %d\n", counter);
  printf("Perncent: %f\n", (float)counter / (float)(N*N));
}


void save_table(int *X, int N){

  FILE *fp;

  char filename[20];

  sprintf(filename, "table%dx%d.bin", N, N);

  printf("Saving table in file %s\n", filename);

  fp = fopen(filename, "w+");

  fwrite(X, sizeof(int), N*N, fp);

  fclose(fp);

}

int main(int argc, char **argv){

  if (argc != 2){
    printf("usage: %s [dimension]\n", argv[0]);
    exit(1);
  }
  
  int N = atoi(argv[1]);

  printf("Generating an %d x %d table\n", N, N);

  int *table = (int *)calloc(N,N*sizeof(int));

  generate_table(table, N);

  save_table(table, N);

  free(table);
  
  return 0;

}
