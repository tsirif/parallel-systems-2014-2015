#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "lib.h"



void data_rearrangement(float *Y, float *X, 
			unsigned int *permutation_vector, 
			int N){

  int i=0;
  for(i=0; i<N; i++){
    memcpy(&Y[i*DIM], &X[permutation_vector[i]*DIM], DIM*sizeof(float));
  }

}
