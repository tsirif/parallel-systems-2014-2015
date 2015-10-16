#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "lib.h"

void find_max(float *max_out,float *X, int N){


  int i = 0, j = 0;
  // For every Dimension
  for(i=0; i<DIM; i++){
    // Initialize the max value to be the minimum possible.
    max_out[i] = -FLT_MAX;
    // For every point in the dataset
    for(j=0; j<N; j++){
      if(max_out[i]<X[j*DIM + i]){
	max_out[i] = X[j*DIM + i];
      }
    }
  }

}

void find_min(float *min_out, float *X, int N){

  int i = 0, j = 0;
  for(i=0; i<DIM; i++){ 
    min_out[i] = FLT_MAX;
    for(j=0; j<N; j++){
      if(min_out[i]>X[j*DIM + i]){
	min_out[i] = X[j*DIM + i];
      }
    }
  }

}
