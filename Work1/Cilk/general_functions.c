#include <stdio.h>
#include <stdlib.h>
#include <float.h>

/*#define CILK*/

#ifdef CILK
#include <cilk/cilk.h>
#endif

#define DIM 3

void find_max(float *max_out,float *X, int N){


  int i;
  for(i=0; i<DIM; i++){
    max_out[i] = -FLT_MAX;
    int j;
    for(j=0; j<N; j++){
      if(max_out[i]<X[j*DIM + i]){
	max_out[i] = X[j*DIM + i];
      }
    }
  }

}

void find_min(float *min_out, float *X, int N){

  int i;
  for(i=0; i<DIM; i++){ 
    min_out[i] = FLT_MAX;
    int j;
    for(j=0; j<N; j++){
      if(min_out[i]>X[j*DIM + i]){
	min_out[i] = X[j*DIM + i];
      }
    }
  }

}
