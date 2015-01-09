#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#define CILK

#ifdef CILK
#include <cilk/cilk.h>
#endif

#define DIM 3


void data_rearrangement(float *Y, float *X, 
			unsigned int *permutation_vector, 
			int N){
  int i;
#ifdef CILK
  cilk_for(i=0; i<N; i++){
#else
  for(i=0; i<N; i++){
#endif
    memcpy(&Y[i*DIM], &X[permutation_vector[i]*DIM], DIM*sizeof(float));
  }

}
