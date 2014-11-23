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
#ifdef CILK
  cilk_for(int i=0; i<N; i++){
#else
    for(int i=0; i<N; i++){
#endif
    memcpy(&Y[i*DIM], &X[permutation_vector[i]*DIM], DIM*sizeof(float));
  }

}
