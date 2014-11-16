#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "lib.h"



void data_rearrangement(float *Y, float *X, 
			unsigned int *permutation_vector, 
			int N){

  int i=0;
  omp_set_num_threads(THREADS);
  int threads = omp_get_num_threads();
  //~ int chunk = N  /  ( 4*DIM* threads ) ;
  //~ #pragma omp parallel for schedule(dynamic , chunk) shared(X,Y) private(i)
  #pragma omp parallel for schedule(guided) shared(X,Y) private(i)
  for(i=0; i<N; i++){
    memcpy(&Y[i*DIM], &X[permutation_vector[i]*DIM], DIM*sizeof(float));
  }

}
