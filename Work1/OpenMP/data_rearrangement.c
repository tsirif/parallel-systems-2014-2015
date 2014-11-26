#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "lib.h"


void data_rearrangement(float *Y, float *X, 
			unsigned int *permutation_vector, 
			int N)
{
  int i=0;
  #pragma omp parallel shared(X, Y, permutation_vector) private(i)
  {
    #pragma omp for schedule(guided)
    for (i = 0; i < N; ++i)
    {
      memcpy(&Y[i*DIM], &X[permutation_vector[i]*DIM], DIM*sizeof(float));
    }
  }
  //~ int index;
  //~ #pragma omp parallel  shared(X,Y,permutation_vector)
  //~ #pragma private(index,i)
  //~ {
    //~ #pragma omp for schedule(guided) nowait
    //~ for(i=0; i<N; i++){
      //~ index = permutation_vector[i]*DIM ;
      //~ Y[i*DIM] = X[ index ];
      //~ Y[i*DIM + 1] = X[ index + 1 ];
      //~ Y[i*DIM + 2] = X[ index + 2 ];
    //~ }
  //~ }
}
