#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "float.h"
#include "lib.h"


unsigned int compute_code(float x, float low, float step){

  return floor((x - low) / step);

}


/* Function that does the quantization */
void quantize(unsigned int *codes, float *X, float *low, float step, int N){


omp_set_num_threads(THREADS);
int threads = omp_get_num_threads();
//~ int chunk = N  /  (2*omp_get_num_threads() ) ;
// Split the data into chunks so that every thread has at least 2 
// chunks to work on and thus we can better balance the data load.
int chunk = N  /  ( 2*DIM* threads ) ;
if (chunk < 1 )
  chunk = 1 ;
int i = 0, j = 0; 

// Index of element to be accessed.
int index;

#pragma omp parallel shared(codes , X , low , step  ) private(i,j,index)
  {
    #pragma omp for schedule( dynamic , chunk ) private(i,j,index) nowait
    //~ #pragma omp for schedule( guided ) private(i,j,index) nowait
    // Increase the step to 2 for every iteration so as to pipeline
    // more instructions .
    for(i=0; i<N; i ++ )
    {
      for(j=0; j<DIM; j++)
      {
        //~ index = 2*i*DIM + j ;
        index = i*DIM + j ;
        codes[index] = compute_code( X[index], low[j], step);
        //~ index2 = i*DIM + j + DIM ;
        //~ codes[index2] = compute_code( X[index2], low[j], step);
      }
    }
  }


}

float max_range(float *x){

  int i=0;
  float max = -FLT_MAX;
  for(i=0; i<DIM; i++){
    if(max<x[i]){
      max = x[i];
    }
  }

  return max;

}

void compute_hash_codes(unsigned int *codes, float *X, int N, 
			int nbins, float *min, 
			float *max){
  
  float range[DIM];
  float qstep;

  int i = 0;
  #pragma omp parallel shared(range  , max  , min   ) private(i)
  {
    #pragma omp for schedule( dynamic  ) nowait
    for(i=0; i<DIM; i++)
    {
      range[i] = fabs(max[i] - min[i]); // The range of the data
      range[i] += 0.01*range[i]; // Add somthing small to avoid having points exactly at the boundaries 
    }
  }
  
  qstep = max_range(range) / nbins; // The quantization step 
  
  quantize(codes, X, min, qstep, N); // Function that does the quantization

}



