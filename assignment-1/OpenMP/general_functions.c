#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "lib.h"

void parallel_max(float *max_out , float * X, int N , int col )
{
  int j ;
  *max_out = -FLT_MAX ;
  int index ;
  #pragma omp parallel shared(X,max_out,N) private(j)
  {
    float max = -FLT_MAX ;
    #pragma omp for schedule(guided) nowait
    for(j=0; j<N; j++)
    {
      index = j*DIM + col ;
      if(max < X[index])
      {
        max = X[index];
      }
    }
    // Lock the comparison to avoid race conditions.
    #pragma omp critical
    {
      if (max > *max_out )
        *max_out = max ;
    }
    
  }
  
}

void find_max(float *max_out, float *X, int N){

  int i = 0, j = 0;
  float parallelMax[DIM] = {-FLT_MAX};
  
  for (i = 0 ; i < DIM ; i++)
  {
    //~ parallel_max(&parallelMax[i] , X , N , i);
    parallel_max(&max_out[i] , X , N , i);
  }

  
  //~ for (i = 0 ; i<DIM ; i++ )
    //~ printf("Parallel max #%d %.30f \n",i,parallelMax[i]);
    //~ printf("Parallel max #%d %.30f \n",i,max_out[i]);

  // Test
  
  //~ 
  //~ for(i=0; i<DIM; i++){ 
    //~ max_out[i] = -FLT_MAX;
    //~ for(j=0; j<N; j++){
      //~ if(max_out[i]<X[j*DIM + i]){
	//~ max_out[i] = X[j*DIM + i];
      //~ }
    //~ }
  //~ }
  //~ 
  //~ for (i = 0 ; i<DIM ; i++ )
    //~ printf("Serial max #%d %.30f \n",i,max_out[i]);
    //~ 
  //~ for (i = 0 ; i<DIM ; i++ )
  //~ {
    //~ if ( max_out[i] != parallelMax[i] )
    //~ {
      //~ printf("error at %i \n",i);
      //~ printf("true value is %.30f \n",max_out[i]);
      //~ printf("value found %.30f \n",parallelMax[i]);
    //~ }
  //~ } 
}

void parallel_min(float *min_out , float * X, int N , int col )
{
  int j ;
  *min_out = FLT_MAX ;
  int index ;
  #pragma omp parallel shared(X,min_out,N) private(j)
  {
    float min = FLT_MAX ;
    #pragma omp for schedule(guided) nowait
    for(j=0; j<N; j++)
    {
      index = j*DIM + col ;
      if(min > X[index])
      {
        min = X[index];
      }
    }
    // Lock the comparison to avoid race conditions.
    #pragma omp critical
    {
      if (min < *min_out )
        *min_out = min ;
    }
    
  }
  
}

void find_min(float *min_out, float *X, int N){

  int i = 0, j = 0;
  float parallelMin[DIM] = {FLT_MAX};
  
  for (i = 0 ; i < DIM ; i++)
  {
    //~ parallel_min(&parallelMin[i] , X , N , i);
    parallel_min(&min_out[i] , X , N , i);
  }

  
  //~ for (i = 0 ; i<DIM ; i++ )
    //~ printf("Parallel min #%d %.30f \n",i,parallelMin[i]);
    //~ printf("Parallel min #%d %.30f \n",i,min_out[i]);

  // Test
  
  //~ 
  //~ for(i=0; i<DIM; i++){ 
    //~ min_out[i] = FLT_MAX;
    //~ for(j=0; j<N; j++){
      //~ if(min_out[i]>X[j*DIM + i]){
	//~ min_out[i] = X[j*DIM + i];
      //~ }
    //~ }
  //~ }
  //~ 
  //~ for (i = 0 ; i<DIM ; i++ )
    //~ printf("Serial min #%d %.30f \n",i,min_out[i]);
    //~ 
  //~ for (i = 0 ; i<DIM ; i++ )
  //~ {
    //~ if ( min_out[i] != parallelMin[i] )
    //~ {
      //~ printf("error at %i \n",i);
      //~ printf("true value is %.30f \n",min_out[i]);
      //~ printf("value found %.30f \n",parallelMin[i]);
    //~ }
  //~ } 
}
