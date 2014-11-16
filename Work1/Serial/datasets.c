/*

This file contains the different distributions used for testing the octree construction

author: Nikos Sismanis
date: Oct 2014

*/

#include "stdio.h" 
#include "stdlib.h"
#include <math.h>

#define DIM 3 // Dimension of the space 
#define PI 3.1415927 



void cube(float *X, int N){

  int i, j;
  for(i=0; i<N; i++){
    for(j=0; j<DIM; j++){
      X[i*DIM + j] = (float)rand() / (float) RAND_MAX;
    }
  }
  
}

void plummer(float *X, int N){

  srand(time(NULL));
  int i = 0;
  for(i=0; i<N; i++){
    float X1 = (float)rand() / (float) RAND_MAX;;
    float X2 = (float)rand() / (float) RAND_MAX;;
    float X3 = (float)rand() / (float) RAND_MAX;;
    float R =  1.0 / sqrt( (pow(X1, -2.0 / 3.0) - 1.0) );
    if(R<100.0){
      float x1 = (1.0 - 2.0 * X2) * R;
      float x2 = sqrt(R * R - X1 * X1) * cos(2.0 * M_PI * X3);
      float x3 = sqrt(R * R - X1 * X1) * sin(2.0 * M_PI * X3);
      float scale = 3.0 * M_PI / 16.0;
      x1 *= scale; x2 *= scale; x3 *= scale;
      X[i*DIM] = x1;
      X[i*DIM + 1] = x2;
      X[i*DIM + 2] = x3;
    }
  }

}

/* Function that creates a dataset for testing 0 (spherical octant) 
1 (uniform cude) */
void create_dataset(float *X, int N, int dist){

  switch(dist){
  case 0:
    cube(X, N);
    break;
  case 1:
    plummer(X, N);
    break;
  default:
    plummer(X, N);
    break;
    
  break;
  }

}
