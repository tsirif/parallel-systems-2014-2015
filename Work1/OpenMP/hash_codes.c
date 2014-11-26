#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "lib.h"


inline unsigned int compute_code(float x, float low, float step)
{
  return floor((x - low) / step);
}


/* Function that does the quantization */
void quantize(unsigned int *codes, float *X,
    float *low, float step, int N)
{

  int i = 0, j = 0; 
  // Index of element to be accessed.
  int index;

  #pragma omp parallel private(i, j, index) if(N > THREADS*THREADS)
  {
    #pragma omp for schedule(guided)
    // Increase the step to 2 for every iteration so as to pipeline
    // more instructions .
    for (i = 0; i < N; ++i)
    {
      for (j = 0; j < DIM; ++j)
      {
        index = i*DIM + j;
        codes[index] = compute_code(X[index], low[j], step);
      }
    }
  }
}

inline float max_range(float *x)
{
  int i=0;
  float max = -FLT_MAX;
  for (i = 0; i < DIM; ++i)
  {
    if (max < x[i])
    {
      max = x[i];
    }
  }
  return max;
}

void compute_hash_codes(unsigned int *codes, float *X, int N, 
			int nbins, float *min, 
			float *max)
{  
  float range[DIM];
  float qstep;
  int i = 0;

  for(i=0; i<DIM; i++)
  {
    range[i] = fabs(max[i] - min[i]);  // The range of the data
    // Add something small to avoid having points exactly at the boundaries 
    range[i] += 0.01*range[i];
  }
  
  qstep = max_range(range) / nbins; // The quantization step 
  
  quantize(codes, X, min, qstep, N); // Function that does the quantization
}



