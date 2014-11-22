#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "float.h"
#include "lib.h"

typedef struct 
{
  // The id of the thread.
  int threadID;
  // The pointer to the array containing the codes.
  unsigned int * codes;
  // The array containing the input points.
  float *X;
  // The min of every dim.
  float *low ;
  // the range .
  float step ;

  // Where the thread will start iterating over the points.
  int start;
  // Where the thread will end iterating over the points.
  int end;
}thread_hash_data;
  


unsigned int compute_code(float x, float low, float step){

  return floor((x - low) / step);

}

void *pthread_hash_callback(void *arg)
{
  // For loop indices.
  int i ;
  int j ;
  
  // Get the pointer to the input data.
  thread_hash_data *dataPtr = (thread_hash_data *)arg ;
  
  // Set the starting and the ending point of the iteration
  int start = dataPtr->start;
  int end = dataPtr->end;
  
  for ( i = start ; i < end ; i++)
  {
    for ( j = 0 ; j < DIM ; j++ )
      dataPtr->codes[i*DIM + j] = compute_code( dataPtr->X[i*DIM + j] ,
        dataPtr->low[j] , dataPtr->step );
  }
  
}


/* Function that does the quantization */
void quantize(unsigned int *codes, float *X, float *low, float step, int N){

  // Initiliaze the static array containing the identities of threads/
  pthread_t threads[THREADS];
  // Declare an attribute for the above threads.
  pthread_attr_t attribute;
  
  // Initiliaze the attribute so that the threads we create are 
  // joinable.
  pthread_attr_init(&attribute);
  pthread_attr_setdetachstate(&attribute, PTHREAD_CREATE_JOINABLE);
  
  // Split the data into even chunks for every thread.
  int chunk = N  /  THREADS  ;
  
  if (chunk < 1 )
    chunk = 1 ;
    
  // Iteration indices.
  int i = 0, j = 0; 
  
  // Index of element to be accessed.
  int index;
  
  // Static array containing the data 
  thread_hash_data thread_data[THREADS];
  
  // Initialize the data that will be passed to every thread.
  for ( i = 0 ; i < THREADS ; i++)
  {
    thread_data[i].threadID = i ;
    thread_data[i].codes = codes ;
    thread_data[i].X = X;
    thread_data[i].low = low ;
    thread_data[i].step = step ;
    thread_data[i].start = i*chunk ;
    thread_data[i].end = (i+1)*chunk ;
    if (i==THREADS-1)
      thread_data[i].end = N ;
      
  }
  
  // The index that iterates over the threads
  int threadIt ;
  // The flag that checks for errors during the creation of the 
  // threads.
  int threadFlag;
  
  
  // Create the threads and perform the hash encoding.
  for( threadIt = 0; threadIt < THREADS ; threadIt ++ )
  {
    // Create the threads.
    threadFlag = pthread_create( &threads[threadIt] , &attribute ,
      pthread_hash_callback, (void*) &thread_data[threadIt] );
    
    // Check if the thread was succesfully created.
    // If not terminate the iteration.
    if (threadFlag )
    {
      printf("Thread %d could not be created. Error Code is %d \n",
        threadIt , threadFlag );
      exit(-1);
    }
    
  } // End of thread Hashing
  
  pthread_attr_destroy(&attribute);
  
  // Pointer used to check if the threads where successfully joined.
  void *status; 

  
  // Join the threads.
  for( threadIt = 0 ; threadIt < THREADS ; threadIt++) 
  {
    // Join thread #threadIt .
    threadFlag = pthread_join( threads[threadIt], &status);
    if (threadFlag ) 
    {
      printf("ERROR; return code from pthread_join()" 
            "is %d\n", threadFlag );
      exit(-1);
    }
    //~ printf("Main: completed join with thread %d having a status   "
          //~ "of %ld\n" , threadIt , (long)status );
  } // End of Joining Threads.
    


}  // End of Hash Codes.

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

  for(i=0; i<DIM; i++)
  {
    range[i] = fabs(max[i] - min[i]); // The range of the data
    range[i] += 0.01*range[i]; // Add somthing small to avoid having points exactly at the boundaries 
  }

  qstep = max_range(range) / nbins; // The quantization step 
  
  quantize(codes, X, min, qstep, N); // Function that does the quantization

}



