#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "lib.h"



typedef struct 
{
  // The id of the thread.
  int threadID;
  // The pointer to the array that will contain the sorted data.
  float *Y;
  // The array containing the input points.
  float *X;
  // The pointer to the permutation vector.
  unsigned int *permutation_vector ;

  // Where the thread will start iterating over the points.
  int start;
  // Where the thread will end iterating over the points.
  int end;
}thread_rearrangement_data;


// Function that will be called by every thread to rearrange the data
// in the memory so as to ensure that points that are close in space
// are also located in a continuous region of the memory.
void *thread_data_rearrangement(void * arg)
{
  // For loop indices.
  int i ;
  
  // Get the pointer to the input data.
  thread_rearrangement_data *dataPtr = 
    (thread_rearrangement_data *)arg ;
  
  // Set the starting and the ending point of the iteration
  int start = dataPtr->start;
  int end = dataPtr->end;
  
 
  for( i = start ; i < end ; i++)
  {
    // Copy to the i-th element of Y the point that corresponds
    // to the i-th entry of the permutation vector.
    memcpy(&dataPtr->Y[i*DIM], 
      &dataPtr->X[ dataPtr->permutation_vector[i]*DIM ], 
      DIM*sizeof(float));
  }
  
}

void data_rearrangement(float *Y, float *X, 
			unsigned int *permutation_vector, 
			int N)
{
  // Initiliaze the static array containing the identities of threads/
  pthread_t threads[THREADS];
  // Declare an attribute for the above threads.
  pthread_attr_t attribute;
  
  // Initiliaze the attribute so that the threads we create are 
  // joinable.
  pthread_attr_init(&attribute);
  // Set the threads to be joinable.
  pthread_attr_setdetachstate(&attribute, PTHREAD_CREATE_JOINABLE);
  
  // Split the data into even chunks for every thread.
  int chunk = N  /  THREADS  ;
  
  if (chunk < 1 )
    chunk = 1 ;
    
  // Iteration indices.
  int i = 0 ; 
  

  
  // Static array containing the data 
  thread_rearrangement_data thread_data[THREADS];
  
  // Initialize the data that will be passed to every thread.
  for ( i = 0 ; i < THREADS ; i++)
  {
    thread_data[i].threadID = i ;
    thread_data[i].Y = Y ;
    thread_data[i].X = X;
    thread_data[i].permutation_vector = permutation_vector ;
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
  
  
  // Create the threads and rearrange the data.
  for( threadIt = 0; threadIt < THREADS ; threadIt ++ )
  {
    // Create the threads.
    threadFlag = pthread_create( &threads[threadIt] , &attribute ,
      thread_data_rearrangement, (void*) &thread_data[threadIt] );
    
    // Check if the thread was succesfully created.
    // If not terminate the iteration.
    if (threadFlag )
    {
      printf("Thread %d could not be created. Error Code is %d \n",
        threadIt , threadFlag );
      exit(-1);
    }
    
  } // End of thread data rearrangement.
  
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
    


}
