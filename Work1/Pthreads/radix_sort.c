#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "lib.h"

#define MAXBINS 8
#define MAXLEVEL 1


// The struct that contains the data that is passed to every thread
// so as to calculate the number of elements in every bin.
typedef struct 
{
  // Unique ID of every thread.
  int threadID;
  // The pointer to the array of the morton codes' of the points
  // that will be sorted recursively. 
  unsigned long int *morton_codes;
  // The shift that will be performed to find the corresponding bin.
  int *sft;
  // A pointer to the final array whose every element contains the
  // number of particles of every bin.
  int *bin_sizes;
  
  // Start of Iterations.
  int start;
  // End of Iterations.
  int end;
  
  // Pointer to a spin lock that will be used to prevent race conditions
  // when calculating the total bin sizes.
  pthread_spinlock_t *lock;
  
}bin_data;

// This struct contains all the necessary data so as to call the 
// truncated radix sort recursively.
typedef struct
{
  // The pointer to the array containing the morton codes.
  unsigned long int *morton_codes;
  // The pointer to the array containing the sorted morton codes.
  unsigned long int *sorted_morton_codes;
  // The permutation vector.
  unsigned int *permutation_vector;
  // The pointer to the array of the indices of the particles.
  unsigned int *index;
  // The array that records the level of every node.
  int *level_record;
  // The number of particles to be sorted.
  int N;
  // The population Threshold
  int population_threshold;
  // The shift to the morton code that will be applied to find
  // the bin where the particle belongs.
  int sft;
  // The level of the node.
  int lv;
  }recursion_data;
  


// The function that is called by every thread to calculate the 
// number of elements in each bin.
void * calculate_bin_sizes(void *arg)
{
  // Iteration Indices.
  int i , j ;
  int local_bin_offsets[MAXBINS] = {0};
  
  bin_data *dataPtr = (bin_data *)arg;
  
  // Assign the start and end of iterations.
  int start = dataPtr->start;
  int end = dataPtr->end;
  
  // Get the necessary shift for finding the correct bin.
  int sft = *(dataPtr->sft) ;
  unsigned long int *morton_codes = dataPtr->morton_codes;
  
  for (j = start ; j < end ; ++j)
  {
    unsigned int ii = ( morton_codes[j] >> sft) & 0x07;
    local_bin_offsets[ii]++;
  }
  
  pthread_spin_lock(dataPtr->lock);
  
  for (i = 0 ; i<MAXBINS ; i++)
    (dataPtr->bin_sizes)[i] += local_bin_offsets[i] ;
    
  pthread_spin_unlock(dataPtr->lock);
  
  } // End of thread function 


void swap_long(unsigned long int **x, unsigned long int **y)
{
  unsigned long int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;
}

void swap(unsigned int **x, unsigned int **y)
{
  unsigned int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;
}

void parallel_scan_exclude(int* a, int n)
{
  if (n == 1)
  {
    a[0] = 0;
    return;
  }
  int i;
  int c[n/2];
  for (i = 0; i < n/2; ++i)
  {
    c[i] = a[2*i] + a[2*i+1];
  }
  parallel_scan_exclude(c, n/2);
  int b[n];
  for (i = 0; i < n; ++i)
  {
    {
      if (i%2 == 0)
        b[i] = c[i/2];
      else
        b[i] = c[(i-1)/2] + a[i-1];
    }
  }
  for (i = 0; i < n; ++i)
  {
    a[i] = b[i];
  }
}


void threaded_radix_sort(void *arg)
{
  }

void truncated_radix_sort(unsigned long int *morton_codes,
    unsigned long int *sorted_morton_codes,
    unsigned int *permutation_vector,
    unsigned int *index,
    int *level_record,
    int N,
    int population_threshold,
    int sft, int lv)
{
  
  if(N<=0){

    return;
  }
  
  else if (N <= population_threshold || sft < 0)
  { 
    // Record the level of the node.
    level_record[0] = lv;
    // Base case. The node is a leaf
    // Copy the pernutation vector
    memcpy(permutation_vector, index, N*sizeof(unsigned int));
    // Copy the Morton codes 
    memcpy(sorted_morton_codes, morton_codes, N*sizeof(unsigned long int));
    return;
  }
  else
  {
    // Record the level of the node.
    level_record[0] = lv;
    int bin_offsets[MAXBINS] = {0};
    int bin_offsets_cnt[MAXBINS] = {0};
    int i = 0, j = 0;
    // If we are not deep enough in the recursive algorithm 
    // use the parallel MSB radix sort. 
    // This is done so as not to create a huge number of threads
    // in the deeper levels where the cost of creating them will
    // be greater than the time we save by parallelizing the code.
    if (lv < MAXLEVEL)
    {
      
      
      // Initiliaze the static array containing 
      // the identities of threads.
      pthread_t threads[THREADS];
      // Declare an attribute for the above threads.
      pthread_attr_t attribute;
      
      // Declare the spin lock and initialize it.
      pthread_spinlock_t lock ;
      
      pthread_spin_init(&lock,PTHREAD_PROCESS_PRIVATE);
      
      // Initialize the attribute so that the threads we create are 
      // joinable.
      pthread_attr_init(&attribute);
      pthread_attr_setdetachstate(&attribute, PTHREAD_CREATE_JOINABLE);
      
      // Split the data into even chunks for every thread.
      int chunk = N  /  THREADS  ;
        
      // Iteration indices.
      int i = 0, j = 0; 
      
      
      // Static array containing the data 
      bin_data thread_data[THREADS];
      
      // Initialize the data that will be passed to every thread.
      for ( i = 0 ; i < THREADS ; i++)
      {
        thread_data[i].threadID = i ;
        thread_data[i].morton_codes = morton_codes ;
        thread_data[i].sft = &sft ;
        thread_data[i].bin_sizes = &bin_offsets[0] ;
        thread_data[i].lock = &lock;
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
      
      
      // Create the threads and calculate the bin sizes.
      for( threadIt = 0; threadIt < THREADS ; threadIt ++ )
      {
        // Create the threads.
        threadFlag = pthread_create( &threads[threadIt] , &attribute ,
          calculate_bin_sizes , (void*) &thread_data[threadIt] );
        
        // Check if the thread was succesfully created.
        // If not terminate the iteration.
        if (threadFlag )
        {
          printf("Thread %d could not be created. Error Code is %d \n",
            threadIt , threadFlag );
          exit(-1);
        }
        
      } // End of bin size calculation using threads.
      
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

      bin_offsets_cnt[0] = 0;
      for (i = 1; i < MAXBINS; ++i)
      {
        bin_offsets_cnt[i] = bin_offsets_cnt[i-1] + bin_offsets[i-1];
        bin_offsets[i-1] = bin_offsets_cnt[i];
      }
      bin_offsets[MAXBINS-1] += bin_offsets[MAXBINS-2];

      // Find the bin in which each point belongs.
      for (j = 0; j < N; ++j)
      {
        unsigned int ii = (morton_codes[j]>>sft) & 0x07;
        permutation_vector[bin_offsets_cnt[ii]] = index[j];
        sorted_morton_codes[bin_offsets_cnt[ii]] = morton_codes[j];
        bin_offsets_cnt[ii]++;
      }
     
      //swap the index pointers  
      swap(&index, &permutation_vector);
      //swap the code pointers 
      swap_long(&morton_codes, &sorted_morton_codes);

      /** 
       * Define a parallel region where we will loop over all the bins
       * and recursively call the radix sort function on each one.
       * Shared Attributes : The input and output arrays , the
       *  parameters of the sorting function and the bin size and offset
       *  arrays.
      **/
      for (i = 0; i < MAXBINS; ++i)
      {
        int offset = (i>0) ? bin_offsets[i-1] : 0;
        int size = bin_offsets[i] - offset;
        truncated_radix_sort(&morton_codes[offset],
            &sorted_morton_codes[offset],
            &permutation_vector[offset],
            &index[offset], &level_record[offset],
            size,
            population_threshold,
            sft-3, lv+1);
      } 
    }  // should we run in parallel?
    // If not execute the recursive call serially.
    else
    {
      for (j = 0; j < N; ++j)
      {
        unsigned int ii = (morton_codes[j]>>sft) & 0x07;
        bin_offsets[ii]++;
      }

      // scan prefix (must change this code)  
      bin_offsets_cnt[0] = 0;
      for (i = 1; i < MAXBINS; ++i)
      {
        bin_offsets_cnt[i] = bin_offsets_cnt[i-1] + bin_offsets[i-1];
        bin_offsets[i-1] = bin_offsets_cnt[i];
      }
      bin_offsets[MAXBINS-1] += bin_offsets[MAXBINS-2];
      
      for (j = 0; j < N; ++j)
      {
        unsigned int ii = (morton_codes[j]>>sft) & 0x07;
        permutation_vector[bin_offsets_cnt[ii]] = index[j];
        sorted_morton_codes[bin_offsets_cnt[ii]] = morton_codes[j];
        bin_offsets_cnt[ii]++;
      }
      
      //swap the index pointers  
      swap(&index, &permutation_vector);

      //swap the code pointers 
      swap_long(&morton_codes, &sorted_morton_codes);

      /* Call the function recursively to split the lower levels */
      for (i = 0; i < MAXBINS; ++i)
      {
        int offset = (i>0) ? bin_offsets[i-1] : 0;
        int size = bin_offsets[i] - offset;
        truncated_radix_sort(&morton_codes[offset], 
             &sorted_morton_codes[offset], 
             &permutation_vector[offset], 
             &index[offset], &level_record[offset], 
             size, 
             population_threshold,
             sft-3, lv+1);
      }
    }  // should we run in parallel?
  }  // have we reached a leaf node?
}
