#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "lib.h"

#define MAXBINS 8


void swap_long(unsigned long int **x, unsigned long int **y){

  unsigned long int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;

}

void swap(unsigned int **x, unsigned int **y){

  unsigned int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;

}

void truncated_radix_sort(unsigned long int *morton_codes, 
			  unsigned long int *sorted_morton_codes, 
			  unsigned int *permutation_vector,
			  unsigned int *index,
			  int *level_record,
			  int N, 
			  int population_threshold,
			  int sft, int lv){

  int BinSizes[MAXBINS] = {0};
  unsigned int *tmp_ptr;
  unsigned long int *tmp_code;
  
  level_record[0] = lv; // record the level of the node

  if( N<=population_threshold || sft < 0) { // Base case. The node is a leaf

    //~ memcpy(permutation_vector, index, N*sizeof(unsigned int)); // Copy the pernutation vector
    //~ memcpy(sorted_morton_codes, morton_codes, N*sizeof(unsigned long int)); // Copy the Morton codes 
    #pragma omp parallel 
    {
      #pragma omp sections 
      {
        #pragma omp section
        //swap the index pointers  
        memcpy(permutation_vector, index, N*sizeof(unsigned int)); // Copy the pernutation vector
    
        #pragma omp section
        //swap the code pointers 
        memcpy(sorted_morton_codes, morton_codes, N*sizeof(unsigned long int)); // Copy the Morton codes 
      }  
    }
    return;
  }
  else{
    // If we are not deep enough in the recursive algorithm 
    // use the parallel MSB radix sort. 
    // This is done so as not to create a huge number of threads
    // in the deeper levels where the cost of creating them will
    // be greater than the time we save by parallelizing the code.
    if ( lv < 4  )
    {
      
      // Loop indexes.
      int i ; 
      int j = 0;

      // Find which child each point belongs to 

    
       // Find which child each point belongs to .
      #pragma omp parallel shared(morton_codes , BinSizes ) private(j)
      {
        // Thread local variable used for calculating the number of
        // elements that are in every bin. 
        int localBinSizes[MAXBINS] = {0};
        {
          // Every thread finds the bin where it's corresponding points
          // belong to.
          #pragma omp for schedule(guided) nowait
          for(j=0; j<N; j++){
            unsigned int ii = (morton_codes[j]>>sft) & 0x07;
            localBinSizes[ii]++;
          }
          
          // Synchronization :
          // In order to avoid race conditions and data loss 
          // this region is declared critical and can be executed 
          // by on thread at any time.
          #pragma omp critical
          {
            for (i = 0 ; i < MAXBINS ; i++ )
            {
              // Add to the total number of elements of bin #i
              // the ones found by the current thread.
              BinSizes[i] += localBinSizes[i] ;
            }
          }
        }
      }
      
      
      
      // scan prefix (must change this code)  
      int offset = 0 ;
      for(i=0; i<MAXBINS; i++){
        int ss = BinSizes[i];
        BinSizes[i] = offset;
        offset += ss;
      }
      
      
      //~ exit(-1);
      for(j=0; j<N; j++){
        unsigned int ii = (morton_codes[j]>>sft) & 0x07;
        permutation_vector[BinSizes[ii]] = index[j];
        sorted_morton_codes[BinSizes[ii]] = morton_codes[j];
        BinSizes[ii]++;
      }
      

      #pragma omp parallel 
      {
        #pragma omp sections 
        {
          #pragma omp section
          //swap the index pointers  
          swap(&index, &permutation_vector);
    
          #pragma omp section
          //swap the code pointers 
          swap_long(&morton_codes, &sorted_morton_codes);
        }  
      }
      
      
      //~ //swap the index pointers  
      //~ swap(&index, &permutation_vector);
  //~ 
      //~ //swap the code pointers 
      //~ swap_long(&morton_codes, &sorted_morton_codes);

      /* Call the function recursively to split the lower levels */
    
      

      int size[MAXBINS] = {0};
      int offSet[MAXBINS] = {0};
      // Set for every bin from where it's elements begin(offset)
      // and how many of them it has.
      size[0] = BinSizes[0] ;
      //~ #pragma omp parallel for schedule(guided) shared(size,offset)
      //~ #pragma private(i)
      
      
      for (i = 1 ; i < MAXBINS ; i++)
      {
        size[i] = BinSizes[i] - BinSizes[i-1];
        offSet[i] = BinSizes[i-1] ;
        //~ printf("i %d size %d \n",i,size[i]);
        //~ printf("i %d offset %d \n",i,offSet[i]);
      }
      
      /** 
       * Define a parallel region where we will loop over all the bins
       * and recursively call the radix sort function on each one.
       * Shared Attributes : The input and output arrays , the
       *  parameters of the sorting function and the bin size and offset
       *  arrays.
      **/
      
      #pragma omp parallel private(i)
      //~ #pragma shared( morton_codes , sorted_morton_codes , 
      //~ #pragma permutation_vector, index , level_record , size , 
      //~ #pragma population_threshold , sft , lv ) private( i )
      {
        // Define a parallel loop for the recursive calls.
        #pragma omp  for schedule(guided) 
        for( i = 0 ; i < MAXBINS ; i++ )
        {
            truncated_radix_sort(&morton_codes[offSet[i]], 
            &sorted_morton_codes[offSet[i]], 
            &permutation_vector[offSet[i]], 
            &index[offSet[i]], &level_record[offSet[i]], 
            size[i], 
            population_threshold,
            sft-3, lv+1);

          } 
      }
      
      
    }
    // If not execute the recursive call serially.
    else
    {
      //~ // If there are enough threads to execute the program in parallel.
      //~ 
      //~ // Find which child each point belongs to 
      int j = 0;
      for(j=0; j<N; j++){
        unsigned int ii = (morton_codes[j]>>sft) & 0x07;
        BinSizes[ii]++;
      }  
      
  
      // scan prefix (must change this code)  
      int offset = 0, i = 0;
      for(i=0; i<MAXBINS; i++){
        int ss = BinSizes[i];
        BinSizes[i] = offset;
        offset += ss;
      }
      
      for(j=0; j<N; j++){
        unsigned int ii = (morton_codes[j]>>sft) & 0x07;
        permutation_vector[BinSizes[ii]] = index[j];
        sorted_morton_codes[BinSizes[ii]] = morton_codes[j];
        BinSizes[ii]++;
      }

      //swap the index pointers  
      swap(&index, &permutation_vector);
  
      //swap the code pointers 
      swap_long(&morton_codes, &sorted_morton_codes);
  
      /* Call the function recursively to split the lower levels */
      offset = 0; 
      for(i=0; i<MAXBINS; i++){
        
        int size = BinSizes[i] - offset;
        
        truncated_radix_sort(&morton_codes[offset], 
           &sorted_morton_codes[offset], 
           &permutation_vector[offset], 
           &index[offset], &level_record[offset], 
           size, 
           population_threshold,
           sft-3, lv+1);
        offset += size;  
      }
    }
    
      
  } 
}

