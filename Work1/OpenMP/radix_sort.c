#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "lib.h"

#define MAXBINS 8
#define MAXLEVEL 3


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
    if (i%2 == 0)
      b[i] = c[i/2];
    else
      b[i] = c[(i-1)/2] + a[i-1];
  }
  for (i = 0; i < n; ++i)
  {
    a[i] = b[i];
  }
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
  level_record[0] = lv; // record the level of the node

  if (N <= population_threshold || sft < 0)
  { 
    // Base case. The node is a leaf
    // Copy the pernutation vector
    memcpy(permutation_vector, index, N*sizeof(unsigned int));
    // Copy the Morton codes 
    memcpy(sorted_morton_codes, morton_codes, N*sizeof(unsigned long int));
    return;
  }
  else
  {
    int bin_sizes[MAXBINS] = {0};
    // If we are not deep enough in the recursive algorithm 
    // use the parallel MSB radix sort. 
    // This is done so as not to create a huge number of threads
    // in the deeper levels where the cost of creating them will
    // be greater than the time we save by parallelizing the code.
    if (lv < MAXLEVEL)
    {
      // Loop indeces.
      int i; 
      int j = 0;
      
      int bin_offsets[MAXBINS] = {0};
    
      // Find which child each point belongs to .
      #pragma omp parallel shared(morton_codes, bin_sizes) private(j)
      {
        // Thread local variable used for calculating the number of
        // elements that are in every bin. 
        int local_bin_sizes[MAXBINS] = {0};

        // Every thread finds the bin where it's corresponding points
        // belong to.
        #pragma omp for schedule(guided) nowait
        for (j = 0; j < N; ++j)
        {
          unsigned int ii = (morton_codes[j]>>sft) & 0x07;
          local_bin_sizes[ii]++;
        }
        
        // Synchronization :
        // In order to avoid race conditions and data loss
        // this region is declared critical and can be executed
        // by one thread at any time.
        #pragma omp critical
        {
          for (i = 0; i < MAXBINS; i++)
          {
            // Add to the total number of elements of bin #i
            // the ones found by the current thread.
            bin_offsets[i] = bin_sizes[i] += local_bin_sizes[i];
          }
        }
      }

      parallel_scan_exclude(bin_offsets, MAXBINS);

      // Find the bin in which each point belongs.
      for (j = 0; j < N; ++j)
      {
        unsigned int ii = (morton_codes[j]>>sft) & 0x07;
        permutation_vector[bin_offsets[ii]] = index[j];
        sorted_morton_codes[bin_offsets[ii]] = morton_codes[j];
        bin_offsets[ii]++;
      }

      printf("\nBIN-OFFSET-0: %d\n", bin_offsets[0]);
     
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
      #pragma omp parallel private(i)
      //~ #pragma shared( morton_codes , sorted_morton_codes , 
      //~ #pragma permutation_vector, index , level_record , size , 
      //~ #pragma population_threshold , sft , lv ) private( i )
      {
        // Define a parallel loop for the recursive calls.
        #pragma omp for schedule(guided) 
        for (i = 0; i < MAXBINS; ++i)
        {
          int offset = 0;
          if (i != 0)
            offset = bin_offsets[i-1];
          truncated_radix_sort(&morton_codes[offset],
              &sorted_morton_codes[offset],
              &permutation_vector[offset],
              &index[offset], &level_record[offset],
              bin_sizes[i],
              population_threshold,
              sft-3, lv+1);
        } 
      }
    }  // if lv < MAXLEVEL
    // If not execute the recursive call serially.
    else
    {
      //~ // If there are enough threads to execute the program in parallel.
      //~ 
      //~ // Find which child each point belongs to 
      int j = 0, i = 0;
      for (j = 0; j < N; ++j)
      {
        unsigned int ii = (morton_codes[j]>>sft) & 0x07;
        bin_sizes[ii]++;
      }  


      parallel_scan_exclude(bin_sizes, MAXBINS);
      // scan prefix (must change this code)  
      /*int offset = 0, i = 0;*/
      /*for(i=0; i<MAXBINS; i++){*/
        /*int ss = BinSizes[i];*/
        /*BinSizes[i] = offset;*/
        /*offset += ss;*/
      /*}*/

      for (j = 0; j < N; ++j)
      {
        unsigned int ii = (morton_codes[j]>>sft) & 0x07;
        permutation_vector[bin_sizes[ii]] = index[j];
        sorted_morton_codes[bin_sizes[ii]] = morton_codes[j];
        bin_sizes[ii]++;
      }

      //swap the index pointers  
      swap(&index, &permutation_vector);

      //swap the code pointers 
      swap_long(&morton_codes, &sorted_morton_codes);

      /* Call the function recursively to split the lower levels */
      int offset = 0;
      for (i = 0; i < MAXBINS; ++i)
      {
        int size = bin_sizes[i] - offset;
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
