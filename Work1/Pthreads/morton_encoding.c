#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "lib.h"

typedef struct
{
  int tid;
  int N;
  unsigned long int* morton_codes;
  unsigned int* simple_codes;
} Codes;

inline unsigned long int splitBy3(unsigned int a)
{
    unsigned long int x = a & 0x1fffff; // we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

inline unsigned long int mortonEncode_magicbits(unsigned int x, unsigned int y, unsigned int z)
{
    unsigned long int answer = 0;
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    return answer;
}

// Function for morton code split calculation
void* calculate_morton_codes(void* arg)
{
  int i;
  Codes* codes = (Codes *) arg;
  for (i = codes->tid ; i < codes->N; i += THREADS)
  {
    codes->morton_codes[i] = mortonEncode_magicbits(
        codes->simple_codes[i*DIM],
        codes->simple_codes[i*DIM + 1],
        codes->simple_codes[i*DIM + 2]);
  }
  pthread_exit((void*) 0);
}

/* The function that transform the morton codes into hash codes */ 
void morton_encoding(unsigned long int *morton_codes,
    unsigned int *simple_codes, int N, int max_level)
{
  long i;
  int rcode;
  pthread_t threads[THREADS];
  Codes codes[THREADS];
  pthread_attr_t tattr;
  void* status;

  // Initialization of static codes [Codes] and tattr [pthread_attr_t]
  pthread_attr_init(&tattr);
  pthread_attr_setdetachstate(&tattr, PTHREAD_CREATE_JOINABLE);

  // Thread creation and job delegation
  for (i = 0; i < THREADS; ++i)
  {
    codes[i].tid = i;
    codes[i].N = N;
    codes[i].morton_codes = morton_codes;
    codes[i].simple_codes = simple_codes;
    /*printf("Starting thread num #%ld\n", i);*/
    rcode = pthread_create(&threads[i], &tattr,
        calculate_morton_codes, (void*) &codes[i]);
    if (rcode)
    {
      printf("ERROR; return code from pthread_create() is %d\n", rcode);
      exit(-1);
    }
  }

  // Thread attribute destruction and thread meeting point
  pthread_attr_destroy(&tattr);

  for (i = 0; i < THREADS; ++i)
  {
    rcode = pthread_join(threads[i], &status);
    if (rcode)
    {
      printf("ERROR; return code from pthread_join() is %d\n", rcode);
      exit(-1);
    }
  }
}
