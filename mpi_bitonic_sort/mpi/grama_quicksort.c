#include <stdlib.h>
#include <string.h>
#include "omp.h"
#include "utils.h"


void grama_quicksort(uint32_t* array, uint32_t* sorted_array, unsigned int N,
                     unsigned int p, int lv)
{
  /* printf("\nlevel of recursion is: %d\n", lv); */
  /* printf("workers available: %d\n", p); */
  /* printf("array - %d - %d - %d:\n", lv, p, N); */
  /* print_array(array, N); */
  /* printf("sorted array - %d - %d - %d:\n", lv, p, N); */
  /* print_array(sorted_array, N); */
  if (p == 1) {
    qsort(array, N, sizeof(uint32_t), cmpfunc);
    memcpy(sorted_array, array, N*sizeof(uint32_t));
  }
  else {
    unsigned int ave = N / p;
    int procs, procl;
    unsigned int len_s[p], len_l[p];
    uint64_t pivot = 0;
    unsigned int div, id, start, end, i, j_s, j_l, offset_s, offset_l;
    omp_set_num_threads(p);
    #pragma omp parallel shared(array, sorted_array, len_s, len_l, pivot) firstprivate(ave, N) private(div, id, start, end, i, j_s, j_l, offset_s, offset_l)
    {
      #pragma omp for reduction(+: pivot)
      for (i = 0; i < N; ++i)
        pivot += array[i];
      #pragma omp single
      {
        pivot = pivot / N;
      }
      id = omp_get_thread_num();
      start = id * ave;
      end = id != omp_get_num_threads() - 1 ? (id+1) * ave : N;

      len_s[id] = 0;
      len_l[id] = 0;
      for (i = start; i < end; i++) {
        if (array[i] < pivot) {
          len_s[id]++;
        }
        else {
          len_l[id]++;
        }
      }
      #pragma omp barrier
      #pragma omp single
      {
        offset_s = 0;
        offset_l = 0;
        for (i = 0; i < p; ++i) {
          len_s[i] += offset_s;
          offset_s = len_s[i];
          len_l[i] += offset_l;
          offset_l = len_l[i];
        }
      }

      j_s = 0;
      j_l = 0;
      offset_s = id != 0 ? len_s[id-1] : 0;
      offset_l = id != 0 ? len_l[id-1] + len_s[p-1] : len_s[p-1];
      for (i = start; i < end; i++) {
        if (array[i] < pivot) {
          sorted_array[offset_s + j_s++] = array[i];
        }
        else {
          sorted_array[offset_l + j_l++] = array[i];
        }
      }
    }

    swap(&array, &sorted_array);

    procs = (int)(((len_s[p-1] / (float)N)) * p + 0.5);
    if (procs == 0) procs = 1;
    procl = p - procs;
    /* procs = p / 2; */
    /* procl = p / 2; */
    div = len_s[p-1];

    #pragma omp parallel shared(array, sorted_array, N, div, procs, procl) firstprivate(lv) num_threads(2)
    {
      #pragma omp sections
      {
        #pragma omp section
        grama_quicksort(&array[0], &sorted_array[0], div, procs, ++lv);
        #pragma omp section
        grama_quicksort(&array[div], &sorted_array[div], N-div, procl, ++lv);
      }
    }
    /* printf("Work division: s-%d l-%d\n", procs, procl); */
    /* printf("Len-s: %d Len-l: %d N: %d\n", div, N-div, N); */
    /* printf("after-after %d - %d:\n", lv, N); */
    /* print_array(sorted_array, N); */
    /* print_array(array, N); */
  }
}
