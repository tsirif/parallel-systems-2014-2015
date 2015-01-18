#include <stdlib.h>
#include "omp.h"
#include "utils.h"

int partition(uint32_t* array, int a, int b, uint32_t pivot,
              unsigned int* len_s, unsigned int* len_l);

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
  }
  else {
    unsigned int ave = N / p;
    int procs, procl;
    unsigned int len_s[p], len_l[p];
    uint64_t pivot;
    unsigned int div, id, start, end, i, j_s, j_l, offset_s, offset_l;
    omp_set_num_threads(p);
    #pragma omp parallel shared(array, sorted_array, len_s, len_l, pivot) firstprivate(ave, N) private(div, id, start, end, i, j_s, j_l, offset_s, offset_l)
    {
      #pragma omp for schedule(static) reduction(+: pivot)
      for (i = 0; i < N; ++i)
        pivot += array[i];
      #pragma omp single
      {
        pivot /= N;
      }
      id = omp_get_thread_num();
      start = id * ave;
      end = id != omp_get_num_threads() - 1 ? (id+1) * ave : N;

      /* div = partition(array, start, end, pivot, &len_s[id], &len_l[id]); */
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

    /* printf("after-after %d - %d:\n", lv, N); */
    /* print_array(sorted_array, N); */
    swap(&array, &sorted_array);
    /* print_array(array, N); */

    // TODO better with round fun
    /* procs = (int)(((len_s[p-1] / (float)N)) * p + 0.5); */
    /* procl = p - procs; */
    procs = p / 2;
    procl = p / 2;
    /* printf("Work division: s-%d l-%d\n", procs, procl); */
    div = len_s[p-1];
    /* printf("Len-s: %d Len-l: %d N: %d\n", div, N-div, N); */

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
  }
}

int partition(uint32_t* array, int a, int b, uint32_t pivot,
              unsigned int* len_s, unsigned int* len_l)
{
  uint32_t lt[b-a], ge[b-a];
  int i, j;
  int lt_n = 0, ge_n = 0;

  for (i = a; i < b; i++) {
    if (array[i] < pivot) {
      lt[lt_n++] = array[i];
    }
    else {
      ge[ge_n++] = array[i];
    }
  }

  for(i = 0; i < lt_n; i++){
    array[a + i] = lt[i];
  }
  for(j = 0; j < ge_n; j++){
    array[a + lt_n + j] = ge[j];
  }

  *len_s = lt_n;
  *len_l = ge_n;

  return a + lt_n;
}
