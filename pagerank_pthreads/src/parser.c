#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "pagerank_pthreads/utils.h"

/**
 * @brief read a directed graph from a file
 * @param filename [char const *] input file's name
 * @param L [unsigned int ***] transition sparse matrix
 * @param C [unsigned int **] output edges per node array
 * @param N [unsigned int *] number of nodes
 * @param E [unsinged int *] number of edges
 * @return if operation is successful or not
 */
int read_graph(char const * filename, uint*** L, uint** C, uint* N, uint* E)
{
  if (capacity) free((void*)capacity);
  capacity = NULL;
  // open input file
  FILE *fin;
  fin = fopen(filename, "r");
  if (fin == NULL)
  {
    printf("Error opening input file: %s\n", filename);
    exit(1);
  }

  char line[1000];

  uint e = 1;
  while (!feof(fin) && e > 0)
  {
    fgets(line, sizeof(line), fin);
    // ignore sentences starting from #
    if (strncmp(line, "#", 1) == 0)
    {
      // read number of nodes and edges
      if (strstr(line, "Nodes") != NULL)
      {
        sscanf(line, "# Nodes: %u Edges: %u\n", N, E);
        e = *E;
        capacity = (uint*) malloc((*N) * sizeof(uint));
        if (capacity == NULL) exit(-1);
        *L = (uint**) malloc((*N) * sizeof(uint*));
        if (*L == NULL) exit(-1);
        *C = (uint*) malloc((*N) * sizeof(uint));
        if (*C == NULL) exit(-1);
        uint i;
        for (i = 0; i < *N; ++i)
        {
          capacity[i] = DFL_CAPACITY;
          C[0][i] = 0;
          L[0][i] = (uint*) malloc(capacity[i] * sizeof(uint));
          if (L[0][i] == NULL) exit(-1);
        }
      }
      continue;
    }
    // read an edge (from and to nodes)
    uint from, to;
    sscanf(line, "%u %u\n", &from, &to);
    e--;
    // condition to ensure the removal of self-transition
    if (from == to)
    {
      (*E)--;
      continue;
    }
    // append to L[from] vector a to node
    append(*L, *C, capacity, from, to);
  }

  fclose(fin);
  return 0;
}

/**
 * @brief read a directed graph from a file
 * @param filename [char const *] input file's name
 * @param R [unsigned int ***] reverse transition sparse matrix
 * @param RC [unsigned int **] output edges coming to a node
 * @param LC [unsigned int **] output edges exiting a node
 * @param N [unsigned int *] number of nodes
 * @param E [unsinged int *] number of edges
 * @return if operation is successful or not
 */
int read_graph_reverse(char const * filename, uint*** R, uint** RC, uint** LC, uint* N, uint* E)
{
  if (capacity) free((void*)capacity);
  capacity = NULL;
  // open input file
  FILE *fin;
  fin = fopen(filename, "r");
  if (fin == NULL)
  {
    printf("Error opening input file: %s\n", filename);
    exit(1);
  }

  char line[1000];

  uint e = 1;
  while (!feof(fin) && e > 0)
  {
    fgets(line, sizeof(line), fin);
    // ignore sentences starting from #
    if (strncmp(line, "#", 1) == 0)
    {
      // read number of nodes and edges
      if (strstr(line, "Nodes") != NULL)
      {
        sscanf(line, "# Nodes: %u Edges: %u\n", N, E);
        e = *E;
        capacity = (uint*) malloc((*N) * sizeof(uint));
        if (capacity == NULL) exit(-1);
        *R = (uint**) malloc((*N) * sizeof(uint*));
        if (*R == NULL) exit(-1);
        *RC = (uint*) malloc((*N) * sizeof(uint));
        if (*RC == NULL) exit(-1);
        *LC = (uint*) malloc((*N) * sizeof(uint));
        if (*LC == NULL) exit(-1);
        uint i;
        for (i = 0; i < *N; ++i)
        {
          RC[0][i] = 0.0;
          LC[0][i] = 0.0;
          capacity[i] = DFL_CAPACITY;
          R[0][i] = (uint*) malloc(DFL_CAPACITY * sizeof(uint));
          if (R[0][i] == NULL) exit(-1);
        }
      }
      continue;
    }
    // read an edge (from and to nodes)
    uint from, to;
    sscanf(line, "%u %u\n", &from, &to);
    e--;
    // condition to ensure the removal of self-transition
    if (from == to)
    {
      (*E)--;
      continue;
    }
    // append to L[from] vector a to node
    LC[0][from] += 1;
    append(*R, *RC, capacity, to, from);
  }

  fclose(fin);
  return 0;
}
