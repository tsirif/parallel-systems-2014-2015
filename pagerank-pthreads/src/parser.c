#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "pagerank-pthreads/utils.h"

/**
 * @brief read a directed graph from a file
 * @param filename [char const *] input file's name
 * @param L [unsigned int **] transition sparse matrix
 * @param C [unsigned int *] output edges per node array
 * @param N [unsigned int *] number of nodes
 * @param E [unsinged int *] number of edges
 * @return if operation is successful or not
 */
int read_graph(char const * filename, uint** L, uint* C, uint* N, uint* E)
{
  if (capacity) free((void*)capacity);
  capacity = NULL;
  // open input file
  FILE *fin;
  fin = fopen(filename, "r");
  if (fin == NULL)
  {
    printf("Error opening input file: %s\n", filename);
    exit(0);
  }

  char line[1000];

  while (!feof(fin))
  {
    fgets(line, sizeof(line), fin);
    // ignore sentences starting from #
    if (strncmp(line, "#", 1) == 0)
    {
      // read number of nodes and edges
      if (strstr(line, "Nodes") != NULL)
      {
        sscanf(line, "# Nodes: %u Edges: %u\n", N, E);
        capacity = (uint*) malloc((*N) * sizeof(uint));
        L = (uint**) malloc((*N) * sizeof(uint*));
        C = (uint*) calloc(0, (*N) * sizeof(uint));
        for (uint i = 0; i < *N; i++)
        {
          capacity[i] = DFL_CAPACITY;
          L[i] = (uint*) malloc(DFL_CAPACITY * sizeof(uint));
        }
      }
      continue;
    }
    // read an edge (from and to nodes)
    uint from, to;
    sscanf(line, "%u %u\n", &from, &to);
    // condition to ensure the removal of self-transition
    if (from == to)
    {
      (*E)--;
      continue;
    }
    // append to L[from] vector a to node
    append(L, C, from, to);
  }
  return 0;
}

/**
 * @brief read a directed graph from a file
 * @param filename [char const *] input file's name
 * @param R [unsigned int **] reverse transition sparse matrix
 * @param RC [unsigned int *] output edges coming to a node
 * @param LC [unsigned int *] output edges exiting a node
 * @param N [unsigned int *] number of nodes
 * @param E [unsinged int *] number of edges
 * @return if operation is successful or not
 */
int read_graph_reverse(char const * filename, uint** R, uint* RC, uint* LC, uint* N, uint* E)
{
  if (capacity) free((void*)capacity);
  capacity = NULL;
  // open input file
  FILE *fin;
  fin = fopen(filename, "r");
  if (fin == NULL)
  {
    printf("Error opening input file: %s\n", filename);
    exit(0);
  }

  char line[1000];

  while (!feof(fin))
  {
    fgets(line, sizeof(line), fin);
    // ignore sentences starting from #
    if (strncmp(line, "#", 1) == 0)
    {
      // read number of nodes and edges
      if (strstr(line, "Nodes") != NULL)
      {
        sscanf(line, "# Nodes: %u Edges: %u\n", N, E);
        capacity = (uint*) malloc((*N) * sizeof(uint));
        R = (uint**) malloc((*N) * sizeof(uint*));
        RC = (uint*) calloc(0, (*N) * sizeof(uint));
        LC = (uint*) calloc(0, (*N) * sizeof(uint));
        for (uint i = 0; i < *N; i++)
        {
          capacity[i] = DFL_CAPACITY;
          R[i] = (uint*) malloc(DFL_CAPACITY * sizeof(uint));
        }
      }
      continue;
    }
    // read an edge (from and to nodes)
    uint from, to;
    sscanf(line, "%u %u\n", &from, &to);
    // condition to ensure the removal of self-transition
    if (from == to)
    {
      (*E)--;
      continue;
    }
    // append to L[from] vector a to node
    LC[from] += 1;
    append(R, RC, to, from);
  }
  return 0;
}
