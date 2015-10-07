#include <stdint.h>

typedef unsigned int uint;

#define DFL_CAPACITY 10;
extern uint* capacity;

void append(uint** L, uint* C, uint index, uint value);
void reverse(uint** L, uint* LC, uint N, uint** R, uint* RC);

int read_graph(char const * filename, uint** L, uint* C, uint* N, uint* E);
int read_graph_reverse(char const * filename, uint** R, uint* C, uint* N, uint* E);
