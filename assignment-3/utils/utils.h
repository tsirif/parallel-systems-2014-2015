/* swap 2 int* pointers */
inline void swap(int **a, int **b)
{
    int *t;
    t = *a;
    *a = *b;
    *b = t;
}

#define DFL_RUNS 10 /* the number of iterations if not else specified */

/* Position of i-th row j-th element using our current data arrangement. */
#define POS(i, j) (i*N + j)

void read_from_file(int *X, char *filename, int N);
void save_table(int *X, int N, const char *filename);
void generate_table(int *X, int N);
void print_table(int *A, int N);
