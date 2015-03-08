/* define colors */
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

/* swap 2 int* pointers */
inline void swap(int** a, int** b)
{
    int *t;
    t = *a;
    *a = *b;
    *b = t;
}

#define DFL_RUNS 10 /* the number of iterations if not else specified */

/* Position of i-th row j-th element using our current data arrangement. */
#define POS(i, j) (i*N + j)
