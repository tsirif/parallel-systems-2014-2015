#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

inline void save_table(int *X, int N);
inline void read_from_file(int *X, const char *filename, int N);
void test_1();
void test_2();
void test_3();
void test_4();

int main()
{
    test_1();
    test_2();
    test_3();
    test_4();
    return 0;
}

inline void save_table(int *X, int N)
{
    FILE *fp;
    char filename[20];
    sprintf(filename, "test.bin");
    fp = fopen(filename, "w+");
    fwrite(X, sizeof(int), N * N, fp);
    fclose(fp);
}

inline void read_from_file(int *X, const char *filename, int N)
{
    FILE *fp = fopen(filename, "r+");
    fread(X, sizeof(int), N * N, fp);
    fclose(fp);
}

void test_1()
{
    printf("Test-1\n\n");
    int i, j;
    int *X = (int *) calloc(5, 5 * sizeof(int));
    int *Y = (int *) calloc(5, 5 * sizeof(int));

    // write the test case
    for (i = 0; i < 5; ++i) {
        for (j = 0; j < 5; ++j)
            X[i * 5 + j] = 0;
    }

    X[2 * 5 + 1] = 1;
    X[2 * 5 + 2] = 1;
    X[2 * 5 + 3] = 1;
    save_table(X, 5);
    // invoke test
    int ret = 0, status = 0;
    pid_t pid;
    char *cmd[] = {"./omp.out", "test.bin", "5", "10", (char *)0};

    if ((pid = fork()) == 0)
        ret = execv("./omp.out", cmd);
    else if (pid < 0)
        ret = -1;
    else {
        do {
            waitpid(pid, &status, WUNTRACED);
        } while (!WIFEXITED(status) && !WIFSIGNALED(status));
    }

    if (ret != 0) {
        printf("Error while executing functional test!\n");
        exit(1);
    }

    // write expectations
    // it's the same!
    // check results
    read_from_file(Y, "results.bin", 5);
    int fail = 0;

    for (i = 0; i < 5; ++i) {
        for (j = 0; j < 5; ++j) {
            if (X[i * 5 + j] != Y[i * 5 + j]) {
                fail = 1;
                break;
            }
        }
    }

    if (fail)
        printf("Test-1 failed!\n");
    else
        printf("Test-1 succeded!\n");

    free(X);
    free(Y);
}

void test_2()
{
    printf("Test-2\n\n");
    int i, j;
    int *X = (int *) calloc(5, 5 * sizeof(int));
    int *Y = (int *) calloc(5, 5 * sizeof(int));

    // write the test case
    for (i = 0; i < 5; ++i) {
        for (j = 0; j < 5; ++j)
            X[i * 5 + j] = 0;
    }

    X[2 * 5 + 1] = 1;
    X[2 * 5 + 2] = 1;
    X[2 * 5 + 3] = 1;
    save_table(X, 5);
    // invoke test
    int ret = 0, status = 0;
    pid_t pid;
    char *cmd[] = {"./omp.out", "test.bin", "5", "9", (char *)0};

    if ((pid = fork()) == 0)
        ret = execv("./omp.out", cmd);
    else if (pid < 0)
        ret = -1;
    else {
        do {
            waitpid(pid, &status, WUNTRACED);
        } while (!WIFEXITED(status) && !WIFSIGNALED(status));
    }

    if (ret != 0) {
        printf("Error while executing functional test!\n");
        exit(1);
    }

    // write expectations
    for (i = 0; i < 5; ++i) {
        for (j = 0; j < 5; ++j)
            X[i * 5 + j] = 0;
    }

    X[1 * 5 + 2] = 1;
    X[2 * 5 + 2] = 1;
    X[3 * 5 + 2] = 1;
    // check results
    read_from_file(Y, "results.bin", 5);
    int fail = 0;

    for (i = 0; i < 5; ++i) {
        for (j = 0; j < 5; ++j) {
            if (X[i * 5 + j] != Y[i * 5 + j]) {
                fail = 1;
                break;
            }
        }
    }

    if (fail)
        printf("Test-2 failed!\n");
    else
        printf("Test-2 succeded!\n");

    free(X);
    free(Y);
}

void test_3()
{
    printf("Test-3\n\n");
    int i, j;
    int *X = (int *) calloc(5, 5 * sizeof(int));
    int *Y = (int *) calloc(5, 5 * sizeof(int));

    // write the test case
    for (i = 0; i < 5; ++i) {
        for (j = 0; j < 5; ++j)
            X[i * 5 + j] = 0;
    }

    X[1 * 5 + 1] = 1;
    X[1 * 5 + 2] = 1;
    X[2 * 5 + 1] = 1;
    X[2 * 5 + 2] = 1;
    save_table(X, 5);
    // invoke test
    int ret = 0, status = 0;
    pid_t pid;
    char *cmd[] = {"./omp.out", "test.bin", "5", "2", (char *)0};

    if ((pid = fork()) == 0)
        ret = execv("./omp.out", cmd);
    else if (pid < 0)
        ret = -1;
    else {
        do {
            waitpid(pid, &status, WUNTRACED);
        } while (!WIFEXITED(status) && !WIFSIGNALED(status));
    }

    if (ret != 0) {
        printf("Error while executing functional test!\n");
        exit(1);
    }

    // write expectations
    // but it's the same!
    // check results
    read_from_file(Y, "results.bin", 5);
    int fail = 0;

    for (i = 0; i < 5; ++i) {
        for (j = 0; j < 5; ++j) {
            if (X[i * 5 + j] != Y[i * 5 + j]) {
                fail = 1;
                break;
            }
        }
    }

    if (fail)
        printf("Test-3a failed!\n");
    else
        printf("Test-3a succeded!\n");

    char *cmd2[] = {"./omp.out", "test.bin", "5", "11", (char *)0};

    if ((pid = fork()) == 0)
        ret = execv("./omp.out", cmd2);
    else if (pid < 0)
        ret = -1;
    else {
        do {
            waitpid(pid, &status, WUNTRACED);
        } while (!WIFEXITED(status) && !WIFSIGNALED(status));
    }

    if (ret != 0) {
        printf("Error while executing functional test!\n");
        exit(1);
    }

    // write expectations
    // but it's the same!
    // check results
    read_from_file(Y, "results.bin", 5);
    fail = 0;

    for (i = 0; i < 5; ++i) {
        for (j = 0; j < 5; ++j) {
            if (X[i * 5 + j] != Y[i * 5 + j]) {
                fail = 1;
                break;
            }
        }
    }

    if (fail)
        printf("Test-3b failed!\n");
    else
        printf("Test-3b succeded!\n");

    free(X);
    free(Y);
}

void test_4()
{
    printf("Test-4\n\n");
    int i, j;
    int *X = (int *) calloc(5, 5 * sizeof(int));
    int *Y = (int *) calloc(5, 5 * sizeof(int));

    // write the test case
    for (i = 0; i < 5; ++i) {
        for (j = 0; j < 5; ++j)
            X[i * 5 + j] = 0;
    }

    X[0 * 5 + 0] = 1;
    X[4 * 5 + 0] = 1;
    X[0 * 5 + 4] = 1;
    save_table(X, 5);
    // invoke test
    int ret = 0, status = 0;
    pid_t pid;
    char *cmd[] = {"./omp.out", "test.bin", "5", "1", (char *)0};

    if ((pid = fork()) == 0)
        ret = execv("./omp.out", cmd);
    else if (pid < 0)
        ret = -1;
    else {
        do {
            waitpid(pid, &status, WUNTRACED);
        } while (!WIFEXITED(status) && !WIFSIGNALED(status));
    }

    if (ret != 0) {
        printf("Error while executing functional test!\n");
        exit(1);
    }

    // write expectations
    X[4 * 5 + 4] = 1;
    // check results
    read_from_file(Y, "results.bin", 5);
    int fail = 0;

    for (i = 0; i < 5; ++i) {
        for (j = 0; j < 5; ++j) {
            if (X[i * 5 + j] != Y[i * 5 + j]) {
                fail = 1;
                break;
            }
        }
    }

    if (fail)
        printf("Test-4 failed!\n");
    else
        printf("Test-4 succeded!\n");

    free(X);
    free(Y);
}
