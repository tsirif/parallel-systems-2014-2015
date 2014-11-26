#GCC=gcc -O4 -std=gnu99
ICC=icc

VERSION=0.03

all:
	$(ICC) test_octree.c datasets.c general_functions.c hash_codes.c morton_encoding.c radix_sort.c data_rearrangement.c verification.c -lm -o test_octree.out

clean:
	rm -f *~ *.o test_octree 

release:
	tar -cvf octree-code-$(VERSION).tar *.c *.h *m *sh Makefile
