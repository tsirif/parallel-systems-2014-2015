GCC=gcc -O4

VERSION=0.01

all:
	$(GCC) test_octree.c datasets.c general_functions.c hash_codes.c morton_encoding.c radix_sort.c data_rearrangement.c verification.c -lm -o test_octree.out

clean:
	rm -f *~ *.o test_octree.out 

release:
	tar -cvf octree-code-$(VERSION).tar *.c *.h Makefile
