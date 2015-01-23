VERSION = 0.03
OPTI = -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800
STD = -std=c99
WARN = -Wmissing-prototypes -Wall #-Winline
INC = -Idcmt/include
LIB = -Ldcmt/lib
CC = icc
CCFLAGS = $(OPTI) $(WARN) $(STD)
# SSE2FLAGS = -msse2 -DHAVE_SSE2
# WITH_SFMT = -DSFMT
WITH_DCMT = -DDCMT
WITH_TEST = -DTEST
WITH_OUTPUT = -DFILEOUT
WITH_COMPARE = -DCOMPARE
WITH_TIME = -DTIME
WITH_GRAMA = -DGRAMA
WITH_ALL = $(WITH_TIME) $(WITH_DCMT) $(WITH_COMPARE)
WITH_ALL_GRAMA = $(WITH_ALL) $(WITH_GRAMA)


# .PHONY: clean

# SFMT.o: SFMT/SFMT.c
	# $(CC) $(CCFLAGS) $(SSE2FLAGS) -DSFMT_MEXP=607 $(INC) -c $<

hybrid-bitonic: bitonic.c grama_quicksort.c
	mpicc $(CCFLAGS) -openmp $(WITH_ALL_GRAMA) $(WITH_TEST) -o ../$@ bitonic.c grama_quicksort.c $(INC) $(LIB) -ldcmt

mpi-bitonic: bitonic.c
	mpicc $(CCFLAGS) -openmp $(WITH_ALL) $(WITH_TEST) -o ../$@ bitonic.c $(INC) $(LIB) -ldcmt

clean:
	rm -f *.o *~ *.a
