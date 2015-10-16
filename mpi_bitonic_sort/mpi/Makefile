VERSION = 0.03
OPTI = -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800
STD = -std=c99
WARN = -Wmissing-prototypes -Wall #-Winline
INC = -Idcmt/include
LIB = -Ldcmt/lib
CC = gcc
CCFLAGS = $(OPTI) $(WARN) $(STD)
WITH_DCMT = -DDCMT
WITH_TEST = -DTEST
WITH_OUTPUT = -DFILEOUT
WITH_COMPARE = -DCOMPARE
WITH_TIME = -DTIME
WITH_GRAMA = -DGRAMA
WITH_ALL = $(WITH_TIME) $(WITH_DCMT) $(WITH_COMPARE)
WITH_ALL_GRAMA = $(WITH_ALL) $(WITH_GRAMA)


.PHONY: clean

all: hybrid-bitonic mpi-bitonic

hybrid-bitonic: bitonic.c grama_quicksort.c dcmt/lib/libdcmt.a
	mpicc $(CCFLAGS) -openmp $(WITH_ALL_GRAMA) $(WITH_TEST) -o ../$@ bitonic.c grama_quicksort.c $(INC) $(LIB) -ldcmt

dcmt/lib/libdcmt.a:
	$(MAKE) -C dcmt/lib/

mpi-bitonic: bitonic.c
	mpicc $(CCFLAGS) -openmp $(WITH_ALL) $(WITH_TEST) -o ../$@ bitonic.c $(INC) $(LIB) -ldcmt

clean:
	rm -f *.o *~ *.a
	$(MAKE) -C dcmt/lib MAKEFLAGS=clean
