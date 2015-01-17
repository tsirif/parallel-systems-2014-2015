VERSION = 0.03
OPTI = -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800
STD = -std=c99
WARN = -Wmissing-prototypes -Wall #-Winline
INC = -Idcmt/include
LIB = -Ldcmt/lib
CC = icc
CCFLAGS = $(OPTI) $(WARN) $(STD)
SSE2FLAGS = -msse2 -DHAVE_SSE2
WITH_SFMT = -DSFMT
WITH_DCMT = -DDCMT
WITH_TEST = -DTEST
WITH_OUTPUT = -DFILEOUT
WITH_COMPARE = -DCOMPARE
WITH_TIME = -DTIME


# .PHONY: clean

# SFMT.o: SFMT/SFMT.c
	# $(CC) $(CCFLAGS) $(SSE2FLAGS) -DSFMT_MEXP=607 $(INC) -c $<

# parallel-bitonic: bitonic.c SFMT.o
	# mpicc $(CCFLAGS) $(WITH_TEST) $(WITH_COMPARE) $(WITH_SFMT) $(WITH_OUTPUT) -o $@ bitonic.c SFMT.o
parallel-bitonic: bitonic.c
	mpicc $(CCFLAGS) $(WITH_TEST) $(WITH_COMPARE) $(WITH_DCMT) $(WITH_OUTPUT) -o $@ $< $(INC) $(LIB) -ldcmt

clean: parallel-bitonic
	rm -f *.o *~ *.a
