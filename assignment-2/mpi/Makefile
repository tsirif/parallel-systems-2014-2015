VERSION = 0.03
OPTI = -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800
STD = -std=c99
WARN = -Wmissing-prototypes -Wall #-Winline
INC = -ISFMT
CC = gcc
CCFLAGS = $(OPTI) $(WARN) $(STD)
SSE2FLAGS = -msse2 -DHAVE_SSE2
TEST = -DTEST
WITH_SFMT = -DSFMT

.PHONY: clean

SFMT.o: SFMT/SFMT.c
	$(CC) $(CCFLAGS) $(SSE2FLAGS) -DSFMT_MEXP=607 $(INC) -c $<

parallel-bitonic: bitonic.c SFMT.o
	mpicc $(CCFLAGS) $(TEST) $(SFMT) -O3 -o $@ bitonic.c SFMT.o

clean: parallel-bitonic
	rm -f *.o *~ *.a
