VERSION = 0.03
OPTI = -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800
STD = -std=c99
WARN = -Wmissing-prototypes -Wall #-Winline
CC = gcc
CCFLAGS = $(OPTI) $(WARN) $(STD)

TARGET = omp
GEN = gen

all:
	$(CC) $(CCFLAGS) -fopenmp -o $(TARGET).out omp.c
	$(CC) $(CCFLAGS) -o $(GEN).out ../init/generate_table.c
test: test.c omp.c
	if [ -a $(TARGET).out ] ; \
	then \
		rm $(TARGET).out ; \
	fi;
	$(CC) $(CCFLAGS) -DTEST -DPRINT -fopenmp -o $(TARGET).out omp.c
	# $(CC) $(CCFLAGS) -DTEST -DPRINT -fopenmp -o $(TARGET).out omp.c
	$(CC) $(CCFLAGS) -o $@.out test.c
run:
	./$(GEN).out $(N)
	./$(TARGET).out table$(N)x$(N).bin $(N) $(T)
clear:
	rm -f *.bin $(GEN).out $(TARGET).out test
