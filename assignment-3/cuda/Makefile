VERSION = 0.03
OPTI = -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800
STD = -std=c99
WARN = -Wmissing-prototypes -Wall #-Winline
SM_ARCH = sm_11
CC = gcc
NVCC = nvcc
CCFLAGS = $(OPTI) $(WARN) $(STD)
NVCCFLAGS = -arch=$(SM_ARCH) -g -G
TARGET = cuda
GEN = gen

all:
	$(NVCC) $(NVCCFLAGS) -o $(TARGET).out cuda.cu
	$(CC) $(CCFLAGS) -o $(GEN).out ../init/generate_table.c
test: test.c cuda.cu
	$(NVCC) $(NVCCFLAGS) -DPRINT -DTEST -o $(TARGET).out cuda.cu
	$(CC) $(CCFLAGS) -o $@ test.c
run:
	./$(GEN).out $(N)
	./$(TARGET).out table$(N)x$(N).bin $(N) $(T)
clear:
	rm -f *.bin $(GEN).out $(TARGET).out test
