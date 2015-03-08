VERSION = 0.03
OPTI = -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800
STD = -std=c99
WARN = -Wmissing-prototypes -Wall #-Winline
SM_ARCH = sm_50
CC = gcc
NVCC = nvcc
CCFLAGS = $(OPTI) $(WARN) $(STD)
NVCCFLAGS = -arch=$(SM_ARCH) -O3
TARGET = cuda
GEN = gen

all:
	$(NVCC) $(NVCCFLAGS) -o $(TARGET).out cuda.cu
	$(CC) $(CCFLAGS) -o $(GEN).out ../utils/generate_table.c
run:
	./$(GEN).out $(N)
	./$(TARGET).out table$(N)x$(N).bin $(N) $(T)
clear:
	rm -f *.bin $(GEN).out $(TARGET).out test.out
test:
	$(NVCC) $(NVCCFLAGS) -g -G -DTEST -DPRINT -o $(TARGET).out cuda.cu
