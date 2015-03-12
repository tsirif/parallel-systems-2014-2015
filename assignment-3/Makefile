OS_SIZE    = $(shell uname -m | sed -e "s/x86_64/64/" -e "s/armv7l/32/" -e "s/aarch64/64/")
OS_ARCH    = $(shell uname -m)
ARCH_FLAGS =

# Take command line flags that override any of these settings
ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif
ifeq ($(ARMv7),1)
	OS_SIZE    = 32
	OS_ARCH    = armv7l
	ARCH_FLAGS = -target-cpu-arch ARM
endif
ifeq ($(aarch64),1)
	OS_SIZE    = 64
	OS_ARCH    = aarch64
	ARCH_FLAGS = -target-cpu-arch ARM
endif

# Gencode arguments
ifeq ($(OS_ARCH),armv7l)
SMS ?= 20 30 32 35 50
else
SMS ?= 11 20 30 35 50
endif

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified <<<)
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

#check if gcc-4.6 and g++-4.6 exist
EXISTS = $(shell gcc-4.6 --version> /dev/null 2>&1 ; echo $$? )
EXISTS += $(shell g++-4.6 --version> /dev/null 2>&1 ; echo $$? )
ifeq ($(EXISTS), 0 0)
#both --version commands return 0. We will use 4.6 gcc/g++ for better compability
NVIDIA_GCC = gcc-4.6
NVIDIA_GPP = g++-4.6
else
NVIDIA_GCC = gcc
NVIDIA_GPP = g++
endif

CC = gcc
NVCC = nvcc

OPTI = -m$(OS_SIZE) -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800
STD = -std=c99
WARN = -Wextra -Wall -Wpointer-arith -Wformat -Wfloat-equal -Winit-self
EXTRA_WARN = -Wcast-qual -Wwrite-strings -pedantic -Wshadow -Wstrict-prototypes -Wundef -Wunreachable-code

#flags for gcc
CCFLAGS = $(OPTI) $(WARN) $(STD) $(EXTRA_WARN)
#flags for g++
PPFLAGS = $(OPTI) $(WARN)
#flags for nvcc
NVCCFLAGS = -m${OS_SIZE} ${ARCH_FLAGS} -O3 --compiler-options "$(PPFLAGS)" --compiler-bindir "$(NVIDIA_GCC)"
NVCCDBG = -m${OS_SIZE} ${ARCH_FLAGS} -g -G -pg -O0 -lineinfo -DTEST -DPRINT --compiler-options "$(WARN)" --compiler-bindir "$(NVIDIA_GCC)"

#the name of the cuda source file / target executable
TARGET = cuda
CPU_TARGET = omp
#binary that generates a NxN table .bin file
GEN = gen
BIN_DIR = bin
TMP_DIR = tmp
CUDA_DIR = cuda
CPU_DIR = cpu
UTILS_DIR = utils
TESTS_DIR = tests

ifeq ($(N),)
N = 1000
endif

.PHONY: clean dbg all clear run debug

# $^ is all dependancies (right of ':')
# $< is 1st dependancy
# $@ is left of ':'

all: $(BIN_DIR) $(BIN_DIR)/$(TARGET).out $(BIN_DIR)/$(GEN).out $(BIN_DIR)/$(CPU_TARGET).out
$(BIN_DIR):
	mkdir $(BIN_DIR)
$(BIN_DIR)/$(TARGET).out: $(BIN_DIR)/generic_functions.o $(BIN_DIR)/$(TARGET).o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ $^
$(BIN_DIR)/$(TARGET).o: $(CUDA_DIR)/$(TARGET).cu $(UTILS_DIR)/utils.h
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
$(BIN_DIR)/generic_functions.o : $(UTILS_DIR)/generic_functions.c
	$(NVIDIA_GPP) $(PPFLAGS) -o $@ -c $^

$(BIN_DIR)/cc_generic_functions.o: $(UTILS_DIR)/generic_functions.c
	$(CC) $(CCFLAGS) -o $@ -c $^
$(BIN_DIR)/generate_table.o: $(UTILS_DIR)/generate_table.c
	$(CC) $(CCFLAGS) -o $@ -c $^
$(BIN_DIR)/$(GEN).out: $(BIN_DIR)/generate_table.o $(BIN_DIR)/cc_generic_functions.o
	$(CC) $(CCFLAGS) -o $@ $^
$(BIN_DIR)/$(CPU_TARGET).o: $(CPU_DIR)/$(CPU_TARGET).c
	$(CC) $(CCFLAGS) -fopenmp -o $@ -c $^
$(BIN_DIR)/$(CPU_TARGET).out: $(BIN_DIR)/cc_generic_functions.o $(BIN_DIR)/$(CPU_TARGET).o
	$(CC) $(CCFLAGS) -fopenmp -o $@ $^

test: all
	rm -rf $(TMP_DIR)
	mkdir $(TMP_DIR)
# we allow the script to fail so the dir is cleaned no matter what
	-$(TESTS_DIR)/checksum_tester.py $(N)
	rm -f *.bin
run: all
	./$(GEN).out $(N)
	./$(TARGET).out table$(N)x$(N).$(BIN_DIR) $(N) $(T)
clear:
	rm -f *.o *.$(BIN_DIR) *.out
	rm -rf $(BIN_DIR)/ $(TMP_DIR)/
dbg: $(BIN_DIR) $(BIN_DIR)/generic_functions.o $(BIN_DIR)/$(TARGET).o
	$(NVCC) $(NVCCDBG) $(GENCODE_FLAGS) -o $(BIN_DIR)/$(TARGET)_dbg.out $(BIN_DIR)/generic_functions.o $(BIN_DIR)/cuda.o
clean: clear
debug: dbg
