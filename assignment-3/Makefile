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

DEV_INFO = devInfo

ifeq ($(wildcard $(DEV_INFO).out), )
$(shell nvcc $(DEV_INFO).cu -o $(DEV_INFO).out)
endif

SMS = $(shell ./$(DEV_INFO).out)

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

NVCC_VERSION = $(shell nvcc --version | grep 'release' | cut -d' ' -f5 | sed 's/[^0-9]//g')
$(info $(NVIDIA_GCC))
$(info $(NVCC_VERSION))

ifeq ($(shell test $(NVCC_VERSION) -gt 60; echo $$?),0)
CUDA_VER_DEFINE = -DCUDA_65
else
CUDA_VER_DEFINE = 
endif

OPTI = -m$(OS_SIZE) -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800 $(CUDA_VER_DEFINE)
STD = -std=c99
WARN = -Wextra -Wall -Wpointer-arith -Wformat -Wfloat-equal -Winit-self
EXTRA_WARN = -Wcast-qual -Wwrite-strings -pedantic -Wshadow -Wstrict-prototypes -Wundef -Wunreachable-code
DBG = -DPRINT -DTEST
SINGLE_PRECISION = -DSINGLE
DOUBLE_PRECISION = -DDOUBLE
#flags for gcc
CCFLAGS = $(OPTI) $(WARN) $(STD) $(EXTRA_WARN)
#flags for g++
PPFLAGS = $(OPTI) $(WARN)
#flags for nvcc
NVCCFLAGS = -m${OS_SIZE} ${ARCH_FLAGS} -lm -O2 --compiler-options "$(PPFLAGS)" --compiler-bindir "$(NVIDIA_GCC)"
NVCCDBG = -m${OS_SIZE} ${ARCH_FLAGS} -lm -g -G -pg -O0 -lineinfo $(DBG) --compiler-options "$(WARN)" --compiler-bindir "$(NVIDIA_GCC)"

#the name of the cuda source file / target executable
TARGET = cuda-int
CPU_TARGET = omp
#binary that generates a NxN table .bin file
GEN = gen
BIN_DIR = bin
TMP_DIR = tmp
CUDA_DIR = cuda
CPU_DIR = cpu
UTILS_DIR = utils
TESTS_DIR = tests
UTILS_C = utils

ifeq ($(N),)
N = 1000
endif

.PHONY: clean dbg all clear run debug

# $^ is all dependancies (right of ':')
# $< is 1st dependancy
# $@ is left of ':'

all: $(BIN_DIR) $(BIN_DIR)/$(TARGET)_32.out $(BIN_DIR)/$(TARGET)_64.out $(BIN_DIR)/$(GEN).out $(BIN_DIR)/$(CPU_TARGET).out

$(BIN_DIR):
	mkdir -p $(BIN_DIR)
dbg: $(BIN_DIR) $(BIN_DIR)/$(TARGET)_dbg_32.out $(BIN_DIR)/$(TARGET)_dbg_64.out $(BIN_DIR)/$(GEN).out $(BIN_DIR)/$(CPU_TARGET)_dbg.out

$(BIN_DIR)/$(TARGET)_32.out: $(BIN_DIR)/$(TARGET)_32.o $(BIN_DIR)/$(UTILS_C)_cpp.o
	$(NVCC) $(NVCCFLAGS) $(SINGLE_PRECISION) $(GENCODE_FLAGS) -o $@ $^
$(BIN_DIR)/$(TARGET)_32.o: $(CUDA_DIR)/$(TARGET).cu $(UTILS_DIR)/utils.h
	$(NVCC) $(NVCCFLAGS) $(SINGLE_PRECISION) $(GENCODE_FLAGS) -o $@ -c $<

$(BIN_DIR)/$(TARGET)_dbg_32.out: $(BIN_DIR)/$(TARGET)_dbg_32.o $(BIN_DIR)/$(UTILS_C)_cpp.o
	$(NVCC) $(NVCCDBG) $(SINGLE_PRECISION) $(GENCODE_FLAGS) -o $@ $^
$(BIN_DIR)/$(TARGET)_dbg_32.o: $(CUDA_DIR)/$(TARGET).cu $(UTILS_DIR)/utils.h
	$(NVCC) $(NVCCDBG) $(SINGLE_PRECISION) $(GENCODE_FLAGS) -o $@ -c $<

$(BIN_DIR)/$(TARGET)_64.out: $(BIN_DIR)/$(TARGET)_64.o $(BIN_DIR)/$(UTILS_C)_cpp.o
	$(NVCC) $(NVCCFLAGS) $(DOUBLE_PRECISION) $(GENCODE_FLAGS) -o $@ $^
$(BIN_DIR)/$(TARGET)_64.o: $(CUDA_DIR)/$(TARGET).cu $(UTILS_DIR)/utils.h
	$(NVCC) $(NVCCFLAGS) $(DOUBLE_PRECISION) $(GENCODE_FLAGS) -o $@ -c $<

$(BIN_DIR)/$(TARGET)_dbg_64.out: $(BIN_DIR)/$(TARGET)_dbg_64.o $(BIN_DIR)/$(UTILS_C)_cpp.o
	$(NVCC) $(NVCCDBG) $(DOUBLE_PRECISION) $(GENCODE_FLAGS) -o $@ $^
$(BIN_DIR)/$(TARGET)_dbg_64.o: $(CUDA_DIR)/$(TARGET).cu $(UTILS_DIR)/utils.h
	$(NVCC) $(NVCCDBG) $(DOUBLE_PRECISION) $(GENCODE_FLAGS) -o $@ -c $<

$(BIN_DIR)/generate_table.o: $(UTILS_DIR)/generate_table.c
	$(CC) $(CCFLAGS) -o $@ -c $^
$(BIN_DIR)/$(GEN).out: $(BIN_DIR)/generate_table.o $(BIN_DIR)/$(UTILS_C).o
	$(CC) $(CCFLAGS) -o $@ $^

$(BIN_DIR)/$(CPU_TARGET).o: $(CPU_DIR)/$(CPU_TARGET).c
	$(CC) $(CCFLAGS) -fopenmp -o $@ -c $^
$(BIN_DIR)/$(CPU_TARGET).out: $(BIN_DIR)/$(CPU_TARGET).o $(BIN_DIR)/$(UTILS_C).o
	$(CC) $(CCFLAGS) -fopenmp -o $@ $^

$(BIN_DIR)/$(CPU_TARGET)_dbg.o: $(CPU_DIR)/$(CPU_TARGET).c
	$(CC) $(CCFLAGS) -g $(DBG) -fopenmp -o $@ -c $^
$(BIN_DIR)/$(CPU_TARGET)_dbg.out: $(BIN_DIR)/$(CPU_TARGET)_dbg.o $(BIN_DIR)/$(UTILS_C).o
	$(CC) $(CCFLAGS) -g $(DBG) -fopenmp -o $@ $^

$(BIN_DIR)/$(UTILS_C).o : $(UTILS_DIR)/$(UTILS_C).c
	$(CC) $(CCFLAGS) -o $@ -c $^
$(BIN_DIR)/$(UTILS_C)_cpp.o : $(UTILS_DIR)/$(UTILS_C).c
	$(NVIDIA_GPP) $(PPFLAGS) -o $@ -c $^

test: all
	rm -rf $(TMP_DIR)
	mkdir $(TMP_DIR)
# we allow the script to fail so the dir is cleaned no matter what
	-$(TESTS_DIR)/checksum_tester.py $(N)
	rm -f *.bin
run: all
	./$(BIN_DIR)/$(GEN).out $(N)
	./$(BIN_DIR)/$(CPU_TARGET).out table$(N)x$(N).bin $(N) $(T)
	./$(BIN_DIR)/$(TARGET)_32.out table$(N)x$(N).bin $(N) $(T)
	./$(BIN_DIR)/$(TARGET)_64.out table$(N)x$(N).bin $(N) $(T)
clear:
	rm -f *.o *.$(BIN_DIR) *.out *.html
	rm -rf $(BIN_DIR)/ $(TMP_DIR)/
clean: clear
debug: dbg
