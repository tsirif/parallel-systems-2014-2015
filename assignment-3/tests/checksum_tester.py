#!/usr/bin/env python
# encoding: utf-8
import os
import hashlib
import time
from sys import argv

dim = int(argv[1])
runs = 10
table_filename = "table{0}x{0}.bin".format(dim)

def file_md5(filename):
    with open(filename, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# generate a table
gen_exit = os.system("bin/gen.out {0}".format(dim))
assert(gen_exit == 0)

# run the cpu and cuda binaries
duration = time.time()
cuda_exit = os.system(
    "bin/cuda-int_32.out {2} {0} {1}".format(dim, runs, table_filename))
duration = time.time() - duration
print("Cuda-Int-32 total binary time: "+str(duration))
assert (cuda_exit == 0)

duration = time.time()
cuda_exit = os.system(
    "bin/cuda-int_64.out {2} {0} {1}".format(dim, runs, table_filename))
duration = time.time() - duration
print("Cuda-Int-32 total binary time: "+str(duration))
assert (cuda_exit == 0)

duration = time.time()
cpu_exit = os.system(
    "bin/omp.out {2} {0} {1}"  .format(dim, runs, table_filename))
duration = time.time() - duration
print("OMP total binary time: "+str(duration))
assert (cpu_exit == 0)

# compare the result file
cuda_out_32_md5 = file_md5("cuda_2_results_32.bin")
cuda_out_64_md5 = file_md5("cuda_2_results_64.bin")
cpu_out_md5 = file_md5("omp-results.bin")

if cuda_out_32_md5 == cpu_out_md5:
    print("CUDA_INT_32 TEST SUCCESS!")
else:
    print("CUDA_INT_32 TEST FAILED!")

if cuda_out_64_md5 == cpu_out_md5:
    print("CUDA_INT_64 TEST SUCCESS!")
else:
    print("CUDA_INT_64 TEST FAILED!")
