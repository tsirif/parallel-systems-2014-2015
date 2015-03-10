#!/usr/bin/env python
# encoding: utf-8
import os
import hashlib

dim = 1000
runs = 10
table_filename = "table{0}x{0}.bin".format(dim)

def file_md5(filename):
    with open(filename, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

#generate a table
gen_exit = os.system("bin/gen.out {0}".format(dim))
assert(gen_exit == 0)

#run the cpu and cuda binaries
cuda_exit = os.system("bin/cuda.out {2} {0} {1}".format(dim, runs, table_filename))
assert (cuda_exit == 0)
cpu_exit = os.system("bin/omp.out {2} {0} {1}"  .format(dim, runs, table_filename))
assert (cpu_exit == 0)

#compare the result file
cuda_out_md5 = file_md5("cuda-results.bin")
cpu_out_md5 = file_md5("omp-results.bin")

if cuda_out_md5 == cpu_out_md5:
    print("TEST SUCCESS!")
else:
    print("TEST FAILED!")
    exit(1)
