#!/usr/bin/env python
# encoding: utf-8

from subprocess32 import call
from itertools import product
import os
import sys

#from: http://it.auth.gr/en/node/1548
script_src=r'''#!/bin/bash
#PBS -q auth
#PBS -N {7}
#PBS -j oe
#PBS -l nodes={1}:ppn={2}
#PBS -l walltime=00:0{6}:00
cd $PBS_O_WORKDIR
export NP=$(cat $PBS_NODEFILE | wc -l)
echo Master process running on `hostname`
echo Directory is `pwd`
echo `cat $PBS_NBODEFILE`
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS processes
args='{5} {4} {3}'
echo qpp is {4}
echo nodes is {1}
echo ppn is {2}
echo threadspp is {3}
export I2G_MPI_TYPE=mpich2
export I2G_MPI_APPLICATION=../{0}
export I2G_MPI_APPLICATION_ARGS=$args
$I2G_MPI_START'''

local_script_src="mpirun -n {4} ../{0} {1} {2} {3} > {5}"

def run_experiments():
    os.chdir("./local-results")
    # nodes = range(3)
    # ranks = range(5)
    # threads = [2, 4, 8, 16]
    # q = range(16, 28)
    ranks = [2]
    threads = [4]
    q = range(28, 30)
    # for nnode, nrank, nthread, nq in product(nodes, ranks, threads, q):
    #     make_script("hybrid-bitonic", nnode, nnode+nrank, nq, nthread)
    # for nnode, nrank, nq in product(nodes, ranks, q):
    #     make_script("mpi-bitonic", nnode, nnode+nrank, nq, -1)
    # for nq in q:
    #     call("../serial-bitonic nq > serial-bitonic-"+str(nq)+".o", shell=True)
    # for nrank, nthread, nq in product(ranks, threads, q):
        # make_local_script("hybrid-bitonic", nrank, nq, nthread)
    # for nrank, nq in product(ranks, q):
        # make_local_script("mpi-bitonic", nrank, nq, -1)
    q = range(16, 28)
    for nq in q:
        call("../serial-bitonic "+str(nq)+" > serial-bitonic-"+str(nq)+".out", shell=True)

def make_script(executable, nodes, total_ranks, total_size, threads):
    if total_size < 25:
        time = 1
    elif total_size < 26:
        time = 2
    else:
        time = 5
    if threads == -1:
        threads = ''
    ranks_per_node = total_ranks - nodes
    size_per_rank = total_size - total_ranks
    result = executable+"-"+str(2**nodes)+"-"+str(2**total_ranks)
    if threads != '':
        result = result + "-"+str(threads)
    result = result +"-"+str(total_size)+".out"
    script = script_src.format(executable, 2**nodes, 2**ranks_per_node, threads,
                               size_per_rank, total_ranks, time, result)
    filename = "script-{0}-{1}-{2}-{3}.sh".format(nodes, 2**total_ranks,
                                                  threads, total_size)
    with open(filename, "w") as f:
        f.write(script)
    call("qsub " + filename, shell=True)
    print "qsub " + filename
    call("rm "+filename, shell=True)

def make_local_script(executable, total_ranks, total_size, threads):
    if threads == -1:
        threads = ''
    size_per_rank = total_size - total_ranks
    result = executable+"-"+str(2**total_ranks)
    if threads != '':
        result = result + "-"+str(threads)
    result = result +"-"+str(total_size)+".out"
    print result
    script = local_script_src.format(executable,
                                     total_ranks, size_per_rank, threads,
                                     2**total_ranks, result)
    call(script, shell=True)

if __name__ == '__main__':
    # argv = sys.argv[1:]
    # os.chdir("./results")
    # make_local_script(argv[0], int(argv[1]), int(argv[2]), int(argv[3]))
    run_experiments()

