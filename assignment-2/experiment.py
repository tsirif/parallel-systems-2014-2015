from subprocess import call
import sys

#from: http://it.auth.gr/en/node/1548
script_src=r'''#!/bin/bash
#PBS -q auth
#PBS -N {0}
#PBS -j oe
#PBS -l nodes={1}:ppn={2}
#PBS -l walltime=00:0{6}:00
cd $PBS_O_WORKDIR
export NP=$(cat $PBS_NODEFILE | wc -l)
echo Master process running on `hostname`
echo Directory is `pwd`
echo PBS has allocated the following nodes:
echo `cat $PBS_NBODEFILE`
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS nodes
args='{5} {4} {3}'
echo qpp is {4}
echo nodes is {1}
echo ppn is {2}
echo threadspp is {3}
export I2G_MPI_TYPE=mpich2
export I2G_MPI_APPLICATION={0}
export I2G_MPI_APPLICATION_ARGS=$args
$I2G_MPI_START'''

# nodes = [1, 2, 4]
# cores = [1,2,4,8,16, 32, 64]
# Q = range(16, 21)
# src = r'/mnt/scratchdir/home/orestisf/run_tests/'
# os.chdir(src)

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
    script = script_src.format(executable, 2**nodes, 2**ranks_per_node, threads,
                               2**size_per_rank, total_ranks, time)
    filename = "script-{0}-{1}-{2}-{3}.sh".format(nodes, total_ranks,
                                                  threads, total_size)
    with open(filename, "w") as f:
        f.write(script)
    call("qsub " + filename, shell=True)
    print "qsub " + filename

if __name__ == '__main__':
    argv = sys.argv[1:]
    make_script(argv[0], argv[1], argv[2], argv[3], argv[4])

