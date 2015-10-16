#!/usr/bin/env python
# encoding: utf-8

import os
from collections import defaultdict
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

results = defaultdict(float)
executable = ["mpi-bitonic", "hybrid-bitonic", "serial-bitonic"]
ranks = range(0, 5)
threads = [2, 4, 8, 16]
q = range(16, 28)


def handle_results():
    os.chdir('./local-results')
    print "indexing mpi-bitonic results"
    for nranks, nq in product(ranks, q):
        filename = executable[0]+"-"+str(2**nranks)
        filename = filename +"-"+str(nq)+".out"
        print filename
        with open(filename, 'r') as f:
            for line in f:
                first = line.split()[0]
                last = line.split()[-1]
                if first == "mpi":
                    results[('mpi-bitonic', nranks, nq)] = float(last)
                if first == "serial":
                    results[('mpi-bitonic', nranks, nq)] /= float(last)
                    break

    print "indexing hybrid-bitonic results"
    for nranks, nq, nthread in product(ranks, q, threads):
        filename = executable[1]+"-"+str(2**nranks)+"-"+str(nthread)
        filename = filename +"-"+str(nq)+".out"
        print filename
        with open(filename, 'r') as f:
            for line in f:
                first = line.split()[0]
                last = line.split()[-1]
                if first == "hybrid":
                    results[('hybrid-bitonic', nranks, nthread, nq)] = float(last)
                if first == "serial":
                    results[('hybrid-bitonic', nranks, nthread, nq)] /= float(last)
                    break

    print "indexing serial-bitonic results"
    for nq in q:
        filename = executable[2]+"-"+str(nq)+".out"
        print filename
        with open(filename, 'r') as f:
            for line in f:
                first = line.split()[0]
                last = line.split()[-1]
                if first == "imperative":
                    results[('serial-bitonic', nq)] = float(last)
                if first == "serial":
                    results[('serial-bitonic', nq)] /= float(last)
                    break

class GraphMaker(object):
    def __init__(self, title, xlabel, bounds=True):
        fig, axes = plt.subplots(sharex=True, sharey=True)
        self.fig = fig
        self.axes = axes
        self.axes.set_title(title)
        self.axes.set_ylabel("wtime over quicksort's wtime")
        self.axes.set_xlabel(xlabel)
        if bounds:
            self.axes.set_ylim([0, 1.25])
        self.axes.autoscale_view()
        self.axes.grid(True)

    def add_plot(self, xdata, ydata, data_fmt, data_label):
        plt.plot(xdata, ydata, data_fmt, label=data_label)

    def show_figure(self):
        self.axes.legend()
        plt.show()

    def save_figure(self, filename):
        self.axes.legend()
        plt.savefig(filename, bbox_inches='tight')

def ranks_over_constq():
    # mpi-bitonic
    mpibit = GraphMaker("Perfomance of mpi bitonic", "#problem size (log2)")
    for nranks in ranks:
        ydata = [results[('mpi-bitonic', nranks, nq)] for nq in q]
        mpibit.add_plot(q, ydata, 'o-', "#procs:"+str(2**nranks))
    mpibit.save_figure("mpi-processes.png")

    # hybrid-bitonic
    hybbit = GraphMaker("Perfomance of hybrid bitonic (2 threads)", "#problem size (log2)")
    for nranks in ranks:
        ydata = [results[('hybrid-bitonic', nranks, 2, nq)] for nq in q]
        mpibit.add_plot(q, ydata, 'o-', "#procs:"+str(2**nranks))
    hybbit.save_figure("hybrid-processes-2threads.png")

    hybbit = GraphMaker("Perfomance of hybrid bitonic (4 threads)", "#problem size (log2)")
    for nranks in ranks:
        ydata = [results[('hybrid-bitonic', nranks, 4, nq)] for nq in q]
        mpibit.add_plot(q, ydata, 'o-', "#procs:"+str(2**nranks))
    hybbit.save_figure("hybrid-processes-4threads.png")

    hybbit = GraphMaker("Perfomance of hybrid bitonic (8 threads)", "#problem size (log2)")
    for nranks in ranks:
        ydata = [results[('hybrid-bitonic', nranks, 8, nq)] for nq in q]
        mpibit.add_plot(q, ydata, 'o-', "#procs:"+str(2**nranks))
    hybbit.save_figure("hybrid-processes-8threads.png")

    hybbit = GraphMaker("Perfomance of hybrid bitonic (16 threads)", "#problem size (log2)")
    for nranks in ranks:
        ydata = [results[('hybrid-bitonic', nranks, 16, nq)] for nq in q]
        mpibit.add_plot(q, ydata, 'o-', "#procs:"+str(2**nranks))
    hybbit.save_figure("hybrid-processes-16threads.png")

    # serial-bitonic
    serialbit = GraphMaker("Perfomance of serial bitonic", "total problem size (log2)", bounds=False)
    ydata = [results[('serial-bitonic', nq)] for nq in q]
    serialbit.add_plot(q, ydata, 'o-', "1 process")
    serialbit.save_figure("serial-quantity.png")

def compare_over_const():
    comprank21 = GraphMaker("Compare mpi with mpi/openmp bitonic (21q)", "#ranks (log2)")
    ydata = [results[('mpi-bitonic', nranks, 21)] for nranks in ranks]
    comprank21.add_plot(ranks, ydata, 'x-', "mpi")
    ydata = [results[('hybrid-bitonic', nranks-2, 4, 21)] for nranks in range(3,5)]
    comprank21.add_plot(range(3, 5), ydata, 'o-', "hybrid rank=proc+thread")
    ydata = [results[('hybrid-bitonic', nranks, 4, 21)] for nranks in ranks]
    comprank21.add_plot(ranks, ydata, 'o-', "hybrid rank=proc")
    comprank21.save_figure("compare-rank-21q.png")

    comprank27 = GraphMaker("Compare mpi with mpi/openmp bitonic (27q)", "#ranks (log2)")
    ydata = [results[('mpi-bitonic', nranks, 27)] for nranks in ranks]
    comprank27.add_plot(ranks, ydata, 'x-', "mpi")
    ydata = [results[('hybrid-bitonic', nranks-2, 4, 27)] for nranks in range(3,5)]
    comprank27.add_plot(range(3, 5), ydata, 'o-', "hybrid rank=proc+thread")
    ydata = [results[('hybrid-bitonic', nranks, 4, 27)] for nranks in ranks]
    comprank27.add_plot(ranks, ydata, 'o-', "hybrid rank=proc")
    comprank27.save_figure("compare-rank-27q.png")

    compq = GraphMaker("Compare mpi with mpi/openmp bitonic", "#problem size (log2)")
    ydata = [results[('mpi-bitonic', 3, nq)] for nq in q]
    compq.add_plot(q, ydata, 'x-', "mpi (8 procs)")
    ydata = [results[('hybrid-bitonic', 1, 4, nq)] for nq in q]
    compq.add_plot(q, ydata, 'o-', "hybrid (2 procs - 4 threads)")
    compq.save_figure("compare-optimized.png")

def threads_over_constrank():
    thcomp = GraphMaker("Perfomance of hybrid bitonic (21 #problem size)", "#processes")
    for nthread in threads:
        ydata = [results[('hybrid-bitonic', nranks, nthread, 21)] for nranks in ranks]
        thcomp.add_plot(ranks, ydata, 'o-', str(nthread))
    thcomp.save_figure("compare-threads-21q.png")

    thcomp = GraphMaker("Perfomance of hybrid bitonic (27 #problem size)", "#processes")
    for nthread in threads:
        ydata = [results[('hybrid-bitonic', nranks, nthread, 27)] for nranks in ranks]
        thcomp.add_plot(ranks, ydata, 'o-', str(nthread))
    thcomp.save_figure("compare-threads-27q.png")

    thcomp = GraphMaker("Perfomance of hybrid bitonic (24 #problem size)", "#processes")
    for nthread in threads:
        ydata = [results[('hybrid-bitonic', nranks, nthread, 24)] for nranks in ranks]
        thcomp.add_plot(ranks, ydata, 'o-', str(nthread))
    thcomp.save_figure("compare-threads-24q.png")

    thcomp = GraphMaker("Perfomance of hybrid bitonic", "#threads processes*threads==32")
    for i in range(3,-1,-1):
        ydata = [results[('hybrid-bitonic', i+1, threads[3-i], nq)] for nq in q]
        thcomp.add_plot(q, ydata, 'o-', str(2**(i+1))+"nodes/"+str(threads[3-i])+"threads")
    thcomp.save_figure("const-parallelism.png")

def output_results():
    print results

if __name__ == '__main__':
    handle_results()
    ranks_over_constq()
    compare_over_const()
    threads_over_constrank()


