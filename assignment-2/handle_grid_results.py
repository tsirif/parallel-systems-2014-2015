#!/usr/bin/env python
# encoding: utf-8

import os
from collections import defaultdict
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

results = defaultdict(float)
executable = ["mpi-bitonic", "hybrid-bitonic", "serial-bitonic"]
nodes = range(3)
ranks = range(1, 5)
threads = [2, 4, 8, 16]
q = range(16, 28)


def handle_results():
    os.chdir('./grid-results')
    print "indexing mpi-bitonic results"
    for nnodes, nranks, nq in product(nodes, ranks, q):
        filename = executable[0]+"-"+str(2**nnodes)+"-"+str(2**(nranks+nnodes))
        filename = filename +"-"+str(nq)+".out"
        try:
            with open(filename, 'r') as f:
                for line in f:
                    if len(line.split()) < 2:
                        continue
                    first = line.split()[0]
                    last = line.split()[-1]
                    if first == "mpi":
                        results[('mpi-bitonic', nnodes, nranks, nq)] = float(last)
                    if first == "serial":
                        results[('mpi-bitonic', nnodes, nranks, nq)] /= float(last)
                        break
            print filename
        except IOError as e:
            print e

    print "indexing hybrid-bitonic results"
    for nnodes, nranks, nq, nthread in product(nodes, ranks, q, threads):
        filename = executable[1]+"-"+str(2**nnodes)+"-"+str(2**(nranks+nnodes))
        filename = filename +"-"+str(nthread)+"-"+str(nq)+".out"
        try:
            with open(filename, 'r') as f:
                for line in f:
                    if len(line.split()) < 2:
                        continue
                    first = line.split()[0]
                    last = line.split()[-1]
                    if first == "hybrid":
                        results[('hybrid-bitonic', nnodes, nranks, nthread, nq)] = float(last)
                    if first == "serial":
                        results[('hybrid-bitonic', nnodes, nranks, nthread, nq)] /= float(last)
                        break
            print filename
        except IOError as e:
            print e

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
    print results

class GraphMaker(object):
    def __init__(self, title, xlabel, bounds=True):
        fig, axes = plt.subplots(sharex=True, sharey=True)
        self.fig = fig
        self.axes = axes
        self.axes.set_title(title)
        self.axes.set_ylabel("wtime over quicksort's wtime")
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylim([0, 1.25])
        self.axes.autoscale_view()
        self.axes.grid(True)
        # self.axes.xaxis.set_major_locator(matplotlib.dates.MinuteLocator())
        # self.axes.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%M:%S'))
        # self.axes.xaxis.set_minor_locator(matplotlib.dates.SecondLocator())

    def add_plot(self, xdata, ydata, data_fmt, data_label):
        plt.plot(xdata, ydata, data_fmt, label=data_label)

    def show_figure(self):
        self.axes.legend()
        # self.fig.autofmt_xdate()
        plt.show()

    def save_figure(self, filename):
        self.axes.legend()
        # self.fig.autofmt_xdate()
        plt.savefig(filename, bbox_inches='tight')

def q_over_constppn():
    # mpi-bitonic
    mpibit = GraphMaker("Perfomance of mpi bitonic", "#problem size (log2)")
    for nnodes in nodes:
        if nnodes == 2:
            break
        ydata = [results[('mpi-bitonic', nnodes, nnodes+2, nq)] for nq in q]
        print ydata
        mpibit.add_plot(q, ydata, 'o-', str(2**nnodes)+"/4ppn")
    mpibit.save_figure("mpi-4ppn.png")

    mpibit = GraphMaker("Perfomance of mpi bitonic", "#problem size (log2)")
    for nnodes in nodes:
        if nnodes == 2:
            break
        ydata = [results[('mpi-bitonic', nnodes, nnodes+3, nq)] for nq in q]
        print ydata
        mpibit.add_plot(q, ydata, 'o-', str(2**nnodes)+"/"+"8ppn")
    mpibit.save_figure("mpi-8ppn.png")
    # hybrid-bitonic
    hybridbit = GraphMaker("Perfomance of hybrid bitonic", "#problem size (log2)")
    for nnodes in nodes:
        if nnodes == 2:
            break
        ydata = [results[('hybrid-bitonic', nnodes, nnodes+2, 4, nq)] for nq in q]
        print ydata
        hybridbit.add_plot(q, ydata, 'o-', str(2**nnodes)+"/4ppn/4threads")
    hybridbit.save_figure("hybrid-4ppn.png")

    hybridbit = GraphMaker("Perfomance of hybrid bitonic", "#problem size (log2)")
    for nnodes in nodes:
        if nnodes == 2:
            break
        ydata = [results[('hybrid-bitonic', nnodes, nnodes+3, 4, nq)] for nq in q]
        print ydata
        hybridbit.add_plot(q, ydata, 'o-', str(2**nnodes)+"/8ppn/4threads")
    hybridbit.save_figure("hybrid-8ppn.png")

def q_over_constranks():
    # mpi-bitonic
    mpibit = GraphMaker("Perfomance of mpi bitonic", "#problem size (log2)")
    for nnodes in nodes:
        ydata = [results[('mpi-bitonic', nnodes, 3-nnodes, nq)] for nq in q]
        mpibit.add_plot(q, ydata, 'o-', str(2**nnodes)+"nodes/"+str(2**(3-nnodes))+"ppn")
    mpibit.save_figure("mpi-8ranks.png")
    # mpi-bitonic
    mpibit = GraphMaker("Perfomance of mpi bitonic", "#problem size (log2)")
    for nnodes in nodes:
        if nnodes == 2:
            break
        ydata = [results[('mpi-bitonic', nnodes, 4-nnodes, nq)] for nq in q]
        mpibit.add_plot(q, ydata, 'o-', str(2**nnodes)+"nodes/"+str(2**(4-nnodes))+"ppn")
    mpibit.save_figure("mpi-16ranks.png")

    # hybrid-bitonic
    hybridbit = GraphMaker("Perfomance of hybrid bitonic", "#problem size (log2)")
    for nnodes in nodes:
        ydata = [results[('hybrid-bitonic', nnodes, 3-nnodes, 4, nq)] for nq in q]
        hybridbit.add_plot(q, ydata, 'o-', str(2**nnodes)+"nodes/"+str(2**(3-nnodes))+"ppn")
    hybridbit.save_figure("hybrid-8ranks.png")
    # hybrid-bitonic
    hybridbit = GraphMaker("Perfomance of hybrid bitonic", "#problem size (log2)")
    for nnodes in nodes:
        ydata = [results[('hybrid-bitonic', nnodes, 4-nnodes, 4, nq)] for nq in q]
        hybridbit.add_plot(q, ydata, 'o-', str(2**nnodes)+"nodes/"+str(2**(4-nnodes))+"ppn")
    hybridbit.save_figure("hybrid-16ranks.png")

def compare_over_const():
    comprank = GraphMaker("Compare mpi with mpi/openmp bitonic", "#problem size (log2)")
    ydata = [results[('mpi-bitonic', 0, 4, nq)] for nq in q]
    comprank.add_plot(q, ydata, 'x-', "mpi 1node/16procs")
    ydata = [results[('mpi-bitonic', 1, 5, nq)] for nq in q]
    comprank.add_plot(q, ydata, 'x-', "mpi 2nodes/32procs")
    ydata = [results[('hybrid-bitonic', 0, 4, 8, nq)] for nq in q]
    comprank.add_plot(q, ydata, 'o-', "hybrid 1node/16procs/8threads")
    ydata = [results[('hybrid-bitonic', 1, 5, 8, nq)] for nq in q]
    comprank.add_plot(q, ydata, 'o-', "hybrid 2node/32procs/8threads")
    comprank.save_figure("compare-rank-21q.png")

if __name__ == '__main__':
    handle_results()
    q_over_constppn()
    q_over_constranks()
    compare_over_const()


