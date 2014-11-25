#!/usr/bin/env python

import subprocess32

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

METHODS = ['Serial', 'OpenMP', 'Pthreads']
LIBRARIES = ['hash', 'morton', 'radix', 'rearrange']
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


def compile_methods():
    for method in METHODS:
        subprocess32.check_call(["make", "-C", "./"+method])


def run(args, method):
    arguments = list(args)
    arguments.insert(0, "./" + method + "/test_octree.out")
    s = subprocess32.check_output(arguments)
    with open("./" + method + "/results.txt", 'w') as f:
        f.write(s)


def run_methods(args):
    for method in METHODS:
        run(args, method)


def get_results(method):
    success = True
    counter = defaultdict(int)
    mean_duration = defaultdict(float)
    with open("./" + method + "/results.txt", 'r') as f:
        for line in f:
            for library in LIBRARIES:
                if library in line:
                    counter[library] += 1
                    duration = float(line.split()[-1])
                    mean_duration[library] += duration
            if "Index" in line:
                if line.split()[-1] != 'PASS':
                    success = False
            if "Encoding" in line:
                if line.split()[-1] != 'PASS':
                    success = False
    for library in LIBRARIES:
        mean_duration[library] /= counter[library]
    return mean_duration, success


def get_method_results():
    time_results = defaultdict(defaultdict)
    success_results = defaultdict(bool)
    for method in METHODS:
        mean_duration, success = get_results(method)
        time_results[method] = mean_duration
        success_results[method] = success
    return time_results, success_results


def compare_results(N, P, L, results, show=False):
    lenL = len(LIBRARIES)
    lenM = len(METHODS)
    width = 0.35
    ind = np.arange(0, (lenL)*(lenM+1)*width, (lenM+1)*width)

    fig, ax = plt.subplots()
    method_rects = []
    for i in range(lenM):
        values = np.array(results[METHODS[i]].values())
        method_rects.append(ax.bar(ind+i*width, values*1000,
                                   width, color=COLORS[i]))

    ax.set_ylabel('Duration (ms)')
    ax.set_title('Duration: N='+N+' P='+P+' L='+L)
    ax.set_xticks(ind+lenM*width/2)
    ax.set_xticklabels(results[METHODS[0]].keys())

    def f(x):
        return x[0]

    ax.legend(map(f, method_rects), METHODS)

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height,
                    '%.2f' % height,
                    ha='center', va='bottom')

    for rects in method_rects:
        autolabel(rects)

    if show:
        plt.show()
    else:
        plt.savefig('compare-'+N+'-'+P+'-'+L+'.png',
                    bbox_inches='tight')


def compare_methods():
    args_list = [[str(i), '0', str(j), '5', str(k)]
                 for i in range(500000, 2500000, 500000)
                 for j in range(68, 138, 30)
                 for k in range(10, 25, 5)]
    compile_methods()
    for args in args_list:
        run_methods(args)
        time_results, success_results = get_method_results()
        for method, success in success_results.items():
            print method + " implementation was successful: " + str(success)
        compare_results(args[0], args[2], args[4], time_results)


class GraphMaker(object):
    def __init__(self, xdata, title, serial_performance, method, graph_type):
        self.xdata = xdata
        self.ydata = defaultdict(list)

        fig, ax = plt.subplots(sharex=True, sharey=True)
        self.fig = fig
        self.ax = ax

        self.ax.set_title(title)
        self.ax.set_ylabel('Performance over Serial')

        self.serial = serial_performance
        self.method = method
        self.graph_type = graph_type

    def addy(self, means):
        for library in LIBRARIES:
            l = len(self.ydata[library])
            self.ydata[library].append(
                means[library]/self.serial[l][library])

    def plot(self):
        for library in LIBRARIES:
            self.ax.plot(self.xdata, self.ydata[library], label=library)

        self.ax.legend()

        plt.savefig('graph_'+self.method+'_'+self.graph_type+'.png',
                    bbox_inches='tight')


def make_graph(title_method, graph_type, xdata, args_list):
    compile_methods()
    serial_performance = list()
    for args in args_list:
        print '.',
        run(args, 'Serial')
        mean_duration, success = get_results('Serial')
        serial_performance.append(mean_duration)
    for method in METHODS[1:]:
        title = method + title_method
        graphMaker = GraphMaker(xdata, title, serial_performance,
                                method, graph_type)
        for args in args_list:
            print '.',
            run(args, method)
            mean_duration, success = get_results(method)
            graphMaker.addy(mean_duration)
        graphMaker.plot()


def show_perfomance():
    xdata = np.arange(1000000, 5000000, 50000)
    args_list = [[str(i), '0', '98', '5', '15']
                 for i in xdata]
    title_method = ' perfomance by N (P=98,L=15,cube)'
    graph_type = 'N_cube'
    print "Making "+graph_type+" graphs!..."
    make_graph(title_method, graph_type, xdata, args_list)
    print "Done!"

    xdata = np.arange(68, 128, 1)
    args_list = [['2500000', '0', str(i), '5', '15']
                 for i in xdata]
    title_method = ' perfomance by P (N=2500000,L=15,cube)'
    graph_type = 'P_cube'
    print "Making "+graph_type+" graphs!..."
    make_graph(title_method, graph_type, xdata, args_list)
    print "Done!"

    xdata = np.arange(10, 25, 1)
    args_list = [['2500000', '0', '98', '5', str(i)]
                 for i in xdata]
    title_method = ' perfomance by L (N=2500000,P=98,cube)'
    graph_type = 'L_cube'
    print "Making "+graph_type+" graphs!..."
    make_graph(title_method, graph_type, xdata, args_list)
    print "Done!"

    xdata = np.arange(1000000, 5000000, 50000)
    args_list = [[str(i), '1', '98', '5', '15']
                 for i in xdata]
    title_method = ' perfomance by N (P=98,L=15,cube)'
    graph_type = 'N_plummer'
    print "Making "+graph_type+" graphs!..."
    make_graph(title_method, graph_type, xdata, args_list)
    print "Done!"

    xdata = np.arange(68, 128, 1)
    args_list = [['2500000', '1', str(i), '5', '15']
                 for i in xdata]
    title_method = ' perfomance by P (N=2500000,L=15,cube)'
    graph_type = 'P_plummer'
    print "Making "+graph_type+" graphs!..."
    make_graph(title_method, graph_type, xdata, args_list)
    print "Done!"

    xdata = np.arange(10, 25, 1)
    args_list = [['2500000', '1', '98', '5', str(i)]
                 for i in xdata]
    title_method = ' perfomance by L (N=2500000,P=98,cube)'
    graph_type = 'L_plummer'
    print "Making "+graph_type+" graphs!..."
    make_graph(title_method, graph_type, xdata, args_list)
    print "Done!"


if __name__ == "__main__":

    import sys
    argv = sys.argv[1:]
    if len(argv) == 1:
        if argv[0] == '-g':
            show_perfomance()
            compare_methods()
    else:
        if len(argv) == 5:
            args = argv
        else:
            args = ['2500000', '0', '98', '5', '10']
        compile_methods()
        run_methods(args)
        time_results, success_results = get_method_results()
        for method, success in success_results.items():
            print method + " implementation was successful: " + str(success)
        compare_results(args[0], args[2], args[4], time_results, show=True)
