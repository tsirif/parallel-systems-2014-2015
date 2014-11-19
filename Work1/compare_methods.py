#!/usr/bin/env python

import sys
import subprocess32

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

METHODS = ['OpenMP', 'Serial']
LIBRARIES = ['hash', 'morton', 'radix', 'rearrange']
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


def compile_and_run_methods(args):
    for method in METHODS:
        subprocess32.check_call(["make", "-C", "./"+method])
        arguments = list(args)
        arguments.insert(0, "./" + method + "/test_octree.out")
        s = subprocess32.check_output(arguments)
        with open("./" + method + "/results.txt", 'w') as f:
            f.write(s)


def get_method_results():
    results = defaultdict(defaultdict)
    for method in METHODS:
        counter = defaultdict(int)
        mean_duration = defaultdict(float)
        with open("./" + method + "/results.txt", 'r') as f:
            for line in f:
                for library in LIBRARIES:
                    if library in line:
                        counter[library] += 1
                        duration = float(line.split()[-1])
                        mean_duration[library] += duration
        for library in LIBRARIES:
            mean_duration[library] /= counter[library]
        results[method] = mean_duration
    return results


def compare_results(results):
    N = len(LIBRARIES)
    ind = np.arange(N)
    width = 0.35

    fig, ax = plt.subplots()
    method_rects = []
    for i in range(len(METHODS)):
        values = np.array(results[METHODS[i]].values())
        method_rects.append(ax.bar(ind+i*width, values*1000,
                                   width, color=COLORS[i]))

    ax.set_ylabel('Duration (ms)')
    ax.set_title('Duration by library and method')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(results[METHODS[0]].keys())
    f = lambda x: x[0]
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

    plt.show()

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        args = ['1000000', '0', '20', '20', '10']
    compile_and_run_methods(args)
    results = get_method_results()
    print results
    compare_results(results)
