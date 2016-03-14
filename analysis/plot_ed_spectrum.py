import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pylab
import glob
import sys
from itertools import groupby
sys.path.append('/home/stephan/mc/ctqmc')
sys.path.append("/net/home/lxtsfs1/tpc/hesselmann/mc/ctqmc")
sys.path.append('/home/stephan/mc/qising-SSE')
from texify import *

latexify()
color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'darkgreen']
marker_cycle = ['o', 'D', '<', 'p', '>', 'v', '*', '^', 's']

list_of_files = glob.glob("../data/spectrum/ed*")
datalist = [ ( pylab.loadtxt(filename), label ) for label, filename in enumerate(list_of_files) ]

for data, label in datalist:
	plt.figure()
	k = int(data[:,0][0])
	L = int(data[:,1][0])
	plt.xlabel(r"$V$")
	plt.ylabel(r"$\frac{E_n(V) - E_0(V)}{E_1(0)}$")
	plt.title(r"$L = " + str(L) + "$")
	x = data[:,2]
	undegenerate_spectrum = {}
	color_spectrum = {}
	for i in range(len(x)):
		undegenerate_spectrum[x[i]] = [list(group)[0] - min(data[i,3:]) for key, group in groupby(data[i,3:])]
		color_spectrum[x[i]] = [color_cycle[(len(list(group))-1)%len(color_cycle)] for key, group in groupby(data[i,3:])]

		y = [p / undegenerate_spectrum[x[0]][1] for p in undegenerate_spectrum[x[i]]]
		plt.scatter([x[i]]*len(y), y, marker='o', color=color_spectrum[x[i]], s=[50.]*len(y))

plt.show()