import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import pylab
import glob
import sys
sys.path.append('/home/stephan/mc/ctqmc')
sys.path.append("/net/home/lxtsfs1/tpc/hesselmann/mc/ctqmc")
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
	for i in range(k):
		x = data[:,2]
		y = data[:,3+i] - data[:,3]
		plt.plot( x, y, "o", markersize=10, color=color_cycle[i%len(color_cycle)] )

plt.xlabel(r"$V$")
plt.ylabel(r"$E_n - E_0$")
plt.show()