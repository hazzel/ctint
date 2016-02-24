import re
import os
import glob
import sys
sys.path.append('/home/stephan/mc/ctqmc')
sys.path.append("/net/home/lxtsfs1/tpc/hesselmann/mc/ctqmc")
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from ParseDataOutput import *
from texify import *

def combinatorial_factor(n, k):
	prod = 1.
	for j in range(n + 1):
		if k != j:
			prod *= (k + j) * (k - j)
	return 1. / prod

def estimator(n, beta, C):
	omega1 = 2. * np.pi / beta
	sum1 = 0.
	sum2 = 0.
	for k in range(n + 1):
		sum1 += k**2. * combinatorial_factor(n, k) * C[k]
		print k**2. * combinatorial_factor(n, k) * C[k], " # ", combinatorial_factor(n, k) * C[k]
		sum2 += combinatorial_factor(n, k) * C[k]
	print "n: ", n, " , ", sum1, " ", sum2
	return omega1 * abs(- sum1 / sum2)**0.5

latexify()
color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'darkgreen']
marker_cycle = ['o', 'D', '<', 'p', '>', 'v', '*', '^', 's']

filelist = []
filelist.append(glob.glob("../bin/job-2/*.out"))
filelist.sort()

Delta = []
filelist = [item for sublist in filelist for item in sublist]
for f in filelist:
	plist = ParseParameters(f)
	elist = ParseEvalables(f)
	
	for i in range(len(plist)):
		n_matsubara = int(plist[i]["matsubara_freqs"])
		n_discrete_tau = int(plist[i]["discrete_tau"])
		h = float(plist[i]["V"])
		T = float(plist[i]["T"])
		beta = 1./T
		L = float(plist[i]["L"])
	
		#x = (np.array(range(0, n_matsubara)) * 2. + 1.) * np.pi * T
		x = (np.array(range(0, n_matsubara)) * 2.) * np.pi * T
		y = np.array(ArrangePlot(elist[i], "dynamical_M2")[0]) * x**2.
		err = np.array(ArrangePlot(elist[i], "dynamical_M2")[1]) * x**2.
		for n in range(1, 10):
			Delta.append(estimator(n, 1./T, y))
		
		c = 0
		ax = plt.gca()
		ax.set_xlabel(r"$\omega_n$")
		ax.set_ylabel(r"$M_2(\omega_n) \cdot \omega_n^2$")
		ax.plot(np.array(x), np.array(y), marker=marker_cycle[c%len(marker_cycle)], color=color_cycle[c%len(color_cycle)], markersize=10.0, linewidth=2.0, label=r'$L='+str(int(L))+'$')
		ax.plot(np.array(x), np.array(y), marker=marker_cycle[c%len(marker_cycle)], color=color_cycle[c%len(color_cycle)], markersize=10.0, linewidth=2.0, label=r'$L='+str(int(L))+'$')
		(_, caps, _) = ax.errorbar(np.array(x), np.array(y), yerr=np.array(err), marker='None', capsize=10, color=color_cycle[c%len(color_cycle)])
		for cap in caps:
			cap.set_markeredgewidth(1.4)
	plt.tight_layout()
print Delta
plt.show()
