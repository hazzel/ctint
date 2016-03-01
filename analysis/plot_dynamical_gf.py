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
filelist.append(glob.glob("../bin/job/*.out"))
#filelist.append(glob.glob("../bin/job-2/*.out"))
filelist.sort()

Delta = []
filelist = [item for sublist in filelist for item in sublist]
for f in filelist:
	figure, (ax1, ax2) = plt.subplots(1, 2)
	plist = ParseParameters(f)
	elist = ParseEvalables(f)
	
	for i in range(len(plist)):
		n_matsubara = int(plist[i]["matsubara_freqs"])
		n_discrete_tau = int(plist[i]["discrete_tau"])
		h = float(plist[i]["V"])
		T = float(plist[i]["T"])
		L = float(plist[i]["L"])
	
		x_mat = (np.array(range(0, n_matsubara)) * 2.) * np.pi * T
		y_mat = np.array(ArrangePlot(elist[i], "dynamical_M2_mat")[0])# * x**2.
		err_mat = np.array(ArrangePlot(elist[i], "dynamical_M2_mat")[1])# * x**2.
		x_tau = np.array(range(0, n_discrete_tau)) / float(n_discrete_tau) / T
		y_tau = np.log(np.array(ArrangePlot(elist[i], "dynamical_M2_tau")[0]))
		err_tau = np.array(ArrangePlot(elist[i], "dynamical_M2_tau")[1]) / y_tau
		for n in range(1, 10):
			Delta.append(estimator(n, 1./T, y_mat))

		c = 0
		ax1.set_xlabel(r"$\omega_n$")
		ax1.set_ylabel(r"$M_2(\omega_n) \cdot \omega_n^2$")
		ax1.plot(np.array(x_mat), np.array(y_mat), marker=marker_cycle[c%len(marker_cycle)], color=color_cycle[c%len(color_cycle)], markersize=10.0, linewidth=2.0, label=r'$L='+str(int(L))+'$')
		ax1.plot(np.array(x_mat), np.array(y_mat), marker=marker_cycle[c%len(marker_cycle)], color=color_cycle[c%len(color_cycle)], markersize=10.0, linewidth=2.0, label=r'$L='+str(int(L))+'$')
		(_, caps, _) = ax1.errorbar(np.array(x_mat), np.array(y_mat), yerr=np.array(err_mat), marker='None', capsize=10, color=color_cycle[c%len(color_cycle)])
		for cap in caps:
			cap.set_markeredgewidth(1.4)
			
		c = 1
		ax2.set_xlabel(r"$\tau$")
		ax2.set_ylabel(r"$M_2(\tau)$")
		ax2.plot(np.array(x_tau), np.array(y_tau), marker=marker_cycle[c%len(marker_cycle)], color=color_cycle[c%len(color_cycle)], markersize=10.0, linewidth=2.0, label=r'$L='+str(int(L))+'$')
		ax2.plot(np.array(x_tau), np.array(y_tau), marker=marker_cycle[c%len(marker_cycle)], color=color_cycle[c%len(color_cycle)], markersize=10.0, linewidth=2.0, label=r'$L='+str(int(L))+'$')
		(_, caps, _) = ax2.errorbar(np.array(x_tau), np.array(y_tau), yerr=np.array(err_tau), marker='None', capsize=10, color=color_cycle[c%len(color_cycle)])
		for cap in caps:
			cap.set_markeredgewidth(1.4)
		
	plt.tight_layout()
print Delta
plt.show()
