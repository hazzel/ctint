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
		sum2 += combinatorial_factor(n, k) * C[k]
	return omega1 * (- sum1 / sum2)**0.5

latexify()
color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'darkgreen']
marker_cycle = ['o', 'D', '<', 'p', '>', 'v', '*', '^', 's']

filelist = []
filelist.append(glob.glob("../bin/job/*.out"))
filelist.sort()

Delta = []
filelist = [item for sublist in filelist for item in sublist]
for f in filelist:
	plist = ParseParameters(f)
	elist = ParseEvalables(f)
	f, ax = plt.subplots(3, 3, sharex = True, sharey = True)
	ax = [item for sublist in ax for item in sublist]
	
	for i in range(len(plist)):
		n_matsubara = int(plist[i]["matsubara_freqs"])
		n_discrete_tau = int(plist[i]["discrete_tau"])
		h = float(plist[i]["V"])
		T = float(plist[i]["T"])
		beta = 1./T
		L = float(plist[i]["L"])
		
		rmax = 0
		while True:
			y = ArrangePlot(elist[i], "G\(omega\)_" + str(rmax))[0]
			if (y == []):
				rmax -= 1
				break
			rmax += 1

		for r in range(rmax + 1):
			x = np.array(range(0, n_matsubara))
			y = np.array(ArrangePlot(elist[i], "G\(omega\)_" + str(r))[0])
			Delta.append([])
			for n in range(1, n_matsubara):
				Delta[r].append(estimator(n, 1./T, y))
				print n, " :", estimator(n, 1./T, y)
			
			ymax = np.abs(y[0])
			y /= ymax
			err = np.array(ArrangePlot(elist[i], "G\(omega\)_" + str(r))[1])
			err /= ymax
			
			c = 0
			#ax[r].set_xlabel(r"$n$")
			#ax[r].set_ylabel(r"$G(\omega_n)_" + str(r) + "$")
			ax[r].text(0.05, 0.95, r"$G_" + str(r) + "(\omega_n) / |G_" + str(r) + "(0)|$", transform=ax[r].transAxes, fontsize=15, va='top')
			ax[r].plot(np.array(x), np.array(y), marker=marker_cycle[c%len(marker_cycle)], color=color_cycle[c%len(color_cycle)], markersize=10.0, linewidth=2.0, label=r'$L='+str(int(L))+'$')
			ax[r].plot(np.array(x), np.array(y), marker=marker_cycle[c%len(marker_cycle)], color=color_cycle[c%len(color_cycle)], markersize=10.0, linewidth=2.0, label=r'$L='+str(int(L))+'$')
			(_, caps, _) = ax[r].errorbar(np.array(x), np.array(y), yerr=np.array(err), marker='None', capsize=10, color=color_cycle[c%len(color_cycle)])
			for cap in caps:
				cap.set_markeredgewidth(1.4)
	plt.tight_layout()
	f.subplots_adjust(hspace=0.2, wspace=0.2)
print Delta[0]
plt.show()
