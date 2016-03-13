import re
import os
import glob
import sys
sys.path.append('/home/stephan/mc/ctqmc')
sys.path.append("/net/home/lxtsfs1/tpc/hesselmann/mc/ctqmc")
import numpy as np
from decimal import *
from scipy.optimize import curve_fit
import matplotlib as plt
import pylab
from ParseDataOutput import *
sys.path.append("/net/home/lxtsfs1/tpc/hesselmann/mc/qising-SSE")
sys.path.append("/home/stephan/mc/qising-SSE")
from texify import *

def FitFunction(x, a, b, c):
	return a+b*np.exp(-c*x)

def combinatorial_factor(n, k):
	prod = Decimal(1)
	for j in range(n + 1):
		if k != j:
			kDec = Decimal(str(k))
			jDec = Decimal(str(j))
			prod *= (kDec + jDec) * (kDec - jDec)
	return Decimal(1) / prod

def estimator(n, beta, C):
	omega1 = Decimal(str(2. * np.pi / beta))
	sum1 = Decimal(0)
	sum2 = Decimal(0)
	for k in range(n + 1):
		sum1 += Decimal(str(k**2.)) * combinatorial_factor(n, k) * Decimal(str(C[k]))
		sum2 += combinatorial_factor(n, k) * Decimal(str(C[k]))
	return float(omega1 * abs(- sum1 / sum2).sqrt())

latexify()
color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'darkgreen']
marker_cycle = ['o', 'D', '<', 'p', '>', 'v', '*', '^', 's']

list_of_files = glob.glob("../data/ed*")
datalist = [ ( pylab.loadtxt(filename), label ) for label, filename in enumerate(list_of_files) ]

for data, label in datalist:
	for i in range(len(data)):
		figure, (ax1, ax2, ax3) = plt.subplots(1, 3)
		
		n_discrete_tau = int(data[i,8])
		h = data[i,2]
		T = data[i,3]
		L = int(data[i,1])
		figure.suptitle(r"$L = " + str(L) + ",\ V = " + str(h) + ",\ T = " + str(T) + "$")
		
		x_tau = np.linspace(0., 1./T, n_discrete_tau)
		y_tau = data[i,9:]

		c = 0
		ax1.set_xlabel(r"$\omega_n$")
		ax1.set_ylabel(r"$M_2(\omega_n) \cdot \omega_n^2$")
				
		c = 1
		ax2.set_xlabel(r"$\tau$")
		ax2.set_ylabel(r"$M_2(\tau)$")
		ax2.set_yscale("log")
		ax2.plot(x_tau, y_tau, marker=marker_cycle[c%len(marker_cycle)], color=color_cycle[c%len(color_cycle)], markersize=10.0, linewidth=2.0, label=r'$L='+str(int(L))+'$')
		
		try:
			nmin = 10; nmax = len(x_tau)/2-18
			parameter, perr = curve_fit(FitFunction, x_tau[nmin:nmax], y_tau[nmin:nmax])
			px = np.linspace(x_tau[nmin], x_tau[nmax], 1000)
			ax2.plot(px, FitFunction(px, *parameter), 'k-', linewidth=3.0)
			ax2.text(0.05, 0.98, r"$\Delta_{FIT} = " + ("{:.4f}").format(parameter[2]) + "$", transform=ax2.transAxes, fontsize=20, va='top')
			ax2.text(0.05, 0.92, r"$\Delta_{ED} = 0.9264$", transform=ax2.transAxes, fontsize=20, va='top')
			print parameter
			print perr
		except:
			print "runtime error"
		
		c = 2
		ax3.set_xlabel(r"$n$")
		ax3.set_ylabel(r"$\Delta_n$")
			
		plt.tight_layout()
plt.show()
