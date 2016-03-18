import re
import os
import glob
import sys
sys.path.append('/home/stephan/mc/ctqmc')
sys.path.append("/net/home/lxtsfs1/tpc/hesselmann/mc/ctqmc")
import numpy as np
from cdecimal import *
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
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
		n_ed_tau = int(data[i,8])
		n_ed_mat = int(data[i,10+n_ed_tau])
		figure.suptitle(r"$L = " + str(L) + ",\ V = " + str(h) + ",\ T = " + str(T) + "$")
		
		x_mat = np.array(range(0, n_ed_mat)) * 2. * np.pi * T
		y_mat = data[i,11+n_ed_tau:]
		
		x_tau = np.linspace(0., 1./T/2., n_ed_tau + 1)
		y_tau = data[i,9:10+n_ed_tau]
		
		N_bootstrap = 250
		x_delta = np.array(range(1, n_ed_mat))
		y_delta = []
		for j in range(N_bootstrap):
			y_delta.append(np.zeros(n_ed_mat - 1))
			y_boot = np.zeros(n_ed_mat)
			for k in range(len(y_boot)):
				y_boot[k] = y_mat[k] + np.random.normal(0., 0.01 * abs(y_mat[k]))
			for n in range(1, n_ed_mat):
				y_delta[j][n-1] = estimator(n, 1./T, y_boot)

		c = 0
		ax1.set_xlabel(r"$\omega_n$")
		ax1.set_ylabel(r"$M_2(\omega_n) \cdot \omega_n^2$")
		ax1.plot(x_mat, y_mat * x_mat**2., marker='o', color="r", markersize=10.0, linewidth=2.0, label=r'$L='+str(int(L))+'$')
				
		c = 1
		ax2.set_xlabel(r"$\tau$")
		ax2.set_ylabel(r"$M_2(\tau)$")
		ax2.set_yscale("log")
		ax2.plot(x_tau, y_tau, marker='o', color="r", markersize=10.0, linewidth=2.0, label=r'$L='+str(int(L))+'$')
		
		try:
			nmin = len(x_tau)/4; nmax = 3.*len(x_tau)/4
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
		ax3.plot(x_delta, np.mean(y_delta, axis=0), marker="o", color="red", markersize=10.0, linewidth=2.0, label=r'$L='+str(int(L))+'$')
		(_, caps, _) = ax3.errorbar(x_delta, np.mean(y_delta, axis=0), yerr=np.std(y_delta, axis=0), marker="None", color="red", markersize=10.0, linewidth=2.0)
		for cap in caps:
			cap.set_markeredgewidth(1.4)
			
		plt.tight_layout()
plt.show()
