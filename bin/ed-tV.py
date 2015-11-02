import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl
import scipy.linalg as la
from itertools import product

def set_bit(int_type, offset):
	mask = 1 << offset
	return (int_type | mask)
 
def clear_bit(int_type, offset):
	mask = ~(1 << offset)
	return (int_type & mask)

def invert_bit(int_type, offset):
	mask = 1 << offset
	return (int_type ^ mask)

def test_bit(int_type, offset):
	mask = 1 << offset
	return (int_type & mask) >> offset

class hilbert:
	def __init__(self, size_):
		self.size = size_

	# psi = (sign, state)
	def n_el(self, psi):
		return psi[0] * sum(test_bit(psi[1], i) for i in xrange(N))
	
	def n_i(self, psi, i):
		return psi[0] * test_bit(psi[1], i)

	def c_i(self, psi, i):
		if not test_bit(psi[1], i):
			return (0, 0)
		sign = psi[0]
		for j in xrange(i + 1, self.size):
			if test_bit(psi[1], j):
				sign *= -1
		return (sign, clear_bit(psi[1], i))

	def c_dag_i(self, psi, i):
		if test_bit(psi[1], i):
			return (0, 0)
		sign = psi[0]
		for j in xrange(i + 1, self.size):
			if test_bit(psi[1], j):
				sign *= -1
		return (sign, set_bit(psi[1], i))

def generate_graph(L):
	graph = {}
	N = 2*L**2
	for i in xrange(N):
		if i % 2 == 1:
			if (i+1) % (2*L) == 0:
				graph[i] = []
				graph[i].append((i - 4*L + 1 + N) % N)
				graph[i].append((i - 2*L + 1 + N) % N)
				graph[i].append((i - 1 + N) % N)
			else:
				graph[i] = []
				graph[i].append((i - 2*L + 1 + N) % N)
				graph[i].append((i + 1 + N) % N)
				graph[i].append((i - 1 + N) % N)
		else:
			if i % (2*L) == 0:
				graph[i] = []
				graph[i].append((i + 4*L - 1 + N) % N)
				graph[i].append((i + 2*L - 1 + N) % N)
				graph[i].append((i + 1 + N) % N)
			else:
				graph[i] = []
				graph[i].append((i + 2*L - 1 + N) % N)
				graph[i].append((i - 1 + N) % N)
				graph[i].append((i + 1 + N) % N)
	return graph

L = 3
V = 1.5
T = 0.1
beta = 1./T
N = 2*L**2
start_time = time.time()
total_h = hilbert(N)
g = generate_graph(L)

print "Construct Hilbert space..."
#Grand canonical at half filling with <n> = 0.5 and mu=0
#basis = filter(lambda x: total_h.n_el((1, x)) >= 0, xrange(2**N))

#Canonical at half filling with n = 0.5
basis = filter(lambda x: total_h.n_el((1, x)) == N/2, xrange(2**N))

index = dict((state, i) for i,state in enumerate(basis))
d = len(basis)
print "Hilbert space dimension:", d
print "Construct matrices..."
H = sp.dok_matrix((d,d))
M2 = sp.dok_matrix((d,d))
M4 = sp.dok_matrix((d,d))

prog = 0
for in_state in basis:
	for i, j in product(xrange(N), xrange(N)):
		# hopping
		if j in g[i]:
			out_state = total_h.c_i((-1, in_state), j)
			out_state = total_h.c_dag_i(out_state, i)
			if out_state[0] != 0:
				H[index[out_state[1]], index[in_state]] += out_state[0] * -1.
		# nearest neighbor V
		if i < j and j in g[i]:
			H[index[in_state], index[in_state]] += V * (total_h.n_i((1, in_state),
				i) - 0.5) * (total_h.n_i((1, in_state), j) - 0.5)
		# M2
		p_i = 1.0 if i % 2 == 0 else -1.0
		p_j = 1.0 if j % 2 == 0 else -1.0
		M2[index[in_state], index[in_state]] += p_i * p_j * 1./N**2 * (
			total_h.n_i((1, in_state), i) - 0.5) * (total_h.n_i((1, in_state), j)
			- 0.5)
		# M4
		for k, l in product(xrange(N), xrange(N)):
			p_k = 1.0 if k % 2 == 0 else -1.0
			p_l = 1.0 if l % 2 == 0 else -1.0
			M4[index[in_state], index[in_state]] += p_i * p_j * p_k * p_l \
				* 1./N**4 * (total_h.n_i((1, in_state), i) - 0.5) * (total_h.n_i(
				(1, in_state), j) - 0.5) * (total_h.n_i((1, in_state), k) - 0.5) \
				* (total_h.n_i((1, in_state), l) - 0.5)
	prog += 1
	if prog % max(1,(d/100)) == 0:
		sys.stdout.write(str(round(100.*prog/d,1)) + "% ")
		sys.stdout.flush()
sys.stdout.write("\n")

H = sp.csc_matrix(H)
k_max = 26
for k in xrange(20, k_max):
	ev, es = sl.eigsh(H, k=k, which='SA')
	V = np.matrix(es)
	D = np.diag(np.exp(-beta * ev))
	Z = np.trace(D)
	Eexp = np.trace(D*np.diag(ev))/Z
	m2 = np.trace(D*V.T*M2*V)/Z
	m4 = np.trace(D*V.T*M4*V)/Z
	print "E(k=" + str(k) + ") = " + str(Eexp)
	print "m2(k=" + str(k) + ") = " + str(m2)
	print "m4(k=" + str(k) + ") = " + str(m4)
	print "B(k=" + str(k) + ") = " + str(m4/m2**2)
	print "----------------"
print "Ellapsed wall clock time:", round(time.time() - start_time, 2), \
	"seconds."
