from itertools import product
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl
import scipy.linalg as la

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

V = 1.5
L = 2
N = 2*L**2
total_h = hilbert(N)
g = generate_graph(L)

#Grand canonical at half filling with <n> = 0.5 and mu=0
basis = filter(lambda x: total_h.n_el((1, x)) >= 0, xrange(2**N))
#Canonical at half filling with n = 0.5
#basis = filter(lambda x: total_h.n_el((1, x)) == N/2, xrange(2**N))
index = dict((state, i) for i,state in enumerate(basis))
d = len(basis)
print "Hilbert space dimension:", d
H = sp.dok_matrix((d,d))

# hopping
for in_state in basis:
	for i, j in product(xrange(N), xrange(N)):
		if j in g[i]:
			out_state = total_h.c_i((-1, in_state), j)
			out_state = total_h.c_dag_i(out_state, i)
			if out_state[0] != 0:
				H[index[out_state[1]], index[in_state]] += out_state[0] * -1.

# nearest neighbor V
for state in basis:
	for i, j in product(xrange(N), xrange(N)):
		if i < j and j in g[i]:
			H[index[state], index[state]] += V * (total_h.n_i((1, state), i) - 0.5) * (total_h.n_i((1, state), j) - 0.5)

# M2
M2 = sp.dok_matrix((d,d))
for state in basis:
	for i, j in product(xrange(N), xrange(N)):
		p_i = 1.0 if i % 2 == 0 else -1.0
		p_j = 1.0 if j % 2 == 0 else -1.0
		M2[index[state], index[state]] += p_i * p_j * 1./N**2 * (total_h.n_i((1, state), i) - 0.5) * (total_h.n_i((1, state), j) - 0.5)

# M4
M4 = sp.dok_matrix((d,d))
for state in basis:
	for i, j, k, l in product(xrange(N), xrange(N), xrange(N), xrange(N)):
		p_i = 1.0 if i % 2 == 0 else -1.0
		p_j = 1.0 if j % 2 == 0 else -1.0
		p_k = 1.0 if k % 2 == 0 else -1.0
		p_l = 1.0 if l % 2 == 0 else -1.0
		M4[index[state], index[state]] += p_i * p_j * p_k * p_l * 1./N**4 * (total_h.n_i((1, state), i) - 0.5) * (total_h.n_i((1, state), j) - 0.5) * (total_h.n_i((1, state), k) - 0.5) * (total_h.n_i((1, state), l) - 0.5)

H = sp.csc_matrix(H)
beta = 5.
k_max = d
for k in xrange(1, k_max):
	Es, estates = sl.eigsh(H, k=k, which='SA')
	Z = sum(np.exp(-beta*E) for E in Es)
	Eexp = sum(E*np.exp(-beta*E)/Z for E in Es)
	V = np.matrix(estates)
	m2 = sum(np.exp(-beta*E)*np.trace(V.T*M2*V)/Z for E in Es)
	m4 = sum(np.exp(-beta*E)*np.trace(V.T*M4*V)/Z for E in Es)
	print "E(k=" + str(k) + ") = " + str(Eexp)
	print "m2(k=" + str(k) + ") = " + str(m2)
	print "m4(k=" + str(k) + ") = " + str(m4)
	print "B(k=" + str(k) + ") = " + str(m4/m2**2)
	print "----------------"
