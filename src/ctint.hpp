#pragma once
#include <string>

// ------------ The main class of the solver -----------------------

class ctint_solver
{
	double beta;
	int n_slices;
	int worm_nhood_dist = -1;
	double zeta2 = 0.75;
	double zeta4 = 16.0;

	public:
		ctint_solver(double beta_, int n_slices_ = 100);
		
		// The method that runs the qmc
		void solve(int L, double V, int n_cycles, int length_cycle = 50,
			int n_warmup_cycles = 5000, std::string random_name = "",
			int max_time = -1);
 
};

