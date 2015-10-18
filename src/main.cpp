#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <boost/mpi.hpp>
#include <triqs/mpi/base.hpp>
#include "ctint.hpp"

int main(int argc, char* argv[])
{
	boost::mpi::environment env(argc, argv);
	
	double L=6, V=1.5, beta=5.0;
	int n_cycles = 1000000, n_cycle_length = 50, n_warmup_cycles = 2000,
		n_slices = 500;
	if(argc==6)
	{
		L = std::atof(argv[1]);
		V = std::atof(argv[2]);
		beta = std::atoi(argv[3]);
		n_cycles = std::atoi(argv[4]);
		n_cycle_length = std::atoi(argv[5]);
	}
	int seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator(seed);
	long unsigned int base_seed = generator();
	ctint_solver mc(base_seed, beta, n_slices);
	mc.solve(L, V, n_cycles, n_cycle_length, n_warmup_cycles, "", -1);

	return 0;
}
