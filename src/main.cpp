#include <iostream>
#include <boost/mpi.hpp>
#include "ctint.hpp"

int main(int argc, char* argv[])
{
	boost::mpi::environment env(argc, argv);
	
	double L=6, V=1.5, beta=5.0;
	int n_cycles = 1000000, n_cycle_length = 50, n_warmup_cycles = 500;
	if(argc==4)
	{
		L = atof(argv[1]);
		V = atof(argv[2]);
		beta = atoi(argv[3]);
		n_cycles = 1000000 ;
	}
	std::cout << "CT-INT with L=" << L << ", V=" << V << ", beta=" << beta
	  << std::endl;	
	ctint_solver mc(beta, 100);
	mc.solve(L, V, n_cycles, n_cycle_length, n_warmup_cycles, "", -1);

	return 0;
}
