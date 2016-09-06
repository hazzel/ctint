#pragma once

struct parameters
{	
	double beta, V, mu, zeta2, zeta4;
	int worm_nhood_dist, n_matsubara, n_discrete_tau, n_prebin, n_static_cycles;
	double ratio_w2, ratio_w4;
	//Proposal probabilities
	std::vector<double> add;
	std::vector<double> rem;
	double W2toZ, ZtoW2, ZtoW4, W4toZ, W2toW4, W4toW2, worm_shift;
};
