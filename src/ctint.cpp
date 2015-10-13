#include <iostream>
#include <cassert>
#include <initializer_list>
#include <triqs/mc_tools.hpp>
#include <triqs/det_manip.hpp>
#include <triqs/statistics.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/complex.hpp>
#include "ctint.hpp"
#include "greens_function.h"
#include "fast_update.h"

#define eigen_impl

// --------------- The QMC configuration ----------------

// Argument type
struct arg_t
{
	double tau;
	int site;
};

enum { nn_int, worm };

// The function that appears in the calculation of the determinant
struct full_g_entry
{
	const greens_function& g0;

	double operator()(const arg_t& x, const arg_t& y) const
	{
		if ((x.tau == y.tau) && (x.site == y.site))
			return 0.0;
		else
			return g0(x.tau - y.tau, x.site, y.site);
	}
};

struct parameters
{	
	double beta, V, zeta2, zeta4;
	int worm_nhood_dist;
};

// The Monte Carlo configuration
struct configuration
{
	const lattice& l;
	triqs::det_manip::det_manip<full_g_entry> Mmatrix;
	fast_update<full_g_entry, arg_t> M;
	parameters params;
	triqs::statistics::observable<double> obs_pert_order;
	triqs::statistics::observable<double> ZtoW2_acc;
	triqs::statistics::observable<double> W2toZ_acc;

	#ifndef eigen_impl
	int perturbation_order() const { return Mmatrix.size() / 2; }
	#else
	int perturbation_order() const { return M.perturbation_order(nn_int); }
	int worms() const { return M.perturbation_order(worm); }
	#endif

	configuration(const lattice& l_, const greens_function& g0, 
		const parameters& params_)
		: l(l_), Mmatrix{full_g_entry{g0}, 100}, M{full_g_entry{g0}, l_, 2},
			params(params_), obs_pert_order()
	{
		#ifndef eigen_impl
		std::cout << "Using 'triqs' for fast updates." << std::endl;
		#else
		std::cout << "Using 'eigen' for fast updates." << std::endl;
		#endif
	}
};

// ------------ QMC move : inserting a vertex ------------------

struct move_insert
{
	configuration* config;
	triqs::mc_tools::random_generator& rng;

	double attempt()
	{
		double tau = rng(config->params.beta);
		int s1 = rng(config->l.n_sites());
		int s2 = config->l.neighbors(s1, 1)[rng(3)];
		int k = config->perturbation_order();
		#ifndef eigen_impl
		double det_ratio = config->Mmatrix.try_insert2(2*k, 2*k+1, 2*k,
			2*k+1, {tau, s1}, {tau, s2}, {tau, s1}, {tau, s2});
		#else
		std::vector<arg_t> vec = {arg_t{tau, s1}, arg_t{tau, s2}};
		double det_ratio = config->M.try_add<1>(vec, nn_int);
		#endif
		assert(det_ratio == det_ratio && "nan value in det ratio");
		return -config->params.beta * config->params.V * config->l.n_bonds()
			/ (k + 1) * det_ratio;
	}

	double accept()
	{
		#ifndef eigen_impl
		config->Mmatrix.complete_operation(); // Finish insertion
		#else
		config->M.finish_add();
		#endif
		return 1.0;
	}

	void reject() {}
};

// ------------ QMC move : Z -> W2 ------------------

struct move_ZtoW2
{
	configuration* config;
	triqs::mc_tools::random_generator& rng;

	double attempt()
	{
		if (config->worms() != 0) return 0.0;
		double tau = rng(config->params.beta);
		int s1 = rng(config->l.n_sites());
		const std::vector<int>& neighbors =
			config->l.neighbors(s1, config->params.worm_nhood_dist);
		int s2 = neighbors[rng(neighbors.size())];
		std::vector<arg_t> vec = {arg_t{tau, s1}, arg_t{tau, s2}};
		double det_ratio = config->M.try_add<1>(vec, worm);
		assert(det_ratio == det_ratio && "nan value in det ratio");
		double acc = config->l.parity(s1) * config->l.parity(s2)
			* config->params.zeta2 * neighbors.size() * det_ratio;
		config->ZtoW2_acc << std::min(1.0, acc);
		return acc;
	}

	double accept()
	{
		config->M.finish_add();
		return 1.0;
	}

	void reject() {}
};

// ------------ QMC move : W2 -> Z ------------------

struct move_W2toZ
{
	configuration* config;
	triqs::mc_tools::random_generator& rng;

	double attempt()
	{
		int w = config->worms();
		if (w != 1) return 0.0;
		int p = rng(w);
		int dist = config->l.distance(config->M.vertex(p, worm).site,
			config->M.vertex(p+1, worm).site);
		double acc = 0.0;
		if (dist > 0 && dist <= config->params.worm_nhood_dist)
		{
			std::vector<int> vec = {p};
			double det_ratio = config->M.try_remove<1>(vec, worm);
			assert(det_ratio == det_ratio && "nan value in det ratio");
			const std::vector<int>& neighbors =
				config->l.neighbors(0, config->params.worm_nhood_dist);
			acc = config->l.parity(config->M.vertex(p, worm).site)
				* config->l.parity(config->M.vertex(p+1, worm).site)
				/ config->params.zeta2 / neighbors.size() * det_ratio;
		}
		config->W2toZ_acc << std::min(1.0, acc);
		return acc;
	}

	double accept()
	{
		config->M.finish_remove();
		return 1.0;
	}

	void reject() {}
};

// ------------ QMC move : deleting a vertex ------------------

struct move_remove
{
	configuration* config;
	triqs::mc_tools::random_generator& rng;

	double attempt()
	{
		int k = config->perturbation_order();
		if (k <= 0) return 0;
		int p = rng(k); // Choose one of the operators for removal
		#ifndef eigen_impl
		double det_ratio = config->Mmatrix.try_remove2(2*p, 2*p+1, 2*p, 2*p+1);
		#else
		std::vector<int> vec = {p};
		double det_ratio = config->M.try_remove<1>(vec, nn_int);
		#endif
		assert(det_ratio == det_ratio && "nan value in det ratio");
		return -k / (config->params.beta * config->params.V
			* config->l.n_bonds()) * det_ratio;
	}

	double accept()
	{
		#ifndef eigen_impl
		config->Mmatrix.complete_operation(); // Finish removal
		#else
		config->M.finish_remove();
		#endif
		return 1.0;
	}

	void reject() {}
};

//  -------------- QMC measurement ----------------

struct measure_M
{
	configuration* config;
	double Z = 0, k = 0;

	measure_M(configuration* config_)
		: config(config_) {}

	void accumulate(double sign)
	{
		Z += sign;
		k += config->perturbation_order();
		config->obs_pert_order << config->perturbation_order();
	};

	void collect_results(const boost::mpi::communicator& c)
	{
		boost::mpi::all_reduce(c, Z, Z, std14::plus<>());
		boost::mpi::all_reduce(c, k, k, std14::plus<>());
		
		if (c.rank() == 0)
		{
			std::cout << "Average perturbation order = "
			<< triqs::statistics::make_jackknife(config->obs_pert_order)
			<< std::endl;
			std::cout << "ZtoW2 = "
			<< triqs::statistics::average_and_error(config->ZtoW2_acc)
			<< std::endl;
			std::cout << "W2toZ = "
			<< triqs::statistics::average_and_error(config->W2toZ_acc)
			<< std::endl;
		}
	}
};

// ------------ The main class of the solver ------------------------

ctint_solver::ctint_solver(double beta_, int n_slices_)
	: beta(beta_), n_slices(n_slices_)
{}

// The method that runs the qmc
void ctint_solver::solve(int L, double V, int n_cycles, int length_cycle,
	int n_warmup_cycles, std::string random_name, int max_time)
{
	boost::mpi::communicator world;

	// Rank-specific variables
	int verbosity = (world.rank() == 0 ? 3 : 0);
	int random_seed = 34789 + 928374 * world.rank();

	// Construct a Monte Carlo loop
	triqs::mc_tools::mc_generic<double> CTQMC(n_cycles, length_cycle,
		n_warmup_cycles, random_name, random_seed, verbosity);

	// Prepare the configuration
	honeycomb h(L);
	lattice l;
	l.generate_graph(h);
	if (worm_nhood_dist == -1)
		worm_nhood_dist = l.max_distance();
	l.generate_neighbor_map(worm_nhood_dist);
	greens_function g0;
	g0.generate_mesh(&l, beta, n_slices);
	auto config = configuration{l, g0,
		{beta, V, zeta2, zeta4, worm_nhood_dist}};

	// Register moves and measurements
	CTQMC.add_move(move_insert{&config, CTQMC.rng()}, "insertion");
	CTQMC.add_move(move_remove{&config, CTQMC.rng()}, "removal");
	CTQMC.add_move(move_ZtoW2{&config, CTQMC.rng()}, "Z -> W2");
	CTQMC.add_move(move_W2toZ{&config, CTQMC.rng()}, "W2 -> Z");
	CTQMC.add_measure(measure_M{&config}, "M measurement");

	// Run and collect results
	CTQMC.start(1.0, triqs::utility::clock_callback(max_time));
	CTQMC.collect_results(world);
}
