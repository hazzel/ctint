#include <iostream>
#include <cassert>
#include <triqs/mc_tools.hpp>
#include <triqs/det_manip.hpp>
#include <triqs/statistics.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/complex.hpp>
#include "ctint.hpp"
#include "greens_function.h"
#include "fast_update.h"

// --------------- The QMC configuration ----------------

// Argument type
struct arg_t
{
	double tau;
	int site;
};

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

// The Monte Carlo configuration
struct configuration
{
	const lattice& l;
	triqs::det_manip::det_manip<full_g_entry> Mmatrix;
	fast_update<full_g_entry, arg_t> M;
	triqs::statistics::observable<double> obs_pert_order;

	//int perturbation_order() const { return Mmatrix.size() / 2; }
	int perturbation_order() const { return M.perturbation_order(); }

	configuration(const lattice& l_, const greens_function& g0)
		: l(l_), Mmatrix{full_g_entry{g0}, 100}, M{full_g_entry{g0}, l_},
			obs_pert_order()
	{}
};

// ------------ QMC move : inserting a vertex ------------------

struct move_insert
{
	configuration* config;
	triqs::mc_tools::random_generator& rng;
	double beta, V;

	double attempt()
	{
		double tau = rng(beta);
		int s1 = rng(config->l.n_sites());
		int s2 = config->l.neighbors(s1, 1)[rng(3)];
		int k = config->perturbation_order();
		//double det_ratio = config->Mmatrix.try_insert2(2*k, 2*k+1, 2*k,
		//	2*k+1, {tau, s1}, {tau, s2}, {tau, s1}, {tau, s2});
		double det_ratio = config->M.try_add({tau, s1}, {tau, s2});
		assert(det_ratio == det_ratio && "nan value in det ratio");
		return -beta * V * config->l.n_bonds() / (k + 1) * det_ratio;
	}

	double accept()
	{
		//config->Mmatrix.complete_operation(); // Finish insertion
		config->M.finish_add();
		return 1.0;
	}

	void reject() {}
};

// ------------ QMC move : deleting a vertex ------------------

struct move_remove
{
	configuration* config;
	triqs::mc_tools::random_generator& rng;
	double beta, V;

	double attempt()
	{
		int k = config->perturbation_order();
		if (k <= 0) return 0;
		int p = rng(k); // Choose one of the operators for removal
		//double det_ratio = config->Mmatrix.try_remove2(2*p, 2*p+1, 2*p, 2*p+1);
		double det_ratio = config->M.try_remove(p);
		assert(det_ratio == det_ratio && "nan value in det ratio");
		return -k / (beta * V * config->l.n_bonds()) * det_ratio;
	}

	double accept()
	{
		//config->Mmatrix.complete_operation(); // Finish removal
		config->M.finish_remove();
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
			<< triqs::statistics::average_and_error(config->obs_pert_order)
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
	greens_function g0;
	g0.generate_mesh(&l, beta, n_slices);
	auto config = configuration{l, g0};

	// Register moves and measurements
	CTQMC.add_move(move_insert{&config, CTQMC.rng(), beta, V}, "insertion");
	CTQMC.add_move(move_remove{&config, CTQMC.rng(), beta, V}, "removal");
	CTQMC.add_measure(measure_M{&config}, "M measurement");

	// Run and collect results
	CTQMC.start(1.0, triqs::utility::clock_callback(max_time));
	CTQMC.collect_results(world);
}
