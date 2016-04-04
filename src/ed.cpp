#include <iostream>
#include <vector>
#include <map>
#include <functional>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <armadillo>
#include <boost/program_options.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include "lattice.h"
#include "honeycomb.h"
#include "hilbert.h"
#include "sparse_storage.h"

namespace po = boost::program_options;

namespace mp = boost::multiprecision;
typedef mp::number<mp::mpfr_float_backend<300> >  mp_float;

template<typename T>
void print_help(const T& desc)
{
	std::cout << desc << "n\n"; std::exit(1);
}

// Imaginary time observables
void write_imaginary_time_obs(std::ostream& out, arma::sp_mat& op,
	int Ntau, double beta, double Z, arma::vec& ev, arma::mat& es,
	arma::mat& esT, arma::vec& boltzmann)
{
	for (int n = 0; n <= Ntau; ++n)
	{
		mp_float tau = static_cast<double>(n) /static_cast<double>(Ntau)
			* beta / 2.;
		mp_float obs = mp_float(0.);
		for (int a = 0; a < ev.n_rows; ++a)
			for (int b = 0; b < ev.n_rows; ++b)
			{
				mp_float omega = ev(a) - ev(b);
				if (omega < 0. || omega > 0.)
				{
					obs += (boltzmann(b) - boltzmann(a))
						* mp::exp(-tau * omega) / (1. - mp::exp(-beta * omega))
						* arma::trace(esT.row(a) * op * es.col(b))
						* arma::trace(esT.row(b) * op * es.col(a));
				}
			}
		obs /= mp_float(Z);
		out << obs << "\t";
		std::cout << obs << "\t";
		std::cout.flush();
	}
}
// Matsubara frequency observables
void write_matsubara_obs(std::ostream& out, arma::sp_mat& op,
	int Nmat, double beta, double Z, arma::vec& ev, arma::mat& es,
	arma::mat& esT, arma::vec& boltzmann)
{
	for (int n = 0; n < Nmat; ++n)
	{
		double obs = 0.;
		for (int a = 0; a < ev.n_rows; ++a)
			for (int b = 0; b < ev.n_rows; ++b)
			{
				double omega = ev(a) - ev(b);
				if (omega < 0. || omega > 0.)
				{
					double omega_n = 2. * 4. * std::atan(1.) * n / beta;
					obs += (boltzmann(b) - boltzmann(a))
						* omega / (omega*omega + omega_n*omega_n)
						* arma::trace(esT.row(a) * op * es.col(b))
						* arma::trace(esT.row(b) * op * es.col(a));
				}
			}
		obs /= Z;
		out << obs << "\t";
		std::cout << obs << "\t";
		std::cout.flush();
	}
}

int main(int ac, char** av)
{
	int L;
	double V;
	std::vector<double> temperature;
	int k;
	std::string ensemble;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("L", po::value<int>(&L)->default_value(3), "linear lattice dimension")
		("V", po::value<double>(&V)->default_value(1.355), "interaction strength")
		("T", po::value<std::vector<double>>()->multitoken(),
		 "temperature interval: T_start T_end steps")
		("k", po::value<int>(&k)->default_value(100), "number of eigenstates")
		("ensemble,e", po::value<std::string>(&ensemble)->default_value("gc"),
			"ensemble: gc or c");
	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);

	if (vm.count("help")) print_help(desc);
	if (vm.count("T"))
		temperature = vm["T"].as<std::vector<double>>();
	else
		print_help(desc);
	std::cout << "L = " << L << std::endl;
	std::cout << "V = " << V << std::endl;
	if (temperature.size() != 3)
		print_help(desc);
	else
		std::cout << "T = " << temperature[0] << "..." << temperature[1]
			<< " in " << temperature[2] << " steps." << std::endl;
	std::cout << "ensemble: " << ensemble << std::endl;
	//Generate lattice
	honeycomb h(L);
	lattice lat;
	lat.generate_graph(h);
	lat.generate_neighbor_map("nearest neighbors", [&lat]
		(lattice::vertex_t i, lattice::vertex_t j) {
		return lat.distance(i, j) == 1; });
	//Generate hilbert space and build basis
	hilbert hspace(lat);
	hspace.build_basis([&hspace, &lat, &ensemble](int_t state_id) {
		if (ensemble == "gc")
			return hspace.n_el({1, state_id}) >= 0;
		else
			return hspace.n_el({1, state_id}) == lat.n_sites()/2; });
	std::cout << "Dimension of total Hilbert space: " << hspace.dimension()
		<< std::endl;
	std::cout << "Dimension of sub space: " << hspace.sub_dimension()
		<< std::endl;

	//Build Hamiltonian
	sparse_storage<int_t> H_st(hspace.sub_dimension());
	hspace.build_operator([&lat, &hspace, &H_st, V](const std::pair<int_t,
		int_t>& n)
	{
		for (int_t i = 0; i < lat.n_sites(); ++i)
			for (int_t j : lat.neighbors(i, "nearest neighbors"))
			{
				//Hopping term: -t sum_<ij> c_i^dag c_j
				state m = hspace.c_i({-1, n.first}, j);
				m = hspace.c_dag_i(m, i);
				if (m.sign != 0)
				{
					H_st(hspace.index(m.id), n.second) += m.sign * (-1.);
				}

				//Interaction: V sum_<ij> (n_i - 0.5) (n_j - 0.5)
				if (i < j)
				{
					H_st(n.second, n.second) += V * (hspace.n_i({1, n.first}, i)
						- 0.5) * (hspace.n_i({1, n.first}, j) - 0.5);
				}
			}
	});
	arma::sp_mat H = H_st.build_matrix();
	
	//Build static observables
	sparse_storage<int_t> M2_st(hspace.sub_dimension()), M4_st(hspace.sub_dimension());
	hspace.build_operator([&lat, &hspace, &M2_st, &M4_st, V]
		(const std::pair<int_t, int_t>& n)
	{
		for (int_t i = 0; i < lat.n_sites(); ++i)
			for (int_t j = 0; j < lat.n_sites(); ++j)
			{
				M2_st(n.second, n.second) += lat.parity(i) * lat.parity(j)
					/ std::pow(lat.n_sites(), 2.0) * (hspace.n_i({1, n.first}, i)
					- 0.5) * (hspace.n_i({1, n.first}, j) - 0.5);

				for (int_t k = 0; k < lat.n_sites(); ++k)
					for (int_t l = 0; l < lat.n_sites(); ++l)
						M4_st(n.second, n.second) += lat.parity(i) * lat.parity(j)
							* lat.parity(k) * lat.parity(l) / std::pow(lat.n_sites(),
							4.0) * (hspace.n_i({1, n.first},i) - 0.5)
							* (hspace.n_i({1, n.first}, j) - 0.5)
							* (hspace.n_i({1, n.first}, k) - 0.5)
							* (hspace.n_i({1, n.first}, l) - 0.5);
			}
	});
	arma::sp_mat M2 = M2_st.build_matrix();
	arma::sp_mat M4 = M4_st.build_matrix();
	std::cout << "Operator construction done." << std::endl;

	std::string out_file = "../data/ed_L_" + std::to_string(L) + "__"
		+ "V_" + std::to_string(V) + "__"
		+ "T_" + std::to_string(temperature[0]) + "_"
		+ std::to_string(temperature[1]) + "_" + std::to_string(temperature[2])
		+ "__" + ensemble;
	std::ofstream out("../data/" + out_file);
	std::cout.precision(8);
	out.precision(12);
	arma::vec ev; arma::mat es;
	if (hspace.sub_dimension() < 1000)
	{
		arma::mat H_dense(H);
		arma::eig_sym(ev, es, H_dense);
	}
	else
	{
		arma::eigs_sym(ev, es, H, std::min(hspace.sub_dimension()-2,
			static_cast<int_t>(k)), "sa");
	}
	arma::mat esT = es.t();
	
	// Build dynamic observables
	sparse_storage<int_t> ni_st(hspace.sub_dimension());
	sparse_storage<int_t> epsilon_st(hspace.sub_dimension());
	hspace.build_operator([&lat, &hspace, &ni_st, &epsilon_st]
		(const std::pair<int_t, int_t>& n)
		{
			for (int i = 0; i < lat.n_sites(); ++i)
			{
				ni_st(n.second, n.second)
					+= lat.parity(i) / lat.n_sites()
						* (hspace.n_i({1, n.first}, i) - 0.5);
				for (int j : lat.neighbors(i, "nearest neighbors"))
				{
					state p = hspace.c_i({-1, n.first}, j);
					p = hspace.c_dag_i(p, i);
					if (p.sign != 0)
						epsilon_st(hspace.index(p.id), n.second) += p.sign
							/ static_cast<double>(lat.n_bonds());
				}
			}
		});
	arma::sp_mat ni_op = ni_st.build_matrix();
	arma::sp_mat epsilon_op = epsilon_st.build_matrix();

	int tmax = (temperature[2] == 1) ? 0 : temperature[2];
	for (int t = 0; t <= tmax; ++t)
	{
		double T = temperature[0] + (temperature[1] - temperature[0])
			* static_cast<double>(t) / temperature[2];
		double beta = 1./T;
		std::cout << "T = " << T << std::endl;
		std::cout << "beta = " << beta << std::endl;
		
		arma::vec boltzmann(ev.n_rows);
		for (int i = 0; i < ev.n_rows; ++i)
			boltzmann(i) = std::exp(-beta * (ev(i) - ev.min()));
		double Z = 0., E = 0., m2 = 0., m4 = 0.;
		for (int i = 0; i < ev.n_rows; ++i)
		{
			Z += boltzmann(i);
			E += boltzmann(i) * ev(i);
			// Trace over 1x1 matrices
			m2 += boltzmann(i) * arma::trace(esT.row(i) * M2 * es.col(i));
			m4 += boltzmann(i) * arma::trace(esT.row(i) * M4 * es.col(i));
		}

		int Ntau = 200, Nmat = 20;
		out << k << "\t" << L << "\t" << V << "\t" << T << "\t"
			<< E/Z << "\t" << m2/Z << "\t" << m4/Z << "\t" << m4/(m2*m2) << "\t"
			<< Ntau << "\t" << Nmat << "\t";
		std::cout << k << "\t" << L << "\t" << V << "\t" << T << "\t"
			<< E/Z << "\t" << m2/Z << "\t" << m4/Z << "\t" << m4/(m2*m2) << "\t"
			<< Ntau << "\t" << Nmat << "\t";

		std::cout << std::endl << std::endl;

		write_imaginary_time_obs(out, ni_op, Ntau, beta, Z, ev, es, esT,
			boltzmann);
		write_matsubara_obs(out, ni_op, Ntau, beta, Z, ev, es, esT, boltzmann);

		write_imaginary_time_obs(out, epsilon_op, Ntau, beta, Z, ev, es, esT,
			boltzmann);
		write_matsubara_obs(out, epsilon_op, Ntau, beta, Z, ev, es, esT,
			boltzmann);

		out << std::endl;
		std::cout << std::endl;

		sparse_storage<int_t> n_total_st(hspace.sub_dimension());
		hspace.build_operator([&lat, &hspace, &n_total_st]
			(const std::pair<int_t, int_t>& n)
			{
				for (int_t i = 0; i < lat.n_sites(); ++i)
					n_total_st(n.second, n.second) += hspace.n_i({1, n.first}, i);
			});
		arma::sp_mat n_total = n_total_st.build_matrix();

		for (int a = 0; a < std::min(10,static_cast<int>(hspace.sub_dimension()));
			++a)
			std::cout << "E(" << a << ") = " << ev(a) << ", <" << a << "|n|"
				<< a << "> = " << arma::trace(esT.row(a) * n_total * es.col(a))
				<< std::endl;
	}
	out.close();
}
