#include <iostream>
#include <vector>
#include <map>
#include <functional>
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

namespace mp {
typedef boost::multiprecision::mpfr_float mp_float;

mp_float trace(arma::mat& M)
{
	boost::multiprecision::mpfr_float::default_precision(1000000);
	mp_float Z = 0;
	for (int_t i = 0; i < M.n_rows; ++i)
		Z += M(i, i);
	return Z;
}
mp_float trace(arma::mat&& M)
{
	boost::multiprecision::mpfr_float::default_precision(1000000);
	mp_float Z = 0;
	for (int_t i = 0; i < M.n_rows; ++i)
		Z += M(i, i);
	return Z;
}}

template<typename T>
void print_help(const T& desc)
{
	std::cout << desc << "n\n"; std::exit(1);
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
		("ensemble,e", po::value<std::string>(&ensemble)->default_value("c"),
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
			return hspace.n_el(state{1, state_id}) >= 0;
		else
			return hspace.n_el(state{1, state_id}) == lat.n_sites()/2; });
	std::cout << "Dimension of total Hilbert space: " << hspace.dimension()
		<< std::endl;
	std::cout << "Dimension of sub space: " << hspace.sub_dimension()
		<< std::endl;

	//Build Hamiltonian
	sparse_storage<int_t> H_st;
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
					H_st(hspace.index(m.id), n.second) += m.sign * (-1.);

				//Interaction: V sum_<ij> (n_i - 0.5) (n_j - 0.5)
				if (i < j)
				{
					H_st(n.second, n.second) += V * (hspace.n_i({1, n.first}, i)
						- 0.5) * (hspace.n_i({1, n.first}, j) - 0.5);
				}
			}
	});
	arma::sp_mat H = H_st.build_matrix();
	
	//Build observables
	sparse_storage<int_t> M2_st, M4_st;
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
	out.precision(12);
	arma::vec ev; arma::mat es;
	arma::eigs_sym(ev, es, H, std::min(hspace.sub_dimension()-2,
		static_cast<int_t>(k)), "sa");
	for (int t = 0; t <= temperature[2]; ++t)
	{
		boost::multiprecision::mpfr_float::default_precision(1000000);
		double T = temperature[0] + (temperature[1]- temperature[0])
			* static_cast<double>(t) / temperature[2];
		double beta = 1./T;
		std::cout << "T = " << T << std::endl;
		std::cout << "beta = " << beta << std::endl;
		arma::vec boltzmann(ev.n_rows);
		for (int i = 0; i < ev.n_rows; ++i)
			boltzmann(i) = std::exp(-beta * ev(i));
		arma::mat D = arma::diagmat(boltzmann);
		mp::mp_float Z = mp::trace(D);
		std::cout << "Z = " << Z << std::endl;
		mp::mp_float E = mp::trace(D*arma::diagmat(ev))/Z;
		mp::mp_float m2 = mp::trace(D*es.t()*M2*es)/Z;
		mp::mp_float m4 = mp::trace(D*es.t()*M4*es)/Z;
		out << k << "\t" << L << "\t" << V << "\t" << T << "\t"
			<< E << "\t" << m2 << "\t" << m4 << "\t" << m4/(m2*m2) << std::endl;
		std::cout << k << "\t" << L << "\t" << V << "\t" << T << "\t"
			<< E << "\t" << m2 << "\t" << m4 << "\t" << m4/(m2*m2) << std::endl;
	}
	out.close();
}