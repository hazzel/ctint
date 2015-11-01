#include <iostream>
#include <map>
#include <functional>
#include <fstream>
#include <cmath>
#include <armadillo>
#include <boost/program_options.hpp>
#include "lattice.h"
#include "honeycomb.h"
#include "hilbert.h"
#include "sparse_storage.h"

namespace po = boost::program_options;

int main(int ac, char** av)
{
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("L", po::value<int>(), "linear lattice dimension")
		("V", po::value<double>(), "interaction strength")
		("ensemble", po::value<std::string>(), "ensemble: gc or c");
	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);

	int L = 3;
	double V = 1.355;
	std::string ensemble = "c";
	if (vm.count("help")) { std::cout << desc << "\n"; return 1; }
	if (vm.count("L"))
		L = vm["L"].as<int>();
	if (vm.count("V"))
		V = vm["V"].as<double>();
	if (vm.count("ensemble"))
		ensemble = vm["ensemble"].as<std::string>();
	std::cout << "L = " << L << std::endl;
	std::cout << "V = " << V << std::endl;
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

	std::ofstream out("../data/ed.txt");
	out.precision(10);
	double Tmin = 0.1, Tmax = 0.1;
	int N = 0, k = 50;
	for (int t = 0; t <= N; ++t)
	{
		double T = Tmin + (Tmax - Tmin) * static_cast<double>(t)
			/ static_cast<double>(N);
		double beta = 1./T;
		arma::vec ev;
		arma::mat es;
		arma::eigs_sym(ev, es, H, k, "sa");
		arma::vec boltzmann(ev.n_rows);
		for (int i = 0; i < ev.n_rows; ++i)
			boltzmann(i) = std::exp(-beta * ev(i));
		arma::mat D = arma::diagmat(boltzmann);
		long double Z = arma::trace(D);
		double E = arma::trace(D*arma::diagmat(ev))/Z;
		double m2 = arma::trace(D*es.t()*M2*es)/Z;
		double m4 = arma::trace(D*es.t()*M4*es)/Z;
		out << k << "\t" << L << "\t" << V << "\t" << T << "\t"
			<< E << "\t" << m2 << "\t" << m4 << "\t" << m4/(m2*m2) << std::endl;
	}
	out.close();
}
