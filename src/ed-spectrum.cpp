#include <iostream>
#include <vector>
#include <map>
#include <functional>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <armadillo>
#include <boost/program_options.hpp>
#include "lattice.h"
#include "honeycomb.h"
#include "hilbert.h"
#include "sparse_storage.h"

namespace po = boost::program_options;

template<typename T>
void print_help(const T& desc)
{
	std::cout << desc << "n\n"; std::exit(1);
}

int main(int ac, char** av)
{
	int L;
	std::vector<double> V;
	int k;
	std::string ensemble;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("L", po::value<int>(&L)->default_value(3), "linear lattice dimension")
		("V", po::value<std::vector<double>>()->multitoken(),
		 "interaction strength interval: V_start V_end steps")
		("k", po::value<int>(&k)->default_value(10), "number of eigenstates")
		("ensemble,e", po::value<std::string>(&ensemble)->default_value("c"),
			"ensemble: gc or c");
	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);

	if (vm.count("help")) print_help(desc);
	if (vm.count("V"))
		V = vm["V"].as<std::vector<double>>();
	else
		print_help(desc);
	std::cout << "L = " << L << std::endl;
	if (V.size() != 3)
		print_help(desc);
	else
		std::cout << "V = " << V[0] << "..." << V[1]
			<< " in " << V[2] << " steps." << std::endl;
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
		
	std::string out_file = "ed_L_" + std::to_string(L) + "__"
		+ "V_" + std::to_string(V[0]) + "_"
		+ std::to_string(V[1]) + "_" + std::to_string(V[2])
		+ "__" + ensemble;
	std::ofstream out("../data/spectrum/" + out_file);
	out.precision(12);
	
	//Build free Hamiltonian
	sparse_storage<int_t> H0_st;
	hspace.build_operator([&lat, &hspace, &H0_st](const std::pair<int_t,
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
					H0_st(hspace.index(m.id), n.second) += m.sign * (-1.);
				}
			}
	});

	for (int i = 0; i <= V[2]; ++i)
	{
		double v = V[0] + (V[1] - V[0])
			* static_cast<double>(i) / V[2];

		//Build Hamiltonian
		sparse_storage<int_t> H_st = H0_st;
		hspace.build_operator([&lat, &hspace, &H_st, v](const std::pair<int_t,
			int_t>& n)
		{
			for (int_t i = 0; i < lat.n_sites(); ++i)
				for (int_t j : lat.neighbors(i, "nearest neighbors"))
				{
					//Interaction: V sum_<ij> (n_i - 0.5) (n_j - 0.5)
					if (i < j)
					{
						H_st(n.second, n.second) += v * (hspace.n_i({1, n.first}, i)
							- 0.5) * (hspace.n_i({1, n.first}, j) - 0.5);
					}
				}
		});
		arma::sp_mat H = H_st.build_matrix();
		std::cout << "Operator construction done." << std::endl;

		arma::vec ev; arma::mat es;
		arma::eigs_sym(ev, es, H, std::min(hspace.sub_dimension()-2,
			static_cast<int_t>(k)), "sa");
			
		out << k << "\t" << L << "\t" << v << "\t";
		for (int j = 0; j < k; ++j)
			out << ev(j) << "\t";
		out << std::endl;
		std::cout << k << "\t" << L << "\t" << v << "\t";
		for (int j = 0; j < k; ++j)
			std::cout << ev(j) << "\t";
		std::cout << std::endl;
	}
	out.close();
}
