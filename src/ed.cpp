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

namespace mp = boost::multiprecision;
typedef mp::number<mp::mpfr_float_backend<300> >  mp_float;

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
	std::cout.precision(8);
	out.precision(12);
	arma::vec ev; arma::mat es;
	if (hspace.sub_dimension() < 1000)
	{
		arma::mat H_dense(H);
		arma::eig_sym(ev, es, H_dense);
		std::cout << H_dense << std::endl;
		std::cout << ev << std::endl;
	}
	else
	{
		arma::eigs_sym(ev, es, H, std::min(hspace.sub_dimension()-2,
			static_cast<int_t>(k)), "sa");
	}
	arma::mat esT = es.t();
	
	// Build dynamic observables
	std::vector<arma::sp_mat> n_i(lat.n_sites());
	for (int_t i = 0; i < lat.n_sites(); ++i)
	{
		sparse_storage<int_t> ni_st;
		hspace.build_operator([&lat, &hspace, &ni_st, i]
			(const std::pair<int_t, int_t>& n) { ni_st(n.second, n.second)
				+= hspace.n_i({1, n.first}, i) - 0.5; });
		n_i[i] = ni_st.build_matrix();
	}

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

		int Ntau = 200;
		out << k << "\t" << L << "\t" << V << "\t" << T << "\t"
			<< E/Z << "\t" << m2/Z << "\t" << m4/Z << "\t" << m4/(m2*m2) << "\t"
			<< Ntau << "\t";
		std::cout << k << "\t" << L << "\t" << V << "\t" << T << "\t"
			<< E/Z << "\t" << m2/Z << "\t" << m4/Z << "\t" << m4/(m2*m2) << "\t"
			<< Ntau << "\t";

		std::cout << std::endl << std::endl;
		// Dynamic structure factor
		for (int n = 0; n <= Ntau; ++n)
		{
			arma::sp_mat M2_tau(H.n_rows, H.n_cols);
			arma::vec U_vec(ev.n_rows), Ut_vec(ev.n_rows);
			double tau = static_cast<double>(n) /static_cast<double>(Ntau)
				* beta / 2.;
			for (int i = 0; i < ev.n_rows; ++i)
			{
				U_vec(i) = std::exp(-tau * (ev(i) - (ev.max() + ev.min())/2.));
				Ut_vec(i) = std::exp(tau * (ev(i) - (ev.max() + ev.min())/2.));
			}
			arma::mat U = es * arma::diagmat(U_vec) * es.t();
			arma::mat Ut = es * arma::diagmat(Ut_vec) * es.t();
			double m2_tau = 0.;
			int i = 0;
			for (int_t j = 0; j < lat.n_sites(); ++j)
			{
				M2_tau = lat.parity(i) * lat.parity(j) / lat.n_sites()
					* Ut * n_i[i] * U * n_i[j];
				for (int k = 0; k < ev.n_rows; ++k)
					m2_tau += boltzmann(k) * arma::trace(esT.row(k) * M2_tau
						* es.col(k));
			}
			m2_tau /= Z;
			out << m2_tau << "\t";
			std::cout << m2_tau << "\t";
			std::cout.flush();
		}
		// Matsubara structure factor
		int Nmat = 100;
		out << Nmat << "\t";
		std::cout << Nmat << "\t";
		for (int n = 0; n < Nmat; ++n)
		{
			double m2_mat = 0.;
			int i = 0;
			for (int j = 0; j < lat.n_sites(); ++j)
				for (int a = 0; a < ev.n_rows; ++a)
					for (int b = 0; b < ev.n_rows; ++b)
					{
						double omega = ev(a) - ev(b);
						if (omega < 0. || omega > 0.)
						{
							double omega_n = 2. * 4. * std::atan(1.) * n * T;
							m2_mat += lat.parity(i) * lat.parity(j) / lat.n_sites()
								* (boltzmann(a) - boltzmann(b))
								* omega / (omega*omega + omega_n*omega_n)
								* arma::trace(esT.row(a) * n_i[i] * es.col(b))
								* arma::trace(esT.row(b) * n_i[j] * es.col(a));
						}
					}
			m2_mat /= Z;
			out << m2_mat << "\t";
			std::cout << m2_mat << "\t";
			std::cout.flush();
		}
		out << std::endl;
		std::cout << std::endl;

		for (int_t i = 0; i < lat.n_sites(); ++i)
		{
			sparse_storage<int_t> ni_st;
			hspace.build_operator([&lat, &hspace, &ni_st, i]
				(const std::pair<int_t, int_t>& n) { ni_st(n.second, n.second)
					+= hspace.n_i({1, n.first}, i); });
			n_i[i] = ni_st.build_matrix();
		}
		std::cout << "GS: " << std::endl;
		std::cout << "E(0) = " << ev(0) << std::endl;
		std::cout << "E(1) = " << ev(1) << std::endl;
		for (int i = 0; i < hspace.sub_dimension(); ++i)
		{
			if (std::abs(es.col(1)(i, 0)) > 0.0000001)
			{
				double n = 0.;
				for (int j = 0; j < lat.n_sites(); ++j)
					n += n_i[j](i, i);
				std::cout << "state: " << i << " with n=" << n 
				<< " and amplitude " << es.col(1)(i, 0) << std::endl;
			}
			else
			{
				double n = 0.;
				for (int j = 0; j < lat.n_sites(); ++j)
					n += n_i[j](i, i);
				std::cout << "no overlap with state " << i << " with n=" << n
					<< std::endl;
			}
		}
		double n = 0.;
		for (int i = 0; i < lat.n_sites(); ++i)
			n += arma::trace(esT.row(0) * n_i[i] * es.col(0));
		std::cout << "<GS|n|GS> = " << n << std::endl;
		n = 0.;
		for (int i = 0; i < lat.n_sites(); ++i)
			n += arma::trace(esT.row(1) * n_i[i] * es.col(1));
		std::cout << "<1|n|1> = " << n << std::endl;
	}
	out.close();
}
