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

// Imaginary time observables
template<typename T>
std::vector<T> get_imaginary_time_obs(arma::SpMat<T>& op, int Ntau,
	double beta, double parity, double Z, arma::vec& ev, arma::mat& es,
	arma::mat& esT, arma::vec& boltzmann)
{
	arma::Mat<T> es_cx = arma::conv_to<arma::Mat<T>>::from(es);
	arma::Mat<T> esT_cx = arma::conv_to<arma::Mat<T>>::from(esT);
	std::vector<T> obs_vec(Ntau + 1);
	for (int n = 0; n <= Ntau; ++n)
	{
		double tau = static_cast<double>(n) / static_cast<double>(Ntau) * beta;
		obs_vec[n] = T(0.);
		for (int a = 0; a < ev.n_rows; ++a)
			for (int b = 0; b < ev.n_rows; ++b)
				obs_vec[n] += boltzmann(a) * std::exp(tau * (ev(a) - ev(b)))
					* arma::trace(esT_cx.row(a) * op * es_cx.col(b))
					* arma::trace(esT_cx.row(b) * op.t() * es_cx.col(a));
		obs_vec[n] /= Z;
	}
	return obs_vec;
}

// Matsubara frequency observables
template<typename T>
std::vector<T> get_matsubara_obs(arma::SpMat<T>& op, int Nmat, double beta,
	double parity, double Z, arma::vec& ev, arma::mat& es, arma::mat& esT,
	arma::vec& boltzmann)
{
	arma::Mat<T> es_cx = arma::conv_to<arma::Mat<T>>::from(es);
	arma::Mat<T> esT_cx = arma::conv_to<arma::Mat<T>>::from(esT);
	std::vector<T> obs_vec(Nmat);
	for (int n = 0; n < Nmat; ++n)
	{
		obs_vec[n] = T(0.);
		for (int a = 0; a < ev.n_rows; ++a)
			for (int b = 0; b < ev.n_rows; ++b)
			{
				double omega = ev(a) - ev(b);
				if (n > 0 || parity == -1)
				{
					double omega_n = (2.*n+(1.-parity)/2.) * 4.*std::atan(1.) / beta;
					obs_vec[n] += -(boltzmann(a) - parity * boltzmann(b))
						* omega / (omega*omega + omega_n*omega_n)
						* arma::trace(esT_cx.row(a) * op * es_cx.col(b))
						* arma::trace(esT_cx.row(b) * op.t() * es_cx.col(a));
				}
			}
		obs_vec[n] /= Z;
	}
	if (parity == 1)
	{
		// Treat n=0 separately
		obs_vec[0] = T(0.);
		for (int a = 0; a < ev.n_rows; ++a)
			for (int b = 0; b < ev.n_rows; ++b)
			{
				if (ev(a) < ev(b) || ev(a) > ev(b))
					obs_vec[0] += (boltzmann(b) - boltzmann(a))
						/ (ev(a) - ev(b))
						* arma::trace(esT_cx.row(a) * op * es_cx.col(b))
						* arma::trace(esT_cx.row(b) * op.t() * es_cx.col(a));
				else
				{
					obs_vec[0] += boltzmann(a) * beta
						* arma::trace(esT_cx.row(a) * op * es_cx.col(b))
						* arma::trace(esT_cx.row(b) * op.t() * es_cx.col(a));
				}
			}
		obs_vec[0] /= Z;
	}
	return obs_vec;
}

int main(int ac, char** av)
{
	int L;
	double V;
	double T;
	int k;
	std::string ensemble;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("L", po::value<int>(&L)->default_value(3), "linear lattice dimension")
		("V", po::value<double>(&V)->default_value(1.355), "interaction strength")
		("T", po::value<double>(&T)->default_value(0.1), "temperature")
		("k", po::value<int>(&k)->default_value(100), "number of eigenstates")
		("ensemble,e", po::value<std::string>(&ensemble)->default_value("gc"),
			"ensemble: gc or c");
	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);

	if (vm.count("help")) print_help(desc);
	std::cout << "L = " << L << std::endl;
	std::cout << "V = " << V << std::endl;
	std::cout << "T = " << T << std::endl;
	std::cout << "ensemble: " << ensemble << std::endl;
	//Generate lattice
	honeycomb h(L);
	lattice lat;
	lat.generate_graph(h);
	h.generate_maps(lat);

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
	sparse_storage<double, int_t> H_st(hspace.sub_dimension());
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
	sparse_storage<double, int_t> M2_st(hspace.sub_dimension()),
		M4_st(hspace.sub_dimension()), cij_st(hspace.sub_dimension());
	hspace.build_operator([&] (const std::pair<int_t, int_t>& n)
	{
		for (int_t i = 0; i < lat.n_sites(); ++i)
		{
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
			for (int_t j : lat.neighbors(i, "nearest neighbors"))
			{
					state p = hspace.c_i({1, n.first}, j);
					p = hspace.c_dag_i(p, i);
					if (p.sign != 0)
						cij_st(hspace.index(p.id), n.second) += p.sign
							/ static_cast<double>(lat.n_bonds());
			}
		}
	});
	arma::sp_mat M2 = M2_st.build_matrix();
	arma::sp_mat M4 = M4_st.build_matrix();
	arma::sp_mat Cij = cij_st.build_matrix();
	std::cout << "Operator construction done." << std::endl;

	std::string out_file = "../data/ed_L_" + std::to_string(L) + "__"
		+ "V_" + std::to_string(V) + "__"
		+ "T_" + std::to_string(T) + "__"
		+ ensemble;
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
	
	// Static expectation values
	double beta = 1./T;
	std::cout << "T = " << T << std::endl;
	std::cout << "beta = " << beta << std::endl;
	
	arma::vec boltzmann(ev.n_rows);
	for (int i = 0; i < ev.n_rows; ++i)
		boltzmann(i) = std::exp(-beta * (ev(i) - ev.min()));
	double Z = 0., E = 0., m2 = 0., m4 = 0., cij = 0.;
	for (int i = 0; i < ev.n_rows; ++i)
	{
		Z += boltzmann(i);
		E += boltzmann(i) * ev(i);
		// Trace over 1x1 matrices
		m2 += boltzmann(i) * arma::trace(esT.row(i) * M2 * es.col(i));
		m4 += boltzmann(i) * arma::trace(esT.row(i) * M4 * es.col(i));
		cij += boltzmann(i) * arma::trace(esT.row(i) * Cij * es.col(i));
	}

	int Ntau = 100, Nmat = 20;
	out << k << "\t" << L << "\t" << V << "\t" << T << "\t"
		<< E/Z << "\t" << m2/Z << "\t" << m4/Z << "\t" << m4/(m2*m2) << "\t"
		<< Ntau << "\t" << Nmat << std::endl;
	std::cout << k << "\t" << L << "\t" << V << "\t" << T << "\t"
		<< E/Z << "\t" << m2/Z << "\t" << m4/Z << "\t" << m4/(m2*m2) << "\t"
		<< Ntau << "\t" << Nmat << std::endl;
	std::cout << "<epsilon> = " << cij/Z << std::endl;
	
	// Build dynamic observables
	sparse_storage<double, int_t> ni_st(hspace.sub_dimension());
	sparse_storage<double, int_t> kekule_st(hspace.sub_dimension());
	sparse_storage<double, int_t> chern_st(hspace.sub_dimension());
	sparse_storage<double, int_t> epsilon_st(hspace.sub_dimension());
	sparse_storage<double, int_t> epsilon_nn_st(hspace.sub_dimension());
	sparse_storage<std::complex<double>, int_t> sp_st(hspace.sub_dimension());
	sparse_storage<std::complex<double>, int_t> sp_2_st(hspace.sub_dimension());
	sparse_storage<std::complex<double>, int_t> tp_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			//kekule
			for (auto& b : lat.bonds("kekule"))
			{
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					kekule_st(hspace.index(p.id), n.second) += p.sign
						/ static_cast<double>(lat.n_bonds());
			}
			for (auto& b : lat.bonds("kekule_2"))
			{
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					kekule_st(hspace.index(p.id), n.second) += -p.sign
						/ static_cast<double>(lat.n_bonds());
			}
			
			//chern
			for (auto& b : lat.bonds("chern"))
			{
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					chern_st(hspace.index(p.id), n.second) += -p.sign;
				p = hspace.c_i({1, n.first}, b.first);
				p = hspace.c_dag_i(p, b.second);
				if (p.sign != 0)
					chern_st(hspace.index(p.id), n.second) += p.sign;
			}

			for (int i = 0; i < lat.n_sites(); ++i)
			{
				//CDW
				ni_st(n.second, n.second)
					+= lat.parity(i) / lat.n_sites()
						* (hspace.n_i({1, n.first}, i) - 0.5);

				//epsilon
				for (int j : lat.neighbors(i, "nearest neighbors"))
				{
					state p = hspace.c_i({1, n.first}, j);
					p = hspace.c_dag_i(p, i);
					if (p.sign != 0)
						epsilon_st(hspace.index(p.id), n.second) += p.sign
							/ static_cast<double>(lat.n_bonds());
							
					epsilon_nn_st(n.second, n.second)
						+= (hspace.n_i({1, n.first}, i) - 0.5)
						* (hspace.n_i({1, n.first}, j) - 0.5)
						/ static_cast<double>(lat.n_bonds());
				}
				
				//sp
				auto& K = lat.symmetry_point("K");
				std::complex<double> phase = std::exp(std::complex<double>(0.,
					K.dot(lat.real_space_coord(i))));
				state p = hspace.c_i({1, n.first}, i);
				if (p.sign != 0)
					sp_st(hspace.index(p.id), n.second) += phase
						* std::complex<double>(p.sign);
				p = hspace.c_dag_i({1, n.first}, i);
				if (p.sign != 0)
					sp_2_st(hspace.index(p.id), n.second) += phase
						* std::complex<double>(p.sign);
				
				//tp
				for (int j = 0; j < lat.n_sites(); ++j)
				{
					phase = std::exp(std::complex<double>(0.,
						K.dot(lat.real_space_coord(i) - lat.real_space_coord(j))));
					state p = hspace.c_i({1, n.first}, i);
					p = hspace.c_i(p, j);
					if (p.sign != 0)
						tp_st(hspace.index(p.id), n.second) += phase
							* std::complex<double>(p.sign);
				}
			}
			epsilon_st(n.second, n.second) -= cij / Z;
		});
	arma::sp_mat ni_op = ni_st.build_matrix();
	arma::sp_mat kekule_op = kekule_st.build_matrix();
	arma::sp_mat chern_op = chern_st.build_matrix();
	arma::sp_mat epsilon_op = epsilon_st.build_matrix();
	arma::sp_mat epsilon_nn_op = epsilon_nn_st.build_matrix();
	arma::sp_cx_mat sp_op = sp_st.build_matrix();
	arma::sp_cx_mat sp_2_op = sp_2_st.build_matrix();
	arma::sp_cx_mat tp_op = tp_st.build_matrix();

	std::vector<std::vector<double>> obs_data;
	obs_data.emplace_back(get_imaginary_time_obs(ni_op, Ntau, beta, 1., Z, ev,
		es, esT, boltzmann));
	obs_data.emplace_back(get_matsubara_obs(ni_op, Nmat, beta, 1., Z, ev, es,
		esT, boltzmann));
	
	obs_data.emplace_back(get_imaginary_time_obs(kekule_op, Ntau, beta, 1., Z,
		ev, es, esT, boltzmann));
	obs_data.emplace_back(get_matsubara_obs(kekule_op, Nmat, beta, 1., Z, ev, es,
		esT, boltzmann));
	
	obs_data.emplace_back(get_imaginary_time_obs(chern_op, Ntau, beta, 1., Z,
		ev, es, esT, boltzmann));
	obs_data.emplace_back(get_matsubara_obs(chern_op, Nmat, beta, 1., Z, ev, es,
		esT, boltzmann));

	obs_data.emplace_back(get_imaginary_time_obs(epsilon_op, Ntau, beta, 1., Z,
		ev, es, esT, boltzmann));
	obs_data.emplace_back(get_matsubara_obs(epsilon_op, Nmat, beta, 1., Z, ev,
		es, esT, boltzmann));
	
	obs_data.emplace_back(get_imaginary_time_obs(epsilon_nn_op, Ntau, beta, 1.,
		Z, ev, es, esT, boltzmann));
	obs_data.emplace_back(get_matsubara_obs(epsilon_nn_op, Nmat, beta, 1., Z, ev,
		es, esT, boltzmann));

	std::vector<std::vector<std::complex<double>>> obs_data_cx;
	obs_data_cx.emplace_back(get_imaginary_time_obs(sp_op, Ntau, beta, 1., Z,
		ev, es, esT, boltzmann));
	obs_data_cx.emplace_back(get_matsubara_obs(sp_op, Nmat, beta, 1., Z, ev,
		es, esT, boltzmann));
	
	obs_data_cx.emplace_back(get_imaginary_time_obs(sp_2_op, Ntau, beta, 1., Z,
		ev, es, esT, boltzmann));
	obs_data_cx.emplace_back(get_matsubara_obs(sp_2_op, Nmat, beta, 1., Z, ev,
		es, esT, boltzmann));
	
	obs_data_cx.emplace_back(get_imaginary_time_obs(tp_op, Ntau, beta, 1., Z,
		ev, es, esT, boltzmann));
	obs_data_cx.emplace_back(get_matsubara_obs(tp_op, Nmat, beta, 1., Z, ev,
		es, esT, boltzmann));

	for (int i = 0; i < obs_data.size(); ++i)
	{
		for (int j = 0; j < obs_data[i].size(); ++j)
		{
			out << obs_data[i][j] << "\t";
			std::cout << obs_data[i][j] << "\t";
		}
		out << std::endl;
		std::cout << std::endl;
	}
	for (int i = 0; i < obs_data_cx.size(); ++i)
	{
		for (int j = 0; j < obs_data_cx[i].size(); ++j)
		{
			out << std::real(obs_data_cx[i][j]) << "\t";
			std::cout << std::real(obs_data_cx[i][j]) << "\t";
		}
		out << std::endl;
		std::cout << std::endl;
	}

	sparse_storage<double, int_t> n_total_st(hspace.sub_dimension());
	hspace.build_operator([&lat, &hspace, &n_total_st]
		(const std::pair<int_t, int_t>& n)
		{
			for (int_t i = 0; i < lat.n_sites(); ++i)
				n_total_st(n.second, n.second) += hspace.n_i({1, n.first}, i);
		});
	arma::sp_mat n_total = n_total_st.build_matrix();

	//for (int a = 0; a < std::min(100, static_cast<int>(hspace.sub_dimension()));
	//	++a)
	for (int a = 0; a < static_cast<int>(hspace.sub_dimension()); ++a)
	{
		double e = arma::trace(esT.row(a) * epsilon_op * es.col(0));
		if (a == 0 || std::abs(e) > std::pow(10, -12))
		{
			std::cout << "E(" << a << ") = " << ev(a) << ", <" << a << "|n|"
				<< a << "> = " << arma::trace(esT.row(a) * n_total * es.col(a))
				<< ", <" << a << "|epsilon|"
				<< 0 << "> = " << e
				<< std::endl;
		}
	}
	for (int a = 0; a < static_cast<int>(hspace.sub_dimension()); ++a)
	{
		double e = arma::trace(esT.row(a) * n_total * es.col(0));
		if (a == 0 || std::abs(e) > std::pow(10, -12))
		{
			std::cout << "E(" << a << ") = " << ev(a) << ", <" << a << "|n|"
				<< a << "> = " << arma::trace(esT.row(a) * n_total * es.col(a))
				<< ", <" << a << "|n|"
				<< 0 << "> = " << e
				<< std::endl;
		}
	}
	for (int a = 0; a < static_cast<int>(hspace.sub_dimension()); ++a)
	{
		double e = arma::trace(esT.row(a) * kekule_op * es.col(0));
		if (a == 0 || std::abs(e) > std::pow(10, -12))
		{
			std::cout << "E(" << a << ") = " << ev(a) << ", <" << a << "|n|"
				<< a << "> = " << arma::trace(esT.row(a) * n_total * es.col(a))
				<< ", <" << a << "|kekule|"
				<< 0 << "> = " << e
				<< std::endl;
		}
	}
	out.close();
}
