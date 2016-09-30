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
template<typename T>
std::vector<T> get_imaginary_time_obs(arma::SpMat<T>& op, int Ntau,
	double t_step, int degeneracy, arma::vec& ev, arma::mat& es,
	arma::mat& esT)
{
	arma::Mat<T> es_cx = arma::conv_to<arma::Mat<T>>::from(es);
	arma::Mat<T> esT_cx = arma::conv_to<arma::Mat<T>>::from(esT);
	std::vector<T> obs_vec(Ntau + 1);
	for (int n = 0; n <= Ntau; ++n)
	{
		double tau = n * t_step;
		obs_vec[n] = T(0.);
		for (int a = 0 ; a < degeneracy; ++a)
			for (int b = 0; b < ev.n_rows; ++b)
				obs_vec[n] += std::exp(tau * (ev(a) - ev(b)))
					* arma::trace(esT_cx.row(a) * op * es_cx.col(b))
					* arma::trace(esT_cx.row(b) * op.t() * es_cx.col(a));
	}
	return obs_vec;
}

template<typename T>
void print_data(std::ostream& out, const T& data)
{
	for (int j = 0; j < data.size(); ++j)
	{
		out << std::abs(data[j]) << "\t";
		std::cout << std::abs(data[j]) << "\t";
	}
	out << std::endl;
	std::cout << std::endl;
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
		("k", po::value<int>(&k)->default_value(100), "number of eigenstates")
		("ensemble,e", po::value<std::string>(&ensemble)->default_value("gc"),
			"ensemble: gc or c");
	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);

	if (vm.count("help")) print_help(desc);
	std::cout << "L = " << L << std::endl;
	std::cout << "V = " << V << std::endl;
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
	std::cout << "Constructing static operators...";
	std::cout.flush();
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
	H_st.clear();
	
	//Build static observables
	sparse_storage<double, int_t> M2_st(hspace.sub_dimension()),
		M4_st(hspace.sub_dimension());
	hspace.build_operator([&] (const std::pair<int_t, int_t>& n)
	{
		for (int_t i = 0; i < lat.n_sites(); ++i)
		{
			for (int_t j = 0; j < lat.n_sites(); ++j)
			{
				M2_st(n.second, n.second) += lat.parity(i) * lat.parity(j)
					/ std::pow(lat.n_sites(), 2.0) * (hspace.n_i({1, n.first}, i)
					- 0.5) * (hspace.n_i({1, n.first}, j) - 0.5);

				for (int_t p = 0; p < lat.n_sites(); ++p)
					for (int_t q = 0; q < lat.n_sites(); ++q)
						M4_st(n.second, n.second) += lat.parity(i) * lat.parity(j)
							* lat.parity(p) * lat.parity(q) / std::pow(lat.n_sites(),
							4.0) * (hspace.n_i({1, n.first},i) - 0.5)
							* (hspace.n_i({1, n.first}, j) - 0.5)
							* (hspace.n_i({1, n.first}, p) - 0.5)
							* (hspace.n_i({1, n.first}, q) - 0.5);
			}
		}
	});
	arma::sp_mat M2 = M2_st.build_matrix();
	M2_st.clear();
	arma::sp_mat M4 = M4_st.build_matrix();
	M4_st.clear();
	std::cout << "Done." << std::endl;

	std::string out_file = "../data/ed_L_" + std::to_string(L) + "__"
		+ "V_" + std::to_string(V) + "__"
		+ "GS__"
		+ ensemble;
	std::ofstream out("../data/" + out_file);
	std::cout.precision(8);
	out.precision(18);
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
	H.clear();
	arma::mat esT = es.t();
	std::cout << "GS space" << std::endl;
	std::cout << "GS energy: " << ev[0] << std::endl;
	int degeneracy = 0;
	for (int i = 0; i < ev.n_rows && ev[i] <= ev[0]; ++i)
		++degeneracy;
	std::cout << "GS degeneracy: " << degeneracy << std::endl;
	
	double E = 0., m2 = 0., m4 = 0., cij = 0.;
	for (int i = 0; i < degeneracy; ++i)
	{
		E += ev(i) / degeneracy;
		// Trace over 1x1 matrices
		m2 += arma::trace(esT.row(i) * M2 * es.col(i));
		m4 += arma::trace(esT.row(i) * M4 * es.col(i));
	}
	M2.clear();
	M4.clear();
	
	int Ntau = 100, Nmat = 0;
	double t_step = 0.2;
	out << k << "\t" << L << "\t" << V << "\t" << T << "\t"
		<< E << "\t" << m2 << "\t" << m4 << "\t" << m4/(m2*m2) << "\t"
		<< Ntau << "\t" << Nmat << std::endl;
	std::cout << k << "\t" << L << "\t" << V << "\t" << T << "\t"
		<< E << "\t" << m2 << "\t" << m4 << "\t" << m4/(m2*m2) << "\t"
		<< Ntau << "\t" << Nmat << std::endl;
	
	// Build dynamic observables
	std::cout << "Constructing dynamic operators...";
	std::cout.flush();
	arma::Mat<std::complex<double>> es_cx = arma::conv_to<arma::Mat<std::complex<double>>>::from(es);
	arma::Mat<std::complex<double>> esT_cx = arma::conv_to<arma::Mat<std::complex<double>>>::from(esT);
	std::vector<std::vector<std::complex<double>>> obs_data_cx;
	std::complex<double> ep, kek, chern = 0;
	
	sparse_storage<std::complex<double>, int_t> ni_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			for (int i = 0; i < lat.n_sites(); ++i)
			{
				//CDW
				ni_st(n.second, n.second)
					+= lat.parity(i) / lat.n_sites()
						* (hspace.n_i({1, n.first}, i) - 0.5);
			}
		});
	arma::sp_cx_mat ni_op = ni_st.build_matrix();
	ni_st.clear();
	obs_data_cx.emplace_back(get_imaginary_time_obs(ni_op, Ntau, t_step, degeneracy, ev,
		es, esT));
	ni_op.clear();
	print_data(out, obs_data_cx[0]);
	
	sparse_storage<std::complex<double>, int_t> kekule_st(hspace.sub_dimension());
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
		});
	arma::sp_cx_mat kekule_op = kekule_st.build_matrix();
	kekule_st.clear();
	obs_data_cx.emplace_back(get_imaginary_time_obs(kekule_op, Ntau, t_step, degeneracy,
		ev, es, esT));
	for (int i = 0; i < degeneracy; ++i)
		kek += arma::trace(esT_cx.row(i) * kekule_op * es_cx.col(i));
	kekule_op.clear();
	print_data(out, obs_data_cx[1]);
	
	sparse_storage<std::complex<double>, int_t> chern_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			//chern
			for (auto& b : lat.bonds("chern"))
			{
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					chern_st(hspace.index(p.id), n.second) += std::complex<double>{0., p.sign
						/ static_cast<double>(lat.n_bonds())};
			}
 			for (auto& b : lat.bonds("chern_2"))
			{
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					chern_st(hspace.index(p.id), n.second) += std::complex<double>{0., -p.sign
						/ static_cast<double>(lat.n_bonds())};
			}
		});
	arma::sp_cx_mat chern_op = chern_st.build_matrix();
	chern_st.clear();
	obs_data_cx.emplace_back(get_imaginary_time_obs(chern_op, Ntau, t_step, degeneracy,
		ev, es, esT));
	for (int i = 0; i < degeneracy; ++i)
		chern += arma::trace(esT_cx.row(i) * chern_op * es_cx.col(i));
	chern_op.clear();
	print_data(out, obs_data_cx[2]);
	
	sparse_storage<std::complex<double>, int_t> epsilon_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			for (int i = 0; i < lat.n_sites(); ++i)
			{
				//epsilon
				for (int j : lat.neighbors(i, "nearest neighbors"))
				{
					state p = hspace.c_i({1, n.first}, j);
					p = hspace.c_dag_i(p, i);
					if (p.sign != 0)
						epsilon_st(hspace.index(p.id), n.second) += p.sign
							/ static_cast<double>(lat.n_bonds());
				}
			}
		});
	arma::sp_cx_mat epsilon_op = epsilon_st.build_matrix();
	for (int i = 0; i < degeneracy; ++i)
		ep += arma::trace(esT_cx.row(i) * epsilon_op * es_cx.col(i));
	
	sparse_storage<std::complex<double>, int_t> sp_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			for (int i = 0; i < lat.n_sites(); ++i)
			{
				//sp
				auto& K = lat.symmetry_point("K");
				std::complex<double> phase = std::exp(std::complex<double>(0.,
					K.dot(lat.real_space_coord(i))));
				state p = hspace.c_i({1, n.first}, i);
				if (p.sign != 0)
					sp_st(hspace.index(p.id), n.second) += phase
						* std::complex<double>(p.sign);
			}
		});
	arma::sp_cx_mat sp_op = sp_st.build_matrix();
	sp_st.clear();
	obs_data_cx.emplace_back(get_imaginary_time_obs(sp_op, Ntau, t_step, degeneracy,
		ev, es, esT));
	sp_op.clear();
	print_data(out, obs_data_cx[3]);
	
	sparse_storage<std::complex<double>, int_t> tp_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			for (int i = 0; i < lat.n_sites(); ++i)
			{
				//tp
				for (int j = 0; j < lat.n_sites(); ++j)
				{
					auto& K = lat.symmetry_point("K");
					std::complex<double> phase = std::exp(std::complex<double>(0.,
						K.dot(lat.real_space_coord(i) - lat.real_space_coord(j))));
					state p = hspace.c_dag_i({1, n.first}, j);
					p = hspace.c_dag_i(p, i);
					if (p.sign != 0)
						tp_st(hspace.index(p.id), n.second) += phase
							* std::complex<double>(p.sign);
				}
			}
		});
	arma::sp_cx_mat tp_op = tp_st.build_matrix();
	tp_st.clear();
	obs_data_cx.emplace_back(get_imaginary_time_obs(tp_op, Ntau, t_step, degeneracy,
		ev, es, esT));
	tp_op.clear();
	print_data(out, obs_data_cx[4]);
	
	//Subtract finite expectation value
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			epsilon_st(n.second, n.second) -= ep;
		});
	epsilon_op = epsilon_st.build_matrix();
	epsilon_st.clear();
	obs_data_cx.emplace_back(get_imaginary_time_obs(epsilon_op, Ntau, t_step, degeneracy,
		ev, es, esT));
	epsilon_op.clear();
	print_data(out, obs_data_cx[5]);
	
	std::cout << "Done" << std::endl;
	std::cout << "<epsilon> = " << ep << std::endl;
	std::cout << "<kekule> = " << kek << std::endl;
	std::cout << "<chern> = " << chern << std::endl;
		
	out.close();
}
