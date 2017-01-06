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
#include "hex_honeycomb.h"
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
std::vector<T> get_imaginary_time_obs(arma::SpMat<T>& op, 
	int Ntau, double t_step, int degeneracy,
	arma::vec& ev, arma::mat& es, arma::mat& esT)
{
	arma::Mat<T> es_cx = arma::conv_to<arma::Mat<T>>::from(es);
	arma::Mat<T> esT_cx = arma::conv_to<arma::Mat<T>>::from(esT);
	arma::SpMat<T> opT = op.t();
	std::vector<T> obs_vec(Ntau + 1);
	for (int n = 0; n <= Ntau; ++n)
	{
		double tau = n * t_step;
		obs_vec[n] = T(0.);
		for (int a = 0 ; a < degeneracy; ++a)
			for (int b = 0; b < ev.n_rows; ++b)
				obs_vec[n] += std::exp(tau * (ev(a) - ev(b)))
					* arma::trace(esT_cx.row(a) * op * es_cx.col(b))
					* arma::trace(esT_cx.row(b) * opT * es_cx.col(a)) / std::complex<double>(degeneracy);
	}
	return obs_vec;
}

template<typename T>
void print_overlap(arma::SpMat<T>& op, const std::string& name,
	int degeneracy, arma::vec& ev, arma::mat& es, arma::mat& esT)
{
	arma::Mat<T> es_cx = arma::conv_to<arma::Mat<T>>::from(es);
	arma::Mat<T> esT_cx = arma::conv_to<arma::Mat<T>>::from(esT);
	std::cout << std::endl;
	std::cout << "----------" << std::endl;
	for (int d = 0; d < degeneracy; ++d)
	{
		int cnt = 0;
		for (int i = 0; i < esT_cx.n_rows; ++i)
		{
			double c = 0.;
			c += std::abs(arma::trace(esT_cx.row(i) * op * es_cx.col(d)));
			if (i < degeneracy || c > std::pow(10., -13.))
			{
				std::cout << "|<" << i << "| " + name + " |" << d << ">|^2 = " << c
					<< ", E(" << i << ") - E(0) = " << ev[i]-ev[0] << std::endl;
				++cnt;
			}
			if (cnt >= 20) break;
		}
	}
	std::cout << "----------" << std::endl;
	std::cout << std::endl;
}

template<typename T>
void print_data(std::ostream& out, const T& data)
{
	for (int j = 0; j < data.size(); ++j)
	{
		out << std::real(data[j]) << "\t";
		std::cout << std::real(data[j]) << "\t";
	}
	out << std::endl;
	std::cout << std::endl;
}
	
int main(int ac, char** av)
{
	int L;
	double tprime;
	double V;
	double mu;
	int k;
	std::string ensemble, geometry;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("L", po::value<int>(&L)->default_value(3), "linear lattice dimension")
		("tprime", po::value<double>(&tprime)->default_value(0.), "d3 hopping")
		("V", po::value<double>(&V)->default_value(1.355), "interaction strength")
		("mu", po::value<double>(&mu)->default_value(0.), "chemical potential")
		("k", po::value<int>(&k)->default_value(100), "number of eigenstates")
		("ensemble,e", po::value<std::string>(&ensemble)->default_value("gc"),
			"ensemble: gc or c")
		("geometry,g", po::value<std::string>(&geometry)->default_value("rhom"),
			"geometry: rhombic or hexagonal");
	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);

	if (vm.count("help")) print_help(desc);
	if (geometry == "hex")
		L = 1;
	else
		geometry = "rhom";
	std::cout << "geometry: " << geometry << std::endl;
	std::cout << "ensemble: " << ensemble << std::endl;
	std::cout << "L = " << L << std::endl;
	std::cout << "tprime = " << tprime << std::endl;
	std::cout << "V = " << V << std::endl;
	std::cout << "mu = " << mu << std::endl;
	//Generate lattice
	lattice lat;
	if (geometry == "hex")
	{
		hex_honeycomb h(L);
		lat.generate_graph(h);
		h.generate_maps(lat);
	}
	else if (geometry == "rhom")
	{
		honeycomb h(L);
		lat.generate_graph(h);
		h.generate_maps(lat);
	}

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
	hspace.build_operator([&](const std::pair<int_t,
		int_t>& n)
	{
		for (int_t i = 0; i < lat.n_sites(); ++i)
		{
			//Chemical potential: -mu sum_i c_i^dag c_i
			H_st(n.second, n.second) -= mu * hspace.n_i({1, n.first}, i);
			
			for (int_t j : lat.neighbors(i, "nearest neighbors"))
			{
				//Hopping term: -t sum_<ij> c_i^dag c_j
				state m = hspace.c_i({-1, n.first}, j);
				m = hspace.c_dag_i(m, i);
				if (m.sign != 0)
					H_st(hspace.index(m.id), n.second) += m.sign * (-1.);

				//Interaction: V sum_<ij> (n_i - 0.5) (n_j - 0.5)
				if (i < j)
					H_st(n.second, n.second) += V * (hspace.n_i({1, n.first}, i)
						- 0.5) * (hspace.n_i({1, n.first}, j) - 0.5);
			}
		}
		for (auto& a : lat.bonds("d3_bonds"))
		{
			state m = hspace.c_i({-1, n.first}, a.second);
			m = hspace.c_dag_i(m, a.first);
			if (m.sign != 0)
				H_st(hspace.index(m.id), n.second) += m.sign * (-1.) * tprime;
		}
	});
	arma::sp_mat H = H_st.build_matrix();
	H_st.clear();
	
	//Build static observables
	sparse_storage<double, int_t> n_total_st(hspace.sub_dimension());
	hspace.build_operator([&] (const std::pair<int_t, int_t>& n)
	{
		for (int_t i = 0; i < lat.n_sites(); ++i)
			n_total_st(n.second, n.second) += hspace.n_i({1, n.first}, i);
	});
	arma::sp_mat n_total_op = n_total_st.build_matrix();
	n_total_st.clear();
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
	for (int i = 0; i < ev.n_rows && std::abs(ev[i] - ev[0]) <= std::pow(10, -12); ++i)
		++degeneracy;
	std::cout << "GS degeneracy: " << degeneracy << std::endl;
	
	double E = 0., m2 = 0., m4 = 0., cij = 0., n_total = 0.;
	for (int i = 0; i < degeneracy; ++i)
	{
		E += ev(i) / degeneracy;
		n_total += arma::trace(esT.row(i) * n_total_op * es.col(i)) / degeneracy;
	}
	
	int Ntau = 50, Nmat = 0;
	double t_step = 0.2;
	out << k << "\t" << L << "\t" << V << "\t" << 0 << "\t"
		<< E << "\t" << m2 << "\t" << m4 << "\t" << m4/(m2*m2) << "\t"
		<< Ntau << "\t" << Nmat << std::endl;
	
	// Build dynamic observables
	std::cout << "Constructing dynamic operators...";
	std::cout.flush();
	arma::Mat<std::complex<double>> es_cx = arma::conv_to<arma::Mat<std::complex<double>>>::from(es);
	arma::Mat<std::complex<double>> esT_cx = arma::conv_to<arma::Mat<std::complex<double>>>::from(esT);
	std::vector<std::vector<std::complex<double>>> obs_data_cx;
	std::complex<double> cdw2 = 0., cdw4 = 0., ep = 0., kek = 0., chern = 0., chern2 = 0., chern4 = 0.;
	
	sparse_storage<std::complex<double>, int_t> ni_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			for (int i = 0; i < lat.n_sites(); ++i)
			{
				//CDW
				ni_st(n.second, n.second)
					+= lat.parity(i) / std::complex<double>(lat.n_sites())
						* (hspace.n_i({1, n.first}, i) - 0.5);
			}
		});
	arma::sp_cx_mat ni_op = ni_st.build_matrix();
	ni_st.clear();
	obs_data_cx.emplace_back(get_imaginary_time_obs(ni_op, Ntau, t_step, degeneracy, ev,
		es, esT));
	print_overlap(ni_op, "cdw", degeneracy, ev, es, esT);
	arma::sp_cx_mat ni2_op = ni_op * ni_op;
	ni_op.clear();
	for (int i = 0; i < degeneracy; ++i)
	{
		cdw2 += arma::trace(esT_cx.row(i) * ni2_op * es_cx.col(i)) / std::complex<double>(degeneracy);
		cdw4 += arma::trace(esT_cx.row(i) * ni2_op * ni2_op * es_cx.col(i)) / std::complex<double>(degeneracy);
	}
	ni2_op.clear();
	print_data(out, obs_data_cx[0]);
	
	sparse_storage<std::complex<double>, int_t> ni_K_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			auto& K = lat.symmetry_point("K");
			for (int i = 0; i < lat.n_sites(); ++i)
			{
				std::complex<double> phase = std::exp(std::complex<double>(0.,
						K.dot(lat.real_space_coord(i))));
				ni_K_st(n.second, n.second)
					+= lat.parity(i) * phase / std::complex<double>(lat.n_sites())
						* (hspace.n_i({1, n.first}, i) - 0.5);
			}
		});
	arma::sp_cx_mat ni_K_op = ni_K_st.build_matrix();
	ni_K_st.clear();
	obs_data_cx.emplace_back(get_imaginary_time_obs(ni_K_op, Ntau, t_step, degeneracy, ev,
		es, esT));
	print_overlap(ni_K_op, "cdw_K", degeneracy, ev, es, esT);
	ni_K_op.clear();
	print_data(out, obs_data_cx[1]);
	
	sparse_storage<std::complex<double>, int_t> ni_sym_K_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			auto& K = lat.symmetry_point("K");
			for (int i = 0; i < lat.n_sites(); ++i)
			{
				std::complex<double> phase = std::exp(std::complex<double>(0.,
						K.dot(lat.real_space_coord(i))));
				ni_sym_K_st(n.second, n.second)
					+= phase / std::complex<double>(lat.n_sites())
						* (hspace.n_i({1, n.first}, i) - 0.5);
			}
		});
	arma::sp_cx_mat ni_sym_K_op = ni_sym_K_st.build_matrix();
	ni_sym_K_st.clear();
	obs_data_cx.emplace_back(get_imaginary_time_obs(ni_sym_K_op, Ntau, t_step, degeneracy, ev,
		es, esT));
	print_overlap(ni_sym_K_op, "cdw_sym_K", degeneracy, ev, es, esT);
	ni_sym_K_op.clear();
	print_data(out, obs_data_cx[2]);
	
	
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
					kekule_st(hspace.index(p.id), n.second) +=
						2.*std::complex<double>(p.sign)
						/ static_cast<double>(lat.n_bonds());
			}
			
			for (auto& b : lat.bonds("kekule_2"))
			{
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					kekule_st(hspace.index(p.id), n.second) +=
						-std::complex<double>(p.sign)
						/ static_cast<double>(lat.n_bonds());
			}
			
			for (auto& b : lat.bonds("kekule_3"))
			{
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					kekule_st(hspace.index(p.id), n.second) +=
						-std::complex<double>(p.sign)
						/ static_cast<double>(lat.n_bonds());
			}
		});
	arma::sp_cx_mat kekule_op = kekule_st.build_matrix();
	kekule_st.clear();
	obs_data_cx.emplace_back(get_imaginary_time_obs(kekule_op, Ntau, t_step, degeneracy,
		ev, es, esT));
	for (int i = 0; i < degeneracy; ++i)
		kek += arma::trace(esT_cx.row(i) * kekule_op * es_cx.col(i)) / std::complex<double>(degeneracy);
	print_overlap(kekule_op, "kekule", degeneracy, ev, es, esT);
	kekule_op.clear();
	print_data(out, obs_data_cx[3]);
	
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
				p = hspace.c_i({1, n.first}, b.first);
				p = hspace.c_dag_i(p, b.second);
				if (p.sign != 0)
					chern_st(hspace.index(p.id), n.second) += std::complex<double>{0., -p.sign
						/ static_cast<double>(lat.n_bonds())};
			}
			/*
			//chern
			for (auto& b : lat.bonds("chern_2"))
			{
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					chern_st(hspace.index(p.id), n.second) += std::complex<double>{0., p.sign
						/ static_cast<double>(lat.n_bonds())};
				p = hspace.c_i({1, n.first}, b.first);
				p = hspace.c_dag_i(p, b.second);
				if (p.sign != 0)
					chern_st(hspace.index(p.id), n.second) += std::complex<double>{0., -p.sign
						/ static_cast<double>(lat.n_bonds())};
			}
			*/
		});
	arma::sp_cx_mat chern_op = chern_st.build_matrix();
	chern_st.clear();
	obs_data_cx.emplace_back(get_imaginary_time_obs(chern_op, Ntau, t_step, degeneracy,
		ev, es, esT));
	print_overlap(chern_op, "chern", degeneracy, ev, es, esT);
	arma::sp_cx_mat chern2_op = chern_op * chern_op;
	for (int i = 0; i < degeneracy; ++i)
	{
		chern += arma::trace(esT_cx.row(i) * chern_op * es_cx.col(i)) / std::complex<double>(degeneracy);
		chern2 += arma::trace(esT_cx.row(i) * chern2_op * es_cx.col(i)) / std::complex<double>(degeneracy);
		chern4 += arma::trace(esT_cx.row(i) * chern2_op * chern2_op * es_cx.col(i)) / std::complex<double>(degeneracy);
	}
	chern_op.clear();
	chern2_op.clear();
	print_data(out, obs_data_cx[4]);
	
	sparse_storage<std::complex<double>, int_t> gamma_mod_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			double pi = 4. * std::atan(1.);
			std::complex<double> im = {0., 1.};
			for (auto& b : lat.bonds("nn_bond_1"))
			{
				std::complex<double> phase = std::exp(std::complex<double>(0., 0.*pi));
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					gamma_mod_st(hspace.index(p.id), n.second) += phase * std::complex<double>(p.sign)
						/ std::complex<double>(lat.n_bonds());
				p = hspace.c_i({1, n.first}, b.first);
				p = hspace.c_dag_i(p, b.second);
				if (p.sign != 0)
					gamma_mod_st(hspace.index(p.id), n.second) += std::conj(phase) * std::complex<double>(p.sign)
						/ std::complex<double>(lat.n_bonds());
			}
			for (auto& b : lat.bonds("nn_bond_2"))
			{
				std::complex<double> phase = std::exp(std::complex<double>(0., 2./3.*pi));
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					gamma_mod_st(hspace.index(p.id), n.second) += phase * std::complex<double>(p.sign)
						/ std::complex<double>(lat.n_bonds());
				p = hspace.c_i({1, n.first}, b.first);
				p = hspace.c_dag_i(p, b.second);
				if (p.sign != 0)
					gamma_mod_st(hspace.index(p.id), n.second) += std::conj(phase) * std::complex<double>(p.sign)
						/ std::complex<double>(lat.n_bonds());
			}
			for (auto& b : lat.bonds("nn_bond_3"))
			{
				std::complex<double> phase = std::exp(std::complex<double>(0., 4./3.*pi));
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					gamma_mod_st(hspace.index(p.id), n.second) += phase * std::complex<double>(p.sign)
						/ std::complex<double>(lat.n_bonds());
				p = hspace.c_i({1, n.first}, b.first);
				p = hspace.c_dag_i(p, b.second);
				if (p.sign != 0)
					gamma_mod_st(hspace.index(p.id), n.second) += std::conj(phase) * std::complex<double>(p.sign)
						/ std::complex<double>(lat.n_bonds());
			}
		});
	arma::sp_cx_mat gamma_mod_op = gamma_mod_st.build_matrix();
	gamma_mod_st.clear();
	obs_data_cx.emplace_back(get_imaginary_time_obs(gamma_mod_op, Ntau, t_step, degeneracy, ev,
		es, esT));
	print_overlap(gamma_mod_op, "gamma_mod", degeneracy, ev, es, esT);
	gamma_mod_op.clear();
	print_data(out, obs_data_cx[5]);
	
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
		ep += arma::trace(esT_cx.row(i) * epsilon_op * es_cx.col(i)) / std::complex<double>(degeneracy);
	//Subtract finite expectation value
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			epsilon_st(n.second, n.second) -= ep;
		});
	epsilon_op = epsilon_st.build_matrix();
	print_overlap(epsilon_op, "epsilon - <ep>", degeneracy, ev, es, esT);
	epsilon_st.clear();
	obs_data_cx.emplace_back(get_imaginary_time_obs(epsilon_op, Ntau, t_step, degeneracy,
		ev, es, esT));
	epsilon_op.clear();
	print_data(out, obs_data_cx[6]);
	
	if (ensemble == "gc")
	{
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
		print_overlap(sp_op, "sp", degeneracy, ev, es, esT);
		sp_st.clear();
		obs_data_cx.emplace_back(get_imaginary_time_obs(sp_op, Ntau, t_step, degeneracy,
			ev, es, esT));
		sp_op.clear();
		print_data(out, obs_data_cx[7]);
		
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
						state p = hspace.c_i({1, n.first}, j);
						p = hspace.c_i(p, i);
						if (p.sign != 0)
							tp_st(hspace.index(p.id), n.second) += phase
								* std::complex<double>(p.sign);
					}
				}
			});
		arma::sp_cx_mat tp_op = tp_st.build_matrix();
		print_overlap(tp_op, "tp", degeneracy, ev, es, esT);
		tp_st.clear();
		obs_data_cx.emplace_back(get_imaginary_time_obs(tp_op, Ntau, t_step, degeneracy,
			ev, es, esT));
		tp_op.clear();
		print_data(out, obs_data_cx[8]);
	}
	
	std::cout << "Done" << std::endl;
	std::cout << "<E> = " << E << std::endl;
	std::cout << "<n> = " << n_total / lat.n_sites() << std::endl;
	std::cout << "<m2> = " << std::real(cdw2) << std::endl;
	std::cout << "<m4> = " << std::real(cdw4) << std::endl;
	std::cout << "B_cdw = " << std::real(cdw4/(cdw2*cdw2)) << std::endl;
	std::cout << "<epsilon> = " << std::real(ep) << std::endl;
	std::cout << "<kekule> = " << std::real(kek) << std::endl;
	std::cout << "<chern> = " << std::imag(chern) << std::endl;
	std::cout << "<chern^2> = " << std::real(chern2) << std::endl;
	std::cout << "<chern^4> = " << std::real(chern4) << std::endl;
	std::cout << "B_chern = " << std::real(chern4/(chern2*chern2)) << std::endl;
	
	out.close();
}
