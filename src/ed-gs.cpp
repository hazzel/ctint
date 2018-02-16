#include <iostream>
#include <iomanip>
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
#include "tilted_honeycomb.h"
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
	arma::vec& ev, arma::cx_mat& es_cx, arma::cx_mat& esT_cx)
{
	arma::SpMat<T> opT = op.t();
	std::vector<T> obs_vec(Ntau + 1);
	int a;
	if (degeneracy == 1)
		a = 0;
	else
		a = 1;
	obs_vec[0] = arma::trace(esT_cx.row(a) * op * opT * es_cx.col(a));
	std::cout << "----------" << std::endl;
	for (int d = 0; d < degeneracy; ++d)
		std::cout << "<" << d << "| O * Ot |" << d << "> = " << arma::trace(esT_cx.row(d) * op * opT * es_cx.col(d)) << std::endl;
	for (int n = 1; n <= Ntau; ++n)
	{
		double tau = n * t_step;
		obs_vec[n] = T(0.);
		//for (int a = 0 ; a < degeneracy; ++a)
			for (int b = 0; b < ev.n_rows; ++b)
				obs_vec[n] += std::exp(tau * (ev(a) - ev(b)))
					* arma::trace(esT_cx.row(a) * op * es_cx.col(b))
					* arma::trace(esT_cx.row(b) * opT * es_cx.col(a));// / std::complex<double>(degeneracy);
	}
	return obs_vec;
}

template<typename T>
void print_overlap(arma::SpMat<T>& op, const std::string& name,
	int degeneracy, arma::vec& ev, arma::cx_mat& es_cx, arma::cx_mat& esT_cx, arma::SpMat<T>& P_cx, arma::SpMat<T>& PH_cx)
{
	std::cout << std::endl;
	std::cout << "Inversion parity: P * O * P - O = " << arma::norm(arma::nonzeros(P_cx * op * P_cx - op)) << std::endl;
	std::cout << "Inversion parity: P * O * P + O = " << arma::norm(arma::nonzeros(P_cx * op * P_cx + op)) << std::endl;
	std::cout << "Particle-hole parity: PH * O * PH - O = " << arma::norm(arma::nonzeros(PH_cx * arma::conj(op) * PH_cx - op)) << std::endl;
	std::cout << "Particle-hole parity: PH * O * PH + O = " << arma::norm(arma::nonzeros(PH_cx * arma::conj(op) * PH_cx + op)) << std::endl;
	for (int d = 0; d < degeneracy; ++d)
	//for (int d = 0; d < 2; ++d)
	{
		int cnt = 0;
		for (int i = 0; i < esT_cx.n_rows; ++i)
		{
			double c = 0.;
			c += std::abs(arma::trace(esT_cx.row(i) * op * es_cx.col(d)));
			if (d == i || c > 1E-14)
			{
				std::cout << "|<" << i << "| " + name + " |" << d << ">|^2 = " << c*c
					<< ", E(" << i << ") - E(0) = " << ev[i]-ev[0] << std::endl;
				std::cout << "<" << d << "|O P O|" << d << "> = " << arma::trace(esT_cx.row(d) * op.t() * P_cx * op * es_cx.col(d))
					/ arma::trace(esT_cx.row(d) * op.t() * op * es_cx.col(d)) << std::endl;
				std::cout << "<" << i << "| P |" << i << "> = " << arma::trace(esT_cx.row(i) * P_cx * es_cx.col(i)) << std::endl;
				std::cout << "<" << d << "|O PH O|" << d << "> = " << arma::trace(esT_cx.row(d) * op.t() * PH_cx * op * es_cx.col(d))
					/ arma::trace(esT_cx.row(d) * op.t() * op * es_cx.col(d)) << std::endl;
				std::cout << "<" << i << "| PH |" << i << "> = " << arma::trace(esT_cx.row(i) * PH_cx * es_cx.col(i)) << std::endl;
				++cnt;
			}
			//if (cnt >= 20) break;
		}
	}
	std::cout << "----------" << std::endl;
	std::cout << std::endl;
}

arma::cx_mat symmetrize_es(arma::vec& ev, arma::cx_mat& es, arma::sp_cx_mat& P)
{
	double epsilon = std::pow(10., -5.);

	arma::cx_mat S_s = es + P * es;
	arma::cx_mat S_a = es - P * es;
	arma::cx_mat S_so(es.n_rows, es.n_cols);
	arma::cx_mat S_ao(es.n_rows, es.n_cols);
	arma::cx_mat S_f = arma::zeros<arma::cx_mat>(es.n_rows, 2*es.n_cols);

	int n_sectors = 1, n_last_sector = 1;
	double E = ev(0);
	for (int i = 0; i < es.n_cols; ++i)
	{
		if (arma::norm(S_s.col(i)) > epsilon)
			S_s.col(i) /= arma::norm(S_s.col(i));
		else
			S_s.col(i) *= 0.;
		if (arma::norm(S_a.col(i)) > epsilon)
			S_a.col(i) /= arma::norm(S_a.col(i));
		else
			S_a.col(i) *= 0.;
		if (std::abs(ev(i) - E) > epsilon)
		{
			++n_sectors;
			E = ev(i);
			n_last_sector = 1;
			std::cout << "Sector " << i << ": " << ev(i) << std::endl;
		}
		else
			++n_last_sector;
	}

	int cnt = 0;
	for (int i = 0; i < es.n_cols - n_last_sector; ++i)
	{
		int j;
		for (j = i; j < es.n_cols - n_last_sector && std::abs(ev[j]-ev[i]) < epsilon ; ++j)
		{
			S_so.col(j) = S_s.col(j);
			S_ao.col(j) = S_a.col(j);
			for (int k = i; k < j; ++k)
			{
				S_so.col(j) -= S_so.col(k) * arma::dot(S_so.col(k), S_s.col(j));
				S_ao.col(j) -= S_ao.col(k) * arma::dot(S_ao.col(k), S_a.col(j));
			}
			//std::cout << "E=" << ev(i) << ", orth: i=" << i << ", j=" << j << ": " << arma::norm(S_so.col(j)) << " " << arma::norm(S_ao.col(j)) << std::endl;
			if (arma::norm(S_so.col(j)) > epsilon)
			{
				S_so.col(j) /= arma::norm(S_so.col(j));
				S_f.col(cnt) = S_so.col(j);
				++cnt;
			}
			if (arma::norm(S_ao.col(j)) > epsilon)
			{
				S_ao.col(j) /= arma::norm(S_ao.col(j));
				S_f.col(cnt) = S_ao.col(j);
				++cnt;
			}
		}
		i = j - 1;
	}
	std::cout << "Total energy sectors: " << n_sectors << std::endl;
	std::cout << "cnt = " << cnt << std::endl;
	std::cout << "evs = " << ev.n_rows << std::endl;
	std::cout << "Symmetrized states: " << cnt << " out of " << es.n_cols << " up to energy E-E_0=" << ev(cnt-1)-ev(0) << std::endl;
	
	if (cnt < es.n_cols)
		S_f.cols(cnt, es.n_cols - 1) = es.cols(cnt, es.n_cols - 1);
	return S_f.cols(0, es.n_cols - 1);
}

std::vector<std::vector<int>> get_energy_levels(arma::vec& en)
{
	double epsilon = std::pow(10., -4.);
	std::vector<std::vector<int>> energy_levels;
	energy_levels.push_back({0});
	for (int i = 1; i < en.n_rows; ++i)
	{
		if (std::abs(en[i] - en[energy_levels.back()[0]]) > epsilon)
			energy_levels.push_back(std::vector<int>());
		energy_levels.back().push_back(i);
	}
	for (int i = 0; i < energy_levels.size(); ++i)
		std::sort(energy_levels[i].begin(), energy_levels[i].end());
	return energy_levels;
}

arma::cx_mat orthonormalize(arma::cx_mat& es)
{
	double epsilon = std::pow(10., -4.);
	arma::cx_mat S_o = es, S_f = arma::zeros<arma::cx_mat>(es.n_rows, es.n_cols);
	
	for (int i = 0; i < S_o.n_cols; ++i)
	{
		if (arma::norm(S_o.col(i)) > epsilon)
			S_o.col(i) /= arma::norm(S_o.col(i));
		else
			S_o.col(i) *= 0.;
	}

	for (int i = 0; i < S_o.n_cols; ++i)
	{
		S_f.col(i) = S_o.col(i);
		for (int k = 0; k < i; ++k)
			S_f.col(i) -= S_f.col(k) * (S_f.col(k).t() * S_o.col(i));
		
		if (arma::norm(S_f.col(i)) > epsilon)
			S_f.col(i) /= arma::norm(S_f.col(i));
	}
	return S_f;
}

void split_quantum_numbers(std::vector<std::vector<int>>& energy_levels, const arma::cx_mat& es, const arma::sp_cx_mat& pm)
{
	double epsilon = std::pow(10., -4.);
	arma::cx_mat esT = es.t();
	for (int i = 0; i < energy_levels.size(); ++i)
	{
		std::vector<std::vector<int>> sub_levels;
		std::vector<std::complex<double>> quantum_numbers;
		sub_levels.push_back({energy_levels[i][0]});
		quantum_numbers.push_back(arma::trace(esT.row(energy_levels[i][0]) * pm * es.col(energy_levels[i][0])));
		for (int j = 1; j < energy_levels[i].size(); ++j)
		{
			std::complex<double> q = arma::trace(esT.row(energy_levels[i][j]) * pm * es.col(energy_levels[i][j]));
			int k;
			for (k = 0; k < quantum_numbers.size();)
				if (std::abs(quantum_numbers[k] - q) > epsilon)
					++k;
				else
					break;
			if (k == quantum_numbers.size())
			{
				quantum_numbers.push_back(q);
				sub_levels.push_back({energy_levels[i][j]});
			}
			else
				sub_levels[k].push_back(energy_levels[i][j]);
		}
		//for (int k = 0; k < quantum_numbers.size(); ++k)
		//	for (int j = 0; j < sub_levels[k].size(); ++j)
		//	std::cout << "sublevel k = " << k << ", j = " << j << ": " << sub_levels[k][j] << std::endl;
		energy_levels.erase(energy_levels.begin() + i);
		for (int k = 0; k < quantum_numbers.size(); ++k)
			energy_levels.insert(energy_levels.begin()+i, sub_levels[sub_levels.size()-1-k]);
	}
	for (int i = 0; i < energy_levels.size(); ++i)
		std::sort(energy_levels[i].begin(), energy_levels[i].end());
}

arma::cx_mat project_symmetry(const arma::cx_mat& es, const std::vector<std::vector<int>>& energy_levels, const arma::sp_cx_mat& pm)
{
	double epsilon = std::pow(10., -4.);
	arma::cx_mat esT = es.t();
	arma::cx_mat S_f = arma::zeros<arma::cx_mat>(es.n_rows, es.n_cols);
	
	for (int i = 0; i < energy_levels.size(); ++i)
	{
		int N = energy_levels[i].size();
		arma::cx_mat projP(N, N);
		arma::cx_mat S_proj = arma::zeros<arma::cx_mat>(es.n_rows, N);
		for (int j = 0; j < N; ++j)
			for (int k = 0; k < N; ++k)
				projP(j, k) = arma::trace(esT.row(energy_levels[i][j]) * pm * es.col(energy_levels[i][k]));
			
		arma::cx_vec proj_ev; arma::cx_mat proj_es;
		arma::eig_gen(proj_ev, proj_es, projP);
		for (int j = 0; j < N; ++j)
			for (int k = 0; k < N; ++k)
				S_proj.col(j) += proj_es(k, j) * es.col(energy_levels[i][k]);
		S_proj = orthonormalize(S_proj);
		for (int j = 0; j < N; ++j)
			S_f.col(energy_levels[i][j]) = S_proj.col(j);
	}
	return S_f;
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
	int Lx, Ly;
	double tprime;
	double V;
	double stag_mu;
	double mu;
	int k;
	std::string ensemble, geometry;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("Lx", po::value<int>(&Lx)->default_value(3), "x linear lattice dimension")
		("Ly", po::value<int>(&Ly)->default_value(3), "y linear lattice dimension")
		("tprime", po::value<double>(&tprime)->default_value(0.), "d3 hopping")
		("V", po::value<double>(&V)->default_value(1.355), "interaction strength")
		("stag_mu", po::value<double>(&stag_mu)->default_value(0.), "staggered chemical potential")
		("mu", po::value<double>(&mu)->default_value(0.), "chemical potential")
		("k", po::value<int>(&k)->default_value(100), "number of eigenstates")
		("ensemble,e", po::value<std::string>(&ensemble)->default_value("gc"),
			"ensemble: gc or c")
		("geometry,g", po::value<std::string>(&geometry)->default_value("rhom"),
			"geometry: rhombic/tilted/hexagonal");
	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);

	if (vm.count("help")) print_help(desc);
	std::cout << "geometry: " << geometry << std::endl;
	std::cout << "ensemble: " << ensemble << std::endl;
	std::cout << "Lx = " << Lx << std::endl;
	std::cout << "Ly = " << Ly << std::endl;
	std::cout << "tprime = " << tprime << std::endl;
	std::cout << "V = " << V << std::endl;
	std::cout << "stag_mu = " << stag_mu << std::endl;
	std::cout << "mu = " << mu << std::endl;
	//Generate lattice
	lattice lat;
	if (geometry == "hex")
	{
		hex_honeycomb h(Lx);
		lat.generate_graph(h);
		h.generate_maps(lat);
	}
	else if (geometry == "tilted")
	{
		tilted_honeycomb hc(Lx, Ly);
		lat.generate_graph(hc);
		hc.generate_maps(lat);
	}
	else if (geometry == "rhom")
	{
		honeycomb h(Lx, Ly);
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


	std::vector<std::map<int, int>> cb_bonds(3);
	for (int i = 0; i < lat.n_sites(); ++i)
	{
		auto& nn = lat.neighbors(i, "nearest neighbors");
		for (int j : nn)
		{
			for (auto& b : cb_bonds)
			{
				if (!b.count(i) && !b.count(j))
				{
					b[i] = j;
					b[j] = i;
					break;
				}
			}
		}
	}
	auto get_bond_type = [&] (const std::pair<int, int>& bond) -> int
	{
		for (int i = 0; i < cb_bonds.size(); ++i)
			if (cb_bonds[i].at(bond.first) == bond.second)
				return i;
	};

	//Build Hamiltonian
	std::cout << "Constructing static operators...";
	std::cout.flush();
	sparse_storage<double, int_t> H_st(hspace.sub_dimension());
	hspace.build_operator([&](const std::pair<int_t,
		int_t>& n)
	{
		for (int_t i = 0; i < lat.n_sites(); ++i)
		{
			//(Staggered) Chemical potential: -mu sum_i c_i^dag c_i
			H_st(n.second, n.second) -= lat.parity(i) * stag_mu * hspace.n_i({1, n.first}, i);
			H_st(n.second, n.second) -= mu * hspace.n_i({1, n.first}, i);

			for (int_t j : lat.neighbors(i, "nearest neighbors"))
			{
				//Hopping term: -t sum_<ij> c_i^dag c_j
				state m = hspace.c_i({1, n.first}, j);
				m = hspace.c_dag_i(m, i);

				if (m.sign != 0)
					H_st(hspace.index(m.id), n.second) += m.sign * (-1.);

				//Interaction: V sum_<ij> (n_i - 0.5) (n_j - 0.5)
				if (i < j)
					H_st(n.second, n.second) += V * (hspace.n_i({1, n.first}, i)
						- 0.5) * (hspace.n_i({1, n.first}, j) - 0.5);
					//H_st(n.second, n.second) += V * (hspace.n_i({1, n.first}, i)
					//	- 0.0) * (hspace.n_i({1, n.first}, j) - 0.0);
			}
		}
		for (auto& a : lat.bonds("t3_bonds"))
		{
			state m = hspace.c_i({1, n.first}, a.second);
			m = hspace.c_dag_i(m, a.first);
			if (m.sign != 0)
				H_st(hspace.index(m.id), n.second) += m.sign * (-1.) * tprime;
		}
	});
	arma::sp_mat H_real = H_st.build_matrix();
	H_st.clear();
	sparse_storage<std::complex<double>, int_t> H_cx_st(hspace.sub_dimension());
	hspace.build_operator([&](const std::pair<int_t,
		int_t>& n)
	{
		for (int_t i = 0; i < lat.n_sites(); ++i)
		{
			//(Staggered) Chemical potential: -mu sum_i c_i^dag c_i
			H_cx_st(n.second, n.second) -= lat.parity(i) * stag_mu * hspace.n_i({1, n.first}, i);
			H_cx_st(n.second, n.second) -= mu * hspace.n_i({1, n.first}, i);

			for (int_t j : lat.neighbors(i, "nearest neighbors"))
			{
				//Hopping term: -t sum_<ij> c_i^dag c_j
				state m = hspace.c_i({1, n.first}, j);
				m = hspace.c_dag_i(m, i);

				if (m.sign != 0)
					H_cx_st(hspace.index(m.id), n.second) += m.sign * (-1.);

				//Interaction: V sum_<ij> (n_i - 0.5) (n_j - 0.5)
				if (i < j)
					H_cx_st(n.second, n.second) += V * (hspace.n_i({1, n.first}, i)
						- 0.5) * (hspace.n_i({1, n.first}, j) - 0.5);
					//H_cx_st(n.second, n.second) += V * (hspace.n_i({1, n.first}, i)
					//	- 0.0) * (hspace.n_i({1, n.first}, j) - 0.0);
			}
		}
		for (auto& a : lat.bonds("t3_bonds"))
		{
			state m = hspace.c_i({1, n.first}, a.second);
			m = hspace.c_dag_i(m, a.first);
			if (m.sign != 0)
				H_cx_st(hspace.index(m.id), n.second) += m.sign * (-1.) * tprime;
		}
	});
	arma::sp_cx_mat H = H_cx_st.build_matrix();
	H_cx_st.clear();

	//Build static observables
	sparse_storage<std::complex<double>, int_t> n_total_st(hspace.sub_dimension());
	hspace.build_operator([&] (const std::pair<int_t, int_t>& n)
	{
		for (int_t i = 0; i < lat.n_sites(); ++i)
			n_total_st(n.second, n.second) += hspace.n_i({1, n.first}, i);
	});
	arma::sp_cx_mat n_total_op = n_total_st.build_matrix();
	n_total_st.clear();

	sparse_storage<std::complex<double>, int_t> P_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			state m = {1, n.first};
			state m_new = {1, 0};
			std::vector<int> sites;
			for (int_t i = 0; i < lat.n_sites(); ++i)
			{
				if (hspace.n_i({1, n.first}, i) > 0)
					m_new = hspace.c_dag_i(m_new, lat.inverted_site(i));
			}
			if (m_new.sign != 0)
				P_st(hspace.index(m_new.id), n.second) += m_new.sign;
		});
	arma::sp_cx_mat P_op = P_st.build_matrix();
	P_st.clear();
	
	double rot_angle = 60.;
	sparse_storage<std::complex<double>, int_t> P_rot60_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			state m = {1, n.first};
			state m_new = {1, 0};
			std::vector<int> sites;
			for (int_t i = 0; i < lat.n_sites(); ++i)
			{
				if (hspace.n_i({1, n.first}, i) > 0)
					m_new = hspace.c_dag_i(m_new, lat.rotated_site(i, rot_angle));
			}
			if (m_new.sign != 0)
				P_rot60_st(hspace.index(m_new.id), n.second) += m_new.sign;
		});
	arma::sp_cx_mat P_rot60_op = P_rot60_st.build_matrix();
	P_rot60_st.clear();
	
	rot_angle = 120.;
	sparse_storage<std::complex<double>, int_t> P_rot120_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			state m = {1, n.first};
			state m_new = {1, 0};
			std::vector<int> sites;
			for (int_t i = 0; i < lat.n_sites(); ++i)
			{
				if (hspace.n_i({1, n.first}, i) > 0)
					m_new = hspace.c_dag_i(m_new, lat.rotated_site(i, rot_angle));
			}
			if (m_new.sign != 0)
				P_rot120_st(hspace.index(m_new.id), n.second) += m_new.sign;
		});
	arma::sp_cx_mat P_rot120_op = P_rot120_st.build_matrix();
	P_rot120_st.clear();
	
	sparse_storage<std::complex<double>, int_t> P_sv_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			state m = {1, n.first};
			state m_new = {1, 0};
			std::vector<int> sites;
			for (int_t i = 0; i < lat.n_sites(); ++i)
			{
				if (hspace.n_i({1, n.first}, i) > 0)
					m_new = hspace.c_dag_i(m_new, lat.reflected_v_site(i));
			}
			if (m_new.sign != 0)
				P_sv_st(hspace.index(m_new.id), n.second) += m_new.sign;
		});
	arma::sp_cx_mat P_sv_op = P_sv_st.build_matrix();
	P_sv_st.clear();
	
	sparse_storage<std::complex<double>, int_t> P_sh_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			state m = {1, n.first};
			state m_new = {1, 0};
			std::vector<int> sites;
			for (int_t i = 0; i < lat.n_sites(); ++i)
			{
				if (hspace.n_i({1, n.first}, i) > 0)
					m_new = hspace.c_dag_i(m_new, lat.reflected_h_site(i));
			}
			if (m_new.sign != 0)
				P_sh_st(hspace.index(m_new.id), n.second) += m_new.sign;
		});
	arma::sp_cx_mat P_sh_op = P_sh_st.build_matrix();
	P_sh_st.clear();
	
	sparse_storage<std::complex<double>, int_t> PH_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			state m = hspace.ph_symmetry({1, n.first});
			PH_st(hspace.index(m.id), n.second) += m.sign;
		});
	arma::sp_cx_mat PH_op = PH_st.build_matrix();
	PH_st.clear();

	std::cout << "Done." << std::endl;

	std::string out_file = "../data/ed_" + geometry + "__Lx_" + std::to_string(Lx) + "__"
		+ "Ly_" + std::to_string(Ly) + "__"
		+ "V_" + std::to_string(V) + "__"
		+ "GS__"
		+ ensemble;
	std::ofstream out("../data/" + out_file);
	std::cout.precision(8);
	out.precision(18);
	arma::vec ev; arma::mat es;
	if (hspace.sub_dimension() < 1000)
	{
		arma::mat H_dense(H_real);
		arma::eig_sym(ev, es, H_dense);
	}
	else
	{
		arma::eigs_sym(ev, es, H_real, std::min(hspace.sub_dimension()-2,
			static_cast<int_t>(k)), "sa");
	}
	H_real.clear();
	arma::cx_mat es_cx = arma::conv_to<arma::cx_mat>::from(es);
	arma::cx_mat esT_cx = es_cx.t();
	
	std::cout << "GS space" << std::endl;
	std::cout << "GS energy: " << ev[0] << std::endl;
	int degeneracy = 0;
	for (int i = 0; i < ev.n_rows && std::abs(ev[i] - ev[0]) <= std::pow(10, -12); ++i)
		++degeneracy;
	std::cout << "GS degeneracy: " << degeneracy << std::endl;
	
	/*
	auto energy_levels = get_energy_levels(ev);
	es_cx = project_symmetry(es_cx, energy_levels, PH_op);
	split_quantum_numbers(energy_levels, es_cx, PH_op);
		
	for (int i = 0; i < es_cx.n_cols; ++i)
	{
		std::cout << "E(" << i << ") = " << (arma::trace(esT_cx.row(i) * H * es_cx.col(i))) << std::endl;
		std::cout << "N(" << i << ") = " << (arma::trace(esT_cx.row(i) * n_total_op * es_cx.col(i))) << std::endl;
	}
	*/
	
	if (ensemble == "c")
	{
		auto energy_levels = get_energy_levels(ev);
		es_cx = project_symmetry(es_cx, energy_levels, P_op);
		split_quantum_numbers(energy_levels, es_cx, P_op);
		es_cx = project_symmetry(es_cx, energy_levels, P_sv_op);
		split_quantum_numbers(energy_levels, es_cx, P_sv_op);
		es_cx = project_symmetry(es_cx, energy_levels, P_sh_op);
		split_quantum_numbers(energy_levels, es_cx, P_sh_op);
		es_cx = project_symmetry(es_cx, energy_levels, P_rot60_op);
		split_quantum_numbers(energy_levels, es_cx, P_rot60_op);
		es_cx = project_symmetry(es_cx, energy_levels, P_rot120_op);
		split_quantum_numbers(energy_levels, es_cx, P_rot120_op);
		es_cx = project_symmetry(es_cx, energy_levels, PH_op);
		split_quantum_numbers(energy_levels, es_cx, PH_op);
		esT_cx = es_cx.t();
		
		//if (degeneracy > 1)
		{
			arma::sp_cx_mat id = arma::speye<arma::sp_cx_mat>(H.n_rows, H.n_cols);
			std::cout << "Inversion symmetry" << std::endl;
			std::cout << "P * H - H * P = " << arma::norm(arma::nonzeros(P_op * H - H * P_op)) << std::endl;
			std::cout << "id - P * P = " << arma::norm(arma::nonzeros(id - P_op * P_op)) << std::endl;
			
			std::cout << "Rotation symmetry: " << lat.check_rotation_symmetry(rot_angle) << std::endl;
			std::cout << "P_rot60 * H - H * P_rot60 = " << arma::norm(arma::nonzeros(P_rot60_op * H - H * P_rot60_op)) << std::endl;
			std::cout << "id - P_rot60 * P_rot60 = " << arma::norm(arma::nonzeros(id - P_rot60_op * P_rot60_op)) << std::endl;
			std::cout << "P_rot120 * H - H * P_rot120 = " << arma::norm(arma::nonzeros(P_rot120_op * H - H * P_rot120_op)) << std::endl;
			std::cout << "id - P_rot120 * P_rot120 = " << arma::norm(arma::nonzeros(id - P_rot120_op * P_rot120_op)) << std::endl;
			
			std::cout << "Reflection symmetry" << std::endl;
			std::cout << "P_sv * H - H * P_sv = " << arma::norm(arma::nonzeros(P_sv_op * H - H * P_sv_op)) << std::endl;
			std::cout << "id - P_sv * P_sv = " << arma::norm(arma::nonzeros(id - P_sv_op * P_sv_op)) << std::endl;
			std::cout << "P_sh * H - H * P_sh = " << arma::norm(arma::nonzeros(P_sh_op * H - H * P_sh_op)) << std::endl;
			std::cout << "id - P_sh * P_sh = " << arma::norm(arma::nonzeros(id - P_sh_op * P_sh_op)) << std::endl;

			std::cout << "Particle hole symmetry" << std::endl;
			std::cout << "PH * H - H * PH = " << arma::norm(arma::nonzeros(PH_op * H - H * PH_op)) << std::endl;
			std::cout << "id - PH * PH = " << arma::norm(arma::nonzeros(id - PH_op * PH_op)) << std::endl;
			
			std::cout << "PH * P - P * PH = " << arma::norm(arma::nonzeros(PH_op * P_op - P_op * PH_op)) << std::endl;
			std::cout << "PH * P60 - P60 * PH = " << arma::norm(arma::nonzeros(PH_op * P_rot60_op - P_rot60_op * PH_op)) << std::endl;
			std::cout << "PH * P120 - P120 * PH = " << arma::norm(arma::nonzeros(PH_op * P_rot120_op - P_rot120_op * PH_op)) << std::endl;
		}
		std::cout << "#\tE\t\tP_inv\t\tP_rot60\t\tP_rot120\t\tP_sv\t\tP_sh\t\tPH" << std::endl;
		for (int i = 0; i < 6; ++i)
		{
			//std::cout << "E(" << i << ") = " << arma::dot(es_cx.col(i), H * es_cx.col(i)) << std::endl;
			//std::cout << "N(" << i << ") = " << arma::dot(es_cx.col(i), n_total_op * es_cx.col(i)) << std::endl;
			//std::cout << "P(" << i << ") = " << arma::dot(es_cx.col(i), P_op * es_cx.col(i)) << std::endl;
			//std::cout << "PH(" << i << ") = " << arma::dot(es_cx.col(i), PH_op * es_cx.col(i)) << std::endl;
			
			std::cout << i << "\t" << std::real(arma::trace(esT_cx.row(i) * H * es_cx.col(i))) << "\t"
				<< std::setprecision(2) << (arma::trace(esT_cx.row(i) * P_op * es_cx.col(i))) << "\t\t"
				<< (arma::trace(esT_cx.row(i) * P_rot60_op * es_cx.col(i))) << "\t\t"
				<< (arma::trace(esT_cx.row(i) * P_rot120_op * es_cx.col(i))) << "\t\t"
				<< (arma::trace(esT_cx.row(i) * P_sv_op * es_cx.col(i))) << "\t\t"
				<< (arma::trace(esT_cx.row(i) * PH_op * es_cx.col(i))) << std::setprecision(6) << std::endl;
		}
		
		for (int i = 0; i < es_cx.n_cols; ++i)
		{
			std::cout << "E(" << i << ") = " << (arma::trace(esT_cx.row(i) * H * es_cx.col(i))) << std::endl;
			std::cout << "P_rot60(" << i << ") = " << (arma::trace(esT_cx.row(i) * P_rot60_op * es_cx.col(i))) << std::endl;
			std::cout << "P_rot120(" << i << ") = " << (arma::trace(esT_cx.row(i) * P_rot120_op * es_cx.col(i))) << std::endl;
			std::cout << "P(" << i << ") = " << (arma::trace(esT_cx.row(i) * P_op * es_cx.col(i))) << std::endl;
			std::cout << "P_sv(" << i << ") = " << (arma::trace(esT_cx.row(i) * P_sv_op * es_cx.col(i))) << std::endl;
			std::cout << "P_sh(" << i << ") = " << (arma::trace(esT_cx.row(i) * P_sh_op * es_cx.col(i))) << std::endl;
			std::cout << "PH(" << i << ") = " << (arma::trace(esT_cx.row(i) * PH_op * es_cx.col(i))) << std::endl;
			std::cout << "-----" << std::endl;
		}
	}

	std::complex<double> E = 0., m2 = 0., m4 = 0., cij = 0., n_total = 0.;
	for (int i = 0; i < degeneracy; ++i)
	{
		E += ev(i) / static_cast<double>(degeneracy);
		n_total += arma::trace(esT_cx.row(i) * n_total_op * es_cx.col(i)) / static_cast<double>(degeneracy);
	}
	H.clear();
	n_total_op.clear();

	int Ntau = 50, Nmat = 0;
	double t_step = 0.2;
	out << k << "\t" << Lx << "\t" << Ly << "\t" << V << "\t" << 0 << "\t"
		<< E << "\t" << m2 << "\t" << m4 << "\t" << m4/(m2*m2) << "\t"
		<< Ntau << "\t" << Nmat << std::endl;

	// Build dynamic observables
	std::cout << "Constructing dynamic operators..." << std::endl;
	std::cout.flush();
	std::vector<std::vector<std::complex<double>>> obs_data_cx;
	std::complex<double> cdw2 = 0., cdw4 = 0., cdw2_0 = 0., cdw2_1 = 0., S_cdw = 0., ep = 0., ep_0 = 0., ep_1 = 0., kek_0 = 0., kek_1 = 0.,
		chern = 0., chern2 = 0., chern4 = 0., S_chern = 0., chernAA2 = 0.;
	std::complex<double> h_t = 0., h_v = 0., h_mu = 0.;

	sparse_storage<std::complex<double>, int_t> ht_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			for (int_t i = 0; i < lat.n_sites(); ++i)
			{
				for (int_t j : lat.neighbors(i, "nearest neighbors"))
				{
					//Hopping term: -t sum_<ij> c_i^dag c_j
					state m = hspace.c_i({-1, n.first}, j);
					m = hspace.c_dag_i(m, i);
					double tp = -1.;

					if (m.sign != 0)
						ht_st(hspace.index(m.id), n.second) += m.sign * tp;
				}
			}
			for (auto& a : lat.bonds("d3_bonds"))
			{
				state m = hspace.c_i({-1, n.first}, a.second);
				m = hspace.c_dag_i(m, a.first);
				if (m.sign != 0)
					ht_st(hspace.index(m.id), n.second) += m.sign * (-1.) * tprime;
			}
		});
	arma::sp_cx_mat ht_op = ht_st.build_matrix();
	ht_st.clear();
	for (int i = 0; i < degeneracy; ++i)
		h_t += arma::trace(esT_cx.row(i) * ht_op * es_cx.col(i)) / std::complex<double>(degeneracy);
	ht_op.clear();

	sparse_storage<std::complex<double>, int_t> hv_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			for (int_t i = 0; i < lat.n_sites(); ++i)
				for (int_t j : lat.neighbors(i, "nearest neighbors"))
				{
					//Interaction: V sum_<ij> (n_i - 0.5) (n_j - 0.5)
					
					if (i < j)
						hv_st(n.second, n.second) += V * (hspace.n_i({1, n.first}, i)
							- 0.5) * (hspace.n_i({1, n.first}, j) - 0.5);
					
					//if (i < j)
					//	hv_st(n.second, n.second) += V * hspace.n_i({1, n.first}, i) * hspace.n_i({1, n.first}, j);
				}
		});
	arma::sp_cx_mat hv_op = hv_st.build_matrix();
	hv_st.clear();
	for (int i = 0; i < degeneracy; ++i)
		h_v += arma::trace(esT_cx.row(i) * hv_op * es_cx.col(i)) / std::complex<double>(degeneracy);
	hv_op.clear();

	sparse_storage<std::complex<double>, int_t> hmu_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			for (int_t i = 0; i < lat.n_sites(); ++i)
			{
				//(Staggered) Chemical potential: -mu sum_i c_i^dag c_i
				hmu_st(n.second, n.second) -= lat.parity(i) * stag_mu * hspace.n_i({1, n.first}, i);
				hmu_st(n.second, n.second) -= mu * hspace.n_i({1, n.first}, i);
			}
		});
	arma::sp_cx_mat hmu_op = hmu_st.build_matrix();
	hmu_st.clear();
	for (int i = 0; i < degeneracy; ++i)
		h_mu += arma::trace(esT_cx.row(i) * hmu_op * es_cx.col(i)) / std::complex<double>(degeneracy);
	hmu_op.clear();

	sparse_storage<std::complex<double>, int_t> ni_st(hspace.sub_dimension());
	sparse_storage<std::complex<double>, int_t> S_cdw_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			for (int i = 0; i < lat.n_sites(); ++i)
			{
				//CDW
				ni_st(n.second, n.second)
					+= lat.parity(i) / std::complex<double>(lat.n_sites())
						* (hspace.n_i({1, n.first}, i) - 0.5);
				auto& r_i = lat.real_space_coord((i/2)*2);
				auto& q = lat.symmetry_point("q");
				std::complex<double> im = {0., 1.};
				S_cdw_st(n.second, n.second)
					+= lat.parity(i) / std::complex<double>(lat.n_sites())
						* (hspace.n_i({1, n.first}, i) - 0.5) * std::exp(im * q.dot(r_i));
			}
		});
	arma::sp_cx_mat ni_op = ni_st.build_matrix();
	ni_st.clear();
	obs_data_cx.push_back(get_imaginary_time_obs(ni_op, Ntau, t_step, degeneracy, ev,
		es_cx, esT_cx));
	print_overlap(ni_op, "cdw", degeneracy, ev, es_cx, esT_cx, P_op, PH_op);
	//print_overlap(ni_op, "cdw", degeneracy, ev, es_cx, esT_cx);
	arma::sp_cx_mat ni2_op = ni_op * ni_op;
	ni_op.clear();
	arma::sp_cx_mat S_cdw_op = S_cdw_st.build_matrix();
	S_cdw_st.clear();
	for (int i = 0; i < degeneracy; ++i)
	{
		cdw2 += arma::trace(esT_cx.row(i) * ni2_op * es_cx.col(i)) / std::complex<double>(degeneracy);
		cdw4 += arma::trace(esT_cx.row(i) * ni2_op * ni2_op * es_cx.col(i)) / std::complex<double>(degeneracy);
		S_cdw += arma::trace(esT_cx.row(i) * S_cdw_op * S_cdw_op.t() * es_cx.col(i)) / std::complex<double>(degeneracy);
	}
	cdw2_0 += arma::trace(esT_cx.row(0) * ni2_op * es_cx.col(0));
	cdw2_1 += arma::trace(esT_cx.row(1) * ni2_op * es_cx.col(1));
	ni2_op.clear();
	S_cdw_op.clear();
	print_data(out, obs_data_cx[0]);
	
	sparse_storage<std::complex<double>, int_t> ep_V_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			for (auto& a : lat.bonds("nearest neighbors"))
			{
				if (a.first > a.second) continue;
				//ep_V_st(n.second, n.second)
				//	+= 1. / std::complex<double>(lat.n_bonds())
				//		* (hspace.n_i({1, n.first}, a.first) - 0.5) * (hspace.n_i({1, n.first}, a.second) - 0.5);

				ep_V_st(n.second, n.second)
					+= 1. / std::complex<double>(lat.n_bonds())
						* std::complex<double>(hspace.n_i({1, n.first}, a.first)) * std::complex<double>(hspace.n_i({1, n.first}, a.second));
			}
		});
	arma::sp_cx_mat ep_V_op = ep_V_st.build_matrix();
	ep_V_st.clear();
	obs_data_cx.push_back(get_imaginary_time_obs(ep_V_op, Ntau, t_step, degeneracy, ev,
		es_cx, esT_cx));
	print_overlap(ep_V_op, "epsilon_V", degeneracy, ev, es_cx, esT_cx, P_op, PH_op);
	ep_V_op.clear();
	print_data(out, obs_data_cx[1]);

	sparse_storage<std::complex<double>, int_t> ep_as_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			std::vector<const std::vector<std::pair<int, int>>*> bonds =
				{&lat.bonds("nn_bond_1"), &lat.bonds("nn_bond_2"),
				&lat.bonds("nn_bond_3")};
			for (int i = 0; i < bonds.size(); ++i)
				for (int j = 0; j < bonds[i]->size(); ++j)
				{
					auto& b = (*bonds[i])[j];
					state p = hspace.c_i({1, n.first}, b.second);
					p = hspace.c_dag_i(p, b.first);
					if (p.sign != 0)
						ep_as_st(hspace.index(p.id), n.second) += std::complex<double>(p.sign)
							/ std::complex<double>(lat.n_bonds());
					p = hspace.c_i({1, n.first}, b.first);
					p = hspace.c_dag_i(p, b.second);
					if (p.sign != 0)
						ep_as_st(hspace.index(p.id), n.second) += -std::complex<double>(p.sign)
							/ std::complex<double>(lat.n_bonds());
				}
		});
	arma::sp_cx_mat ep_as_op = ep_as_st.build_matrix();
	ep_as_st.clear();
	obs_data_cx.push_back(get_imaginary_time_obs(ep_as_op, Ntau, t_step, degeneracy, ev,
		es_cx, esT_cx));
	print_overlap(ep_as_op, "ep_as", degeneracy, ev, es_cx, esT_cx, P_op, PH_op);
	ep_as_op.clear();
	print_data(out, obs_data_cx[2]);

	sparse_storage<std::complex<double>, int_t> ni_sym_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			for (auto& b : lat.bonds("chern"))
			{
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					ni_sym_st(hspace.index(p.id), n.second) +=
						std::complex<double>(p.sign)
						/ static_cast<double>(lat.n_bonds());
						
				p = hspace.c_i({1, n.first}, b.first);
				p = hspace.c_dag_i(p, b.second);
				if (p.sign != 0)
					ni_sym_st(hspace.index(p.id), n.second) +=
						std::complex<double>(p.sign)
						/ static_cast<double>(lat.n_bonds());
			}

			for (auto& b : lat.bonds("chern_2"))
			{
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					ni_sym_st(hspace.index(p.id), n.second) +=
						-std::complex<double>(p.sign)
						/ static_cast<double>(lat.n_bonds());
						
				p = hspace.c_i({1, n.first}, b.first);
				p = hspace.c_dag_i(p, b.second);
				if (p.sign != 0)
					ni_sym_st(hspace.index(p.id), n.second) +=
						-std::complex<double>(p.sign)
						/ static_cast<double>(lat.n_bonds());
			}
		});
	arma::sp_cx_mat ni_sym_op = ni_sym_st.build_matrix();
	ni_sym_st.clear();
	print_overlap(ni_sym_op, "cdw_sym", degeneracy, ev, es_cx, esT_cx, P_op, PH_op);
	//print_overlap(ni_sym_op, "cdw_sym_K", degeneracy, ev, es_cx, esT_cx);
	ni_sym_op.clear();

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
						-std::complex<double>(p.sign)
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
						2.*std::complex<double>(p.sign)
						/ static_cast<double>(lat.n_bonds());
			}
		});
	arma::sp_cx_mat kekule_op = kekule_st.build_matrix();
	kekule_st.clear();
	obs_data_cx.push_back(get_imaginary_time_obs(kekule_op, Ntau, t_step, degeneracy,
		ev, es_cx, esT_cx));
	kek_0 += arma::trace(esT_cx.row(0) * kekule_op * kekule_op.t() * es_cx.col(0));
	kek_1 += arma::trace(esT_cx.row(1) * kekule_op * kekule_op.t() * es_cx.col(1));
	print_overlap(kekule_op, "kekule", degeneracy, ev, es_cx, esT_cx, P_op, PH_op);
	kekule_op.clear();
	print_data(out, obs_data_cx[3]);
	
	sparse_storage<std::complex<double>, int_t> kekule2_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			for (auto& b : lat.bonds("kekule"))
			{
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					kekule2_st(hspace.index(p.id), n.second) +=
						std::complex<double>(p.sign)
						/ static_cast<double>(lat.n_bonds());
			}

			for (auto& b : lat.bonds("kekule_2"))
			{
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					kekule2_st(hspace.index(p.id), n.second) +=
						-std::complex<double>(p.sign)
						/ static_cast<double>(lat.n_bonds());
			}
			
			/*
			std::complex<double> im = {0., 1.};
			for (auto& b : lat.bonds("kekule"))
			{
				auto& r_i = lat.real_space_coord((b.first/2)*2);
				auto& q = lat.symmetry_point("K");
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					kekule2_st(hspace.index(p.id), n.second) +=
						std::complex<double>(p.sign)
						/ static_cast<double>(lat.n_bonds()) * std::exp(im * q.dot(r_i));
			}
			for (auto& b : lat.bonds("kekule_2"))
			{
				auto& r_i = lat.real_space_coord((b.first/2)*2);
				auto& q = lat.symmetry_point("K");
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					kekule2_st(hspace.index(p.id), n.second) +=
						std::complex<double>(p.sign)
						/ static_cast<double>(lat.n_bonds()) * std::exp(im * q.dot(r_i));
			}
			*/
		});
	arma::sp_cx_mat kekule2_op = kekule2_st.build_matrix();
	kekule2_st.clear();
	obs_data_cx.push_back(get_imaginary_time_obs(kekule2_op, Ntau, t_step, degeneracy,
		ev, es_cx, esT_cx));
	print_overlap(kekule2_op, "kekule_as", degeneracy, ev, es_cx, esT_cx, P_op, PH_op);
	kekule2_op.clear();
	print_data(out, obs_data_cx[4]);
	
	sparse_storage<std::complex<double>, int_t> kekule_alpha_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			double pi = 4. * std::atan(1.);
			//double alpha = pi/2.; //kekule_as
			//double alpha = pi; //kekule_s
			double alpha = 4.*pi/3.;
			
			for (auto& b : lat.bonds("kekule"))
			{
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					kekule_alpha_st(hspace.index(p.id), n.second) +=
						std::complex<double>(p.sign) * std::cos(alpha - 2.*pi/3.)
						/ static_cast<double>(lat.n_bonds());
			}

			for (auto& b : lat.bonds("kekule_2"))
			{
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					kekule_alpha_st(hspace.index(p.id), n.second) +=
						std::complex<double>(p.sign) * std::cos(alpha + 2.*pi/3.)
						/ static_cast<double>(lat.n_bonds());
			}
			
			for (auto& b : lat.bonds("kekule_3"))
			{
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
					kekule_alpha_st(hspace.index(p.id), n.second) +=
						std::complex<double>(p.sign) * std::cos(alpha)
						/ static_cast<double>(lat.n_bonds());
			}
		});
	arma::sp_cx_mat kekule_alpha_op = kekule_alpha_st.build_matrix();
	kekule_alpha_st.clear();
	obs_data_cx.push_back(get_imaginary_time_obs(kekule_alpha_op, Ntau, t_step, degeneracy,
		ev, es_cx, esT_cx));
	print_overlap(kekule_alpha_op, "kekule_alpha", degeneracy, ev, es_cx, esT_cx, P_op, PH_op);
	kekule_alpha_op.clear();
	print_data(out, obs_data_cx[5]);

	sparse_storage<std::complex<double>, int_t> chern_st(hspace.sub_dimension());
	sparse_storage<std::complex<double>, int_t> S_chern_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			//chern
			for (auto& b : lat.bonds("chern"))
			{
				auto& r_i = lat.real_space_coord(b.first);
				auto& q = lat.symmetry_point("q");
				std::complex<double> im = {0., 1.};
				
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
				{
					chern_st(hspace.index(p.id), n.second) += std::complex<double>{0., p.sign
						/ static_cast<double>(lat.n_bonds())};
					S_chern_st(hspace.index(p.id), n.second) += std::complex<double>{0., p.sign
						/ static_cast<double>(lat.n_bonds())} * std::exp(im * q.dot(r_i));
				}
				p = hspace.c_i({1, n.first}, b.first);
				p = hspace.c_dag_i(p, b.second);
				if (p.sign != 0)
				{
					chern_st(hspace.index(p.id), n.second) += std::complex<double>{0., -p.sign
						/ static_cast<double>(lat.n_bonds())};
					S_chern_st(hspace.index(p.id), n.second) += std::complex<double>{0., -p.sign
						/ static_cast<double>(lat.n_bonds())} * std::exp(im * q.dot(r_i));
				}
			}
			
			for (auto& b : lat.bonds("chern_2"))
			{
				auto& r_i = lat.real_space_coord(b.first);
				auto& q = lat.symmetry_point("q");
				std::complex<double> im = {0., 1.};
				
				state p = hspace.c_i({1, n.first}, b.second);
				p = hspace.c_dag_i(p, b.first);
				if (p.sign != 0)
				{
					chern_st(hspace.index(p.id), n.second) += std::complex<double>{0., -p.sign
						/ static_cast<double>(lat.n_bonds())};
					S_chern_st(hspace.index(p.id), n.second) += std::complex<double>{0., -p.sign
						/ static_cast<double>(lat.n_bonds())} * std::exp(im * q.dot(r_i));
				}
				p = hspace.c_i({1, n.first}, b.first);
				p = hspace.c_dag_i(p, b.second);
				if (p.sign != 0)
				{
					chern_st(hspace.index(p.id), n.second) += std::complex<double>{0., p.sign
						/ static_cast<double>(lat.n_bonds())};
					S_chern_st(hspace.index(p.id), n.second) += std::complex<double>{0., p.sign
						/ static_cast<double>(lat.n_bonds())} * std::exp(im * q.dot(r_i));
				}
			}
		});
	arma::sp_cx_mat chern_op = chern_st.build_matrix();
	chern_st.clear();
	obs_data_cx.push_back(get_imaginary_time_obs(chern_op, Ntau, t_step, degeneracy,
		ev, es_cx, esT_cx));
	print_overlap(chern_op, "chern", degeneracy, ev, es_cx, esT_cx, P_op, PH_op);
	arma::sp_cx_mat S_chern_op = S_chern_st.build_matrix();
	S_chern_st.clear();
	arma::sp_cx_mat chern2_op = chern_op * chern_op;
	for (int i = 0; i < degeneracy; ++i)
	{
		chern += arma::trace(esT_cx.row(i) * chern_op * es_cx.col(i)) / std::complex<double>(degeneracy);
		chern2 += arma::trace(esT_cx.row(i) * chern2_op * es_cx.col(i)) / std::complex<double>(degeneracy);
		chern4 += arma::trace(esT_cx.row(i) * chern2_op * chern2_op * es_cx.col(i)) / std::complex<double>(degeneracy);
		S_chern += arma::trace(esT_cx.row(i) * S_chern_op * S_chern_op.t() * es_cx.col(i)) / std::complex<double>(degeneracy);
	}
	chern_op.clear();
	chern2_op.clear();
	S_chern_op.clear();
	print_data(out, obs_data_cx[6]);
	
	std::vector<std::complex<double>> S_chernAA;
	std::vector<Eigen::Vector2d> hexagon_pos;
	std::vector<Eigen::Vector2d> q_vec;
	auto& bonds = lat.bonds("chern");
	Eigen::Vector2d delta = {1., 0};
	for (int i = 0; i < bonds.size(); i+=3)
	{
		Eigen::Vector2d r = lat.real_space_coord(bonds[i].first) + delta;
		hexagon_pos.push_back(r);
	}
	auto& G = lat.symmetry_point("Gamma");
	auto M = lat.symmetry_point("M");
	auto& K = lat.symmetry_point("K");
	for (int i = 0; i <= Lx / 2; ++i)
		q_vec.push_back(G + (lat.b1 + lat.b2) * static_cast<double>(i) / static_cast<double>(Lx));
	if (Lx % 2 != 0)
	{
		M = q_vec[Lx / 2] + lat.b1 / static_cast<double>(Lx);
		q_vec.push_back(M);
	}
	for (int i = 1; i <= Lx / 6; ++i)
		q_vec.push_back(M + (lat.b1 - lat.b2) * static_cast<double>(i) / static_cast<double>(Lx));
	for (int i = 1; i < Lx / 3; ++i)
		q_vec.push_back(K + (-2.*lat.b1 - lat.b2) * static_cast<double>(i) / static_cast<double>(Lx));
	for (int k = 0; k < q_vec.size(); ++k)
	{
		sparse_storage<std::complex<double>, int_t> chernAA_st(hspace.sub_dimension());
		hspace.build_operator([&]
			(const std::pair<int_t, int_t>& n)
			{
				std::complex<double> im = {0., 1.};
				//chern
				for (int i = 0; i < bonds.size(); ++i)
				{
					state p = hspace.c_i({1, n.first}, bonds[i].second);
					p = hspace.c_dag_i(p, bonds[i].first);
					if (p.sign != 0)
						chernAA_st(hspace.index(p.id), n.second) += std::complex<double>{0., p.sign
							/ static_cast<double>(lat.n_bonds())} * std::exp(im * q_vec[k].dot(hexagon_pos[i/3]));
					p = hspace.c_i({1, n.first}, bonds[i].first);
					p = hspace.c_dag_i(p, bonds[i].second);
					if (p.sign != 0)
						chernAA_st(hspace.index(p.id), n.second) += std::complex<double>{0., -p.sign
							/ static_cast<double>(lat.n_bonds())} * std::exp(im * q_vec[k].dot(hexagon_pos[i/3]));
				}
			});
		arma::sp_cx_mat chernAA_op = chernAA_st.build_matrix();
		chernAA_st.clear();
		arma::sp_cx_mat chernAA2_op = chernAA_op * chernAA_op.t();
		chernAA_op.clear();
		for (int i = 0; i < 2; ++i)
			S_chernAA.push_back(arma::trace(esT_cx.row(i) * chernAA2_op * es_cx.col(i)));
		chernAA2_op.clear();
	}

	sparse_storage<std::complex<double>, int_t> gamma_mod_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			double pi = 4. * std::atan(1.);
			std::complex<double> im = {0., 1.};
			
			std::vector<const std::vector<std::pair<int, int>>*> bonds =
				{&lat.bonds("nn_bond_1"), &lat.bonds("nn_bond_2"),
				&lat.bonds("nn_bond_3")};
			std::vector<std::complex<double>> phases_cx = {2.*im*std::sin(0. * pi), 2.*im*std::sin(2./3. * pi), 2.*im*std::sin(4./3. * pi)};
			for (int i = 0; i < bonds.size(); ++i)
				for (int j = 0; j < bonds[i]->size(); ++j)
				{
					auto& b = (*bonds[i])[j];
					state p = hspace.c_i({1, n.first}, b.second);
					p = hspace.c_dag_i(p, b.first);
					if (p.sign != 0)
						gamma_mod_st(hspace.index(p.id), n.second) += phases_cx[i] * std::complex<double>(p.sign)
							/ std::complex<double>(lat.n_bonds());
					p = hspace.c_i({1, n.first}, b.first);
					p = hspace.c_dag_i(p, b.second);
					if (p.sign != 0)
						gamma_mod_st(hspace.index(p.id), n.second) += -phases_cx[i] * std::complex<double>(p.sign)
							/ std::complex<double>(lat.n_bonds());
				}
		});
	arma::sp_cx_mat gamma_mod_op = gamma_mod_st.build_matrix();
	gamma_mod_st.clear();
	obs_data_cx.push_back(get_imaginary_time_obs(gamma_mod_op, Ntau, t_step, degeneracy, ev,
		es_cx, esT_cx));
	print_overlap(gamma_mod_op, "gamma_mod", degeneracy, ev, es_cx, esT_cx, P_op, PH_op);
	gamma_mod_op.clear();
	print_data(out, obs_data_cx[7]);
	
	sparse_storage<std::complex<double>, int_t> gamma_mod_s_st(hspace.sub_dimension());
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			double pi = 4. * std::atan(1.);
			std::complex<double> im = {0., 1.};
			
			std::vector<const std::vector<std::pair<int, int>>*> bonds =
				{&lat.bonds("nn_bond_1"), &lat.bonds("nn_bond_2"),
				&lat.bonds("nn_bond_3")};
			std::vector<std::complex<double>> phases_cx = {2.*im*std::sin(0. * pi), 2.*im*std::sin(2./3. * pi), 2.*im*std::sin(4./3. * pi)};
			for (int i = 0; i < bonds.size(); ++i)
				for (int j = 0; j < bonds[i]->size(); ++j)
				{
					auto& b = (*bonds[i])[j];
					state p = hspace.c_i({1, n.first}, b.second);
					p = hspace.c_dag_i(p, b.first);
					if (p.sign != 0)
						gamma_mod_s_st(hspace.index(p.id), n.second) += phases_cx[i] * std::complex<double>(p.sign)
							/ std::complex<double>(lat.n_bonds());
					p = hspace.c_i({1, n.first}, b.first);
					p = hspace.c_dag_i(p, b.second);
					if (p.sign != 0)
						gamma_mod_s_st(hspace.index(p.id), n.second) += phases_cx[i] * std::complex<double>(p.sign)
							/ std::complex<double>(lat.n_bonds());
				}
		});
	arma::sp_cx_mat gamma_mod_s_op = gamma_mod_s_st.build_matrix();
	gamma_mod_s_st.clear();
	obs_data_cx.push_back(get_imaginary_time_obs(gamma_mod_s_op, Ntau, t_step, degeneracy, ev,
		es_cx, esT_cx));
	print_overlap(gamma_mod_s_op, "gamma_mod_s", degeneracy, ev, es_cx, esT_cx, P_op, PH_op);
	gamma_mod_s_op.clear();
	print_data(out, obs_data_cx[8]);

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
	ep_0 += arma::trace(esT_cx.row(0) * epsilon_op * es_cx.col(0));
	ep_1 += arma::trace(esT_cx.row(1) * epsilon_op * es_cx.col(1));
	//Subtract finite expectation value
	hspace.build_operator([&]
		(const std::pair<int_t, int_t>& n)
		{
			epsilon_st(n.second, n.second) -= ep;
		});
	epsilon_op = epsilon_st.build_matrix();
	print_overlap(epsilon_op, "epsilon - <ep>", degeneracy, ev, es_cx, esT_cx, P_op, PH_op);
	//print_overlap(epsilon_op, "epsilon - <ep>", degeneracy, ev, es_cx, esT_cx);
	epsilon_st.clear();
	obs_data_cx.push_back(get_imaginary_time_obs(epsilon_op, Ntau, t_step, degeneracy,
		ev, es_cx, esT_cx));
	epsilon_op.clear();
	print_data(out, obs_data_cx[9]);
	
	if (ensemble == "gc")
	{
		sparse_storage<std::complex<double>, int_t> sp_st(hspace.sub_dimension());
		hspace.build_operator([&]
			(const std::pair<int_t, int_t>& n)
			{
				for (int i = 0; i < lat.n_sites(); i+=2)
				{
					//sp
					auto& K = lat.symmetry_point("K");
					std::complex<double> phase = std::exp(std::complex<double>(0.,
						K.dot(lat.real_space_coord(i))));
					state p = hspace.c_dag_i({1, n.first}, i);
					if (p.sign != 0)
						sp_st(hspace.index(p.id), n.second) += phase
							* std::complex<double>(p.sign) / std::sqrt(Lx * Ly);
					/*
					p = hspace.c_dag_i({1, n.first}, i);
					if (p.sign != 0)
						sp_st(hspace.index(p.id), n.second) += phase
							* std::complex<double>(p.sign) / static_cast<double>(lat.n_sites());
					*/
				}
			});
		arma::sp_cx_mat sp_op = sp_st.build_matrix();
		print_overlap(sp_op, "sp", degeneracy, ev, es_cx, esT_cx, P_op, PH_op);
		//print_overlap(sp_op, "sp", degeneracy, ev, es_cx, esT_cx);
		sp_st.clear();
		obs_data_cx.push_back(get_imaginary_time_obs(sp_op, Ntau, t_step, degeneracy,
			ev, es_cx, esT_cx));
		sp_op.clear();
		print_data(out, obs_data_cx[10]);

		sparse_storage<std::complex<double>, int_t> tp_st(hspace.sub_dimension());
		hspace.build_operator([&]
			(const std::pair<int_t, int_t>& n)
			{
				for (int i = 0; i < lat.n_sites(); i+=2)
				{
					//tp
					for (int j = 0; j < lat.n_sites(); j+=2)
					{
						auto& K = lat.symmetry_point("K");
						auto& Kp = lat.symmetry_point("Kp");
						//std::complex<double> phase = std::exp(std::complex<double>(0.,
						//	K.dot(lat.real_space_coord(i) - lat.real_space_coord(j))));
						std::complex<double> phase = std::exp(std::complex<double>(0.,
							K.dot(lat.real_space_coord(i)) + Kp.dot(lat.real_space_coord(j))));
						state p = hspace.c_dag_i({1, n.first}, j+1);
						p = hspace.c_dag_i(p, i);
						if (p.sign != 0)
							tp_st(hspace.index(p.id), n.second) += phase
								* std::complex<double>(p.sign) / static_cast<double>(Lx * Ly);
					}
				}
			});
		arma::sp_cx_mat tp_op = tp_st.build_matrix();
		print_overlap(tp_op, "tp", degeneracy, ev, es_cx, esT_cx, P_op, PH_op);
		//print_overlap(tp_op, "tp", degeneracy, ev, es_cx, esT_cx);
		tp_st.clear();
		obs_data_cx.push_back(get_imaginary_time_obs(tp_op, Ntau, t_step, degeneracy,
			ev, es_cx, esT_cx));
		tp_op.clear();
		print_data(out, obs_data_cx[11]);
	}
	
	std::cout << "Done" << std::endl;
	std::cout << "<E> = " << E << std::endl;
	std::cout << "<H_t> = " << h_t << std::endl;
	std::cout << "<H_V> = " << h_v << std::endl;
	std::cout << "<H_mu> = " << h_mu << std::endl;
	std::cout << "<n> = " << n_total / static_cast<double>(lat.n_sites()) << std::endl;
	std::cout << "<m2> = " << std::real(cdw2) << std::endl;
	std::cout << "<0|m2|0> = " << std::real(cdw2_0) << std::endl;
	std::cout << "<1|m2|1> = " << std::real(cdw2_1) << std::endl;
	std::cout << "<m4> = " << std::real(cdw4) << std::endl;
	std::cout << "S_cdw_q = " << std::real(S_cdw) << std::endl;
	std::cout << "B_cdw = " << std::real(cdw4/(cdw2*cdw2)) << std::endl;
	std::cout << "R_cdw = " << std::real(1. - S_cdw/cdw2) << std::endl;
	std::cout << "<epsilon> = " << std::real(ep) << std::endl;
	std::cout << "<0|epsilon|0> = " << std::real(ep_0) << std::endl;
	std::cout << "<1|epsilon|1> = " << std::real(ep_1) << std::endl;
	std::cout << "<0|kekule^2|0> = " << std::real(kek_0) << std::endl;
	std::cout << "<1|kekule^2|1> = " << std::real(kek_1) << std::endl;
	std::cout << "<chern> = " << std::imag(chern) << std::endl;
	std::cout << "<chern^2> = " << std::real(chern2) << std::endl;
	std::cout << "<chern^4> = " << std::real(chern4) << std::endl;
	std::cout << "S_chern_q = " << std::real(S_chern) << std::endl;
	std::cout << "R_chern = " << std::real(1. - S_chern/chern2) << std::endl;
	std::cout << "B_chern = " << std::real(chern4/(chern2*chern2)) << std::endl;
	std::cout << "S_chernAA:" << std::endl;
	std::cout << "|0> :" << std::endl;
	for (int i = 0; i < S_chernAA.size(); i+=2)
		std::cout << "k = " << i << " : " << S_chernAA[i] << std::endl;
	std::cout << "|1> :" << std::endl;
	for (int i = 1; i < S_chernAA.size(); i+=2)
		std::cout << "k = " << i << " : " << S_chernAA[i] << std::endl;

	out.close();
}
