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
#include <Eigen/Dense>
#include "lattice.h"
#include "honeycomb.h"

namespace po = boost::program_options;

namespace mp = boost::multiprecision;
typedef mp::number<mp::mpfr_float_backend<300> >  mp_float;

using complex_t = std::complex<double>;
template<int n, int m>
using matrix_t = Eigen::Matrix<complex_t, n, m, Eigen::ColMajor>; 
using dmatrix_t = matrix_t<Eigen::Dynamic, Eigen::Dynamic>;

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

int get_bond_type(std::vector<std::map<int, int>>& cb_bonds, const std::pair<int, int>& bond)
{
	for (int i = 0; i < cb_bonds.size(); ++i)
		if (cb_bonds[i].at(bond.first) == bond.second)
			return i;
}

std::vector<std::map<int, int>> create_checkerboard(lattice& l)
{
	std::vector<std::map<int, int>> cb_bonds(3);
	for (int i = 0; i < l.n_sites(); ++i)
	{
		auto& nn = l.neighbors(i, "nearest neighbors");
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
	return cb_bonds;
}

template <class data_t, class index_t>
class SortIndicesInc
{
	protected:
		const data_t& data;
	public:
		SortIndicesInc(const data_t& data_) : data(data_) {}
		bool operator()(const index_t& i, const index_t& j) const
		{
			return data[i] < data[j];
		}
};
	
int main(int ac, char** av)
{
	int L;
	double T;
	int k;
	std::string ensemble;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("L", po::value<int>(&L)->default_value(3), "linear lattice dimension")
		("k", po::value<int>(&k)->default_value(100), "number of eigenstates")
		("ensemble,e", po::value<std::string>(&ensemble)->default_value("gc"),
			"ensemble: gc or c");
	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);

	if (vm.count("help")) print_help(desc);
	std::cout << "L = " << L << std::endl;
	std::cout << "ensemble: " << ensemble << std::endl;
	//Generate lattice
	honeycomb h(L);
	lattice l;
	l.generate_graph(h);
	h.generate_maps(l);
	auto cb_bonds = create_checkerboard(l);
	
	dmatrix_t broken_H0 = dmatrix_t::Zero(l.n_sites(), l.n_sites());
	for (auto& a : l.bonds("nearest neighbors"))
	{
		if (static_cast<int>(std::sqrt(l.n_sites()/2)) % 3 == 0
			&& get_bond_type(cb_bonds, a) == 0)
			broken_H0(a.first, a.second) = 1.001;
		else
			broken_H0(a.first, a.second) = 1.;
	}
	Eigen::SelfAdjointEigenSolver<dmatrix_t> solver(broken_H0);
	std::vector<int> indices(l.n_sites());
	for (int i = 0; i < l.n_sites(); ++i)
		indices[i] = i;
	SortIndicesInc<Eigen::VectorXd, int> inc(solver.eigenvalues());
	std::sort(indices.begin(), indices.end(), inc);
	dmatrix_t P = dmatrix_t::Zero(l.n_sites(), l.n_sites() / 2);
	for (int i = 0; i < l.n_sites() / 2; ++i)
		P.col(i) = solver.eigenvectors().col(indices[i]);
	dmatrix_t Pt = P.adjoint();
	dmatrix_t V = solver.eigenvectors();
	dmatrix_t Vt = V.adjoint();
	
	dmatrix_t kek_op = dmatrix_t::Zero(l.n_sites(), l.n_sites());
	for (auto& a : l.bonds("kekule"))
		kek_op(a.first, a.second) += 1.;
	for (auto& a : l.bonds("kekule_2"))
		kek_op(a.first, a.second) += -1.;

	std::string out_file = "../data/ed_L_" + std::to_string(L) + "__"
		+ "V_0__" + "GS__" + ensemble;
	std::ofstream out("../data/" + out_file);
	std::cout.precision(8);
	out.precision(18);
	
	complex_t E = 0.;
	for (int i = 0; i < l.n_sites() / 2; ++i)
		E += solver.eigenvalues()[i];
	std::cout << "<GS|E|GS>: " << std::real(E) << std::endl;
	std::cout << "<GS|kekule|GS>: " << std::real((Pt * kek_op * P).trace()) << std::endl;
	std::cout << Pt * kek_op * P << std::endl;
	
	out.close();
}