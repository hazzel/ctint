#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include "boost/multi_array.hpp"
#include "lattice.h"
#include <triqs/arrays/array.hpp>
#include <triqs/arrays/matrix.hpp>
#include <triqs/arrays/linalg/eigenelements.hpp>

class greens_function
{
	public:
		greens_function() {}

		void generate_mesh(lattice* l_, double beta_, int n_slices_)
		{
			l = l_; beta = beta_; n_slices = n_slices_;
			dtau = beta / (2 * static_cast<double>(n_slices));
			
			triqs::arrays::matrix<double> K(l->n_sites(), l->n_sites());
			triqs::arrays::assign_foreach(K, [this](int i, int j)
				{ return (l->distance(i, j) == 1) ? -1.0 : 0.0; });
			auto v = triqs::arrays::linalg::eigenelements(K);
			
			generate_index_map(v.first, v.second);
			fill_mesh(v.first, v.second);
		}
		
		double operator()(double tau, int i, int j) const
		{
			double tau_p;
			if (std::abs(tau) > beta/2.0)
				tau_p = beta - std::abs(tau);
			else
				tau_p = std::abs(tau);
			int t = static_cast<int>(std::abs(tau_p) / dtau);
			int x = index_map[i][j];
			double tau_t = t * dtau;
			double G_t = mesh[x][t], G_tt = mesh[x][t + 1];
			double g = G_t + (tau_p - tau_t) * (G_tt - G_t) / dtau;
			double sign = 1.0;
			bool same_sl = l->sublattice(i) == l->sublattice(j);
			if (std::abs(tau) > beta/2.0 && (!same_sl))
				sign *= -1.0;
			if (tau < 0.0 && same_sl)
				sign *= -1.0;
			return sign * g;
		}
	private:
		triqs::arrays::matrix<double> bare_gf(double tau,
			const triqs::arrays::array<double, 1>& ev,
			const triqs::arrays::matrix<double>& V)
		{
			triqs::arrays::matrix<double> D(l->n_sites(), l->n_sites());
			triqs::arrays::assign_foreach(D, [&tau, &ev, this](int i, int j)
			{
				if (i != j)
					return std::exp(-tau * ev(i)) /
						(1.0 + std::exp(-beta * ev(i)));
				else
					return 0.0; });
			return V * D * V.transpose();
		}
		
		void generate_index_map(const triqs::arrays::array<double, 1>& ev,
			const triqs::arrays::matrix<double>& V)
		{
			index_map.resize(boost::extents[l->n_sites()][l->n_sites()]);
			double threshold = std::pow(10.0, -13.0);
			std::vector<double> values;
			
			auto g0 = bare_gf(0.1 * beta, ev, V);
			for (int i = 0; i < l->n_sites(); ++i)
			{
				for (int j = i; j < l->n_sites(); ++j)
				{
					bool is_stored = false;
					for (int k = 0; k < values.size(); ++k)
					{
						if (std::abs(values[k] - g0(i, j)) < threshold)
						{
							is_stored = true;
							index_map[i][j] = k;
							index_map[j][i] = k;
						}
					}
					if (!is_stored)
					{
						values.push_back(g0(i, j));
						index_map[i][j] = values.size() - 1;
						index_map[j][i] = values.size() - 1;
					}
				}
			}
			mesh.resize(boost::extents[values.size()][n_slices + 1]);
		}
		
		void fill_mesh(const triqs::arrays::array<double, 1>& ev,
			const triqs::arrays::matrix<double>& V)
		{
			for (int t = 0; t <= n_slices; ++t)
			{
				auto g0 = bare_gf(dtau * t, ev, V);
				for (int i = 0; i < l->n_sites(); ++i)
				{
					for (int j = i; j < l->n_sites(); ++j)
					{
						mesh[index_map[i][j]][t] = g0(i, j);
					}
				}
			}
		}
	private:
		lattice* l;
		double beta;
		double dtau;
		int n_slices;
		boost::multi_array<int, 2> index_map;
		boost::multi_array<double, 2> mesh;
};
