#pragma once
#include <vector>
#include <Eigen/Dense>
#include "lattice.h"

struct helper_matrices
{
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
		Eigen::ColMajor> dmatrix_t;

	dmatrix_t u;
	dmatrix_t v;
	dmatrix_t Mu;
	Eigen::Matrix<double, 2, 2, Eigen::ColMajor> a;
	Eigen::Matrix<double, 2, 2, Eigen::ColMajor> invS;
};

template<typename function_t, typename arg_t>
class fast_update
{
	public:
		typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
			Eigen::ColMajor> dmatrix_t;

		fast_update(const function_t& function_, const lattice& l_)
			: function(function_), l(l_)
		{
			buffer.resize(2);
		}

		int n_vertices()
		{
			return vertices.size();
		}

		double try_add(const arg_t& x, const arg_t& y)
		{
			int k = vertices.size();
			const int n = 2;
			
			helper.u.resize(k, n);
			helper.v.resize(n, k);
			buffer[0] = x;
			buffer[1] = y;
			fill_helper_matrices();
			helper.Mu = M * helper.u;
			helper.invS = helper.a - helper.v * helper.Mu;

			return helper.invS.determinant();
		}

		void finish_add()
		{
			int k = vertices.size(); const int n = 2;
			vertices.insert(vertices.begin(), buffer.begin(), buffer.begin() + 2);

			helper.invS = helper.invS.inverse().eval();
			dmatrix_t vM = M.transpose() * helper.v.transpose();
			vM.transposeInPlace();
			M.conservativeResize(k + n, k + n);
			M.block(k, 0, n, k) = -helper.invS * vM;
			M.topLeftCorner(k, k) -= helper.Mu * M.block(k, 0, n, k);
			M.block(0, k, k, n) = -helper.Mu * helper.invS;
					
			/*
			dmatrix_t vinvG = v * invG.topLeftCorner(k, k);
			invG.block(k, 0, n, k) = -S * vinvG;
			invG.topLeftCorner(k, k) -= invGu * invG.block(k, 0, n, k);
			invG.block(0, k, k, n) = -invGu * S;
			*/
			M.template block<n, n>(k, k) = helper.invS;
		}
	private:
		void fill_helper_matrices()
		{
			for (int i = 0; i < helper.u.rows(); ++i)
			{
				helper.u(i, 0) = function(vertices[i], buffer[0]);
				helper.v(0, i) = helper.u(i, 0) * ((l.sublattice(vertices[i].site)
					== l.sublattice(buffer[0].site)) ? -1.0 : 1.0);
				helper.u(i, 1) = function(vertices[i], buffer[1]);
				helper.v(1, i) = helper.u(i, 1) * ((l.sublattice(vertices[i].site)
					== l.sublattice(buffer[1].site)) ? -1.0 : 1.0);
			}
			helper.a(0, 0) = 0.0; helper.a(1, 1) = 0.0;
			helper.a(0, 1) = function(buffer[0], buffer[1]);
			helper.a(1, 0) = helper.a(0, 1) * ((l.sublattice(buffer[0].site) ==
				l.sublattice(buffer[1].site)) ? -1.0 : 1.0);
		}
	private:
		function_t function;
		const lattice& l;
		std::vector<arg_t> vertices;
		std::vector<arg_t> buffer;
		dmatrix_t M;
		helper_matrices helper;
};
