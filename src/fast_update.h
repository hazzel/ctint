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
	Eigen::Matrix<double, 2, 2, Eigen::ColMajor> S;
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
			arg_buffer.resize(2);
			pos_buffer.resize(1);
		}

		int perturbation_order() const
		{
			return vertices.size() / 2;
		}

		double try_add(const arg_t& x, const arg_t& y)
		{
			int k = vertices.size();
			const int n = 2;
			
			helper.u.resize(k, n);
			helper.v.resize(n, k);
			arg_buffer[0] = x;
			arg_buffer[1] = y;
			fill_helper_matrices();
			helper.Mu.noalias() = M * helper.u;
			helper.S.noalias() = helper.a - helper.v * helper.Mu;

			return helper.S.determinant();
		}

		void finish_add()
		{
			int k = vertices.size(); const int n = 2;
			vertices.insert(vertices.end(), arg_buffer.begin(),
				arg_buffer.begin() + 2);

			helper.S = helper.S.inverse().eval();
			dmatrix_t vM = M.transpose() * helper.v.transpose();
			vM.transposeInPlace();
			M.conservativeResize(k + n, k + n);

			M.block(k, 0, n, k).noalias() = -helper.S * vM;
			M.topLeftCorner(k, k).noalias() -= helper.Mu * M.block(k, 0, n, k);
			M.block(0, k, k, n).noalias() = -helper.Mu * helper.S;
					
			/*
			dmatrix_t vinvG = v * invG.topLeftCorner(k, k);
			invG.block(k, 0, n, k) = -S * vinvG;
			invG.topLeftCorner(k, k) -= invGu * invG.block(k, 0, n, k);
			invG.block(0, k, k, n) = -invGu * S;
			*/
			M.template block<n, n>(k, k) = helper.S;
		}

		double try_remove(int pos)
		{
			const int n = 2;
			if (vertices.size() < n)
				return 0.0;
			pos_buffer[0] = pos;
			helper.S = M.block(2*pos, 2*pos, n, n);
			return helper.S.determinant();
		}

		void finish_remove()
		{
			permute_M();
			int k = vertices.size(); const int n = 2;
				
			helper.S.transposeInPlace();
			dmatrix_t t = M.block(k - n, 0, n, k - n).transpose()
				* helper.S.inverse();
			t.transposeInPlace();
			M.topLeftCorner(k - n, k - n).noalias()
				-= M.block(0, k - n, k - n, n) * t;
			M.conservativeResize(k - n, k - n);

			for (int i = 0; i < n; ++i)
				vertices.erase(vertices.end() - 1);
		}
	private:
		void fill_helper_matrices()
		{
			for (int i = 0; i < helper.u.rows(); ++i)
			{
				helper.u(i, 0) = function(vertices[i], arg_buffer[0]);
				helper.v(0, i) = helper.u(i, 0) * ((l.sublattice(vertices[i].site)
					== l.sublattice(arg_buffer[0].site)) ? -1.0 : 1.0);
				helper.u(i, 1) = function(vertices[i], arg_buffer[1]);
				helper.v(1, i) = helper.u(i, 1) * ((l.sublattice(vertices[i].site)
					== l.sublattice(arg_buffer[1].site)) ? -1.0 : 1.0);
			}
			helper.a(0, 0) = 0.0; helper.a(1, 1) = 0.0;
			helper.a(0, 1) = function(arg_buffer[0], arg_buffer[1]);
			helper.a(1, 0) = helper.a(0, 1) * ((l.sublattice(arg_buffer[0].site)
				== l.sublattice(arg_buffer[1].site)) ? -1.0 : 1.0);
		}

		void swap_rows_cols(int i, int j)
		{
			int k = vertices.size();
			dmatrix_t cols = M.block(0, i, k, 2);
			M.block(0, i, k, 2) = M.block(0, j, k, 2);
			M.block(0, j, k, 2) = cols;
			dmatrix_t rows = M.block(i, 0, 2, k);
			M.block(i, 0, 2, k) = M.block(j, 0, 2, k);
			M.block(j, 0, 2, k) = rows;
		}
		
		void permute_M()
		{	
			int k = vertices.size();
			const int n = 2;
			for (int i = 0; i < n/2; ++i)
			{
				swap_rows_cols(2*pos_buffer[n/2 - i - 1], k - 2*i - 2);
				std::swap(vertices[2*pos_buffer[n/2 - i - 1]],
					vertices[k - 2*i - 2]);
				std::swap(vertices[2*pos_buffer[n/2 - i - 1] + 1],
					vertices[k - 2*i - 1]);
			}
		}
	private:
		function_t function;
		const lattice& l;
		std::vector<arg_t> vertices;
		std::vector<arg_t> arg_buffer;
		std::vector<int> pos_buffer;
		dmatrix_t M;
		helper_matrices helper;
};
