#pragma once
#include <vector>
#include <array>
#include <Eigen/Dense>
#include "lattice.h"

struct helper_matrices
{
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
		Eigen::RowMajor> dmatrix_t;

	dmatrix_t u;
	dmatrix_t v;
	dmatrix_t Mu;
	dmatrix_t a;
	dmatrix_t S;
	dmatrix_t m;
};

template<typename function_t, typename arg_t>
class fast_update
{
	public:
		typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
			Eigen::ColMajor> dmatrix_t;

		fast_update(const function_t& function_, const lattice& l_,
			int n_flavors_)
			: function(function_), l(l_), flavor_cnt(n_flavors_)
		{}

		int perturbation_order(int flavor) const
		{
			return flavor_cnt[flavor] / 2;
		}

		const arg_t& vertex(int index, int flavor)
		{
			int offset = 0;
			for (int f = 0; f < flavor; ++f)
				offset += flavor_cnt[f];
			return vertices[offset + index]; 
		}

		void print_vertices() const
		{
			std::cout << "print vertices:" << std::endl;
			int offset = 0;
			for (int f = 0; f < flavor_cnt.size(); f++)
			{
				std::cout << "flavor " << f << std::endl;
				for (int i = 0; i < flavor_cnt[f]/2; ++i)
				{
					std::cout << 2*i << ": " << vertices[offset+2*i].tau << ", "
						<< vertices[offset+2*i].site << " , w: "
						<< vertices[offset+2*i].worm << " ; "
						<< vertices[offset+2*i+1].tau
						<< ", " << vertices[offset+2*i+1].site << " , w: "
						<< vertices[offset+2*i+1].worm << std::endl;
				}
				offset += flavor_cnt[f];
			}
		}

		void rebuild()
		{
			if (M.rows() == 0) return;
			for (int i = 0; i < M.rows(); ++i)
			{
				M(i, i) = 0.0;
				for (int j = i+1; j < M.cols(); ++j)
				{
					M(i, j) = function(vertices[i], vertices[j]);
					M(j, i) = -M(i, j) * l.parity(vertices[i].site)
						* l.parity(vertices[j].site);
				}
			}
			M = M.inverse().eval();
		}

		template<int N>
		double try_add(std::vector<arg_t>& args, int flavor=0)
		{
			int k = M.rows();
			const int n = 2*N;
			last_flavor = flavor;
			
			arg_buffer.swap(args);
			helper.u.resize(k, n);
			helper.v.resize(n, k);
			helper.a.resize(n, n);
			fill_helper_matrices();
			helper.Mu.noalias() = M * helper.u;
			helper.S = helper.a;
			helper.S.noalias() -= helper.v * helper.Mu;

			return helper.S.determinant();
		}

		void finish_add()
		{
			int k = M.rows();
			int n = arg_buffer.size();

			helper.S = helper.S.inverse().eval();
			dmatrix_t vM = M.transpose() * helper.v.transpose();
			vM.transposeInPlace();
			M.conservativeResize(k + n, k + n);

			M.block(k, 0, n, k).noalias() = -helper.S * vM;
			M.topLeftCorner(k, k).noalias() -= helper.Mu * M.block(k, 0, n, k);
			M.block(0, k, k, n).noalias() = -helper.Mu * helper.S;
			M.block(k, k, n, n) = helper.S;
			
			vertices.insert(vertices.end(), arg_buffer.begin(), arg_buffer.end());
			pos_buffer.resize(n/2);
			for (int i = 0; i < n/2; ++i)
				pos_buffer[i] = vertices.size() - n + 2*i;
			if(last_flavor == 0)
				permute_backward();
			flavor_cnt[last_flavor] += n;
		}

		template<int N>
		double try_remove(std::vector<int>& pos, int flavor=0)
		{
			if (flavor_cnt[flavor] < 2*N)
				return 0.0;
			last_flavor = flavor;
			pos_buffer.swap(pos);
			int pos_offset = 0;
			for (int f = 0; f < flavor; ++f)
				pos_offset += flavor_cnt[f];
			for (int& p : pos_buffer)
				p = 2*p + pos_offset;
			fill_S_matrix();
			return helper.S.determinant();
		}

		void finish_remove()
		{
			permute_forward();
			int k = M.rows();
			int n = 2*pos_buffer.size();
				
			helper.S.transposeInPlace();
			dmatrix_t t = M.block(k - n, 0, n, k - n).transpose()
				* helper.S.inverse();
			t.transposeInPlace();
			M.topLeftCorner(k - n, k - n).noalias()
				-= M.block(0, k - n, k - n, n) * t;
			M.conservativeResize(k - n, k - n);

			for (int i = 0; i < n; ++i)
				vertices.erase(vertices.end() - 1);
			flavor_cnt[last_flavor] -= n;
		}
	
		double try_shift(std::vector<arg_t>& args)
		{
			int k = M.rows() - args.size();
			int n = args.size();
			last_flavor = 1; //shift worm vertices
			
			arg_buffer.swap(args);
			helper.u.resize(k, n);
			helper.v.resize(n, k);
			helper.a.resize(n, n);
			helper.m.resize(k, k);
			fill_helper_matrices();
			
			helper.m = M.topLeftCorner(k, k);
			helper.m.noalias() -= M.block(0, k, k, n)
				* M.bottomRightCorner(n, n).inverse() * M.block(k, 0, n, k);
			helper.S = helper.a;
			helper.S.noalias() -= helper.v * helper.m * helper.u;
			return helper.S.determinant()
				* M.bottomRightCorner(n, n).determinant();
		}
		
		void finish_shift()
		{
			int k = M.rows() - arg_buffer.size();
			int n = arg_buffer.size();
			
			helper.S = helper.S.inverse().eval();
			dmatrix_t vM = helper.v * helper.m;
			helper.Mu.noalias() = helper.m * helper.u;
			M.block(k, 0, n, k).noalias() = -helper.S * vM;
			M.topLeftCorner(k, k) = helper.m;
			M.topLeftCorner(k, k).noalias() -= helper.Mu * M.block(k, 0, n, k);
			M.block(0, k, k, n).noalias() = -helper.Mu * helper.S;
			M.block(k, k, n, n) = helper.S;
			
			for (int i = 0; i < arg_buffer.size(); ++i)
				vertices[vertices.size() - arg_buffer.size() + i] = arg_buffer[i];
		}
	private:
		void fill_helper_matrices()
		{
			for (int i = 0; i < helper.u.rows(); ++i)
			{
				for (int j = 0; j < helper.u.cols(); ++j)
				{
					helper.u(i, j) = function(vertices[i], arg_buffer[j]);
					helper.v(j, i) = -helper.u(i, j) * l.parity(vertices[i].site)
						* l.parity(arg_buffer[j].site);
				}
			}
			for (int i = 0; i < helper.a.rows(); ++i)
			{
				helper.a(i, i) = 0.0;
				for (int j = i+1; j < helper.a.cols(); ++j)
				{
					helper.a(i, j) = function(arg_buffer[i], arg_buffer[j]);
					helper.a(j, i) = -helper.a(i, j) * l.parity(arg_buffer[i].site)
						* l.parity(arg_buffer[j].site);
				}
			}
		}

		void fill_S_matrix()
		{
			int n = pos_buffer.size();
			helper.S.resize(2*n, 2*n);
			for (int i = 0; i < n; ++i)
			{
				for (int j = 0; j < n; ++j)
				{
					helper.S.template block<2, 2>(2*i, 2*j) =
						M.template block<2, 2>(pos_buffer[i], pos_buffer[j]);
				}
			}
		}

		void swap_rows_cols(int i, int j)
		{
			int k = M.rows();
			dmatrix_t cols = M.block(0, i, k, 2);
			M.block(0, i, k, 2) = M.block(0, j, k, 2);
			M.block(0, j, k, 2) = cols;
			dmatrix_t rows = M.block(i, 0, 2, k);
			M.block(i, 0, 2, k) = M.block(j, 0, 2, k);
			M.block(j, 0, 2, k) = rows;
		}
		
		void permute_forward()
		{	
			int n = 2*pos_buffer.size();
			int block_end = 0;
			for (int f = 0; f < last_flavor; ++f)
				block_end += flavor_cnt[f];
			for (int f = last_flavor; f < flavor_cnt.size(); ++f)
			{
				block_end += flavor_cnt[f];
				for (int i = 0; i < n/2; ++i)
				{
					swap_rows_cols(pos_buffer[n/2 - i - 1], block_end - 2*i - 2);
					std::swap(vertices[pos_buffer[n/2 - i - 1]],
						vertices[block_end - 2*i - 2]);
					std::swap(vertices[pos_buffer[n/2 - i - 1] + 1],
						vertices[block_end - 2*i - 1]);
					pos_buffer[n/2 - i - 1] = block_end - 2*i - 2;
				}
			}
		}

		void permute_backward()
		{
			int n = 2*pos_buffer.size();
			int block_end = 0;
			for (int f = 0; f < flavor_cnt.size(); ++f)
				block_end += flavor_cnt[f];
			for (int f = flavor_cnt.size()-1; f > last_flavor; --f)
			{
				block_end -= flavor_cnt[f];
				for (int i = 0; i < n/2; ++i)
				{
					swap_rows_cols(pos_buffer[i], block_end + 2*i);
					std::swap(vertices[pos_buffer[i]], vertices[block_end + 2*i]);
					std::swap(vertices[pos_buffer[i] + 1],
						vertices[block_end + 2*i + 1]);
					pos_buffer[i] = block_end + 2*i;
				}
			}
		}
		
		void print_matrix(const dmatrix_t& m)
		{
			Eigen::IOFormat clean(4, 0, ", ", "\n", "[", "]");
			std::cout << m.format(clean) << std::endl;
		}
	private:
		function_t function;
		const lattice& l;
		std::vector<arg_t> vertices;
		std::vector<int> flavor_cnt;
		std::vector<arg_t> arg_buffer;
		std::vector<int> pos_buffer;
		int last_flavor;
		dmatrix_t M;
		helper_matrices helper;
};
