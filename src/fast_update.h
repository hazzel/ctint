#pragma once
#include <vector>
#include <Eigen/Dense>

template<typename function_t, typename arg_t>
class fast_update
{
	public:
		typedef Eigen::matrix<Eigen::Dynamic, Eigen::Dynamic,
			Eigen::ColMajor> matrix_t;

		fast_update(const function_t& function_)
			: function(function_)
		{}

		int n_vertices()
		{
			return vertices.size();
		}

		double try_add(const arg_t& x, const arg_t& y)
		{
			
			return 0.;
		}
	private:
		function_t function;
		std::vector<arg_t> vertices;
		matrix_t M;
};
