#pragma once
#include <map>
#include <armadillo>

template<typename int_t>
struct sparse_storage
{
	std::map<std::pair<int_t, int_t>, double> data;
	int dimension;
	
	sparse_storage(int dimension_)
		: dimension(dimension_)
	{}

	double& operator()(int_t i, int_t j)
	{
		auto pos = std::make_pair(i, j);
		if (data.count(pos) == 0)
			data[pos] = 0.;
		return data[pos];
	}

	arma::sp_mat build_matrix()
	{
		for (int i = 0; i < dimension; ++i)
			operator()(i, i) += 0.;
		arma::umat pos(2, data.size());
		arma::vec values(data.size());
		int_t cnt = 0;
		for (auto& x : data)
		{
			pos(0, cnt) = x.first.first;
			pos(1, cnt) = x.first.second;
			values(cnt) = x.second;
			++cnt;
		}
		return arma::sp_mat(pos, values);
	}
};
