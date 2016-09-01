#pragma once
#include <fstream>
#include "measurements.h"
#include "configuration.h"

struct event_rebuild
{
	configuration& config;
	measurements& measure;

	void trigger()
	{
		config.M.rebuild();
	}
};

struct event_print_M
{
	configuration& config;
	measurements& measure;
	
	event_print_M(configuration& config_,
		measurements& measure_)
		: config(config_), measure(measure_)
	{
		std::ofstream file("M.txt", std::fstream::out);
		file.close();
	}

	void trigger()
	{
		std::ofstream file("M.txt", std::fstream::app);
		config.M.print_M_matrix(file);
		file.close();
	}
};

struct event_build
{
	configuration& config;
	Random& rng;

	void trigger()
	{
		int n0 = 0.13 * config.param.beta * config.param.V
			* config.l.n_sites();
		std::vector<arg_t> initial_vertices;
		for (int i = 0; i < n0; ++i)
		{
			double tau = rng() * config.param.beta;
			int s1 = rng() * config.l.n_sites();
			int s2 =  config.l.neighbors(s1, "nearest neighbors")
				[rng() * config.l.neighbors(s1, "nearest neighbors").size()];
			initial_vertices.push_back({tau, s1, nn_int});
			initial_vertices.push_back({tau, s2, nn_int});
		}
		config.M.build(initial_vertices, {2*n0, 0});
	}
};
