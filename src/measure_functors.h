#pragma once
#include <ostream>
#include <vector>
#include "measurements.h"
#include "parser.h"
#include "configuration.h"

void eval_M2(double& out, std::vector< std::valarray<double>* >& o, double* p)
{
	double z=(*o[0])[0];
	double w2=(*o[1])[0];
	
	out = (w2 / z) / p[0];
}

void eval_M4(double& out, std::vector< std::valarray<double>* >& o, double* p)
{
	double z=(*o[0])[0];
	double w4=(*o[1])[0];
	
	out = (w4 / z) / p[1];
}

void eval_B(double& out, std::vector< std::valarray<double>* >& o, double* p)
{
	double z=(*o[0])[0];
	double w2=(*o[1])[0];
	double w4=(*o[2])[0];
	
	out = (w4 * z) / (w2 * w2) * (p[0] * p[0] / p[1]);
}

void eval_corr(std::valarray<double>& out,
	std::vector<std::valarray<double>*>& o, double* p)
{
	std::valarray<double>* corr = o[0];
	double z=(*o[1])[0];
	out.resize(corr->size());
	for (int i = 0; i < corr->size(); ++i)
		out[i] = (*corr)[i] / z * p[2] / p[3];
}

struct measure_worm
{
	configuration& config;
	measurements& measure;
	parser& pars;
	std::vector<double> correlations;

	void perform()
	{
		std::fill(correlations.begin(), correlations.end(), 0.0);
		if (config.worms() == 0) //measure Z
		{
			measure.add("<k>_Z", config.perturbation_order());
			measure.add("deltaZ", 1.0);
			measure.add("deltaW2", 0.0);
			measure.add("deltaW4", 0.0);
		}
		else if (config.worms() == 1) //measure W2
		{
			measure.add("<k>_W2", config.perturbation_order());
			measure.add("deltaZ", 0.0);
			measure.add("deltaW2", 1.0);
			measure.add("deltaW4", 0.0);
			int sites[] = {config.M.vertex(0, worm).site,
				config.M.vertex(1, worm).site};
			int R = config.l.distance(sites[0], sites[1]);
			correlations[R] = config.l.parity(sites[0])
				* config.l.parity(sites[1])
				/ static_cast<double>(config.shellsize[R]);
		}
		else if (config.worms() == 2) //measure W4
		{
			measure.add("<k>_W4", config.perturbation_order());
			measure.add("deltaZ", 0.0);
			measure.add("deltaW2", 0.0);
			measure.add("deltaW4", 1.0);
		}
		measure.add("corr", correlations);
	}

	void collect(std::ostream& os)
	{
		double eval_param[] = {config.param.zeta2, config.param.zeta4,
			static_cast<double>(config.l.n_sites()), config.param.zeta2};
		measure.add_evalable("M2", "deltaZ", "deltaW2", eval_M2, eval_param);
		measure.add_evalable("M4", "deltaZ", "deltaW4", eval_M4, eval_param);
		measure.add_evalable("BinderRatio", "deltaZ", "deltaW2", "deltaW4",
			eval_B, eval_param);
		measure.add_vectorevalable("Correlations", "corr", "deltaZ", eval_corr,
			eval_param);
		
		os << "PARAMETERS" << std::endl;
		pars.get_all(os);
		measure.get_statistics(os);
	}
};

struct measure_estimator
{
	configuration& config;
	Random& rng;
	measurements& measure;
	parser& pars;
	std::vector<std::vector<double>> mgf_r;
	std::vector<double> mgf_omega;

	void perform()
	{
		measure.add("<k>_Z", config.perturbation_order());
		for (auto& g : mgf_r)
			std::fill(g.begin(), g.end(), 0.);
		int i = rng() * config.l.n_sites();
//		for (int i = 0; i < config.l.n_sites(); ++i)
		for (int j = 0; j < config.l.n_sites(); ++j)
		{
			std::vector<arg_t> vec = {arg_t{0., i, 0}, arg_t{0., j, 0}};
			config.M.matsubara_gf<1>(config.param.beta, vec, mgf_omega);
			for (int n = 0; n < config.param.n_matsubara; ++n)
				mgf_r[config.l.distance(i, j)][n] += config.l.parity(i)
					* config.l.parity(j) * mgf_omega[n]
					/ config.shellsize[config.l.distance(i, j)];
//					/ config.l.n_sites();
		}
		for (int r = 0; r < config.l.max_distance() + 1; ++r)
			measure.add("G(omega)_" + std::to_string(r), mgf_r[r]);
	}

	void collect(std::ostream& os)
	{
		os << "PARAMETERS" << std::endl;
		pars.get_all(os);
		measure.get_statistics(os);
	}
};
