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
	Random& rng;
	parser& pars;
	std::vector<double> correlations;
	
	measure_worm(configuration& config_, Random& rng_, parser& pars_)
		: config(config_), rng(rng_), pars(pars_)
	{	
		correlations.resize(config.l.max_distance() + 1, 0);
		
		//Set up measurements
		config.measure.add_observable("sign", config.param.n_prebin);
		config.measure.add_observable("N", config.param.n_prebin);
		config.measure.add_observable("<k>_Z", config.param.n_prebin);
		config.measure.add_observable("<k>_W2", config.param.n_prebin);
		config.measure.add_observable("<k>_W4", config.param.n_prebin);
		config.measure.add_observable("deltaZ", config.param.n_prebin);
		config.measure.add_observable("deltaW2", config.param.n_prebin);
		config.measure.add_observable("deltaW4", config.param.n_prebin);
		config.measure.add_vectorobservable("corr", config.l.max_distance() + 1,
			config.param.n_prebin);
		
		//Measure acceptance probabilities
		if (config.param.add[0] > 0.)
			config.measure.add_observable("insertion n=1", config.param.n_prebin * config.param.n_static_cycles);
		if (config.param.rem[0] > 0.)
			config.measure.add_observable("removal n=1", config.param.n_prebin * config.param.n_static_cycles);
		if (config.param.add[1] > 0.)
			config.measure.add_observable("insertion n=2", config.param.n_prebin * config.param.n_static_cycles);
		if (config.param.rem[1] > 0.)
			config.measure.add_observable("removal n=2", config.param.n_prebin * config.param.n_static_cycles);
		if (config.param.ZtoW2 > 0.)
			config.measure.add_observable("Z -> W2", config.param.n_prebin * config.param.n_static_cycles);
		if (config.param.W2toZ > 0.)
			config.measure.add_observable("W2 -> Z", config.param.n_prebin * config.param.n_static_cycles);
		if (config.param.W2toW4 > 0.)
			config.measure.add_observable("W2 -> W4", config.param.n_prebin * config.param.n_static_cycles);
		if (config.param.W4toW2 > 0.)
			config.measure.add_observable("W4 -> W2", config.param.n_prebin * config.param.n_static_cycles);
		if (config.param.ZtoW4 > 0.)
			config.measure.add_observable("Z -> W4", config.param.n_prebin * config.param.n_static_cycles);
		if (config.param.W4toZ > 0.)
			config.measure.add_observable("W4 -> Z", config.param.n_prebin * config.param.n_static_cycles);
		if (config.param.worm_shift > 0.)
			config.measure.add_observable("worm shift", config.param.n_prebin * config.param.n_static_cycles);
	}

	void perform()
	{
		config.measure.add("sign", config.sign);
		std::fill(correlations.begin(), correlations.end(), 0.0);
		if (config.worms() == 0) //measure Z
		{
			config.measure.add("<k>_Z", config.perturbation_order());
			config.measure.add("deltaZ", 1.0);
			config.measure.add("deltaW2", 0.0);
			config.measure.add("deltaW4", 0.0);
			
			double N = measure_N();
			config.measure.add("N", N);
		}
		else if (config.worms() == 1) //measure W2
		{
			config.measure.add("<k>_W2", config.perturbation_order());
			config.measure.add("deltaZ", 0.0);
			config.measure.add("deltaW2", 1.0);
			config.measure.add("deltaW4", 0.0);
			int sites[] = {config.M.vertex(0, worm).site,
				config.M.vertex(1, worm).site};
			int R = config.l.distance(sites[0], sites[1]);
			correlations[R] = config.l.parity(sites[0])
				* config.l.parity(sites[1])
				/ static_cast<double>(config.shellsize[R]);
		}
		else if (config.worms() == 2) //measure W4
		{
			config.measure.add("<k>_W4", config.perturbation_order());
			config.measure.add("deltaZ", 0.0);
			config.measure.add("deltaW2", 0.0);
			config.measure.add("deltaW4", 1.0);
		}
		config.measure.add("corr", correlations);
	}
	
	double measure_N()
	{
		int kmax = 1;
		double N = 0;
		for (int k = 0; k < kmax; ++k)
		{
			double tau_0 = rng() * config.param.beta;
			for (int i = 0; i < config.l.n_sites(); ++i)
			{
				std::vector<arg_t> rows = {arg_t{tau_0, i, 0}};
				std::vector<arg_t> cols = {arg_t{tau_0, i, 0}};
				N += config.sign * config.M.get_obs<1>(rows, cols, 0)
					/ config.l.n_sites() / kmax;
			}
		}
		return N;
	}

	void collect(std::ostream& os)
	{
		double eval_param[] = {config.param.zeta2, config.param.zeta4,
			static_cast<double>(config.l.n_sites()), config.param.zeta2};
		config.measure.add_evalable("M2", "deltaZ", "deltaW2", eval_M2, eval_param);
		config.measure.add_evalable("M4", "deltaZ", "deltaW4", eval_M4, eval_param);
		config.measure.add_evalable("BinderRatio", "deltaZ", "deltaW2", "deltaW4",
			eval_B, eval_param);
		config.measure.add_vectorevalable("Correlations", "corr", "deltaZ", eval_corr,
			eval_param);
		
		os << "PARAMETERS" << std::endl;
		pars.get_all(os);
		config.measure.get_statistics(os);
	}
};

struct measure_dynamics
{
	configuration& config;
	Random& rng;
	parser& pars;
	std::vector<double> dyn_M2;
	std::vector<double> dyn_M2_tau;
	std::vector<double> mgf_omega;

	measure_dynamics(configuration& config_, Random& rng_, parser& pars_)
		: config(config_), rng(rng_), pars(pars_)
	{
		dyn_M2.resize(config.param.n_matsubara, 0.0);
		dyn_M2_tau.resize(config.param.n_discrete_tau + 1, 0.0);
		mgf_omega.resize(config.param.n_matsubara, 0.0);
		
		//Set up measurements
		config.measure.add_observable("sign", config.param.n_prebin);
		config.measure.add_observable("<k>_Z", config.param.n_prebin);
		if (config.param.n_matsubara > 0)
			config.measure.add_vectorobservable("dyn_M2_mat",
				config.param.n_matsubara, config.param.n_prebin);
		if (config.param.n_discrete_tau > 0)
		{
			config.measure.add_vectorobservable("dyn_M2_tau",
				config.param.n_discrete_tau, config.param.n_prebin);
			config.measure.add_vectorobservable("dyn_sp_tau",
				config.param.n_discrete_tau, config.param.n_prebin);
			config.measure.add_vectorobservable("dyn_tp_tau",
				config.param.n_discrete_tau, config.param.n_prebin);
		}
		//Measure acceptance probabilities
		if (config.param.add[0] > 0.)
			config.measure.add_observable("insertion n=1", config.param.n_prebin * config.param.n_static_cycles);
		if (config.param.rem[0] > 0.)
			config.measure.add_observable("removal n=1", config.param.n_prebin * config.param.n_static_cycles);
		if (config.param.add[1] > 0.)
			config.measure.add_observable("insertion n=2", config.param.n_prebin * config.param.n_static_cycles);
		if (config.param.rem[1] > 0.)
			config.measure.add_observable("removal n=2", config.param.n_prebin * config.param.n_static_cycles);
	}
	
	void measure_dynamical_M2_mat()
	{
		std::fill(dyn_M2.begin(), dyn_M2.end(), 0.);
		int i = rng() * config.l.n_sites();
		for (int j = 0; j < config.l.n_sites(); ++j)
		{
			std::vector<arg_t> vec = {arg_t{0., i, 0}, arg_t{0., j, 0}};
			config.M.matsubara_gf<2>(config.param.beta, vec, mgf_omega);
			for (int n = 0; n < config.param.n_matsubara; ++n)
				dyn_M2[n] += config.l.parity(i) * config.l.parity(j) * mgf_omega[n]
					/ config.l.n_sites();
		}
		config.measure.add("dyn_M2_mat", dyn_M2);
	}
	
	void measure_dynamical_M2_tau()
	{
		std::fill(dyn_M2_tau.begin(), dyn_M2_tau.end(), 0.);
		for (int t = 0; t <= config.param.n_discrete_tau; ++t)
		{
			double tau = config.param.beta * static_cast<double>(t)
				/ static_cast<double>(config.param.n_discrete_tau);
			int kmax = 1;
			for (int k = 0; k < kmax; ++k)
			{
				double tau_0 = rng() * config.param.beta;
				double end = tau + tau_0;
				if (end > config.param.beta)
					end -= config.param.beta;
				int i = rng() * config.l.n_sites();
				for (int j = 0; j < config.l.n_sites(); ++j)
				for (int i = 0; i < config.l.n_sites(); ++i)
				{
					std::vector<arg_t> vec = {arg_t{end, i, 0}, arg_t{tau_0, j, 0}};
					dyn_M2_tau[t] += config.l.parity(i) * config.l.parity(j)
						* config.M.try_add<2>(vec, 0)
						/ std::pow(config.l.n_sites(), 2.)
						/ kmax;
				}
			}
		}
		config.measure.add("dyn_M2_tau", dyn_M2_tau);
	}
	
	void measure_dynamical_sp_tau()
	{
		std::fill(dyn_M2_tau.begin(), dyn_M2_tau.end(), 0.);
		double pi = 4.*std::atan(1.);
		//Eigen::Vector2d K(2.*pi/9., 2.*pi/9.*(2.-1./std::sqrt(3.)));
		auto& K = config.l.symmetry_point("K");
		for (int t = 0; t <= config.param.n_discrete_tau; ++t)
		{
			double tau = config.param.beta * static_cast<double>(t)
				/ static_cast<double>(config.param.n_discrete_tau);
			int kmax = 1;
			for (int k = 0; k < kmax; ++k)
			{
				double tau_0 = rng() * config.param.beta;
				double end = tau + tau_0;
				if (end > config.param.beta)
					end -= config.param.beta;
				for (int i = 0; i < config.l.n_sites(); ++i)
				for (int j = 0; j < config.l.n_sites(); ++j)
				{
					auto& r_i = config.l.real_space_coord(i);
					auto& r_j = config.l.real_space_coord(j);
					double kdot = K.dot(r_j - r_i);

					std::vector<arg_t> rows = {arg_t{end, i, 0}};
					std::vector<arg_t> cols = {arg_t{tau_0, j, 0}};
					dyn_M2_tau[t] += std::cos(K.dot(r_j - r_i)) * config.M.get_obs<1>(rows, cols, 0)
						/ std::pow(config.l.n_sites(), 2.) / kmax;
				}
			}
		}
		config.measure.add("dyn_sp_tau", dyn_M2_tau);
	}
	
	void measure_dynamical_tp_tau()
	{
		std::fill(dyn_M2_tau.begin(), dyn_M2_tau.end(), 0.);
		double pi = 4.*std::atan(1.);
		Eigen::Vector2d K(2.*pi/9., 2.*pi/9.*(2.-1./std::sqrt(3.)));
		int i = rng() * config.l.n_sites();
		for (int j = 0; j < config.l.n_sites(); ++j)
			for (int m = 0; m < config.l.n_sites(); ++m)
				for (int n = 0; n < config.l.n_sites(); ++n)
					for (int t = 0; t <= config.param.n_discrete_tau; ++t)
					{
						auto& r_i = config.l.real_space_coord(i);
						auto& r_j = config.l.real_space_coord(j);
						auto& r_m = config.l.real_space_coord(m);
						auto& r_n = config.l.real_space_coord(n);
						double tau = config.param.beta * static_cast<double>(t)
							/ static_cast<double>(config.param.n_discrete_tau);
						double tau_0 = rng() * (config.param.beta - tau);
						std::vector<arg_t> rows_i = {arg_t{tau + tau_0, i, 0}};
						std::vector<arg_t> rows_j = {arg_t{tau + tau_0, j, 0}};
						std::vector<arg_t> cols_m = {arg_t{tau_0, m, 0}};
						std::vector<arg_t> cols_n = {arg_t{tau_0, n, 0}};
//						dyn_M2_tau[t] = std::cos(K.dot(r_j - r_i))*config.l.n_sites()
//							* (config.M.get_obs<1>(rows_i, cols_n, 0)
//							* config.M.get_obs<1>(rows_j, cols_m, 0)
//							- config.M.get_obs<1>(rows_i, cols_m, 0)
//							* config.M.get_obs<1>(rows_j, cols_n, 0));
						
						std::vector<arg_t> rows = {arg_t{tau + tau_0, i, 0},
							arg_t{tau + tau_0, j, 0}};
						std::vector<arg_t> cols = {arg_t{tau_0, n, 0},
							arg_t{tau_0, m, 0}};
						dyn_M2_tau[t] += std::cos(K.dot(r_j - r_i + r_m - r_n))
							* config.l.n_sites()
							* config.M.get_obs<2>(rows, cols, 0);
					}
		config.measure.add("dyn_tp_tau", dyn_M2_tau);
	}

	void perform()
	{
		config.measure.add("sign", config.sign);
		config.measure.add("<k>_Z", config.perturbation_order());
//		measure_dynamical_M2_mat();
		measure_dynamical_M2_tau();
		measure_dynamical_sp_tau();
//		measure_dynamical_tp_tau();
	}

	void collect(std::ostream& os)
	{
		os << "PARAMETERS" << std::endl;
		pars.get_all(os);
		config.measure.get_statistics(os);
	}
};
