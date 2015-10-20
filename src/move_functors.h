#pragma once
#include "fast_update.h"
#include "lattice.h"
#include "greens_function.h"
#include "Random.h"

// Argument type
struct arg_t
{
	double tau;
	int site;
	bool worm;
};

enum { nn_int, worm };

// The function that appears in the calculation of the determinant
struct full_g_entry
{
	const greens_function& g0;

	double operator()(const arg_t& x, const arg_t& y) const
	{
		if ((x.tau == y.tau) && (x.site == y.site))
			return 0.0;
		else
			return g0(x.tau - y.tau, x.site, y.site);
	}
};

struct parameters
{	
	double beta, V, zeta2, zeta4;
	int worm_nhood_dist;
	double ratio_w2, ratio_w4;
};

// The Monte Carlo configuration
struct configuration
{
	const lattice& l;
	fast_update<full_g_entry, arg_t> M;
	parameters params;

	int perturbation_order() const { return M.perturbation_order(nn_int); }
	int worms() const { return M.perturbation_order(worm); }

	configuration(const lattice& l_, const greens_function& g0, 
		const parameters& params_)
		: l(l_), M{full_g_entry{g0}, l_, 2}, params(params_)
	{}
};

// ------------ QMC move : inserting a vertex ------------------

struct move_insert
{
	configuration* config;
	Random& rng;

	double attempt()
	{
		double tau = config->params.beta * rng();
		int s1 = config->l.n_sites() * rng();
		int s2 = config->l.neighbors(s1, 1)
			[config->l.neighbors(s1, 1).size() * rng()];
		int k = config->perturbation_order();
		std::vector<arg_t> vec = {arg_t{tau, s1, false}, arg_t{tau, s2, false}};
		double det_ratio = config->M.try_add<1>(vec, nn_int);
		assert(det_ratio == det_ratio && "nan value in det ratio");
		return -config->params.beta * config->params.V * config->l.n_bonds()
			/ (k + 1) * det_ratio;
	}

	double accept()
	{
		config->M.finish_add();
		return 1.0;
	}

	void reject() {}
};

// ------------ QMC move : deleting a vertex ------------------

struct move_remove
{
	configuration* config;
	Random& rng;

	double attempt()
	{
		int k = config->perturbation_order();
		if (k <= 0) return 0;
		int p = k * rng(); // Choose one of the operators for removal
		std::vector<int> vec = {p};
		double det_ratio = config->M.try_remove<1>(vec, nn_int);
		assert(det_ratio == det_ratio && "nan value in det ratio");
		return -k / (config->params.beta * config->params.V
			* config->l.n_bonds()) * det_ratio;
	}

	double accept()
	{
		config->M.finish_remove();
		return 1.0;
	}

	void reject() {}
};

// ------------ QMC move : Z -> W2 ------------------

struct move_ZtoW2
{
	configuration* config;
	Random& rng;
	bool save_acc;

	double attempt()
	{
		if (config->worms() != 0)
		{
			save_acc = false;
			return 0.0;
		}
		double tau = config->params.beta * rng();
		int s1 = config->l.n_sites() * rng();
		const std::vector<int>& neighbors =
			config->l.neighbors(s1, config->params.worm_nhood_dist);
		int s2 = neighbors[neighbors.size() * rng()];
		std::vector<arg_t> vec = {arg_t{tau, s1, true}, arg_t{tau, s2, true}};
		double det_ratio = config->M.try_add<1>(vec, worm);
		assert(det_ratio == det_ratio && "nan value in det ratio");
		save_acc = true;
		return config->l.parity(s1) * config->l.parity(s2)
			* config->params.zeta2 * config->params.ratio_w2 * det_ratio;
	}

	double accept()
	{
		config->M.finish_add();
		return 1.0;
	}

	void reject()
	{
	}
};

// ------------ QMC move : W2 -> Z ------------------

struct move_W2toZ
{
	configuration* config;
	Random& rng;
	bool save_acc;

	double attempt()
	{
		if (config->worms()	!= 1)
		{
			save_acc = false;
			return 0.0;
		}
		int p = 0; //only one worm exists	
		int sites[] = {config->M.vertex(p, worm).site,
							config->M.vertex(p+1, worm).site};
		const std::vector<int>& neighbors =
			config->l.neighbors(sites[0], config->params.worm_nhood_dist);
		if (std::find(neighbors.begin(), neighbors.end(), sites[1])
			== neighbors.end())
		{
			save_acc = false;
			return 0.0;
		}
		std::vector<int> vec = {p};
		double det_ratio = config->M.try_remove<1>(vec, worm);
		assert(det_ratio == det_ratio && "nan value in det ratio");
		save_acc = true;
		return config->l.parity(sites[0]) * config->l.parity(sites[1])
			/ config->params.zeta2 / config->params.ratio_w2 * det_ratio;
	}

	double accept()
	{
		config->M.finish_remove();
		return 1.0;
	}

	void reject()
	{
	}
};

// ------------ QMC move : W2 -> W4 ------------------

struct move_W2toW4
{
	configuration* config;
	Random& rng;
	bool save_acc;

	double attempt()
	{
		if (config->worms() != 1)
		{
			save_acc = false;
			return 0.0;
		}
		double tau = config->M.vertex(0, worm).tau;
		int s1 = config->l.n_sites() * rng();
		const std::vector<int>& neighbors =
			config->l.neighbors(s1, config->params.worm_nhood_dist);
		int s2 = neighbors[neighbors.size() * rng()];
		std::vector<arg_t> vec = {arg_t{tau, s1, true}, arg_t{tau, s2, true}};
		double det_ratio = config->M.try_add<1>(vec, worm);
		assert(det_ratio == det_ratio && "nan value in det ratio");
		save_acc = true;
		return config->l.parity(s1) * config->l.parity(s2)
			* config->params.zeta4 / config->params.zeta2
			* config->params.ratio_w2 * det_ratio;
	}

	double accept()
	{
		config->M.finish_add();
		return 1.0;
	}

	void reject()
	{
	}
};

// ------------ QMC move : W4 -> W2 ------------------

struct move_W4toW2
{
	configuration* config;
	Random& rng;
	bool save_acc;

	double attempt()
	{
		if (config->worms()	!= 2)
		{
			save_acc = false;
			return 0.0;
		}
		int p = config->worms() * rng(); //only one worm exists	
		int sites[] = {config->M.vertex(2*p, worm).site,
							config->M.vertex(2*p+1, worm).site};
		const std::vector<int>& neighbors =
			config->l.neighbors(sites[0], config->params.worm_nhood_dist);
		if (std::find(neighbors.begin(), neighbors.end(), sites[1])
			== neighbors.end())
		{
			save_acc = false;
			return 0.0;
		}
		std::vector<int> vec = {p};
		double det_ratio = config->M.try_remove<1>(vec, worm);
		assert(det_ratio == det_ratio && "nan value in det ratio");
		save_acc = true;
		return config->l.parity(sites[0]) * config->l.parity(sites[1])
			* config->params.zeta2 / config->params.zeta4
			/ config->params.ratio_w2 * det_ratio;
	}

	double accept()
	{
		config->M.finish_remove();
		return 1.0;
	}

	void reject()
	{
	}
};

// ------------ QMC move : Z -> W4 ------------------

struct move_ZtoW4
{
	configuration* config;
	Random& rng;
	bool save_acc;

	double attempt()
	{
		if (config->worms() != 0)
		{
			save_acc = false;
			return 0.0;
		}
		double tau = config->params.beta * rng();
		int s1 = config->l.n_sites() * rng();
		const std::vector<int>& neighbors =
			config->l.neighbors(s1, config->params.worm_nhood_dist);
		int s2 = neighbors[neighbors.size() * rng()];
		int s3 = neighbors[neighbors.size() * rng()];
		int s4 = neighbors[neighbors.size() * rng()];
		std::vector<arg_t> vec = {arg_t{tau, s1, true}, arg_t{tau, s2, true},
			arg_t{tau, s3, true}, arg_t{tau, s4, true}};
		double det_ratio = config->M.try_add<2>(vec, worm);
		assert(det_ratio == det_ratio && "nan value in det ratio");
		save_acc = true;
		return config->l.parity(s1) * config->l.parity(s2) * config->l.parity(s3)
			* config->l.parity(s4) * config->params.zeta4
			* config->params.ratio_w4 * det_ratio;
	}

	double accept()
	{
		config->M.finish_add();
		return 1.0;
	}

	void reject()
	{	
	}
};

// ------------ QMC move : W4 -> Z ------------------

struct move_W4toZ
{
	configuration* config;
	Random& rng;
	bool save_acc;

	double attempt()
	{
		if (config->worms() != 2)
		{
			save_acc = false;
			return 0.0;
		}
		int p = 0;
		int sites[] = {config->M.vertex(p, worm).site,
							config->M.vertex(p+1, worm).site,
							config->M.vertex(p+2, worm).site,
							config->M.vertex(p+3, worm).site};
		const std::vector<int>& neighbors =
			config->l.neighbors(sites[0], config->params.worm_nhood_dist);
		for (int i = 1; i < 4; ++i)	
			if (std::find(neighbors.begin(), neighbors.end(), sites[i])
				== neighbors.end())
			{
				save_acc = false;
				return 0.0;
			}
		std::vector<int> vec = {p, p+1};
		double det_ratio = config->M.try_remove<2>(vec, worm);
		assert(det_ratio == det_ratio && "nan value in det ratio");
		save_acc = true;
		return config->l.parity(config->M.vertex(p, worm).site)
			* config->l.parity(config->M.vertex(p+1, worm).site)
			* config->l.parity(config->M.vertex(p+2, worm).site)
			* config->l.parity(config->M.vertex(p+3, worm).site)
			/ config->params.zeta4 / config->params.ratio_w4 * det_ratio;
	}

	double accept()
	{
		config->M.finish_remove();
	}

	void reject()
	{
	}
};

// ------------ QMC move : worm shift ------------------

struct move_shift
{
	configuration* config;
	Random& rng;
	bool save_acc;

	double attempt()
	{
		if (config->worms() == 0)
		{
			save_acc = false;
			return 0.0;
		}
		std::vector<arg_t> worm_vert(2*config->worms());
		for (int i = 0; i < worm_vert.size(); ++i)
			worm_vert[i] = config->M.vertex(i, worm);
		double tau_shift = -0.05*config->params.beta + 0.1*config->params.beta
			* rng();
		if (worm_vert[0].tau + tau_shift > config->params.beta)
			tau_shift -= config->params.beta;
		else if(worm_vert[0].tau + tau_shift < 0.0)
			tau_shift += config->params.beta;
		for (int i = 0; i < worm_vert.size(); ++i)
			worm_vert[i].tau += tau_shift;
		int p = worm_vert.size() * rng();
		const std::vector<int>& neighbors = config->l.neighbors(
			worm_vert[p].site, config->params.worm_nhood_dist);
		int old_site = worm_vert[p].site;
		worm_vert[p].site = neighbors[neighbors.size() * rng()];
		double det_ratio = config->M.try_shift(worm_vert);
		assert(det_ratio == det_ratio && "nan value in det ratio");
		save_acc = true;
		//std::cout << "worm shift try:" << std::endl;
		//std::cout << "shift buffer: " << tau_shift << std::endl;
		//std::cout << worm_vert[0].tau << " , " << worm_vert[0].site << std::endl;
		//std::cout << worm_vert[1].tau << " , " << worm_vert[1].site << std::endl;
		//config->M.print_vertices();
		return config->l.parity(old_site) * config->l.parity(worm_vert[p].site)
			* det_ratio;
	}

	double accept()
	{
		config->M.finish_shift();
		//std::cout << "worm shift done:" << std::endl;
		//config->M.print_vertices();
		//std::cout << std::endl;
		//std::cin.get();
		return 1.0;
	}

	void reject()
	{}
};

