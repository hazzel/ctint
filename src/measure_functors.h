#pragma once
#include <ostream>
#include "measurements.h"
#include "move_functors.h"

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

struct measure_M
{
	configuration* config;
	measurements& measure;
	parser& pars;

	void perform()
	{
		if (config->worms() == 0) //measure Z
		{
			measure.add("<k>_Z", config->perturbation_order());
			measure.add("deltaZ", 1.0);
			measure.add("deltaW2", 0.0);
			measure.add("deltaW4", 0.0);
		}
		else if (config->worms() == 1) //measure W2
		{
			measure.add("<k>_W2", config->perturbation_order());
			measure.add("deltaZ", 0.0);
			measure.add("deltaW2", 1.0);
			measure.add("deltaW4", 0.0);		}
		else if (config->worms() == 2) //measure W4
		{
			measure.add("<k>_W4", config->perturbation_order());
			measure.add("deltaZ", 0.0);
			measure.add("deltaW2", 0.0);
			measure.add("deltaW4", 1.0);
		}
	}

	void collect(std::ostream& os)
	{
		double eval_param[] = {config->params.zeta2, config->params.zeta4};
		measure.add_evalable("M2", "deltaZ", "deltaW2", "deltaW4", eval_M2,
			eval_param);
		measure.add_evalable("M4", "deltaZ", "deltaW2", "deltaW4", eval_M4,
			eval_param);
		measure.add_evalable("BinderRatio", "deltaZ", "deltaW2", "deltaW4",
			eval_B, eval_param);
		
		os << "PARAMETERS" << std::endl;
		pars.get_all(os);
		measure.get_statistics(os);
	}
};
