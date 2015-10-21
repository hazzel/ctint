#pragma once
#include "measurements.h"
#include "move_base.h"

struct measure_M
{
	configuration* config;
	measurements& measure;

	measure_M(configuration* config_, measurements& measure_)
		: config(config_), measure(measure_)
	{}

	void perform()
	{
		if (config->worms() == 0) //measure Z
		{
			measure.add("<k>_Z", config->perturbation_order());
			measure.add("Z", 1.0);
			measure.add("W2", 0.0);
			measure.add("W4", 0.0);
		}
		else if (config->worms() == 1) //measure W2
		{
			measure.add("<k>_W2", config->perturbation_order());
			measure.add("Z", 0.0);
			measure.add("W2", 1.0);
			measure.add("W4", 0.0);		}
		else if (config->worms() == 2) //measure W4
		{
			measure.add("<k>_W4", config->perturbation_order());
			measure.add("Z", 0.0);
			measure.add("W2", 0.0);
			measure.add("W4", 1.0);
		}
	}

	void collect_results()
	{
	}
};


