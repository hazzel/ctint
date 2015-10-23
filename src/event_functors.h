#pragma once
#include "measurements.h"
#include "move_functors.h"

struct event_rebuild
{
	configuration* config;
	measurements& measure;

	void trigger()
	{
		config->M.rebuild();
	}
};
