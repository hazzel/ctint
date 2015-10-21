#include <sstream>
#include <fstream>
#include <string>
#include <limits>
#include <functional>
#include "mc.h"
#include "measure_functors.h"

mc::mc(const std::string& dir)
	: rng(Random()), qmc(rng)
{
	pars.read_file(dir);
	sweep = 0;
	n_cycles = pars.value_or_default<int>("cycles", 300);
	n_warmup = pars.value_or_default<int>("warmup", 100000);
	n_prebin = pars.value_or_default<int>("prebin", 500);
	hc.L = pars.value_or_default<int>("L", 9);
	param.beta = 1./pars.value_or_default<double>("T", 0.2);
	param.V = pars.value_or_default<double>("V", 1.355);
	param.zeta2 = pars.value_or_default<double>("zeta2", 10.0);
	param.zeta4 = pars.value_or_default<double>("zeta4", 30.0);
	param.worm_nhood_dist = pars.value_or_default<int>("nhood_dist", 4);
	n_tau_slices = pars.value_or_default<int>("tau_slices", 500);
}

mc::~mc()
{
	delete[] evalableParameters;
	delete config;
}

void mc::random_write(odump& d)
{
	rng.RngHandle()->write(d);
}
void mc::seed_write(const std::string& fn)
{
	std::ofstream s;
	s.open(fn.c_str());
	s << rng.Seed() << std::endl;
	s.close();
}
void mc::random_read(idump& d)
{
	rng.NewRng();
	rng.RngHandle()->read(d);
}
void mc::init()
{
	//Initialize lattice
	lat.generate_graph(hc);
	if (param.worm_nhood_dist == -1)
		param.worm_nhood_dist = lat.max_distance();
	lat.generate_neighbor_map(param.worm_nhood_dist);
	lat.generate_neighbor_map(1);
	param.ratio_w2 = static_cast<double>(lat.neighbors(0,
		param.worm_nhood_dist).size()) / static_cast<double>(lat.n_sites());
	param.ratio_w4 = std::pow(static_cast<double>(lat.neighbors(0,
		param.worm_nhood_dist).size()) / static_cast<double>(lat.n_sites()), 3.0);

	//Set up bare greens function look up
	g0.generate_mesh(&lat, param.beta, n_tau_slices);

	//Set up Monte Carlo moves
	config = new configuration(lat, g0, param);
	qmc.add_move(move_insert{config, rng}, "insertion");
	qmc.add_move(move_remove{config, rng}, "removal");
	//qmc.add_move(move_ZtoW2{config, rng, false}, "Z -> W2");
	//qmc.add_move(move_W2toZ{config, rng, false}, "W2 -> Z");
	//qmc.add_move(move_ZtoW4{config, rng, false}, "Z -> W4");
	//qmc.add_move(move_W4toZ{config, rng, false}, "W4 -> Z");
	//qmc.add_move(move_W2toW4{config, rng, false}, "W2 -> W4");
	//qmc.add_move(move_W4toW2{config, rng, false}, "W4 -> W2");
	//qmc.add_move(move_shift{config, rng, false}, "worm shift");

	//Set up measurements
	measure.add_observable("<k>_Z", n_prebin);
	measure.add_observable("<k>_W2", n_prebin);
	measure.add_observable("<k>_W4", n_prebin);
	measure.add_observable("Z", n_prebin);
	measure.add_observable("W2", n_prebin);
	measure.add_observable("W4", n_prebin);
	qmc.add_measure(measure_M{config, measure}, "measurement");
}
void mc::write(const std::string& dir)
{
	odump d(dir+"dump");
	random_write(d);
	d.write(sweep);
	d.close();
	seed_write(dir+"seed");
}
bool mc::read(const std::string& dir)
{
	idump d(dir+"dump");
	if (!d)
	{
		std::cout << "read fail" << std::endl;
		return false;
	}
	else
	{
		random_read(d);
		d.read(sweep);
		d.close();
		return true;
	}
}

void mc::write_output(const std::string& dir)
{
	std::ofstream f;
	f.open(dir.c_str());
	f << "PARAMETERS" << std::endl;
	pars.get_all(f);
	measure.get_statistics(f);
	f.close();

	const std::vector<std::pair<std::string, double>>& acc =
		qmc.acceptance_rates();
	for (auto a : acc)
		std::cout << a.first << " : " << a.second << std::endl;
	std::cout << "Average sign: " << qmc.average_sign() << std::endl;
}

bool mc::is_thermalized()
{
	return sweep >= n_warmup;
}

void mc::do_update()
{
	if (!is_thermalized())
		qmc.do_update();
	else
		for (int i = 0; i < n_cycles; ++i)
			qmc.do_update();
	++sweep;
	status();
}

void mc::do_measurement()
{
	qmc.do_measurement();
}

void mc::status()
{
	if (sweep == n_warmup)
		std::cout << "Thermalization done." << std::endl;
	if (is_thermalized() && sweep % (10000) == 0)
	{
		std::cout << "sweep: " << sweep << std::endl;
		std::cout << "pert order: " << config->perturbation_order() << std::endl;
	}
}
