#include <sstream>
#include <fstream>
#include <string>
#include <limits>
#include <functional>
#include "mc.h"

mc::mc(const std::string& dir)
	: rng(Random())
{
	sweep = 0;
	pars.read_file(dir);
	hc.L = 6;
	param = parameters{5.0, 1.5, 10.0, 30.0, 4, 0.0, 0.0};
	n_tau_slices = 500;
}

mc::~mc()
{
	delete[] evalableParameters;
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
	configuration config{lat, g0, param};
	qmc.add_move(move_insert{&config, rng}, "insertion");
	qmc.add_move(move_remove{&config, rng}, "removal");
	qmc.add_move(move_ZtoW2{&config, rng, false}, "Z -> W2");
	qmc.add_move(move_W2toZ{&config, rng, false}, "W2 -> Z");
	qmc.add_move(move_ZtoW4{&config, rng, false}, "Z -> W4");
	qmc.add_move(move_W4toZ{&config, rng, false}, "W4 -> Z");
	qmc.add_move(move_W2toW4{&config, rng, false}, "W2 -> W4");
	qmc.add_move(move_W4toW2{&config, rng, false}, "W4 -> W2");
	qmc.add_move(move_shift{&config, rng, false}, "worm shift");
	
	measure.add_observable("k", n_prebin);
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
}

bool mc::is_thermalized()
{
	return sweep >= n_warmup;
}

void mc::do_update()
{
	std::cout << sweep << std::endl;
	++sweep;
}

void mc::do_measurement()
{
	//std::cout << "measure" << std::endl;
}
