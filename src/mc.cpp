#include <string>
#include <fstream>
#include "mc.h"
#include "move_functors.h"
#include "measure_functors.h"
#include "event_functors.h"

mc::mc(const std::string& dir)
	: rng(Random()), config(measure), qmc(rng, config)
{
	//Read parameters
	pars.read_file(dir);
	sweep = 0;
	config.param.n_static_cycles = pars.value_or_default<int>("cycles", 300);
	n_warmup = pars.value_or_default<int>("warmup", 100000);
	config.param.n_prebin = pars.value_or_default<int>("prebin", 500);
	n_rebuild = pars.value_or_default<int>("rebuild", 1000);
	n_tau_slices = pars.value_or_default<int>("tau_slices", 500);
	config.param.n_matsubara = pars.value_or_default<int>("matsubara_freqs",
		0);
	config.param.n_discrete_tau = pars.value_or_default<int>("discrete_tau",
		0);
	//hc.Lx = pars.value_or_default<int>("L", 9);
	//hc.Ly = pars.value_or_default<int>("L", 9);
	hc.L = pars.value_or_default<int>("L", 9);
	config.param.beta = 1./pars.value_or_default<double>("T", 0.2);
	config.param.V = pars.value_or_default<double>("V", 1.355);
	config.param.mu = pars.value_or_default<double>("mu", 0.);
	config.param.zeta2 = pars.value_or_default<double>("zeta2", 10.0);
	config.param.zeta4 = pars.value_or_default<double>("zeta4", 30.0);
	config.param.worm_nhood_dist = pars.value_or_default<int>("nhood_dist", 4);

	//Proposal probabilites
	config.param.add.push_back(pars.value_or_default<double>("add_1", 1.0));
	config.param.rem.push_back(pars.value_or_default<double>("rem_1", 1.0));
	config.param.add.push_back(pars.value_or_default<double>("add_2", 1.0));
	config.param.rem.push_back(pars.value_or_default<double>("rem_2", 1.0));
	config.param.ZtoW2 = pars.value_or_default<double>("ZtoW2", 1.0);
	config.param.W2toZ = pars.value_or_default<double>("W2toZ", 1.0);
	config.param.ZtoW4 = pars.value_or_default<double>("ZtoW4", 1.0);
	config.param.W4toZ = pars.value_or_default<double>("W4toZ", 1.0);
	config.param.W2toW4 = pars.value_or_default<double>("W2toW4", 1.0);
	config.param.W4toW2 = pars.value_or_default<double>("W4toW2", 1.0);
	config.param.worm_shift = pars.value_or_default<double>("worm_shift", 1.0);

	//Initialize lattice
	config.l.generate_graph(hc);
	if (config.param.worm_nhood_dist == -1)
		config.param.worm_nhood_dist = config.l.max_distance();
	hc.generate_maps(config.l);
	config.l.generate_neighbor_map("worm nhood", [this]
		(lattice::vertex_t i, lattice::vertex_t j) {
		return config.l.distance(i, j) <= config.param.worm_nhood_dist; });
	config.l.generate_neighbor_map("shift nhood", [this] (lattice::vertex_t i,
		lattice::vertex_t j) { return i != j && config.l.distance(i, j) <=
		config.param.worm_nhood_dist; });
	config.param.ratio_w2 = static_cast<double>(config.l.neighbors(0,
		"worm nhood").size()) / static_cast<double>(config.l.n_sites());
	config.param.ratio_w4 = std::pow(static_cast<double>(config.l.neighbors(0,
		"worm nhood").size()) / static_cast<double>(config.l.n_sites()), 3.0);

	//Set up bare greens function look up
	config.g0.generate_mesh(&config.l, config.param.beta, config.param.mu,
		n_tau_slices, config.param.n_matsubara);
	
	//Initialize configuration class
	config.initialize();

	//Set up Monte Carlo moves
	qmc.add_move(move_insert<1>{config, rng}, "insertion n=1",
		config.param.add[0]);
	qmc.add_move(move_remove<1>{config, rng}, "removal n=1",
		config.param.rem[0]);
	qmc.add_move(move_insert<2>{config, rng}, "insertion n=2",
		config.param.add[1]);
	qmc.add_move(move_remove<2>{config, rng}, "removal n=2",
		config.param.rem[1]);
	qmc.add_move(move_ZtoW2{config, rng, false}, "Z -> W2",
		config.param.ZtoW2);
	qmc.add_move(move_W2toZ{config, rng, false}, "W2 -> Z",
		config.param.W2toZ);
	qmc.add_move(move_ZtoW4{config, rng, false}, "Z -> W4",
		config.param.ZtoW4);
	qmc.add_move(move_W4toZ{config, rng, false}, "W4 -> Z",
		config.param.W4toZ);
	qmc.add_move(move_W2toW4{config, rng, false}, "W2 -> W4",
		config.param.W2toW4);
	qmc.add_move(move_W4toW2{config, rng, false}, "W4 -> W2",
		config.param.W4toW2);
	qmc.add_move(move_shift{config, rng, false}, "worm shift",
		config.param.worm_shift);
	
	//qmc.add_measure(measure_worm{config, rng, pars}, "measurement");
	qmc.add_measure(measure_dynamics{config, rng, pars}, "measurement");
	
	//Set up events
	qmc.add_event(event_rebuild{config, config.measure}, "rebuild");
	//qmc.add_event(event_print_M{config, config.measure}, "print_M");
	qmc.add_event(event_print_vertices{config, config.measure}, "print_vertices");
	qmc.add_event(event_build{config, rng}, "initial build");
	
	std::vector<std::map<int, int>> cb_bonds(3);
	for (int i = 0; i < config.l.n_sites(); ++i)
	{
		auto& nn = config.l.neighbors(i, "nearest neighbors");
		for (int j : nn)
		{
			for (auto& b : cb_bonds)
			{
				if (!b.count(i) && !b.count(j))
				{
					b[i] = j;
					b[j] = i;
					break;
				}
			}
		}
	}
	auto get_bond_type = [&] (const std::pair<int, int>& bond) -> int
	{
		for (int i = 0; i < cb_bonds.size(); ++i)
			if (cb_bonds[i].at(bond.first) == bond.second)
				return i;
	};
	
	std::ofstream f_epsilon("ep_lattice.txt");
	std::ofstream f_kek("kek_lattice.txt");
	std::ofstream f_kek_2("kek_2_lattice.txt");
	std::ofstream f_kek_3("kek_3_lattice.txt");
	std::ofstream f_chern("chern_lattice.txt");
	std::ofstream f_chern_2("chern_2_lattice.txt");
	std::ofstream f_bond_type("bond_type_lattice.txt");
	std::ofstream t3_bonds("t3_lattice.txt");
	for (auto& b : config.l.bonds("nearest neighbors"))
		f_epsilon << b.first << "," << config.l.real_space_coord(b.first)[0] << ","
			<< config.l.real_space_coord(b.first)[1] << "," << b.second << ","
			<< config.l.real_space_coord(b.second)[0] << ","
			<< config.l.real_space_coord(b.second)[1] << std::endl;
	
	for (auto& b : config.l.bonds("kekule"))
		f_kek << b.first << "," << config.l.real_space_coord(b.first)[0] << ","
			<< config.l.real_space_coord(b.first)[1] << "," << b.second << ","
			<< config.l.real_space_coord(b.second)[0] << ","
			<< config.l.real_space_coord(b.second)[1] << std::endl;
	for (auto& b : config.l.bonds("kekule_2"))
		f_kek_2 << b.first << "," << config.l.real_space_coord(b.first)[0] << ","
			<< config.l.real_space_coord(b.first)[1] << "," << b.second << ","
			<< config.l.real_space_coord(b.second)[0] << ","
			<< config.l.real_space_coord(b.second)[1] << std::endl;
	for (auto& b : config.l.bonds("kekule_3"))
		f_kek_3 << b.first << "," << config.l.real_space_coord(b.first)[0] << ","
			<< config.l.real_space_coord(b.first)[1] << "," << b.second << ","
			<< config.l.real_space_coord(b.second)[0] << ","
			<< config.l.real_space_coord(b.second)[1] << std::endl;
	for (auto& b : config.l.bonds("chern"))
		f_chern << b.first << "," << config.l.real_space_coord(b.first)[0] << ","
			<< config.l.real_space_coord(b.first)[1] << "," << b.second << ","
			<< config.l.real_space_coord(b.second)[0] << ","
			<< config.l.real_space_coord(b.second)[1] << std::endl;
	for (auto& b : config.l.bonds("chern_2"))
		f_chern_2 << b.first << "," << config.l.real_space_coord(b.first)[0] << ","
			<< config.l.real_space_coord(b.first)[1] << "," << b.second << ","
			<< config.l.real_space_coord(b.second)[0] << ","
			<< config.l.real_space_coord(b.second)[1] << std::endl;
	for (auto& b : cb_bonds[1])
		f_bond_type << b.first << "," << config.l.real_space_coord(b.first)[0] << ","
			<< config.l.real_space_coord(b.first)[1] << "," << b.second << ","
			<< config.l.real_space_coord(b.second)[0] << ","
			<< config.l.real_space_coord(b.second)[1] << std::endl;
	for (auto& b : config.l.bonds("t3_bonds"))
		t3_bonds << b.first << "," << config.l.real_space_coord(b.first)[0] << ","
			<< config.l.real_space_coord(b.first)[1] << "," << b.second << ","
			<< config.l.real_space_coord(b.second)[0] << ","
			<< config.l.real_space_coord(b.second)[1] << std::endl;
	f_epsilon.close();
	f_kek.close();
	f_kek_2.close();
	f_kek_3.close();
	f_chern.close();
	f_chern_2.close();
	f_bond_type.close();
	t3_bonds.close();
}

mc::~mc()
{}

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
	//Initialize vertex list to reduce warm up time
	qmc.trigger_event("initial build");
}

void mc::write(const std::string& dir)
{
	odump d(dir+"dump");
	random_write(d);
	d.write(sweep);
	config.serialize(d);
	d.close();
	seed_write(dir+"seed");
	std::ofstream f(dir+"bins");
	if (is_thermalized())
	{
		f << "Thermalization: Done." << std::endl;
		f << "Sweeps: " << (sweep - n_warmup) << std::endl;
		f << "Bins: " << static_cast<int>((sweep - n_warmup) / config.param.n_prebin)
			<< std::endl;
	}
	else
	{
		f << "Thermalization: " << sweep << std::endl;
		f << "Sweeps: 0" << std::endl;
		f << "Bins: 0" << std::endl;
	}
	f.close();
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
		config.serialize(d);
		d.close();
		return true;
	}
}

void mc::write_output(const std::string& dir)
{
	std::ofstream f(dir);
	qmc.collect_results(f);
	f.close();
	/*
	const std::vector<std::pair<std::string, double>>& acc =
		qmc.acceptance_rates();
	for (auto a : acc)
		std::cout << a.first << " : " << a.second << std::endl;
	std::cout << "Average sign: " << qmc.average_sign() << std::endl;
	*/
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
		for (int i = 0; i < config.param.n_static_cycles; ++i)
			qmc.do_update();
	++sweep;
	if (sweep % n_rebuild == 0)
		qmc.trigger_event("rebuild");
	//if (is_thermalized() && sweep % 1000 == 0)
	//	qmc.trigger_event("print_vertices");
	status();
}

void mc::do_measurement()
{
	qmc.do_measurement();
}

void mc::status()
{
//	if (sweep == n_warmup)
//		std::cout << "Thermalization done." << std::endl;
//	if (is_thermalized() && sweep % (10000) == 0)
//	{
//		std::cout << "sweep: " << sweep << std::endl;
//		std::cout << "pert order: " << config.perturbation_order() << std::endl;
//	}
}
