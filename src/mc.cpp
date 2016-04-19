#include <string>
#include <fstream>
#include "mc.h"
#include "move_functors.h"
#include "measure_functors.h"
#include "event_functors.h"

mc::mc(const std::string& dir)
	: rng(Random()), qmc(rng), config{}
{
	//Read parameters
	pars.read_file(dir);
	sweep = 0;
	n_cycles = pars.value_or_default<int>("cycles", 300);
	n_warmup = pars.value_or_default<int>("warmup", 100000);
	n_prebin = pars.value_or_default<int>("prebin", 500);
	n_rebuild = pars.value_or_default<int>("rebuild", 1000);
	n_tau_slices = pars.value_or_default<int>("tau_slices", 500);
	config.param.n_matsubara = pars.value_or_default<int>("matsubara_freqs",
		10);
	config.param.n_discrete_tau = pars.value_or_default<int>("discrete_tau",
		100);
	hc.L = pars.value_or_default<int>("L", 9);
	config.param.beta = 1./pars.value_or_default<double>("T", 0.2);
	config.param.V = pars.value_or_default<double>("V", 1.355);
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
	config.l.generate_neighbor_map("nearest neighbors", [this]
		(lattice::vertex_t i, lattice::vertex_t j) {
		return config.l.distance(i, j) == 1; });
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
	config.g0.generate_mesh(&config.l, config.param.beta, n_tau_slices,
		config.param.n_matsubara);
	
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

	//Set up measurements
	config.measure.add_observable("<k>_Z", n_prebin);
	if (config.param.ZtoW2 + config.param.ZtoW4 > 0.)
	{
		config.measure.add_observable("<k>_W2", n_prebin);
		config.measure.add_observable("<k>_W4", n_prebin);
		config.measure.add_observable("deltaZ", n_prebin);
		config.measure.add_observable("deltaW2", n_prebin);
		config.measure.add_observable("deltaW4", n_prebin);
		config.measure.add_vectorobservable("corr", config.l.max_distance() + 1,
			n_prebin);
	}
	config.measure.add_vectorobservable("dyn_M2_mat",
		config.param.n_matsubara, n_prebin);
	config.measure.add_vectorobservable("dyn_M2_tau",
		config.param.n_discrete_tau, n_prebin);
	config.measure.add_vectorobservable("dyn_sp_tau",
		config.param.n_discrete_tau, n_prebin);
	config.measure.add_vectorobservable("dyn_tp_tau",
		config.param.n_discrete_tau, n_prebin);
	//Measure acceptance probabilities
	if (config.param.add[0] > 0.)
		config.measure.add_observable("insertion n=1", n_prebin * n_cycles);
	if (config.param.rem[0] > 0.)
		config.measure.add_observable("removal n=1", n_prebin * n_cycles);
	if (config.param.add[1] > 0.)
		config.measure.add_observable("insertion n=2", n_prebin * n_cycles);
	if (config.param.rem[1] > 0.)
		config.measure.add_observable("removal n=2", n_prebin * n_cycles);
	if (config.param.ZtoW2 > 0.)
		config.measure.add_observable("Z -> W2", n_prebin * n_cycles);
	if (config.param.W2toZ > 0.)
		config.measure.add_observable("W2 -> Z", n_prebin * n_cycles);
	if (config.param.W2toW4 > 0.)
		config.measure.add_observable("W2 -> W4", n_prebin * n_cycles);
	if (config.param.W4toW2 > 0.)
		config.measure.add_observable("W4 -> W2", n_prebin * n_cycles);
	if (config.param.ZtoW4 > 0.)
		config.measure.add_observable("Z -> W4", n_prebin * n_cycles);
	if (config.param.W4toZ > 0.)
		config.measure.add_observable("W4 -> Z", n_prebin * n_cycles);
	if (config.param.worm_shift > 0.)
		config.measure.add_observable("worm shift", n_prebin * n_cycles);
	config.measure.add_observable("sign", n_prebin * n_cycles);
	
	//qmc.add_measure(measure_worm{config, config.measure, pars,
	//	std::vector<double>(config.l.max_distance() + 1, 0.0)}, "measurement");
	//config.measure.add_vectorobservable("Correlations", config.l.max_distance(),
	//	n_prebin);
	qmc.add_measure(measure_estimator{config, rng, config.measure, pars,
		std::vector<double>(config.param.n_matsubara, 0.0),
		std::vector<double>(config.param.n_discrete_tau + 1, 0.0),
		std::vector<double>(config.param.n_matsubara, 0.0)}, "measurement");
	
	//Set up events
	qmc.add_event(event_rebuild{config, config.measure}, "rebuild");
	qmc.add_event(event_build{config, rng}, "initial build");
	//Initialize vertex list to reduce warm up time
	qmc.trigger_event("initial build");
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
void mc::init() {}

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
		f << "Bins: " << static_cast<int>((sweep - n_warmup) / n_prebin)
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
		qmc.do_update(config.measure);
	else
		for (int i = 0; i < n_cycles; ++i)
			qmc.do_update(config.measure);
	++sweep;
	if (sweep % n_rebuild == 0)
		qmc.trigger_event("rebuild");
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
