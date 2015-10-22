#pragma once
#include <vector>
#include <functional>
#include <utility>
#include <algorithm>
#include <ostream>
#include <iostream>
#include "Random.h"
#include "move_base.h"
#include "measure_base.h"

class mctools
{
	public:
		mctools(Random& rng_) : rng(rng_) {}
		~mctools()
		{
			//moves.reserve(12);
		}

		template<typename T>
		void add_move(T&& functor, const std::string& name, double prop_rate=1.0)
		{
			moves.push_back(move_base{std::forward<T>(functor), name, prop_rate});
			normalize_proposal_rates();
			acceptance.push_back(std::make_pair(name, 0.0));
		}

		template<typename T>
		void add_move(T* functor, const std::string& name, double prop_rate=1.0)
		{
			moves.push_back(move_base{functor, name, prop_rate});
			normalize_proposal_rates();
			acceptance.push_back(std::make_pair(name, 0.0));
		}

		template<typename T>
		void add_measure(T&& functor, const std::string& name)
		{
			measures.push_back(measure_base{std::forward<T>(functor), name});
		}

		void do_update()
		{
			double r = rng();
			for (int i = 0; i < moves.size(); ++i)
			{
				if (r < proposal[i])
				{
					double q = moves[i].attempt();
					if (q < 0.0)
						std::cout << "Negative sign at " << moves[i].name()
							<< std::endl;
					if (rng() < std::abs(q))
						moves[i].accept();
					else
						moves[i].reject();
					break;
				}
			}
		}

		void do_measurement()
		{
			for (measure_base& m : measures)
				m.perform();
		}
		
		void collect_results(std::ostream& os)
		{
			for (measure_base& m : measures)
				m.collect(os);
		}

		const std::vector<std::pair<std::string, double>>& acceptance_rates()
		{
			for (int i = 0; i < moves.size(); ++i)
				acceptance[i].second = moves[i].acceptance_rate();
			return acceptance;
		}

		double average_sign()
		{
			double sign = 0.0;
			for (int i = 0; i < moves.size(); ++i)
				sign += moves[i].sign();
			return sign / static_cast<double>(moves.size());
		}
	private:
		void normalize_proposal_rates()
		{
			proposal.assign(moves.size(), 0.0);
			double sum = 0.0;
			for (move_base& m : moves)
				sum += m.proposal_rate();
			for (move_base& m : moves)
				m.proposal_rate(m.proposal_rate() / sum);
			proposal[0] = moves[0].proposal_rate();
			for (int i = 1; i < moves.size(); ++i)
				proposal[i] = proposal[i-1] + moves[i].proposal_rate();
		}
	private:
		Random& rng;
		std::vector<move_base> moves;
		std::vector<measure_base> measures;
		std::vector<double> proposal;
		std::vector<std::pair<std::string, double>> acceptance;
};
