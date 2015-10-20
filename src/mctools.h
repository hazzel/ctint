#pragma once
#include <vector>
#include <functional>
#include <algorithm>
#include <iostream>
#include "move_base.h"

class mctools
{
	public:
		mctools()
		{
			//moves.reserve(10);
		};
		~mctools() {};

		template<typename T>
		void add_move(T&& functor, const std::string& name, double prop_rate=1.0)
		{
			moves.emplace_back(std::forward<move_base>(move_base{
				std::forward<T>(functor), name, prop_rate}));
			normalize_proposal_rates();
		}
	private:
		void normalize_proposal_rates()
		{
			double sum = 0.0;
			for (move_base& m : moves)
				sum += m.proposal_rate();
			for (move_base& m : moves)
				m.proposal_rate(m.proposal_rate() / sum);
		}
	private:
		std::vector<move_base> moves;
};
