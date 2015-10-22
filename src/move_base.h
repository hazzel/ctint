#pragma once
#include <vector>
#include <functional>
#include <utility>
#include <memory>
#include <iostream>

class move_base
{
	public:
		template<typename T>
		move_base(T&& functor, const std::string& name_, double prop_rate_=1.0)
			: name_str(name_), prop_rate(prop_rate_), n_attempted(0),
				n_accepted(0)
		{
			std::cout << "move_base constructor T&& " << name_str << std::endl;
			construct_delegation(new typename std::remove_reference<T>::type(
				std::forward<T>(functor)));
		}

		template<typename T>
		move_base(T* functor, const std::string& name_, double prop_rate_=1.0)
			: name_str(name_), prop_rate(prop_rate_), n_attempted(0),
				n_accepted(0)
		{
			std::cout << "move_base constructor T* " << name_str << std::endl;
			construct_delegation(functor);
		}
		
		move_base(const move_base& rhs)
		{
			std::cout << "move_base const copy " << name_str << std::endl;
			*this = rhs;
		}
		// to avoid clash with tempalte construction  !
		move_base(move_base& rhs)
		{
			*this = rhs;
			std::cout << "move_base copy= " << name_str << std::endl;
		} 
		move_base(move_base&& rhs)
		{
			*this = std::move(rhs);
			std::cout << "move_base move constructor= " << name_str << std::endl;
		}
		move_base& operator=(const move_base& rhs)
		{
			*this = rhs.clone_fun();
			std::cout << "move_base operator= " << name_str << std::endl;
			return *this;
		}
		move_base& operator=(move_base&& rhs) = default;

		double attempt()
		{
			double p = attempt_fun();
			avg_sign *= static_cast<double>(n_attempted);
			avg_sign += (p >= 0.0) - (p < 0.0);
			++n_attempted;
			avg_sign /= static_cast<double>(n_attempted);
			return p;
		}
		double accept() { ++n_accepted; return accept_fun(); }
		void reject() { reject_fun(); }
		std::string name() { return name_str; }
		double proposal_rate() const { return prop_rate; }
		void proposal_rate(double prop_rate_) { prop_rate = prop_rate_; }
		double acceptance_rate() const { return static_cast<double>(n_accepted)
			/ static_cast<double>(n_attempted); }
		double sign() { return avg_sign; }
	private:
		template<typename T>
		void construct_delegation (T* functor)
		{
			impl = std::shared_ptr<T>(functor);
			attempt_fun = [functor]() { return functor->attempt(); };
			accept_fun = [functor]() { return functor->accept(); };
			reject_fun = [functor]() { functor->reject(); };
			clone_fun = [functor, this]() { return move_base(*functor, name_str); };
			//clone_fun = [functor, this]() { return move_base(new T(*functor),
			//	name_str); };
		}
	private:
		std::shared_ptr<void> impl;
		std::function<double()> attempt_fun;
		std::function<double()> accept_fun;
		std::function<void()> reject_fun;
		std::function<move_base()> clone_fun;
		std::string name_str;
		double prop_rate;
		double avg_sign;
		unsigned int n_attempted;
		unsigned int n_accepted;
};