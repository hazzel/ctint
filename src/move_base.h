#pragma once
#include <vector>
#include <functional>
#include <memory>
#include <iostream>

class move_base
{
	public:
		template<typename T>
		move_base(T&& functor, const std::string& name_, double prop_rate_=1.0)
			: name_str(name_), prop_rate(prop_rate_)
		{
			construct_delegation(new typename std::remove_reference<T>::type(
				std::forward<T>(functor)));
		}
		
		move_base(const move_base& rhs) {*this = rhs;}
		move_base(move_base& rhs) {*this = rhs;} // to avoid clash with
		//tempalte construction  !
		move_base(move_base&& rhs) {*this = std::move(rhs);}
		move_base& operator = (const move_base& rhs) { *this = rhs.clone_fun();
			return *this;}
		move_base& operator = (move_base&& rhs) = default;

		double attempt() { return attempt_fun(); }
		double accept() { return accept_fun(); }
		void reject() { reject_fun(); }
		std::string name() { return name_str; }
		double proposal_rate() const { return prop_rate; }
		void proposal_rate(double prop_rate_) { prop_rate = prop_rate_; }
	private:
		template<typename T>
		void construct_delegation (T* functor)
		{
			impl = std::shared_ptr<T>(functor);
			attempt_fun = [functor]() { return functor->attempt(); };
			accept_fun = [functor]() { return functor->accept(); };
			reject_fun = [functor]() { functor->reject(); };
			clone_fun = [functor, this]() { return move_base(*functor, name_str); };
		}
	private:
		std::shared_ptr<void> impl;
		std::function<double()> attempt_fun;
		std::function<double()> accept_fun;
		std::function<void()> reject_fun;
		std::function<move_base()> clone_fun;
		std::string name_str;
		double prop_rate;
};
