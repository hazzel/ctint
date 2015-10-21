#pragma once
#include <vector>
#include <functional>
#include <utility>
#include <memory>
#include <iostream>

class measure_base
{
	public:
		template<typename T>
		measure_base(T&& functor, const std::string& name_)
			: name_str(name_)
		{
			construct_delegation(new typename std::remove_reference<T>::type(
				std::forward<T>(functor)));
		}
		
		measure_base(const measure_base& rhs) {*this = rhs;}
		measure_base(measure_base& rhs) {*this = rhs;} // to avoid clash with
		//tempalte construction  !
		measure_base(measure_base&& rhs) {*this = std::move(rhs);}
		measure_base& operator = (const measure_base& rhs) { *this =
			rhs.clone_fun(); return *this;}
		measure_base& operator = (measure_base&& rhs) = default;

		void perform() { perform_fun(); }	
		std::string name() { return name_str; }
	private:
		template<typename T>
		void construct_delegation (T* functor)
		{
			impl = std::shared_ptr<T>(functor);
			perform_fun = [functor]() { functor->perform(); };
			clone_fun = [functor, this]() { return measure_base(*functor,
				name_str); };
		}
	private:
		std::shared_ptr<void> impl;
		std::function<void()> perform_fun;
		std::function<measure_base()> clone_fun;
		std::string name_str;
};
