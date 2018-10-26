#pragma once
#include <iostream>
#include <array>
#include <vector>
#include <map>
#include <string>
#include <functional>
#include <Eigen/Dense>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/named_function_params.hpp>
#include <boost/graph/visitors.hpp>
#include "boost/multi_array.hpp"

class lattice
{
	public:
		typedef boost::adjacency_list<boost::setS, boost::vecS,
				boost::undirectedS> graph_t;
		typedef graph_t::vertex_descriptor vertex_t;
		typedef graph_t::vertex_iterator vertex_it_t;
		typedef graph_t::edge_descriptor edge_t;
		typedef graph_t::edge_iterator edge_it_t;
		typedef std::vector<std::vector<int>> nested_vector_t;
		typedef std::pair<int, int> pair_t;
		typedef std::vector<pair_t> pair_vector_t;
		typedef std::map<std::string, nested_vector_t> neighbor_map_t;
		typedef std::map<std::string, pair_vector_t> bond_map_t;
		typedef std::vector<Eigen::Vector2d> real_space_map_t;
		typedef std::map<std::string, Eigen::Vector2d> point_map_t;

		lattice()
			: graph(0)
		{}
		~lattice() { delete graph; }

		template<typename T>
		void generate_graph(T& generator)
		{
			delete graph;
			graph = generator.graph(*this);
			real_space_map = generator.real_space_map;
			generate_distance_map();
			generator.generate_maps(*this);
		}

		void generate_neighbor_map(const std::string& name,
			std::function<bool(vertex_t, vertex_t)> fun)
		{
			if (neighbor_maps.count(name))
			{
				std::cerr << "Neighbor map already exists." << std::endl;
				return;
			}
			neighbor_maps[name] = nested_vector_t(n_sites());
			for (int i = 0; i < n_sites(); ++i)
			{
				for (int j = 0; j < n_sites(); ++j)
				{
					if (fun(i, j))
						neighbor_maps[name][i].push_back(j);
				}
			}
		}
		
		void generate_bond_map(const std::string& name,
			std::function<bool(vertex_t, vertex_t)> fun)
		{
			if (bond_maps.count(name))
			{
				std::cerr << "Bond map already exists." << std::endl;
				return;
			}
			bond_maps[name] = pair_vector_t();
			for (int i = 0; i < n_sites(); ++i)
			{
				for (int j = 0; j < n_sites(); ++j)
				{
					if (fun(i, j))
						bond_maps[name].push_back({i, j});
				}
			}
		}
		
		void generate_bond_map(const std::string& name,
			std::function<void(pair_vector_t&)> fun)
		{
			if (bond_maps.count(name))
			{
				std::cerr << "Bond map already exists." << std::endl;
				return;
			}
			bond_maps[name] = pair_vector_t();
			fun(bond_maps[name]);
		}

		inline void add_symmetry_points(const point_map_t& points)
		{
			symmetry_points.insert(points.begin(), points.end());
		}

		inline const Eigen::Vector2d& symmetry_point(const std::string& name) const
		{
			return symmetry_points.at(name);
		}

		inline int n_sites() const
		{
			return boost::num_vertices(*graph);
		}

		// Edges on graph = nearest neighbor bonds
		int n_bonds() const
		{
			return boost::num_edges(*graph);
		}

		inline int max_distance() const
		{
			return distance_list.size();
		}

		inline int distance(vertex_t i, vertex_t j) const
		{
			if (i < j)
				return distance_map[(i*(2*n_sites()-1-i))/2 + j - i - 1];
			else if (i > j)
				return distance_map[(j*(2*n_sites()-1-j))/2 + i - j - 1];
			else
				return 0;
		}

		inline const std::vector<int>& neighbors(vertex_t site, const std::string& name)
			const
		{
			return neighbor_maps.at(name)[site];
		}
		
		inline const pair_vector_t& bonds(const std::string& name) const
		{
			return bond_maps.at(name);
		}

		//TODO: generalize as vertex property on graph
		inline int sublattice(vertex_t site) const
		{
			return site % 2;
		}

		inline double parity(vertex_t site) const
		{
			return (site % 2 == 0) ? 1.0 : -1.0;
		}

		inline const Eigen::Vector2d& real_space_coord(vertex_t i) const
		{
			return real_space_map[i];
		}
		
		vertex_t site_at_position(const Eigen::Vector2d& R) const
		{
			std::vector<double> fx, fy;
			fx.push_back(0);
			fy.push_back(0);
			for (int i = 1; i <= Lx; ++i)
			{
				fx.push_back(i);
				fx.push_back(-i);
			}
			for (int i = 1; i <= Ly; ++i)
			{
				fy.push_back(i);
				fy.push_back(-i);
			}
			for (int a = 0; a < fx.size(); ++a)
				for (int b = 0; b < fy.size(); ++b)
				{
					Eigen::Vector2d new_R = R + fx[a] * Lx * a1 + fy[b] * Ly * a2;
					for (int i = 0; i < n_sites(); ++i)
					{
						auto& R_i = real_space_coord(i);
						if ((new_R - R_i).norm() < std::pow(10., -12.))
							return i;
					}
				}
			throw std::runtime_error("No lattice site at this position.");
		}
		
		bool is_lattice_site(const Eigen::Vector2d& R) const
		{
			std::vector<double> fx, fy;
			fx.push_back(0);
			fy.push_back(0);
			for (int i = 1; i <= Lx; ++i)
			{
				fx.push_back(i);
				fx.push_back(-i);
			}
			for (int i = 1; i <= Ly; ++i)
			{
				fy.push_back(i);
				fy.push_back(-i);
			}
			for (int a = 0; a < fx.size(); ++a)
				for (int b = 0; b < fy.size(); ++b)
				{
					Eigen::Vector2d new_R = R + fx[a] * Lx * a1 + fy[b] * Ly * a2;
					for (int i = 0; i < n_sites(); ++i)
					{
						auto& R_i = real_space_coord(i);
						if ((new_R - R_i).norm() < std::pow(10., -12.))
							return true;
					}
				}
			return false;
		}
		
		vertex_t inverted_site(vertex_t i) const
		{
			double pi = 4. * std::atan(1.);
			Eigen::Rotation2D<double> rot(pi);
			auto& R = real_space_coord(i) - center;
			Eigen::Vector2d rot_R = rot * R + center;
			return site_at_position(rot_R);
		}
		
		vertex_t rotated_site(vertex_t i, double angle) const
		{
			double pi = 4. * std::atan(1.);
			Eigen::Rotation2D<double> rot(angle / 180. * pi);
			auto& R = real_space_coord(i) - center;
			Eigen::Vector2d rot_R = rot * R + center;
			return site_at_position(rot_R);
		}
		
		vertex_t reflected_v_site(vertex_t i) const
		{
			Eigen::Vector2d R = real_space_coord(i);
			R[0] = 2.*center[0] - R[0];
			return site_at_position(R);
		}
		
		vertex_t reflected_h_site(vertex_t i) const
		{
			Eigen::Vector2d R = real_space_coord(i);
			R[1] = 2.*center[1] - R[1];
			return site_at_position(R);
		}
		
		bool check_rotation_symmetry(double angle) const
		{
			double pi = 4. * std::atan(1.);
			Eigen::Rotation2D<double> rot(angle / 180. * pi);
			for (int i = 0; i < n_sites(); ++i)
			{
				auto& R = real_space_coord(i) - center;
				Eigen::Vector2d rot_R = rot * R + center;
				if (!is_lattice_site(rot_R))
					return false;
			}
			return true;
		}
		
		pair_vector_t generate_kekule_bonds(int i, int j, int alpha)
		{
			Eigen::Vector2d kek_base_1 = a1 + a2;
			Eigen::Vector2d kek_base_2 = 2.*a1 - a2;
			Eigen::Vector2d R_ij = i * kek_base_1 + j * kek_base_2;
			std::vector<int> r;
			r.push_back(site_at_position(R_ij + Eigen::Vector2d{-1.0, 0.}));
			r.push_back(site_at_position(R_ij + Eigen::Vector2d{-0.5, std::sqrt(3.)/2.}));
			r.push_back(site_at_position(R_ij + Eigen::Vector2d{0.5, std::sqrt(3.)/2.}));
			r.push_back(site_at_position(R_ij + Eigen::Vector2d{-0.5, -std::sqrt(3.)/2.}));
			r.push_back(site_at_position(R_ij + Eigen::Vector2d{0.5, -std::sqrt(3.)/2.}));
			r.push_back(site_at_position(R_ij + Eigen::Vector2d{1.0, 0.}));
			r.push_back(site_at_position(R_ij + Eigen::Vector2d{1.0, std::sqrt(3.)}));
			r.push_back(site_at_position(R_ij + Eigen::Vector2d{2.0, 0.}));
			r.push_back(site_at_position(R_ij + Eigen::Vector2d{1.0, -std::sqrt(3.)}));
			pair_vector_t kek_bonds;
			
			if (alpha == 0)
			{
				kek_bonds.push_back({r[1], r[2]});
				kek_bonds.push_back({r[2], r[1]});
				
				kek_bonds.push_back({r[0], r[3]});
				kek_bonds.push_back({r[3], r[0]});
				
				kek_bonds.push_back({r[4], r[5]});
				kek_bonds.push_back({r[5], r[4]});
			}
			else if (alpha == 1)
			{
				kek_bonds.push_back({r[0], r[1]});
				kek_bonds.push_back({r[1], r[0]});
				
				kek_bonds.push_back({r[2], r[5]});
				kek_bonds.push_back({r[5], r[2]});
				
				kek_bonds.push_back({r[3], r[4]});
				kek_bonds.push_back({r[4], r[3]});
			}
			else if (alpha == 2)
			{
				kek_bonds.push_back({r[2], r[6]});
				kek_bonds.push_back({r[6], r[2]});
				
				kek_bonds.push_back({r[5], r[7]});
				kek_bonds.push_back({r[7], r[5]});
				
				kek_bonds.push_back({r[4], r[9]});
				kek_bonds.push_back({r[9], r[4]});
			}
			return kek_bonds;
		}

		void print_sites() const
		{
			std::pair<vertex_it_t, vertex_it_t> vs = boost::vertices(*graph);

			std::copy(vs.first, vs.second,
				std::ostream_iterator<vertex_t>{std::cout, "\n"});
		}

		// Edges on graph = nearest neighbor bonds
		void print_bonds() const
		{
			std::pair<edge_it_t, edge_it_t> es = boost::edges(*graph);

			std::copy(es.first, es.second,
				std::ostream_iterator<edge_t>{std::cout, "\n"});
		}

		void print_distance_map() const
		{
			for (int i = 0; i < n_sites(); ++i)
				for (int j = i; j < n_sites(); ++j)
					std::cout << "d(" << i << ", " << j << ") = " << distance(i, j) << std::endl;
		}
	private:
		double min_distance(int i, int j)
		{
			auto& r_i = real_space_coord(i);
			auto& r_j = real_space_coord(j);
			auto R_Lx = Lx * a1, R_Ly = Ly * a2;
			
			double R = (r_i - r_j).norm();
			for (int n = -1; n <= 1; ++n)
				for (int m = -1; m <= 1; ++m)
				{
					double R_nm = (r_i - r_j + n * R_Lx + m * R_Ly).norm();
					R = std::min(R, R_nm);
				}
			return R;
		}
		
		void generate_distance_map()
		{
			distance_map.resize((n_sites() * (n_sites() - 1)) / 2, 0);
			int i = 0;
			for (int j = i; j < n_sites(); ++j)
			{
				double R_ij = min_distance(i, j);
				bool add = true;
				for (double R : distance_list)
					if (std::abs(R - R_ij) < 1E-10)
						add = false;
				if (add)
					distance_list.push_back(R_ij);
			}
			std::sort(distance_list.begin(), distance_list.end());
			for (int i = 0; i < n_sites(); ++i)
				for (int j = i+1; j < n_sites(); ++j)
				{
					double R_ij = min_distance(i, j);
					for (int n = 0; n < distance_list.size(); ++n)
					{
						if (std::abs(distance_list[n] - R_ij) < 1E-10)
							distance_map[(i*(2*n_sites()-1-i))/2 + j - i - 1] = n;
					}
				}
			//print_distance_map();
		}
	public:
		// Base vectors of Bravais lattice
		Eigen::Vector2d a1;
		Eigen::Vector2d a2;
		// Base vectors of reciprocal lattice
		Eigen::Vector2d b1;
		Eigen::Vector2d b2;
		// Center of inversion symmetry
		Eigen::Vector2d center;
		// Nearest neighbor vector_wick_base
		Eigen::Vector2d delta;
		int Lx;
		int Ly;
	private:
		graph_t* graph;
		int neighbor_dist;
		std::vector<int> distance_map;
		std::vector<double> distance_list;
		neighbor_map_t neighbor_maps;
		bond_map_t bond_maps;
		real_space_map_t real_space_map;
		point_map_t symmetry_points;
};
