#pragma once
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <Eigen/Dense>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include "lattice.h"


struct hex_honeycomb
{
	//typedef lattice::graph_t graph_t;
	typedef boost::adjacency_list<boost::setS, boost::vecS,
		boost::undirectedS> graph_t;

	int L;
	std::vector<Eigen::Vector2d> real_space_map;
	std::vector<std::pair<int, int>> coord_map;
	std::vector<std::array<int, 3>> vertex_list;
	std::map<std::array<int, 3>, int> tuple_map;
	// Base vectors of Bravais lattice
	Eigen::Vector2d a1;
	Eigen::Vector2d a2;
	Eigen::Vector2d c1;
	Eigen::Vector2d c2;
	Eigen::Vector2d c3;
	// Base vectors of reciprocal lattice
	Eigen::Vector2d b1;
	Eigen::Vector2d b2;
	// Vector to second sublattice point
	Eigen::Vector2d delta;
	// Center of inversion symmetry
	Eigen::Vector2d center;
	double pi = 4. * std::atan(1.);

	hex_honeycomb(int L_ = 6)
		: L(L_),
			a1(3./2.*L, std::sqrt(3.)/2.*L), a2(3./2.*L, -std::sqrt(3.)/2.*L),
			c1(1./2., std::sqrt(3.)/2.), c2(-1., 0.), c3(1./2., -std::sqrt(3.)/2.),
			delta(1./2., std::sqrt(3.)/2.)
	{
		b1 = Eigen::Vector2d(2.*pi/3., 2.*pi/std::sqrt(3.));
		b2 = Eigen::Vector2d(2.*pi/3., -2.*pi/std::sqrt(3.));
		center = Eigen::Vector2d(0., 0.);
	}
	
	std::array<int, 3> neighbor_site(std::array<int, 3> site, int type)
	{
		int x = site[0], y = site[1], z = site[2];
		//even sublattice
		if (x + y + z == 1)
		{
			if (type == 0)
			{
				if (x == L)
					return {-L+1, y+L, z+L};
				else
					return {x+1, y, z};
			}
			else if (type == 1)
			{
				if (y == L)
					return {x+L, -L+1, z+L};
				else				
					return {x, y+1, z};
			}
			else if (type == 2)
			{
				if (z == L)
					return {x+L, y+L, -L+1};
				else				
					return {x, y, z+1};
			}
		}
		//odd sublattice
		else
		{
			if (type == 0)
			{
				if (x == -L+1)
					return {L, y-L, z-L};
				else
					return {x-1, y, z};
			}
			else if (type == 1)
			{
				if (y == -L+1)
					return {x-L, L, z-L};
				else				
					return {x, y-1, z};
			}
			else if (type == 2)
			{
				if (z == -L+1)
					return {x-L, y-L, L};
				else				
					return {x, y, z-1};
			}
		}
	}

	graph_t* graph()
	{
		int n_sites = 6 * L * L;
		graph_t* g = new graph_t(n_sites);
		add_edges(g);
		return g;
	}
	
	int distance(std::array<int, 3> s1, std::array<int, 3> s2)
	{
		int u1 = s1[0];
		int v1 = s1[1];
		int w1 = s1[2];
		int u2 = s2[0];
		int v2 = s2[1];
		int w2 = s2[2];
		int d0 = std::abs(u2 - u1) + std::abs(v2 - v1) + std::abs(w2 - w1);
		int d1 = std::abs(u2 - u1 + 2 * L) + std::abs(v2 - v1 - L) + std::abs(w2 - w1 - L);
		int d2 = std::abs(u2 - u1 - 2 * L) + std::abs(v2 - v1 + L) + std::abs(w2 - w1 + L);
		int d3 = std::abs(u2 - u1 - L) + std::abs(v2 - v1 + 2 * L) + std::abs(w2 - w1 - L);
		int d4 = std::abs(u2 - u1 + L) + std::abs(v2 - v1 - 2 * L) + std::abs(w2 - w1 + L);
		int d5 = std::abs(u2 - u1 - L) + std::abs(v2 - v1 - L) + std::abs(w2 - w1 + 2 * L);
		int d6 = std::abs(u2 - u1 + L) + std::abs(v2 - v1 + L) + std::abs(w2 - w1 - 2 * L);
		
		return std::min({ d0, d1, d2, d3, d4, d5, d6 });
	}
	
	void add_edges(graph_t* g)
	{
		typedef std::pair<int, int> edge_t;
		int n_vertices = boost::num_vertices(*g);
		
		vertex_list.resize(n_vertices);
		real_space_map.resize(n_vertices);
		int n_even = 0, n_odd = 1;
		for (int i = -L + 1; i <= L; ++i)
			for (int j = -L + 1; j <= L; ++j)
				for (int k = -L + 1; k <= L; ++k)
				{
					int sum = i + j + k;
					if (sum == 1)
					{
						vertex_list[n_even] = {i, j, k};
						tuple_map[{i, j, k}] = n_even;
						real_space_map[n_even] = Eigen::Vector2d{i*c1 + j*c2 + k*c3};
						n_even += 2;
					}
					else if (sum == 2)
					{
						vertex_list[n_odd] = {i, j, k};
						tuple_map[{i, j, k}] = n_odd;
						real_space_map[n_odd] = Eigen::Vector2d{i*c1 + j*c2 + k*c3};
						n_odd += 2;
					}
				}
		for (auto i = 0; i < vertex_list.size(); ++i)
			for (auto j = 0; j < vertex_list.size(); ++j)
				if (distance(vertex_list[i], vertex_list[j]) == 1)
				{
					boost::add_edge(i, j, *g);
					boost::add_edge(j, i, *g);
				}
	}

	void generate_maps(lattice& l)
	{
		l.a1 = a1;
		l.a2 = a2;
		l.b1 = b1;
		l.b2 = b2;
		l.center = center;
		l.Lx = L;
		l.Ly = L;
		
		//Symmetry points
		std::map<std::string, Eigen::Vector2d> points;

		points["K"] = {2.*pi/3., 2.*pi/3./std::sqrt(3.)};
		points["Kp"] = {2.*pi/3., -2.*pi/3./std::sqrt(3.)};
		points["Gamma"] = {0., 0.};
		points["M"] = {2.*pi/3., 0.};
		l.add_symmetry_points(points);

		//Site maps
		l.generate_neighbor_map("nearest neighbors", [&]
			(lattice::vertex_t i, lattice::vertex_t j) {
			return l.distance(i, j) == 1; });
		l.generate_bond_map("nearest neighbors", [&]
			(lattice::vertex_t i, lattice::vertex_t j)
			{ return l.distance(i, j) == 1; });
		l.generate_bond_map("single_d1_bonds", [&]
			(lattice::vertex_t i, lattice::vertex_t j)
			{ return l.distance(i, j) == 1 && i < j; });
		l.generate_bond_map("d3_bonds", [&]
			(lattice::vertex_t i, lattice::vertex_t j)
			{ return l.distance(i, j) == 3; });
		
		l.generate_bond_map("t3_bonds", [&]
			(lattice::pair_vector_t& list)
		{
			for (int i = -L + 1; i <= L; ++i)
				for (int j = -L + 1; j <= L; ++j)
					for (int k = -L + 1; k <= L; ++k)
					{
						int sum = i + j + k;
						std::array<int, 3> t = {i, j, k};
						if (sum == 1)
						{
							int n = tuple_map.at(t);
							auto ns_1 = neighbor_site(t, 0);
							auto ns_2 = neighbor_site(ns_1, 1);
							auto ns_3 = neighbor_site(ns_2, 2);
							list.push_back({tuple_map.at(ns_3), n});
							list.push_back({n, tuple_map.at(ns_3)});
							
							ns_1 = neighbor_site(t, 1);
							ns_2 = neighbor_site(ns_1, 2);
							ns_3 = neighbor_site(ns_2, 0);
							list.push_back({tuple_map.at(ns_3), n});
							list.push_back({n, tuple_map.at(ns_3)});
							
							ns_1 = neighbor_site(t, 2);
							ns_2 = neighbor_site(ns_1, 0);
							ns_3 = neighbor_site(ns_2, 1);
							list.push_back({tuple_map.at(ns_3), n});
							list.push_back({n, tuple_map.at(ns_3)});
						}
					}
		});
		
		l.generate_bond_map("kekule", [&]
			(lattice::pair_vector_t& list)
		{});
		
		l.generate_bond_map("kekule_2", [&]
			(lattice::pair_vector_t& list)
		{});
		
		l.generate_bond_map("kekule_3", [&]
			(lattice::pair_vector_t& list)
		{});
		
		l.generate_bond_map("single_kekule", [&]
			(lattice::pair_vector_t& list)
		{
			for (auto& bond : l.bonds("kekule"))
				if (bond.first < bond.second)
					list.push_back(bond);
		});
		
		l.generate_bond_map("single_kekule_2", [&]
			(lattice::pair_vector_t& list)
		{
			for (auto& bond : l.bonds("kekule_2"))
				if (bond.first < bond.second)
					list.push_back(bond);
		});
		
		l.generate_bond_map("single_kekule_3", [&]
			(lattice::pair_vector_t& list)
		{
			for (auto& bond : l.bonds("kekule_3"))
				if (bond.first < bond.second)
					list.push_back(bond);
		});
		
		l.generate_bond_map("chern", [&]
		(lattice::pair_vector_t& list)
		{});
		
		l.generate_bond_map("chern_2", [&]
		(lattice::pair_vector_t& list)
		{});
		
		/*
		l.generate_bond_map("nn_bond_1", [&]
		(lattice::pair_vector_t& list)
		{
			for (int j = 0; j < Ly; ++j)
				for (int i = 0; i < Lx; ++i)
					for (int d = 0; d < 6; d+=2)
					{
						int n = j * 6 * Lx + i * 6 + d;
						list.push_back({neighbor_site(i, j, d, 0), n});
					}
		});
		
		l.generate_bond_map("nn_bond_2", [&]
		(lattice::pair_vector_t& list)
		{
			for (int j = 0; j < Ly; ++j)
				for (int i = 0; i < Lx; ++i)
					for (int d = 0; d < 6; d+=2)
					{
						int n = j * 6 * Lx + i * 6 + d;
						list.push_back({neighbor_site(i, j, d, 2), n});
					}
		});
		
		l.generate_bond_map("nn_bond_3", [&]
		(lattice::pair_vector_t& list)
		{
			for (int j = 0; j < Ly; ++j)
				for (int i = 0; i < Lx; ++i)
					for (int d = 0; d < 6; d+=2)
					{
						int n = j * 6 * Lx + i * 6 + d;
						list.push_back({neighbor_site(i, j, d, 1), n});
					}
		});
		
		l.generate_bond_map("inv_nn_bond_1", [&]
		(lattice::pair_vector_t& list)
		{
			for (int j = 0; j < Ly; ++j)
				for (int i = 0; i < Lx; ++i)
					for (int d = 0; d < 6; d+=2)
					{
						int n = j * 6 * Lx + i * 6 + d;
						list.push_back({l.inverted_site(neighbor_site(i, j, d, 0)), l.inverted_site(n)});
					}
		});
		
		l.generate_bond_map("inv_nn_bond_2", [&]
		(lattice::pair_vector_t& list)
		{
			for (int j = 0; j < Ly; ++j)
				for (int i = 0; i < Lx; ++i)
					for (int d = 0; d < 6; d+=2)
					{
						int n = j * 6 * Lx + i * 6 + d;
						list.push_back({l.inverted_site(neighbor_site(i, j, d, 2)), l.inverted_site(n)});
					}
		});
		
		l.generate_bond_map("inv_nn_bond_3", [&]
		(lattice::pair_vector_t& list)
		{
			for (int j = 0; j < Ly; ++j)
				for (int i = 0; i < Lx; ++i)
					for (int d = 0; d < 6; d+=2)
					{
						int n = j * 6 * Lx + i * 6 + d;
						list.push_back({l.inverted_site(neighbor_site(i, j, d, 1)), l.inverted_site(n)});
					}
		});
		*/
	}
};
