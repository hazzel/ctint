#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include "lattice.h"


struct tilted_honeycomb
{
	//typedef lattice::graph_t graph_t;
	typedef boost::adjacency_list<boost::setS, boost::vecS,
		boost::undirectedS> graph_t;

	int Lx;
	int Ly;
	std::vector<Eigen::Vector2d> real_space_map;
	// Base vectors of Bravais lattice
	Eigen::Vector2d a1;
	Eigen::Vector2d a2;
	Eigen::Vector2d d1;
	Eigen::Vector2d d2;
	// Base vectors of reciprocal lattice
	Eigen::Vector2d b1;
	Eigen::Vector2d b2;
	// Vector to second sublattice point
	std::vector<Eigen::Vector2d> delta;
	// Center of inversion symmetry
	Eigen::Vector2d center;
	double pi = 4. * std::atan(1.);

	tilted_honeycomb(int Lx_ = 6, int Ly_ = 6)
		: Lx(Lx_), Ly(Ly_),
			a1(3., 0.), a2(3./2., -3.*std::sqrt(3.)/2.),
			d1(3./2., std::sqrt(3.)/2.), d2(3./2., -std::sqrt(3.)/2.)
	{
		b1 = Eigen::Vector2d(2.*pi/3., 2.*pi/std::sqrt(3.));
		b2 = Eigen::Vector2d(2.*pi/3., -2.*pi/std::sqrt(3.));
		delta.push_back({1./2., -std::sqrt(3.)/2.});
		delta.push_back({1./2., std::sqrt(3.)/2.});
		delta.push_back({-1., 0.});
		center = Eigen::Vector2d(1., 0.);
	}
	
	int neighbor_site(int i, int j, int d, int type)
	{
		//int i = site % (6 * Lx) / 6, j = site / (6 * Lx),  n_vertices = 6 * Lx * Ly;
		int site = j * 6 * Lx + i * 6 + d;
		int n_vertices = 6 * Lx * Ly;
		if (type == 0)
		{
			if (d == 0)
				return (site + 6 * Lx * (Ly-1) + 5 + n_vertices) % n_vertices;
			else if (d == 2)
				return (site + 1 + n_vertices) % n_vertices;
			else if (d == 4)
			{
				if (i == Lx - 1)
					return (site - 6 * Lx + 3 + n_vertices) % n_vertices;
				else
					return (site + 3 + n_vertices) % n_vertices;
			}
		}
		else if (type == 1)
		{
			if (d == 0)
			{
				if (i == 0)
					return (site + 6 * Lx - 3 + n_vertices) % n_vertices;
				else
					return (site - 3 + n_vertices) % n_vertices;
			}
			else if (d == 2)
				return (site - 1 + n_vertices) % n_vertices;
			else if (d == 4)
				return (site + 1 + n_vertices) % n_vertices;
		}
		else if (type == 2)
		{
			if (d == 0)
				return (site + 1 + n_vertices) % n_vertices;
			else if (d == 2)
				return (site + 3 + n_vertices) % n_vertices;
			else if (d == 4)
				return (site + 6 * Lx - 1 + n_vertices) % n_vertices;
		}
	}

	graph_t* graph()
	{
		int n_sites = 6 * Lx * Ly;
		graph_t* g = new graph_t(n_sites);
		add_edges(g);
		return g;
	}

	void add_edges(graph_t* g)
	{
		typedef std::pair<int, int> edge_t;
		for (int j = 0; j < Ly; ++j)
			for (int i = 0; i < Lx; ++i)
				for (int d = 0; d < 6; d+=2)
				{
					int n = j * 6 * Lx + i * 6 + d;
					
					boost::add_edge(n, neighbor_site(i, j, d, 0), *g);
					boost::add_edge(neighbor_site(i, j, d, 0), n, *g);
					
					boost::add_edge(n, neighbor_site(i, j, d, 1), *g);
					boost::add_edge(neighbor_site(i, j, d, 1), n, *g);
					
					boost::add_edge(n, neighbor_site(i, j, d, 2), *g);
					boost::add_edge(neighbor_site(i, j, d, 2), n, *g);
					
					real_space_map.push_back(Eigen::Vector2d{i * a1 + j * a2 + d/2 * d2});
					real_space_map.push_back(Eigen::Vector2d{i * a1 + j * a2 + d/2 * d2 + delta[d/2]});
				}
	}

	Eigen::Vector2d closest_k_point(const Eigen::Vector2d& K)
	{
		Eigen::Vector2d x = {0., 0.};
		double dist = (x - K).norm();
		for (int i = 0; i < Lx; ++i)
			for (int j = 0; j < Ly; ++j)
			{
				Eigen::Vector2d y = static_cast<double>(i) / static_cast<double>(Lx)
					* b1 + static_cast<double>(j) / static_cast<double>(Ly) * b2;
				double d = (y - K).norm();
				if (d < dist)
				{
					x = y;
					dist = d;
				}
			}
		return x;
	}

	void generate_maps(lattice& l)
	{
		l.a1 = a1;
		l.a2 = a2;
		l.b1 = b1;
		l.b2 = b2;
		l.center = center;
		l.Lx = Lx;
		l.Ly = Ly;
	
		//Symmetry points
		std::map<std::string, Eigen::Vector2d> points;

		points["K"] = closest_k_point({2.*pi/3., 2.*pi/3./std::sqrt(3.)});
		points["Kp"] = closest_k_point({2.*pi/3., -2.*pi/3./std::sqrt(3.)});
		points["Gamma"] = closest_k_point({0., 0.});
		points["M"] = closest_k_point({2.*pi/3., 0.});
		points["q"] = closest_k_point(b1 / Lx);
		l.add_symmetry_points(points);
		
		/*
		for (int i = 0; i < Lx; ++i)
			for (int j = 0; j < Ly; ++j)
			{
				Eigen::Vector2d y = static_cast<double>(i) / static_cast<double>(Lx)
					* b1 + static_cast<double>(j) / static_cast<double>(Ly) * b2;
				//std::cout << i << ", " << j << " : (" << y[0] << ", " << y[1] << ")" << std::endl;
				std::cout << y[0] << " " << y[1] << std::endl;
			}
		std::cout << "K point: (" << 2.*pi/3. << ", " << 2.*pi/3./std::sqrt(3.) << ")" << std::endl;
		std::cout << "closest to K point: (" << points["K"][0] << ", " << points["K"][1] << ")" << std::endl;
		*/

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
		{});
				
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
