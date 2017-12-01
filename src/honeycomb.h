#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include "lattice.h"


struct honeycomb
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
	// Base vectors of reciprocal lattice
	Eigen::Vector2d b1;
	Eigen::Vector2d b2;
	// Vector to second sublattice point
	Eigen::Vector2d delta;
	// Center of inversion symmetry
	Eigen::Vector2d center;
	double pi = 4. * std::atan(1.);

	honeycomb(int Lx_ = 6, int Ly_ = 6)
		: Lx(Lx_), Ly(Ly_),
			a1(3./2., std::sqrt(3.)/2.), a2(3./2., -std::sqrt(3.)/2.),
			delta(1./2., std::sqrt(3.)/2.)
	{
		b1 = Eigen::Vector2d(2.*pi/3., 2.*pi/std::sqrt(3.));
		b2 = Eigen::Vector2d(2.*pi/3., -2.*pi/std::sqrt(3.));
		center = Eigen::Vector2d(1., 0.);
	}
	
	int neighbor_site(int site, int type)
	{
		int i = site % (2 * Lx) / 2, n_vertices = 2 * Lx * Ly;
		if (type == 0)
			return (site + 1 + n_vertices) % n_vertices;
		else if (type == 1)
		{
			if (i == 0)
				return (site + 2*Lx - 1 + n_vertices) % n_vertices;
			else
				return (site - 1 + n_vertices) % n_vertices;
		}
		else if (type == 2)
		{
			if (i == 0)
				return (site + 4*Lx - 1 + n_vertices) % n_vertices;
			else
				return (site + 2*Lx - 1 + n_vertices) % n_vertices;
		}
	}

	graph_t* graph()
	{
		int n_sites = 2 * Lx * Ly;
		graph_t* g = new graph_t(n_sites);
		add_edges(g);
		return g;
	}

	void add_edges(graph_t* g)
	{
		typedef std::pair<int, int> edge_t;
		for (int j = 0; j < Ly; ++j)
			for (int i = 0; i < Lx; ++i)
			{
				int n = j * 2 * Lx + i * 2;
				
				boost::add_edge(n, neighbor_site(n, 0), *g);
				boost::add_edge(neighbor_site(n, 0), n, *g);
				
				boost::add_edge(n, neighbor_site(n, 1), *g);
				boost::add_edge(neighbor_site(n, 1), n, *g);
				
				boost::add_edge(n, neighbor_site(n, 2), *g);
				boost::add_edge(neighbor_site(n, 2), n, *g);
				
				real_space_map.push_back(Eigen::Vector2d{i * a1 + j * a2});
				real_space_map.push_back(Eigen::Vector2d{i * a1 + j * a2 + delta});
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
		{
			int N = l.n_sites();
			
			//list.push_back({0, 7});
			//list.push_back({7, 0});
			//list.push_back({2, 11});
			//list.push_back({11, 2});
			//list.push_back({1, 6});
			//list.push_back({6, 1});
			
			for (int j = 0; j < Ly; ++j)
			{
				for (int i = 0; i < Lx; ++i)
				{
					int n = j * 2 * Lx + i * 2;
					list.push_back({n, (n + 2 * Lx + 1 + N) % N});
					list.push_back({(n + 2 * Lx + 1 + N) % N, n});
					
					list.push_back({n, (n - 2 * Lx + 1 + N) % N});
					list.push_back({(n - 2 * Lx + 1 + N) % N, n});
					
					if (i == 0 || i == 1)
					{
						list.push_back({n, (n + 4 * Lx - 3 + N) % N});
						list.push_back({(n + 4 * Lx - 3 + N) % N, n});
					}
					else
					{
						list.push_back({n, (n + 2 * Lx - 3 + N) % N});
						list.push_back({(n + 2 * Lx - 3 + N) % N, n});
					}
				}
			}
		});
		
		l.generate_bond_map("kekule", [&]
			(lattice::pair_vector_t& list)
		{
			int N = l.n_sites();
			if (Lx == 2 && Ly == 2)
			{
				list = {{0, 1}, {1, 0}, {4, 7}, {7, 4}, {2, 5}, {5, 2}};
				return;
			}
			
			int j_type = 0, i_type = 0;
			for (int j = 0; j < Ly; ++j)
			{
				i_type = j_type;
				for (int i = 0; i < Lx; ++i)
				{
					int n = j * 2 * Lx + i * 2;
					list.push_back({n, neighbor_site(n, i_type)});
					list.push_back({neighbor_site(n, i_type), n});
					i_type = (i_type - 1 + 3) % 3;
				}
				j_type = (j_type + 1 + 3) % 3;
			}
		});
		
		l.generate_bond_map("kekule_2", [&]
			(lattice::pair_vector_t& list)
		{
			int N = l.n_sites();
			if (Lx == 2 && Ly == 2)
			{
				list = {{1, 2}, {2, 1}, {4, 5}, {5, 4}, {0, 7}, {7, 0}};
				return;
			}

			int j_type = 2, i_type = 0;
			for (int j = 0; j < Ly; ++j)
			{
				i_type = j_type;
				for (int i = 0; i < Lx; ++i)
				{
					int n = j * 2 * Lx + i * 2;
					list.push_back({n, neighbor_site(n, i_type)});
					list.push_back({neighbor_site(n, i_type), n});
					i_type = (i_type - 1 + 3) % 3;
				}
				j_type = (j_type + 1 + 3) % 3;
			}
		});
		
		l.generate_bond_map("kekule_3", [&]
			(lattice::pair_vector_t& list)
		{
			int N = l.n_sites();
			if (Lx == 2 && Ly == 2)
			{
				list = {{2, 3}, {3, 2}, {5, 6}, {6, 5}, {3, 4}, {4, 3}};
				return;
			}

			int j_type = 1, i_type = 0;
			for (int j = 0; j < Ly; ++j)
			{
				i_type = j_type;
				for (int i = 0; i < Lx; ++i)
				{
					int n = j * 2 * Lx + i * 2;
					list.push_back({n, neighbor_site(n, i_type)});
					list.push_back({neighbor_site(n, i_type), n});
					i_type = (i_type - 1 + 3) % 3;
				}
				j_type = (j_type + 1 + 3) % 3;
			}
		});
		
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
		{
			int N = l.n_sites();
			if (Lx == 2 && Ly == 2)
			{
				list = {{0, 4}, {4, 2}, {2, 0}, {2, 6}, {6, 4}, {0, 6}};
				return;
			}

			for (int i = 0; i < Lx; ++i)
				for (int j = 0; j < Ly; ++j)
				{
					int x = 2 * i + 2 * Lx * j;
					int y = x + 2 * Lx;
					list.push_back({x % N, y % N});
	
					if (i == Lx - 1)
						y = x - 2 * (Lx - 1);
					else
						y = x + 2;
					list.push_back({(y+N) % N, x % N});
				
					if (i == Lx - 1)
					{
						x = 2 * Lx * j;
						y = x + 4 * Lx - 2;
					}
					else
					{
						x = 2 * (i+1) + 2 * Lx * j;
						y = x + 2 * (Lx - 1);
					}
					list.push_back({y % N, x % N});
				}
		});
		
		l.generate_bond_map("chern_2", [&]
		(lattice::pair_vector_t& list)
		{
			int N = l.n_sites();
			if (Lx == 2 && Ly == 2)
			{
				list = {{1, 5}, {5, 3}, {3, 1}, {3, 7}, {7, 5}, {1, 7}};
				return;
			}

			for (int i = 0; i < Lx; ++i)
				for (int j = 0; j < Ly; ++j)
				{
					int x = 2 * i + 1 + 2 * Lx * j;
					int y;
					
					if (i == 0)
						y = x + 4 * Lx - 2;
					else
						y = x + 2 * (Lx - 1);
					list.push_back({x % N, y % N});
					
					y = x + 2 * Lx;
					list.push_back({y % N, x % N});
					
					x = (2 * i + 1 + 2 * Lx * (j + 1)) % N;
					if (i == 0)
						y = x + 2 * (Lx - 1);
					else
						y = x - 2;
					list.push_back({y % N, x % N});
				}
		});
		
		l.generate_bond_map("nn_bond_1", [&]
		(lattice::pair_vector_t& list)
		{
			for (int j = 0; j < Ly; ++j)
				for (int i = 0; i < Lx; ++i)
				{
					int n = j * 2 * Lx + i * 2;
					list.push_back({neighbor_site(n, 0), n});
				}
		});
		
		l.generate_bond_map("nn_bond_2", [&]
		(lattice::pair_vector_t& list)
		{
			for (int j = 0; j < Ly; ++j)
				for (int i = 0; i < Lx; ++i)
				{
					int n = j * 2 * Lx + i * 2;
					list.push_back({neighbor_site(n, 2), n});
				}
		});
		
		l.generate_bond_map("nn_bond_3", [&]
		(lattice::pair_vector_t& list)
		{
			for (int j = 0; j < Ly; ++j)
				for (int i = 0; i < Lx; ++i)
				{
					int n = j * 2 * Lx + i * 2;
					list.push_back({neighbor_site(n, 1), n});
				}
		});
		
		l.generate_bond_map("inv_nn_bond_1", [&]
		(lattice::pair_vector_t& list)
		{
			for (int j = 0; j < Ly; ++j)
				for (int i = 0; i < Lx; ++i)
				{
					int n = j * 2 * Lx + i * 2;
					list.push_back({l.inverted_site(neighbor_site(n, 0)), l.inverted_site(n)});
				}
		});
		
		l.generate_bond_map("inv_nn_bond_2", [&]
		(lattice::pair_vector_t& list)
		{
			for (int j = 0; j < Ly; ++j)
				for (int i = 0; i < Lx; ++i)
				{
					int n = j * 2 * Lx + i * 2;
					list.push_back({l.inverted_site(neighbor_site(n, 2)), l.inverted_site(n)});
				}
		});
		
		l.generate_bond_map("inv_nn_bond_3", [&]
		(lattice::pair_vector_t& list)
		{
			for (int j = 0; j < Ly; ++j)
				for (int i = 0; i < Lx; ++i)
				{
					int n = j * 2 * Lx + i * 2;
					list.push_back({l.inverted_site(neighbor_site(n, 1)), l.inverted_site(n)});
				}
		});
	}
};
