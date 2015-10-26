#pragma once
#include <iostream>
#include <array>
#include <vector>
#include <map>
#include <string>
#include <functional>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/named_function_params.hpp>
#include <boost/graph/visitors.hpp>
#include "boost/multi_array.hpp"
#include "honeycomb.h"

class lattice
{
	public:
		typedef boost::adjacency_list<boost::setS, boost::vecS,
				boost::undirectedS> graph_t;
		typedef graph_t::vertex_descriptor vertex_t;
		typedef graph_t::vertex_iterator vertex_it_t;
		typedef graph_t::edge_descriptor edge_t;
		typedef graph_t::edge_iterator edge_it_t;
		typedef boost::multi_array<int, 2> multi_array_t;
		typedef std::vector<std::vector<int>> nested_vector_t;
		typedef std::map<std::string, nested_vector_t> neighbor_map_t;

		lattice()
			: graph(0) {}
		~lattice() { delete graph; }

		template<typename T>
		void generate_graph(T& generator)
		{
			delete graph;
			graph = generator.graph();
			generate_distance_map();
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

		int n_sites() const
		{
			return boost::num_vertices(*graph);
		}

		int n_bonds() const
		{
			return boost::num_edges(*graph);
		}

		int max_distance() const
		{
			return max_dist;
		}

		int distance(vertex_t i, vertex_t j) const
		{
			return distance_map[i][j];
		}

		const std::vector<int>& neighbors(vertex_t site,
			const std::string& name) const
		{
			return neighbor_maps.at(name)[site];
		}

		//TODO: generalize as vertex property on graph
		int sublattice(vertex_t site) const
		{
			return site % 2;
		}

		double parity(vertex_t site) const
		{
			return (site % 2 == 0) ? 1.0 : -1.0;
		}

		void print_sites() const
		{
			std::pair<vertex_it_t, vertex_it_t> vs = boost::vertices(*graph);

			std::copy(vs.first, vs.second,
				std::ostream_iterator<vertex_t>{std::cout, "\n"});
		}

		void print_bonds() const
		{
			std::pair<edge_it_t, edge_it_t> es = boost::edges(*graph);

			std::copy(es.first, es.second,
				std::ostream_iterator<edge_t>{std::cout, "\n"});
		}

		void print_distance_map() const
		{
			for (int i = 0; i < n_sites(); ++i)
			{
				for (int j = 0; j < n_sites(); ++j)
				{
					std::cout << "d(" << i << ", " << j << ") = "
						<< distance_map[i][j] << std::endl;
				}
			}
		}
	private:
		void generate_distance_map()
		{
			distance_map.resize(boost::extents[n_sites()][n_sites()]);

			for (int i = 0; i < n_sites(); ++i)
			{
				boost::breadth_first_search(*graph, i, boost::visitor(
					boost::make_bfs_visitor(boost::record_distances(
					&distance_map[i][0], boost::on_tree_edge{}))));
			}
			max_dist = *std::max_element(distance_map.origin(),
							distance_map.origin() + distance_map.num_elements());
		}
	private:
		graph_t* graph;
		int neighbor_dist;
		int max_dist;
		multi_array_t distance_map;
		neighbor_map_t neighbor_maps;
};
