#include "./read_data.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <vector>

using std::min;
using std::vector;
using std::unordered_map;
using std::pair;
using std::make_pair;
using std::cout;
using std::endl;


struct compareByTwoKey{
    bool operator()(const pair<size_t,double> &t1, const pair<size_t, double> &t2){

        if(t1.first<t2.first)
            return true;
        else
            return false;
    }
};



unordered_map<size_t, vector<pair<size_t,double > > >  
read_graph(std::string network_file, size_t & num_edge, const bool have_weight){


	std::ifstream embFile(network_file);
	
	if (embFile.is_open()){

		size_t x,y;
		auto rule_compare = compareByTwoKey();
		unordered_map<size_t,vector<pair<size_t,double > > > adj_list;
		if(have_weight){

			double weight;
			while(embFile>>x>>y>>weight){
				pair<size_t,double> ele = make_pair(y,1);
				auto pos = lower_bound(adj_list[x].begin(),adj_list[x].end(),
					ele,rule_compare);
				adj_list[x].insert(pos,ele);
				++num_edge;
			}

		}
		else{
			while(embFile>>x>>y){
				pair<size_t,double> ele = make_pair(y,1);
				auto pos = lower_bound(adj_list[x].begin(), adj_list[x].end(),
									ele, rule_compare);
				adj_list[x].insert(pos,ele);
				++num_edge;
			}
		}
		embFile.close();

		return adj_list;
	}
	else {

		cout<<"read error";
	    exit(0);
	}

}


void adjList2CSR(
	unordered_map<size_t, vector<pair<size_t,double> > > & adj_list,
	double *weights, size_t *col_id, size_t *row_shift ){

	size_t i = 0;
	vector<pair<size_t, double> >  node_neighbors;

	row_shift[0] = 0;

	for(size_t row_id=0; row_id < adj_list.size(); ++row_id){

		node_neighbors = adj_list[row_id];

		for(auto adj_node : node_neighbors){
			weights[i] = adj_node.second;
			col_id[i] = adj_node.first;
			++i;
		}
		row_shift[row_id+1] = i;
	}

}