#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <thrust/host_vector.h>
#include <unordered_map>
#include <algorithm>

using std::vector,std::unordered_map,pair;

class csr_graph
{
public :
	size_t num_node;
	size_t *node_id;
	size_t *offset;
	double *weight;
   __global__ csr_graph(std::string);
   __device__ ~csr_graph();
};

__global__ csr_graph::csr_graph(std::string network_file,bool have_weight){

	std::ifstream embFile("edges.csv");
	
	if (embFile.is_open()){

		size_t x,y;
		

		if(have_weight){
			double weight;
			unordered_map<vector<pair<size_t,weight>>> adj_list;
			while(embFile>>x>>y>>weight){
				auto pos = lower_bound(adj_list[x].begin(),adj_list[x].end(),
					std::make_pair(y,weight))
				adj_list[x].insert(pos,y);
			}

		}
		else{
			unordered_map<vector<size_t>> adj_list;
			while(embFile>>x>>y){
				auto pos = lower_bound(adj_list[x].begin(),adj_list[x].end(),y)
				adj_list[x].insert(pos,y);
			}
		}
		embFile.close();
	}
	else {
	    exit(0);
	}


}

#include <sstream>

int main(int argc, char const *argv[]){

	// std::string a("123 a \n 456 b ");
	// std::istringstream out(a);

	double x,y;

	std::ifstream embFile("edges.csv");
	if (embFile.is_open()){
	    // std::string edge;
	    // std::stringstream edge2pair;
		while(embFile>>x>>y){

			std::cout<<x<<' '<<y<<std::endl;

		}
	    embFile.close();
	}
	else {
	    exit(0);
	}



	return 0;
}




