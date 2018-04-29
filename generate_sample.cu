#include "generate_sample.h"
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <assert.h>
#include <time.h>  
#include "tool.h"


using std::vector;
using std::unordered_map;
using std::pair;
using std::make_pair;
using std::cout;
using std::endl;



static void HandleError(cudaError_t err, const char *file,int line){    
    if (err != cudaSuccess){        
        printf( "%s in %s at line %d\n", cudaGetErrorString(err),file,line);        
        exit(EXIT_FAILURE);
    }
}


#define HANDLE_ERROR(err)  (HandleError(err, __FILE__, __LINE__ ))



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

	size_t i=0;
	vector<pair<size_t, double> >  node_neighbors;

	for(size_t row_id=0; row_id<adj_list.size(); ++row_id){

		node_neighbors = adj_list[row_id];

		for(auto adj_node : node_neighbors){
			weights[i] = adj_node.second;
			col_id[i] = adj_node.first;
			++i;
		}
		row_shift[row_id+1] = i;
	}

}

size_t * init_graph(){

	size_t num_edge=0;
	auto adj_list = read_graph("edges.csv",num_edge,false);


	double * weights = new double[num_edge];

	size_t * col_id = new size_t[num_edge];
	size_t * row_shift = new size_t[adj_list.size()+1];
	row_shift[0] = 0;


	adjList2CSR(adj_list,weights,col_id,row_shift);


	csr_graph * host_g = new csr_graph(col_id,row_shift,weights, adj_list.size(),num_edge,1.0,1.0);
	
	

}


size_t * get_samples_batch(csr_graph * g, const size_t batchsize, const size_t len, const size_t *nodes){

	size_t *host_walks = new size_t[batchsize*len];

	size_t *dev_walks = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&dev_walks, len * batchsize * sizeof(size_t)));

	curandState *devStates;
	HANDLE_ERROR(cudaMalloc((void **)&devStates, batchsize * sizeof(curandState)));

	size_t blocks = min(batchsize,(size_t)512);

	sample_generator<<<blocks,1>>>(dev_walks, g, devStates,
							unsigned(time(NULL)), nodes, batchsize, len);

	HANDLE_ERROR(
		cudaMemcpy(	host_walks,
					dev_walks,
					sizeof(size_t)*batchsize*len,
					cudaMemcpyDeviceToHost)); 

	cudaFree(dev_walks);
	cudaFree(devStates);

	return host_walks;
}


size_t * get_samples_epoch(csr_graph * g, const size_t len){

	size_t batchsize = min((size_t)512,g->num_node);

	size_t *host_walks = new size_t[(g->num_node)*len];

	size_t *dev_walks = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&dev_walks, len * batchsize * sizeof(size_t)));

	curandState *devStates;
	HANDLE_ERROR(cudaMalloc((void **)&devStates, batchsize * sizeof(curandState)));

	size_t *nodes = new size_t[batchsize];

	for(size_t i=0;i < g->num_node; i+=batchsize){

		for(size_t j=0;j<batchsize;++j)
			nodes[j] = i+j;

		sample_generator<<<batchsize,1>>>(dev_walks, g, devStates,
							unsigned(time(NULL)), nodes, batchsize, len);

		HANDLE_ERROR(
			cudaMemcpy(	&host_walks[i],
						dev_walks,
						sizeof(size_t)*batchsize*len,
						cudaMemcpyDeviceToHost)); 

	}

	

	cudaFree(dev_walks);
	cudaFree(devStates);

	return host_walks;
}