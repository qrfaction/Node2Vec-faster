#include "./env_init.h"
#include "./generate_sample.h"
#include "./read_data.h"
#include "./tool.h"
#include <iostream>

#define HANDLE_ERROR(err)  (HandleError(err, __FILE__, __LINE__ ))

using std::cout;
using std::endl;

size_t N = 0;
bool in_env = false;
size_t max_threads_per_block = 0;
size_t max_blocks_per_grid = 0;
csr_graph *G = NULL;

static void weight_norm(float *w, const size_t *row_shift){

	size_t num_adj;
	for(size_t i=0;i<N;++i){

		num_adj = row_shift[i+1] - row_shift[i];

		for(size_t j=row_shift[i]; j < row_shift[i+1] ;++j)
			w[j]/=num_adj;
	}

}

static void init_setting(){

	cudaDeviceProp devProp;
	HANDLE_ERROR(cudaGetDeviceProperties(&devProp, 0));  // 默认使用第一个设备

	max_threads_per_block = devProp.maxThreadsDim[0];
	max_blocks_per_grid = devProp.maxGridSize[0];


}

static csr_graph *
get_graph(const float p, const float q, const bool have_weight){


	// 读取graph的邻接链表
	size_t num_edge=0;
	auto adj_list = read_graph("../edges.csv",num_edge,have_weight);


	// 转成csr格式
	float * weights = new float[num_edge];
	size_t * col_id = new size_t[num_edge];
	size_t * row_shift = new size_t[adj_list.size()+1];
	adjList2CSR(adj_list,weights,col_id,row_shift);

	size_t num_node = adj_list.size();
	N = num_node;

	// 如果采用随机游走  提前进行规范化权重
	if(p==q && p==1)
		weight_norm(weights,row_shift);


	csr_graph * h_g = new csr_graph();

	h_g->num_node = num_node;
	h_g->num_edge = num_edge;
	h_g->p = p;
	h_g->q = q;

	HANDLE_ERROR(cudaMalloc((void **)&(h_g->neighbor), num_edge * sizeof(size_t)));
	HANDLE_ERROR(cudaMalloc((void **)&(h_g->offset), (num_node+1) * sizeof(size_t)));
	HANDLE_ERROR(cudaMalloc((void **)&(h_g->weights), num_edge * sizeof(float)));

	HANDLE_ERROR(
		cudaMemcpy(	h_g->neighbor,
					col_id,
					sizeof(size_t)*num_edge,
					cudaMemcpyHostToDevice)); 
	HANDLE_ERROR(
		cudaMemcpy(	h_g->offset,
					row_shift,
					sizeof(size_t)*(num_node+1),
					cudaMemcpyHostToDevice)); 
	HANDLE_ERROR(
		cudaMemcpy(	h_g->weights,
					weights,
					sizeof(float)*num_edge,
					cudaMemcpyHostToDevice)); 

	delete []weights;
	delete []row_shift;
	delete []col_id;

	return h_g;

}


void graph_init(const float p, const float q, const bool have_weight){

	if(in_env==true){
		cout<<"Already initialized"<<endl;
		exit(0);
	}

	in_env = true;

	init_setting();



	csr_graph *h_g = get_graph(p,q,have_weight);
	
	N = h_g->num_node;


	HANDLE_ERROR(cudaMalloc((void **)&G, sizeof(csr_graph)));

	HANDLE_ERROR(
		cudaMemcpy(	G,
					h_g,
					sizeof(csr_graph),
					cudaMemcpyHostToDevice)); 
}

void graph_close(){

	if(in_env==false){
		cout<<"Not initialized yet"<<endl;
		exit(0);
	}
	in_env = false;

	csr_graph *h_g = new csr_graph();

	HANDLE_ERROR(
		cudaMemcpy(	h_g,
					G,
					sizeof(csr_graph),
					cudaMemcpyDeviceToHost)); 


	cudaFree(h_g->weights);
	cudaFree(h_g->offset);
	cudaFree(h_g->neighbor);

	cudaFree(G);
	delete h_g;
}

size_t get_num_node(){
	if(in_env==false){
		cout<<"Not initialized yet"<<endl;
		exit(0);
	}
	return N;
}