#include "./generate_sample.h"
#include "./read_data.h"
#include "./tool.h"
#include <cuda.h>
#include <iostream>
#include <curand.h>
#include <time.h>  


using std::min;
using std::cout;
using std::endl;

#define HANDLE_ERROR(err)  (HandleError(err, __FILE__, __LINE__ ))


static size_t max_threads_per_block;
static size_t max_blocks_per_grid;


static void HandleError(cudaError_t err, const char *file,int line){    
    if (err != cudaSuccess){        
        printf( "%s %d in %s at line %d\n", cudaGetErrorString(err),err,file,line);        
        exit(EXIT_FAILURE);
    }
}


static void init_setting(){

	cudaDeviceProp devProp;
	HANDLE_ERROR(cudaGetDeviceProperties(&devProp, 0));  // 默认使用第一个设备

	max_threads_per_block = devProp.maxThreadsDim[0];
	max_blocks_per_grid = devProp.maxGridSize[0];


}

static void weight_norm(double *w, const size_t *row_shift, const size_t N){

	size_t num_adj;
	for(size_t i=0;i<N;++i){

		num_adj = row_shift[i+1] - row_shift[i];

		for(size_t j=row_shift[i]; j < row_shift[i+1] ;++j)
			w[j]/=num_adj;
	}

}

static csr_graph *
init_graph(const double p, const double q){


	// 读取graph的邻接链表
	size_t num_edge=0;
	auto adj_list = read_graph("edges.csv",num_edge,false);


	// 转成csr格式
	double * weights = new double[num_edge];
	size_t * col_id = new size_t[num_edge];
	size_t * row_shift = new size_t[adj_list.size()+1];
	adjList2CSR(adj_list,weights,col_id,row_shift);

	size_t num_node = adj_list.size();

	// 如果采用随机游走  提前进行规范化权重
	if(p==q && p==1)
		weight_norm(weights,row_shift,num_node);


	csr_graph * h_g = new csr_graph();

	h_g->num_node = num_node;
	h_g->num_edge = num_edge;
	h_g->p = p;
	h_g->q = q;

	HANDLE_ERROR(cudaMalloc((void **)&(h_g->neighbor), num_edge * sizeof(size_t)));
	HANDLE_ERROR(cudaMalloc((void **)&(h_g->offset), (num_node+1) * sizeof(size_t)));
	HANDLE_ERROR(cudaMalloc((void **)&(h_g->weights), num_edge * sizeof(double)));

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
					sizeof(double)*num_edge,
					cudaMemcpyHostToDevice)); 

	delete []weights;
	delete []row_shift;
	delete []col_id;

	return h_g;

}

csr_graph * get_dev_graph(const double p, const double q, size_t *num_node){

	init_setting();

	csr_graph *h_g = init_graph(p,q);
	
	*num_node = h_g->num_node;

	
	csr_graph * dev_g;

	HANDLE_ERROR(cudaMalloc((void **)&dev_g, sizeof(csr_graph)));

	HANDLE_ERROR(
		cudaMemcpy(	dev_g,
					h_g,
					sizeof(csr_graph),
					cudaMemcpyHostToDevice)); 

	return dev_g;
}

void destroy_dev_graph(csr_graph *dev_g){

	csr_graph *h_g = new csr_graph();


	HANDLE_ERROR(
		cudaMemcpy(	h_g,
					dev_g,
					sizeof(csr_graph),
					cudaMemcpyDeviceToHost)); 


	cudaFree(h_g->weights);
	cudaFree(h_g->offset);
	cudaFree(h_g->neighbor);

	cudaFree(dev_g);
	delete h_g;
}

void get_samples_batch(csr_graph * g, const size_t batchsize, const size_t len, 
						const size_t *nodes , size_t *host_walks){


	size_t *dev_walks = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&dev_walks, len * batchsize * sizeof(size_t)));

	curandState *devStates;
	HANDLE_ERROR(cudaMalloc((void **)&devStates, batchsize * sizeof(curandState)));


	size_t *dev_nodes = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&dev_nodes, batchsize*sizeof(size_t)));
	HANDLE_ERROR(
		cudaMemcpy(	dev_nodes,
					nodes,
					sizeof(size_t)*batchsize,
					cudaMemcpyHostToDevice)); 


	int num_thread = min( max_threads_per_block , batchsize );
	int num_block = min( max_blocks_per_grid , (batchsize + num_thread-1)/num_thread );

	sample_generator<<<num_block,num_thread>>>(dev_walks, g, devStates,
							size_t(time(NULL)), dev_nodes, batchsize, len);

	HANDLE_ERROR(
		cudaMemcpy(	host_walks,
					dev_walks,
					sizeof(size_t)*batchsize*len,
					cudaMemcpyDeviceToHost)); 

	cudaFree(dev_walks);
	cudaFree(devStates);
	cudaFree(dev_nodes);

}


void get_samples_epoch(csr_graph * g, const size_t len, const size_t N, size_t *host_walks){


	size_t *dev_walks = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&dev_walks, len * N * sizeof(size_t)));

	curandState *devStates;
	HANDLE_ERROR(cudaMalloc((void **)&devStates, N*sizeof(curandState)));



	size_t *h_nodes = new size_t[N];
	for(size_t i=0;i < N;++i)
		h_nodes[i] = i;
	size_t *dev_nodes = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&dev_nodes, N*sizeof(size_t)));

	HANDLE_ERROR(
		cudaMemcpy(	dev_nodes,
					h_nodes,
					sizeof(size_t)*N,
					cudaMemcpyHostToDevice)); 


	int num_thread = min( max_threads_per_block , N );
	int num_block = min( max_blocks_per_grid , (N+num_thread-1)/num_thread );

	sample_generator<< < num_block,num_thread >> >(dev_walks, g, devStates,
						(size_t)(time(NULL)), dev_nodes, N, len);


	HANDLE_ERROR(
		cudaMemcpy(	host_walks,
					dev_walks,
					sizeof(size_t)*N*len,
					cudaMemcpyDeviceToHost)); 
	

	delete [] h_nodes;
	cudaFree(dev_walks);
	cudaFree(devStates);
	cudaFree(dev_nodes);

}

