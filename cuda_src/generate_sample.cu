#include "./generate_sample.h"
#include "./read_data.h"
#include "./tool.h"
#include "./env_init.h"
#include <cuda.h>
#include <iostream>
#include <curand.h>
#include <time.h>  

#define HANDLE_ERROR(err)  (HandleError(err, __FILE__, __LINE__ ))

using std::min;
using std::cout;
using std::endl;


void get_samples_batch(const size_t batchsize, const size_t len, 
						const size_t *nodes , size_t *host_walks){
	if(in_env==false){
		cout<<"Not initialized yet"<<endl;
		exit(0);
	}


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

	sample_generator<<<num_block,num_thread>>>(dev_walks,G, devStates,
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


void get_samples_epoch(const size_t len, size_t *host_walks){

	if(in_env==false){
		cout<<"Not initialized yet"<<endl;
		exit(0);
	}
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

	sample_generator<< < num_block,num_thread >> >(dev_walks,G, devStates,
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

