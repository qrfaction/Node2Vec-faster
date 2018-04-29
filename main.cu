#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <thrust/host_vector.h>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>
#include <curand.h>
#include <assert.h>


using std::vector;
using std::unordered_map;
using std::pair;
using std::make_pair;
using std::cout;
using std::endl;

// static void HandleError(cudaError_t err, const char *file,int line){    
// 	if (err != cudaSuccess){        
// 		printf( "%s in %s at line %d\n", cudaGetErrorString(err),file,line);        
// 		exit(EXIT_FAILURE);
// 	}
// }


// #define HANDLE_ERROR(err)  (HandleError(err, __FILE__, __LINE__ ))

// class csr_graph{

// public :

//     csr_graph(size_t *,size_t *,double *,size_t ,size_t ,double ,double);
//     ~csr_graph();
//     __device__ void node2vec_walk(size_t *, size_t, size_t, curandState *);

// private :

// 	size_t num_node;
// 	size_t num_edge;
// 	size_t *neighbor;
// 	double *weights;
// 	size_t *offset;
// 	size_t p;
// 	size_t q;


// 	__device__ void set_weight(const size_t *,const size_t *, double *, const size_t, const size_t, const size_t);

// 	__device__ bool is_adj(size_t &, size_t &, const size_t *, const size_t);

// 	__device__ size_t sample(const size_t, double *, curandState *);

// 	__device__ void weight_norm(double *,size_t);
// };

// struct compareByTwoKey{
//     bool operator()(const pair<size_t,double> &t1, const pair<size_t, double> &t2){

//         if(t1.first<t2.first)
//             return true;
//         else
//             return false;
//     }
// };

// __device__ bool 
// csr_graph::is_adj(size_t & L, size_t & U, const size_t *prev_adj, const size_t k){

// 	size_t mid;
// 	while(L <= U){
// 			mid = (L+U) >> 1;
// 			if(k < prev_adj[ mid ])
// 				U = mid - 1;
// 			else if(k > prev_adj[ mid ])
// 				L = mid + 1;
// 			else
// 				return true;
// 		}
// 	return false;
// }

// __device__ void 
// csr_graph::set_weight(const size_t *prev_adj,const size_t *curr_adj, double *w, 
// 					const size_t prev_n_len, const size_t curr_n_len, const size_t prev){

// 	assert(curr_n_len > 0);
// 	assert(prev_n_len > 0);

// 	size_t curr_l = 0;
// 	size_t curr_r = curr_n_len-1;
// 	size_t node_id;

// 	size_t L = 0;
// 	size_t l_bound = L;
// 	size_t U = prev_n_len;
// 	size_t u_bound = U;


// 	while(curr_n_len > 0){

// 		node_id = curr_adj[curr_l];

// 		if(is_adj(l_bound,u_bound,prev_adj,node_id)==false)
// 			w[curr_l] = 1/q;
// 		else if(node_id == prev)
// 			w[curr_l] = 1/p;

// 		++curr_l;
// 		if(curr_r < curr_l)
// 			break;
		
// 		L = l_bound;          //  限界减小搜索空间
// 		u_bound = U;
// 		node_id = curr_adj[curr_r];


// 		if(is_adj(l_bound,u_bound,prev_adj,node_id)==false)
// 			w[curr_r] = 1/q;
// 		else if(node_id == prev)
// 			w[curr_l] = 1/p;

// 		--curr_r;
// 		if(curr_r < curr_l)
// 			break;
		
// 		U = u_bound;        //  限界减小搜索空间
// 		l_bound = L;

// 	}

// }

// __device__ void 
// csr_graph::weight_norm(double *w,const size_t N){

// 	double summation = 0;

// 	for(size_t i=0;i<N;++i)
// 		summation +=w[i];
	
// 	for(size_t i=0;i<N;++i)
// 		w[i]/=summation;
// }


// __device__ size_t 
// csr_graph::sample(const size_t N, double *w, curandState * state){

// 	double x = curand_uniform(state);
// 	double sum_w = 0;
// 	size_t i;
// 	for(i=0;i<N;++i){
// 		sum_w+=w[i];
// 		if(sum_w > x)
// 			break;
// 	}    
// 	assert(i<N);
// 	return i;
// }

// __device__ void
// csr_graph::node2vec_walk(size_t * walk, const size_t start_node, const size_t len, curandState * state){


// 	walk[0] = start_node;
// 	size_t prev = start_node;
// 	size_t curr = start_node;
// 	size_t prev_n_len,curr_n_len;
// 	size_t * prev_adj;
// 	size_t * curr_adj;
// 	size_t start_i,end_i;
// 	double *w;

	

// 	for(size_t i=1;i<len;++i){


// 		// 获取当前时刻结点信息
// 		start_i = this->offset[curr];
// 		end_i = this->offset[curr+1];
// 		curr_n_len = start_i - end_i;
// 		curr_adj = &(this->neighbor[start_i]);


// 		w = new double[curr_n_len];
// 		for(size_t j=0;j<curr_n_len;++j)
// 			w[j] = weights[start_i+j];



// 		// 获取上一时刻结点信息
// 		start_i = this->offset[prev];
// 		end_i = this->offset[prev+1];
// 		prev_n_len = start_i - end_i;
// 		prev_adj = &(this->neighbor[start_i]);

// 		// 依据上时刻的结点与当前结点修改状态转移概率
// 		set_weight(prev_adj,curr_adj,w,prev_n_len,curr_n_len,prev);
// 		weight_norm(w,curr_n_len);

// 		// 采样结点  更新状态
// 		curr = curr_adj[ sample(curr_n_len, w, state) ];
// 		walk[i] = curr;
// 		prev = walk[i-1];

// 		delete [] w;
// 	}

// }

// csr_graph::csr_graph(size_t *col_id, size_t *row_shift, double *w,
// 	size_t num_v, size_t num_e, double p, double q):num_node(num_v), num_edge(num_e){

	
// 	this->p = p;
// 	this->q = q;

// 	HANDLE_ERROR(
// 		cudaMalloc((void**)&neighbor, num_e*sizeof(size_t)));
// 	HANDLE_ERROR(
// 		cudaMalloc((void**)&weights, num_e*sizeof(double)));
// 	HANDLE_ERROR(
// 		cudaMalloc((void**)&offset, num_v*sizeof(size_t)));


// 	HANDLE_ERROR(
// 		cudaMemcpy(neighbor, col_id,
// 			num_e*sizeof(size_t), cudaMemcpyHostToDevice));
// 	HANDLE_ERROR(
// 		cudaMemcpy(weights, w,
// 			num_e*sizeof(double), cudaMemcpyHostToDevice));
// 	HANDLE_ERROR(
// 		cudaMemcpy(offset, row_shift,
// 			num_v*sizeof(size_t), cudaMemcpyHostToDevice));

	
// }

// __global__ void 
// sample_generator(size_t *walks, csr_graph * g, curandState *state, 
// 					const size_t *nodes, const size_t batchsize, const size_t len){
	

// 	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

// 	while(tid < batchsize){ 

// 		g->node2vec_walk(&walks[ tid * len ],nodes[tid],len,&state[tid]);
// 		tid += blockDim.x * gridDim.x;

// 		for(size_t i=0;i<len;++i)
// 			printf("%d  ss",walks[ tid * len + i]);
// 		printf("\n");
// 	}
// }


// csr_graph::~csr_graph(){

// 	cudaFree(neighbor);
// 	cudaFree(weights);
// 	cudaFree(offset);

// }









int main(int argc, char const *argv[]){


	
	size_t num_edge=0;
	auto adj_list = read_graph("edges.csv",num_edge,false);


	double * weights = new double[num_edge];

	size_t * col_id = new size_t[num_edge];
	size_t * row_shift = new size_t[adj_list.size()+1];
	row_shift[0] = 0;


	adjList2CSR(adj_list,weights,col_id,row_shift);


	csr_graph * g = new csr_graph(col_id, row_shift, weights, adj_list.size(), num_edge, 1.0, 1.0);
	

	size_t batchsize = 5;
	size_t len = 6;


	size_t *walks = NULL;
	HANDLE_ERROR(cudaMalloc((void **)&walks, len*batchsize*sizeof(size_t)));
	

	size_t n[] = {2,5,6,7,8};

	size_t * nodes;
	HANDLE_ERROR(cudaMalloc((void **)&nodes, batchsize*sizeof(size_t)));
	cudaMemcpy(nodes,n,sizeof(size_t)*batchsize,cudaMemcpyHostToDevice); 

	curandState *devStates;
	HANDLE_ERROR(cudaMalloc((void **)&devStates, batchsize*sizeof(curandState)));

	sample_generator<<<batchsize,1>>>(walks,g,devStates,nodes,batchsize,len);

	cudaDeviceSynchronize();  

	size_t *h_walks = new size_t[batchsize*len];

	cudaMemcpy(h_walks,walks,sizeof(size_t)*batchsize*len,cudaMemcpyDeviceToHost); 

	// for(int i=0;i<adj_list.size()-1;++i){
	// 	for(int j=row_shift[i];j<row_shift[i+1];++j){
	// 		cout<<row_shift[i]<<"__"<<col_id[j]<<endl;
	// 	}
	// }

	// for(size_t i=0;i<batchsize;++i){
	// 	for(size_t j=0;j<len;++j){
	// 		cout<<h_walks[i*len + j]<<"   ";
	// 	}
	// 	cout<<endl;
	// }

	return 0;
}




