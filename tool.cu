#include "./tool.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <assert.h>


__device__ bool 
is_adj(size_t & L, size_t & U, const size_t *prev_adj, const size_t k){

	size_t mid;
	while(L <= U){
			mid = (L+U) >> 1;
			if(k < prev_adj[ mid ])
				U = mid - 1;
			else if(k > prev_adj[ mid ])
				L = mid + 1;
			else
				return true;
		}
	return false;
}

__device__ void 
set_weight(	csr_graph * g, 
			const size_t *prev_adj	,	const size_t *curr_adj	, 	double *w, 
			const size_t prev_n_len	, 	const size_t curr_n_len , 	const size_t prev){

	assert(curr_n_len > 0);
	assert(prev_n_len > 0);

	size_t curr_l = 0;
	size_t curr_r = curr_n_len-1;
	size_t node_id;

	size_t L = 0;
	size_t l_bound = L;
	size_t U = prev_n_len;
	size_t u_bound = U;


	while(curr_n_len > 0){

		node_id = curr_adj[curr_l];

		if(is_adj(l_bound,u_bound,prev_adj,node_id)==false)
			w[curr_l] = 1/(g->q);
		else if(node_id == prev)
			w[curr_l] = 1/(g->p);

		++curr_l;
		if(curr_r < curr_l)
			break;
		
		L = l_bound;          //  限界减小搜索空间
		u_bound = U;
		node_id = curr_adj[curr_r];


		if(is_adj(l_bound,u_bound,prev_adj,node_id)==false)
			w[curr_r] = 1/(g->q);
		else if(node_id == prev)
			w[curr_l] = 1/(g->p);

		--curr_r;
		if(curr_r < curr_l)
			break;
		
		U = u_bound;        //  限界减小搜索空间
		l_bound = L;

	}

}

__device__ void 
weight_norm(double *w,const size_t N){

	double summation = 0;

	for(size_t i=0;i<N;++i)
		summation +=w[i];
	
	for(size_t i=0;i<N;++i)
		w[i]/=summation;
}


__device__ size_t 
sample(const size_t N, double *w, curandState * state){

	double x = curand_uniform(state);
	double sum_w = 0;
	size_t i;
	for(i=0;i<N;++i){
		sum_w+=w[i];
		if(sum_w > x)
			break;
	}    
	assert(i<N);
	return i;
}


__device__ void
node2vec_walk(csr_graph * g, size_t * walk, const size_t start_node, 
				const size_t len, curandState * state){


	walk[0] = start_node;
	size_t prev = start_node;
	size_t curr = start_node;
	size_t prev_n_len,curr_n_len;
	size_t * prev_adj;
	size_t * curr_adj;
	size_t start_i,end_i;
	double *w;

	

	for(size_t i=1;i<len;++i){


		// 获取当前时刻结点信息
		start_i = g->offset[curr];
		end_i = g->offset[curr+1];
		curr_n_len = start_i - end_i;
		curr_adj = &(g->neighbor[start_i]);


		w = new double[curr_n_len];

		for(size_t j=0;j<curr_n_len;++j)
			w[j] = g->weights[start_i+j];

		// 获取上一时刻结点信息
		start_i = g->offset[prev];
		end_i = g->offset[prev+1];
		prev_n_len = start_i - end_i;
		prev_adj = &(g->neighbor[start_i]);

		// 依据上时刻的结点与当前结点修改状态转移概率
		set_weight(g,prev_adj,curr_adj,w,prev_n_len,curr_n_len,prev);
		weight_norm(w,curr_n_len);

		// 采样结点  更新状态
		curr = curr_adj[ sample(curr_n_len, w, state) ];
		walk[i] = curr;
		prev = walk[i-1];

		delete [] w;
	}

}


__global__ void 
sample_generator(size_t *walks, csr_graph * g, curandState *state, unsigned long seed,
					const size_t *nodes, const size_t batchsize, const size_t len){
	

	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	while(tid < batchsize){ 

		curand_init (seed, tid, 0, &state[tid]);  

		node2vec_walk(g, &walks[ tid * len ], nodes[tid], len, &state[tid]);
		
		tid += blockDim.x * gridDim.x;

	}
}
