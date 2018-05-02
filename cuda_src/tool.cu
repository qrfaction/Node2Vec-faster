#include "./tool.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>


__device__ static bool
search_node(int64_t & L, int64_t & U, const size_t *prev_adj, const size_t k){

	int64_t mid;
	while(L <= U){
			
			mid = (L+U)>>1;

			if(k < prev_adj[ mid ])
				U = mid - 1;
			else if(k > prev_adj[ mid ])
				L = mid + 1;
			else
				return true;
		}
	return false;
}

__device__ static void 
set_weight(	csr_graph * g, 
			const size_t *prev_adj	,	const size_t *curr_adj	, 	double *w, 
			const size_t prev_n_len	, 	const size_t curr_n_len ,   const size_t prev){

	assert(curr_n_len > 0);
	assert(prev_n_len > 0);

	size_t curr_l = 0;
	size_t curr_r = curr_n_len-1;
	size_t node_id;

	int64_t L = 0;
	int64_t U = prev_n_len;

	int64_t u_bound = U;
	int64_t l_bound = L;

	while(true){

		node_id = curr_adj[curr_l];

		if(search_node(l_bound,u_bound,prev_adj,node_id)==false)
			w[curr_l] = 1/(g->q);
		else if(prev == node_id)
			w[curr_l] = 1/(g->p);

		++curr_l;
		if(curr_r < curr_l)
			break;
		
		L = l_bound;          //  限界减小搜索空间
		u_bound = U;

		
		
		node_id = curr_adj[curr_r];

		if(search_node(l_bound,u_bound,prev_adj,node_id)==false)
			w[curr_r] = 1/(g->q);
		else if(prev == node_id)
			w[curr_r] = 1/(g->p);

		--curr_r;
		if(curr_r < curr_l)
			break;
		
		U = u_bound;        //  限界减小搜索空间
		l_bound = L;

	}
}

__device__ static void 
weight_norm(double *w,const size_t N){

	double summation = 0;

	for(size_t i=0;i<N;++i)
		summation +=w[i];
	
	for(size_t i=0;i<N;++i)
		w[i]/=summation;
}


__device__ static size_t 
sampling(const size_t N, double *w, curandState * state){

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




__device__ static void
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
		curr_n_len = end_i - start_i;
		curr_adj = &(g->neighbor[start_i]);


		w = new double[curr_n_len];
		for(size_t j=0;j<curr_n_len;++j)
			w[j] = g->weights[start_i+j];
		
		// 获取上一时刻结点信息
		start_i = g->offset[prev];
		end_i = g->offset[prev+1];
		prev_n_len = end_i - start_i;
		prev_adj = &(g->neighbor[start_i]);




		// 依据上时刻的结点与当前结点修改状态转移概率
		set_weight(g,prev_adj,curr_adj,w,prev_n_len,curr_n_len,prev);
		weight_norm(w,curr_n_len);

		// 采样结点  更新状态

		curr = curr_adj[ sampling(curr_n_len, w, state) ];
		walk[i] = curr;
		prev = walk[i-1];

		delete [] w;
	}

}

__device__ static void
random_walk(csr_graph * g, size_t * walk, const size_t start_node, 
				const size_t len, curandState * state){


	walk[0] = start_node;
	size_t curr = start_node;
	size_t curr_i;
	size_t start_i,end_i;
	size_t num_adj;
	double *w;

	for(size_t i=1;i<len;++i){
		// 获取当前时刻结点信息
		start_i = g->offset[curr];
		end_i = g->offset[curr+1];
		num_adj = end_i - start_i;

		w = &(g->weights[start_i]);

		curr_i = start_i + sampling(num_adj, w, state);
		curr = g->neighbor[ curr_i ];

		walk[i] = curr;
	}

}


__global__ void 
sample_generator(size_t * walks, csr_graph * g, curandState *state, size_t seed,
					const size_t *nodes, const size_t batchsize, const size_t len){
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	auto get_samples = node2vec_walk;
	if((g->p)==(g->q) && (g->p)==1)
		get_samples = random_walk;


	while(tid < batchsize){ 

		curand_init(seed * tid, tid, 0, &state[tid]);  
		get_samples(g, &(walks[ tid * len ]), nodes[tid], len, &state[tid]);

		tid += blockDim.x * gridDim.x ;
		
	}
}
