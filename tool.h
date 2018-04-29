#ifndef TOOL_H_
#define TOOL_H_

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "./generate_sample.h"





__device__ void set_weight(	csr_graph * g, 
							const size_t *,const size_t *, double *, 
                           	const size_t, const size_t, const size_t);

__device__ bool is_adj(size_t &, size_t &, const size_t *, const size_t);

__device__ size_t sample(const size_t, double *, curandState *);

__device__ void weight_norm(double *,size_t);

__device__ void node2vec_walk(csr_graph * g,size_t *, size_t, size_t, curandState *);

__global__ void sample_generator(size_t *, csr_graph *, curandState *, unsigned long,
                    const size_t *, const size_t, const size_t);

#endif