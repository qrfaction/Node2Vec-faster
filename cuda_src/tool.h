#ifndef TOOL_H_
#define TOOL_H_

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "./generate_sample.h"




__global__ void sample_generator(size_t *, csr_graph *, curandState *, size_t,
                    const size_t *, const size_t, const size_t);

#endif