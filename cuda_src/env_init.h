#ifndef ENV_INIT_H_
#define ENV_INIT_H_

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "./generate_sample.h"



#ifdef __cplusplus
#define GRAPH_C extern "C"
#else
#define GRAPH_C
#endif

GRAPH_C {

    typedef struct{

        size_t num_node;
        size_t num_edge;
        size_t *neighbor;
        size_t *offset;
        float *weights;
        float p;
        float q;       

    }csr_graph;

    void graph_init(const float, const float, const bool);
    void graph_close();
    size_t get_num_node();

}

extern size_t max_threads_per_block;

extern size_t max_blocks_per_grid;

extern csr_graph *G;

extern size_t N;

extern bool in_env;

#endif