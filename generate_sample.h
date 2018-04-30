#ifndef GENERATE_SAMPLE_H_
#define GENERATE_SAMPLE_H_

#include <cuda_runtime.h>
#include <curand_kernel.h>


#ifdef __cplusplus
#define GRAPH_EXTERN_C extern "C"
#else
#define GRAPH_EXTERN_C
#endif

GRAPH_EXTERN_C {

    typedef struct{

        size_t num_node;
        size_t num_edge;
        size_t *neighbor;
        size_t *offset;
        double *weights;
        double p;
        double q;       

    }csr_graph;

}


size_t * get_samples_batch(csr_graph *, const size_t, const size_t, const size_t *);

size_t * get_samples_epoch(csr_graph * , const size_t , const size_t N);

csr_graph * init_graph(const double ,const double, csr_graph *);

#endif
