#ifndef GENERATE_SAMPLE_H_
#define GENERATE_SAMPLE_H_

#include <cuda_runtime.h>
#include <curand_kernel.h>

typedef struct{

    size_t num_node;
    size_t num_edge;
    size_t *neighbor;
    size_t *offset;
    double *weights;
    double p;
    double q;

}csr_graph;

size_t * get_samples_batch(csr_graph *, const size_t, const size_t, const size_t *);

size_t * get_samples_epoch(csr_graph * , const size_t );

#endif
