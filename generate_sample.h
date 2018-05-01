#ifndef GENERATE_SAMPLE_H_
#define GENERATE_SAMPLE_H_


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


void get_samples_batch(csr_graph *, const size_t, const size_t, const size_t *, size_t *);

void get_samples_epoch(csr_graph * , const size_t , const size_t N, size_t *);

csr_graph * get_dev_graph(const double ,const double, size_t *);

void destroy_dev_graph(csr_graph *);


#endif
