#ifndef GENERATE_SAMPLE_H_
#define GENERATE_SAMPLE_H_


#ifdef __cplusplus
#define GENERATOR_C extern "C"
#else
#define GENERATOR_C
#endif

GENERATOR_C {

    void get_samples_batch(const size_t, const size_t, const size_t *, size_t *);

	void get_samples_epoch(const size_t , size_t *);


}

#endif
