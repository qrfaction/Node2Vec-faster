#include <iostream>
#include <assert.h>
#include "../cuda_src/generate_sample.h"
#include <stdlib.h>

using std::cout;
using std::endl;



void test_sample_batch(const double p,const double q){

	size_t *num_node = new size_t;
	size_t len = 5;
	size_t batchsize = 6;
	csr_graph *dev_g = get_dev_graph(p,q,num_node);

	size_t *walks = new size_t[batchsize*len];

	size_t nodes[] = {1,2,3,4,5,8};

	get_samples_batch(dev_g,batchsize,len,nodes,walks);

	for(int i=0;i<batchsize;++i){
		for(int j=0;j<len;++j){

			assert(nodes[i]==walks[i*len]);

			cout<< "walks[" << i << "][" << j <<"]=" 
				<< walks[i*len+j] << " ";
		}

		cout << endl;
	}

	destroy_dev_graph(dev_g);

	delete num_node;
	delete  [] walks ;

}


void test_sample_epoch(const double p,const double q){

	size_t *num_node = new size_t;
	size_t len = 5;
	csr_graph *dev_g = get_dev_graph(p,q,num_node);
	size_t *walks = new size_t[(*num_node)*len];


	get_samples_epoch(dev_g,len,*num_node,walks);

	for(int i=0;i<*num_node;++i){
		for(int j=0;j<len;++j){

			assert(i==walks[i*len]);

			cout<< "walks[" << i << "][" << j <<"]=" 
				<< walks[i*len+j] << " ";
		}

		cout << endl;
	}

	destroy_dev_graph(dev_g);

	delete num_node;
	delete  [] walks ;
}

int main(int argc, char const *argv[]){


	test_sample_epoch(1,1);

	cout<<endl<<endl;

	test_sample_epoch(1.5,1.8);

	cout<<endl<<endl;

	test_sample_batch(1,1);

	cout<<endl<<endl;
	
	test_sample_batch(1.2,1.9);

	return 0;
}
