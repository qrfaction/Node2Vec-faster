#include <iostream>
#include <assert.h>
#include "../cuda_src/generate_sample.h"
#include "../cuda_src/env_init.h"
#include <stdlib.h>

using std::cout;
using std::endl;



void test_sample_batch(const float p,const float q){

	graph_init(p,q,false);


	size_t len = 5;
	size_t batchsize = 6;
	size_t *walks = new size_t[batchsize*len];

	

	size_t nodes[] = {1,2,3,4,5,8};

	get_samples_batch(batchsize,len,nodes,walks);

	for(int i=0;i<batchsize;++i){
		for(int j=0;j<len;++j){

			assert(nodes[i]==walks[i*len]);

			cout<< "walks[" << i << "][" << j <<"]=" 
				<< walks[i*len+j] << " ";
		}

		cout << endl;
	}

	graph_close();

	delete  [] walks ;

}


void test_sample_epoch(const float p,const float q){

	graph_init(p,q,false);
	size_t num_node = get_num_node();

	size_t len = 5;
	size_t *walks = new size_t[num_node*len];

	

	get_samples_epoch(len,walks);

	

	for(int i=0;i < num_node;++i){
		for(int j=0;j<len;++j){

			assert(i==walks[i*len]);

			cout<< "walks[" << i << "][" << j <<"]=" 
				<< walks[i*len+j] << " ";
		}

		cout << endl;
	}

	graph_close();

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
