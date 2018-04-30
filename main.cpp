#include <iostream>
#include "generate_sample.h"


using std::cout;
using std::endl;





int main(int argc, char const *argv[]){


	csr_graph *h_g = new csr_graph();

	csr_graph *dev_g = init_graph(1,1,h_g);

	cout<<h_g->num_node<<' '<<h_g->offset[10]<<endl;

	// for(size_t i=0;i<h_g->num_node+1;++i)
	// 	cout<<h_g->offset[i]<<endl;

	size_t * walks= get_samples_epoch(dev_g, 5, h_g->num_node);




	return 0;
}




