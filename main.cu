#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <thrust/host_vector.h>
#include <unordered_map>
#include <algorithm>
#include <sstream>

using std::vector;
using std::unordered_map;
using std::pair;
using std::make_pair;
using std::cout;
using std::endl;

// class csr_graph
// {
// public :
// 	size_t num_node;
// 	size_t num_edge;
// 	size_t *neighbor;
// 	double *weight;
// 	size_t *offset;
	
//    __global__ csr_graph(std::string,bool);
//    __device__ ~csr_graph();

// };

struct compareByTwoKey
{
    bool operator()(const pair<size_t,double> &t1,const pair<size_t,double> &t2)
    {
        if(t1.first<t2.first)
            return true;
        else
            return false;
    }
};

// csr_graph::csr_graph():num_edge(0)
// {

	

// 	num_node=adj_list.size();

// 	HANDLE_ERROR(
// 		cudaMalloc((void**)&neighbor ; num_edge*sizeof(size_t);));
// 	HANDLE_ERROR(
// 		cudaMalloc((void**)&weight ; num_edge*sizeof(size_t);));
// 	HANDLE_ERROR(
// 		cudaMalloc((void**)&neighbor ; num_node*sizeof(size_t);));

// 	if(have_weight)
// 	{
// 		for(auto iter=adj_list.begin();iter!=adj_list.end();++iter)
// 		{
			
			
// 		}
// 	}
	



// }



unordered_map<size_t,vector<pair<size_t,double > > >  
read_graph(std::string network_file,int & num_edge,const bool have_weight){

	std::ifstream embFile(network_file);
	
	if (embFile.is_open()){

		size_t x,y;
		auto rule_compare = compareByTwoKey();
		unordered_map<size_t,vector<pair<size_t,double > > > adj_list;
		if(have_weight){

			double weight;
			while(embFile>>x>>y>>weight){
				auto pos = lower_bound(adj_list[x].begin(),adj_list[x].end(),
					make_pair(y,weight),rule_compare);
				adj_list[x].insert(pos,make_pair(y,weight));
				++num_edge;
			}

		}
		else{
			while(embFile>>x>>y){
				pair<size_t,double > ele = make_pair(y,1);
				auto pos = lower_bound(adj_list[x].begin(),adj_list[x].end(),
									ele,rule_compare);
				adj_list[x].insert(pos,ele);
				++num_edge;
			}
		}
		embFile.close();

		return adj_list;
	}
	else {

		cout<<"read error";
	    exit(0);
	}

}

//   edges.csvsaddasdasdsadsadasddasdasdasdasd

void adjList2CSR(
	unordered_map<size_t,vector<pair<size_t,double> > > & adj_list,
	size_t *weights,size_t *col_id,size_t *row_shift ){

	size_t i=0;
	vector<pair<size_t,double> >  node_neighbors;

	for(size_t row_id=0;row_id<adj_list.size();++row_id){

		node_neighbors = adj_list[row_id];
		for(auto adj_node=node_neighbors.begin();adj_node!=node_neighbors.end();++i,++adj_node){
			weights[i] = adj_node->second;
			col_id[i] = adj_node->first;
		}
		row_shift[row_id+1] = i;
	}

}

int main(int argc, char const *argv[]){

	int num_edge;
	auto adj_list = read_graph("edges.csv",num_edge,false);

	size_t * weights = new size_t[num_edge];
	size_t * col_id = new size_t[num_edge];
	size_t * row_shift = new size_t[adj_list.size()+1];
	row_shift[0] = 0;

	adjList2CSR(adj_list,weights,col_id,row_shift);


	
	for(int i=0;i<adj_list.size()-1;++i){
		for(int j=row_shift[i];j<row_shift[i+1];++j){
			cout<<row_shift[i]<<"__"<<col_id[j]<<endl;
		}
	}


	return 0;
}




