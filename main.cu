#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <thrust/host_vector.h>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

using std::vector;
using std::unordered_map;
using std::pair;
using std::make_pair;
using std::cout;
using std::endl;

static void HandleError(cudaError_t err, const char *file,int line){    
	if (err != cudaSuccess){        
		printf( "%s in %s at line %d\n", cudaGetErrorString(err),file,line);        
		exit(EXIT_FAILURE);
	}
}


#define HANDLE_ERROR(err)  (HandleError(err, __FILE__, __LINE__ ))


class csr_graph{

public :

    csr_graph(size_t *,size_t *,double *,size_t ,size_t );
    ~csr_graph();


private :

	size_t num_node;
	size_t num_edge;
	size_t *neighbor;
	double *weights;
	size_t *offset;
	size_t p;
	size_t q;

	__device__ void node2vec_walk(thrust::device_vector<size_t> &,size_t,size_t);

	__device__ void sample(size_t *, size_t *);

	__device__ void set_weight(const size_t *,const size_t *, double *, const size_t, const size_t);


};

struct compareByTwoKey{
    bool operator()(const pair<size_t,double> &t1, const pair<size_t, double> &t2){

        if(t1.first<t2.first)
            return true;
        else
            return false;
    }
};

__device__ void set_weight(const size_t *prev_adj,const size_t *curr_adj, double *w, 
					const size_t prev_n_len, const size_t curr_n_len){

	size_t tid = blockIdx.x;
	size_t curr;
	size_t l_bound;
	size_t u_bound;

	while(tid < curr_n_len){

		curr = curr_adj[tid];
		l_bound = 0;
		u_bound = prev_n_len;
		w[tid] /= q;

		while(l_bound < u_bound){
			if(curr < prev_adj[ (l_bound+u_bound)/2 ]){
				u_bound = (l_bound+u_bound)/2 - 1;
			}
			elif(curr > prev_adj[ (l_bound+u_bound)/2 ]){
				l_bound = (l_bound+u_bound)/2 + 1;
			}
			else{
				w[tid] *= q;
				break;
			}
		}
		tid += blockDim.x * gridDim.x;
	}
}



__device__ void sample(size_t * src,size_t * curr){

}

// __device__ void
// csr_graph::random_walk(thrust::device_vector<size_t> & walk, 
// 				size_t start_node, size_t len){

// 	walk.push_back(start_node);
// 	size_t & curr;
// 	size_t * neighbor;
// 	size_t start_i,end_i;

// 	for(size_t i=0;i<len;++i){
// 		curr = walk.back();
// 		start_i = this->offset[curr];
// 		end_i = this->offset[curr+1];
// 		neighbor = &(this->neighbor[start_i]);

// 	}

// }

csr_graph::csr_graph(size_t *col_id, size_t *row_shift, double *w,
	size_t num_v, size_t num_e, size_t p, size_t q):num_node(num_v), num_edge(num_e){

	this->p = p;
	this->q = q;

	HANDLE_ERROR(
		cudaMalloc((void**)&neighbor, num_e*sizeof(size_t)));
	HANDLE_ERROR(
		cudaMalloc((void**)&weights, num_e*sizeof(double)));
	HANDLE_ERROR(
		cudaMalloc((void**)&offset, num_v*sizeof(size_t)));


	HANDLE_ERROR(
		cudaMemcpy(neighbor, col_id,
			num_e*sizeof(size_t), cudaMemcpyHostToDevice));
	HANDLE_ERROR(
		cudaMemcpy(weights, w,
			num_e*sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(
		cudaMemcpy(offset, row_shift,
			num_v*sizeof(size_t), cudaMemcpyHostToDevice));

	
}

csr_graph::~csr_graph(){

	cudaFree(neighbor);
	cudaFree(weights);
	cudaFree(offset);

}




unordered_map<size_t, vector<pair<size_t,double > > >  
read_graph(std::string network_file, int & num_edge, const bool have_weight){

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


void adjList2CSR(
	unordered_map<size_t, vector<pair<size_t,double> > > & adj_list,
	double *weights, size_t *col_id, size_t *row_shift ){

	size_t i=0;
	vector<pair<size_t, double> >  node_neighbors;

	for(size_t row_id=0; row_id<adj_list.size(); ++row_id){

		node_neighbors = adj_list[row_id];

		for(auto adj_node : node_neighbors){
			weights[i] = adj_node.second;
			col_id[i] = adj_node.first;
			++i;
		}
		row_shift[row_id+1] = i;
	}

}




int main(int argc, char const *argv[]){

	int num_edge;
	auto adj_list = read_graph("edges.csv",num_edge,false);

	double * weights = new double[num_edge];
	size_t * col_id = new size_t[num_edge];
	size_t * row_shift = new size_t[adj_list.size()+1];
	row_shift[0] = 0;

	adjList2CSR(adj_list,weights,col_id,row_shift);


	
	for(int i=0;i<adj_list.size()-1;++i){
		for(int j=row_shift[i];j<row_shift[i+1];++j){
			cout<<row_shift[i]<<"__"<<col_id[j]<<endl;
		}
	}
	thrust::device_vector<int> a(10000,5);

	// cout<<a[5]<<endl;
	#include <typeinfo>

	// auto b = a.begin();

	// cout<<b[2]<<endl<<typeid(b).name();
	sum<<<1,1>>>(a);
	cout<<a[9999];



	return 0;
}




