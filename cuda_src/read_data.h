#ifndef READ_DATA_H_
#define READ_DATA_H_

#include <unordered_map>
#include <vector>


using std::vector;
using std::unordered_map;
using std::pair;
using std::string;

unordered_map<size_t, vector<pair<size_t,double> >>  
read_graph(string, size_t & , const bool );


void adjList2CSR(
	unordered_map<size_t, vector<pair<size_t,double> > > & ,
	double *, size_t *, size_t *);



#endif