import ctypes
from load_lib import _LIB
import numpy as np

_LIB.get_samples_batch.argtypes = (ctypes.c_size_t,
                                   ctypes.c_size_t,
                                   ctypes.POINTER(ctypes.c_size_t),
                                   ctypes.POINTER(ctypes.c_size_t))

_LIB.get_samples_epoch.argtypes = (ctypes.c_size_t,
                                   ctypes.POINTER(ctypes.c_size_t))

_LIB.get_num_node.restype = ctypes.c_size_t

graph_init = _LIB.graph_init
graph_init.argtypes = (ctypes.c_float, ctypes.c_float)

graph_close = _LIB.graph_close



def get_num_node():
    return _LIB.get_num_node()


def get_samples_batch(batchsize,length,nodes):

    assert isinstance(nodes,np.ndarray)
    assert batchsize == len(nodes)

    _nodes = nodes.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))

    _walks = (ctypes.c_size_t*(batchsize * length))()

    _LIB.get_samples_batch(batchsize,length,
                           _nodes,_walks)
    walks = [_walks[i*length : (i+1)*length] for i in range(batchsize)]
    return walks


def get_samples_epoch(length):
    N = get_num_node()
    _walks = (ctypes.c_size_t*(N * length))()
    _LIB.get_samples_epoch(length,_walks)

    walks = [_walks[i*length : (i+1)*length] for i in range(N)]
    return walks



class openGraph:

    def __init__(self,p,q,weighted):
        self.p = p
        self.q = q
        self.weighted = weighted

    def __enter__(self):
        graph_init(self.p,self.q,self.weighted)

    def __exit__(self,exc_type, exc_val, exc_tb):
        graph_close()


if __name__ == "__main__":

    with openGraph(1,1,False):
        print(get_samples_batch(5,6,np.array([1,2,3,4,5])))
        print(get_samples_epoch(6))







