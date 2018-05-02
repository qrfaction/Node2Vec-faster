CUDA_DIR = $(LD_LIBRARY_PATH)


CPP_SRCS := $(wildcard ../cuda_src/*.cpp)
CUDA_SRCS := $(wildcard ../cuda_src/*.cu)
SRCS := main.cpp $(CUDA_SRCS) $(CPP_SRCS)


FLAGS = -std=c++11 

NVCC = nvcc
ARCH = -gencode arch=compute_30,code=sm_30 \
       -gencode arch=compute_35,code=sm_35 \
       -gencode arch=compute_50,code=[sm_50,compute_50] \
       -gencode arch=compute_52,code=[sm_52,compute_52]

OPTIONS += $(FLAGS) $(ARCH)


all: generator_test.out

generator_test.out: $(SRCS)
	$(NVCC)  $(OPTIONS) -run $^ -o $@


.PHONY: clean
clean:
	rm  generator_test.out
