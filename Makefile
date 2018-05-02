
CUDA_DIR = $(LD_LIBRARY_PATH)


CPP_SRCS := $(wildcard cuda_src/*.cpp)
CPP_OBJS := ${CPP_SRCS:cuda_src/%.cpp=build/obj/%.o}
CUDA_SRCS := $(wildcard cuda_src/*.cu)
CUDA_OBJS := ${CUDA_SRCS:cuda_src/%.cu=build/obj/%.o}
OBJS := $(CPP_OBJS) $(CUDA_OBJS)

CC = g++
WARNINGS = -Wall -Wfatal-errors -Wno-unused -Wno-unused-result
CPP_FLAGS = -std=c++11 -fPIC $(WARNINGS) -I$(CUDA_DIR)/include
LD_FLAGS = -L$(CUDA_DIR)/lib64 

NVCC = nvcc
NVCC_FLAGS = -std=c++11 --compiler-options '-fPIC'
ARCH = -gencode arch=compute_30,code=sm_30 \
       -gencode arch=compute_35,code=sm_35 \
       -gencode arch=compute_50,code=[sm_50,compute_50] \
       -gencode arch=compute_52,code=[sm_52,compute_52]

all: build/lib/generator_api.so

build/lib/generator_api.so: $(OBJS)
	@mkdir -p build/lib
	$(CC)  -shared $^ -o $@ $(LD_FLAGS)

build/obj/%.o: cuda_src/%.cpp
	@mkdir -p build/obj
	$(CC) $(CPP_FLAGS) -c $< -o $@

build/obj/%.o: cuda_src/%.cu
	@mkdir -p build/obj
	$(NVCC)  $(ARCH)  $(NVCC_FLAGS) -c $< -o $@



.PHONY: clean
clean:
	rm -rf build
