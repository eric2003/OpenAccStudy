EXES=test
CXX=nvc++
GPUFLAGS = -gpu=managed -Minfo=all,ccff -Mneginfo
CXXFLAGS = -fast -acc $(GPUFLAGS)
CC=nvc
CFLAGS=$(CXXFLAGS)
CFLAGS += -DENABLE_CUDA
CFLAGS += -DENABLE_OPENACC
CUDAC=nvcc
CUDAFLAGS=-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80
#LDFLAGS=-Mcuda 

GPUFLAGS += -DENABLE_CUDA
GPUFLAGS += -DENABLE_OPENACC

all: $(EXES)
test: main.o cuda_sub.o laplace2d_kernels.o myopenacc.o
	$(CXX) -o $@ $(CFLAGS) $^ $(LDFLAGS)
main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp
laplace2d_kernels.o: laplace2d_kernels.cpp
	$(CXX) $(CXXFLAGS) -c laplace2d_kernels.cpp
myopenacc.o: myopenacc.cpp
	$(CXX) $(CXXFLAGS) -c myopenacc.cpp
cuda_sub.o: cuda_sub.cu
	$(CXX) $(CXXFLAGS) -c cuda_sub.cu
clean:
	rm -rf *.o $(EXES)
