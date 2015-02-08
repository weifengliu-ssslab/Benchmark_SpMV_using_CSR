#compilers
CC=nvcc

#GLOBAL_PARAMETERS
USE_DOUBLE = 1
USE_SVM_ALWAYS = 0

#includes
CUDA_INSTALL_PATH = /usr/local/cuda
INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc

#libs
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib -lcudart -lcusparse
LIBS = $(CUDA_LIBS)

all:
	$(CC)  -arch=sm_32 -O3 -Xcompiler -fopenmp main.cu -o spmv $(INCLUDES) $(LIBS) -D USE_DOUBLE=$(USE_DOUBLE) -D USE_SVM_ALWAYS=$(USE_SVM_ALWAYS)
