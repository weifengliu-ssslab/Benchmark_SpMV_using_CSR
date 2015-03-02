#compilers
CC=g++

#GLOBAL_PARAMETERS
USE_DOUBLE = 0
USE_SVM_ALWAYS = 1

#includes
INCLUDES = -I/opt/AMDAPPSDK-3.0-0-Beta/include

#libs
OPENCL_LIBS = -L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64 -lOpenCL
LIBS = $(OPENCL_LIBS)

all:
	$(CC) -O3 main.cpp -o spmv $(INCLUDES) $(LIBS) -D USE_DOUBLE=$(USE_DOUBLE) -D USE_SVM_ALWAYS=$(USE_SVM_ALWAYS)
