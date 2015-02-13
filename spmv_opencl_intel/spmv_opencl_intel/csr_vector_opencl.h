#ifndef CSR_VECTOR_OPENCL_H
#define CSR_VECTOR_OPENCL_H

#include "common.h"
#include "basiccl.h"

class csr_vector_opencl
{
public:
    csr_vector_opencl();
    int init_platform();
	int init_kernels(string compile_flags);
    int prepare_mem(int m, int n, int nnzA, int *csrRowPtrA, int *csrColIdxA, value_type *csrValA,
                    value_type *x, value_type *y);
    int run_benchmark();
	int get_y();
    int free_platform();
    int free_mem();
	void sync_device();

private:
    bool _profiling;

    // basic OpenCL variables
    BasicCL _basicCL;

    char _platformVendor[CL_STRING_LENGTH];
    char _platformVersion[CL_STRING_LENGTH];

    char _gpuDeviceName[CL_STRING_LENGTH];
    char _gpuDeviceVersion[CL_STRING_LENGTH];
    int  _gpuDeviceComputeUnits;
    cl_ulong  _gpuDeviceGlobalMem;
    cl_ulong  _gpuDeviceLocalMem;
    int  _localDeviceComputeUnits;

    cl_uint             _numPlatforms;           // OpenCL platform
    cl_platform_id*     _cpPlatforms;

    cl_uint             _numGpuDevices;          // OpenCL Gpu device
    cl_device_id*       _cdGpuDevices;
    cl_device_id        _cdGpuDevice;

    cl_context          _cxLocalContext;         // OpenCL Local context
    cl_command_queue    _cqLocalCommandQueue;    // OpenCL Local command queues

    cl_program          _cpCSRSpMV;               // OpenCL Gpu program
    cl_kernel           _ckCSRVecSpMV;            // OpenCL Gpu kernel

    int _m;
    int _n;
    int _nnzA;

	int _THREADS_PER_VECTOR;

#if USE_SVM_ALWAYS
	// A
    value_type *_svm_csrValA;
    int        *_svm_csrRowPtrA;
    int        *_svm_csrColIdxA;
	// x and y
    value_type *_svm_x;
	value_type *_svm_y;
#else
	// A
    cl_mem      _d_csrValA;
    cl_mem      _d_csrColIdxA;
	cl_mem      _d_csrRowPtrA;
	// x and y
	cl_mem      _d_x;
	cl_mem      _d_y;
#endif

	value_type *_h_y;
};

csr_vector_opencl::csr_vector_opencl()
{
}

int csr_vector_opencl::init_platform()
{
    int err = CL_SUCCESS;
    _profiling = false;
    int select_device = 0;

    // platform
    err = _basicCL.getNumPlatform(&_numPlatforms);
    if(err != CL_SUCCESS) return err;
    //cout << "#Platform = " << _numPlatforms << endl;

    _cpPlatforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * _numPlatforms);

    err = _basicCL.getPlatformIDs(_cpPlatforms, _numPlatforms);
    if(err != CL_SUCCESS) return err;

    for (unsigned int i = 0; i < _numPlatforms; i++)
    {
        err = _basicCL.getPlatformInfo(_cpPlatforms[i], _platformVendor, _platformVersion);
        if(err != CL_SUCCESS) return err;

        // Gpu device
        err = _basicCL.getNumGpuDevices(_cpPlatforms[i], &_numGpuDevices);

        if (_numGpuDevices > 0)
        {
            _cdGpuDevices = (cl_device_id *)malloc(_numGpuDevices * sizeof(cl_device_id) );

            err |= _basicCL.getGpuDeviceIDs(_cpPlatforms[i], _numGpuDevices, _cdGpuDevices);

            err |= _basicCL.getDeviceInfo(_cdGpuDevices[select_device], _gpuDeviceName, _gpuDeviceVersion,
                                         &_gpuDeviceComputeUnits, &_gpuDeviceGlobalMem,
                                         &_gpuDeviceLocalMem, NULL);
            if(err != CL_SUCCESS) return err;

            //cout << "Platform [" << i <<  "] Vendor: " << _platformVendor << ", Version: " << _platformVersion << endl;
            //cout << _numGpuDevices << " Gpu device: "
            //     << _gpuDeviceName << " ("
            //     << _gpuDeviceComputeUnits << " compute units, "
            //     << _gpuDeviceLocalMem / 1024 << " KB local, "
            //     << _gpuDeviceGlobalMem / (1024 * 1024) << " MB global, "
            //     << _gpuDeviceVersion << ")" << endl;

            break;
        }
        else
        {
            continue;
        }
    }

    // Gpu context
    err = _basicCL.getContext(&_cxLocalContext, _cdGpuDevices, _numGpuDevices);
    if(err != CL_SUCCESS) return err;

    // Gpu commandqueue
    if (_profiling)
        err = _basicCL.getCommandQueueProfilingEnable(&_cqLocalCommandQueue, _cxLocalContext, _cdGpuDevices[select_device]);
    else
        err = _basicCL.getCommandQueue(&_cqLocalCommandQueue, _cxLocalContext, _cdGpuDevices[select_device]);
    if(err != CL_SUCCESS) return err;

    return err;
}

int csr_vector_opencl::init_kernels(string compile_flags)
{
	int err = CL_SUCCESS;

	// get programs
    err  = _basicCL.getProgramFromFile(&_cpCSRSpMV, _cxLocalContext, "csr_vector_spmv_kernels.cl", compile_flags);
    if(err != CL_SUCCESS) return err;

    // get kernels
    err  = _basicCL.getKernel(&_ckCSRVecSpMV, _cpCSRSpMV, "spmv_csr_vector_kernel");
    if(err != CL_SUCCESS) return err;

    return err;
}

int csr_vector_opencl::free_platform()
{
    int err = CL_SUCCESS;

    // free OpenCL kernels
    err = clReleaseKernel(_ckCSRVecSpMV);    if(err != CL_SUCCESS) return err;

    // free OpenCL programs
    err = clReleaseProgram(_cpCSRSpMV);    if(err != CL_SUCCESS) return err;

    return err;
}


int csr_vector_opencl::free_mem()
{
    int err = CL_SUCCESS;

	err = clFinish(_cqLocalCommandQueue);

#if USE_SVM_ALWAYS
	// A
	clSVMFree(_cxLocalContext, _svm_csrRowPtrA);
    clSVMFree(_cxLocalContext, _svm_csrValA);
    clSVMFree(_cxLocalContext, _svm_csrColIdxA);
	// x and y
	clSVMFree(_cxLocalContext, _svm_x);
	clSVMFree(_cxLocalContext, _svm_y);
#else
	// A
	if(_d_csrRowPtrA) err = clReleaseMemObject(_d_csrRowPtrA);
	if(_d_csrValA) err = clReleaseMemObject(_d_csrValA);
    if(_d_csrColIdxA) err = clReleaseMemObject(_d_csrColIdxA);
	// x and y
	if(_d_x) err = clReleaseMemObject(_d_x);
	if(_d_y) err = clReleaseMemObject(_d_y);
#endif

    return err;
}

void csr_vector_opencl::sync_device()
{
	clFinish(_cqLocalCommandQueue);
}

int csr_vector_opencl::run_benchmark()

{
    int err = CL_SUCCESS;

	const int nnz_per_row = _nnzA / _m;

    if (nnz_per_row <=  4)
        _THREADS_PER_VECTOR = 4;
    else
        _THREADS_PER_VECTOR = 8;

    const size_t THREADS_PER_BLOCK  = 64;
    const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / _THREADS_PER_VECTOR;

    // compute kernel launch parameters
    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = THREADS_PER_BLOCK;
    int num_blocks = ceil((double)_nnzA / (double)num_threads);

	szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

	err  = clSetKernelArg(_ckCSRVecSpMV, 0, sizeof(cl_int), (void*)&_m);
#if USE_SVM_ALWAYS
	err |= clSetKernelArgSVMPointer(_ckCSRVecSpMV, 1,  _svm_csrRowPtrA);
    err |= clSetKernelArgSVMPointer(_ckCSRVecSpMV, 2,  _svm_csrColIdxA);
	err |= clSetKernelArgSVMPointer(_ckCSRVecSpMV, 3,  _svm_csrValA);
	err |= clSetKernelArgSVMPointer(_ckCSRVecSpMV, 4,  _svm_x);
    err |= clSetKernelArgSVMPointer(_ckCSRVecSpMV, 5,  _svm_y);
#else
    err |= clSetKernelArg(_ckCSRVecSpMV, 1, sizeof(cl_mem), (void*)&_d_csrRowPtrA);
    err |= clSetKernelArg(_ckCSRVecSpMV, 2, sizeof(cl_mem), (void*)&_d_csrColIdxA);
    err |= clSetKernelArg(_ckCSRVecSpMV, 3, sizeof(cl_mem), (void*)&_d_csrValA);
    err |= clSetKernelArg(_ckCSRVecSpMV, 4, sizeof(cl_mem), (void*)&_d_x);
    err |= clSetKernelArg(_ckCSRVecSpMV, 5, sizeof(cl_mem), (void*)&_d_y);
#endif
    err |= clSetKernelArg(_ckCSRVecSpMV, 6, sizeof(value_type) * (VECTORS_PER_BLOCK * _THREADS_PER_VECTOR + _THREADS_PER_VECTOR / 2), NULL);
    err |= clSetKernelArg(_ckCSRVecSpMV, 7, sizeof(cl_int)     * (VECTORS_PER_BLOCK * 2), NULL);
    err |= clSetKernelArg(_ckCSRVecSpMV, 8, sizeof(cl_uint), (void*)&VECTORS_PER_BLOCK);
    err |= clSetKernelArg(_ckCSRVecSpMV, 9, sizeof(cl_uint), (void*)&_THREADS_PER_VECTOR);
    if(err != CL_SUCCESS) { cout << "arg error = " << err << endl; return err; }

    // run kernel
    err = clEnqueueNDRangeKernel(_cqLocalCommandQueue, _ckCSRVecSpMV, 1,
                                     NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
        if(err != CL_SUCCESS) { cout << "kernel run error = " << err << endl; return err; }

    return err;
}

int csr_vector_opencl::get_y()
{
	int err = CL_SUCCESS;

#if USE_SVM_ALWAYS
    // copy svm_y to h_y
	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_READ, _svm_y, _m * sizeof(value_type), 0, 0, 0);
    memcpy(_h_y, _svm_y, _m * sizeof(value_type));
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_y, 0, 0, 0 );
#else
	err = clEnqueueReadBuffer(_cqLocalCommandQueue,
                              _d_y, CL_TRUE, 0, _m * sizeof(value_type),
                              _h_y, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;
#endif

    return err;
}

int csr_vector_opencl::prepare_mem(int m, int n, int nnzA,
                                      int *csrRowPtrA, int *csrColIdxA, value_type *csrValA,
                                      value_type *x, value_type *y)
{
    int err = CL_SUCCESS;

    _m = m;
    _n = n;
    _nnzA = nnzA;

    _h_y = y;

    // prepare shared virtual memory (unified memory)
#if USE_SVM_ALWAYS
    cout << endl << "CUSP is using shared virtual memory.";

	// Matrix A
	_svm_csrRowPtrA = (int *)clSVMAlloc(_cxLocalContext, CL_MEM_READ_ONLY, (_m+1)  * sizeof(int), 0 );
	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_WRITE, _svm_csrRowPtrA, (_m+1)  * sizeof(int), 0, 0, 0);
	memcpy(_svm_csrRowPtrA, csrRowPtrA, (_m+1)  * sizeof(int));
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_csrRowPtrA, 0, 0, 0 );

	_svm_csrValA    = (value_type *)clSVMAlloc(_cxLocalContext, CL_MEM_READ_ONLY, _nnzA  * sizeof(value_type), 0 );
	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_WRITE, _svm_csrValA, _nnzA  * sizeof(value_type), 0, 0, 0);
	memcpy(_svm_csrValA, csrValA, _nnzA  * sizeof(value_type));
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_csrValA, 0, 0, 0 );

	_svm_csrColIdxA = (int *)clSVMAlloc(_cxLocalContext, CL_MEM_READ_ONLY, _nnzA * sizeof(int), 0 );
	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_WRITE, _svm_csrColIdxA, _nnzA * sizeof(int), 0, 0, 0);
	memcpy(_svm_csrColIdxA, csrColIdxA, _nnzA * sizeof(int));
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_csrColIdxA, 0, 0, 0 );

	// Vector x
	_svm_x = (value_type *)clSVMAlloc(_cxLocalContext, CL_MEM_READ_ONLY, _n  * sizeof(value_type), 0 );
	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_WRITE, _svm_x, _n * sizeof(value_type), 0, 0, 0);
	memcpy(_svm_x, x, _n * sizeof(value_type));
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_x, 0, 0, 0 );

	// Vector y
    _svm_y = (value_type *)clSVMAlloc(_cxLocalContext, CL_MEM_READ_WRITE, _m  * sizeof(value_type), 0 );
	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_WRITE, _svm_y, _m * sizeof(value_type), 0, 0, 0);
	memset(_svm_y, 0, _m * sizeof(value_type));
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_y, 0, 0, 0 );
    // prepare device memory
#else
	cout << endl << "CUSP is using dedicated GPU memory";

	// Matrix A
    _d_csrColIdxA = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY, _nnzA  * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) return err;
    _d_csrRowPtrA = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY, (_m+1) * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) return err;
    _d_csrValA    = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY, _nnzA  * sizeof(value_type), NULL, &err);
    if(err != CL_SUCCESS) return err;

    err = clEnqueueWriteBuffer(_cqLocalCommandQueue, _d_csrColIdxA, CL_TRUE, 0, _nnzA  * sizeof(int), csrColIdxA, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;
    err = clEnqueueWriteBuffer(_cqLocalCommandQueue, _d_csrRowPtrA, CL_TRUE, 0, (_m+1) * sizeof(int), csrRowPtrA, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;
    err = clEnqueueWriteBuffer(_cqLocalCommandQueue, _d_csrValA, CL_TRUE, 0, _nnzA  * sizeof(value_type), csrValA, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;

	// Vector x
    _d_x    = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY, _n  * sizeof(value_type), NULL, &err);
    if(err != CL_SUCCESS) return err;
    err = clEnqueueWriteBuffer(_cqLocalCommandQueue, _d_x, CL_TRUE, 0, _n  * sizeof(value_type), x, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;

	// Vector y
    _d_y    = clCreateBuffer(_cxLocalContext, CL_MEM_READ_WRITE, _m  * sizeof(value_type), NULL, &err);
    if(err != CL_SUCCESS) return err;
    err = clEnqueueWriteBuffer(_cqLocalCommandQueue, _d_y, CL_TRUE, 0, _m  * sizeof(value_type), _h_y, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;
#endif

    return err;
}
	

#endif // CSR_VECTOR_OPENCL_H
