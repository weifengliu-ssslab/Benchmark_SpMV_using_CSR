#ifndef BHSPARSE_SPMV_OPENCL_H
#define BHSPARSE_SPMV_OPENCL_H

#include "common.h"
#include "basiccl.h"

class bhsparse_spmv_opencl
{
public:
    bhsparse_spmv_opencl();
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
	bool checkSVMAvailability(cl_device_id device);
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
    cl_kernel           _ckCSRSpMV;               // OpenCL Gpu kernel

    int _m;
    int _n;
    int _nnzA;

    // A
    value_type *_svm_csrValA;
    int        *_svm_csrRowPtrA;
    int        *_svm_csrColIdxA;

    cl_mem      _d_csrValA;
    cl_mem      _d_csrColIdxA;

    // x and y
    value_type *_svm_x;
    cl_mem      _d_x;

    value_type *_h_y;
    value_type *_svm_y;
    value_type *_svm_y_temp;

    // workload partition parameters
    int _partition_size;
    int _partition_num;
    int _threadbunch_num;
    int _threadbunch_per_block;

    // speculative execution buffer
    int      *_svm_speculator;
    int      *_svm_dirty_counter;

    // inter-warp y entry value store
    int        *_svm_synchronizer_idx;
    value_type *_svm_synchronizer_val;
};

bhsparse_spmv_opencl::bhsparse_spmv_opencl()
{
}

int bhsparse_spmv_opencl::init_platform()
{
    int err = CL_SUCCESS;
    _profiling = false;
    int select_device = 0;

    // platform
    err = _basicCL.getNumPlatform(&_numPlatforms);
    if(err != CL_SUCCESS) return err;
	cout << "#Platform = " << _numPlatforms << endl;

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

            cout << "Platform [" << i <<  "] Vendor: " << _platformVendor << ", Version: " << _platformVersion << endl;
			cout << "#Gpu device = " << _numGpuDevices << endl
				 << "GPU device [" << select_device << "] = "
                 << _gpuDeviceName << " ("
                 << _gpuDeviceComputeUnits << " compute units, "
                 << _gpuDeviceLocalMem / 1024 << " KB local, "
                 << _gpuDeviceGlobalMem / (1024 * 1024) << " MB global, "
                 << _gpuDeviceVersion << ")" << endl;

            break;
        }
        else
        {
            continue;
        }
    }

	if(!checkSVMAvailability(_cdGpuDevices[select_device]))
    {
        cout << "Cannot detect Shared Virtual Memory (SVM) capabilities of the device." << endl;
		return BHSPARSE_UNSUPPORTED_DEVICE;
    }
	else
	{
		cout << "The device supports Shared Virtual Memory (SVM)." << endl;
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

bool bhsparse_spmv_opencl::checkSVMAvailability(cl_device_id device)
{
    cl_device_svm_capabilities caps;
    int err = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities), &caps, 0 );

    // Coarse-grained buffer SVM should be available on any OpenCL 2.0 device.
    // So it is either not an OpenCL 2.0 device or it must support coarse-grained buffer SVM:
    return err == CL_SUCCESS && (caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER);
}

int bhsparse_spmv_opencl::init_kernels(string compile_flags)
{
	int err = CL_SUCCESS;

	// get programs
    err  = _basicCL.getProgramFromFile(&_cpCSRSpMV, _cxLocalContext, "bhsparse_csr_spmv_kernels.cl", compile_flags);
    if(err != CL_SUCCESS) return err;

    // get kernels
    err  = _basicCL.getKernel(&_ckCSRSpMV, _cpCSRSpMV, "SpMV_kernel");
    if(err != CL_SUCCESS) return err;

    return err;
}

int bhsparse_spmv_opencl::free_platform()
{
    int err = CL_SUCCESS;

    // free OpenCL kernels
    err = clReleaseKernel(_ckCSRSpMV);    if(err != CL_SUCCESS) return err;

    // free OpenCL programs
    err = clReleaseProgram(_cpCSRSpMV);    if(err != CL_SUCCESS) return err;

    return err;
}

int bhsparse_spmv_opencl::free_mem()
{
    int err = CL_SUCCESS;

	err = clFinish(_cqLocalCommandQueue);

    // A
	clSVMFree(_cxLocalContext, _svm_csrRowPtrA);
#if USE_SVM_ALWAYS
    clSVMFree(_cxLocalContext, _svm_csrValA);
    clSVMFree(_cxLocalContext, _svm_csrColIdxA);
#else
	if(_d_csrValA) err = clReleaseMemObject(_d_csrValA);
    if(_d_csrColIdxA) err = clReleaseMemObject(_d_csrColIdxA);
#endif

	// vector
#if USE_SVM_ALWAYS
	clSVMFree(_cxLocalContext, _svm_x);
#else
	if(_d_x) err = clReleaseMemObject(_d_x);
#endif

    clSVMFree(_cxLocalContext, _svm_y);
    clSVMFree(_cxLocalContext, _svm_y_temp);

	// other buffers
	clSVMFree(_cxLocalContext, _svm_speculator);
    clSVMFree(_cxLocalContext, _svm_dirty_counter);
    clSVMFree(_cxLocalContext, _svm_synchronizer_idx);
	clSVMFree(_cxLocalContext, _svm_synchronizer_val);

    return err;
}

void bhsparse_spmv_opencl::sync_device()
{
	clFinish(_cqLocalCommandQueue);
}

int bhsparse_spmv_opencl::run_benchmark()

{
    int err = CL_SUCCESS;

    _threadbunch_per_block = THREADGROUP / THREADBUNCH;

    // compute kernel launch parameters
    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = THREADGROUP;
    int num_blocks = ceil((double)_threadbunch_num / (double)_threadbunch_per_block);

    //cout << "#PARTITIONS = " << _partition_num
    //         << ", #THREADBUNCH = " << _threadbunch_num
    //         << ", #THREADS/BLOCK = " << num_threads
    //         << ", #BLOCKS = " << num_blocks << endl;

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    err  = clSetKernelArgSVMPointer(_ckCSRSpMV, 0,  _svm_csrRowPtrA);
#if USE_SVM_ALWAYS
    err |= clSetKernelArgSVMPointer(_ckCSRSpMV, 1,  _svm_csrColIdxA);
    err |= clSetKernelArgSVMPointer(_ckCSRSpMV, 2,  _svm_csrValA);
	err |= clSetKernelArgSVMPointer(_ckCSRSpMV, 3,  _svm_x);
#else
	err |= clSetKernelArg(_ckCSRSpMV, 1,  sizeof(cl_mem), (void*)&_d_csrColIdxA);
    err |= clSetKernelArg(_ckCSRSpMV, 2,  sizeof(cl_mem), (void*)&_d_csrValA);
	err |= clSetKernelArg(_ckCSRSpMV, 3,  sizeof(cl_mem), (void*)&_d_x);
#endif
    err |= clSetKernelArgSVMPointer(_ckCSRSpMV, 4,  _svm_y);
    err |= clSetKernelArgSVMPointer(_ckCSRSpMV, 5,  _svm_speculator);
    err |= clSetKernelArgSVMPointer(_ckCSRSpMV, 6,  _svm_dirty_counter);
    err |= clSetKernelArgSVMPointer(_ckCSRSpMV, 7,  _svm_synchronizer_idx);
    err |= clSetKernelArgSVMPointer(_ckCSRSpMV, 8,  _svm_synchronizer_val);
    err |= clSetKernelArg(_ckCSRSpMV, 9, sizeof(value_type)  * (_threadbunch_per_block * SEG_H * THREADBUNCH), NULL);
    err |= clSetKernelArg(_ckCSRSpMV, 10, sizeof(value_type) * (_threadbunch_per_block * THREADBUNCH), NULL);
	err |= clSetKernelArg(_ckCSRSpMV, 11, sizeof(cl_int)     * (_threadbunch_per_block * (STEP + 1)), NULL);
    err |= clSetKernelArg(_ckCSRSpMV, 12, sizeof(cl_uint)    * (_threadbunch_per_block * (THREADBUNCH+1)), NULL);
    err |= clSetKernelArg(_ckCSRSpMV, 13, sizeof(value_type) * (_threadbunch_per_block), NULL);
	err |= clSetKernelArg(_ckCSRSpMV, 14, sizeof(cl_bool)    * (_threadbunch_per_block), NULL);
    err |= clSetKernelArg(_ckCSRSpMV, 15, sizeof(cl_bool)    * (_threadbunch_per_block * (THREADBUNCH + 1)), NULL);
    err |= clSetKernelArg(_ckCSRSpMV, 16, sizeof(cl_int), (void*)&_partition_size);
    err |= clSetKernelArg(_ckCSRSpMV, 17, sizeof(cl_int), (void*)&_partition_num);
    err |= clSetKernelArg(_ckCSRSpMV, 18, sizeof(cl_int), (void*)&_threadbunch_num);
    err |= clSetKernelArg(_ckCSRSpMV, 19, sizeof(cl_int), (void*)&_nnzA);
    err |= clSetKernelArg(_ckCSRSpMV, 20, sizeof(cl_int), (void*)&_m);
    if(err != CL_SUCCESS) { cout << "arg error = " << err << endl; return err; }

    // run kernel

	//bhsparse_timer spmv_timer;
    //spmv_timer.start();

    // clear dirty_counter
	_svm_dirty_counter[0] = 0;

    err = clEnqueueNDRangeKernel(_cqLocalCommandQueue, _ckCSRSpMV, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
    if(err != CL_SUCCESS) { cout << "kernel run error = " << err << endl; return err; }

	//cout << "[1/3] cudaCSRSpMV execution time: " << spmv_timer.stop() << " ms." << endl;

	// step 8. inter-warp value calibration
	//bhsparse_timer cali_timer;
	//cali_timer.start();

	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_WRITE, _svm_y, _m * sizeof(value_type), 0, 0, 0);
	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_READ, _svm_synchronizer_idx, _threadbunch_num * sizeof(int), 0, 0, 0);
    err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_READ, _svm_synchronizer_val, _threadbunch_num * sizeof(value_type), 0, 0, 0);

	for (int i = 0; i < _threadbunch_num; i++)
	{
		_svm_y[_svm_synchronizer_idx[i]] += _svm_synchronizer_val[i];
	}

	//double cali_time = cali_timer.stop();
	//cout << "[2/3] Inter-warp value calibration execution time: " << cali_time << " ms." << endl;

	// step 7. speculative execution

	// make a duplicate that contains original values
	//bhsparse_timer spec_timer;
	//spec_timer.start();

	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_READ, _svm_dirty_counter, sizeof(int), 0, 0, 0);
	//cout << "dirty_counter = " << dirty_counter << endl;

	if (_svm_dirty_counter[0])
	{
		// copy a temp y
		err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_WRITE, _svm_y_temp, _m * sizeof(value_type), 0, 0, 0);
		memcpy(_svm_y_temp, _svm_y, _m * sizeof(value_type));

		// read back speculative array
		err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_READ, _svm_csrRowPtrA, (_m+1)  * sizeof(int), 0, 0, 0);
		err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_READ, _svm_speculator, 2 * _partition_num * sizeof(int), 0, 0, 0);

		for (int i = 0; i < _svm_dirty_counter[0]; i++)
		{
			// get start index
            int spec_start = _svm_speculator[2 * i];
            // get stop  index
            int spec_stop  = _svm_speculator[2 * i + 1];
			//cout << "Doing speculative " << i <<  " - start = " << spec_start << ", stop = " << spec_stop << endl;

			// clear the area in final vector y
			//memset(&_h_y[spec_start], 0, (spec_stop - spec_start) * sizeof(value_type));

			int y_ptr = spec_start;
			int row_offset = _svm_csrRowPtrA[spec_start];

			for (int j = spec_start + 1; j <= spec_stop + 1; j++)
			{
				int row_offset_next = _svm_csrRowPtrA[j];

				// the row a is not an empty row
				if (row_offset != row_offset_next)
				{
                    
					if (j-1 == spec_stop)
					{
						_svm_y[j-1] += _svm_y_temp[y_ptr];
						_svm_y_temp[j-1] += _svm_y_temp[y_ptr];
					}
					else
						_svm_y[j-1] = _svm_y_temp[y_ptr];
					y_ptr++;
				}
				else
				{
					_svm_y[j-1] = 0;
				}
				// if it is en empty row, do nothing

				row_offset = row_offset_next;
			}
		}

		err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_y_temp, 0, 0, 0 );
	    err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_speculator, 0, 0, 0 );
	    err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_csrRowPtrA, 0, 0, 0 );
	}

	//double spec_time = spec_timer.stop();
	//cout << "[3/3] Speculative execution time: " << spec_time << " ms." << endl;

	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_y, 0, 0, 0 );
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_dirty_counter, 0, 0, 0 );
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_synchronizer_idx, 0, 0, 0 );
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_synchronizer_val, 0, 0, 0 );

    return err;
}

int bhsparse_spmv_opencl::get_y()
{
	int err = CL_SUCCESS;

    // copy svm_y to h_y
	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_READ, _svm_y, _m * sizeof(value_type), 0, 0, 0);
    memcpy(_h_y, _svm_y, _m * sizeof(value_type));
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_y, 0, 0, 0 );

    return err;
}

int bhsparse_spmv_opencl::prepare_mem(int m, int n, int nnzA,
                                      int *csrRowPtrA, int *csrColIdxA, value_type *csrValA,
                                      value_type *x, value_type *y)
{
    int err = CL_SUCCESS;

    _m = m;
    _n = n;
    _nnzA = nnzA;

    _h_y = y;

    // malloc mem space and copy data from host to device

    // Matrix A
	_svm_csrRowPtrA = (int *)clSVMAlloc(_cxLocalContext, CL_MEM_READ_ONLY, (_m+1)  * sizeof(int), 0 );
	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_WRITE, _svm_csrRowPtrA, (_m+1)  * sizeof(int), 0, 0, 0);
	memcpy(_svm_csrRowPtrA, csrRowPtrA, (_m+1)  * sizeof(int));
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_csrRowPtrA, 0, 0, 0 );

    // prepare shared virtual memory (unified memory)
#if USE_SVM_ALWAYS
    cout << endl << "bhSPARSE is always using shared virtual memory.";
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
    // prepare device memory
#else
	cout << endl << "bhSPARSE is using dedicated GPU memory for [col_idx_A, val_A and x] and shared virtual memory for the other arrays.";
    _d_csrValA    = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY, _nnzA  * sizeof(value_type), NULL, &err);
    if(err != CL_SUCCESS) return err;
    err = clEnqueueWriteBuffer(_cqLocalCommandQueue, _d_csrValA, CL_TRUE, 0, _nnzA  * sizeof(value_type), csrValA, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;
	
	_d_csrColIdxA = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY, _nnzA  * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) return err;
	err = clEnqueueWriteBuffer(_cqLocalCommandQueue, _d_csrColIdxA, CL_TRUE, 0, _nnzA  * sizeof(int), csrColIdxA, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;
    
    _d_x    = clCreateBuffer(_cxLocalContext, CL_MEM_READ_ONLY, _n  * sizeof(value_type), NULL, &err);
    if(err != CL_SUCCESS) return err;
    err = clEnqueueWriteBuffer(_cqLocalCommandQueue, _d_x, CL_TRUE, 0, _n  * sizeof(value_type), x, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;
#endif

    // Vector y
    _svm_y = (value_type *)clSVMAlloc(_cxLocalContext, CL_MEM_READ_WRITE, _m  * sizeof(value_type), 0 );
	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_WRITE, _svm_y, _m * sizeof(value_type), 0, 0, 0);
	memset(_svm_y, 0, _m * sizeof(value_type));
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_y, 0, 0, 0 );

    // Vector y_temp
    _svm_y_temp = (value_type *)clSVMAlloc(_cxLocalContext, CL_MEM_READ_WRITE, _m  * sizeof(value_type), 0 );
	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_WRITE, _svm_y_temp, _m * sizeof(value_type), 0, 0, 0);
	memset(_svm_y_temp, 0, _m * sizeof(value_type));
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_y_temp, 0, 0, 0 );

    // compute work parameters

    // work-partition size and number
    _partition_size = THREADBUNCH * SEG_H;
    _partition_num  = ceil((double)_nnzA / (double)_partition_size);
    _threadbunch_num = ceil((double)_partition_num / (double)STEP);

    //cout << "THREADBUNCH SIZE = " << THREADBUNCH
    //     << ", SEG_H SIZE = " << SEG_H
    //     << ", PARTITION SIZE = " << _partition_size
    //     << ", STEP = " << STEP
    //     << endl;

    // allocate buffer for speculative execution
	_svm_speculator = (int *)clSVMAlloc(_cxLocalContext, CL_MEM_READ_WRITE, 2 * _partition_num * sizeof(int), 0 );
	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_WRITE, _svm_speculator, 2 * _partition_num * sizeof(int), 0, 0, 0);
	memset(_svm_speculator, 0, 2 * _partition_num * sizeof(int));
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_speculator, 0, 0, 0 );

	_svm_dirty_counter = (int *)clSVMAlloc(_cxLocalContext, CL_MEM_READ_WRITE, sizeof(int), 0 );
	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_WRITE, _svm_dirty_counter, sizeof(int), 0, 0, 0);
	memset(_svm_dirty_counter, 0, sizeof(int));
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_dirty_counter, 0, 0, 0 );

	// allocate buffer for inter-warp y entry value store
	_svm_synchronizer_idx = (int *)clSVMAlloc(_cxLocalContext, CL_MEM_READ_WRITE, _threadbunch_num * sizeof(int), 0 );
	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_WRITE, _svm_synchronizer_idx, _threadbunch_num * sizeof(int), 0, 0, 0);
	memset(_svm_synchronizer_idx, 0, _threadbunch_num * sizeof(int));
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_synchronizer_idx, 0, 0, 0 );

	_svm_synchronizer_val = (value_type *)clSVMAlloc(_cxLocalContext, CL_MEM_READ_WRITE, _threadbunch_num * sizeof(value_type), 0 );
	err = clEnqueueSVMMap(_cqLocalCommandQueue, CL_TRUE, CL_MAP_WRITE, _svm_synchronizer_val, _threadbunch_num * sizeof(value_type), 0, 0, 0);
	memset(_svm_synchronizer_val, 0, _threadbunch_num * sizeof(value_type));
	err = clEnqueueSVMUnmap(_cqLocalCommandQueue, _svm_synchronizer_val, 0, 0, 0 );

    return err;
}
	

#endif // BHSPARSE_SPMV_OPENCL_H
