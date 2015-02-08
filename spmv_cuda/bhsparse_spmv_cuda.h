#ifndef BHSPARSE_SPMV_CUDA_H
#define BHSPARSE_SPMV_CUDA_H

#include "common.h"
#include "kernels_cuda.h"

class bhsparse_spmv_cuda
{
public:
    bhsparse_spmv_cuda();
    int init_platform();
    int prepare_mem(int m, int n, int nnzA, int *csrRowPtrA, int *csrColIdxA, value_type *csrValA,
                    value_type *x, value_type *y);
    int run_benchmark();
    void get_y();
    int free_platform();
    int free_mem();

private:
    int _m;
    int _n;
    int _nnzA;

    // A
    value_type *_svm_csrValA;
    int        *_svm_csrRowPtrA;
    int        *_svm_csrColIdxA;

    value_type *_d_csrValA;
    int        *_d_csrColIdxA;

    // x and y
    value_type *_svm_x;
    value_type *_d_x;
    cudaTextureObject_t  _svm_x_tex;
    cudaTextureObject_t  _d_x_tex;

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

bhsparse_spmv_cuda::bhsparse_spmv_cuda()
{
}

int bhsparse_spmv_cuda::init_platform()
{
    int err = 0;

    // set device
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    cout << "Device [" <<  device_id << "] " << deviceProp.name << ", "
         << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << endl;

    return err;
}

int bhsparse_spmv_cuda::free_platform()
{
    int err = 0;

    return err;
}

int bhsparse_spmv_cuda::free_mem()
{
    int err = 0;

    checkCudaErrors(cudaDeviceSynchronize());

    // A
    checkCudaErrors(cudaFree(_svm_csrRowPtrA));
#if USE_SVM_ALWAYS
    checkCudaErrors(cudaFree(_svm_csrValA));
    checkCudaErrors(cudaFree(_svm_csrColIdxA));
#else
    checkCudaErrors(cudaFree(_d_csrValA));
    checkCudaErrors(cudaFree(_d_csrColIdxA));
#endif

    // vectors
#if USE_SVM_ALWAYS
    cudaDestroyTextureObject(_svm_x_tex);
    checkCudaErrors(cudaFree(_svm_x));
#else
    cudaDestroyTextureObject(_d_x_tex);
    checkCudaErrors(cudaFree(_d_x));
#endif
    checkCudaErrors(cudaFree(_svm_y));
    checkCudaErrors(cudaFree(_svm_y_temp));

    // other buffers
    checkCudaErrors(cudaFree(_svm_speculator));
    checkCudaErrors(cudaFree(_svm_dirty_counter));
    checkCudaErrors(cudaFree(_svm_synchronizer_idx));
    checkCudaErrors(cudaFree(_svm_synchronizer_val));

    return err;
}


int bhsparse_spmv_cuda::run_benchmark()

{
    int err = BHSPARSE_SUCCESS;

    _threadbunch_per_block = THREADGROUP / THREADBUNCH;

    // compute kernel launch parameters
    int num_threads = THREADGROUP;
    int num_blocks = ceil((double)_threadbunch_num / (double)_threadbunch_per_block);

    //if (iter == 1)
    //    cout << "#PARTITIONS = " << _partition_num
    //         << ", #THREADBUNCH = " << _threadbunch_num
    //         << ", #THREADS/BLOCK = " << num_threads
    //         << ", #BLOCKS = " << num_blocks << endl;

    //    bhsparse_timer spmv_timer;
    //    spmv_timer.start();

    // clear dirty_counter
    checkCudaErrors(cudaDeviceSynchronize());
    _svm_dirty_counter[0] = 0;

#if USE_SVM_ALWAYS
    SpMV_kernel<<< num_blocks, num_threads >>>
                   (_svm_csrRowPtrA, _svm_csrColIdxA, _svm_csrValA,
                    _svm_x_tex, _svm_x, _svm_y, _svm_speculator, _svm_dirty_counter,
                    _svm_synchronizer_idx, _svm_synchronizer_val,
                    _partition_size, _partition_num, _threadbunch_num,
                    _nnzA, _m);
#else
    SpMV_kernel<<< num_blocks, num_threads >>>
                   (_svm_csrRowPtrA, _d_csrColIdxA, _d_csrValA,
                    _d_x_tex, _d_x, _svm_y, _svm_speculator, _svm_dirty_counter,
                    _svm_synchronizer_idx, _svm_synchronizer_val,
                    _partition_size, _partition_num, _threadbunch_num,
                    _nnzA, _m);
#endif

    checkCudaErrors(cudaDeviceSynchronize());

    //    double spmv_time = spmv_timer.stop() / (double)iter;
    //    cout << "[1/3] speculative stage: " << spmv_time << " ms." << endl;

    // step 8. inter-warp value calibration
    //    bhsparse_timer cali_timer;
    //    cali_timer.start();

    for (int i = 0; i < _threadbunch_num; i++)
    {
        _svm_y[_svm_synchronizer_idx[i]] += _svm_synchronizer_val[i];
    }

    //    double cali_time = cali_timer.stop();
    //    cout << "[2/3] synchronization: " << cali_time << " ms." << endl;

    // step 9. check prediction

    // make a duplicate that contains original values
    //    bhsparse_timer spec_timer;
    //    spec_timer.start();

    if (_svm_dirty_counter[0])
    {
        // copy a temp y
        memcpy(_svm_y_temp, _svm_y, _m * sizeof(value_type));

        for (int i = 0; i < _svm_dirty_counter[0]; i++)
        {
            // get start index
            int spec_start = _svm_speculator[2 * i];
            // get stop  index
            int spec_stop  = _svm_speculator[2 * i + 1];

            int y_ptr = spec_start;
            int row_offset = _svm_csrRowPtrA[spec_start];

            for (int j = spec_start + 1; j <= spec_stop + 1; j++)
            {
                int row_offset_next = _svm_csrRowPtrA[j];

                if (row_offset != row_offset_next) // the row a is not an empty row
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
                else // if it is en empty row, set to 0
                {
                    _svm_y[j-1] = 0;
                }

                row_offset = row_offset_next;
            }
        }
    }

    //    double spec_time = spec_timer.stop();
    //    cout << "[3/3] checking prediction: " << spec_time << " ms." << endl;

    return err;
}

void bhsparse_spmv_cuda::get_y()
{
    // copy svm_y to h_y
    memcpy(_h_y, _svm_y, _m * sizeof(value_type));

    return;
}

int bhsparse_spmv_cuda::prepare_mem(int m, int n, int nnzA,
                                    int *csrRowPtrA, int *csrColIdxA, value_type *csrValA,
                                    value_type *x, value_type *y)
{
    int err = BHSPARSE_SUCCESS;

    _m = m;
    _n = n;
    _nnzA = nnzA;

    _h_y = y;

    checkCudaErrors(cudaMallocManaged(&_svm_csrRowPtrA, (_m+1) * sizeof(int)));
    memcpy(_svm_csrRowPtrA, csrRowPtrA, (_m+1) * sizeof(int));

    // prepare shared virtual memory (unified memory)
#if USE_SVM_ALWAYS
    cout << endl << "bhSPARSE is always using shared virtual memory (unified memory).";
    // Matrix A
    checkCudaErrors(cudaMallocManaged(&_svm_csrColIdxA, _nnzA  * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&_svm_csrValA,    _nnzA  * sizeof(value_type)));
    memcpy(_svm_csrColIdxA, csrColIdxA, _nnzA  * sizeof(int));
    memcpy(_svm_csrValA,    csrValA,    _nnzA  * sizeof(value_type));

    checkCudaErrors(cudaMallocManaged(&_svm_x, _n  * sizeof(value_type)));
    memcpy(_svm_x,    x,    _n  * sizeof(value_type));
    // prepare device memory
#else
    cout << endl << "bhSPARSE is using dedicated GPU memory for [col_idx_A, val_A and x] and shared virtual memory (unified memory) for the other arrays.";
    checkCudaErrors(cudaMalloc((void **)&_d_csrColIdxA, _nnzA  * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&_d_csrValA,    _nnzA  * sizeof(value_type)));
    checkCudaErrors(cudaMemcpy(_d_csrColIdxA, csrColIdxA, _nnzA  * sizeof(int),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_d_csrValA,    csrValA,    _nnzA  * sizeof(value_type),   cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **)&_d_x, _n * sizeof(value_type)));
    checkCudaErrors(cudaMemcpy(_d_x, x, _n * sizeof(value_type), cudaMemcpyHostToDevice));
#endif

    // create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
#if USE_SVM_ALWAYS
    resDesc.res.linear.devPtr = _svm_x;
#else
    resDesc.res.linear.devPtr = _d_x;
#endif
    resDesc.res.linear.sizeInBytes = _n * sizeof(value_type);
    if (sizeof(value_type) == sizeof(float))
    {
        resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x = 32; // bits per channel
    }
    else if (sizeof(value_type) == sizeof(double))
    {
        resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.desc.y = 32; // bits per channel
    }
    else
    {
        return BHSPARSE_UNSUPPORTED_VALUE_TYPE;
    }

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    // create texture object: we only have to do this once!
#if USE_SVM_ALWAYS
    _svm_x_tex = 0;
    cudaCreateTextureObject(&_svm_x_tex, &resDesc, &texDesc, NULL);
#else
    _d_x_tex = 0;
    cudaCreateTextureObject(&_d_x_tex, &resDesc, &texDesc, NULL);
#endif

    // Vector y
    checkCudaErrors(cudaMallocManaged(&_svm_y, _m  * sizeof(value_type)));
    memcpy(_svm_y,    y,    _m  * sizeof(value_type));

    // Vector y_temp
    checkCudaErrors(cudaMallocManaged(&_svm_y_temp, _m * sizeof(value_type)));
    memset(_svm_y_temp, 0, _m * sizeof(value_type));

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
    checkCudaErrors(cudaMallocManaged(&_svm_speculator, 2 * _partition_num * sizeof(int)));
    memset(_svm_speculator, 0, 2 * _partition_num * sizeof(int));

    checkCudaErrors(cudaMallocManaged(&_svm_dirty_counter, sizeof(int)));
    memset(_svm_dirty_counter, 0, sizeof(int));

    // allocate buffer for inter-warp y entry value store
    checkCudaErrors(cudaMallocManaged(&_svm_synchronizer_idx, _threadbunch_num * sizeof(int)));
    memset(_svm_synchronizer_idx, 0, _threadbunch_num * sizeof(int));

    checkCudaErrors(cudaMallocManaged(&_svm_synchronizer_val, _threadbunch_num * sizeof(value_type)));
    memset(_svm_synchronizer_val, 0, _threadbunch_num * sizeof(value_type));

    return err;
}

#endif // BHSPARSE_SPMV_CUDA_H
