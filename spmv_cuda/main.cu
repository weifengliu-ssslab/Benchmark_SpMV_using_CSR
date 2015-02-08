#include "common.h"
#include "bhsparse_spmv_cuda.h"
#include "mmio.h"

cusparseStatus_t cusparse_spmv(cusparseHandle_t handle, cusparseMatDescr_t descr, 
                   int m, int n, int nnz, 
                   int *csrRowPtrA, int *csrColIdxA, double *csrValA, 
                   double *x, double *y, double alpha, double beta)
{
    return cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha, descr, csrValA, csrRowPtrA, csrColIdxA, x, &beta, y);
}

cusparseStatus_t cusparse_spmv(cusparseHandle_t handle, cusparseMatDescr_t descr,
                   int m, int n, int nnz,
                   int *csrRowPtrA, int *csrColIdxA, float *csrValA,
                   float *x, float *y, float alpha, float beta)
{
    return cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha, descr, csrValA, csrRowPtrA, csrColIdxA, x, &beta, y);
}

template <unsigned int THREADS_PER_VECTOR>
void cusp_spmv(int m, int n, int nnz, int *svm_csrRowPtrA, int *svm_csrColIdxA, double *svm_csrValA, double *svm_x, double *svm_y)
{
    const size_t THREADS_PER_BLOCK  = 128;
    const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

    const size_t NUM_BLOCKS = ceil((double)m / (double)VECTORS_PER_BLOCK);

    spmv_csr_vector_kernel
            <int, double, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>
            <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>
            (m, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y);
}

template <unsigned int THREADS_PER_VECTOR>
void cusp_spmv(int m, int n, int nnz, int *svm_csrRowPtrA, int *svm_csrColIdxA, float *svm_csrValA, float *svm_x, float *svm_y)
{
    const size_t THREADS_PER_BLOCK  = 128;
    const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

    const size_t NUM_BLOCKS = ceil((double)m / (double)VECTORS_PER_BLOCK);

    spmv_csr_vector_kernel
            <int, float, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>
            <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>
            (m, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y);
}

void call_cusparse_ref(int m, int n, int nnz, 
                       int *csrRowPtrA, int *csrColIdxA, value_type *csrValA, 
                       value_type *x, value_type *y, value_type *y_ref)
{
    // prepare shared virtual memory (unified memory)
#if USE_SVM_ALWAYS
    cout << endl << "cuSPARSE is using shared virtual memory (unified memory).";
    int *svm_csrRowPtrA;
    int *svm_csrColIdxA;
    value_type *svm_csrValA;
    value_type *svm_x;
    value_type *svm_y;

    checkCudaErrors(cudaMallocManaged(&svm_csrRowPtrA, (m+1) * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&svm_csrColIdxA, nnz  * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&svm_csrValA,    nnz  * sizeof(value_type)));
    memcpy(svm_csrRowPtrA, csrRowPtrA, (m+1) * sizeof(int));
    memcpy(svm_csrColIdxA, csrColIdxA, nnz  * sizeof(int));
    memcpy(svm_csrValA,    csrValA,    nnz  * sizeof(value_type));

    checkCudaErrors(cudaMallocManaged(&svm_x, n  * sizeof(value_type)));
    memcpy(svm_x,    x,    n  * sizeof(value_type));
    checkCudaErrors(cudaMallocManaged(&svm_y, m  * sizeof(value_type)));
    memcpy(svm_y,    y,    m  * sizeof(value_type));
    // prepare device memory
#else
    cout << endl << "cuSPARSE is using dedicated GPU memory.";
    int *d_csrRowPtrA;
    int *d_csrColIdxA;
    value_type *d_csrValA;
    value_type *d_x;
    value_type *d_y;

    checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, (m+1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrColIdxA, nnz  * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrValA,    nnz  * sizeof(value_type)));
    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, csrRowPtrA, (m+1) * sizeof(int),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColIdxA, csrColIdxA, nnz  * sizeof(int),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrValA,    csrValA,    nnz  * sizeof(value_type),   cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **)&d_x, n * sizeof(value_type)));
    checkCudaErrors(cudaMemcpy(d_x, x, n * sizeof(value_type), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **)&d_y, m  * sizeof(value_type)));
    checkCudaErrors(cudaMemcpy(d_y, y, m * sizeof(value_type), cudaMemcpyHostToDevice));
#endif

    double gb = (double)((m + 1 + nnz) * sizeof(int) + (2 * nnz + m) * sizeof(value_type));
    double gflop = (double)(2 * nnz);

// run cuSPARSE START
    cusparseHandle_t handle = 0;
    cusparseStatus_t status;
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        printf("CUSPARSE initialization error\n");
        //return -1;
    }

    cusparseMatDescr_t descr = 0;
    status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        printf("CUSPARSE cusparseCreateMatDescr error\n");
        //return -2;
    }
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    value_type alpha = 1.0;
    value_type beta = 0.0;

    checkCudaErrors(cudaDeviceSynchronize());
    bhsparse_timer cusparse_timer;
    cusparse_timer.start();
    for (int i = 0; i < NUM_RUN; i++)
    {
#if USE_SVM_ALWAYS
        status = cusparse_spmv(handle, descr, m, n, nnz, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y, alpha, beta);
#else
        status = cusparse_spmv(handle, descr, m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y, alpha, beta);
#endif
    }
    checkCudaErrors(cudaDeviceSynchronize());
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        printf("CUSPARSE cusparseS/Dcsrmv error\n");
        //return -3;
    }
    double cusparseTime = cusparse_timer.stop() / NUM_RUN;

    cout << endl << "Checking cuSPARSE SpMV Correctness ... ";

#if USE_SVM_ALWAYS == 0
    value_type *y_cusparse_ref = (value_type *)malloc(m * sizeof(value_type));
    checkCudaErrors(cudaMemcpy(y_cusparse_ref, d_y, m * sizeof(value_type), cudaMemcpyDeviceToHost));
#endif

    int error_count = 0;
    for (int i = 0; i < m; i++)
#if USE_SVM_ALWAYS
        if (y_ref[i] != svm_y[i])
            error_count++;
#else
        if (y_ref[i] != y_cusparse_ref[i])
            error_count++;
#endif
    if (error_count)
        cout << "NO PASS. Error count = " << error_count << " out of " << m << " entries.";
    else
        cout << "PASS!";
    cout << endl;

    cout << "cuSPARSE time = " << cusparseTime
         << " ms. Bandwidth = " << gb/(1.0e+6 * cusparseTime)
         << " GB/s. GFlops = " << gflop/(1.0e+6 * cusparseTime)  << " GFlops." << endl << endl;
// run cuSPARSE STOP

#if USE_SVM_ALWAYS
    checkCudaErrors(cudaFree(svm_csrValA));
    checkCudaErrors(cudaFree(svm_csrRowPtrA));
    checkCudaErrors(cudaFree(svm_csrColIdxA));
    checkCudaErrors(cudaFree(svm_x));
    checkCudaErrors(cudaFree(svm_y));
#else
    free(y_cusparse_ref);
    checkCudaErrors(cudaFree(d_csrRowPtrA));
    checkCudaErrors(cudaFree(d_csrColIdxA));
    checkCudaErrors(cudaFree(d_csrValA));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
#endif

    return;
}




void call_cusp_ref(int m, int n, int nnz,  
                   int *csrRowPtrA, int *csrColIdxA, value_type *csrValA,
                   value_type *x, value_type *y, value_type *y_ref)
{
#if USE_SVM_ALWAYS
    cout << endl << "CUSP is using shared virtual memory (unified memory).";
    int *svm_csrRowPtrA;
    int *svm_csrColIdxA;
    value_type *svm_csrValA;
    value_type *svm_x;
    value_type *svm_y;

    checkCudaErrors(cudaMallocManaged(&svm_csrRowPtrA, (m+1) * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&svm_csrColIdxA, nnz  * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&svm_csrValA,    nnz  * sizeof(value_type)));
    memcpy(svm_csrRowPtrA, csrRowPtrA, (m+1) * sizeof(int));
    memcpy(svm_csrColIdxA, csrColIdxA, nnz  * sizeof(int));
    memcpy(svm_csrValA,    csrValA,    nnz  * sizeof(value_type));

    checkCudaErrors(cudaMallocManaged(&svm_x, n  * sizeof(value_type)));
    memcpy(svm_x,    x,    n  * sizeof(value_type));
    checkCudaErrors(cudaMallocManaged(&svm_y, m  * sizeof(value_type)));
    memcpy(svm_y,    y,    m  * sizeof(value_type));
    // prepare device memory
#else
    cout << endl << "CUSP is using dedicated GPU memory.";
    int *d_csrRowPtrA;
    int *d_csrColIdxA;
    value_type *d_csrValA;
    value_type *d_x;
    value_type *d_y;

    checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, (m+1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrColIdxA, nnz  * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrValA,    nnz  * sizeof(value_type)));
    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, csrRowPtrA, (m+1) * sizeof(int),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColIdxA, csrColIdxA, nnz  * sizeof(int),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrValA,    csrValA,    nnz  * sizeof(value_type),   cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **)&d_x, n * sizeof(value_type)));
    checkCudaErrors(cudaMemcpy(d_x, x, n * sizeof(value_type), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **)&d_y, m  * sizeof(value_type)));
    checkCudaErrors(cudaMemcpy(d_y, y, m * sizeof(value_type), cudaMemcpyHostToDevice));
#endif


    double gb = (double)((m + 1 + nnz) * sizeof(int) + (2 * nnz + m) * sizeof(value_type));
    double gflop = (double)(2 * nnz);

// run CUSP START
    const int nnz_per_row = nnz / m;

    bhsparse_timer cusp_timer;
    cusp_timer.start();

    if (nnz_per_row <=  2)
    {
        for (int i = 0; i < NUM_RUN; i++)
#if USE_SVM_ALWAYS
            cusp_spmv<2>(m, n, nnz, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y);
#else
            cusp_spmv<2>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
#endif
    }
    else if (nnz_per_row <=  4)
    {
        for (int i = 0; i < NUM_RUN; i++)
#if USE_SVM_ALWAYS
            cusp_spmv<4>(m, n, nnz, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y);
#else
            cusp_spmv<4>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
#endif
    }
    else if (nnz_per_row <=  8)
    {
        for (int i = 0; i < NUM_RUN; i++)
#if USE_SVM_ALWAYS
            cusp_spmv<8>(m, n, nnz, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y);
#else
            cusp_spmv<8>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
#endif
    }
    else if (nnz_per_row <= 16)
    {
        for (int i = 0; i < NUM_RUN; i++)
#if USE_SVM_ALWAYS
            cusp_spmv<16>(m, n, nnz, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y);
#else
            cusp_spmv<16>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
#endif
    }
    else
    {
        for (int i = 0; i < NUM_RUN; i++)
#if USE_SVM_ALWAYS
            cusp_spmv<32>(m, n, nnz, svm_csrRowPtrA, svm_csrColIdxA, svm_csrValA, svm_x, svm_y);
#else
            cusp_spmv<32>(m, n, nnz, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_x, d_y);
#endif
    }

    checkCudaErrors(cudaDeviceSynchronize());
    double cuspTime = cusp_timer.stop() / NUM_RUN;

    cout << endl << "Checking CUSP SpMV Correctness ... ";

#if USE_SVM_ALWAYS == 0
    value_type *y_cusp_ref = (value_type *)malloc(m * sizeof(value_type));
    checkCudaErrors(cudaMemcpy(y_cusp_ref, d_y, m * sizeof(value_type), cudaMemcpyDeviceToHost));
#endif

    int error_count = 0;
    for (int i = 0; i < m; i++)
#if USE_SVM_ALWAYS
        if (y_ref[i] != svm_y[i])
            error_count++;
#else
        if (y_ref[i] != y_cusp_ref[i])
            error_count++;
#endif
    if (error_count)
        cout << "NO PASS. Error count = " << error_count << " out of " << m << " entries.";
    else
        cout << "PASS!";
    cout << endl;

    cout << "CUSP time = " << cuspTime
         << " ms. Bandwidth = " << gb/(1.0e+6 * cuspTime)
         << " GB/s. GFlops = " << gflop/(1.0e+6 * cuspTime)  << " GFlops." << endl << endl;
// run CUSP STOP

#if USE_SVM_ALWAYS
    checkCudaErrors(cudaFree(svm_csrValA));
    checkCudaErrors(cudaFree(svm_csrRowPtrA));
    checkCudaErrors(cudaFree(svm_csrColIdxA));
    checkCudaErrors(cudaFree(svm_x));
    checkCudaErrors(cudaFree(svm_y));
#else
    free(y_cusp_ref);
    checkCudaErrors(cudaFree(d_csrRowPtrA));
    checkCudaErrors(cudaFree(d_csrColIdxA));
    checkCudaErrors(cudaFree(d_csrValA));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
#endif

    return;
}

void call_omp_ref(int m, int n, int nnz,
                  int *csrRowPtrA, int *csrColIdxA, value_type *csrValA,
                  value_type *x, value_type *y, value_type *y_ref)
{
#if USE_SVM_ALWAYS
    cout << endl << "OpenMP is using shared virtual memory (unified memory).";
    int *svm_csrRowPtrA;
    int *svm_csrColIdxA;
    value_type *svm_csrValA;
    value_type *svm_x;
    value_type *svm_y;

    // prepare shared virtual memory (unified memory)
    checkCudaErrors(cudaMallocManaged(&svm_csrRowPtrA, (m+1) * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&svm_csrColIdxA, nnz  * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&svm_csrValA,    nnz  * sizeof(value_type)));
    memcpy(svm_csrRowPtrA, csrRowPtrA, (m+1) * sizeof(int));
    memcpy(svm_csrColIdxA, csrColIdxA, nnz  * sizeof(int));
    memcpy(svm_csrValA,    csrValA,    nnz  * sizeof(value_type));

    checkCudaErrors(cudaMallocManaged(&svm_x, n  * sizeof(value_type)));
    memcpy(svm_x,    x,    n  * sizeof(value_type));
    checkCudaErrors(cudaMallocManaged(&svm_y, m  * sizeof(value_type)));
    memcpy(svm_y,    y,    m  * sizeof(value_type));
#else
    cout << endl << "OpenMP is using dedicated HOST memory.";
    value_type *y_omp_ref = (value_type *)malloc(m * sizeof(value_type));
#endif

    double gb = (double)((m + 1 + nnz) * sizeof(int) + (2 * nnz + m) * sizeof(value_type));
    double gflop = (double)(2 * nnz);

// run OpenMP START
    omp_set_num_threads(4);
    cout << endl << "OpenMP is using 4 threads.";
    checkCudaErrors(cudaDeviceSynchronize());

    bhsparse_timer omp_timer;
    omp_timer.start();

    for (int iter = 0; iter < NUM_RUN; iter++)
    {
        #pragma omp parallel for
        for (int i = 0; i < m; i++)
        {
            value_type sum = 0;
#if USE_SVM_ALWAYS
            for (int j = svm_csrRowPtrA[i]; j < svm_csrRowPtrA[i+1]; j++)
                sum += svm_x[svm_csrColIdxA[j]] * svm_csrValA[j];
            svm_y[i] = sum;
#else
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
                sum += x[csrColIdxA[j]] * csrValA[j];
            y_omp_ref[i] = sum;
#endif
        }
    }

    double ompTime = omp_timer.stop() / NUM_RUN;

    cout << endl << "Checking OpenMP SpMV Correctness ... ";
    int error_count = 0;
    for (int i = 0; i < m; i++)
#if USE_SVM_ALWAYS
        if (y_ref[i] != svm_y[i])
            error_count++;
#else
        if (y_ref[i] != y_omp_ref[i])
            error_count++;
#endif
    if (error_count)
        cout << "NO PASS. Error count = " << error_count << " out of " << m << " entries.";
    else
        cout << "PASS!";
    cout << endl;

    cout << "OpenMP time = " << ompTime
         << " ms. Bandwidth = " << gb/(1.0e+6 * ompTime)
         << " GB/s. GFlops = " << gflop/(1.0e+6 * ompTime)  << " GFlops." << endl << endl;
// run OpenMP STOP

#if USE_SVM_ALWAYS
    checkCudaErrors(cudaFree(svm_csrValA));
    checkCudaErrors(cudaFree(svm_csrRowPtrA));
    checkCudaErrors(cudaFree(svm_csrColIdxA));
    checkCudaErrors(cudaFree(svm_x));
    checkCudaErrors(cudaFree(svm_y));
#else
    free(y_omp_ref);
#endif

    return;
}

int call_bhsparse_small()
{
    int err = 0;

    int m, n, nnzA;

    int *csrColIdxA;
    int *csrRowPtrA;
    value_type *csrValA;

    m = 6;
    n = 6;
    nnzA = 15;

    csrColIdxA = (int *)malloc(nnzA * sizeof(int));
    csrRowPtrA = (int *)malloc((m+1) * sizeof(int));
    csrValA = (value_type *)malloc(nnzA * sizeof(value_type));

    int row_ptr[7]     = {0,       3,                9,    11, 11, 12,      15};
    int col_idx[15]    = {0, 2, 5, 0, 1, 2, 3, 4, 5, 2, 4,      4,  2, 3, 4};
    value_type val[15] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    memcpy(csrRowPtrA, row_ptr, (m+1) * sizeof(int));
    memcpy(csrColIdxA, col_idx, nnzA * sizeof(int));
    memcpy(csrValA, val, nnzA * sizeof(value_type));

    cout << "row_ptr = [ ";
    for (int i = 0; i < m + 1; i++)
        cout << csrRowPtrA[i] << ", ";
    cout << " ]" << endl;

    cout << "col_idx = [ ";
    for (int i = 0; i < nnzA; i++)
        cout << csrColIdxA[i] << ", ";
    cout << " ]" << endl;

    cout << "value   = [ ";
    for (int i = 0; i < nnzA; i++)
        cout << csrValA[i] << ", ";
    cout << " ]" << endl << endl;

    value_type *x = (value_type *)malloc(n * sizeof(value_type));
    for (int i = 0; i < n; i++)
        x[i] = 1.0;

    value_type *y = (value_type *)malloc(m * sizeof(value_type));
    value_type *y_ref = (value_type *)malloc(m * sizeof(value_type));

    // compute cpu results
    for (int i = 0; i < m; i++)
    {
        value_type sum = 0;
        for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
            sum += x[csrColIdxA[j]] * csrValA[j];
        y_ref[i] = sum;
    }

    memset(y, 0, m * sizeof(value_type));

    bhsparse_spmv_cuda *bhsparse = new bhsparse_spmv_cuda();
    err = bhsparse->init_platform();
    cout << "Initializing CUDA platform ... ";
    if (!err)
        cout << "Done.";
    else
        cout << "Failed. Error code = " << err;
    cout << endl;

    err = bhsparse->prepare_mem(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y);

    err = bhsparse->run_benchmark();

    cout << endl;
    // print y_ref
    cout << "(CPU) y = ";
    for (int i = 0; i < m; i++)
    {
        cout << y_ref[i] << ", ";
        if ((i+1) % 16 == 0)
            cout << endl;
    }
    cout << endl;

    // print y
    cout << "(GPU) y = ";
    for (int i = 0; i < m; i++)
    {
        cout << y[i] << ", ";
        if ((i+1) % 16 == 0)
            cout << endl;
    }
    cout << endl;

    // compare cpu and gpu results
    cout << endl << "Checking bhSPARSE SpMV Correctness ... ";
    int error_count = 0;
    for (int i = 0; i < m; i++)
        if (y_ref[i] != y[i])
        {
            error_count++;
            cout << "ERROR ROW [ " << i << " ] " "cpu = " << y_ref[i] << ", gpu = " << y[i] << endl;
        }
    if (error_count)
        cout << "NO PASS. Error count = " << error_count << " out of " << m << " entries.";
    else
        cout << "PASS!";
    cout << endl;

    free(y_ref);

    err = bhsparse->free_platform();
    err = bhsparse->free_mem();

    return err;
}

int call_bhsparse(const char *datasetpath)
{
    int err = 0;

    // report precision of floating-point
    char  *precision;
    if (sizeof(value_type) == 4)
    {
        precision = "32-bit Single Precision";
    }
    else if (sizeof(value_type) == 8)
    {
        precision = "64-bit Double Precision";
    }
    else
    {
        cout << "Wrong precision. Program exit!" << endl;
        return 0;
    }

    cout << "PRECISION = " << precision << endl;
    cout << "RUN SpMV " << NUM_RUN << " times" << endl;

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    value_type *csrValA;

    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;

    // load matrix
    if ((f = fopen(datasetpath, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        cout << "Could not process Matrix Market banner." << endl;
        return -2;
    }

    if ( mm_is_complex( matcode ) )
    {
        cout <<"Sorry, data type 'COMPLEX' is not supported. " << endl;
        return -3;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*cout << "type = Pattern" << endl;*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*cout << "type = real" << endl;*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*cout << "type = integer" << endl;*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric = 1;
        //cout << "symmetric = true" << endl;
    }
    else
    {
        //cout << "symmetric = false" << endl;
    }

    int *csrRowPtrA_counter = (int *)malloc((m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    int *csrRowIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    int *csrColIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    value_type *csrValA_tmp    = (value_type *)malloc(nnzA_mtx_report * sizeof(value_type));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi, idxj;
        double fval;
        int ival;

        if (isReal)
            int count = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        else if (isInteger)
        {
            int count = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            int count = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtrA_counter[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;

    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i-1];
        old_val = new_val;
    }

    nnzA = csrRowPtrA_counter[m];
    csrRowPtrA = (int *)malloc((m+1) * sizeof(int));
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    csrColIdxA = (int *)malloc(nnzA * sizeof(int));
    csrValA    = (value_type *)malloc(nnzA * sizeof(value_type));

    double gb = (double)((m + 1 + nnzA) * sizeof(int) + (2 * nnzA + m) * sizeof(value_type));
    double gflop = (double)(2 * nnzA);

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }

    // free tmp space
    free(csrColIdxA_tmp);
    free(csrValA_tmp);
    free(csrRowIdxA_tmp);
    free(csrRowPtrA_counter);

    cout << " ( " << m << ", " << n << " ) nnz = " << nnzA << endl;

    srand(time(NULL));
    for (int i = 0; i < nnzA; i++)
    {
        csrValA[i] = rand() % 10;
    }

    value_type *x = (value_type *)malloc(n * sizeof(value_type));
    for (int i = 0; i < n; i++)
        x[i] = rand() % 10;

    value_type *y = (value_type *)malloc(m * sizeof(value_type));
    value_type *y_ref = (value_type *)malloc(m * sizeof(value_type));

    // compute cpu results
    bhsparse_timer ref_timer;
    ref_timer.start();

    int ref_iter = 1;
    for (int iter = 0; iter < ref_iter; iter++)
    {
        for (int i = 0; i < m; i++)
        {
            value_type sum = 0;
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
                sum += x[csrColIdxA[j]] * csrValA[j];
            y_ref[i] = sum;
        }
    }

    double ref_time = ref_timer.stop() / (double)ref_iter;
    cout << "cpu sequential time = " << ref_time
         << " ms. Bandwidth = " << gb/(1.0e+6 * ref_time)
         << " GB/s. GFlops = " << gflop/(1.0e+6 * ref_time)  << " GFlops." << endl << endl;

    memset(y, 0, m * sizeof(value_type));


    bhsparse_spmv_cuda *bhsparse = new bhsparse_spmv_cuda();
    err = bhsparse->init_platform();


    // test OpenMP, cuSPARSE and CUSP v0.4.0
    call_cusp_ref(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, y_ref);
    call_cusparse_ref(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, y_ref);
    call_omp_ref(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, y_ref);

    // run bhSPARSE
    err = bhsparse->prepare_mem(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y);

    double time = 0.0;
    err = bhsparse->run_benchmark();
    bhsparse->get_y();

    // compare ref and our results
    cout << endl << "Checking bhSPARSE SpMV Correctness ... ";
    int error_count = 0;
    for (int i = 0; i < m; i++)
        if (y_ref[i] != y[i])
        {
            error_count++;
//            cout << "ROW [ " << i << " ] "
//                 << csrRowPtrA[i] << " - "
//                 << csrRowPtrA[i+1]
//                 << " warp = " <<  csrRowPtrA[i+1]/(31*256)
//                 << "\t cpu = " << y_ref[i]
//                 << ", \t gpu = " << y[i]
//                 << ", \t error = " << y_ref[i] - y[i]
//                 << endl;
        }

    if (error_count)
        cout << "NO PASS. Error count = " << error_count << " out of " << m << " entries.";
    else
    {
        cout << "PASS!";

        bhsparse_timer spmv_timer;
        spmv_timer.start();        
        for (int i = 0; i < NUM_RUN; i++)
        {
            err = bhsparse->run_benchmark();
        }
        time = spmv_timer.stop()/(double)NUM_RUN;

        cout << endl << "bhSPARSE time = " << time
             << " ms. Bandwidth = " << gb/(1.0e+6 * time)
             << " GB/s. GFlops = " << gflop/(1.0e+6 * time) << " GFlops." << endl;
    }

    err = bhsparse->free_platform();
    err = bhsparse->free_mem();

    free(csrRowPtrA);
    free(csrColIdxA);
    free(csrValA);
    free(x);
    free(y);
    free(y_ref);

    return err;
}

int main(int argc, char ** argv)
{
    int argi = 1;

    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }
    // use ./spmv example.mtx
    // launch compute
    int err = 0;

    if (strcmp(filename, "0") == 0)
        err = call_bhsparse_small();
    else
    {
        cout << "--------------" << filename << "--------------" << endl;
        err = call_bhsparse(filename);
    }
    cout << "------------------------------------------------------" << endl;

    return err;
}

