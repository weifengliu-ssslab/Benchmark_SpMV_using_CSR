#include "common.h"
#include "bhsparse_spmv_opencl.h"
#include "csr_vector_opencl.h"
#include "mmio.h"

void call_cusp_ref(int m, int n, int nnz,  
                   int *csrRowPtrA, int *csrColIdxA, value_type *csrValA,
                   value_type *x, value_type *y, value_type *y_ref)
{
    int err = 0;

    value_type *y_cusp_ref = (value_type *)malloc(m * sizeof(value_type));
    memset(y_cusp_ref, 0, m * sizeof(value_type));

    double gb = (double)((m + 1 + nnz) * sizeof(int) + (2 * nnz + m) * sizeof(value_type));
    double gflop = (double)(2 * nnz);

// run CUSP START
    csr_vector_opencl *cusp = new csr_vector_opencl();
    err = cusp->init_platform();

    char flags[64];
    sprintf(flags," -DUSE_DOUBLE=%d", USE_DOUBLE);

    err = cusp->init_kernels(string(flags));

    err = cusp->prepare_mem(m, n, nnz, csrRowPtrA, csrColIdxA, csrValA, x, y);

    err = cusp->run_benchmark();

    err = cusp->get_y();

    // compare ref and our results
    cout << endl << "Checking SpMV Correctness ... ";
    int error_count = 0;
    for (int i = 0; i < m; i++)
        if (y_ref[i] != y[i])
        {
            error_count++;
        }
    if (error_count)
    {
        cout << "NO PASS. Error count = " << error_count << " out of " << m << " entries.";
    }
    else
    {
        cout << "PASS!";
    }

    cout << "\n";

    // warmup
    for (int i = 0; i < 50; i++)
    {
        err = cusp->run_benchmark();
    }
    cusp->sync_device();

    // do benchmark

    bhsparse_timer cusp_timer;
    cusp_timer.start();

    for (int i = 0; i < NUM_RUN; i++)
    {
        err = cusp->run_benchmark();
    }

    cusp->sync_device();
    double time = cusp_timer.stop() / (double)NUM_RUN;

    cout << "CUSP SpMV time = " << time
         << " ms. Bandwidth = " << gb/(1.0e+6 * time)
         << " GB/s. GFlops = " << gflop/(1.0e+6 * time) << " GFlops." << endl;

    err = cusp->free_platform();
    err = cusp->free_mem();
// run CUSP STOP

    free(y_cusp_ref);

    return;
}

int call_bhsparse(const char *datasetpath)
{
    int err = 0;

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
        cout << "Could not process Matrix Market banner.\n";
        return -2;
    }

    if ( mm_is_complex( matcode ) )
    {
        cout <<"Sorry, data type 'COMPLEX' is not supported.\n";
        return -3;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*cout << "type = Pattern\n";*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*cout << "type = real\n";*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*cout << "type = integer\n";*/ }

    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report)) != 0)
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
    
    for (int i = 0; i < nnzA; i++)
    {
        csrValA[i] = 1; //rand() % 10;
    }

    value_type *x = (value_type *)malloc(n * sizeof(value_type));
    
    for (int i = 0; i < n; i++)
        x[i] = 1; //rand() % 10;

    value_type *y = (value_type *)malloc(m * sizeof(value_type));
    value_type *y_ref = (value_type *)malloc(m * sizeof(value_type));

    // compute cpu results
    bhsparse_timer ref_timer;
    ref_timer.start();

    for (int iter = 0; iter < NUM_RUN; iter++)
    {
        for (int i = 0; i < m; i++)
        {
            value_type sum = 0;
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
                sum += x[csrColIdxA[j]] * csrValA[j];
            y_ref[i] = sum;
        }
    }
    
    double ref_time = ref_timer.stop() / (double)NUM_RUN;

    cout << "\ncpu sequential time = " << ref_time
         << " ms. Bandwidth = " << gb/(1.0e+6 * ref_time)
         << " GB/s. GFlops = " << gflop/(1.0e+6 * ref_time)  << " GFlops." << endl << endl;

    memset((void *)y, 0, m * sizeof(value_type));

    bhsparse_spmv_opencl *bhsparse = new bhsparse_spmv_opencl();
    err = bhsparse->init_platform();
    cout << "Initializing OpenCL platform ... ";
    if (!err)
        cout << "Done.\n";
    else
    {
        cout << "Failed. Error code = " << err << "\n";
        return err;
    }

    // pass opencl compile flags
    char flags[64];
    sprintf(flags," -DUSE_DOUBLE=%d -DTHREADBUNCH=%d -DSEG_H=%d -DSTEP=%d", USE_DOUBLE, THREADBUNCH, SEG_H, STEP);

    err = bhsparse->init_kernels(string(flags));
    

    // test CUSP v0.4.0
    call_cusp_ref(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, y_ref);

    err = bhsparse->prepare_mem(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y);

    err = bhsparse->run_benchmark();

    err = bhsparse->get_y();

    // compare ref and our results
    cout << endl << "Checking SpMV Correctness ... ";
    int error_count = 0;
    for (int i = 0; i < m; i++)
        if (y_ref[i] != y[i])
        {
            error_count++;
            if (i < 10) printf("rowid = %d, ref = %f, y = %f \n", i, y_ref[i], y[i]);
        }
    if (error_count)
    {
        cout << "NO PASS. Error count = " << error_count << " out of " << m << " entries.";
    }
    else
    {
        cout << "PASS!";
    }

    cout << "\n";

    // warmup
    for (int i = 0; i < 50; i++)
    {
        err = bhsparse->run_benchmark();
    }
    bhsparse->sync_device();

    // do benchmark

    bhsparse_timer spmv_timer;
    spmv_timer.start();

    for (int i = 0; i < NUM_RUN; i++)
    {
        err = bhsparse->run_benchmark();
    }

    bhsparse->sync_device();
    double time = spmv_timer.stop() / (double)NUM_RUN;

    cout << "bhSPARSE SpMV time = " << time
         << " ms. Bandwidth = " << gb/(1.0e+6 * time)
         << " GB/s. GFlops = " << gflop/(1.0e+6 * time) << " GFlops." << endl;

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
    int err = 0;

    int argi = 1;

    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }

    cout << "------------------------------------------------------" << endl;

    cout << "----------" << filename << "----------" << endl;

    // report precision of floating-point
    if (sizeof(value_type) == 4)
    {
        cout << "PRECISION = " << "32-bit Single Precision" << endl;
    }
    else if (sizeof(value_type) == 8)
    {
        cout << "PRECISION = " << "64-bit Double Precision" << endl;
    }
    else
    {
        cout << "Wrong precision. Program exit!" << endl;
        return 0;
    }

    cout << "RUN SpMV " << NUM_RUN << " times" << endl;

    err = call_bhsparse(filename);

    cout << "------------------------------------------------------" << endl;

    return 0;
}

