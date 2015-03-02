#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#if USE_DOUBLE
typedef double   vT;
#else
typedef float   vT;
#endif

// csr_spmv kernel extracted from the CUSP library v0.4.0
__kernel void
spmv_csr_vector_kernel(const int num_rows,
                       __global const int * Ap,
                       __global const int * Aj,
                       __global const vT * Ax,
                       __global const vT * x,
                       __global vT * y,
                       volatile __local vT           *sdata, //[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2]
                       volatile __local int           *ptrs, //[VECTORS_PER_BLOCK][2]
                       const unsigned int VECTORS_PER_BLOCK,
                       const unsigned int THREADS_PER_VECTOR)
{
    const int local_id    = get_local_id(0);

    const int thread_lane = local_id % THREADS_PER_VECTOR; // thread index within the vector
    const int vector_id   = get_global_id(0)   /  THREADS_PER_VECTOR;               // global vector index
    const int vector_lane = local_id /  THREADS_PER_VECTOR;               // vector index within the block
    const int num_vectors = VECTORS_PER_BLOCK * get_num_groups(0);                   // total number of active vectors

    for(int row = vector_id; row < num_rows; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[vector_lane * 2 + thread_lane] = Ap[row + thread_lane];

        const int row_start = ptrs[vector_lane * 2 + 0];                   //same as: row_start = Ap[row];
        const int row_end   = ptrs[vector_lane * 2 + 1];                   //same as: row_end   = Ap[row+1];

        // initialize local sum
        vT sum = 0;

        if (THREADS_PER_VECTOR == 64 && row_end - row_start > 64)
        {
            // ensure aligned memory access to Aj and Ax

            int jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

            // accumulate local sums
            if(jj >= row_start && jj < row_end)
                sum += Ax[jj] * x[Aj[jj]]; 

            // accumulate local sums
            for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR)
                sum += Ax[jj] * x[Aj[jj]]; 
        }
        else
        {
            // accumulate local sums
            for(int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
                sum += Ax[jj] * x[Aj[jj]]; 
        }

        // store local sum in shared memory
        sdata[local_id] = sum;

        // reduce local sums to row sum
        if (THREADS_PER_VECTOR > 32) sdata[local_id] = sum = sum + sdata[local_id + 32];
        if (THREADS_PER_VECTOR > 16) sdata[local_id] = sum = sum + sdata[local_id + 16];
        if (THREADS_PER_VECTOR >  8) sdata[local_id] = sum = sum + sdata[local_id +  8];
        if (THREADS_PER_VECTOR >  4) sdata[local_id] = sum = sum + sdata[local_id +  4];
        if (THREADS_PER_VECTOR >  2) sdata[local_id] = sum = sum + sdata[local_id +  2];
        if (THREADS_PER_VECTOR >  1) sdata[local_id] = sum = sum + sdata[local_id +  1];

        // first thread writes the result
        if (thread_lane == 0)
            y[row] = sdata[local_id];
    }
}
