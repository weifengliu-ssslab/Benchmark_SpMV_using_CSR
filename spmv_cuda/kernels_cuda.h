#ifndef KERNELS_CUDA_H
#define KERNELS_CUDA_H

#include "common.h"
#include "utils_cuda.h"

__global__
void SpMV_kernel(const int                      *d_csrRowPtrA,
                 const int                      *d_csrColIdxA,
                 const value_type               *d_csrValA,
                 cudaTextureObject_t             d_x_tex,
                 const value_type               *d_x,
                 value_type                     *d_y,
                 int                            *d_speculator,
                 int                            *d_dirty_counter,
                 int                            *d_synchronizer_idx,
                 value_type                     *d_synchronizer_val,
                 const int                       partition_size,
                 const int                       partition_num,
                 const int                       tbunch_num,
                 const int                       nnz,
                 const int                       m)
{
    __shared__ unsigned int           s_ibuff[THREADGROUP];
    __shared__ value_type             s_interres[SEG_H * (THREADBUNCH + 1) * THREADGROUP / THREADBUNCH];
    volatile __shared__ value_type    s_tail[THREADGROUP / THREADBUNCH];
    __shared__ bool                   s_present[(THREADBUNCH + 1) * THREADGROUP / THREADBUNCH];
    volatile __shared__ bool          s_dirty[THREADGROUP / THREADBUNCH];

    const int local_id    = threadIdx.x;
    const int tbunch_tid  = local_id % THREADBUNCH; // thread bunch thread (lane) id
    const int tbunch_lid  = local_id / THREADBUNCH; // thread bunch local id
    const int tbunch_gid  = (blockIdx.x * blockDim.x + threadIdx.x) / THREADBUNCH; // thread bunch global id

    if (tbunch_gid >= tbunch_num) return;

    // step 1. binary search
    // compute offset
    const int partition_offset = tbunch_gid * STEP * partition_size;

    int bs_start_stop;
    // do binary search
    for (int i = tbunch_tid; i <= STEP; i += THREADBUNCH)
    {
        int boundary = partition_offset + i * partition_size;
        // clamp partition_nnz_start to [0, nnz]
        boundary = boundary > nnz ? nnz : boundary;
        bs_start_stop = binary_search_right_boundary_kernel<int>(d_csrRowPtrA, boundary, m+1) - 1;
    }

    int bs_start = 0;
    int bs_stop  = __shfl(bs_start_stop, 0);
    int tbunch_start = bs_stop;
 
    if (!tbunch_tid)
    {
        s_present[tbunch_lid * (THREADBUNCH + 1) + THREADBUNCH] = true;
        s_tail[tbunch_lid] = 0;
    }

    value_type *s_interres_local = &s_interres[tbunch_lid * SEG_H * (THREADBUNCH+1)];

    #pragma unroll
    for (int i = 0; i < STEP; i++)
    {
        // keep computation in the correct scope
        if (i + tbunch_gid * STEP >= partition_num)
            break;

        // set spec-flag to fasle, which means this partition has no empty row
        bool dirty = false;
        if (!tbunch_tid)
            s_dirty[tbunch_lid] = false;

        bs_start = bs_stop;
        bs_stop  = __shfl(bs_start_stop, i+1);

        //do fast track
        if (bs_start == bs_stop)
        {
            value_type fsum = 0;
            #pragma unroll
            for (int j = 0; j < SEG_H; j++)
            {
                int candidate_idx = partition_offset + i * partition_size + j * THREADBUNCH + tbunch_tid;

                // load indices & value, and keep index in the right scope [0,nnz)
                fsum += candidate_idx < nnz ? candidate(d_csrValA, d_x, d_x_tex, d_csrColIdxA, candidate_idx, 1.0) : 0;
            }

            fsum = sum_32_shfl<value_type>(fsum);

            if (!tbunch_tid)
            {
                int local_boundary = partition_offset + i * partition_size;
                fsum = (d_csrRowPtrA[bs_start] == local_boundary) ? fsum : (s_tail[tbunch_lid] + fsum);

                if (bs_start == tbunch_start)
                {
                    d_synchronizer_idx[tbunch_gid] = bs_start;
                    d_synchronizer_val[tbunch_gid] = fsum;
                }
                else
                {
                    d_y[bs_start] = fsum;
                }

                s_tail[tbunch_lid] = fsum;
            }

            continue;
        }

        // step 2. bit-flag generation
        // init bit-flag to 0
        s_ibuff[local_id] = 0;

        int local_boundary = partition_offset + i * partition_size;
        int stop = bs_stop == m ? m - 1 : bs_stop;

        for (int tid = bs_start + tbunch_tid; tid <= stop; tid += THREADBUNCH)
        {
            int row_idx = d_csrRowPtrA[tid];
            int row_idx_next = d_csrRowPtrA[tid + 1];

            if (row_idx != row_idx_next) // not an empty row
            {
                int offset = row_idx - local_boundary;
                if (offset >= 0)
                {
                    int slice_id = offset / SEG_H;
                    const unsigned int slice_bit_offset = (slice_id < THREADBUNCH) ? (1 << (31 - (offset % SEG_H))) : 0;
                    slice_id = (slice_id < THREADBUNCH) ? slice_id : 0;

                    // atomic or
                    atomicOr(&s_ibuff[tbunch_lid * THREADBUNCH + slice_id], slice_bit_offset);
                }
            }
            else // is empty
            {
                dirty = true;
            }
        }

        if (dirty) s_dirty[tbunch_lid] = true;

        // move bit-flag from scratchpad to register
        unsigned int bf = s_ibuff[local_id];

        // step 3. collect candidate spmv results
        // step 3-0. compute candidate spmv results
        #pragma unroll
        for (int j = 0; j < SEG_H; j++)
        {
            int candidate_idx = partition_offset + i * partition_size + j * THREADBUNCH + tbunch_tid;

            int y = tbunch_tid % SEG_H;
            int x = (j * THREADBUNCH + tbunch_tid) / SEG_H;

            // load indices & value, and keep index in the right scope [0,nnz)
            s_interres_local[y * (THREADBUNCH+1) + x] = candidate_idx < nnz ? candidate(d_csrValA, d_x, d_x_tex, d_csrColIdxA, candidate_idx, 1.0) : 0;
        }

        // step 3-1. transmit the last entry of the previous partition to this partition
        if (!tbunch_tid)
        {
            // if the first entry is not a segment start,
            // add the "s_tail" to its value and update its bit-flag to "1"
            s_interres_local[0] = ((bf >> 31) & 0x1) ? s_interres_local[0] : s_interres_local[0] + s_tail[tbunch_lid];
        }
        // the first bit should be 1 any way
        bf = tbunch_tid ? bf : bf | 0x80000000;

        // step 4. segmented scan
        // step 4-0. each thread does its own sub-list sequentially

        // extract the first bit-flag packet
        bool bf_bit = (bf >> 31) & 0x1;
        int lstart = !bf_bit;
        int lstop = 0;
        value_type sum  = s_interres_local[tbunch_tid];
        bool present = !tbunch_tid;
        present |= bf_bit;

        #pragma unroll
        for (int j = 1; j < SEG_H; j++)
        {
            bf_bit = (bf >> (31 - j)) & 0x1;

            if (bf_bit)
            {
                s_interres_local[lstop * (THREADBUNCH+1) + tbunch_tid] = sum;
                sum = 0;
                lstop++;
            }
            sum += s_interres_local[j * (THREADBUNCH+1) + tbunch_tid];
            present |= bf_bit;
        }
        s_interres_local[lstop * (THREADBUNCH+1) + tbunch_tid] = sum;

        int segn_scan = lstop - lstart + present;
        segn_scan = segn_scan > 0 ? segn_scan : 0;
        int tmp_segn_scan = segn_scan;

        // exclusive scan
        segn_scan = scan_32_shfl<int>(segn_scan, tbunch_tid);
        segn_scan -= tmp_segn_scan;

        // segmented scan the s_sum array
        value_type tmp_sum = lstart * s_interres_local[tbunch_tid];;
        tmp_sum = __shfl_down(tmp_sum, 1);
        tmp_sum = tbunch_tid == THREADBUNCH-1 ? 0 : tmp_sum;
        sum = s_interres_local[tbunch_tid];
        s_interres_local[tbunch_tid] = tmp_sum;

        s_present[tbunch_lid * (THREADBUNCH + 1) + tbunch_tid] = present;

        // step 4-1. spanning segments
        tmp_sum = 0;
        int id_current;
        if (present)
        {
            tmp_sum = s_interres_local[tbunch_tid];
            id_current = tbunch_tid + 1;
            while (!s_present[tbunch_lid * (THREADBUNCH + 1) + id_current])
            {
                tmp_sum += s_interres_local[id_current];
                id_current++;
            }
        }

        s_interres_local[tbunch_tid] = sum;

        // step 4-2. add scansum to segments
        s_interres_local[lstop * (THREADBUNCH+1) + tbunch_tid] += tmp_sum;

        // step 5. save vector_y to global
        // step 6. transmit the last entry to the next partition
        int location_counter = bs_start + segn_scan;

        for (int j = lstart; j <= lstop; j++)
        {
            if (location_counter == tbunch_start)
            {
                d_synchronizer_idx[tbunch_gid] = tbunch_start;
                d_synchronizer_val[tbunch_gid] = s_interres_local[j * (THREADBUNCH+1) + tbunch_tid];
            }
            else
            {
                d_y[location_counter] = s_interres_local[j * (THREADBUNCH+1) + tbunch_tid];
            }

            if (location_counter == bs_stop)
                s_tail[tbunch_lid] = s_interres_local[j * (THREADBUNCH+1) + tbunch_tid];

            location_counter++;
        }

        // step 7. if spec-flag is true, this partition has at least one empty row,
        //         so write start/stop to global
        if (!tbunch_tid && s_dirty[tbunch_lid])
        {
            s_tail[tbunch_lid] = 0;
            int dirty_idx = atomicAdd(d_dirty_counter, 1);
            d_speculator[2 * dirty_idx] = bs_start;
            d_speculator[2 * dirty_idx + 1] = bs_stop;
        }
    }
}


// csr_spmv kernel extracted from the CUSP library v0.4.0
template <typename IndexType, typename ValueType, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__launch_bounds__(VECTORS_PER_BLOCK * THREADS_PER_VECTOR,1)
__global__ void
spmv_csr_vector_kernel(const IndexType num_rows,
                       const IndexType * Ap,
                       const IndexType * Aj,
                       const ValueType * Ax,
                       const ValueType * x,
                       ValueType * y)
{
    __shared__ volatile ValueType sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
    __shared__ volatile IndexType ptrs[VECTORS_PER_BLOCK][2];

    const IndexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const IndexType thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const IndexType vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const IndexType vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const IndexType num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for(IndexType row = vector_id; row < num_rows; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];

        const IndexType row_start = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const IndexType row_end   = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];

        // initialize local sum
        ValueType sum = 0;

        if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32)
        {
            // ensure aligned memory access to Aj and Ax

            IndexType jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

            // accumulate local sums
            if(jj >= row_start && jj < row_end)
                sum += Ax[jj] * x[ Aj[jj] ];

            // accumulate local sums
            for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR)
                sum += Ax[jj] * x[ Aj[jj] ];
        }
        else
        {
            // accumulate local sums
            for(IndexType jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
                sum += Ax[jj] * x[ Aj[jj] ];
        }

        // store local sum in shared memory
        sdata[threadIdx.x] = sum;

        // reduce local sums to row sum
        if (THREADS_PER_VECTOR > 16) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16];
        if (THREADS_PER_VECTOR >  8) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8];
        if (THREADS_PER_VECTOR >  4) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4];
        if (THREADS_PER_VECTOR >  2) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2];
        if (THREADS_PER_VECTOR >  1) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1];

        // first thread writes the result
        if (thread_lane == 0)
            y[row] = sdata[threadIdx.x];
    }
}
#endif // KERNELS_CUDA_H
