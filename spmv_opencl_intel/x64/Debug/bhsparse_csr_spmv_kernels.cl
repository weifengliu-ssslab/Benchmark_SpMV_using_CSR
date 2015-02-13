#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#if USE_DOUBLE
typedef double   vT;
#else
typedef float   vT;
#endif

inline
int binarysearch_right_boundary(__global const int* d_csrRowPtrA,
                                const int           key_input,
                                const int           size)
{
    int start = 0;
    int stop  = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;
        key_median = d_csrRowPtrA[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }
    return start;
}

inline
void sum_8(volatile __local vT *s_sum,
           const int            local_id)
{
    if (local_id < 4)  s_sum[local_id] += s_sum[local_id + 4];
    s_sum[local_id] += s_sum[local_id + 2];
    s_sum[local_id] += s_sum[local_id + 1];
}

inline
void sum_16(volatile __local vT *s_sum,
            const int            local_id)
{
    if (local_id < 8)  s_sum[local_id] += s_sum[local_id + 8];
	s_sum[local_id] += s_sum[local_id + 4];
    s_sum[local_id] += s_sum[local_id + 2];
    s_sum[local_id] += s_sum[local_id + 1];
}

inline
void scan_8(volatile __local unsigned int *s_scan,
            const int                      local_id)
{
    int ai, bi;
    unsigned int temp;

    if (local_id < 4)  { ai = 2 * local_id;      bi = ai + 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)  { ai = 4 * local_id + 1;  bi = ai + 2;   s_scan[bi] += s_scan[ai]; }
    if (!local_id)     { s_scan[7] = s_scan[3]; s_scan[3] = 0; }
    if (local_id < 2)  { ai = 4 * local_id + 1;  bi = ai + 2;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)  { ai = 2 * local_id;      bi = ai + 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

inline
void scan_16(volatile __local unsigned int *s_scan,
             const int                      local_id)
{
    int ai, bi;
    int baseai = 1 + 2 * local_id;
    int basebi = baseai + 1;
    unsigned int temp;

    if (local_id < 8)  { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    if (local_id < 4)  { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
	if (local_id < 2)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (!local_id) { s_scan[15] = s_scan[7]; s_scan[7] = 0; }
	if (local_id < 2)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)  { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)  { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

__kernel
void SpMV_kernel(__global const int             *d_csrRowPtrA,
                 __global const int             *d_csrColIndA,
                 __global const vT              *d_csrValA,
                 __global const vT              *d_x,
                 __global vT                    *d_y,
                 __global int                   *d_speculator,
                 volatile __global int          *d_dirty_counter,
                 __global int                   *d_synchronizer_idx,
                 __global vT                    *d_synchronizer_val,
                 __local vT                     *s_interres, //s_interres[][SEG_H * THREADBUNCH + SEG_H * THREADBUNCH / 16],
                 volatile __local vT            *s_sum, //s_sum[][THREADBUNCH],
				 volatile __local int           *s_bs, //s_bs[][STEP+1],
                 volatile __local unsigned int  *s_ibuff, //s_bf[][THREADBUNCH+1],
                 volatile __local vT            *s_tail, //s_tail[][THREADBUNCH_PER_BLOCK],
				 volatile __local bool          *s_dirty, //s_dirty[][THREADBUNCH_PER_BLOCK],
                 volatile __local bool          *s_present, //s_present[][THREADBUNCH+1],
                 const int                       partition_size,
                 const int                       partition_num,
                 const int                       tbunch_num,
                 const int                       nnz,
                 const int                       m)
{
    const int local_id    = get_local_id(0);
    const int tbunch_tid  = local_id % THREADBUNCH; // thread bunch thread (lane) id
    const int tbunch_lid  = local_id / THREADBUNCH; // thread bunch local id
    const int tbunch_gid  = get_global_id(0) / THREADBUNCH; // thread bunch global id

	const int tbunch_tid_ext  = tbunch_lid * (THREADBUNCH + 1) + tbunch_tid;

    if (tbunch_gid >= tbunch_num) return;

    // step 1. binary search
    // compute offset
    const int partition_offset = tbunch_gid * STEP * partition_size;
    
    // do binary search
	for (int i = tbunch_tid; i <= STEP; i += THREADBUNCH)
	{
	    int boundary = partition_offset + i * partition_size;
		// clamp partition_nnz_start to [0, nnz]
        boundary = boundary > nnz ? nnz : boundary;
	    s_bs[tbunch_lid * (STEP + 1) + i] = binarysearch_right_boundary(d_csrRowPtrA, boundary, m+1) - 1;
	}

    int bs_start = 0;
    int bs_stop  = s_bs[tbunch_lid * (STEP + 1)];
	int tbunch_start = bs_stop;

    if (!tbunch_tid)
    {
        s_present[tbunch_lid * (THREADBUNCH + 1) + THREADBUNCH] = true;
        s_sum[tbunch_lid * THREADBUNCH + THREADBUNCH - 1] = 0;
        s_tail[tbunch_lid] = 0;
    }

    volatile __local vT *s_interres_local = &s_interres[tbunch_lid * 17 * (SEG_H * THREADBUNCH) / 16];

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
        bs_stop  = s_bs[tbunch_lid * (STEP + 1) + i + 1];

        //do fast track
        if (bs_start == bs_stop)
        {
            vT fsum = 0;
            #pragma unroll
            for (int j = 0; j < SEG_H; j++)
            {
                int candidate_idx = partition_offset + i * partition_size + j * THREADBUNCH + tbunch_tid;

                // load indices & value, and keep index in the right scope [0,nnz)
                fsum += candidate_idx < nnz ? d_csrValA[candidate_idx] * d_x[d_csrColIndA[candidate_idx]] : 0;
            }

            s_interres_local[tbunch_tid] = fsum;
#if THREADBUNCH == 8
            sum_8(s_interres_local, tbunch_tid);
#elif THREADBUNCH == 16
            sum_16(s_interres_local, tbunch_tid);
#endif
            if (!tbunch_tid)
            {
                int local_boundary = partition_offset + i * partition_size;
                fsum = d_csrRowPtrA[bs_start] == local_boundary ? s_interres_local[0] : s_tail[tbunch_lid] + s_interres_local[0];

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
        s_ibuff[tbunch_tid_ext] = 0;

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
                    unsigned int slice_bit_offset = 1 << (31 - (offset % SEG_H));

                    // atomic or
                    atomic_or(&s_ibuff[tbunch_lid * (THREADBUNCH+1) + slice_id], slice_bit_offset);
                }
            }
            else // is empty
            {
                dirty = 1;
            }
        }

        if (dirty) s_dirty[tbunch_lid] = true;

        // move bit-flag from scratchpad to register
        unsigned int bf = s_ibuff[tbunch_tid_ext];

        // step 3. collect candidate spmv results
        // step 3-0. compute candidate spmv results
        #pragma unroll
        for (int j = 0; j < SEG_H; j++)
        {
            const int candidate_idx = partition_offset + i * partition_size + j * THREADBUNCH + tbunch_tid;

#if SEG_H >= THREADBUNCH
            const int y = tbunch_tid % SEG_H + (j % (SEG_H / THREADBUNCH)) * THREADBUNCH;
#else
            const int y = tbunch_tid % SEG_H;
#endif
            const int x = (j * THREADBUNCH + tbunch_tid) / SEG_H;

            // load indices & value, and keep index in the right scope [0,nnz)
            s_interres_local[17 * (y * THREADBUNCH + x) / 16] = candidate_idx < nnz ? d_csrValA[candidate_idx] * d_x[d_csrColIndA[candidate_idx]] : 0;
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
        vT sum  = s_interres_local[tbunch_tid];
        bool present = !tbunch_tid;
        present |= bf_bit;

        #pragma unroll
        for (int j = 1; j < SEG_H; j++)
        {
            bf_bit = (bf >> (31 - j)) & 0x1;

            if (bf_bit)
            {
                s_interres_local[17 * (lstop * THREADBUNCH + tbunch_tid) / 16] = sum;
                sum = 0;
                lstop++;
            }
            sum += s_interres_local[17 * (j * THREADBUNCH + tbunch_tid) / 16];
            present |= bf_bit;
        }
        s_interres_local[17 * (lstop * THREADBUNCH + tbunch_tid) / 16] = sum;

        int segn_scan = lstop - lstart + present;
        segn_scan = segn_scan > 0 ? segn_scan : 0;
        s_ibuff[tbunch_tid_ext] = segn_scan;
#if THREADBUNCH == 8
        scan_8(&s_ibuff[tbunch_lid * (THREADBUNCH+1)], tbunch_tid);
#elif THREADBUNCH == 16
        scan_16(&s_ibuff[tbunch_lid * (THREADBUNCH+1)], tbunch_tid);
#endif
        // segmented scan the s_sum array
        if (tbunch_tid)
            s_sum[tbunch_lid * THREADBUNCH + tbunch_tid - 1] = lstart * s_interres_local[tbunch_tid];
        s_present[tbunch_tid_ext] = present;

        // step 4-1. spanning segments
        sum = 0;
        int id_current;
        if (present)
        {
            sum = s_sum[tbunch_lid * THREADBUNCH + tbunch_tid];
            id_current = tbunch_tid + 1;
            while (!s_present[tbunch_lid * (THREADBUNCH + 1) + id_current])
            {
                sum += s_sum[tbunch_lid * THREADBUNCH + id_current];
                id_current++;
            }
        }

        // step 4-2. add scansum to segments
        s_interres_local[17 * (lstop * THREADBUNCH + tbunch_tid) / 16] += sum;

        // step 5. save vector_y to global
        // step 6. transmit the last entry to next partition
        int location_counter = bs_start + s_ibuff[tbunch_tid_ext];

        for (int j = lstart; j <= lstop; j++)
        {
		    const int idx = 17 * (j * THREADBUNCH + tbunch_tid) / 16;
            if (location_counter == tbunch_start)
            {
                d_synchronizer_idx[tbunch_gid] = tbunch_start;
                d_synchronizer_val[tbunch_gid] = s_interres_local[idx];
            }
            else
            {
                d_y[location_counter] = s_interres_local[idx];
            }

            if (location_counter == bs_stop)
                s_tail[tbunch_lid] = s_interres_local[idx];

            location_counter++;
        }

        // step 7. if spec-flag is true, this partition has at least one empty row,
        //         so write start/stop to global
        if (!tbunch_tid && s_dirty[tbunch_lid])
        {
            s_tail[tbunch_lid] = 0;
            int dirty_idx = atomic_inc(d_dirty_counter);
            d_speculator[2 * dirty_idx] = bs_start;
            d_speculator[2 * dirty_idx + 1] = bs_stop;
        }
    }
}
