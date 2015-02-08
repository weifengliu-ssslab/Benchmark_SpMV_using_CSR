#ifndef UTILS_CUDA_H
#define UTILS_CUDA_H

#include "common.h"

template<typename iT>
__inline__ __device__
iT binary_search_right_boundary_kernel(const iT *d_row_pointer,
                                       const iT  key_input,
                                       const iT  size)
{
    iT start = 0;
    iT stop  = size - 1;
    iT median;
    iT key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

#if __CUDA_ARCH__ >= 350
        key_median = __ldg(&d_row_pointer[median]);
#else
        key_median = d_row_pointer[median];
#endif

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}

#if __CUDA_ARCH__ <= 320

__device__ __forceinline__
double __shfl_down(double var, unsigned int srcLane, int width=32)
{
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, srcLane, width);
    a.y = __shfl_down(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
}

__device__ __forceinline__
double __shfl_up(double var, unsigned int srcLane, int width=32)
{
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_up(a.x, srcLane, width);
    a.y = __shfl_up(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
}

__device__ __forceinline__
double __shfl_xor(double var, int srcLane, int width=32)
{
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_xor(a.x, srcLane, width);
    a.y = __shfl_xor(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
}

#endif


// sum
template<typename vT>
__forceinline__ __device__
vT sum_32_shfl(vT sum)
{
    #pragma unroll
    for(int mask = THREADBUNCH / 2 ; mask > 0 ; mask >>= 1)
        sum += __shfl_xor(sum, mask);

    return sum;
}

// inclusive scan
template<typename T>
__forceinline__ __device__
T scan_32_shfl(T         x,
               const int local_id)
{
    T y = __shfl_up(x, 1);
    x = local_id >= 1 ? x + y : x;
    y = __shfl_up(x, 2);
    x = local_id >= 2 ? x + y : x;
    y = __shfl_up(x, 4);
    x = local_id >= 4 ? x + y : x;
    y = __shfl_up(x, 8);
    x = local_id >= 8 ? x + y : x;
    y = __shfl_up(x, 16);
    x = local_id >= 16 ? x + y : x;

    return x;
}

template<typename iT>
__forceinline__ __device__
void fetch_x(cudaTextureObject_t  d_x_tex,
             const iT             i,
             float               *x)
{
    *x = tex1Dfetch<float>(d_x_tex, i);
}

template<typename iT>
__forceinline__ __device__
void fetch_x(cudaTextureObject_t  d_x_tex,
             const iT             i,
             double              *x)
{
    int2 x_int2 = tex1Dfetch<int2>(d_x_tex, i);
    *x = __hiloint2double(x_int2.y, x_int2.x);
}

__inline__ __device__
float candidate(const float           *d_value_partition,
                const float           *d_x,
                cudaTextureObject_t    d_x_tex,
                const int             *d_column_index_partition,
                const int              candidate_index,
                const float            alpha)
{
    float x = 0;
#if __CUDA_ARCH__ >= 350
    x = __ldg(&d_x[d_column_index_partition[candidate_index]]);
#else
    fetch_x<int>(d_x_tex, d_column_index_partition[candidate_index], &x);
#endif
    return d_value_partition[candidate_index] * x;// * alpha;
}

__inline__ __device__
double candidate(const double           *d_value_partition,
                 const double           *d_x,
                 cudaTextureObject_t     d_x_tex,
                 const int              *d_column_index_partition,
                 const int               candidate_index,
                 const double            alpha)
{
    double x = 0;
#if __CUDA_ARCH__ >= 350
    x = __ldg(&d_x[d_column_index_partition[candidate_index]]);
#else
    fetch_x<int>(d_x_tex, d_column_index_partition[candidate_index], &x);
#endif
    return d_value_partition[candidate_index] * x;// * alpha;
}

#endif // UTILS_CUDA_H
