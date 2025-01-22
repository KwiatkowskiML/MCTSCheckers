#pragma once
#include "cuda_runtime.h"

template <class T>
class Queue {
public:
    int first;
    int last;
    int size;
    T* Q;

    __host__ __device__ Queue(T* Q, int size)
    {
        this->Q = Q;
        this->size = size;
        first = last = 0;
    }
    __host__ __device__ void push(const T v)
    {
        Q[last] = v;
        last = (last + 1) % size;
    }
    __host__ __device__ void pop()
    {
        first = (first + 1) % size;
    }
    __host__ __device__ T front()
    {
        return Q[first];
    }
    __host__ __device__ bool empty()
    {
        return last == first;
    }
    __host__ __device__ int length()
    {
        return last - first + (last < first) * size;
    }
    __host__ __device__ void clear() 
    {
        first = last = 0;
    }
	__host__ __device__ T operator[](int i)
	{
		return Q[(first + i) % size];
	}
};