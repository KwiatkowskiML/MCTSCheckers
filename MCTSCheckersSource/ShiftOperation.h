#pragma once
#include "Types.h"

struct ShiftOperation {
    int shiftAmount;
    UINT mask;

    __device__ __host__ constexpr ShiftOperation(int amount, UINT m = 0xFFFFFFFF)
        : shiftAmount(amount), mask(m) { }

    __device__ __host__ UINT Apply(UINT position) const 
    {
        position &= mask;
        return shiftAmount >= 0 ? position << shiftAmount : position >> (-shiftAmount);
    }
};