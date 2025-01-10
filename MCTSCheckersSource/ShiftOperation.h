#pragma once
#include "Types.h"

struct ShiftOperation {
    int shiftAmount;
    UINT mask;

    constexpr ShiftOperation(int amount, UINT m = 0xFFFFFFFF)
        : shiftAmount(amount), mask(m) { }

    UINT Apply(UINT position) const 
    {
        position &= mask;
        return shiftAmount >= 0 ? position << shiftAmount : position >> -shiftAmount;
    }
};