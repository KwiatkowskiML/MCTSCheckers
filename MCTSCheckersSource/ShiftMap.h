#pragma once
#include "ShiftOperation.h"
#include "Utils.h"
#include "BitShift.h"

class ShiftMap {
private:
	static constexpr ShiftOperation shifts[6] = {
		ShiftOperation(SHIFT_L3, MASK_L3),
		ShiftOperation(SHIFT_BASE),
		ShiftOperation(SHIFT_L5, MASK_L5),
		ShiftOperation(-SHIFT_R5, MASK_R5),
		ShiftOperation(-SHIFT_BASE),
		ShiftOperation(-SHIFT_R3, MASK_R3)
	};

public:
	static UINT shift(UINT position, BitShift bitShift) {
		return shifts[static_cast<int>(bitShift)].Apply(position);
	}

	static BitShift getOpposite(BitShift bitShift) {
		static const BitShift opposites[] = {
			BitShift::BIT_SHIFT_R3,
			BitShift::BIT_SHIFT_R4,
			BitShift::BIT_SHIFT_R5,
			BitShift::BIT_SHIFT_L3,
			BitShift::BIT_SHIFT_L4,
			BitShift::BIT_SHIFT_L5
		};
		return opposites[static_cast<int>(bitShift)];
	}
};