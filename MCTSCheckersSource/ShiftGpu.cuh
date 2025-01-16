#pragma once
#include "cuda_runtime.h"
#include "Types.h"
#include "BitShift.h"
#include "Utils.h"

__device__ __host__ BitShift getOppositeShift(BitShift bitShift) 
{
	BitShift opposite;

	switch (bitShift)
	{
	case BitShift::BIT_SHIFT_R3:
		opposite = BitShift::BIT_SHIFT_L3;
		break;
	case BitShift::BIT_SHIFT_R4:
		opposite = BitShift::BIT_SHIFT_L4;
		break;
	case BitShift::BIT_SHIFT_R5:
		opposite = BitShift::BIT_SHIFT_L5;
		break;
	case BitShift::BIT_SHIFT_L3:
		opposite = BitShift::BIT_SHIFT_R3;
		break;
	case BitShift::BIT_SHIFT_L4:
		opposite = BitShift::BIT_SHIFT_R4;
		break;
	case BitShift::BIT_SHIFT_L5:
		opposite = BitShift::BIT_SHIFT_R5;
		break;
	default:
		opposite = BitShift::BIT_SHIFT_NONE;
		break;
	}

	return opposite;
}

__device__ __host__ UINT applyShift(UINT position, BitShift bitShift)
{
	UINT shiftedPosition;

	switch (bitShift)
	{
	case BitShift::BIT_SHIFT_R3:
		shiftedPosition = (position & MASK_R3) >> SHIFT_R3;
		break;
	case BitShift::BIT_SHIFT_R4:
		shiftedPosition = position >> SHIFT_BASE;
		break;
	case BitShift::BIT_SHIFT_R5:
		shiftedPosition = (position & MASK_R5) >> SHIFT_R5;
		break;
	case BitShift::BIT_SHIFT_L3:
		shiftedPosition = (position & MASK_L3) << SHIFT_L3;
		break;
	case BitShift::BIT_SHIFT_L4:
		shiftedPosition = position << SHIFT_BASE;
		break;
	case BitShift::BIT_SHIFT_L5:
		shiftedPosition = (position & MASK_L5) << SHIFT_L5;
		break;
	default:
		shiftedPosition = position;
		break;
	}

	return shiftedPosition;
}