#pragma once
#include "cuda_runtime.h"

enum class PieceColor{
    White,
    Black
};

__device__ __host__ constexpr PieceColor getEnemyColor(PieceColor playerColor)
{
    return playerColor == PieceColor::White ? PieceColor::Black : PieceColor::White;
}