#pragma once
#include "Types.h"
#include "PieceColor.h"
#include "cuda_runtime.h"

struct BitBoard {
    UINT whitePawns;
    UINT blackPawns;
    UINT kings;

    __device__ __host__ BitBoard(UINT white = 0, UINT black = 0, UINT k = 0)
        : whitePawns(white), blackPawns(black), kings(k) {
    }

	// Getters
    __device__ __host__ UINT getAllPieces() const { return whitePawns | blackPawns; }
    __device__ __host__ UINT getEmptyFields() const { return ~getAllPieces(); }
    __device__ __host__ UINT getEnemyPieces(PieceColor playerColor) const { return playerColor == PieceColor::White ? blackPawns : whitePawns; }
    __device__ __host__ UINT getPieces(PieceColor playerColor) const { return playerColor == PieceColor::White ? whitePawns : blackPawns; }
};