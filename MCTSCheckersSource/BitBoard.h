#pragma once
#include "Types.h"
#include "PieceColor.h"
#include "cuda_runtime.h"

struct BitBoard {
    UINT whitePawns;
    UINT blackPawns;
    UINT kings;

	// Getters
    UINT getAllPieces() { return whitePawns | blackPawns; }
    UINT getEmptyFields() { return ~getAllPieces(); }
	UINT getEnemyPieces(PieceColor playerColor) { return playerColor == PieceColor::White ? blackPawns : whitePawns; }
	UINT getPieces(PieceColor playerColor) { return playerColor == PieceColor::White ? whitePawns : blackPawns; }
};

