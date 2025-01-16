#pragma once
#include "Types.h"
#include "PieceColor.h"

struct BitBoard {
    UINT whitePawns;
    UINT blackPawns;
    UINT kings;

    //BitBoard(UINT white = 0, UINT black = 0, UINT k = 0)
    //    : whitePawns(white), blackPawns(black), kings(k) {
    //}

	// Getters
    UINT getAllPieces() const { return whitePawns | blackPawns; }
    UINT getEmptyFields() const { return ~getAllPieces(); }
	UINT getEnemyPieces(PieceColor playerColor) const { return playerColor == PieceColor::White ? blackPawns : whitePawns; }
	UINT getPieces(PieceColor playerColor) const { return playerColor == PieceColor::White ? whitePawns : blackPawns; }
};