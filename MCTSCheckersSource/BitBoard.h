#pragma once
#include "Types.h"

struct BitBoard {
    UINT whitePawns;
    UINT blackPawns;
    UINT kings;

    BitBoard(UINT white = 0, UINT black = 0, UINT k = 0)
        : whitePawns(white), blackPawns(black), kings(k) {
    }

    UINT getAllPieces() const { return whitePawns | blackPawns; }
    UINT getEmptyFields() const { return ~getAllPieces(); }
};