#pragma once
#include "Types.h"
#include <vector>;

struct Move {
    UINT source;
    UINT destination;
    UINT captured;
	bool isKingMove;

    Move(UINT src, UINT dst, UINT capt = 0, bool king = false)
        : source(src), destination(dst), captured(capt), isKingMove(king) {
    }

    bool isCapture() const { return captured != 0; }
};

using MoveList = std::vector<Move>;