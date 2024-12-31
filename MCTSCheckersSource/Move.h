#pragma once
#include <vector>;
#include "Types.h"
#include "PieceColor.h"

struct Move {
    UINT source;
    UINT destination;
    UINT captured;
	bool isKingMove;
    PieceColor color;

    Move(UINT src, UINT dst, UINT capt = 0, bool king = false, PieceColor col = PieceColor::White)
        : source(src), destination(dst), captured(capt), isKingMove(king), color(col) {
    }

    bool isCapture() const { return captured != 0; }
};

using MoveList = std::vector<Move>;