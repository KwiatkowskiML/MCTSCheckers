#pragma once
#include <vector>
#include "Types.h"
#include "PieceColor.h"
#include <cassert>
#include "Utils.h"
#include "BitBoard.h"
#include <string>

class Move2 {
private:
    std::vector<UINT> steps{};
    UINT captured;
    PieceColor color;

public:
    Move2(UINT src, UINT dst, UINT capt = 0, PieceColor col = PieceColor::White)
        : captured(capt), color(col) 
    {
		steps.push_back(src);
		steps.push_back(dst);
    }

    Move2(std::vector<UINT> steps, UINT capt = 0, PieceColor col = PieceColor::White)
        : steps(steps), captured(capt), color(col) { }

    Move2 getExtendedMove(UINT dst, UINT capt) const;
    BitBoard getBitboardAfterMove(const BitBoard& sourceBitboard) const;

    UINT getDestination() const;
    UINT getSource() const;
    bool isCapture() const;
    std::string toString() const;

    bool operator==(const Move2& other) const {
        return steps == other.steps && captured == other.captured && color == other.color;
    }
};
