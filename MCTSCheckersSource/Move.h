#pragma once
#include <vector>
#include "Types.h"
#include "PieceColor.h"
#include <cassert>
#include "Utils.h"
#include "BitBoard.h"
#include <string>

class Move {
private:
    std::vector<UINT> steps{};
    UINT captured;
    PieceColor color;

public:
    Move(UINT src, UINT dst, UINT capt = 0, PieceColor col = PieceColor::White)
        : captured(capt), color(col) 
    {
		steps.push_back(src);
		steps.push_back(dst);
    }

    Move(std::vector<UINT> steps, UINT capt = 0, PieceColor col = PieceColor::White)
        : steps(steps), captured(capt), color(col) { }

    Move getExtendedMove(Move continuation, UINT capt) const;
    BitBoard getBitboardAfterMove(const BitBoard& sourceBitboard) const;
    const std::vector<UINT>& getSteps() const;

    UINT getDestination() const;
    UINT getSource() const;
	UINT getCaptured() const { return captured; }
    bool isCapture() const;
    std::string toString() const;
	PieceColor getColor() const { return color; }

    bool operator==(const Move& other) const {
        return steps == other.steps && captured == other.captured && color == other.color;
    }
};

using MoveList = std::vector<Move>;
