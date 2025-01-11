#pragma once
#include <vector>
#include "Types.h"
#include "PieceColor.h"
#include <cassert>
#include "Utils.h"
#include "BitBoard.h"
#include <string>

struct Move {
    UINT source;
    UINT destination;
    UINT captured;
	bool isKingMove; // TODO: remove this
    PieceColor color;

    Move(UINT src, UINT dst, UINT capt = 0, bool king = false, PieceColor col = PieceColor::White)
        : source(src), destination(dst), captured(capt), isKingMove(king), color(col) {
    }

    BitBoard getBitboardAfterMove(const BitBoard& sourceBitboard) const
    {
        // TODO: Implement for black pieces
        assert(color == PieceColor::White);

        // Deleting the initial position of the moved piece
        UINT newWhitePawns = sourceBitboard.whitePawns & ~source;
        UINT newBlackPawns = sourceBitboard.blackPawns;
        UINT newKings = sourceBitboard.kings;

        // Deleting captured pieces
        if (isCapture())
        {
            newBlackPawns = sourceBitboard.blackPawns & ~captured;
            newKings = sourceBitboard.kings & ~captured;
        }

        // Adding new piece position
        newWhitePawns |= destination;

        // Handing the case when the pawn becomes a king, or the king is moved
        if (source & sourceBitboard.kings)
        {
            newKings = sourceBitboard.kings & ~source;
            newKings |= destination;
        }
        else if (destination & WHITE_CROWNING)
            newKings |= destination;

        BitBoard newbitBoard(newWhitePawns, newBlackPawns, newKings);
        return newbitBoard;
    }

	UINT getDestination() const 
    { 
        return destination;
    }

    bool isCapture() const 
    { 
        return captured != 0; 
    }

    std::string toString() const {
        std::string result;

        if (captured == 0)
			result = std::to_string(source) + "-" + std::to_string(destination);
        else
        {
			result = std::to_string(source) + "x" + std::to_string(destination);
        }
    }
};

using MoveList = std::vector<Move>;