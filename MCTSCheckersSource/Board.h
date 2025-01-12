#pragma once
#include "Utils.h"
#include "PieceColor.h"
#include "BitBoard.h"
#include <vector>
#include "Move.h"
#include <unordered_map>

class Board {
private:
    
public:
    BitBoard _pieces;
    Board(UINT whitePieces, UINT blackPieces, UINT kings) : _pieces(whitePieces, blackPieces, kings) {};

	// Move generation
    MoveList getAvailableMoves(PieceColor color) const;
    Board getBoardAfterMove(const Move& move) const;

    // Visualization
    void printBoard() const;
    static void printBitboard(UINT bitboard);

    // Field positioning
    const static std::unordered_map<UINT, std::string> fieldMapping;
};
