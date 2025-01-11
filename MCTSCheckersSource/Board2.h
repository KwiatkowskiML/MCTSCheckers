#pragma once
#include "Utils.h"
#include "PieceColor.h"
#include "BitBoard.h"
#include <vector>
#include "Move2.h"

class Board2 {
private:
    

public:
    BitBoard _pieces;
    Board2(UINT whitePieces, UINT blackPieces, UINT kings) : _pieces(whitePieces, blackPieces, kings) {};

    MoveList getAvailableMoves(PieceColor color) const;
    Board2 getBoardAfterMove(const Move2& move) const;

    // Visualization
    void printBoard() const;
    static void printBitboard(UINT bitboard);
    static void printPossibleMoves(const std::vector<Board2>& moves);
};
