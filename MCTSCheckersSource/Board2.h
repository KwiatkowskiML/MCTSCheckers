#pragma once
#include "Utils.h"
#include "Move.h"
#include "PieceColor.h"
#include "BitBoard.h"
#include <vector>

class Board2 {
private:
    

public:
    BitBoard _pieces;
    Board2(UINT whitePieces, UINT blackPieces, UINT kings) : _pieces(whitePieces, blackPieces, kings) {};

    std::vector<Board2> getAvailableMoves(PieceColor color) const;
    void makeMove(const Move& move);

    // Visualization
    void printBoard() const;
    static void printBitboard(UINT bitboard);
    static void printPossibleMoves(const std::vector<Board2>& moves);
};
