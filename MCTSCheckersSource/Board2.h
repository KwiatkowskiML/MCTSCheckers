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

    MoveList getAvailableMoves(PieceColor color) const;
    Board2 getBoardAfterMove(const Move& move) const;

    // Visualization
    void printBoard() const;
    static void printBitboard(UINT bitboard);
    static void printPossibleMoves(const std::vector<Board2>& moves);
};
