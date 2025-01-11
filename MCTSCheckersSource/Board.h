#pragma once
#include "Utils.h"
#include "PieceColor.h"
#include "BitBoard.h"
#include <vector>
#include "Move.h"

class Board {
private:
    

public:
    BitBoard _pieces;
    Board(UINT whitePieces, UINT blackPieces, UINT kings) : _pieces(whitePieces, blackPieces, kings) {};

    MoveList getAvailableMoves(PieceColor color) const;
    Board getBoardAfterMove(const Move& move) const;

    // Visualization
    void printBoard() const;
    static void printBitboard(UINT bitboard);
    static void printPossibleMoves(const std::vector<Board>& moves);
};
