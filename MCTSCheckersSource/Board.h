#pragma once
#include "Utils.h"
#include "PieceColor.h"
#include "BitBoard.h"
#include <vector>
#include "Move.h"
#include <unordered_map>

class Board {
private:
    BitBoard _pieces;
public:
    Board(UINT whitePieces, UINT blackPieces, UINT kings) : _pieces(whitePieces, blackPieces, kings) {};
	Board() : _pieces(0, 0, 0) {};

	// Getters
	BitBoard getBitBoard() const { return _pieces; }
	UINT getKings() const { return _pieces.kings; }

	// Move generation
    MoveList getAvailableMoves(PieceColor color) const;
    Board getBoardAfterMove(const Move& move) const;

    // Simulation
	int simulateGame(PieceColor color) const;

    // Visualization
    std::string toString() const;
    static void printBitboard(UINT bitboard);

    // Field positioning
    const static std::unordered_map<UINT, std::string> fieldMapping;
};
