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
	Board() : _pieces(0, 0, 0) {};

	// Getters
	BitBoard getBitBoard() const { return _pieces; }
	UINT getKings() const { return _pieces.kings; }
	UINT getWhitePawns() const { return _pieces.whitePawns; }
	UINT getBlackPawns() const { return _pieces.blackPawns; }

	// Move generation
	MoveList getAvailableMoves(PieceColor playerColor) const;
	Board getBoardAfterMove(const Move& move) const;
	static UINT getAllFieldsBetween(UINT start, UINT end);

	// Simulation
	int simulateGame(PieceColor playerColor) const;

	// Visualization
	std::string toString() const;
	static void printBitboard(UINT bitboard);

	// Field positioning
	const static std::unordered_map<UINT, std::string> fieldToStringMapping;
	const static std::unordered_map<std::string, UINT> stringToFieldMapping;
};
