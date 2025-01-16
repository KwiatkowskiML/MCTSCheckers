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
	Board(UINT whitePieces, UINT blackPieces, UINT kings) 
	{
		_pieces.whitePawns = whitePieces;
		_pieces.blackPawns = blackPieces;
		_pieces.kings = kings;
	};
	
	Board() 
	{
		_pieces.whitePawns = INIT_WHITE_PAWNS;
		_pieces.blackPawns = INIT_BLACK_PAWNS;
		_pieces.kings = 0;
	};

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
