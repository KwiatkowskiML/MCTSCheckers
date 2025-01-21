#pragma once
#include <vector>
#include <cassert>
#include <string>

#include "Types.h"
#include "PieceColor.h"
#include "Utils.h"
#include "BitBoard.h"

class Move {
private:
	// Stores the sequence of steps in the move
	std::vector<UINT> steps{};

	// Captured piece (if any)
	UINT captured;

public:
	// The color of the player making the move
	PieceColor playerColor;

	//----------------------------------------------------------------
	// Constructors
	//----------------------------------------------------------------
	Move(UINT src, UINT dst, UINT capt = 0, PieceColor col = PieceColor::White)
		: captured(capt), playerColor(col) {
		steps.push_back(src);
		steps.push_back(dst);
	}

	Move(std::vector<UINT> steps, UINT capt = 0, PieceColor col = PieceColor::White)
		: steps(steps), captured(capt), playerColor(col) {}

	Move(const std::string& moveString, PieceColor col = PieceColor::White);

	Move(const Move& other) : steps(other.steps), captured(other.captured), playerColor(other.playerColor) {}

	//----------------------------------------------------------------
	// Move transformations
	//----------------------------------------------------------------

	// Creates a new extended move by appending a continuation move to the current move and updating the capture status
	Move getExtendedMove(Move continuation, UINT capt) const;

	// Returns the bitboard after the move is made
	BitBoard getBitboardAfterMove(const BitBoard& sourceBitboard, bool includeCoronation = true) const;

	//----------------------------------------------------------------
	// Getters
	//----------------------------------------------------------------

	// Returns the steps in the move
	const std::vector<UINT>& getSteps() const;

	// Returns the destination of the move
	UINT getDestination() const;

	// Returns the source of the move
	UINT getSource() const;

	// Returns the captured pieces
	UINT getCaptured() const { return captured; }

	// Checks whether move is a capture move
	bool isCapture() const;

	// Converts the move to a string representation
	std::string toString() const;

	// Returns the color of moving player
	PieceColor getColor() const { return playerColor; }

	//----------------------------------------------------------------
	// Utilities
	//----------------------------------------------------------------

	// Checks if a move exists in a given move list
	static bool containsMove(const std::vector<Move>& moveList, const Move& move) {
		return std::find(moveList.begin(), moveList.end(), move) != moveList.end();
	}

	//----------------------------------------------------------------
	// Operators
	//----------------------------------------------------------------

	bool operator==(const Move& other) const {
		return steps == other.steps && captured == other.captured && playerColor == other.playerColor;
	}
};

using MoveList = std::vector<Move>;

