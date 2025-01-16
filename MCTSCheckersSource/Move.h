#pragma once
#include <vector>
#include "Types.h"
#include "PieceColor.h"
#include <cassert>
#include "Utils.h"
#include "BitBoard.h"
#include <string>



class Move {
private:
	std::vector<UINT> steps{};
	UINT captured;
public:
	PieceColor playerColor;

	Move(UINT src, UINT dst, UINT capt = 0, PieceColor col = PieceColor::White)
		: captured(capt), playerColor(col)
	{
		steps.push_back(src);
		steps.push_back(dst);
	}
	Move(std::vector<UINT> steps, UINT capt = 0, PieceColor col = PieceColor::White)
		: steps(steps), captured(capt), playerColor(col) {
	}
	Move(const std::string& moveString, PieceColor col = PieceColor::White);
	Move(const Move& other) : steps(other.steps), captured(other.captured), playerColor(other.playerColor) {}

	Move getExtendedMove(Move continuation, UINT capt) const;
	BitBoard getBitboardAfterMove(BitBoard sourceBitboard, bool includeCoronation = true) const;
	const std::vector<UINT>& getSteps() const;

	static bool containsMove(const std::vector<Move>& moveList, const Move& move) {
		return std::find(moveList.begin(), moveList.end(), move) != moveList.end();
	}

	UINT getDestination() const;
	UINT getSource() const;
	UINT getCaptured() const { return captured; }
	bool isCapture() const;
	std::string toString() const;
	PieceColor getColor() const { return playerColor; }

	bool operator==(const Move& other) const {
		return steps == other.steps && captured == other.captured && playerColor == other.playerColor;
	}
};

using MoveList = std::vector<Move>;

