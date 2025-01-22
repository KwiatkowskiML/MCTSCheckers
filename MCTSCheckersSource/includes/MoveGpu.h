#pragma once
#include <sstream>

#include "Types.h"
#include "PieceColor.h"
#include "BitBoard.h"
#include "Utils.h"
#include "Board.h"

class MoveGpu {
public:
	// The color of the player making the move
	PieceColor playerColor;

	// Source of the move
	UINT src;	

	// Destination of the move
	UINT dst;

	// Captured piece (if any)
	UINT captured;
	

	//----------------------------------------------------------------
	// Constructors
	//----------------------------------------------------------------

	__host__ __device__ MoveGpu() : src(0), dst(0), playerColor(PieceColor::White), captured(0) {}

	__host__ __device__ MoveGpu(UINT src, UINT dst, PieceColor col, UINT capt = 0)
		: src(src), dst(dst), playerColor(col), captured(capt) {}

	__host__ __device__ MoveGpu(const MoveGpu& other) : src(other.src), dst(other.dst), playerColor(other.playerColor), captured(other.captured) {}

	//----------------------------------------------------------------
	// Move transformations
	//----------------------------------------------------------------

	// Creates a new extended move by appending a continuation move to the current move and updating the capture status
	__host__ __device__ MoveGpu getExtendedMove(MoveGpu continuation) const
	{
		UINT newSrc = src;
		UINT newDst = continuation.dst;
		UINT newCaptured = captured | continuation.captured;

		return MoveGpu(newSrc, newDst, playerColor, newCaptured);
	};

	// Returns the bitboard after the move is made
	__host__ __device__ BitBoard getBitboardAfterMove(const BitBoard& sourceBitboard, bool includeCoronation = true) const
	{
		UINT currentPlayerPieces = sourceBitboard.getPieces(playerColor);
		UINT enemyPlayerPieces = sourceBitboard.getPieces(getEnemyColor(playerColor));
		UINT kings = sourceBitboard.kings;

		// Deleting the initial position of the moved piece
		UINT newCurrentPlayerPieces = currentPlayerPieces & ~src;
		UINT newEnemyPlayerPieces = enemyPlayerPieces;
		UINT newKings = kings;

		// Deleting captured pieces
		if (captured)
		{
			newEnemyPlayerPieces = enemyPlayerPieces & ~captured;
			newKings = kings & ~captured;
		}

		// Adding new piece position
		newCurrentPlayerPieces |= dst;

		// Handing the case when the king is moved
		if (src & kings)
		{
			newKings = newKings & ~src;
			newKings |= dst;
		}

		// Handling the case when the pawn is crowned
		if (includeCoronation)
		{
			if (playerColor == PieceColor::White && (dst & WHITE_CROWNING))
				newKings |= dst;

			if (playerColor == PieceColor::Black && (dst & BLACK_CROWNING))
				newKings |= dst;
		}

		UINT newWhitePawns = playerColor == PieceColor::White ? newCurrentPlayerPieces : newEnemyPlayerPieces;
		UINT newBlackPawns = playerColor == PieceColor::Black ? newCurrentPlayerPieces : newEnemyPlayerPieces;

		return BitBoard(newWhitePawns, newBlackPawns, newKings);
	}

	// kernel debiging
	__device__ __host__ void printMove() {
		printf("Move: 0x%08x%s0x%08x 0x%08x\n",
			src,
			captured ? ":" : "-",
			dst,
			captured);
	}

	// Get string representation of the move
	std::string toString() const
	{
		std::string result = "";

		try {
			result += Board::fieldToStringMapping.at(src);
			if (!captured)
			{
				result += "-";
				result += Board::fieldToStringMapping.at(dst);
			}
			else
			{
				result += ":";
				result += Board::fieldToStringMapping.at(dst);
			}
		}
		catch (const std::exception& e) {
			result = "Error in Move2::toString()";
		}

		result += " 0x" + toHex(captured);
		return result;
	}

	std::string toHex(int number) const {
		std::ostringstream oss;
		oss << std::hex << number;
		return oss.str();
	}

	bool operator==(const MoveGpu& other) const {
		return src == other.src && dst == other.dst && captured == other.captured && playerColor == other.playerColor;
	}
};