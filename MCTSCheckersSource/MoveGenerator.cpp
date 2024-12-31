#include <cassert>

#include "MoveGenerator.h"
#include "Utils.h"

MoveList MoveGenerator::generateMoves(const BitBoard& pieces, PieceColor color)
{
	return MoveList();
}

UINT MoveGenerator::getJumpers(const BitBoard& pieces, PieceColor color)
{
	// TODO: Implement for black pieces
	assert(color == PieceColor::White);

	const UINT emptyFields = pieces.getEmptyFields();
	const UINT whiteKings = pieces.whitePawns & pieces.kings;
	UINT jumpers = 0;

	// Get the black pawns that might be captured in base diagonal direction
	UINT captBlackPawns = (emptyFields << BASE_DIAGONAL_SHIFT) & pieces.blackPawns;

	// Check whether previously specified black pawns can actually be captured
	if (captBlackPawns)
	{
		// Get the white pawns that can capture black pawn in the base diagonal direction
		jumpers |= ((captBlackPawns & MOVES_UP_LEFT_AVAILABLE) << UP_LEFT_SHIFT) & pieces.whitePawns;
		jumpers |= ((captBlackPawns & MOVES_UP_RIGHT_AVAILABLE) << UP_RIGHT_SHIFT) & pieces.whitePawns;
	}

	// Get the black pawns that might be captured in the other diagonal direction
	captBlackPawns = ((emptyFields & MOVES_UP_LEFT_AVAILABLE) << UP_LEFT_SHIFT) & pieces.blackPawns;
	captBlackPawns |= ((emptyFields & MOVES_UP_RIGHT_AVAILABLE) << UP_RIGHT_SHIFT) & pieces.blackPawns;

	jumpers |= (captBlackPawns << BASE_DIAGONAL_SHIFT) & pieces.whitePawns;

	// Find all of the black pawns that might be captured backwards in base diagonal
	captBlackPawns = (emptyFields >> BASE_DIAGONAL_SHIFT) & pieces.blackPawns;

	// Check whether previously specified black pawns can actually be captured
	if (captBlackPawns)
	{
		jumpers |= ((captBlackPawns & MOVES_DOWN_LEFT_AVAILABLE) >> DOWN_LEFT_SHIFT) & pieces.whitePawns;
		jumpers |= ((captBlackPawns & MOVES_DOWN_RIGHT_AVAILABLE) >> DOWN_RIGHT_SHIFT) & pieces.whitePawns;
	}

	// Find all of the black pawns that might be captured backwards in the other diagonal
	captBlackPawns = ((emptyFields & MOVES_DOWN_LEFT_AVAILABLE) >> DOWN_LEFT_SHIFT) & pieces.blackPawns;
	captBlackPawns |= ((emptyFields & MOVES_DOWN_RIGHT_AVAILABLE) >> DOWN_RIGHT_SHIFT) & pieces.blackPawns;
	jumpers |= (captBlackPawns >> BASE_DIAGONAL_SHIFT) & pieces.whitePawns;

	// TODO: Consider if there is a need for analizing kings - there IS

	return jumpers;
}

UINT MoveGenerator::getMovers(const BitBoard& pieces, PieceColor color)
{
	return 0;
}
