#include <cassert>

#include "MoveGenerator.h"
#include "Utils.h"

MoveList MoveGenerator::generateMoves(const BitBoard& pieces, PieceColor color)
{
	// Find all of the jumpers
	UINT jumpers = getJumpers(pieces, color);
	MoveList moves;

	if (jumpers)
	{
		generateCapturingMoves(pieces, color, jumpers, moves);
		return moves;
	}

	UINT movers = getMovers(pieces, color);
	generateBasicMoves(pieces, color, movers, moves);	
	return moves;
}

//----------------------------------------------------------------
// Getting moveable pieces
//----------------------------------------------------------------

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
	// TODO: Implement for black pieces
	assert(color == PieceColor::White);

	const UINT emptyFields = pieces.getEmptyFields();
	const UINT whiteKings = pieces.whitePawns & pieces.kings;

	// Get the white pieces that can move in the basic diagonal direction (right down or left down, depending on the row)
	UINT movers = (emptyFields << BASE_DIAGONAL_SHIFT) & pieces.whitePawns;

	// Get the white pieces that can move in the right down direction
	movers |= ((emptyFields & MOVES_UP_LEFT_AVAILABLE) << UP_LEFT_SHIFT) & pieces.whitePawns;

	// Get the white pieces that can move in the left down direction
	movers |= ((emptyFields & MOVES_UP_RIGHT_AVAILABLE) << UP_RIGHT_SHIFT) & pieces.whitePawns;

	// Get the white kings that can move in the upper diagonal direction (right up or left up)
	if (whiteKings)
	{
		movers |= (emptyFields >> BASE_DIAGONAL_SHIFT) & whiteKings;
		movers |= ((emptyFields & MOVES_DOWN_RIGHT_AVAILABLE) >> DOWN_RIGHT_SHIFT) & whiteKings;
		movers |= ((emptyFields & MOVES_DOWN_LEFT_AVAILABLE) >> DOWN_LEFT_SHIFT) & whiteKings;
	}

	return movers;
}

//----------------------------------------------------------------
// Generating a list of moves
//----------------------------------------------------------------

void MoveGenerator::generateBasicMoves(const BitBoard& pieces, PieceColor color, UINT movers, MoveList& moves)
{
	while (movers) {
		UINT mover = movers & -movers;
		movers ^= mover;

		if (mover & pieces.kings) {
			generateKingMoves(pieces, color, mover, moves);
		}
		else {
			generatePawnMoves(pieces, color, mover, moves);
		}
	}
}

void MoveGenerator::generateCapturingMoves(const BitBoard& pieces, PieceColor color, UINT jumpers, MoveList& moves)
{
	while (jumpers) {
		UINT jumper = jumpers & -jumpers;
		jumpers ^= jumper;

		if (jumper & pieces.kings) {
			generateKingCaptures(pieces, color, jumper, moves);
		}
		else {
			generatePawnCaptures(pieces, color, jumper, moves);
		}
	}
}

//----------------------------------------------------------------
// Generating specified move
//----------------------------------------------------------------

void MoveGenerator::generateKingCaptures(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves)
{
	// TODO: Implement for black pieces
	assert(color == PieceColor::White);
}

void MoveGenerator::generatePawnCaptures(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves)
{
	// TODO: Implement for black pieces
	assert(color == PieceColor::White);
}

void MoveGenerator::generateKingMoves(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves)
{
	// TODO: Implement for black pieces
	assert(color == PieceColor::White);
}

void MoveGenerator::generatePawnMoves(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves)
{
	// TODO: Implement for black pieces
	assert(color == PieceColor::White);

	UINT empty_fields = pieces.getEmptyFields();

	// Generate moves in the base diagonal direction
	if ((position >> BASE_DIAGONAL_SHIFT) & empty_fields)
	{
		moves.emplace_back(position, position >> BASE_DIAGONAL_SHIFT);
	}

	if (position & MOVES_DOWN_LEFT_AVAILABLE)
	{
		// Generate moves in the down left direction
		if ((position >> DOWN_LEFT_SHIFT) & empty_fields)
		{
			moves.emplace_back(position, position >> DOWN_LEFT_SHIFT);
		}
	}

	if (position & MOVES_DOWN_RIGHT_AVAILABLE)
	{
		// Generate moves in the down right direction
		if ((position >> DOWN_RIGHT_SHIFT) & empty_fields)
		{
			moves.emplace_back(position, position >> DOWN_RIGHT_SHIFT);
		}
	}
}
