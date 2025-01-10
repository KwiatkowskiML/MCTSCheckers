#include <cassert>

#include "MoveGenerator.h"
#include "Utils.h"
#include "Board2.h"
#include "ShiftMap.h"

//----------------------------------------------------------------
// Generating moves
//----------------------------------------------------------------

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
	UINT moversL3 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_L3);
	UINT moversL4 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_L4);
	UINT moversL5 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_L5);
	UINT moversR3 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_R3);
	UINT moversR4 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_R4);
	UINT moversR5 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_R5);

	UINT movers2 = moversL3 | moversL4 | moversL5 | moversR3 | moversR4 | moversR5;

	printf("Movers: %d\n", movers);
	printf("Movers2: %d\n", movers2);
	assert(movers == movers2);
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
	UINT captBlackPawns = (emptyFields << SHIFT_BASE) & pieces.blackPawns;

	// Check whether previously specified black pawns can actually be captured
	if (captBlackPawns)
	{
		// Get the white pawns that can capture black pawn in the base diagonal direction
		jumpers |= ((captBlackPawns & MASK_L3) << SHIFT_L3) & pieces.whitePawns;
		jumpers |= ((captBlackPawns & MASK_L5) << SHIFT_L5) & pieces.whitePawns;
	}

	// Get the black pawns that might be captured in the other diagonal direction
	captBlackPawns = ((emptyFields & MASK_L3) << SHIFT_L3) & pieces.blackPawns;
	captBlackPawns |= ((emptyFields & MASK_L5) << SHIFT_L5) & pieces.blackPawns;

	jumpers |= (captBlackPawns << SHIFT_BASE) & pieces.whitePawns;

	// Find all of the black pawns that might be captured backwards in base diagonal
	captBlackPawns = (emptyFields >> SHIFT_BASE) & pieces.blackPawns;

	// Check whether previously specified black pawns can actually be captured
	if (captBlackPawns)
	{
		jumpers |= ((captBlackPawns & MASK_R5) >> SHIFT_R5) & pieces.whitePawns;
		jumpers |= ((captBlackPawns & MASK_R3) >> SHIFT_R3) & pieces.whitePawns;
	}

	// Find all of the black pawns that might be captured backwards in the other diagonal
	captBlackPawns = ((emptyFields & MASK_R5) >> SHIFT_R5) & pieces.blackPawns;
	captBlackPawns |= ((emptyFields & MASK_R3) >> SHIFT_R3) & pieces.blackPawns;
	jumpers |= (captBlackPawns >> SHIFT_BASE) & pieces.whitePawns;

	// TODO: Consider if there is a need for analizing kings - there IS

	return jumpers;
}

UINT MoveGenerator::getMovers(const BitBoard& pieces, PieceColor color)
{
	// TODO: Implement for black pieces
	assert(color == PieceColor::White);

	//const UINT emptyFields = pieces.getEmptyFields();
	//const UINT whiteKings = pieces.whitePawns & pieces.kings;

	//// Get the white pieces that can move in the basic diagonal direction (right down or left down, depending on the row)
	//UINT movers = (emptyFields << SHIFT_BASE) & pieces.whitePawns;

	//// Get the white pieces that can move in the right down direction
	//movers |= ((emptyFields & MASK_L3) << SHIFT_L3) & pieces.whitePawns;

	//// Get the white pieces that can move in the left down direction
	//movers |= ((emptyFields & MASK_L5) << SHIFT_L5) & pieces.whitePawns;

	//// Get the white kings that can move in the upper diagonal direction (right up or left up)
	//if (whiteKings)
	//{
	//	movers |= (emptyFields >> SHIFT_BASE) & whiteKings;
	//	movers |= ((emptyFields & MASK_R3) >> SHIFT_R3) & whiteKings;
	//	movers |= ((emptyFields & MASK_R5) >> SHIFT_R5) & whiteKings;
	//}


	UINT moversL3 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_L3);
	UINT moversL4 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_L4);
	UINT moversL5 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_L5);
	UINT moversR3 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_R3);
	UINT moversR4 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_R4);
	UINT moversR5 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_R5);

	UINT movers = moversL3 | moversL4 | moversL5 | moversR3 | moversR4 | moversR5;

	return movers;
}

UINT MoveGenerator::getMoversInShift(const BitBoard& pieces, PieceColor color, BitShift shift)
{
	// TODO: Implement for black pieces
	assert(color == PieceColor::White);

	const UINT emptyFields = pieces.getEmptyFields();
	const UINT whiteKings = pieces.whitePawns & pieces.kings;

	UINT movers = 0;
	BitShift reverseShift = ShiftMap::getOpposite(shift);

	if (shift == BitShift::BIT_SHIFT_R4 || shift == BitShift::BIT_SHIFT_R3 || shift == BitShift::BIT_SHIFT_R5)
	{
		movers |= ShiftMap::shift(emptyFields, reverseShift) & pieces.whitePawns;
	}
	else if (whiteKings && (shift == BitShift::BIT_SHIFT_L4 || shift == BitShift::BIT_SHIFT_L3 || shift == BitShift::BIT_SHIFT_L5))
	{
		movers |= ShiftMap::shift(emptyFields, reverseShift) & whiteKings;
	}
	
	return movers;
}

//----------------------------------------------------------------
// Generating a list of moves
//----------------------------------------------------------------

void MoveGenerator::generateBasicMoves(const BitBoard& pieces, PieceColor color, UINT movers, MoveList& moves)
{
	while (movers) {
		UINT mover = movers & -movers; // TODO: reconsider this
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
	UINT empty_fields = pieces.getEmptyFields();
	MoveList singleCaptureMoves;

	//--------------------------------------------------------------------------------
	// Capturing black pawns below the white pawn
	//--------------------------------------------------------------------------------

	// Generate capturing moves in the base down diagonal direction
	UINT newPosition = position >> SHIFT_BASE;
	if (newPosition & pieces.blackPawns)
	{
		UINT captured = newPosition;
		if (newPosition & MASK_R5)
		{
			newPosition >>= SHIFT_R5;
			if (newPosition & empty_fields)
				singleCaptureMoves.emplace_back(position, newPosition, captured);
		}
		else if (newPosition & MASK_R3)
		{
			newPosition >>= SHIFT_R3;
			if (newPosition & empty_fields)
				singleCaptureMoves.emplace_back(position, newPosition, captured);
		}
	}

	// Generate capturing moves in the down left direction
	if (position & MASK_R5)
	{
		newPosition = position >> SHIFT_R5;
		if (newPosition & pieces.blackPawns)
		{
			UINT captured = newPosition;
			newPosition >>= SHIFT_BASE;
			if (newPosition & empty_fields)
				singleCaptureMoves.emplace_back(position, newPosition, captured);
		}
	}

	// Generate capturing moves in the down right direction
	if (position & MASK_R3)
	{
		newPosition = position >> SHIFT_R3;
		if (newPosition & pieces.blackPawns)
		{
			UINT captured = newPosition;
			newPosition >>= SHIFT_BASE;
			if (newPosition & empty_fields)
				singleCaptureMoves.emplace_back(position, newPosition, captured);
		}
	}

	//--------------------------------------------------------------------------------
	// Capturing black pawns above the white pawn
	//--------------------------------------------------------------------------------

	// Generate capturing moves in the base upper diagonal direction
	newPosition = position << SHIFT_BASE;
	if (newPosition & pieces.blackPawns)
	{
		UINT captured = newPosition;
		if (newPosition & MASK_L3)
		{
			newPosition <<= SHIFT_L3;
			if (newPosition & empty_fields)
				singleCaptureMoves.emplace_back(position, newPosition, captured);
		}
		else if (newPosition & MASK_L5)
		{
			newPosition <<= SHIFT_L5;
			if (newPosition & empty_fields)
				singleCaptureMoves.emplace_back(position, newPosition, captured);
		}
	}

	// Generate capturing moves in the up left direction
	if (position & MASK_L3)
	{
		newPosition = position << SHIFT_L3;
		if (newPosition & pieces.blackPawns)
		{
			UINT captured = newPosition;
			newPosition <<= SHIFT_BASE;
			if (newPosition & empty_fields)
				singleCaptureMoves.emplace_back(position, newPosition, captured);
		}
	}

	// Generate capturing moves in the up right direction
	if (position & MASK_L5)
	{
		newPosition = position << SHIFT_L5;
		if (newPosition & pieces.blackPawns)
		{
			UINT captured = newPosition;
			newPosition <<= SHIFT_BASE;
			if (newPosition & empty_fields)
				singleCaptureMoves.emplace_back(position, newPosition, captured);
		}
	}

	Board2 currentState(pieces.whitePawns, pieces.blackPawns, pieces.kings);

	// Process each single capture and check for continuations
	for (const Move& singleCapture : singleCaptureMoves) {

		// Create new board state after capture
		BitBoard newState = singleCapture.getBitboardAfterMove(pieces);

		// Check for additional captures from the new position
		UINT newJumpers = getJumpers(newState, PieceColor::White);

		if (newJumpers & singleCapture.destination) {
			// Recursively generate additional captures
			MoveList continuationMoves;
			generatePawnCaptures(newState, PieceColor::White, singleCapture.destination, continuationMoves);

			// If no continuations found, add the single capture
			if (continuationMoves.empty()) {
				moves.push_back(singleCapture);
			}
			// Add all continuation moves
			for (const Move& continuation : continuationMoves) {
				Move combinedMove = singleCapture;
				combinedMove.destination = continuation.destination;
				combinedMove.captured |= continuation.captured;
				moves.push_back(combinedMove);
			}
		}
		else {
			// No continuations possible, add the single capture
			moves.push_back(singleCapture);
		}
	}
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
	if ((position >> SHIFT_BASE) & empty_fields)
	{
		moves.emplace_back(position, position >> SHIFT_BASE);
	}

	if (position & MASK_R5)
	{
		// Generate moves in the down left direction
		if ((position >> SHIFT_R5) & empty_fields)
		{
			moves.emplace_back(position, position >> SHIFT_R5);
		}
	}

	if (position & MASK_R3)
	{
		// Generate moves in the down right direction
		if ((position >> SHIFT_R3) & empty_fields)
		{
			moves.emplace_back(position, position >> SHIFT_R3);
		}
	}
}
