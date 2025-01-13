#include <cassert>

#include "MoveGenerator.h"
#include "Utils.h"
#include "Board.h"
#include "ShiftMap.h"
#include <list>
#include <tuple>

//----------------------------------------------------------------
// Generating moves
//----------------------------------------------------------------

MoveList MoveGenerator::generateMoves(const BitBoard& pieces, PieceColor color)
{
	// Find all of the jumpers
	MoveList moves;

	UINT jumpersL3 = getJumpersInShift(pieces, color, BitShift::BIT_SHIFT_L3);
	UINT jumpersL4 = getJumpersInShift(pieces, color, BitShift::BIT_SHIFT_L4);
	UINT jumpersL5 = getJumpersInShift(pieces, color, BitShift::BIT_SHIFT_L5);
	UINT jumpersR3 = getJumpersInShift(pieces, color, BitShift::BIT_SHIFT_R3);
	UINT jumpersR4 = getJumpersInShift(pieces, color, BitShift::BIT_SHIFT_R4);
	UINT jumpersR5 = getJumpersInShift(pieces, color, BitShift::BIT_SHIFT_R5);

	UINT jumpers2 = jumpersL3 | jumpersL4 | jumpersL5 | jumpersR3 | jumpersR4 | jumpersR5;

	if (jumpers2)
	{
		generateCapturingMovesInShift(pieces, color, jumpersL3, BitShift::BIT_SHIFT_L3, moves);
		generateCapturingMovesInShift(pieces, color, jumpersL4, BitShift::BIT_SHIFT_L4, moves);
		generateCapturingMovesInShift(pieces, color, jumpersL5, BitShift::BIT_SHIFT_L5, moves);
		generateCapturingMovesInShift(pieces, color, jumpersR3, BitShift::BIT_SHIFT_R3, moves);
		generateCapturingMovesInShift(pieces, color, jumpersR4, BitShift::BIT_SHIFT_R4, moves);
		generateCapturingMovesInShift(pieces, color, jumpersR5, BitShift::BIT_SHIFT_R5, moves);
		return moves;
	}

	UINT moversL3 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_L3);
	UINT moversL4 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_L4);
	UINT moversL5 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_L5);
	UINT moversR3 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_R3);
	UINT moversR4 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_R4);
	UINT moversR5 = getMoversInShift(pieces, color, BitShift::BIT_SHIFT_R5);

	generateBasicMovesInShift(pieces, color, moversL3, BitShift::BIT_SHIFT_L3, moves);
	generateBasicMovesInShift(pieces, color, moversL4, BitShift::BIT_SHIFT_L4, moves);
	generateBasicMovesInShift(pieces, color, moversL5, BitShift::BIT_SHIFT_L5, moves);
	generateBasicMovesInShift(pieces, color, moversR3, BitShift::BIT_SHIFT_R3, moves);
	generateBasicMovesInShift(pieces, color, moversR4, BitShift::BIT_SHIFT_R4, moves);
	generateBasicMovesInShift(pieces, color, moversR5, BitShift::BIT_SHIFT_R5, moves);
	return moves;
}

//----------------------------------------------------------------
// Getting moveable pieces
//----------------------------------------------------------------

UINT MoveGenerator::getAllMovers(const BitBoard& pieces, PieceColor color)
{
	UINT movers = 0;

	for (int i = 0; i < static_cast<int>(BitShift::COUNT); ++i) {
		BitShift shift = static_cast<BitShift>(i);
		movers |= getMoversInShift(pieces, color, shift);
	}

	return movers;
}

UINT MoveGenerator::getAllJumpers(const BitBoard& pieces, PieceColor color)
{
	UINT jumpers = 0;

	for (int i = 0; i < static_cast<int>(BitShift::COUNT); ++i) {
		BitShift shift = static_cast<BitShift>(i);
		jumpers |= getJumpersInShift(pieces, color, shift);
	}

	return jumpers;
}

UINT MoveGenerator::getJumpersInShift(const BitBoard& pieces, PieceColor color, BitShift shift, UINT captured)
{
	const UINT emptyFields = pieces.getEmptyFields();
	const UINT currentPieces = pieces.getPieces(color);
	const UINT kings = currentPieces & pieces.kings;
	const UINT enemyPieces = pieces.getPieces(getEnemyColor(color));

	UINT jumpers = 0;
	UINT captPieces = 0;

	// Finding capturable pieces
	if (shift == BitShift::BIT_SHIFT_R3 || shift == BitShift::BIT_SHIFT_R5)
	{
		captPieces |= ShiftMap::shift(emptyFields, BitShift::BIT_SHIFT_L4) & enemyPieces;
	}
	else if (shift == BitShift::BIT_SHIFT_L3 || shift == BitShift::BIT_SHIFT_L5)
	{
		captPieces |= ShiftMap::shift(emptyFields, BitShift::BIT_SHIFT_R4) & enemyPieces;
	}
	else if (shift == BitShift::BIT_SHIFT_R4)
	{
		captPieces |= ShiftMap::shift(emptyFields, BitShift::BIT_SHIFT_L3) & enemyPieces;
		captPieces |= ShiftMap::shift(emptyFields, BitShift::BIT_SHIFT_L5) & enemyPieces;
	}
	else if (shift == BitShift::BIT_SHIFT_L4)
	{
		captPieces |= ShiftMap::shift(emptyFields, BitShift::BIT_SHIFT_R3) & enemyPieces;
		captPieces |= ShiftMap::shift(emptyFields, BitShift::BIT_SHIFT_R5) & enemyPieces;
	}

	// No pieces to capture
	if (!captPieces)
		return jumpers;

	// Get the pieces that can capture enemy pieces
	BitShift reverseShift = ShiftMap::getOpposite(shift);
	jumpers |= ShiftMap::shift(captPieces, reverseShift) & currentPieces;

	// Find the kings that can capture black pawns
	UINT nonTaggedKings = kings & ~jumpers;
	if (nonTaggedKings)
	{
		while (nonTaggedKings) {
			UINT currentKing = nonTaggedKings & -nonTaggedKings;
			nonTaggedKings ^= currentKing;
			UINT movedCurrentKing = currentKing;
			int iteration = 0;
			bool foundEnemyPiece = false;

			// Move the kings untill capturable enemy pawn is found
			while (movedCurrentKing)
			{
				BitShift nextShift = shift;
				if (shift == BitShift::BIT_SHIFT_L3 || shift == BitShift::BIT_SHIFT_L5)
				{
					if (iteration & 1)
						nextShift = BitShift::BIT_SHIFT_L4;
				}

				if (shift == BitShift::BIT_SHIFT_R3 || shift == BitShift::BIT_SHIFT_R5)
				{
					if (iteration & 1)
						nextShift = BitShift::BIT_SHIFT_R4;
				}

				if (shift == BitShift::BIT_SHIFT_L4)
				{
					if (iteration & 1)
					{
						if (movedCurrentKing & MASK_L3)
							nextShift = BitShift::BIT_SHIFT_L3;
						else if (movedCurrentKing & MASK_L5)
							nextShift = BitShift::BIT_SHIFT_L5;
						else
							break;
					}
				}

				if (shift == BitShift::BIT_SHIFT_R4)
				{
					if (iteration & 1)
					{
						if (movedCurrentKing & MASK_R3)
							nextShift = BitShift::BIT_SHIFT_R3;
						else if (movedCurrentKing & MASK_R5)
							nextShift = BitShift::BIT_SHIFT_R5;
						else
							break;
					}
				}

				movedCurrentKing = ShiftMap::shift(movedCurrentKing, nextShift);
				iteration++;

				// found captured piece on its way, there is nothing to do
				if (movedCurrentKing & captured)
					break;

				// found current color piece on the way
				if (movedCurrentKing & currentPieces)
					break;

				// found empty field on the way
				if (movedCurrentKing & emptyFields && !foundEnemyPiece)
					continue;

				// found empty field right after enemy piece
				if (movedCurrentKing & emptyFields && foundEnemyPiece)
				{
					jumpers |= currentKing;
					break;
				}

				// found enemy piece on the way
				if (movedCurrentKing & enemyPieces)
				{
					// found enemy piece right after enemy piece
					if (foundEnemyPiece)
						break;

					foundEnemyPiece = true;
					continue;
				}
			}
		}
	}

	return jumpers;
}

UINT MoveGenerator::getMoversInShift(const BitBoard& pieces, PieceColor color, BitShift shift)
{
	const UINT emptyFields = pieces.getEmptyFields();
	const UINT pawns = pieces.getPieces(color);
	const UINT kings = pawns & pieces.kings;

	UINT movers = 0;
	BitShift reverseShift = ShiftMap::getOpposite(shift);

	if (color == PieceColor::White)
	{
		if (shift == BitShift::BIT_SHIFT_R4 || shift == BitShift::BIT_SHIFT_R3 || shift == BitShift::BIT_SHIFT_R5)
		{
			movers |= ShiftMap::shift(emptyFields, reverseShift) & pawns;
		}
		else if (kings && (shift == BitShift::BIT_SHIFT_L4 || shift == BitShift::BIT_SHIFT_L3 || shift == BitShift::BIT_SHIFT_L5))
		{
			movers |= ShiftMap::shift(emptyFields, reverseShift) & kings;
		}
	}
	else
	{
		if (shift == BitShift::BIT_SHIFT_L4 || shift == BitShift::BIT_SHIFT_L3 || shift == BitShift::BIT_SHIFT_L5)
		{
			movers |= ShiftMap::shift(emptyFields, reverseShift) & pawns;
		}
		else if (kings && (shift == BitShift::BIT_SHIFT_R4 || shift == BitShift::BIT_SHIFT_R3 || shift == BitShift::BIT_SHIFT_R5))
		{
			movers |= ShiftMap::shift(emptyFields, reverseShift) & kings;
		}
	}

	return movers;
}

//----------------------------------------------------------------
// Generating a list of moves
//----------------------------------------------------------------

void MoveGenerator::generateBasicMovesInShift(const BitBoard& pieces, PieceColor color, UINT movers, BitShift shift, MoveList& moves)
{
	while (movers) {
		UINT mover = movers & -movers; // TODO: reconsider this
		movers ^= mover;

		if (mover & pieces.kings) {
			generateKingMoves(pieces, color, mover, shift, moves);
		}
		else {
			generatePawnMovesInShift(pieces, color, mover, shift, moves);
		}
	}
}

void MoveGenerator::generateCapturingMovesInShift(const BitBoard& pieces, PieceColor color, UINT jumpers, BitShift shift, MoveList& moves)
{
	while (jumpers) {
		UINT jumper = jumpers & -jumpers;
		jumpers ^= jumper;

		if (jumper & pieces.kings) {
			generateKingCaptures(pieces, color, jumper, shift, moves);
		}
		else {
			generatePawnCapturesInShift(pieces, color, jumper, shift, moves);
		}
	}
}

//----------------------------------------------------------------
// Generating specified move
//----------------------------------------------------------------

void MoveGenerator::generateKingCaptures(const BitBoard& pieces, PieceColor color, UINT position, BitShift shift, MoveList& moves, UINT captured)
{
	UINT emptyFields = pieces.getEmptyFields();
	UINT newPosition = position;
	UINT currentPieces = pieces.getPieces(color);
	UINT enemyPieces = pieces.getPieces(getEnemyColor(color));

	int iteration = 0;
	UINT foundEnemyPiece = 0;

	std::list<std::tuple<Move, BitShift>> singleCapturedPieces;

	while (newPosition)
	{
		BitShift nextShift = shift;
		if (shift == BitShift::BIT_SHIFT_L3 || shift == BitShift::BIT_SHIFT_L5)
		{
			if (iteration & 1)
				nextShift = BitShift::BIT_SHIFT_L4;
		}

		if (shift == BitShift::BIT_SHIFT_R3 || shift == BitShift::BIT_SHIFT_R5)
		{
			if (iteration & 1)
				nextShift = BitShift::BIT_SHIFT_R4;
		}

		if (shift == BitShift::BIT_SHIFT_L4)
		{
			if (iteration & 1)
			{
				if (newPosition & MASK_L3)
					nextShift = BitShift::BIT_SHIFT_L3;
				else if (newPosition & MASK_L5)
					nextShift = BitShift::BIT_SHIFT_L5;
				else
					break;
			}
		}

		if (shift == BitShift::BIT_SHIFT_R4)
		{
			if (iteration & 1)
			{
				if (newPosition & MASK_R3)
					nextShift = BitShift::BIT_SHIFT_R3;
				else if (newPosition & MASK_R5)
					nextShift = BitShift::BIT_SHIFT_R5;
				else
					break;
			}
		}

		// Make the shift
		newPosition = ShiftMap::shift(newPosition, nextShift);
		iteration++;

		// If the newPosition contains captured piece, there is nothing to do
		if (newPosition & captured)
			break;

		// Shifted out of the board
		if (newPosition == 0)
			break;

		// There must not be any piece in the same color on the way
		assert((newPosition & currentPieces) == 0);
		if (newPosition & currentPieces)
			break;

		// If the newPosition contains empty field continue looking for enemy pieces
		if ((newPosition & emptyFields) > 0 && !foundEnemyPiece)
			continue;

		// If the newPosition contains empty field and we have already found enemy piece, add the move
        if ((newPosition & emptyFields) > 0 && foundEnemyPiece)
        {
			singleCapturedPieces.emplace_back(Move(position, newPosition, foundEnemyPiece, color), shift);
			continue;
        }

		// Being here means the enemy piece is on the new position
		assert((newPosition & enemyPieces) > 0);

		// If we have already found enemy piece, we must not find another one
		if (foundEnemyPiece > 0)
			break;

		foundEnemyPiece = newPosition;
	}

    for (const auto& moveTuple : singleCapturedPieces)
    {
		const Move& move = std::get<0>(moveTuple);
		const BitShift& nextShift = std::get<1>(moveTuple);

		// Create new board state after capture
		BitBoard newState = move.getBitboardAfterMove(pieces);

		// Generate all possible continuations
		std::queue<std::tuple<UINT, BitShift>> newJumpers;
		for (int i = 0; i < static_cast<int>(BitShift::COUNT); ++i) {
			BitShift nextShift = static_cast<BitShift>(i);
			UINT jumpers = getJumpersInShift(newState, color, nextShift, captured | move.getCaptured());

			// Destination cannot move any further 
			if ((jumpers & move.getDestination()) == 0)
				continue;

			newJumpers.push(std::make_tuple(jumpers, nextShift));
		}

		if (!newJumpers.empty()) {

			// Recursively generate additional captures
			while (!newJumpers.empty()) {
				UINT newJumper;
				BitShift nextShift;
				std::tie(newJumper, nextShift) = newJumpers.front();
				newJumpers.pop();
				MoveList continuationMoves;
				generateKingCaptures(newState, color, move.getDestination(), nextShift, continuationMoves, captured | move.getCaptured());

				// Add all continuation moves
				for (const Move& continuation : continuationMoves) {
					Move combinedMove = move.getExtendedMove(continuation, continuation.getCaptured());
					moves.push_back(combinedMove);
				}
			}
		}
		else {
			// No continuations possible, add the single capture
			moves.push_back(move);
		}
    }
}

void MoveGenerator::generatePawnCapturesInShift(const BitBoard& pieces, PieceColor color, UINT position, BitShift shift, MoveList& moves)
{
	// TODO: Add validation for the shift

	UINT emptyFields = pieces.getEmptyFields();
	UINT captured = ShiftMap::shift(position, shift);
	UINT currentPieces = pieces.getPieces(color);
	UINT enemyPieces = pieces.getPieces(getEnemyColor(color));
	UINT newPosition = 0;

	// There must be a captured piece
	assert((captured & enemyPieces) != 0);

	// Capturing pieces below the pawn
	if (shift == BitShift::BIT_SHIFT_R3 || shift == BitShift::BIT_SHIFT_R5)
		newPosition = ShiftMap::shift(captured, BitShift::BIT_SHIFT_R4);

	if (shift == BitShift::BIT_SHIFT_R4)
	{
		if (captured & MASK_R3)
		{
			newPosition = ShiftMap::shift(captured, BitShift::BIT_SHIFT_R3);
		}

		if (captured & MASK_R5)
		{
			newPosition = ShiftMap::shift(captured, BitShift::BIT_SHIFT_R5);
		}
	}

	// Capturing pieces above the pawn
	if (shift == BitShift::BIT_SHIFT_L3 || shift == BitShift::BIT_SHIFT_L5)
		newPosition = ShiftMap::shift(captured, BitShift::BIT_SHIFT_L4);

	if (shift == BitShift::BIT_SHIFT_L4)
	{
		if (captured & MASK_L3)
		{
			newPosition = ShiftMap::shift(captured, BitShift::BIT_SHIFT_L3);
		}
		if (captured & MASK_L5)
		{
			newPosition = ShiftMap::shift(captured, BitShift::BIT_SHIFT_L5);
		}
	}

	// Create the move
	assert(newPosition != 0);
	Move singleCapture = Move(position, newPosition, captured, color);

	// Create new board state after capture
	BitBoard newState = singleCapture.getBitboardAfterMove(pieces);

	// Generate all possible continuations
	std::queue<std::tuple<UINT, BitShift>> newJumpers;
	BitShift reverseShift = ShiftMap::getOpposite(shift);

	for (int i = 0; i < static_cast<int>(BitShift::COUNT); ++i) {
		BitShift nextShift = static_cast<BitShift>(i);
		UINT jumpers = getJumpersInShift(newState, color, nextShift);
		if (jumpers & singleCapture.getDestination())
			newJumpers.push(std::make_tuple(jumpers, nextShift));
	}

	if (!newJumpers.empty()) {
		// Recursively generate additional captures
		while (!newJumpers.empty()) {
			UINT newJumper;
			BitShift nextShift;
			std::tie(newJumper, nextShift) = newJumpers.front();
			newJumpers.pop();
			MoveList continuationMoves;
			generatePawnCapturesInShift(newState, color, singleCapture.getDestination(), nextShift, continuationMoves);

			// Add all continuation moves
			for (const Move& continuation : continuationMoves) {
				Move combinedMove = singleCapture.getExtendedMove(continuation, continuation.getCaptured());
				moves.push_back(combinedMove);
			}
		}
	}
	else {
		// No continuations possible, add the single capture
		moves.push_back(singleCapture);
	}

}

void MoveGenerator::generateKingMoves(const BitBoard& pieces, PieceColor color, UINT position, BitShift shift, MoveList& moves)
{
	UINT emptyFields = pieces.getEmptyFields();
	UINT newPosition = position;
	int iteration = 0;

	while (newPosition)
	{
		if (shift == BitShift::BIT_SHIFT_L3 || shift == BitShift::BIT_SHIFT_L5)
		{
			if (iteration & 1)
				newPosition = ShiftMap::shift(newPosition, BitShift::BIT_SHIFT_L4);
			else
				newPosition = ShiftMap::shift(newPosition, shift);
		}

		if (shift == BitShift::BIT_SHIFT_R3 || shift == BitShift::BIT_SHIFT_R5)
		{
			if (iteration & 1)
				newPosition = ShiftMap::shift(newPosition, BitShift::BIT_SHIFT_R4);
			else
				newPosition = ShiftMap::shift(newPosition, shift);
		}

		if (shift == BitShift::BIT_SHIFT_L4)
		{
			if (iteration & 1)
			{
				if (newPosition & MASK_L3)
					newPosition = ShiftMap::shift(newPosition, BitShift::BIT_SHIFT_L3);
				else if (newPosition & MASK_L5)
					newPosition = ShiftMap::shift(newPosition, BitShift::BIT_SHIFT_L5);
				else
					break;
			}
			else
				newPosition = ShiftMap::shift(newPosition, shift);
		}

		if (shift == BitShift::BIT_SHIFT_R4)
		{
			if (iteration & 1)
			{
				if (newPosition & MASK_R3)
					newPosition = ShiftMap::shift(newPosition, BitShift::BIT_SHIFT_R3);
				else if (newPosition & MASK_R5)
					newPosition = ShiftMap::shift(newPosition, BitShift::BIT_SHIFT_R5);
				else
					break;
			}
			else
				newPosition = ShiftMap::shift(newPosition, shift);
		}

		if (!(newPosition & emptyFields))
			break;

		moves.emplace_back(position, newPosition, 0, color);
		iteration++;
	}
}

void MoveGenerator::generatePawnMovesInShift(const BitBoard& pieces, PieceColor color, UINT position, BitShift shift, MoveList& moves)
{
	UINT newPosition = ShiftMap::shift(position, shift);
	moves.emplace_back(position, newPosition, 0, color);
}

//----------------------------------------------------------------
// Getting enemy color
//----------------------------------------------------------------

PieceColor MoveGenerator::getEnemyColor(PieceColor color)
{
	return color == PieceColor::White ? PieceColor::Black : PieceColor::White;
}

