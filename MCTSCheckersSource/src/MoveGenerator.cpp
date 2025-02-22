#include <cassert>
#include <list>
#include <tuple>

#include "../includes/MoveGenerator.h"
#include "../includes/Utils.h"
#include "../includes/Board.h"
#include "../includes/ShiftMap.h"

//----------------------------------------------------------------
// Generating moves
//----------------------------------------------------------------

MoveList MoveGenerator::generateMoves(const BitBoard& pieces, PieceColor playerColor)
{
	// Find all of the jumpers
	MoveList moves;

	UINT jumpersL3 = getJumpersInShift(pieces, playerColor, BitShift::BIT_SHIFT_L3);
	UINT jumpersL4 = getJumpersInShift(pieces, playerColor, BitShift::BIT_SHIFT_L4);
	UINT jumpersL5 = getJumpersInShift(pieces, playerColor, BitShift::BIT_SHIFT_L5);
	UINT jumpersR3 = getJumpersInShift(pieces, playerColor, BitShift::BIT_SHIFT_R3);
	UINT jumpersR4 = getJumpersInShift(pieces, playerColor, BitShift::BIT_SHIFT_R4);
	UINT jumpersR5 = getJumpersInShift(pieces, playerColor, BitShift::BIT_SHIFT_R5);

	UINT jumpers2 = jumpersL3 | jumpersL4 | jumpersL5 | jumpersR3 | jumpersR4 | jumpersR5;

	if (jumpers2)
	{
		generateCapturingMovesInShift(pieces, playerColor, jumpersL3, BitShift::BIT_SHIFT_L3, moves);
		generateCapturingMovesInShift(pieces, playerColor, jumpersL4, BitShift::BIT_SHIFT_L4, moves);
		generateCapturingMovesInShift(pieces, playerColor, jumpersL5, BitShift::BIT_SHIFT_L5, moves);
		generateCapturingMovesInShift(pieces, playerColor, jumpersR3, BitShift::BIT_SHIFT_R3, moves);
		generateCapturingMovesInShift(pieces, playerColor, jumpersR4, BitShift::BIT_SHIFT_R4, moves);
		generateCapturingMovesInShift(pieces, playerColor, jumpersR5, BitShift::BIT_SHIFT_R5, moves);
		return moves;
	}

	UINT moversL3 = getMoversInShift(pieces, playerColor, BitShift::BIT_SHIFT_L3);
	UINT moversL4 = getMoversInShift(pieces, playerColor, BitShift::BIT_SHIFT_L4);
	UINT moversL5 = getMoversInShift(pieces, playerColor, BitShift::BIT_SHIFT_L5);
	UINT moversR3 = getMoversInShift(pieces, playerColor, BitShift::BIT_SHIFT_R3);
	UINT moversR4 = getMoversInShift(pieces, playerColor, BitShift::BIT_SHIFT_R4);
	UINT moversR5 = getMoversInShift(pieces, playerColor, BitShift::BIT_SHIFT_R5);

	generateBasicMovesInShift(pieces, playerColor, moversL3, BitShift::BIT_SHIFT_L3, moves);
	generateBasicMovesInShift(pieces, playerColor, moversL4, BitShift::BIT_SHIFT_L4, moves);
	generateBasicMovesInShift(pieces, playerColor, moversL5, BitShift::BIT_SHIFT_L5, moves);
	generateBasicMovesInShift(pieces, playerColor, moversR3, BitShift::BIT_SHIFT_R3, moves);
	generateBasicMovesInShift(pieces, playerColor, moversR4, BitShift::BIT_SHIFT_R4, moves);
	generateBasicMovesInShift(pieces, playerColor, moversR5, BitShift::BIT_SHIFT_R5, moves);
	return moves;
}

//----------------------------------------------------------------
// Generating a list of moves
//----------------------------------------------------------------

void MoveGenerator::generateBasicMovesInShift(const BitBoard& pieces, PieceColor playerColor, UINT movers, BitShift shift, MoveList& moves)
{
	while (movers) {
		UINT mover = movers & -movers; 
		movers ^= mover;

		if (mover & pieces.kings) {
			generateKingMoves(pieces, playerColor, mover, shift, moves);
		}
		else {
			generatePawnMovesInShift(pieces, playerColor, mover, shift, moves);
		}
	}
}

void MoveGenerator::generateCapturingMovesInShift(const BitBoard& pieces, PieceColor playerColor, UINT jumpers, BitShift shift, MoveList& moves)
{
	while (jumpers) {
		UINT jumper = jumpers & -jumpers;
		jumpers ^= jumper;

		if (jumper & pieces.kings) {
			generateKingCaptures(pieces, playerColor, jumper, shift, moves);
		}
		else {
			generatePawnCapturesInShift(pieces, playerColor, jumper, shift, moves);
		}
	}
}

//----------------------------------------------------------------
// Generating specified move
//----------------------------------------------------------------

void MoveGenerator::generateKingCaptures(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves, UINT captured)
{
	UINT emptyFields = pieces.getEmptyFields();
	UINT newPosition = position;
	UINT currentPieces = pieces.getPieces(playerColor);
	UINT enemyPieces = pieces.getPieces(getEnemyColor(playerColor));

	assert(position & pieces.kings & currentPieces);
	assert((currentPieces & enemyPieces) == 0);

	int iteration = 0;
	UINT foundEnemyPiece = 0;

	std::list<std::tuple<Move, BitShift>> singleCaptureMoves;

	while (newPosition)
	{
		BitShift nextShift = getNextShift(shift, iteration, newPosition);
		if (nextShift == BitShift::BIT_SHIFT_NONE)
			break;

		// Make the shift
		newPosition = ShiftMap::shift(newPosition, nextShift);
		iteration++;

		// If the newPosition contains captured piece, there is nothing to do
		if (newPosition & captured)
			break;

		// Shifted out of the board
		if (newPosition == 0)
			break;

		// There must not be any piece in the same color on the way if the captured pawn has not been found yet
		assert((newPosition & currentPieces) == 0 || foundEnemyPiece);
		if (newPosition & currentPieces)
			break;

		// If the newPosition contains empty field continue looking for enemy pieces
		if ((newPosition & emptyFields) > 0 && !foundEnemyPiece)
			continue;

		// If the newPosition contains empty field and we have already found enemy piece, add the move
		if ((newPosition & emptyFields) > 0 && foundEnemyPiece)
		{
			singleCaptureMoves.emplace_back(Move(position, newPosition, foundEnemyPiece, playerColor), shift);
			continue;
		}

		// Being here means the enemy piece is on the new position
		assert((newPosition & enemyPieces) > 0);

		// If we have already found enemy piece, we must not find another one
		if (foundEnemyPiece > 0)
			break;

		foundEnemyPiece = newPosition;
	}

	for (const auto& moveTuple : singleCaptureMoves)
	{
		const Move& move = std::get<0>(moveTuple);
		// const BitShift& nextShift = std::get<1>(moveTuple);

		// Create new board state after capture
		BitBoard newState = move.getBitboardAfterMove(pieces);

		// Generate all possible continuations
		std::queue<std::tuple<UINT, BitShift>> newJumpers;
		for (int i = 0; i < static_cast<int>(BitShift::COUNT); ++i) {
			BitShift nextShift = static_cast<BitShift>(i);
			UINT jumpers = getJumpersInShift(newState, playerColor, nextShift, captured | move.getCaptured());

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
				generateKingCaptures(newState, playerColor, move.getDestination(), nextShift, continuationMoves, captured | move.getCaptured());

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

void MoveGenerator::generatePawnCapturesInShift(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves)
{
	UINT emptyFields = pieces.getEmptyFields();
	UINT captured = ShiftMap::shift(position, shift);
	UINT currentPieces = pieces.getPieces(playerColor);
	UINT enemyPieces = pieces.getPieces(getEnemyColor(playerColor));
	UINT newPosition = 0;

	// Make sure the position is right color
	assert(position & currentPieces);

	// Make sure the captured piece is enemy piece
	assert(captured & enemyPieces);

	// Make sure the captured piece is not the same as the position
	assert((captured & currentPieces) == 0);

	// Make sure the captured piece is not empty field
	assert((captured & emptyFields) == 0);

	// Make sure the pieces are marked correctly
	assert((enemyPieces & currentPieces) == 0);

	// There must be a captured piece
	if ((captured & enemyPieces) == 0)
	{
		printf("position:\n");
		Board::printBitboard(position);
		printf("currentPieces:\n");
		Board::printBitboard(currentPieces);
		printf("enemyPieces:\n");
		Board::printBitboard(enemyPieces);
		printf("kings:\n");
		Board::printBitboard(pieces.kings);
		printf("captured:\n");
		Board::printBitboard(captured);
		printf("shift:\n");
		switch (shift)
		{
		case BitShift::BIT_SHIFT_L3:
			printf("L3\n");
			break;
		case BitShift::BIT_SHIFT_L4:
			printf("L4\n");
			break;
		case BitShift::BIT_SHIFT_L5:
			printf("L5\n");
			break;
		case BitShift::BIT_SHIFT_R3:
			printf("R3\n");
			break;
		case BitShift::BIT_SHIFT_R4:
			printf("R4\n");
			break;
		case BitShift::BIT_SHIFT_R5:
			printf("R5\n");
			break;
		default:
			break;
		}

		printf("gotit\n");
	}
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
	Move singleCapture = Move(position, newPosition, captured, playerColor);

	// Create new board state after capture
	BitBoard newState = singleCapture.getBitboardAfterMove(pieces, false);

	// Generate all possible continuations
	std::queue<std::tuple<UINT, BitShift>> newJumpers;
	BitShift reverseShift = ShiftMap::getOpposite(shift);

	for (int i = 0; i < static_cast<int>(BitShift::COUNT); ++i) {
		BitShift nextShift = static_cast<BitShift>(i);
		UINT jumpers = getJumpersInShift(newState, playerColor, nextShift);
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
			
			// debug log
			generatePawnCapturesInShift(newState, playerColor, singleCapture.getDestination(), nextShift, continuationMoves);

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

void MoveGenerator::generateKingMoves(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves)
{
	UINT emptyFields = pieces.getEmptyFields();
	UINT newPosition = position;
	int iteration = 0;

	while (newPosition)
	{
		BitShift nextShift = getNextShift(shift, iteration, newPosition);
		if (nextShift == BitShift::BIT_SHIFT_NONE)
			break;

		newPosition = ShiftMap::shift(newPosition, nextShift);

		if (!(newPosition & emptyFields))
			break;

		moves.emplace_back(position, newPosition, 0, playerColor);
		iteration++;
	}
}

void MoveGenerator::generatePawnMovesInShift(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves)
{
	UINT newPosition = ShiftMap::shift(position, shift);
	moves.emplace_back(position, newPosition, 0, playerColor);
}
