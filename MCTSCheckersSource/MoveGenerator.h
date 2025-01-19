#pragma once
#include "Move.h"
#include "PieceColor.h"
#include "BitBoard.h"
#include "BitShift.h"
#include "ShiftMap.h"
#include "Queue.h"
#include "Move2.h"

#define ASSERTS_ON

class MoveGenerator {
private:
	
public:
	//----------------------------------------------------------------
	// Generating moves
	//----------------------------------------------------------------

	__device__ __host__ static void generateMovesGpu(const BitBoard& pieces, PieceColor playerColor, Queue<Move2>* moves)
	{
		UINT jumpersL3 = getJumpersInShift(pieces, playerColor, BitShift::BIT_SHIFT_L3);
		UINT jumpersL4 = getJumpersInShift(pieces, playerColor, BitShift::BIT_SHIFT_L4);
		UINT jumpersL5 = getJumpersInShift(pieces, playerColor, BitShift::BIT_SHIFT_L5);
		UINT jumpersR3 = getJumpersInShift(pieces, playerColor, BitShift::BIT_SHIFT_R3);
		UINT jumpersR4 = getJumpersInShift(pieces, playerColor, BitShift::BIT_SHIFT_R4);
		UINT jumpersR5 = getJumpersInShift(pieces, playerColor, BitShift::BIT_SHIFT_R5);

		UINT jumpers2 = jumpersL3 | jumpersL4 | jumpersL5 | jumpersR3 | jumpersR4 | jumpersR5;

		if (jumpers2)
		{
			generateCapturingMovesInShiftGpu(pieces, playerColor, jumpersL3, BitShift::BIT_SHIFT_L3, moves);
			generateCapturingMovesInShiftGpu(pieces, playerColor, jumpersL4, BitShift::BIT_SHIFT_L4, moves);
			generateCapturingMovesInShiftGpu(pieces, playerColor, jumpersL5, BitShift::BIT_SHIFT_L5, moves);
			generateCapturingMovesInShiftGpu(pieces, playerColor, jumpersR3, BitShift::BIT_SHIFT_R3, moves);
			generateCapturingMovesInShiftGpu(pieces, playerColor, jumpersR4, BitShift::BIT_SHIFT_R4, moves);
			generateCapturingMovesInShiftGpu(pieces, playerColor, jumpersR5, BitShift::BIT_SHIFT_R5, moves);
			return;
		}

		UINT moversL3 = getMoversInShift(pieces, playerColor, BitShift::BIT_SHIFT_L3);
		UINT moversL4 = getMoversInShift(pieces, playerColor, BitShift::BIT_SHIFT_L4);
		UINT moversL5 = getMoversInShift(pieces, playerColor, BitShift::BIT_SHIFT_L5);
		UINT moversR3 = getMoversInShift(pieces, playerColor, BitShift::BIT_SHIFT_R3);
		UINT moversR4 = getMoversInShift(pieces, playerColor, BitShift::BIT_SHIFT_R4);
		UINT moversR5 = getMoversInShift(pieces, playerColor, BitShift::BIT_SHIFT_R5);

		generateBasicMovesInShiftGpu(pieces, playerColor, moversL3, BitShift::BIT_SHIFT_L3, moves);
		generateBasicMovesInShiftGpu(pieces, playerColor, moversL4, BitShift::BIT_SHIFT_L4, moves);
		generateBasicMovesInShiftGpu(pieces, playerColor, moversL5, BitShift::BIT_SHIFT_L5, moves);
		generateBasicMovesInShiftGpu(pieces, playerColor, moversR3, BitShift::BIT_SHIFT_R3, moves);
		generateBasicMovesInShiftGpu(pieces, playerColor, moversR4, BitShift::BIT_SHIFT_R4, moves);
		generateBasicMovesInShiftGpu(pieces, playerColor, moversR5, BitShift::BIT_SHIFT_R5, moves);
		return;
	}
	
	__device__ __host__ static void generateBasicMovesInShiftGpu(const BitBoard& pieces, PieceColor playerColor, UINT movers, BitShift shift, Queue<Move2>* moves)
	{
		while (movers) {
			UINT mover = movers & -movers; // TODO: reconsider this
			movers ^= mover;

			if (mover & pieces.kings) {
				generateKingMovesGpu(pieces, playerColor, mover, shift, moves);
			}
			else {
				generatePawnMovesGpu(pieces, playerColor, mover, shift, moves);
			}
		}
	}

	__device__ __host__ static void generateCapturingMovesInShiftGpu(const BitBoard& pieces, PieceColor playerColor, UINT jumpers, BitShift shift, Queue<Move2>* moves)
	{
		while (jumpers) {
			UINT jumper = jumpers & -jumpers;
			jumpers ^= jumper;

			if (jumper & pieces.kings) {
				generateKingCapturesGpu(pieces, playerColor, jumper, shift, moves);
			}
			else {
				generatePawnCapturesGpu(pieces, playerColor, jumper, shift, moves);
			}
		}
	}

	__device__ __host__ static BitShift getNextShift(BitShift shift, int iteration, UINT position)
	{
		BitShift nextShift = BitShift::BIT_SHIFT_NONE;

		if (shift == BitShift::BIT_SHIFT_L3 || shift == BitShift::BIT_SHIFT_L5)
		{
			if (iteration & 1)
				nextShift = BitShift::BIT_SHIFT_L4;
			else
				nextShift = shift;
		}

		if (shift == BitShift::BIT_SHIFT_R3 || shift == BitShift::BIT_SHIFT_R5)
		{
			if (iteration & 1)
				nextShift = BitShift::BIT_SHIFT_R4;
			else
				nextShift = shift;
		}

		if (shift == BitShift::BIT_SHIFT_L4)
		{
			if (iteration & 1)
			{
				if (position & MASK_L3)
					nextShift = BitShift::BIT_SHIFT_L3;
				else if (position & MASK_L5)
					nextShift = BitShift::BIT_SHIFT_L5;
				else
					return BitShift::BIT_SHIFT_NONE;
			}
			else
				nextShift = shift;
		}

		if (shift == BitShift::BIT_SHIFT_R4)
		{
			if (iteration & 1)
			{
				if (position & MASK_R3)
					nextShift = BitShift::BIT_SHIFT_R3;
				else if (position & MASK_R5)
					nextShift = BitShift::BIT_SHIFT_R5;
				else
					return BitShift::BIT_SHIFT_NONE;
			}
			else
				nextShift = shift;
		}

		return nextShift;
	}

    //---------------------------------------------------------------
    // Getting specified moveable pieces
	//---------------------------------------------------------------

	__device__ __host__ static UINT getMoversInShift(const BitBoard& pieces, PieceColor playerColor, BitShift shift)
	{
		const UINT emptyFields = pieces.getEmptyFields();
		const UINT pawns = pieces.getPieces(playerColor);
		const UINT kings = pawns & pieces.kings;

		UINT movers = 0;
		BitShift reverseShift = ShiftMap::getOpposite(shift);

		if (playerColor == PieceColor::White)
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

	__device__ __host__ static UINT getJumpersInShift(const BitBoard& pieces, PieceColor playerColor, BitShift shift, UINT captured = 0)
	{
		const UINT emptyFields = pieces.getEmptyFields();
		const UINT currentPieces = pieces.getPieces(playerColor);
		const UINT kings = currentPieces & pieces.kings;
		const UINT enemyPieces = pieces.getPieces(getEnemyColor(playerColor));

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
					BitShift nextShift = getNextShift(shift, iteration, movedCurrentKing);
					if (nextShift == BitShift::BIT_SHIFT_NONE)
						break;

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

	//---------------------------------------------------------------
	// Getting moveable pieces
	//---------------------------------------------------------------

	__device__ __host__ static UINT getAllMovers(const BitBoard& pieces, PieceColor playerColor)
	{
		UINT movers = 0;

		for (int i = 0; i < static_cast<int>(BitShift::COUNT); ++i) {
			BitShift shift = static_cast<BitShift>(i);
			movers |= getMoversInShift(pieces, playerColor, shift);
		}

		return movers;
	}

	__device__ __host__ static UINT getAllJumpers(const BitBoard& pieces, PieceColor playerColor)
	{
		UINT jumpers = 0;

		for (int i = 0; i < static_cast<int>(BitShift::COUNT); ++i) {
			BitShift shift = static_cast<BitShift>(i);
			jumpers |= getJumpersInShift(pieces, playerColor, shift);
		}

		return jumpers;
	}

	//---------------------------------------------------------------
	// Generating king moves
	//---------------------------------------------------------------

	__device__ __host__ static void generateKingMovesGpu(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, Queue<Move2>* moves)
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

			Move2 move(position, newPosition, playerColor);
			moves->push(move);
			iteration++;
		}
	}

	__device__ __host__ static void generateKingCapturesGpu(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, Queue<Move2>* moves)
	{
		UINT emptyFields = pieces.getEmptyFields();
		UINT newPosition = position;
		UINT currentPieces = pieces.getPieces(playerColor);
		UINT enemyPieces = pieces.getPieces(getEnemyColor(playerColor));

#ifdef ASSERTS_ON
		assert(position & pieces.kings & currentPieces);
		assert((currentPieces & enemyPieces) == 0);
#endif

		int iteration = 0;
		UINT foundEnemyPiece = 0;

		// Initialize continuation array
		Move2 localMovesArray[QUEUE_SIZE];
		Queue<Move2> localMovesQueue(localMovesArray, QUEUE_SIZE);

		while (newPosition)
		{
			BitShift nextShift = getNextShift(shift, iteration, newPosition);
			if (nextShift == BitShift::BIT_SHIFT_NONE)
				break;

			// Make the shift
			newPosition = ShiftMap::shift(newPosition, nextShift);
			iteration++;

			// Shifted out of the board
			if (newPosition == 0)
				break;

			// There must not be any piece in the same color on the way if the captured pawn has not been found yet 
#ifdef ASSERTS_ON
			assert((newPosition & currentPieces) == 0 || foundEnemyPiece);
#endif
			if (newPosition & currentPieces)
				break;

			// If the newPosition contains empty field continue looking for enemy pieces
			if ((newPosition & emptyFields) > 0 && !foundEnemyPiece)
				continue;

			// If the newPosition contains empty field and we have already found enemy piece, add the move
			if ((newPosition & emptyFields) > 0 && foundEnemyPiece)
			{
				Move2 move(position, newPosition, playerColor, foundEnemyPiece);
				localMovesQueue.push(move);
				continue;
			}

			// Being here means the enemy piece is on the new position
#ifdef ASSERTS_ON
			assert((newPosition & enemyPieces) > 0);
#endif
			if ((newPosition & enemyPieces) == 0)
				break;

			// If we have already found enemy piece, we must not find another one
			if (foundEnemyPiece > 0)
				break;

			foundEnemyPiece = newPosition;
		}

		while (!localMovesQueue.empty())
		{
			Move2 move = localMovesQueue.front();
			localMovesQueue.pop();
			
			// Create new board state after capture
			BitBoard newBoardState = move.getBitboardAfterMove(pieces);
			bool foundContinuation = false;

			for (int i = 0; i < static_cast<int>(BitShift::COUNT); ++i) 
			{
				// Get all jumpers
				BitShift newShift = static_cast<BitShift>(i);
				UINT jumpers = getJumpersInShift(newBoardState, playerColor, newShift, move.captured);
				
				// Destination jumper
				UINT jumper = jumpers & move.dst;

				// Destination cannot move any further 
				if (!jumper)
					continue;

				foundContinuation = true;
				UINT newJumperPosition = jumper;
				UINT newEmptyFields = newBoardState.getEmptyFields();
				UINT newCurrentPieces = newBoardState.getPieces(playerColor);
				UINT newEnemyPieces = newBoardState.getPieces(getEnemyColor(playerColor));
				UINT newFoundEnemyPiece = 0;
				int newIteration = 0;

				while (newJumperPosition)
				{
					BitShift nextShift = getNextShift(newShift, newIteration, newJumperPosition);
					if (nextShift == BitShift::BIT_SHIFT_NONE)
						break;

					// Make the shift
					newJumperPosition = ShiftMap::shift(newJumperPosition, nextShift);

					// Shifted out of the board
					if (newJumperPosition == 0)
						break;

					// There must not be any piece in the same color on the way if the captured pawn has not been found yet
#ifdef ASSERTS_ON
					assert((newJumperPosition & newCurrentPieces) == 0 || newFoundEnemyPiece);
#endif
					if (newJumperPosition & newCurrentPieces)
						break;

					// If the newPosition contains empty field continue looking for enemy pieces
					if ((newJumperPosition & newEmptyFields) > 0 && !newFoundEnemyPiece)
						continue;

					// If the newPosition contains empty field and we have already found enemy piece, add the move
					if ((newJumperPosition & newEmptyFields) > 0 && newFoundEnemyPiece)
					{
						Move2 newMove(move.src, newJumperPosition, playerColor, move.captured | newFoundEnemyPiece);
						localMovesQueue.push(newMove);
						continue;
					}

					// Being here means the enemy piece is on the new position
#ifdef ASSERTS_ON
					assert((newJumperPosition & newEnemyPieces) > 0);
#endif // ASSERTS_ON
					if ((newJumperPosition & newEnemyPieces) == 0)
						break;

					if (newFoundEnemyPiece > 0)
						break;

					newFoundEnemyPiece = newJumperPosition;
				}
			}

			if (!foundContinuation)
				moves->push(move);
		}
	}

	//---------------------------------------------------------------
	// Generating pawn moves
	//---------------------------------------------------------------
	__device__ __host__ static void generatePawnMovesGpu(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, Queue<Move2>* moves)
	{
		UINT newPosition = ShiftMap::shift(position, shift);
		Move2 move(position, newPosition, playerColor);
		moves->push(move);
	};

	__device__ __host__ static void generatePawnCapturesGpu(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, Queue<Move2>* moves)
	{
		UINT emptyFields = pieces.getEmptyFields();
		UINT captured = ShiftMap::shift(position, shift);
		UINT currentPieces = pieces.getPieces(playerColor);
		UINT enemyPieces = pieces.getPieces(getEnemyColor(playerColor));
		UINT newPosition = 0;
			
#ifdef ASSERTS_ON
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
		assert((captured & enemyPieces) != 0);
#endif

		BitShift nextShift = getNextShift(shift, 1, captured);

		// There must be a next shift
		if (nextShift == BitShift::BIT_SHIFT_NONE)
			return;

		// Setting up pawn position after capture
		newPosition = ShiftMap::shift(captured, nextShift);

		// Create the move
		Move2 singleCapture = Move2(position, newPosition, playerColor, captured);

		// Initialize continuation array
		Move2 localMovesArray[QUEUE_SIZE];
		Queue<Move2> localMovesQueue(localMovesArray, QUEUE_SIZE);
		localMovesQueue.push(singleCapture);

		// Generate all possible continuations
		while (!localMovesQueue.empty())
		{
			// Get move
			Move2 captureMove = localMovesQueue.front();
			localMovesQueue.pop();

			// Create new board state after capture
			BitBoard newState = captureMove.getBitboardAfterMove(pieces, false);
			bool foundContinuation = false;

			// Generate continuations
			for (int i = 0; i < static_cast<int>(BitShift::COUNT); ++i)
			{
				BitShift newShift = static_cast<BitShift>(i);
				UINT jumpers = getJumpersInShift(newState, playerColor, newShift);

				// There must be a jumper which is the continuation of the previous move
				UINT jumper = jumpers & captureMove.dst;
				if (jumper == 0)
					continue;

				// Mark that there is a continuation of the move
				foundContinuation = true;

				// Get new moves attributes
				UINT newCaptured = ShiftMap::shift(jumper, newShift);
				BitShift newPosShift = getNextShift(newShift, 1, newCaptured);
				UINT newDst = ShiftMap::shift(newCaptured, newPosShift);

				// Create new move
				Move2 newMove = Move2(captureMove.src, newDst, playerColor, newCaptured | captureMove.captured);
				localMovesQueue.push(newMove);
			}

			if (!foundContinuation)
				moves->push(captureMove);
		}
	}




	//---------------------------------------------------------------
	// Deprecated
	//---------------------------------------------------------------
	static MoveList generateMoves(const BitBoard& pieces, PieceColor playerColor);

	static void generateBasicMovesInShift(const BitBoard& pieces, PieceColor playerColor, UINT movers, BitShift shift, MoveList& moves);
	static void generateCapturingMovesInShift(const BitBoard& pieces, PieceColor playerColor, UINT jumpers, BitShift shift, MoveList& moves);

	static void generateKingCaptures(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves, UINT captured = 0);
	static void generatePawnCapturesInShift(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves);
	static void generateKingMoves(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves);
	static void generatePawnMovesInShift(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves);
};
