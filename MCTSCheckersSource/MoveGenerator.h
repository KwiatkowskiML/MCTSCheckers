#pragma once
#include "Move.h"
#include "PieceColor.h"
#include "BitBoard.h"
#include "BitShift.h"
#include "ShiftMap.h"
#include "Queue.h"
#include "Move2.h"

class MoveGenerator {
private:
	
public:
    __device__ __host__ static void DoNothing() {};


    static MoveList generateMoves(const BitBoard& pieces, PieceColor playerColor);
	
	__device__ __host__ static BitShift getNextShift(BitShift shift, int iteration, UINT position)
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
				if (position & MASK_L3)
					nextShift = BitShift::BIT_SHIFT_L3;
				else if (position & MASK_L5)
					nextShift = BitShift::BIT_SHIFT_L5;
				else
					return BitShift::BIT_SHIFT_NONE;
			}
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
    // Generating a list of moves
	//---------------------------------------------------------------
    static void generateBasicMovesInShift(const BitBoard& pieces, PieceColor playerColor, UINT movers, BitShift shift, MoveList& moves);
    static void generateCapturingMovesInShift(const BitBoard& pieces, PieceColor playerColor, UINT jumpers, BitShift shift, MoveList& moves);

	__device__ __host__ void generateBasicMovesInShiftGpu(const BitBoard& pieces, PieceColor playerColor, UINT movers, BitShift shift, Queue<Move2>* moves)
	{
		while (movers) {
			UINT mover = movers & -movers; // TODO: reconsider this
			movers ^= mover;

			if (mover & pieces.kings) {
				// generateKingMoves(pieces, playerColor, mover, shift, moves);
			}
			else {
				generatePawnMovesGpu(pieces, playerColor, mover, shift, moves);
			}
		}
	}

	__device__ __host__ void generateCapturingMovesInShiftGpu(const BitBoard& pieces, PieceColor playerColor, UINT jumpers, BitShift shift, Queue<Move2>* moves)
	{
		while (jumpers) {
			UINT jumper = jumpers & -jumpers;
			jumpers ^= jumper;

			if (jumper & pieces.kings) {
				// generateKingCaptures(pieces, playerColor, jumper, shift, moves);
			}
			else {
				generatePawnCapturesGpu(pieces, playerColor, jumper, shift, moves);
			}
		}
	}
    
	//---------------------------------------------------------------
    // Generating specified move
	//---------------------------------------------------------------
    static void generateKingCaptures(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves, UINT captured = 0);
    static void generatePawnCapturesInShift(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves);
    static void generateKingMoves(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves);
    static void generatePawnMovesInShift(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves);

	//---------------------------------------------------------------
	// Generating specified move with queue
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

		// Assertions
		{
			// Make sure the position is right color
			// assert(position & currentPieces);

			// Make sure the captured piece is enemy piece
			// assert(captured & enemyPieces);

			// Make sure the captured piece is not the same as the position
			// assert((captured & currentPieces) == 0);

			// Make sure the captured piece is not empty field
			// assert((captured & emptyFields) == 0);

			// Make sure the pieces are marked correctly
			// assert((enemyPieces & currentPieces) == 0);

			// There must be a captured piece
			/*if ((captured & enemyPieces) == 0)
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
			}*/
			// assert((captured & enemyPieces) != 0);
		}

		BitShift nextShift = getNextShift(shift, 1, position);

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
				BitShift nextShift = static_cast<BitShift>(i);
				UINT jumpers = getJumpersInShift(newState, playerColor, nextShift);

				// There must be a jumper which is the continuation of the previous move
				UINT jumper = jumpers & singleCapture.dst;
				if (jumper == 0)
					continue;

				// Mark that there is a continuation of the move
				foundContinuation = true;

				// Get new moves attributes
				UINT newCaptured = ShiftMap::shift(jumper, nextShift);
				BitShift newPosShift = getNextShift(nextShift, 1, jumper);
				UINT newDst = ShiftMap::shift(newCaptured, newPosShift);

				// Create new move
				Move2 newMove = Move2(captureMove.src, newDst, playerColor, newCaptured | captureMove.captured);
				localMovesQueue.push(newMove);
			}

			if (!foundContinuation)
				moves->push(singleCapture);
		}
	}
};
