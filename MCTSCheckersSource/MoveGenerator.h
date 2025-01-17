#pragma once
#include "Move.h"
#include "PieceColor.h"
#include "BitBoard.h"
#include "BitShift.h"
#include "ShiftMap.h"
#include "Queue.h"

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
	static void generatePawnMovesInShift(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, Queue<Move>* moves)
	{
	};


};
