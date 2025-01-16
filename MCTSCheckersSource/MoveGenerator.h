#pragma once
#include "Move.h"
#include "PieceColor.h"
#include "BitBoard.h"
#include "BitShift.h"

class MoveGenerator {
private:
	
public:
    static MoveList generateMoves(const BitBoard& pieces, PieceColor playerColor);
    static BitShift getNextShift(BitShift shift, int iteration, UINT position);

	// Getting moveable pieces
	static UINT getAllMovers(const BitBoard& pieces, PieceColor playerColor);
	static UINT getAllJumpers(const BitBoard& pieces, PieceColor playerColor);

    // Getting specified moveable pieces
	__device__ __host__ static UINT getMoversInShift(const BitBoard& pieces, PieceColor playerColor, BitShift shift);
	static UINT getJumpersInShift(const BitBoard& pieces, PieceColor playerColor, BitShift shift, UINT captured = 0);

    // Generating a list of moves
    static void generateBasicMovesInShift(const BitBoard& pieces, PieceColor playerColor, UINT movers, BitShift shift, MoveList& moves);
    static void generateCapturingMovesInShift(const BitBoard& pieces, PieceColor playerColor, UINT jumpers, BitShift shift, MoveList& moves);
    
    // Generating specified move
    static void generateKingCaptures(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves, UINT captured = 0);
    static void generatePawnCapturesInShift(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves);
    static void generateKingMoves(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves);
    static void generatePawnMovesInShift(const BitBoard& pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves);
};
