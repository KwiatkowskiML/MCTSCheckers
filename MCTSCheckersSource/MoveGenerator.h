#pragma once
#include "Move.h"
#include "PieceColor.h"
#include "BitBoard.h"
#include "BitShift.h"

class MoveGenerator {
private:
	
public:
    static MoveList generateMoves(BitBoard pieces, PieceColor playerColor);
    static BitShift getNextShift(BitShift shift, int iteration, UINT position);
    static void doNothing() {};

	// Getting moveable pieces
	static UINT getAllMovers(BitBoard pieces, PieceColor playerColor);
	static UINT getAllJumpers(BitBoard pieces, PieceColor playerColor);

    // Getting specified moveable pieces
	static UINT getMoversInShift(BitBoard pieces, PieceColor playerColor, BitShift shift);
	static UINT getJumpersInShift(BitBoard pieces, PieceColor playerColor, BitShift shift, UINT captured = 0);

    // Generating a list of moves
    static void generateBasicMovesInShift(BitBoard pieces, PieceColor playerColor, UINT movers, BitShift shift, MoveList& moves);
    static void generateCapturingMovesInShift(BitBoard pieces, PieceColor playerColor, UINT jumpers, BitShift shift, MoveList& moves);
    
    // Generating specified move
    static void generateKingCaptures(BitBoard pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves, UINT captured = 0);
    static void generatePawnCapturesInShift(BitBoard pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves);
    static void generateKingMoves(BitBoard pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves);
    static void generatePawnMovesInShift(BitBoard pieces, PieceColor playerColor, UINT position, BitShift shift, MoveList& moves);
};
