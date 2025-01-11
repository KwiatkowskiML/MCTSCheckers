#pragma once
#include "Move.h"
#include "PieceColor.h"
#include "BitBoard.h"
#include "BitShift.h"

class MoveGenerator {
public:
    static MoveList generateMoves(const BitBoard& pieces, PieceColor color);

	// Getting moveable pieces
    static UINT getJumpers(const BitBoard& pieces, PieceColor color);
    static UINT getMovers(const BitBoard& pieces, PieceColor color);

    // Getting specified moveable pieces
	static UINT getMoversInShift(const BitBoard& pieces, PieceColor color, BitShift shift);
	static UINT getJumpersInShift(const BitBoard& pieces, PieceColor color, BitShift shift);

    // Generating a list of moves
    static void generateBasicMoves(const BitBoard& pieces, PieceColor color, UINT movers, MoveList& moves);
    static void generateBasicMovesInShift(const BitBoard& pieces, PieceColor color, UINT movers, BitShift shift, MoveList& moves);
    static void generateCapturingMoves(const BitBoard& pieces, PieceColor color, UINT jumpers, MoveList& moves);
    static void generateCapturingMovesInShift(const BitBoard& pieces, PieceColor color, UINT jumpers, BitShift shift, MoveList& moves);
    
    // Generating specified move
    static void generateKingCaptures(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves);
    static void generatePawnCaptures(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves);
    static void generatePawnCapturesInShift(const BitBoard& pieces, PieceColor color, UINT position, BitShift shift, MoveList& moves);
    static void generateKingMoves(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves);
    static void generatePawnMoves(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves);
    static void generatePawnMovesInShift(const BitBoard& pieces, PieceColor color, UINT position, BitShift shift, MoveList& moves);
};
