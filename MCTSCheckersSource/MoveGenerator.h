#pragma once
#include "Direction.h"
#include "Move.h"
#include "PieceColor.h"
#include "BitBoard.h"

class MoveGenerator {
public:
    static MoveList generateMoves(const BitBoard& pieces, PieceColor color);

	// Getting moveable pieces
    static UINT getJumpers(const BitBoard& pieces, PieceColor color);
    static UINT getMovers(const BitBoard& pieces, PieceColor color);

    // Generating a list of moves
    static void generateBasicMoves(const BitBoard& pieces, PieceColor color, UINT movers, MoveList& moves);
    static void generateCapturingMoves(const BitBoard& pieces, PieceColor color, UINT jumpers, MoveList& moves);
    
    // Generating specified move
    static void generateKingCaptures(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves);
    static void generatePawnCaptures(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves);
    static void generateKingMoves(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves);
    static void generatePawnMoves(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves);
};
