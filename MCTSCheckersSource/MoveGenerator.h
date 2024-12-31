#pragma once
#include "Direction.h"
#include "Move.h"
#include "PieceColor.h"
#include "BitBoard.h"

class MoveGenerator {
public:
    static MoveList generateMoves(const BitBoard& pieces, PieceColor color);
//private:
    static UINT getJumpers(const BitBoard& pieces, PieceColor color);
    static UINT getMovers(const BitBoard& pieces, PieceColor color);

    //static MoveList generateBasicMoves(const BitBoard& pieces, UINT position, bool isKing);
    //static MoveList generateCapturingMoves(const BitBoard& pieces, UINT position, bool isKing);
    //
    //static void generateKingCaptures(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves);
    //static void generatePawnCaptures(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves);
    //static void generateKingMoves(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves);
    //static void generatePawnMoves(const BitBoard& pieces, PieceColor color, UINT position, MoveList& moves);
};
