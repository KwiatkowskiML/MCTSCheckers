#pragma once
#include "Utils.h"

class Board
{
private:
	UINT _whitePieces;
	UINT _blackPieces;
	UINT _kings;

	

public:
	Board(UINT whitePieces, UINT blackPieces, UINT kings) : _whitePieces(whitePieces), _blackPieces(blackPieces), _kings(kings) {};
	void GetWhiteAvailableMoves(std::queue<Board>& availableMoves);
	void GetBlackAvailableMoves(std::queue<Board>& availableMoves);
	void PrintBoard();

	UINT GetBlackJumpers();
	UINT GetBlackMovers();

	UINT GetWhiteJumpers();
	UINT GetWhiteMovers();

	static void PrintBitboard(UINT bitboard);
};

