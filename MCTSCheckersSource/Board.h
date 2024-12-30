#pragma once
#include "Utils.h"

class Board
{
private:
	UINT _whitePawns;
	UINT _blackPawns;
	UINT _kings;

	void AddWhiteMove(std::queue<Board>& availableMoves, UINT src, UINT dst);
	void GenerateKingBasicMoves(std::queue<Board>& availableMoves, UINT position);
	void GenerateWhitePawnBasicMoves(std::queue<Board>& availableMoves, UINT position);
	void GenerateWhitePawnCapturingMoves(std::queue<Board>& availableMoves, UINT position);

public:
	Board(UINT whitePieces, UINT blackPieces, UINT kings) : _whitePawns(whitePieces), _blackPawns(blackPieces), _kings(kings) {};
	void GetWhiteAvailableMoves(std::queue<Board>& availableMoves);
	void GetBlackAvailableMoves(std::queue<Board>& availableMoves);
	void PrintBoard();

	UINT GetBlackJumpers();
	UINT GetBlackMovers();

	UINT GetWhiteJumpers();
	UINT GetWhiteMovers();

	static void PrintBitboard(UINT bitboard);
	static void PrintPossibleMoves(const std::queue<Board>& availableMoves);
};

