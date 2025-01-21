#include <stdio.h>
#include "Board.h"
#include "MoveGenerator.h"
#include "CheckersTestSuite.h"
#include "PlayerCPU.h"
#include "PlayerGPU.cuh"
#include "Game.h"
#include <random>

void CompareSimulations(Board board, int times, PieceColor playerColor, PieceColor playerToMove)
{
	int resultcpu = 0;
	int resultgpu = 0;

    for (int i = 0; i < times; i++)
    {
		int winner = board.simulateGame(playerToMove);
        int result = 0;

        if (winner == BLACK_WIN && playerColor == PieceColor::Black)
            result = WIN;
        else if (winner == WHITE_WIN && playerColor == PieceColor::White)
            result = WIN;
        else if (winner == DRAW)
            result = DRAW;
        else
            result = LOOSE;

		resultcpu += result;
    }

    for (int i = 0; i < times; i++)
    {
        std::random_device rd; // Seed
        std::mt19937 gen(rd()); // Mersenne Twister engine
        std::uniform_int_distribution<> dist(-999999, 999999);

		int z = dist(gen);
		int w = dist(gen);
		int jsr = dist(gen);
		int jcong = dist(gen);

		int winner = simulateGameGpu(board.getWhitePawns(), board.getBlackPawns(), board.getKings(), playerToMove, z, w, jsr, jcong);

		int result = 0;
		if (winner == BLACK_WIN && playerColor == PieceColor::Black)
			result = WIN;
		else if (winner == WHITE_WIN && playerColor == PieceColor::White)
			result = WIN;
		else if (winner == DRAW)
			result = DRAW;
		else
			result = LOOSE;

		resultgpu += result;
    }

    printf("Result cpu: %d\n", resultcpu);
    printf("Result gpu: %d\n", resultgpu);
}

int main()
{
    UINT whitePieces = (1ULL << 24) | (1ULL << 26) | (1ULL << 27) | (1ULL << 20) | (1ULL << 16);
    UINT blackPieces = (1ULL << 17) | (1ULL << 18) | (1ULL << 12) | (1ULL << 9) | (1ULL << 7) | (1ULL << 1);
    UINT kings = 0;


    UINT whitePiecesAfter = (1ULL << 24) | (1ULL << 26) | (1ULL << 27) | (1ULL << 20) | (1ULL << 16);
    UINT blackPiecesAfter = (1ULL << 17) | (1ULL << 18) | (1ULL << 12) | (1ULL << 9) | (1ULL << 7) | (1ULL << 4);
	Board board(whitePiecesAfter, blackPiecesAfter, 0);

	// CompareSimulations(board, 1000, PieceColor::Black, PieceColor::Black);

    /*simulateGameGpu(whitePiecesAfter, blackPiecesAfter, 0, PieceColor::Black, 0, 0, 0, 0);
	int result = board.simulateGame(PieceColor::Black);
	printf("Result: %d\n", result);*/

	// CheckersTestSuite::runAll();

    return 0;
}