#include <stdio.h>
#include <random>

#include "Board.h"
#include "MoveGenerator.h"
#include "CheckersTestSuite.h"
#include "PlayerCPU.h"
#include "PlayerGPU.cuh"
#include "Game.h"

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
	Game game;
	game.PlayGame();

	// CheckersTestSuite::runAll();

	//UINT white = (1ull << 22) | (1ull << 23) | (1ull << 15);
	//UINT black = (1ull << 2) | (1ull << 4) | (1ull << 6) | (1ull << 14) | (1ull << 17);

	//UINT white = (1ull << 23) | (1ull << 7);
	//UINT black = (1ull << 2) | (1ull << 4) | (1ull << 14) | (1ull << 26) | (1ull << 17);

	//BitBoard bitboard(white, black, 0);
	//Board board(white, black, 0);
	//std::cout << board.toString() << std::endl;

	//Player* blackPlayerCpu = new PlayerCPU(PieceColor::Black, DEFAULT_TIME_LIMIT);
	//Player* blackPlayerGpu = new PlayerGPU(PieceColor::Black, DEFAULT_TIME_LIMIT);

	//blackPlayerCpu->SetBoard(board);
	//blackPlayerGpu->SetBoard(board);

	//Move* bestMoveCpu = blackPlayerCpu->GetBestMove();
	//Move* bestMoveGpu = blackPlayerGpu->GetBestMove();

	//std::cout << "Best move CPU: " << bestMoveCpu->toString() << std::endl;
	//std::cout << "Best move GPU: " << bestMoveGpu->toString() << std::endl;

	//blackPlayerCpu->GenerateDotFile(TREE_VISUALIZATION_FILE_CPU);
	//blackPlayerGpu->GenerateDotFile(TREE_VISUALIZATION_FILE_GPU);

	//delete blackPlayerCpu;
	//delete blackPlayerGpu;

	// CompareSimulations(board, 1000, PieceColor::Black, PieceColor::Black);

    return 0;
}