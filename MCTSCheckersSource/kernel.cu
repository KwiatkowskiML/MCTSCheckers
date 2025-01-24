#include <stdio.h>
#include <random>

#include "includes/Board.h"
#include "includes/MoveGenerator.h"
#include "includes/CheckersTestSuite.h"
#include "includes/PlayerCPU.h"
#include "includes/PlayerGPU.h"
#include "includes/PlayerReal.h"
#include "includes/Game.h"

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

void CompareGetBestMove(UINT white, UINT black, UINT kings)
{
	BitBoard bitboard(white, black, kings);
	Board board(white, black, kings);
	std::cout << board.toString() << std::endl;

	Player* blackPlayerCpu = new PlayerCPU(PieceColor::Black, DEFAULT_TIME_LIMIT);
	Player* blackPlayerGpu = new PlayerGPU(PieceColor::Black, DEFAULT_TIME_LIMIT);

	blackPlayerCpu->SetBoard(board);
	blackPlayerGpu->SetBoard(board);

	Move* bestMoveCpu = blackPlayerCpu->GetBestMove();
	Move* bestMoveGpu = blackPlayerGpu->GetBestMove();

	std::cout << "Best move CPU: " << bestMoveCpu->toString() << std::endl;
	std::cout << "Games Played: " << blackPlayerCpu->root->gamesPlayed << std::endl;
	
	std::cout << "Best move GPU: " << bestMoveGpu->toString() << std::endl;
	std::cout << "Games Played: " << blackPlayerGpu->root->gamesPlayed << std::endl;

	blackPlayerCpu->GenerateDotFile(TREE_VISUALIZATION_FILE_CPU);
	blackPlayerGpu->GenerateDotFile(TREE_VISUALIZATION_FILE_GPU);

	delete blackPlayerCpu;
	delete blackPlayerGpu;
}

void PlayGameTest()
{
	UINT white = 1ull << 28;
	UINT black = 0; // 1ull << 3;
	UINT kings = 0; // 1ull << 28 | 1ull << 3;

	Player* whitePlayer = new PlayerGPU(PieceColor::White, DEFAULT_TIME_LIMIT);
	Player* blackPlayer = new PlayerReal(PieceColor::Black, DEFAULT_TIME_LIMIT);

	whitePlayer->SetBoard(Board(white, black, kings));

	Game game(whitePlayer, blackPlayer);
	game.PlayGame();

	if (!whitePlayer)
		delete whitePlayer;
	if (!blackPlayer)
		delete blackPlayer;
}

int main()
{
	//cudaDeviceProp prop; 
	//cudaGetDeviceProperties(&prop, 0);  // 0 = first GPU 

	//int num_sms = prop.multiProcessorCount;
	//int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;

	//printf("Number of SMs: %d\n", num_sms);
	//printf("Max threads per SM: %d\n", max_threads_per_sm);

	Game game;
	game.PlayGame();

	// CheckersTestSuite::runAll();
	
	// PlayGameTest();

	// CompareGetBestMove(INIT_WHITE_PAWNS, INIT_BLACK_PAWNS, 0);

	return 0;
}