
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

    {
        /*{
            Player* whitePlayer = nullptr;
            Player* blackPlayer = nullptr;
            int setup = Game::GetGameSetup(whitePlayer, blackPlayer);

            Game game(whitePlayer, blackPlayer);

            switch (setup)
            {
            case 1:
                game.PlayGameAsWhite();
                break;
            default:
                game.PlayGame();
                break;
            }

            if (whitePlayer != nullptr)
                delete whitePlayer;

            if (blackPlayer != nullptr)
                delete blackPlayer;
        }*/

        {
            /*UINT whitePieces = (1ULL << 28) | (1ULL << 21) | (1ULL << 22) | (1ULL << 23) | (1ULL << 17) | (1ULL << 13);
            UINT blackPieces = (1ULL << 24) | (1ULL << 16) | (1ULL << 14) | (1ULL << 15) | (1ULL << 8) | (1ULL << 10) | (1ULL << 4) | (1ULL << 6);
            UINT kings = 0;

            UINT whitePieces2 = (1ULL << 24) | (1ULL << 22) | (1ULL << 19);
            UINT blackPieces2 = (1ULL << 17) | (1ULL << 11) | (1ULL << 4);

            UINT whitePieces3 = (1ULL << 28) | (1ULL << 31) | (1ULL << 24) | (1ULL << 25) | (1ULL << 26) | (1ULL << 27) | (1ULL << 22) | (1ULL << 16) | (1ULL << 8);
            UINT blackPieces3 = (1ULL << 17) | (1ULL << 15) | (1ULL << 9) | (1ULL << 10) | (1ULL << 5) | (1ULL << 3) | 1ULL;
            UINT kings3 = (1ULL << 8);

            UINT whitePieces4 = (1ULL << 28);
            UINT blackPieces4 = (1ULL << 25) | (1ULL << 21) | (1ULL << 20);

            UINT whitePieces5 = (1ULL << 10) | (1ULL << 11);
            UINT blackPieces5 = (1ULL << 13);
			UINT kings5 = (1ULL << 13);*/

           /*Board boardAfterMove(whitePieces, blackPieces, kings);
            std::cout << boardAfterMove.toString() << std::endl;

            Player* blackPlayer = new PlayerCPU(PieceColor::Black, DEFAULT_TIME_LIMIT);
            blackPlayer->SetBoard(boardAfterMove);
            Move* bestMove = blackPlayer->GetBestMove();
            std::cout << "Best move: " << bestMove->toString() << std::endl;

            blackPlayer->GenerateDotFile(TREE_VISUALIZATION_FILE);
            delete blackPlayer;*/
        }

        

        Board board2(whitePieces, blackPieces, kings);
        std::cout << board2.toString() << std::endl;

        // gpu
        Player* blackPlayerGPU = new PlayerGPU(PieceColor::Black, DEFAULT_TIME_LIMIT);
        blackPlayerGPU->SetBoard(board2);

        Move* bestMoveGpu = blackPlayerGPU->GetBestMove();
		std::cout << "Best move: " << bestMoveGpu->toString() << std::endl;
		std::cout << "Run simulations: " << blackPlayerGPU->root->gamesPlayed << std::endl;

        blackPlayerGPU->GenerateDotFile(TREE_VISUALIZATION_FILE_GPU);

        // cpu
        Player* blackPlayerCPU = new PlayerCPU(PieceColor::Black, DEFAULT_TIME_LIMIT);
        blackPlayerCPU->SetBoard(board2);

        Move* bestMoveCpu = blackPlayerCPU->GetBestMove();
		std::cout << "Best move: " << bestMoveCpu->toString() << std::endl;
        std::cout << "Run simulations: " << blackPlayerCPU->root->gamesPlayed << std::endl;

        blackPlayerCPU->GenerateDotFile(TREE_VISUALIZATION_FILE_CPU);

		delete blackPlayerGPU;
		delete blackPlayerCPU;

    }

	/*Player* whitePlayer = new PlayerGPU(PieceColor::White, DEFAULT_TIME_LIMIT);
    whitePlayer->Simulate(whitePlayer->root);*/

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