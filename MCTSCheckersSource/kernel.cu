
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "Board.h"
#include "MoveGenerator.h"
#include "CheckersTestSuite.h"
#include "PlayerCPU.h"
#include "PlayerGPU.cuh"
#include "Game.h"

int main()
{
    {
        /*PlayerCPU* player = new PlayerCPU(PieceColor::White, DEFAULT_TIME_LIMIT);
        Move* bestMove = player->GetBestMove();
        player->GenerateDotFile(TREE_VISUALIZATION_FILE);

        std::cout << "Best move: " << bestMove->toString() << std::endl;

        delete player;*/

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
            UINT whitePieces = (1ULL << 28) | (1ULL << 21) | (1ULL << 22) | (1ULL << 23) | (1ULL << 17) | (1ULL << 13);
            UINT blackPieces = (1ULL << 24) | (1ULL << 16) | (1ULL << 14) | (1ULL << 15) | (1ULL << 8) | (1ULL << 10) | (1ULL << 4) | (1ULL << 6);
            UINT kings = 0;

  /*          UINT whitePieces2 = (1ULL << 24) | (1ULL << 22) | (1ULL << 19);
            UINT blackPieces2 = (1ULL << 17) | (1ULL << 11) | (1ULL << 4);

            UINT whitePieces3 = (1ULL << 28) | (1ULL << 31) | (1ULL << 24) | (1ULL << 25) | (1ULL << 26) | (1ULL << 27) | (1ULL << 22) | (1ULL << 16) | (1ULL << 8);
            UINT blackPieces3 = (1ULL << 17) | (1ULL << 15) | (1ULL << 9) | (1ULL << 10) | (1ULL << 5) | (1ULL << 3) | 1ULL;
            UINT kings3 = (1ULL << 8);

            UINT whitePieces4 = (1ULL << 28);
            UINT blackPieces4 = (1ULL << 25) | (1ULL << 21) | (1ULL << 20);

            UINT whitePieces5 = (1ULL << 10) | (1ULL << 11);
            UINT blackPieces5 = (1ULL << 13);
			UINT kings5 = (1ULL << 13);*/

           /* Board boardAfterMove(whitePieces, blackPieces, kings);
            std::cout << boardAfterMove.toString() << std::endl;

            Player* blackPlayer = new PlayerCPU(PieceColor::Black, DEFAULT_TIME_LIMIT);
            blackPlayer->SetBoard(boardAfterMove);
            Move* bestMove = blackPlayer->GetBestMove();
            std::cout << "Best move: " << bestMove->toString() << std::endl;

            blackPlayer->GenerateDotFile(TREE_VISUALIZATION_FILE);
            delete blackPlayer;*/
        }

        /*Board board2(INIT_WHITE_PAWNS, INIT_BLACK_PAWNS, 0);
        std::cout << board2.toString() << std::endl;
        Player* whitePlayer = new PlayerCPU(PieceColor::White, DEFAULT_TIME_LIMIT);
        whitePlayer->SetBoard(board2);
        Move* bestMove2 = whitePlayer->GetBestMove();
        std::cout << "Best move: " << bestMove2->toString() << std::endl;
        whitePlayer->GenerateDotFile(TREE_VISUALIZATION_FILE);*/
    }

	Player* whitePlayer = new PlayerGPU(PieceColor::White, DEFAULT_TIME_LIMIT);
    whitePlayer->Simulate(whitePlayer->root);

	// simulateGameGpu(INIT_WHITE_PAWNS, INIT_BLACK_PAWNS, 0, PieceColor::White);


	// CheckersTestSuite::runAll();

    return 0;
}