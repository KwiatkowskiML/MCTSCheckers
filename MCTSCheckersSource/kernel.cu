
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "Board.h"
#include "MoveGenerator.h"
#include "CheckersTestSuite.h"
#include "PlayerCPU.h"
#include "PlayerGPU.cuh"
#include "Game.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

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
			UINT kings5 = (1ULL << 13);

            Board boardAfterMove(whitePieces, blackPieces, kings);
            std::cout << boardAfterMove.toString() << std::endl;

            Player* blackPlayer = new PlayerCPU(PieceColor::Black, DEFAULT_TIME_LIMIT);
            blackPlayer->SetBoard(boardAfterMove);
            Move* bestMove = blackPlayer->GetBestMove();
            std::cout << "Best move: " << bestMove->toString() << std::endl;

            blackPlayer->GenerateDotFile(TREE_VISUALIZATION_FILE);
            delete blackPlayer;*/
        }

        {
            /*Board board2(INIT_WHITE_PAWNS, INIT_BLACK_PAWNS, 0);
            std::cout << board2.toString() << std::endl;
            Player* whitePlayer = new PlayerCPU(PieceColor::White, DEFAULT_TIME_LIMIT);
            whitePlayer->SetBoard(board2);
            Move* bestMove2 = whitePlayer->GetBestMove();
            std::cout << "Best move: " << bestMove2->toString() << std::endl;
            whitePlayer->GenerateDotFile(TREE_VISUALIZATION_FILE);*/
        }
    }

	Player* whitePlayer = new PlayerGPU(PieceColor::White, DEFAULT_TIME_LIMIT);
    whitePlayer->Simulate(whitePlayer->root);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
