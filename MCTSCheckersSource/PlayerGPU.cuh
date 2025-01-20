#pragma once
#include "Player.h"
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include "ShiftMap.h"
#include "MoveGenerator.h"
#include "Queue.h"

__host__ __device__ int simulateGameGpu(UINT white, UINT black, UINT kings, PieceColor playercolor)
{
    PieceColor currentMoveColor = playercolor;
    BitBoard currentBitBoard = BitBoard(white, black, kings);

    // Keeping track of the moves with no captures
    int noCaptureMoves = 0;

    // Moves Queue initialization
	Move2 movesArray[QUEUE_SIZE];
	Queue<Move2> movesQueue(movesArray, QUEUE_SIZE);

    while (true)
    {
		// Clear the moves queue
		movesQueue.clear();

		// Generate all possible moves
		MoveGenerator::generateMovesGpu(currentBitBoard, currentMoveColor, &movesQueue);

        // No moves available - game is over
        if (movesQueue.empty()) {
            return currentMoveColor == PieceColor::White ? BLACK_WIN : WHITE_WIN;
        }

        // Check if the no capture moves limit has beeen exceeded
        if (noCaptureMoves >= MAX_NO_CAPTURE_MOVES) {
            return DRAW;
        }

        //// Random number generation
        //std::random_device rd; // Seed
        //std::mt19937 gen(rd()); // Mersenne Twister engine
        //std::uniform_int_distribution<> dist(0, moves.size() - 1);

        //// Select a random move
        //int randomIndex = dist(gen);
        //Move randomMove = moves[randomIndex];

		Move2 randomMove = movesQueue.front();
		randomMove.printMove();

        // Check if the move is a capture
        if (!randomMove.captured && (randomMove.src & currentBitBoard.kings) > 0) {
            noCaptureMoves++;
        }
        else {
            noCaptureMoves = 0;
        }

        currentBitBoard = randomMove.getBitboardAfterMove(currentBitBoard);
        currentBitBoard.print();
        currentMoveColor = getEnemyColor(currentMoveColor);     
    }
}

__global__ void simulateKernel(UINT white, UINT black, UINT kings, PieceColor playerColor, char* dev_results)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    dev_results[idx] = simulateGameGpu(white, black, kings, playerColor);

	printf("Hello from block %d, thread %d, idx = %d, results = %d\n", blockIdx.x, threadIdx.x, idx, dev_results[idx]);
}

class PlayerGPU : public Player
{
public:
	PlayerGPU(PieceColor color, int timeLimit) : Player(color, timeLimit) {}

	// Simulation on GPU
	std::pair<int,int> Simulate(Node* node) override {
        cudaError_t cudaStatus;

        // Kernel parameters setup
        UINT white = node->boardAfterMove.getWhitePawns();
        UINT black = node->boardAfterMove.getBlackPawns();
        UINT kings = node->boardAfterMove.getKings();
		int simulationToRun = NUMBER_OF_BLOCKS * THREADS_PER_BLOCK;
        
        // Results array
		char* dev_results = 0;
		int result = 0;

        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

        // Allocate results array
        cudaStatus = cudaMalloc((void**)&dev_results, simulationToRun * sizeof(char));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }
		
        // Launch kernel
		simulateKernel <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (white, black, kings, node->moveColor, dev_results);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "simulateKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching simulateKernel!\n", cudaStatus);
            goto Error;
        }

        // Sum up simulation results
        result = thrust::reduce(thrust::device, dev_results, dev_results + simulationToRun, 0, thrust::plus<int>());
        printf("Result = %d\n", result);

        // TODO: check if the results should be negated
    Error:
		cudaFree(dev_results);
		return std::make_pair(result, simulationToRun);
	};
};