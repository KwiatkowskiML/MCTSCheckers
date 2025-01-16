#pragma once
#include "Player.h"
#include "BitBoard.h"
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "MoveGenerationGpu.cuh"
#include "ShiftGpu.cuh"

__host__ __device__ UINT simulateGame(UINT white, UINT black, UINT kings, bool whiteToPlay)
{
	BitBoard bitboard;
	bitboard.blackPawns = black;
	bitboard.whitePawns = white;
	bitboard.kings = kings;

	PieceColor playerColor = whiteToPlay ? PieceColor::White : PieceColor::Black;
	PieceColor opponentColor = getEnemyColorGpu(playerColor);
    BitShift shift = BitShift::BIT_SHIFT_L4;
   
	UINT movers = getMoversInShift(bitboard, playerColor, shift);
	UINT allmovers = getAllMovers(bitboard, playerColor);
	UINT jumpers = getJumpersInShift(bitboard, playerColor, shift, 0);

	return 0;
}

__global__ void simulateKernel(UINT white, UINT black, UINT kings, bool whiteToPlay, char* dev_results)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    dev_results[idx] = -1; // temp

	simulateGame(white, black, kings, whiteToPlay);

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
        bool whiteToPlay = node->moveColor == PieceColor::Black; // if black moved, white is to play
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
		simulateKernel <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (white, black, kings, whiteToPlay, dev_results);

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