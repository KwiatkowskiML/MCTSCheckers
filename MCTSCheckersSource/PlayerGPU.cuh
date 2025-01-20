#pragma once
#include "Player.h"
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <curand.h>
#include <curand_kernel.h>
#include "ShiftMap.h"
#include "MoveGenerator.h"
#include "Queue.h"

// http://www.cse.yorku.ca/~oz/marsaglia-rng.html
#define znew (z=36969*(z&65535)+(z>>16))
#define wnew (w=18000*(w&65535)+(w>>16))
#define MWC ((znew<<16)+wnew )
#define SHR3 (jsr^=(jsr<<17), jsr^=(jsr>>13), jsr^=(jsr<<5))
#define CONG (jcong=69069*jcong+1234567)
#define FIB ((b=a+b),(a=b-a))
#define KISS ((MWC^CONG)+SHR3)

#define RANDOM_PER_THREAD 4

__host__ __device__ int simulateGameGpu(UINT white, UINT black, UINT kings, PieceColor playercolor, uint32_t z, uint32_t w, uint32_t jsr, uint32_t jcong)
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

		// Random number generation
		int randomIndex = KISS % movesQueue.length();
		Move2 randomMove = movesQueue[randomIndex];

		// Check if the move is a capture
		if (!randomMove.captured && (randomMove.src & currentBitBoard.kings) > 0) {
			noCaptureMoves++;
		}
		else {
			noCaptureMoves = 0;
		}

		currentBitBoard = randomMove.getBitboardAfterMove(currentBitBoard);
		currentMoveColor = getEnemyColor(currentMoveColor);
	}
}

__global__ void simulateKernel(UINT white, UINT black, UINT kings, PieceColor playerColor, char* dev_results, uint32_t* dev_random)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	uint32_t z = dev_random[idx * RANDOM_PER_THREAD];
	uint32_t w = dev_random[idx * RANDOM_PER_THREAD + 1];
	uint32_t jsr = dev_random[idx * RANDOM_PER_THREAD + 2];
	uint32_t jcong = dev_random[idx * RANDOM_PER_THREAD + 3];

	int simulationResult = simulateGameGpu(white, black, kings, playerColor, z, w, jsr, jcong);

	if (simulationResult == DRAW) {
		dev_results[idx] = DRAW;
	}
	else if (simulationResult == WHITE_WIN) {
		dev_results[idx] = playerColor == PieceColor::White ? WIN : LOOSE;
	}
	else {
		dev_results[idx] = playerColor == PieceColor::Black ? WIN : LOOSE;
	}
}

class PlayerGPU : public Player
{
public:
	PlayerGPU(PieceColor color, int timeLimit) : Player(color, timeLimit) {}

	// Simulation on GPU
	std::pair<int, int> Simulate(Node* node) override {
		cudaError_t cudaStatus;

		// Random number generator initialization
		curandGenerator_t gen;

		// Kernel parameters setup
		UINT white = node->boardAfterMove.getWhitePawns();
		UINT black = node->boardAfterMove.getBlackPawns();
		UINT kings = node->boardAfterMove.getKings();
		int simulationToRun = NUMBER_OF_BLOCKS * THREADS_PER_BLOCK;

		// Results array
		char* dev_results = 0;
		uint32_t* dev_random = 0;
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

		// Allocate random numbers array
		cudaStatus = cudaMalloc((void**)&dev_random, simulationToRun * RANDOM_PER_THREAD * sizeof(uint32_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		// Random number generator setup
		curandStatus_t cuRandStatus;
		cuRandStatus = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
		if (cuRandStatus != CURAND_STATUS_SUCCESS) {
			fprintf(stderr, "curandCreateGenerator failed!");
			goto Error;
		}

		// Setting up seed
		cuRandStatus = curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
		if (cuRandStatus != CURAND_STATUS_SUCCESS) {
			fprintf(stderr, "curandSetPseudoRandomGeneratorSeed failed!");
			goto Error;
		}

		// Setting up seed
		cuRandStatus = curandGenerate(gen, dev_random, simulationToRun * RANDOM_PER_THREAD);
		if (cuRandStatus != CURAND_STATUS_SUCCESS) {
			fprintf(stderr, "curandGenerate failed!");
			goto Error;
		}

		// Launch kernel
		simulateKernel << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>> > (white, black, kings, node->moveColor, dev_results, dev_random);

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

	Error:
		cudaFree(dev_results);
		cudaFree(dev_random);
		curandDestroyGenerator(gen);

		return std::make_pair(result, simulationToRun);
	};
};