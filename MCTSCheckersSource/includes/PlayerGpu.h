#pragma once
#include "Player.h"
#include "MoveGpu.h"
#include "Queue.h"
#include "MoveGenerator.h"

// http://www.cse.yorku.ca/~oz/marsaglia-rng.html
#define znew (z=36969*(z&65535)+(z>>16))
#define wnew (w=18000*(w&65535)+(w>>16))
#define MWC ((znew<<16)+wnew )
#define SHR3 (jsr^=(jsr<<17), jsr^=(jsr>>13), jsr^=(jsr<<5))
#define CONG (jcong=69069*jcong+1234567)
#define FIB ((b=a+b),(a=b-a))
#define KISS ((MWC^CONG)+SHR3)

#define RANDOM_PER_THREAD 4

__host__ __device__ int simulateGameGpu(UINT white, UINT black, UINT kings, PieceColor colorToMove, uint32_t z, uint32_t w, uint32_t jsr, uint32_t jcong);

class PlayerGPU : public Player
{
public:
	PlayerGPU(PieceColor color, int timeLimit) : Player(color, timeLimit) {}

	// Simulation on GPU
	std::pair<int, int> Simulate(Node* node) override;
};