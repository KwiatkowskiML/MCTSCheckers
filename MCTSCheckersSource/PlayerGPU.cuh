#pragma once
#include "Player.h"

class PlayerGPU : public Player
{
public:
	PlayerGPU(PieceColor color, int timeLimit) : Player(color, timeLimit) {}

	// Simulation on GPU
	int Simulate(Node* node) override {
        cudaError_t cudaStatus;

        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

    Error:
        return -1;
	};
};