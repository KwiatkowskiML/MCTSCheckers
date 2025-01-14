#pragma once
#include "Player.h"

class PlayerCPU : public Player
{
public:
	PlayerCPU(PieceColor color, int timeLimit) : Player(color, timeLimit) {};
	int Simulate(Node* node) override;
};

