#pragma once
#include "Player.h"

class PlayerCPU : public Player
{
public:
	PlayerCPU(PieceColor playerColor, int timeLimit) : Player(playerColor, timeLimit) {};
	std::pair<int,int> Simulate(Node* node) override;
};

