#pragma once
#include "Player.h"

class PlayerReal : public Player
{
public:
	PlayerReal(PieceColor playerColor, int timeLimit) : Player(playerColor, timeLimit) {};
	std::pair<int, int> Simulate(Node* node) override;
	Move* GetBestMove() override;
};

