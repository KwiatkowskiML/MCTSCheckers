#include "PlayerCPU.h"

int PlayerCPU::Simulate(Node* node)
{
	int result = node->board.simulateGame(node->colorToPlay);
	return result;
}
