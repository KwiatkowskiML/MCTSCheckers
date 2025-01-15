#include "PlayerCPU.h"

int PlayerCPU::Simulate(Node* node)
{
	int result = node->boardAfterMove.simulateGame(getEnemyColor(node->moveColor));
	return result;
}
