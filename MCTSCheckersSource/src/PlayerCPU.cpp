#include "../includes/PlayerCPU.h"

std::pair<int,int> PlayerCPU::Simulate(Node* node)
{
	// simulate game after the move
	int winner = node->boardAfterMove.simulateGame(getEnemyColor(node->moveColor));
	int result = 0;

	if (winner == BLACK_WIN && playerColor == PieceColor::Black)
		result = WIN;
	else if (winner == WHITE_WIN && playerColor == PieceColor::White)
		result = WIN;
	else if (winner == DRAW)
		result = DRAW;
	else
		result = LOOSE;

	return std::make_pair(result,1);
}
