#include "../includes/PlayerCPU.h"
#include <random>
#include "../includes/PlayerGPU.h"

std::pair<int,int> PlayerCPU::Simulate(Node* node)
{
	// simulate game after the move
	//int winner = node->boardAfterMove.simulateGame(getEnemyColor(node->moveColor));
	
	std::random_device rd; // Seed
	std::mt19937 gen(rd()); // Mersenne Twister engine
	std::uniform_int_distribution<> dist(-999999, 999999);

	int z = dist(gen);
	int w = dist(gen);
	int jsr = dist(gen);
	int jcong = dist(gen);

	int winner = simulateGameGpu(node->boardAfterMove.getWhitePawns(), node->boardAfterMove.getBlackPawns(), node->boardAfterMove.getKings(),
		getEnemyColor(node->moveColor), z, w, jsr, jcong);
	
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
