#include "Game.h"
#include "PlayerCPU.h"
#include <fstream>

void Game::PlayGame()
{
	Board newBoard;
	bool gameEnded = false;
	std::ofstream logFile(GAME_LOG_FILE);

	if (!logFile) {
		std::cerr << "Error opening log file!" << std::endl;
		return;
	}

	// Game loop
	for (int i = 0; i < 100; i++)
	{
		// White player move
		Move* whiteMove = _whitePlayer->GetBestMove();
		if (whiteMove == nullptr)
		{
			logFile << "White player has no moves left. Black player wins!" << std::endl;
			break;
		}
		newBoard = _whitePlayer->root->board.getBoardAfterMove(*whiteMove);

		logFile << "White player move: " << whiteMove->toString() << std::endl;
		logFile << newBoard.toString() << std::endl;

		// Update black players board
		_blackPlayer->SetBoard(newBoard);

		// Black player move
		Move* blackMove = _blackPlayer->GetBestMove();
		if (blackMove == nullptr)
		{
			logFile << "Black player has no moves left. White player wins!" << std::endl;
			break;
		}
		newBoard = _blackPlayer->root->board.getBoardAfterMove(*blackMove);

		logFile << "Black player move: " << blackMove->toString() << std::endl;
		logFile << newBoard.toString() << std::endl;
	}
}

void Game::GetGameSetup(Player*& whitePlayer, Player*& blackPlayer)
{
	std::string input;
	bool shouldAskAgain;

	// Get the white player type
	std::cout << "White:" << std::endl;
	std::cout << "1. Player" << std::endl;
	std::cout << "2. CPU" << std::endl;
	do
	{
		shouldAskAgain = false;
		std::cin >> input;
		switch (stoi(input))
		{
		case 1:
			std::cout << "Not implemented yet" << std::endl;
			break;
		case 2:
			whitePlayer = new PlayerCPU(PieceColor::White, DEFAULT_TIME_LIMIT);
			break;
		default:
			shouldAskAgain = true;
			std::cout << "Invalid value!" << std::endl;
			break;
		}
	} while (shouldAskAgain);

	// Get the black player type
	std::cout << "Black:" << std::endl;
	std::cout << "1. Player" << std::endl;
	std::cout << "2. CPU" << std::endl;
	do
	{
		shouldAskAgain = false;
		std::cin >> input;
		switch (stoi(input))
		{
		case 1:
			std::cout << "Not implemented yet" << std::endl;
			break;
		case 2:
			blackPlayer = new PlayerCPU(PieceColor::Black, DEFAULT_TIME_LIMIT);
			break;
		default:
			shouldAskAgain = true;
			std::cout << "Invalid value!" << std::endl;
			break;
		}
	} while (shouldAskAgain);
}
