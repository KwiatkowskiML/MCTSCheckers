#include "../includes/Game.h"
#include "../includes/PlayerCPU.h"
#include "../includes/PlayerGPU.h"
#include "../includes/PlayerReal.h"
#include <fstream>

#define LOG_GAME_TREE

Game::Game()
{
	std::string input;
	bool shouldAskAgain;

	// Get the white player type
	std::cout << "White:" << std::endl;
	std::cout << "1. Player" << std::endl;
	std::cout << "2. CPU" << std::endl;
	std::cout << "3. GPU" << std::endl;
	do
	{
		shouldAskAgain = false;
		std::cin >> input;
		switch (stoi(input))
		{
		case 1:
			_whitePlayer = new PlayerReal(PieceColor::White, DEFAULT_TIME_LIMIT);
			break;
		case 2:
			_whitePlayer = new PlayerCPU(PieceColor::White, DEFAULT_TIME_LIMIT);
			break;
		case 3:
			_whitePlayer = new PlayerGPU(PieceColor::White, DEFAULT_TIME_LIMIT);
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
	std::cout << "3. GPU" << std::endl;
	do
	{
		shouldAskAgain = false;
		std::cin >> input;
		switch (stoi(input))
		{
		case 1:
			_blackPlayer = new PlayerReal(PieceColor::Black, DEFAULT_TIME_LIMIT);
			break;
		case 2:
			_blackPlayer = new PlayerCPU(PieceColor::Black, DEFAULT_TIME_LIMIT);
			break;
		case 3:
			_blackPlayer = new PlayerGPU(PieceColor::Black, DEFAULT_TIME_LIMIT);
			break;
		default:
			shouldAskAgain = true;
			std::cout << "Invalid value!" << std::endl;
			break;
		}
	} while (shouldAskAgain);
}

void Game::PlayGame()
{
	Board newBoard = _whitePlayer->root->boardAfterMove;
	bool gameEnded = false;
	std::ofstream logFile(GAME_LOG_FILE);
	int iteration = 0;
	int noCaptureMoves = 0;

	if (!logFile) {
		std::cerr << "Error opening log file!" << std::endl;
		return;
	}

	std::cout << newBoard.toString() << std::endl;

	// Game loop
	while (true)
	{
		// Keeping count of the iterations done
		iteration++;

		// White player move
		Move* whiteMove = _whitePlayer->GetBestMove();
		if (whiteMove == nullptr)
		{
			logFile << "White player has no moves left. Black player wins!" << std::endl;
			std::cout << "White player has no moves left. Black player wins!" << std::endl;
			return;
		}
		newBoard = _whitePlayer->root->boardAfterMove.getBoardAfterMove(*whiteMove);

		logFile << iteration << ". White player move: " << whiteMove->toString() << std::endl;
		std::cout << iteration << ". White player move: " << whiteMove->toString() << std::endl;
		logFile << newBoard.toString() << std::endl;
		std::cout << newBoard.toString() << std::endl;

		// Checking if the move is a capture
		if (!whiteMove->isCapture() && (whiteMove->getSource() & newBoard.getKings()) > 0) {
			noCaptureMoves++;
		}
		else {
			noCaptureMoves = 0;
		}

		// Checking for the draw
		if (noCaptureMoves >= MAX_NO_CAPTURE_MOVES) {
			logFile << "DRAW" << std::endl;
			std::cout << "DRAW" << std::endl;
			return;
		}

		// Update black players board
		_blackPlayer->SetBoard(newBoard);

		// Black player move
		Move* blackMove = _blackPlayer->GetBestMove();
		if (blackMove == nullptr)
		{
			logFile << "Black player has no moves left. White player wins!" << std::endl;
			std::cout << "Black player has no moves left. White player wins!" << std::endl;
			return;
		}
		newBoard = _blackPlayer->root->boardAfterMove.getBoardAfterMove(*blackMove);

		logFile << iteration << ". Black player move: " << blackMove->toString() << std::endl;
		std::cout << iteration << ". Black player move: " << blackMove->toString() << std::endl;
		logFile << newBoard.toString() << std::endl;
		std::cout << newBoard.toString() << std::endl;

		// Checking if the move is a capture
		if (!blackMove->isCapture() && (blackMove->getSource() & newBoard.getKings()) > 0) {
			noCaptureMoves++;
		}
		else {
			noCaptureMoves = 0;
		}

		// Checking for the draw
		if (noCaptureMoves >= MAX_NO_CAPTURE_MOVES) {
			logFile << "DRAW" << std::endl;
			std::cout << "DRAW" << std::endl;
			return;
		}

		// Set up white players board
		_whitePlayer->SetBoard(newBoard);
	}
 }