#include "Game.h"
#include "PlayerCPU.h"
#include <fstream>

#define LOG_GAME_TREE

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
		newBoard = _whitePlayer->root->boardAfterMove.getBoardAfterMove(*whiteMove);

		logFile << i << ". White player move: " << whiteMove->toString() << std::endl;
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
		newBoard = _blackPlayer->root->boardAfterMove.getBoardAfterMove(*blackMove);

		logFile << "Black player move: " << blackMove->toString() << std::endl;
		logFile << newBoard.toString() << std::endl;

		// Set up white players board
		_whitePlayer->SetBoard(newBoard);
	}
}

void Game::PlayGameAsWhite()
{
	Board newBoard(INIT_WHITE_PAWNS, INIT_BLACK_PAWNS, 0);
	bool gameEnded = false;
	std::ofstream logFile(GAME_LOG_FILE);

	if (!logFile) {
		std::cerr << "Error opening log file!" << std::endl;
		return;
	}

	for (int i = 0; i < 100; i++) // TODO: correct
	{
		// White player move
		std::cout << newBoard.toString() << std::endl;

		// get white player move
		std::string move;
		Move whiteMove(0,0);
		MoveList possibleWhiteMoves = newBoard.getAvailableMoves(PieceColor::White);

		// Cehck if player input move is correct
		bool askAgain = false;
		do {
			std::cout << "Enter your move: ";
			std::cin >> move;
			try
			{
				whiteMove = Move(move, PieceColor::White); 
				if (!Move::containsMove(possibleWhiteMoves, whiteMove))
				{
					std::cout << "Invalid move!" << std::endl << "Possible moves:" << std::endl;
					for (const Move& m : possibleWhiteMoves)
					{
						std::cout << m.toString() << std::endl;
					}
					askAgain = true;
				}
				else
				{
					askAgain = false;
				}
			}
			catch (const std::exception&)
			{
				std::cout << "Invalid input!" << std::endl;
				askAgain = true;
			}
		} while (askAgain);

		// update board
		newBoard = newBoard.getBoardAfterMove(whiteMove);

		// Logging white move
		logFile << "White player move: " << whiteMove.toString() << std::endl;
		logFile << newBoard.toString() << std::endl;
		std::cout << newBoard.toString() << std::endl;

		// Update black players board
		_blackPlayer->SetBoard(newBoard);

		// Black player move
		Move* blackMove = _blackPlayer->GetBestMove();
		if (blackMove == nullptr)
		{
			logFile << "Black player has no moves left. White player wins!" << std::endl;
			break;
		}
		newBoard = _blackPlayer->root->boardAfterMove.getBoardAfterMove(*blackMove);

#ifdef LOG_GAME_TREE
		std::string filename = TREE_GAME_LOG_PREFIX + std::to_string(i) + ".dot";
		_blackPlayer->GenerateDotFile(filename);
#endif // LOG_GAME_TREE

		// logging into file
		logFile << "Black player move: " << blackMove->toString() << std::endl;
		logFile << newBoard.toString() << std::endl;

		std::cout << "Black player move: " << blackMove->toString() << std::endl;
	}
}

int Game::GetGameSetup(Player*& whitePlayer, Player*& blackPlayer)
{
	std::string input;
	bool shouldAskAgain;
	int result = 0;

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
			result = 1;
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

	return result;
}
