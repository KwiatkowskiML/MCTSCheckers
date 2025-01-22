#include <fstream>
#include "PlayerReal.h"

std::pair<int, int> PlayerReal::Simulate(Node* node)
{
    return std::pair<int, int>();
}

Move* PlayerReal::GetBestMove()
{
	// Cehck if player input move is correct
	std::string stringMove;
	bool askAgain = false;
	Move move(0, 0, 0, playerColor);

	do {
		std::cout << "Enter your move: ";
		std::cin >> stringMove;
		askAgain = false;
		try
		{
			// Check if the move is valid
			move = Move(stringMove, playerColor);
			for (Node* child : root->children)
			{
				if (child->prevMove->getSource() == move.getSource() &&
					child->prevMove->getDestination() == move.getDestination() &&
					child->prevMove->getSteps() == move.getSteps())
				{
					return child->prevMove;
				}
			}
			std::cout << "Invalid move!" << std::endl;
			askAgain = true;
		}
		catch (const std::exception&)
		{
			std::cout << "Invalid input!" << std::endl;
			askAgain = true;
		}
	} while (askAgain);
}
