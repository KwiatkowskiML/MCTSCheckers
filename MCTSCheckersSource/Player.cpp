#include "Player.h"

// This function is called when a simulation is done. It backpropagates the score from the simulation node to the root node.
void Player::BackPropagate(Node* node, int score)
{
	Node* currentNode = node;
	while (currentNode != nullptr)
	{
		currentNode->gamesPlayed++;
		currentNode->score += score;
		currentNode = currentNode->parent;
	}
}

// Expansion phase
bool Player::ExpandNode(Node* node)
{
	if (!node->isLeaf())
		return;

	// Get all available moves
	MoveList moves = node->board.getAvailableMoves(node->colorToPlay);
	if (moves.empty())
		return false;

	// Create a new node for each move
	for (Move move : moves)
	{
		Board newBoard = node->board.getBoardAfterMove(move);
		Node* newNode = new Node(newBoard, node, node->colorToPlay == PieceColor::White ? PieceColor::Black : PieceColor::Black);
		node->children.push_back(newNode);
	}

	return true;
}

Move Player::GetBestMove()
{
	// -----------------------------------------------
	// Building the tree
	// ----------------------------------------------

	// Selection phase
	Node* selectedNode = SelectNode();

	int simulationResult = 0;

	// If the selected node is a leaf, simulate the game
	if (selectedNode->gamesPlayed == 0)
	{
		simulationResult = Simulate(selectedNode);
		BackPropagate(selectedNode, simulationResult);
	}
	else
	{
		// Expansion phase
		bool expansionResult = ExpandNode(selectedNode);

		if (expansionResult)
		{
			// Simulation phase
			simulationResult = Simulate(selectedNode->children[0]);
			BackPropagate(selectedNode->children[0], simulationResult);
		}
		else
		{
			// No move possible during the expansion stage, backpropagate the loss
			BackPropagate(selectedNode, LOOSE);
		}
	}

	// ----------------------------------------------
	// Select the best move
	// ----------------------------------------------
	float maxUCT = -FLT_MAX;
	Node* bestNode = nullptr;
	for (Node* child : root->children)
	{
		float uct = child->calculateUCT(color);
		if (uct > maxUCT)
		{
			maxUCT = uct;
			bestNode = child;
		}
	}

	assert(bestNode != nullptr);
	assert(bestNode->prevMove != nullptr);

	return *bestNode->prevMove;
}

Node* Player::SelectNode()
{
	Node* currentNode = root;
	while (currentNode && !currentNode->isLeaf())
	{
		// Calculate the UCT value for each child
		float maxUCT = -FLT_MAX;
		Node* bestNode = nullptr;
		for (Node* child : currentNode->children)
		{
			float uct = child->calculateUCT(color);
			if (uct > maxUCT)
			{
				maxUCT = uct;
				bestNode = child;
			}
		}
		currentNode = bestNode;
	}
	return currentNode;
}
