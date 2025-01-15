#include "Player.h"
#include <fstream>
#include <string>
#include <functional>
#include <chrono>
#include <sstream>

// #define TREE_BUILD_DEBUG

// Set players board
void Player::SetBoard(Board board)
{
	// Clear the tree
	if (root)
		delete root;

	// Initialize the root node
	root = new Node(board, nullptr, playerColor);

	// Roots children initialization with saving each move
	MoveList moves = root->boardAfterMove.getAvailableMoves(playerColor);
	for (Move move : moves)
	{
		Board newBoard = root->boardAfterMove.getBoardAfterMove(move);
		Node* newNode = new Node(newBoard, root, playerColor, new Move(move));
		root->children.push_back(newNode);
	}
}

// Selection phase
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
			float uct = child->calculateUCT(playerColor);
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

// Expansion phase
bool Player::ExpandNode(Node* node)
{
	if (!node->isLeaf())
		return false;

	// Get all available enemy moves
	MoveList moves = node->boardAfterMove.getAvailableMoves(getEnemyColor(node->moveColor));
	if (moves.empty())
		return false;

	// Create a new node for each move
	for (Move move : moves)
	{
		Board newBoard = node->boardAfterMove.getBoardAfterMove(move);
		Node* newNode = new Node(newBoard, node, getEnemyColor(node->moveColor), new Move(move));
		node->children.push_back(newNode);
	}

	return true;
}

// This function is called when a simulation is done. It backpropagates the score from the simulation node to the root node.
void Player::BackPropagate(Node* node, int score)
{
	Node* currentNode = node;

	// Update current node score
	currentNode->gamesPlayed++;
	currentNode->score += score;
	currentNode = currentNode->parent;

	while (currentNode != nullptr)
	{
		currentNode->gamesPlayed++;
		currentNode->score += score;
		currentNode = currentNode->parent;
	}
}

Move* Player::GetBestMove()
{
	// -----------------------------------------------
	// Building the tree
	// ----------------------------------------------

	// Start the timer
	//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	//while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() < timeLimit)
	for (int i = 0; i < 1000; i++)
	{ 

		Node* selectedNode = SelectNode();
		assert(selectedNode != nullptr);
		assert(selectedNode->isLeaf());

		int simulationResult = 0;

		// If the selected node is a leaf, simulate the game
		if (selectedNode->gamesPlayed == 0)
		{
			simulationResult = Simulate(selectedNode);

			int simulationScore;

			if (simulationResult == WHITE_WIN && playerColor == PieceColor::White)
				simulationScore = WIN;
			else if (simulationResult == BLACK_WIN && playerColor == PieceColor::Black)
				simulationScore = WIN;
			else if (simulationResult == DRAW)
				simulationScore = DRAW;
			else
				simulationScore = LOOSE;

			BackPropagate(selectedNode, simulationScore);
		}
		else
		{
			// Expansion phase
			bool expansionResult = ExpandNode(selectedNode);

			if (expansionResult)
			{
				// Simulation phase
				simulationResult = Simulate(selectedNode->children[0]);

				int simulationScore;

				if (simulationResult == WHITE_WIN && playerColor == PieceColor::White)
					simulationScore = WIN;
				else if (simulationResult == BLACK_WIN && playerColor == PieceColor::Black)
					simulationScore = WIN;
				else if (simulationResult == DRAW)
					simulationScore = DRAW;
				else
					simulationScore = LOOSE;

				BackPropagate(selectedNode->children[0], simulationScore);
			}
			else
			{
				// No move possible during the expansion stage, backpropagate the loss
				BackPropagate(selectedNode, LOOSE);
			}
		}

#ifdef TREE_BUILD_DEBUG
		std::string filename = TREE_VISUALIZATION_PREFIX + std::to_string(i) + ".dot";
		GenerateDotFile(filename);
#endif // TREE_BUILD_DEBUG		
	}

	// ----------------------------------------------
	// Select the best move
	// ----------------------------------------------
	float bestAverage = -FLT_MAX;
	Node* bestNode = nullptr;
	Move* bestMove = nullptr;
	for (Node* child : root->children)
	{
		float average = (float)child->score / (float)child->gamesPlayed;
		if (average > bestAverage)
		{
			bestAverage = average;
			bestNode = child;
		}
	}

	if (bestNode)
		bestMove = bestNode->prevMove;

	return bestMove;
}

//----------------------------------------------
// Generate a dot file to visualize the tree
//----------------------------------------------
void Player::GenerateDotFile(const std::string& filename)
{
	// Generate the tree as a string in DOT format
	std::string treeDot = GenerateTreeString();

	// Open the file to write the DOT string
	std::ofstream dotFile(filename);
	if (!dotFile.is_open())
	{
		throw std::runtime_error("Unable to open file: " + filename);
	}

	// Write the DOT string to the file
	dotFile << treeDot;

	// Close the file
	dotFile.close();
}


std::string Player::GenerateTreeString()
{
	std::ostringstream treeString;  // To store the string representation of the tree

	// Start the DOT graph definition
	treeString << "digraph Tree {\n";
	treeString << "    node [shape=box, fontname=\"Arial\"];\n";

	// Helper function to traverse and generate nodes
	std::function<void(const Node*, int&)> writeNode;
	writeNode = [&treeString, &writeNode, this](const Node* node, int& nodeId) -> void {
		if (!node)
			return;

		int currentNodeId = nodeId++;
		// Write the current node with gamesPlayed and score
		treeString << "    node" << currentNodeId << " [label=\"Games Played: "
			<< node->gamesPlayed << "\\nScore: " << node->score;

		// Write the UCT score if parent exists
		if (node->parent != nullptr)
			treeString << "\\nUCT: " << node->calculateUCT(playerColor);

		// Write the previous move if exists
		if (node->prevMove != nullptr)
			treeString << "\\nMove: " << node->prevMove->toString();

		// color to play
		treeString << "\nTurn: " << (node->moveColor == PieceColor::White ? "White" : "Black");

		treeString << "\"];\n";

		for (const Node* child : node->children)
		{
			if (child)
			{
				int childNodeId = nodeId;
				writeNode(child, nodeId); // Recursively write child nodes

				// Connect current node to child node
				treeString << "    node" << currentNodeId << " -> node" << childNodeId << ";\n";
			}
		}
		};

	int nodeId = 0;
	writeNode(root, nodeId);

	// End the DOT graph definition
	treeString << "}\n";

	return treeString.str();  // Return the generated DOT string
}
