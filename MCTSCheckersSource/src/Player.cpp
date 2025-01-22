#include <fstream>
#include <string>
#include <functional>
#include <chrono>
#include <sstream>

#include "../includes/Player.h"

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
	assert(node->isLeaf());

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
void Player::BackPropagate(Node* node, std::pair<int,int> simulationResult)
{
	Node* currentNode = node;
	int simulationScore = simulationResult.first;
	int numberOfGamesPlayed = simulationResult.second;

	// Update current node score
	currentNode->gamesPlayed += numberOfGamesPlayed;
	currentNode->score += simulationScore;
	currentNode = currentNode->parent;

	while (currentNode != nullptr)
	{
		currentNode->gamesPlayed += numberOfGamesPlayed;
		currentNode->score += simulationScore;
		currentNode = currentNode->parent;
	}
}

Move* Player::GetBestMove()
{
	// -----------------------------------------------
	// Building the tree
	// ----------------------------------------------

	// Start the timer
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() < timeLimit)
	{ 

		Node* selectedNode = SelectNode();
		assert(selectedNode != nullptr);
		assert(selectedNode->isLeaf());

		if (selectedNode->parent == nullptr)
			return nullptr;

		std::pair<int,int> simulationResult;

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
				// No move possible during the expansion stage
				if (selectedNode->moveColor == playerColor)
					simulationResult = std::make_pair(WIN, 1);
				else
					simulationResult = std::make_pair(LOOSE, 1);

				BackPropagate(selectedNode, simulationResult);
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

		// std::cout << "Move: " << child->prevMove->toString() << " Score: " << child->score << " Games Played: " << child->gamesPlayed << " Average: " << average << std::endl;
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
	treeString << "    node [shape=box, fontname=\"Arial\", style=filled];\n";

	// Helper function to traverse and generate nodes
	std::function<void(const Node*, int&)> writeNode;
	writeNode = [&treeString, &writeNode, this](const Node* node, int& nodeId) -> void {
		if (!node)
			return;

		int currentNodeId = nodeId++;
		// Start writing the node
		treeString << "    node" << currentNodeId << " [label=\"Games Played: "
			<< node->gamesPlayed << "\\nScore: " << node->score;

		// Write the UCT score if parent exists
		if (node->parent != nullptr)
			treeString << "\\nUCT: " << node->calculateUCT(playerColor);

		// Write the previous move if exists
		if (node->prevMove != nullptr)
			treeString << "\\nMove: " << node->prevMove->toString();

		// Color to play
		treeString << "\\nTurn: " << (node->moveColor == PieceColor::White ? "White" : "Black");

		// Check if the parent is the root and set fill color to red
		if (node->parent && node->parent == root)
			treeString << "\", fillcolor=red";
		else if (node->parent)
			treeString << "\", fillcolor=white";
		else if (!node->parent)
			treeString << "\", fillcolor=green";

		// Close the node definition
		treeString << "];\n";

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

