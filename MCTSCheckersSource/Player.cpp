#include "Player.h"
#include <fstream>
#include <string>
#include <functional>
#include <chrono>
#include <sstream>

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

// Expansion phase
bool Player::ExpandNode(Node* node)
{
	if (!node->isLeaf())
		return false;

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

// Set players board
void Player::SetBoard(Board board)
{
	// Clear the tree
	if (root)
		delete root;

	// Initialize the root node
	root = new Node(board, nullptr, color);
	
	// Roots children initialization with saving each move
	MoveList moves = root->board.getAvailableMoves(color);
	for (Move move : moves)
	{
		Board newBoard = root->board.getBoardAfterMove(move);
		Node* newNode = new Node(newBoard, root, color == PieceColor::White ? PieceColor::Black : PieceColor::White, new Move(move));
		root->children.push_back(newNode);
	}
}

// This function is called when a simulation is done. It backpropagates the score from the simulation node to the root node.
void Player::BackPropagate(Node* node, int score)
{
	Node* currentNode = node;

	// if the node was simulating the enemy color, invert the score
	if (currentNode->colorToPlay != color)
		score = -score;

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
	for (int i = 0; i < 10; i++)
	{ 
		Node* selectedNode = SelectNode();
		assert(selectedNode != nullptr);
		assert(selectedNode->isLeaf());

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

		std::string filename = TREE_VISUALIZATION_PREFIX + std::to_string(i) + ".dot";
		GenerateDotFile(filename);
	}

	// ----------------------------------------------
	// Select the best move
	// ----------------------------------------------
	float maxUCT = -FLT_MAX;
	Node* bestNode = nullptr;
	Move* bestMove = nullptr;
	for (Node* child : root->children)
	{
		float uct = child->calculateUCT(color);
		if (uct > maxUCT)
		{
			maxUCT = uct;
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
			treeString << "\\nUCT: " << node->calculateUCT(color);

		// Write the previous move if exists
		if (node->prevMove != nullptr)
			treeString << "\\nMove: " << node->prevMove->toString();

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
