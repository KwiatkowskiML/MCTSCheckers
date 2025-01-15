#include "Player.h"
#include <fstream>
#include <string>
#include <functional>
#include <chrono>

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
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() < timeLimit)
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
	std::ofstream dotFile(filename);
	if (!dotFile.is_open())
	{
		throw std::runtime_error("Unable to open file: " + filename);
	}

	// Start the DOT graph definition
	dotFile << "digraph Tree {\n";
	dotFile << "    node [shape=box, fontname=\"Arial\"];\n";

	// Helper function to traverse and generate nodes
	std::function<void(const Node*, int&)> writeNode;
	writeNode = [&dotFile, &writeNode, this](const Node* node, int& nodeId) -> void {
		if (!node)
			return;

		int currentNodeId = nodeId++;
		// Write the current node with gamesPlayed and score
		dotFile << "    node" << currentNodeId << " [label=\"Games Played: "
			<< node->gamesPlayed << "\\nScore: " << node->score;

		// write the uct score
		if (node->parent != nullptr)
			dotFile << "\\nUCT: " << node->calculateUCT(color);

		if (node->prevMove != nullptr)
			dotFile << "\\nMove: " << node->prevMove->toString();

		dotFile << "\"];\n";

		for (const Node* child : node->children)
		{
			if (child)
			{
				int childNodeId = nodeId;
				writeNode(child, nodeId); // Recursively write child nodes

				// Connect current node to child node
				dotFile << "    node" << currentNodeId << " -> node" << childNodeId << ";\n";
			}
		}
		};

	int nodeId = 0;
	writeNode(root, nodeId);

	// End the DOT graph definition
	dotFile << "}\n";

	dotFile.close();
}