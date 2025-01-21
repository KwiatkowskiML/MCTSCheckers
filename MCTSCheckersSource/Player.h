#pragma once
#include <utility>

#include "PieceColor.h"
#include "node.h"

class Player
{
public:
	int timeLimit;
	const PieceColor playerColor;
	Node* root = nullptr;

	// Constructor to initialize the player and the root of the decision tree
	Player(PieceColor playerColor, int timeLimit) : playerColor(playerColor), timeLimit(timeLimit) 
	{
		SetBoard(Board(INIT_WHITE_PAWNS, INIT_BLACK_PAWNS, 0));
	};

	// Reset the tree with a new board
	void SetBoard(Board boardAfterMove);

	// This must be implemented by derived classes, first element of the pair is the simulation result and the sceond is the number of simulations
    virtual std::pair<int, int> Simulate(Node* node) = 0;

	// Backpropagate the result of a simulation up the tree
	void BackPropagate(Node* node, std::pair<int,int> simulationResult);

	// Expand the tree by generating children for the given node
	bool ExpandNode(Node* node);

	// Get the best move based on the results of simulations
	Move* GetBestMove();

	// Select the most promising node 
	Node* SelectNode();

	// Generate a DOT file for visualizing the decision tree using Graphviz
	void GenerateDotFile(const std::string& filename);
	std::string GenerateTreeString();
	
	// Destructor to clean up the decision tree
	~Player() 
	{
		delete root;
	};
};

