#pragma once
#include "PieceColor.h"
#include "node.h"

class Player
{
public:
	int timeLimit;
	const PieceColor color;
	Node* root = nullptr;

	// Constructor to initialize the player and the root of the decision tree
	Player(PieceColor color, int timeLimit) : color(color), timeLimit(timeLimit) 
	{
		SetBoard(Board(INIT_WHITE_PAWNS, INIT_BLACK_PAWNS, 0));
	};

	// Reset the tree with a new board
	void SetBoard(Board board);

	// This must be implemented by derived classes
	virtual int Simulate(Node* node) = 0;

	// Backpropagate the result of a simulation up the tree
	void BackPropagate(Node* node, int score);

	// Expand the tree by generating children for the given node
	bool ExpandNode(Node* node);

	// Get the best move based on the results of simulations
	Move* GetBestMove();

	// Select the most promising node 
	Node* SelectNode();

	// Generate a DOT file for visualizing the decision tree using Graphviz
	void GenerateDotFile(const std::string& filename);
	
	// Destructor to clean up the decision tree
	~Player() 
	{
		delete root;
	};
};

