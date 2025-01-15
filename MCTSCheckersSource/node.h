#pragma once
#include "Board.h"

class Node
{
public:
	// Neighbors nodes
	Node* parent;
	std::vector<Node*> children;

	// Move information
	Board boardAfterMove;
	PieceColor moveColor;
	Move* prevMove;

	// UCT parameters
	int gamesPlayed = 0;
	int score = 0;

	Node(Board boardAfterMove, Node* parent, PieceColor moveColor, Move* prevMove = nullptr) : boardAfterMove(boardAfterMove), parent(parent), moveColor(moveColor), prevMove(prevMove) {};

	// Checking whether the node is a leaf
	bool isLeaf() const { return children.empty(); }

	// UCT formula calculation
	float calculateUCT(PieceColor playerColor) const 
	{ 
		float result;

		if (gamesPlayed == 0)
			return FLT_MAX;

		// If colorToPlay is the color of the player, it means that the move has been done by the enemy
		float k = moveColor != playerColor ? -1.0f : 1.0f;

		// Calculating the formula
		result = k * ((float)score) + C * sqrt(log((float)parent->gamesPlayed) / (float)gamesPlayed);

		return result;
	}

	static void DeleteTree(Node* node)
	{
		if (node)
		{
			for (Node* child : node->children)
			{
				DeleteTree(child);
			}
			delete node;
		}
	}
	
	~Node()
	{
		for (Node* child : children)
		{
			delete child;
		}

		if (prevMove)
			delete prevMove;
	}
};