#pragma once
#include "Board.h"

struct Node
{
	Board board;
	Node* parent;
	std::vector<Node*> children;
	PieceColor colorToPlay;
	Move* prevMove;

	int gamesPlayed = 0;
	int score = 0;

	bool isLeaf() const { return children.empty(); }
	float calculateUCT(PieceColor color) const 
	{ 
		float result;

		if (gamesPlayed == 0)
			return FLT_MAX;

		// If colorToPlay is the color of the player, it means that the move has been done by the enemy
		float k = colorToPlay == color ? -1.0f : 1.0f;

		// Calculating the formula
		result = ((float)score) + C * sqrt(log((float)parent->gamesPlayed) / (float)gamesPlayed);

		return result * k;
	}

	Node(Board board, Node* parent, PieceColor colorToPlay, Move* prevMove = nullptr) : board(board), parent(parent), colorToPlay(colorToPlay), prevMove(prevMove) {};
	
	~Node()
	{
		for (Node* child : children)
		{
			if (child)
			{
				delete child;
			}
		}
		children.clear();

		if (prevMove)
			delete prevMove;
	}
};