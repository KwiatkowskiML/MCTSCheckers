#pragma once
#include "Board.h"

struct Node
{
	Board board;
	Node* parent;
	std::vector<Node*> children;
	PieceColor colorToPlay;

	int gamesPlayed = 0;
	int score = 0;

	Node(Board board, Node* parent, PieceColor colorToPlay) : board(board), parent(parent), colorToPlay(colorToPlay) {};
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
	}
};