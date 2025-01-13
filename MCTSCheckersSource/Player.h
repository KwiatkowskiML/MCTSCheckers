#pragma once
#include "PieceColor.h"
#include "node.h"

class Player
{
public:
	int timeLimit;
	const PieceColor color;
	Node* root;

	Player(PieceColor color, int timeLimit) : color(color), timeLimit(timeLimit) 
	{
		root = new Node(Board(INIT_WHITE_PAWNS, INIT_BLACK_PAWNS, 0), nullptr, color);
		root->colorToPlay = color;
	};

	virtual void Simulate(Node* node) = 0;
	
	~Player() 
	{
		delete root;
	};
};

