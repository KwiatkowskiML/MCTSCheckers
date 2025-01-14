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

		// Roots children initialization with saving each move
		MoveList moves = root->board.getAvailableMoves(color);
		for (Move move : moves)
		{
			Board newBoard = root->board.getBoardAfterMove(move);
			Node* newNode = new Node(newBoard, root, color == PieceColor::White ? PieceColor::Black : PieceColor::White, new Move(move));
			root->children.push_back(newNode);
		}
	};

	virtual int Simulate(Node* node) = 0;
	void BackPropagate(Node* node, int score);
	bool ExpandNode(Node* node);
	Move* GetBestMove();
	Node* SelectNode();
	
	~Player() 
	{
		delete root;
	};
};

