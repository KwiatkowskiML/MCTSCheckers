#pragma once
#include "Player.h"

class Game
{
private:
	Player* _whitePlayer = nullptr;
	Player* _blackPlayer = nullptr;
public:
	Game();
	void PlayGame();
	~Game()
	{
		delete _whitePlayer;
		delete _blackPlayer;
	};
};
