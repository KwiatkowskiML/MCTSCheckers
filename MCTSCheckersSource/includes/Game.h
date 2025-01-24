#pragma once
#include "Player.h"

class Game
{
private:
	Player* _whitePlayer = nullptr;
	Player* _blackPlayer = nullptr;
	std::string _logFileName = "gamelog.file";
public:
	Game();
	Game(Player* whitePlayer, Player* blackPlayer) : _whitePlayer(whitePlayer), _blackPlayer(blackPlayer) {};
	void PlayGame();
	~Game()
	{
		delete _whitePlayer;
		delete _blackPlayer;
	};
};
