#pragma once
#include "Player.h"

class Game
{
private:
	Player* _whitePlayer;
	Player* _blackPlayer;
public:
	Game(Player* whitePlayer, Player* blackPlayer) : _whitePlayer(whitePlayer), _blackPlayer(blackPlayer) {};
	void PlayGame();
	void PlayGameAsWhite();
	static int GetGameSetup(Player*& whitePlayer, Player*& blackPlayer);
	~Game() {};
};

