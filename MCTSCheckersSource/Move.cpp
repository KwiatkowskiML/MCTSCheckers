#include "Move.h"
#include "Board.h"
#include "MoveGenerator.h"

Move::Move(const std::string& moveStr, PieceColor col) : playerColor(col) {
	try {
		// Initialize captured to 0
		captured = 0;

		// Check if it's a capture move (contains ':') or a simple move (contains '-')
		bool isCapture = moveStr.find(':') != std::string::npos;
		char delimiter = isCapture ? ':' : '-';

		// Split the string into fields
		std::vector<std::string> fields;
		size_t start = 0;
		size_t end = moveStr.find(delimiter);

		while (end != std::string::npos) {
			fields.push_back(moveStr.substr(start, end - start));
			start = end + 1;
			end = moveStr.find(delimiter, start);
		}
		fields.push_back(moveStr.substr(start));

		// Convert string fields to board positions using stringToFieldMapping
		steps.clear();
		for (const auto& field : fields) {
			auto it = Board::stringToFieldMapping.find(field);
			if (it == Board::stringToFieldMapping.end()) {
				throw std::invalid_argument("Invalid field notation: " + field);
			}
			steps.push_back(it->second);
		}

		// For capture moves, calculate the captured pieces
		if (isCapture && steps.size() > 1) {
			for (size_t i = 1; i < steps.size(); i++) {
				UINT start_pos = steps[i - 1];
				UINT end_pos = steps[i];

				// Calculate position of captured piece
				UINT captured_pos = Board::getAllFieldsBetween(start_pos, end_pos); // TODO: fix it
				captured |= captured_pos;
			}
		}

	}
	catch (const std::exception& e) {
		throw std::invalid_argument("Invalid move string format: " + moveStr);
	}
}

Move Move::getExtendedMove(Move continuation, UINT capt) const
{
	std::vector<UINT> newSteps = steps;
	for (int i = 1; i < continuation.getSteps().size(); ++i)
	{
		newSteps.push_back(continuation.getSteps()[i]);
	}

	UINT newCaptured = captured | capt;
	return Move(newSteps, newCaptured, playerColor);
}

BitBoard Move::getBitboardAfterMove(const BitBoard& sourceBitboard, bool includeCoronation) const
{
	UINT source = getSource();
	UINT destination = getDestination();
	UINT currentPieces = sourceBitboard.getPieces(playerColor);
	UINT enemyPieces = sourceBitboard.getPieces(getEnemyColor(playerColor));

	// Deleting the initial position of the moved piece
	UINT newCurrentPieces = currentPieces & ~source;
	UINT newEnemyPieces = enemyPieces;
	UINT newKings = sourceBitboard.kings;

	// Deleting captured pieces
	if (isCapture())
	{
		newEnemyPieces = enemyPieces & ~captured;
		newKings = newKings & ~captured;
	}

	// Adding new piece position
	newCurrentPieces |= destination;

	// Handing the case when the king is moved
	if (source & sourceBitboard.kings)
	{
		newKings = newKings & ~source;
		newKings |= destination;
	}

	// Handling the case when the pawn is crowned
	if (includeCoronation)
	{

		if (playerColor == PieceColor::White && (getDestination() & WHITE_CROWNING))
			newKings |= destination;
		if (playerColor == PieceColor::Black && (getDestination() & BLACK_CROWNING))
			newKings |= destination;
	
	}

	UINT newWhitePawns = playerColor == PieceColor::White ? newCurrentPieces : newEnemyPieces;
	UINT newBlackPawns = playerColor == PieceColor::Black ? newCurrentPieces : newEnemyPieces;

	BitBoard newbitBoard(newWhitePawns, newBlackPawns, newKings);
	return newbitBoard;
}

const std::vector<UINT>& Move::getSteps() const
{
	return steps;
}

UINT Move::getDestination() const
{
	return steps.back();
}

UINT Move::getSource() const
{
	return steps.front();
}

bool Move::isCapture() const
{
	return captured > 0;
}

std::string Move::toString() const
{
	std::string result;
	try {
		if (isCapture())
		{
			result += Board::fieldToStringMapping.at(getSource());
			for (size_t i = 1; i < steps.size(); ++i)
			{
				result += ":" + Board::fieldToStringMapping.at(steps[i]);
			}
		}
		else
		{
			result = Board::fieldToStringMapping.at(getSource()) + "-" + Board::fieldToStringMapping.at(getDestination());
		}
	}
	catch (const std::out_of_range& e)
	{
		result = "There is no such field in the fields dictionary";
	}

	return result;
}
