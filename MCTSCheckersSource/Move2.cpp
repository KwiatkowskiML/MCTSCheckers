#include "Move2.h"

Move2 Move2::getExtendedMove(UINT newDestination, UINT capt) const
{
	std::vector<UINT> newSteps = steps;
	newSteps.push_back(newDestination);

	UINT newCaptured = captured | capt;
	return Move2(newSteps, newCaptured, color);
}

BitBoard Move2::getBitboardAfterMove(const BitBoard& sourceBitboard) const
{
    {
        // TODO: Implement for black pieces
        assert(color == PieceColor::White);

        UINT source = getSource();
		UINT destination = getDestination();

        // Deleting the initial position of the moved piece
        UINT newWhitePawns = sourceBitboard.whitePawns & ~source;
        UINT newBlackPawns = sourceBitboard.blackPawns;
        UINT newKings = sourceBitboard.kings;

        // Deleting captured pieces
        if (isCapture())
        {
            newBlackPawns = sourceBitboard.blackPawns & ~captured;
            newKings = sourceBitboard.kings & ~captured;
        }

        // Adding new piece position
        newWhitePawns |= destination;

        // Handing the case when the pawn becomes a king, or the king is moved
        if (source & sourceBitboard.kings)
        {
            newKings = sourceBitboard.kings & ~source;
            newKings |= destination;
        }
        else if (destination & WHITE_CROWNING)
            newKings |= destination;

        BitBoard newbitBoard(newWhitePawns, newBlackPawns, newKings);
        return newbitBoard;
    }
}

UINT Move2::getDestination() const
{
    return steps.back();
}

UINT Move2::getSource() const
{
    return steps.front();
}

bool Move2::isCapture() const
{
    return captured > 0;
}

std::string Move2::toString() const
{
    std::string result;

    if (isCapture())
    {
        result = std::to_string(getSource());
		for (int i = 1; i < steps.size(); i++)
			result += ";" + std::to_string(steps[i]);
    }
	else
		result = std::to_string(getSource()) + "-" + std::to_string(getDestination());

	return result;
}
