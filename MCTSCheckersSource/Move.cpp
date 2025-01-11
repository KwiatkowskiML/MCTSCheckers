#include "Move.h"
#include <sstream> 
#include <iomanip>

Move Move::getExtendedMove(Move continuation, UINT capt) const
{
	std::vector<UINT> newSteps = steps;
	for (int i = 1; i < continuation.getSteps().size(); ++i)
	{
		newSteps.push_back(continuation.getSteps()[i]);
	}

	UINT newCaptured = captured | capt;
	return Move(newSteps, newCaptured, color);
}

BitBoard Move::getBitboardAfterMove(const BitBoard& sourceBitboard) const
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
    std::ostringstream resultStream;
    resultStream << std::hex << std::setfill('0'); // Set output to hexadecimal and pad with zeros

    if (isCapture())
    {
        resultStream << std::setw(8) << getSource(); // Add leading zeros to ensure 32 bits (8 hex digits)
        for (size_t i = 1; i < steps.size(); ++i)
        {
            resultStream << ";" << std::setw(8) << steps[i];
        }
    }
    else
    {
        resultStream << std::setw(8) << getSource() << "-" << std::setw(8) << getDestination();
    }

    return resultStream.str();
}
