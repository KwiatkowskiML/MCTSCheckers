#include "Move2.h"
#include <sstream> 
#include <iomanip>

Move2 Move2::getExtendedMove(Move2 continuation, UINT capt) const
{
	std::vector<UINT> newSteps = steps;
	for (int i = 1; i < continuation.getSteps().size(); ++i)
	{
		newSteps.push_back(continuation.getSteps()[i]);
	}

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

const std::vector<UINT>& Move2::getSteps() const
{
    return steps;
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
