#include "Board.h"

UINT Board::GetBlackJumpers()
{
	uint32_t jumpers = 0;
	return jumpers;
}

UINT Board::GetBlackMovers()
{
	return 0;
}

void Board::GetBlackAvailableMoves(std::queue<Board>& availableMoves)
{
    UINT blackMovers = 0;
    UINT blackJumpers = 0;


}

UINT Board::GetWhiteJumpers()
{
    const UINT emptyFields = ~(_whitePawns | _blackPawns);
    const UINT whiteKings = _whitePawns & _kings;

	UINT jumpers = 0;
    
	// Get the black pawns that might be captured in base diagonal direction
    UINT captBlackPawns = (emptyFields << BASE_DIAGONAL_SHIFT) & _blackPawns;

	// Check whether previously specified black pawns can actually be captured
	if (captBlackPawns)
	{
		// Get the white pawns that can capture black pawn in the base diagonal direction
		jumpers |= ((captBlackPawns & MOVES_UP_LEFT_AVAILABLE) << UP_LEFT_SHIFT) & _whitePawns;
		jumpers |= ((captBlackPawns & MOVES_UP_RIGHT_AVAILABLE) << UP_RIGHT_SHIFT) & _whitePawns;
	}

	// Get the black pawns that might be captured in the other diagonal direction
	captBlackPawns = ((emptyFields & MOVES_UP_LEFT_AVAILABLE) << UP_LEFT_SHIFT) & _blackPawns;
	captBlackPawns |= ((emptyFields & MOVES_UP_RIGHT_AVAILABLE) << UP_RIGHT_SHIFT) & _blackPawns;

	jumpers |= (captBlackPawns << BASE_DIAGONAL_SHIFT) & _whitePawns;

	// Find all of the black pawns that might be captured backwards in base diagonal
    captBlackPawns = (emptyFields >> BASE_DIAGONAL_SHIFT) & _blackPawns;

    // Check whether previously specified black pawns can actually be captured
    if (captBlackPawns)
    {
		jumpers |= ((captBlackPawns & MOVES_DOWN_LEFT_AVAILABLE) >> DOWN_LEFT_SHIFT) & _whitePawns;
		jumpers |= ((captBlackPawns & MOVES_DOWN_RIGHT_AVAILABLE) >> DOWN_RIGHT_SHIFT) & _whitePawns;
    }

	// Find all of the black pawns that might be captured backwards in the other diagonal
	captBlackPawns = ((emptyFields & MOVES_DOWN_LEFT_AVAILABLE) >> DOWN_LEFT_SHIFT) & _blackPawns;
	captBlackPawns |= ((emptyFields & MOVES_DOWN_RIGHT_AVAILABLE) >> DOWN_RIGHT_SHIFT) & _blackPawns;
	jumpers |= (captBlackPawns >> BASE_DIAGONAL_SHIFT) & _whitePawns;

	// TODO: Consider if there is a need for analizing kings - there IS

    return jumpers;
}

// TODO: reconsider last row edge case
UINT Board::GetWhiteMovers()
{
	const UINT emptyFields = ~(_whitePawns | _blackPawns);
	const UINT whiteKings = _whitePawns & _kings;

	// Get the white pieces that can move in the basic diagonal direction (right down or left down, depending on the row)
	UINT movers = (emptyFields << BASE_DIAGONAL_SHIFT) & _whitePawns;
	
	// Get the white pieces that can move in the right down direction
	movers |= ((emptyFields & MOVES_UP_LEFT_AVAILABLE) << UP_LEFT_SHIFT) & _whitePawns;

	// Get the white pieces that can move in the left down direction
	movers |= ((emptyFields & MOVES_UP_RIGHT_AVAILABLE) << UP_RIGHT_SHIFT) & _whitePawns;

	// Get the white kings that can move in the upper diagonal direction (right up or left up)
	if (whiteKings)
	{
		movers |= (emptyFields >> BASE_DIAGONAL_SHIFT) & whiteKings;
		movers |= ((emptyFields & MOVES_DOWN_RIGHT_AVAILABLE) >> DOWN_RIGHT_SHIFT) & whiteKings;
		movers |= ((emptyFields & MOVES_DOWN_LEFT_AVAILABLE) >> DOWN_LEFT_SHIFT) & whiteKings;
	}

	return movers;
}

void Board::AddWhiteMove(std::queue<Board>& availableMoves, UINT src, UINT dst)
{
	// Adding the move to the board
    UINT newWhitePawns = _whitePawns & ~src;
    newWhitePawns |= dst;
    UINT newKings = _kings;

    if (src & _kings)
    {
        newKings = _kings & ~src;
        newKings |= dst;
    }
    else if (dst & WHITE_CROWNING)
        newKings |= dst;

    availableMoves.push(Board(newWhitePawns, _blackPawns, newKings));
}

void Board::GenerateWhiteKingBasicMoves(std::queue<Board>& availableMoves, UINT position)
{
    UINT empty_fields = ~(_whitePawns | _blackPawns);
	UINT newPosition = position;
    int iteration = 0;

	while (newPosition)
	{
        if (iteration & 1)
        {
            if (newPosition & MOVES_DOWN_LEFT_AVAILABLE)
                newPosition >>= DOWN_LEFT_SHIFT;
            else if (newPosition & MOVES_DOWN_RIGHT_AVAILABLE)
                newPosition >>= DOWN_RIGHT_SHIFT;
            else
                break;
        }
        else
        {
            newPosition >>= BASE_DIAGONAL_SHIFT;
        }

		if (!(newPosition & empty_fields))
			break;

        iteration++;
		AddWhiteMove(availableMoves, position, newPosition);
	}

    newPosition = position;
    iteration = 0;
    while (newPosition)
    {
        if (!(iteration & 1))
        {
            if (newPosition & MOVES_DOWN_LEFT_AVAILABLE)
                newPosition >>= DOWN_LEFT_SHIFT;
            else if (newPosition & MOVES_DOWN_RIGHT_AVAILABLE)
                newPosition >>= DOWN_RIGHT_SHIFT;
            else
                break;
        }
        else
        {
            newPosition >>= BASE_DIAGONAL_SHIFT;
        }
        if (!(newPosition & empty_fields))
            break;
        iteration++;
        AddWhiteMove(availableMoves, position, newPosition);
    }

	newPosition = position;
	iteration = 0;
    while (newPosition)
    {
        if (iteration & 1)
        {
            if (newPosition & MOVES_UP_LEFT_AVAILABLE)
                newPosition <<= UP_LEFT_SHIFT;
            else if (newPosition & MOVES_UP_RIGHT_AVAILABLE)
                newPosition <<= UP_RIGHT_SHIFT;
            else
                break;
        }
        else
        {
            newPosition <<= BASE_DIAGONAL_SHIFT;
        }
        if (!(newPosition & empty_fields))
            break;
        iteration++;
        AddWhiteMove(availableMoves, position, newPosition);
    }

    newPosition = position;
    iteration = 0;
    while (newPosition)
    {
        if (!(iteration & 1))
        {
            if (newPosition & MOVES_UP_LEFT_AVAILABLE)
                newPosition <<= UP_LEFT_SHIFT;
            else if (newPosition & MOVES_UP_RIGHT_AVAILABLE)
                newPosition <<= UP_RIGHT_SHIFT;
            else
                break;
        }
        else
        {
            newPosition <<= BASE_DIAGONAL_SHIFT;
        }
        if (!(newPosition & empty_fields))
            break;
        iteration++;
        AddWhiteMove(availableMoves, position, newPosition);
    }
}

void Board::GetWhiteAvailableMoves(std::queue<Board>& availableMoves)
{
	// Find all of the jumpers
    UINT jumpers = GetWhiteJumpers();

    if (jumpers)
    {
        // Generate moves for jumpers
    }
    else
    {
        // Generate moves for movers
		UINT whiteMovers = GetWhiteMovers();
		UINT empty_fields = ~(_whitePawns | _blackPawns);

        while (whiteMovers)
        {
            // Get the first bit set
			UINT mover = whiteMovers & -whiteMovers;  
            
            // Clear the bit from the bitboard
            whiteMovers ^= mover;  

            // King moves generation, without capturing enemy pieces
            if (mover & _kings)
            {
				GenerateWhiteKingBasicMoves(availableMoves, mover);
            }
            else
            {
                // Generate moves in the base diagonal direction
                if ((mover >> BASE_DIAGONAL_SHIFT) & empty_fields)
                {
                    AddWhiteMove(availableMoves, mover, mover >> BASE_DIAGONAL_SHIFT);
                }

                if (mover & MOVES_DOWN_LEFT_AVAILABLE)
                {
                    // Generate moves in the down left direction
                    if ((mover >> DOWN_LEFT_SHIFT) & empty_fields)
                    {
                        AddWhiteMove(availableMoves, mover, mover >> DOWN_LEFT_SHIFT);
                    }
                }

                if (mover & MOVES_DOWN_RIGHT_AVAILABLE)
                {
                    // Generate moves in the down right direction
                    if ((mover >> DOWN_RIGHT_SHIFT) & empty_fields)
                    {
                        AddWhiteMove(availableMoves, mover, mover >> DOWN_RIGHT_SHIFT);
                    }
                }
            }            
        }
    }
}

void Board::PrintBoard()
{
    std::cout << "   A B C D E F G H\n";
    std::cout << "  -----------------\n";
    for (int row = 0; row < 8; row++) {
        std::cout << 8 - row << "| ";
        for (int col = 0; col < 8; col++) {
            // Only dark squares can have pieces
            bool isDarkSquare = (row + col) % 2 != 0;
            if (!isDarkSquare) {
                std::cout << "  ";  // Light square - always empty
                continue;
            }

            // Calculate bit position for dark squares (bottom to top, left to right)
            int darkSquareNumber = (7 - row) * 4 + (col / 2);
            UINT mask = 1U << darkSquareNumber;

            char piece = ' ';  // Empty dark square

            // Check if square has a piece
            if (_whitePawns & mask) {
                piece = (_kings & mask) ? 'W' : 'w';
            }
            else if (_blackPawns & mask) {
                piece = (_kings & mask) ? 'B' : 'b';
            }

            // Print the piece
            std::cout << piece << ' ';
        }
        std::cout << "|" << 8 - row << '\n';
    }
    std::cout << "  -----------------\n";
    std::cout << "   A B C D E F G H\n\n";
}

void Board::PrintBitboard(UINT bitboard)
{
    std::cout << "   A B C D E F G H\n";
    std::cout << "  -----------------\n";
    for (int row = 0; row < 8; row++) {
        std::cout << 8 - row << "| ";
        for (int col = 0; col < 8; col++) {
            // Only dark squares are used
            bool isDarkSquare = (row + col) % 2 != 0;
            if (!isDarkSquare) {
                std::cout << "  ";  // Light square - always empty
                continue;
            }
            // Calculate bit position for dark squares
            int darkSquareNumber = (7 - row) * 4 + (col / 2);  // Bottom to top, left to right
            UINT mask = 1U << darkSquareNumber;  // No need to subtract from 31 anymore
            // Check if bit is set in the bitboard
            char piece = (bitboard & mask) ? '1' : '0';
            std::cout << piece << ' ';
        }
        std::cout << "|" << 8 - row;

        // Calculate row bits based on the new numbering scheme
        int startBit = (7 - row) * 4;  // Bottom row starts at bit 0
        UINT rowBits = (bitboard >> startBit) & 0xF;
        std::cout << "  " << std::hex << rowBits;
        std::cout << '\n';
    }
    std::cout << "  -----------------\n";
    std::cout << "   A B C D E F G H\n";
    printf("Full values: 0x%08X\n\n", bitboard);
}

void Board::PrintPossibleMoves(const std::queue<Board>& availableMoves)
{
    if (availableMoves.empty()) {
        std::cout << "No moves available!\n\n";
        return;
    }

    std::queue<Board> movesCopy = availableMoves;
    int moveNumber = 1;

    while (!movesCopy.empty()) {
        std::cout << "Move #" << moveNumber << ":\n";
        movesCopy.front().PrintBoard();
        movesCopy.pop();
        moveNumber++;
    }

    std::cout << "Total number of possible moves: " << moveNumber - 1 << "\n\n";
}
