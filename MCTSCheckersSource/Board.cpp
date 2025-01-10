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
    UINT captBlackPawns = (emptyFields << SHIFT_BASE) & _blackPawns;

	// Check whether previously specified black pawns can actually be captured
	if (captBlackPawns)
	{
		// Get the white pawns that can capture black pawn in the base diagonal direction
		jumpers |= ((captBlackPawns & MASK_L3) << SHIFT_L3) & _whitePawns;
		jumpers |= ((captBlackPawns & MASK_L5) << SHIFT_L5) & _whitePawns;
	}

	// Get the black pawns that might be captured in the other diagonal direction
	captBlackPawns = ((emptyFields & MASK_L3) << SHIFT_L3) & _blackPawns;
	captBlackPawns |= ((emptyFields & MASK_L5) << SHIFT_L5) & _blackPawns;

	jumpers |= (captBlackPawns << SHIFT_BASE) & _whitePawns;

	// Find all of the black pawns that might be captured backwards in base diagonal
    captBlackPawns = (emptyFields >> SHIFT_BASE) & _blackPawns;

    // Check whether previously specified black pawns can actually be captured
    if (captBlackPawns)
    {
		jumpers |= ((captBlackPawns & MASK_R5) >> SHIFT_R5) & _whitePawns;
		jumpers |= ((captBlackPawns & MASK_R3) >> SHIFT_R3) & _whitePawns;
    }

	// Find all of the black pawns that might be captured backwards in the other diagonal
	captBlackPawns = ((emptyFields & MASK_R5) >> SHIFT_R5) & _blackPawns;
	captBlackPawns |= ((emptyFields & MASK_R3) >> SHIFT_R3) & _blackPawns;
	jumpers |= (captBlackPawns >> SHIFT_BASE) & _whitePawns;

	// TODO: Consider if there is a need for analizing kings - there IS

    return jumpers;
}

// TODO: reconsider last row edge case
UINT Board::GetWhiteMovers()
{
	const UINT emptyFields = ~(_whitePawns | _blackPawns);
	const UINT whiteKings = _whitePawns & _kings;

	// Get the white pieces that can move in the basic diagonal direction (right down or left down, depending on the row)
	UINT movers = (emptyFields << SHIFT_BASE) & _whitePawns;
	
	// Get the white pieces that can move in the right down direction
	movers |= ((emptyFields & MASK_L3) << SHIFT_L3) & _whitePawns;

	// Get the white pieces that can move in the left down direction
	movers |= ((emptyFields & MASK_L5) << SHIFT_L5) & _whitePawns;

	// Get the white kings that can move in the upper diagonal direction (right up or left up)
	if (whiteKings)
	{
		movers |= (emptyFields >> SHIFT_BASE) & whiteKings;
		movers |= ((emptyFields & MASK_R3) >> SHIFT_R3) & whiteKings;
		movers |= ((emptyFields & MASK_R5) >> SHIFT_R5) & whiteKings;
	}

	return movers;
}

void Board::AddWhiteBasicMove(std::queue<Board>& availableMoves, UINT src, UINT dst)
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

void Board::AddWhiteCapturingMove(std::queue<Board>& availableMoves, UINT src, UINT captured, UINT dst)
{
	// Adding the move to the board
    UINT newWhitePawns = _whitePawns & ~src;
    newWhitePawns |= dst;
    UINT newKings = _kings;

	// Handing the case when the pawn becomes a king, or the king is moved
    if (src & _kings)
    {
        newKings = _kings & ~src;
        newKings |= dst;
    }
    else if (dst & WHITE_CROWNING)
        newKings |= dst;

	// Removing the captured piece from the board
	UINT newBlackPawns = _blackPawns & ~captured;
	Board newBoard(newWhitePawns, newBlackPawns, newKings);
    
	// Capturing continuation
    UINT newJumpers = newBoard.GetWhiteJumpers();
	if (newJumpers & dst)
	{
		// Continue capturing
		std::queue<Board> capturingMovesContinuation;
		newBoard.GenerateWhitePawnCapturingMoves(capturingMovesContinuation, dst);

		while (!capturingMovesContinuation.empty())
		{
			availableMoves.push(capturingMovesContinuation.front());
			capturingMovesContinuation.pop();
		}
	}
	else
	{
		availableMoves.push(newBoard);
	}
}

// TODO: add capturing moves
void Board::GenerateKingBasicMoves(std::queue<Board>& availableMoves, UINT position)
{
    UINT empty_fields = ~(_whitePawns | _blackPawns);
	UINT newPosition = position;
    int iteration = 0;

	while (newPosition)
	{
        if (iteration & 1)
        {
            if (newPosition & MASK_R5)
                newPosition >>= SHIFT_R5;
            else if (newPosition & MASK_R3)
                newPosition >>= SHIFT_R3;
            else
                break;
        }
        else
        {
            newPosition >>= SHIFT_BASE;
        }

		if (!(newPosition & empty_fields))
			break;

        iteration++;
		AddWhiteBasicMove(availableMoves, position, newPosition);
	}

    newPosition = position;
    iteration = 0;
    while (newPosition)
    {
        if (!(iteration & 1))
        {
            if (newPosition & MASK_R5)
                newPosition >>= SHIFT_R5;
            else if (newPosition & MASK_R3)
                newPosition >>= SHIFT_R3;
            else
                break;
        }
        else
        {
            newPosition >>= SHIFT_BASE;
        }
        if (!(newPosition & empty_fields))
            break;
        iteration++;
        AddWhiteBasicMove(availableMoves, position, newPosition);
    }

	newPosition = position;
	iteration = 0;
    while (newPosition)
    {
        if (iteration & 1)
        {
            if (newPosition & MASK_L3)
                newPosition <<= SHIFT_L3;
            else if (newPosition & MASK_L5)
                newPosition <<= SHIFT_L5;
            else
                break;
        }
        else
        {
            newPosition <<= SHIFT_BASE;
        }
        if (!(newPosition & empty_fields))
            break;
        iteration++;
        AddWhiteBasicMove(availableMoves, position, newPosition);
    }

    newPosition = position;
    iteration = 0;
    while (newPosition)
    {
        if (!(iteration & 1))
        {
            if (newPosition & MASK_L3)
                newPosition <<= SHIFT_L3;
            else if (newPosition & MASK_L5)
                newPosition <<= SHIFT_L5;
            else
                break;
        }
        else
        {
            newPosition <<= SHIFT_BASE;
        }
        if (!(newPosition & empty_fields))
            break;
        iteration++;
        AddWhiteBasicMove(availableMoves, position, newPosition);
    }
}

void Board::GenerateWhitePawnBasicMoves(std::queue<Board>& availableMoves, UINT position)
{
    UINT empty_fields = ~(_whitePawns | _blackPawns);

    // Generate moves in the base diagonal direction
    if ((position >> SHIFT_BASE) & empty_fields)
    {
        AddWhiteBasicMove(availableMoves, position, position >> SHIFT_BASE);
    }

    if (position & MASK_R5)
    {
        // Generate moves in the down left direction
        if ((position >> SHIFT_R5) & empty_fields)
        {
            AddWhiteBasicMove(availableMoves, position, position >> SHIFT_R5);
        }
    }

    if (position & MASK_R3)
    {
        // Generate moves in the down right direction
        if ((position >> SHIFT_R3) & empty_fields)
        {
            AddWhiteBasicMove(availableMoves, position, position >> SHIFT_R3);
        }
    }
}

void Board::GenerateWhitePawnCapturingMoves(std::queue<Board>& availableMoves, UINT position)
{
	UINT empty_fields = ~(_whitePawns | _blackPawns);

	//--------------------------------------------------------------------------------
	// Capturing black pawns below the white pawn
	//--------------------------------------------------------------------------------
    
	// Generate capturing moves in the base down diagonal direction
    UINT newPosition = position >> SHIFT_BASE;
	if (newPosition & _blackPawns)
	{
		UINT captured = newPosition;
		if (newPosition & MASK_R5)
		{
			newPosition >>= SHIFT_R5;
			if (newPosition & empty_fields)
                AddWhiteCapturingMove(availableMoves, position, captured, newPosition);
		}
		else if (newPosition & MASK_R3)
		{
			newPosition >>= SHIFT_R3;
			if (newPosition & empty_fields)
                AddWhiteCapturingMove(availableMoves, position, captured, newPosition);
		}
	}

	// Generate capturing moves in the down left direction
	if (position & MASK_R5)
	{
		newPosition = position >> SHIFT_R5;
		if (newPosition & _blackPawns)
		{
            UINT captured = newPosition;
			newPosition >>= SHIFT_BASE;
			if (newPosition & empty_fields)
                AddWhiteCapturingMove(availableMoves, position, captured, newPosition);
		}
	}

	// Generate capturing moves in the down right direction
	if (position & MASK_R3)
	{
		newPosition = position >> SHIFT_R3;
		if (newPosition & _blackPawns)
		{
            UINT captured = newPosition;
			newPosition >>= SHIFT_BASE;
			if (newPosition & empty_fields)
                AddWhiteCapturingMove(availableMoves, position, captured, newPosition);
		}
	}

	//--------------------------------------------------------------------------------
	// Capturing black pawns above the white pawn
	//--------------------------------------------------------------------------------

	// Generate capturing moves in the base upper diagonal direction
	newPosition = position << SHIFT_BASE;
	if (newPosition & _blackPawns)
	{
        UINT captured = newPosition;
		if (newPosition & MASK_L3)
		{
			newPosition <<= SHIFT_L3;
			if (newPosition & empty_fields)
                AddWhiteCapturingMove(availableMoves, position, captured, newPosition);
		}
		else if (newPosition & MASK_L5)
		{
			newPosition <<= SHIFT_L5;
			if (newPosition & empty_fields)
                AddWhiteCapturingMove(availableMoves, position, captured, newPosition);
		}
	}

	// Generate capturing moves in the up left direction
	if (position & MASK_L3)
	{
		newPosition = position << SHIFT_L3;
		if (newPosition & _blackPawns)
		{
            UINT captured = newPosition;
			newPosition <<= SHIFT_BASE;
			if (newPosition & empty_fields)
                AddWhiteCapturingMove(availableMoves, position, captured, newPosition);
		}
	}

	// Generate capturing moves in the up right direction
    if (position & MASK_L5)
    {
        newPosition = position << SHIFT_L5;
        if (newPosition & _blackPawns)
        {
            UINT captured = newPosition;
            newPosition <<= SHIFT_BASE;
            if (newPosition & empty_fields)
                AddWhiteCapturingMove(availableMoves, position, captured, newPosition);
        }
    }
}

void Board::GetWhiteAvailableMoves(std::queue<Board>& availableMoves)
{
	// Find all of the jumpers
    UINT jumpers = GetWhiteJumpers();

    if (jumpers)
    {
        // Generate moves for jumpers
        while (jumpers)
        {
			// Get the first bit set
			UINT jumper = jumpers & -jumpers;

			// Clear the bit from the bitboard
			jumpers ^= jumper;

			// King moves generation, with capturing enemy pieces
            if (jumper & _kings)
            {
				// Generate king moves
			}
            else
            {
                // Generate white pawn moves
				GenerateWhitePawnCapturingMoves(availableMoves, jumper);
            }
        }
    }
    else
    {
        // Generate moves for movers
		UINT whiteMovers = GetWhiteMovers();

        while (whiteMovers)
        {
            // Get the first bit set
			UINT mover = whiteMovers & -whiteMovers;  
            
            // Clear the bit from the bitboard
            whiteMovers ^= mover;  

            // King moves generation, without capturing enemy pieces
            if (mover & _kings)
				GenerateKingBasicMoves(availableMoves, mover);
            else
				GenerateWhitePawnBasicMoves(availableMoves, mover);         
        }
    }
}

void Board::PrintBoard()
{
    std::cout << "     A   B   C   D   E   F   G   H\n";
    std::cout << "   +---+---+---+---+---+---+---+---+\n";
    for (int row = 0; row < 8; row++) {
        std::cout << " " << 8 - row << " |";
        for (int col = 0; col < 8; col++) {
            // Only dark squares can have pieces
            bool isDarkSquare = (row + col) % 2 != 0;
            if (!isDarkSquare) {
                std::cout << "   |";  // Light square - always empty with no vertical line
            }
            else {
                // Calculate bit position for dark squares (bottom to top, left to right)
                int darkSquareNumber = (7 - row) * 4 + (col / 2);
                UINT mask = 1U << darkSquareNumber;
                // Check if square has a piece
                if (_whitePawns & mask) {
                    std::cout << " " << (_kings & mask ? 'W' : 'w') << " |";
                }
                else if (_blackPawns & mask) {
                    std::cout << " " << (_kings & mask ? 'B' : 'b') << " |";
                }
                else {
                    std::cout << "   |";  // Empty dark square
                }
            }
        }
        std::cout << " " << 8 - row << '\n';
        std::cout << "   +---+---+---+---+---+---+---+---+\n";  // Horizontal separator after each row
    }
    std::cout << "     A   B   C   D   E   F   G   H\n\n";
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

	printf("Total number of possible moves: %d\n\n", moveNumber - 1);
}
