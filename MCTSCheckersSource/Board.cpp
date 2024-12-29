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
		//PrintBitboard(captBlackPawns);
  //      printf("before values: 0x%08X\n", captBlackPawns & MOVES_UP_LEFT_AVAILABLE);
		//PrintBitboard(captBlackPawns & MOVES_UP_LEFT_AVAILABLE);
  //      printf("after values: 0x%08X\n", (captBlackPawns & MOVES_UP_LEFT_AVAILABLE) << UP_LEFT_SHIFT);
		//PrintBitboard((captBlackPawns & MOVES_UP_LEFT_AVAILABLE) << UP_LEFT_SHIFT);
		//PrintBitboard(1 << 16);
		jumpers |= ((captBlackPawns & MOVES_UP_RIGHT_AVAILABLE) << UP_RIGHT_SHIFT) & _whitePawns;
	}

    return jumpers;
}

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

void Board::GetWhiteAvailableMoves(std::queue<Board>& availableMoves)
{
	// Find all of the moveable white pieces
	// Get the first index of the movable piece and mark it in the bitboard as 0
	// For each moveable piece, find all of the available moves
}

void Board::GetBlackAvailableMoves(std::queue<Board>& availableMoves)
{
	UINT blackMovers = 0;
	UINT blackJumpers = 0;


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
