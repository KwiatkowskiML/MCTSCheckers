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
	return 0;
}

UINT Board::GetWhiteMovers()
{
	const UINT emptyFields = ~(_whitePieces | _blackPieces);
	const UINT whiteKings = _whitePieces & _kings;

	// Get the white pieces that can move in the basic diagonal direction (right down or left down, depending on the row)
	UINT movers = (emptyFields << BASE_DIAGONAL_SHIFT) & _whitePieces;
	
	// Get the white pieces that can move in the right down direction
	movers |= ((emptyFields & MOVES_UP_LEFT_AVAILABLE) << LEFT_UP_SHIFT) & _whitePieces;

	// Get the white pieces that can move in the left down direction
	movers |= ((emptyFields & MOVES_UP_RIGHT_AVAILABLE) << RIGHT_UP_SHIFT) & _whitePieces;

	// Get the white kings that can move in the upper diagonal direction (right up or left up)
	if (whiteKings)
	{
		movers |= (emptyFields >> BASE_DIAGONAL_SHIFT) & whiteKings;
		movers |= ((emptyFields & MOVES_DOWN_RIGHT_AVAILABLE) >> RIGHT_DOWN_SHIFT) & whiteKings;
		movers |= ((emptyFields & MOVES_DOWN_LEFT_AVAILABLE) >> LEFT_DOWN_SHIFT) & whiteKings;
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

            // Calculate bit position for dark squares
            // We need to map board position to bit position
            int darkSquareNumber = (row * 4) + (col / 2);  // Calculate which dark square this is
            UINT mask = 1U << (31 - darkSquareNumber);  // Map to bitboard position

            char piece = ' ';  // Empty dark square

            // Check if square has a piece
            if (_whitePieces & mask) {
                piece = (_kings & mask) ? 'W' : 'w';
            }
            else if (_blackPieces & mask) {
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
            int darkSquareNumber = (row * 4) + (col / 2);  // Which dark square (0-31)
            UINT mask = 1U << (31 - darkSquareNumber);     // Corresponding bit in the bitboard

            // Check if bit is set in the bitboard
            char piece = (bitboard & mask) ? '1' : '0';

            std::cout << piece << ' ';
        }

        std::cout << "|" << 8 - row;

        int startBit = 28 - (row * 4);
        UINT rowBits = (bitboard >> startBit) & 0xF;
        std::cout << "  " << std::hex << rowBits;

        std::cout << '\n';
    }

    std::cout << "  -----------------\n";
    std::cout << "   A B C D E F G H\n";
    printf("Full values: 0x%08X\n\n", bitboard);
}
