#include "Board.h"
#include <cassert>
#include "MoveGenerator.h"

//----------------------------------------------------------------
// Move generation
//----------------------------------------------------------------
MoveList Board::getAvailableMoves(PieceColor color) const
{
    return MoveGenerator::generateMoves(_pieces, color);
}

Board Board::getBoardAfterMove(const Move& move) const
{
    // TODO: Implement for black pieces
    assert(move.getColor() == PieceColor::White);
	UINT source = move.getSource();
	UINT destination = move.getDestination();
	UINT captured = move.getCaptured();

    // Deleting the initial position of the moved piece
    UINT newWhitePawns = _pieces.whitePawns & ~source;
	UINT newBlackPawns = _pieces.blackPawns;
    UINT newKings = _pieces.kings;

    // Deleting captured pieces
	if (move.isCapture())
	{
		newBlackPawns = _pieces.blackPawns & ~captured;
		newKings = _pieces.kings & ~captured;
  	}

    // Adding new piece position
    newWhitePawns |= destination;

    // Handing the case when the pawn becomes a king, or the king is moved
    if (source & _pieces.kings)
    {
        newKings = _pieces.kings & ~source;
        newKings |= destination;
    }
    else if (destination & WHITE_CROWNING)
        newKings |= destination;

	Board newBoard(newWhitePawns, newBlackPawns, newKings);
    return newBoard;

	// TODO: consider capturing continuation here
}

//----------------------------------------------------------------
// Visualization
//----------------------------------------------------------------

void Board::printBoard() const
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
                if (_pieces.whitePawns & mask) {
                    std::cout << " " << (_pieces.kings & mask ? 'W' : 'w') << " |";
                }
                else if (_pieces.blackPawns & mask) {
                    std::cout << " " << (_pieces.kings & mask ? 'B' : 'b') << " |";
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

void Board::printBitboard(UINT bitboard)
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

//----------------------------------------------------------------
// Field positioning
//----------------------------------------------------------------

const std::unordered_map<UINT, std::string> Board::fieldMapping = {
    {1u, "a1"}, {1u << 1, "c1"}, {1u << 2, "e1"}, {1u << 3, "g1"},
    {1u << 4, "b2"}, {1u << 5, "d2"}, {1u << 6, "f2"}, {1u << 7, "h2"},
    {1u << 8, "a3"}, {1u << 9, "c3"}, {1u << 10, "e3"}, {1u << 11, "g3"},
    {1u << 12, "b4"}, {1u << 13, "d4"}, {1u << 14, "f4"}, {1u << 15, "h4"},
    {1u << 16, "a5"}, {1u << 17, "c5"}, {1u << 18, "e5"}, {1u << 19, "g5"},
    {1u << 20, "b6"}, {1u << 21, "d6"}, {1u << 22, "f6"}, {1u << 23, "h6"},
    {1u << 24, "a7"}, {1u << 25, "c7"}, {1u << 26, "e7"}, {1u << 27, "g7"},
    {1u << 28, "b8"}, {1u << 29, "d8"}, {1u << 30, "f8"}, {1u << 31, "h8"}
};
