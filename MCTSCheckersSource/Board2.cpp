#include "Board2.h"

//----------------------------------------------------------------
// Move generation
//----------------------------------------------------------------
std::vector<Board2> Board2::getAvailableMoves(PieceColor color) const
{
    return std::vector<Board2>();
}

//----------------------------------------------------------------
// Visualization
//----------------------------------------------------------------
void Board2::printBoard() const
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

void Board2::printBitboard(UINT bitboard)
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
