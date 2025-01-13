#include "Board.h"
#include <cassert>
#include "MoveGenerator.h"
#include <random>

#define DEBUG

//----------------------------------------------------------------
// Move generation
//----------------------------------------------------------------
MoveList Board::getAvailableMoves(PieceColor color) const
{
    return MoveGenerator::generateMoves(_pieces, color);
}

Board Board::getBoardAfterMove(const Move& move) const
{
	BitBoard newBitboard = move.getBitboardAfterMove(_pieces);
	UINT newWhitePawns = newBitboard.getPieces(PieceColor::White);
	UINT newBlackPawns = newBitboard.getPieces(PieceColor::Black);
	UINT newKings = newBitboard.kings;

	Board newBoard(newWhitePawns, newBlackPawns, newKings);
    return newBoard;
}

//----------------------------------------------------------------
// Simulation
//----------------------------------------------------------------
int Board::simulateGame(PieceColor color) const
{
    PieceColor currentMoveColor = color;
	Board newBoard = *this;

    int noCaptureMoves = 0;

    while (true)
    {
        MoveList moves = newBoard.getAvailableMoves(currentMoveColor);

        // No moves available - game is over
        if (moves.empty()) {
            return currentMoveColor == color ? LOOSE : WIN;
        }

		// Check if the no capture moves limit has beeen exceeded
		if (noCaptureMoves >= MAX_NO_CAPTURE_MOVES) {
			return DRAW;
		}

        // Random number generation
        std::random_device rd; // Seed
        std::mt19937 gen(rd()); // Mersenne Twister engine
        std::uniform_int_distribution<> dist(0, moves.size() - 1);

        // Select a random move
        int randomIndex = dist(gen);
        Move randomMove = moves[randomIndex];

		// Check if the move is a capture
		if (!randomMove.isCapture()) {
			noCaptureMoves++;
		}
		else {
			noCaptureMoves = 0;
		}

        newBoard = newBoard.getBoardAfterMove(randomMove);
        currentMoveColor = getEnemyColor(currentMoveColor);

#ifdef DEBUG
        std::cout << "Chosen move: " << randomMove.toString() << std::endl;
		newBoard.printBoard();
#endif // DEBUG       
    }
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
