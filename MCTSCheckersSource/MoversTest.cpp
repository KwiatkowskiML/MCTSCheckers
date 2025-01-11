#include "MoversTest.h"

bool MoversTest::testMover(const char* testName, UINT expectedMovers)
{
    UINT actualMovers = moveGen.getAllMovers(board, PieceColor::White);
    totalTests++;
    if (verifyTest(testName, expectedMovers, actualMovers)) {
        passedTests++;
        return true;
    }
    return false;
}

void MoversTest::runAllTests()
{
    setUp();

    // Basic pawn movement tests
    {
        setUp();
        board.whitePawns = 1ULL << 24;  // White pawn at position 24
        testMover("Basic move right diagonal", 1ULL << 24);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 27;  // White pawn at position 27
        testMover("Basic move left diagonal", 1ULL << 27);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 25;  // White pawn at position 25
        testMover("Basic move both diagonals", 1ULL << 25);
    }

    // Edge positions tests
    {
        setUp();
        board.whitePawns = 1ULL << 28;  // White pawn at left edge
        testMover("Left edge pawn movement", 1ULL << 28);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 31;  // White pawn at right edge
        testMover("Right edge pawn movement", 1ULL << 31);
    }

    // Multiple pieces movement
    {
        setUp();
        board.whitePawns = (1ULL << 24) | (1ULL << 26);  // White pawns at 24 and 26
        testMover("Two pawns can move", (1ULL << 24) | (1ULL << 26));
    }
    {
        setUp();
        board.whitePawns = (1ULL << 24) | (1ULL << 26) | (1ULL << 28);  // Three pawns
        testMover("Three pawns can move", (1ULL << 24) | (1ULL << 26) | (1ULL << 28));
    }

    // Blocking tests
    {
        setUp();
        board.whitePawns = 1ULL << 24;  // White pawn at position 24
        board.blackPawns = 1ULL << 20;  // Blocking black pawn
        testMover("Blocked by black pawn", 0);
    }
    {
        setUp();
        board.whitePawns = (1ULL << 24) | (1ULL << 20);  // White pawns blocking each other
        testMover("Blocked by friendly pawn", 1ULL << 20);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 25;  // White pawn at position 25
        board.blackPawns = 1ULL << 21;  // Blocking one diagonal
        testMover("Partially blocked pawn", 1ULL << 25);  // Can still move in other diagonal
    }

    // King movement tests
    {
        setUp();
        board.whitePawns = 1ULL << 17;  // White king at central position
        board.kings = 1ULL << 17;       // Mark as king
        testMover("King in center", 1ULL << 17);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 28;  // White king at edge
        board.kings = 1ULL << 28;       // Mark as king
        testMover("King at edge", 1ULL << 28);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 17;  // White king
        board.kings = 1ULL << 17;       // Mark as king
        board.blackPawns = (1ULL << 13) | (1ULL << 12);  // Blocking some directions
        testMover("King partially blocked", 1ULL << 17);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 17;  // White king
        board.kings = 1ULL << 17;       // Mark as king
        board.blackPawns = (1ULL << 13) | (1ULL << 12) | (1ULL << 21) | (1ULL << 20);
        testMover("King fully blocked", 0);
    }

    // Mixed pieces tests
    {
        setUp();
        board.whitePawns = (1ULL << 24) | (1ULL << 17);  // Pawn and king
        board.kings = 1ULL << 17;       // Mark one as king
        testMover("Pawn and king can move", (1ULL << 24) | (1ULL << 17));
    }

    // Special edge cases
    {
        setUp();
        board.whitePawns = (1ULL << 24) | (1ULL << 25) | (1ULL << 26) | (1ULL << 27);  // Full row
        testMover("Full row of pawns", (1ULL << 24) | (1ULL << 25) | (1ULL << 26) | (1ULL << 27));
    }
    {
        setUp();
        board.whitePawns = (1ULL << 28) | (1ULL << 31);  // Both corner pieces
        board.kings = (1ULL << 28) | (1ULL << 31);       // Both are kings
        testMover("Both corners kings", (1ULL << 28) | (1ULL << 31));
    }
    {
        setUp();
        board.whitePawns = 1ULL << 25;  // White pawn
        board.blackPawns = (1ULL << 20) | (1ULL << 21);  // Both diagonals blocked
        testMover("Completely blocked pawn", 0);
    }

    printSummary("Movers");
}