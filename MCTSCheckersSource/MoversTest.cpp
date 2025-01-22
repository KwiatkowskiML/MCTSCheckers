#include "MoversTest.h"

bool MoversTest::testWhiteMover(const char* testName, UINT expectedMovers)
{
    UINT actualMovers = moveGen.getAllMovers(boardAfterMove, PieceColor::White);
    totalTests++;
    if (verifyTest(testName, expectedMovers, actualMovers)) {
        passedTests++;
        return true;
    }
    return false;
}

bool MoversTest::testBlackMover(const char* testName, UINT expectedMovers)
{
    UINT actualMovers = moveGen.getAllMovers(boardAfterMove, PieceColor::Black);
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
        boardAfterMove.whitePawns = 1ULL << 24;  // White pawn at position 24
        testWhiteMover("Basic move right diagonal", 1ULL << 24);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 27;  // White pawn at position 27
        testWhiteMover("Basic move left diagonal", 1ULL << 27);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 25;  // White pawn at position 25
        testWhiteMover("Basic move both diagonals", 1ULL << 25);
    }

    // Edge positions tests
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 28;  // White pawn at left edge
        testWhiteMover("Left edge pawn movement", 1ULL << 28);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 31;  // White pawn at right edge
        testWhiteMover("Right edge pawn movement", 1ULL << 31);
    }

    // Multiple pieces movement
    {
        setUp();
        boardAfterMove.whitePawns = (1ULL << 24) | (1ULL << 26);  // White pawns at 24 and 26
        testWhiteMover("Two pawns can move", (1ULL << 24) | (1ULL << 26));
    }
    {
        setUp();
        boardAfterMove.whitePawns = (1ULL << 24) | (1ULL << 26) | (1ULL << 28);  // Three pawns
        testWhiteMover("Three pawns can move", (1ULL << 24) | (1ULL << 26) | (1ULL << 28));
    }

    // Blocking tests
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 24;  // White pawn at position 24
        boardAfterMove.blackPawns = 1ULL << 20;  // Blocking black pawn
        testWhiteMover("Blocked by black pawn", 0);
    }
    {
        setUp();
        boardAfterMove.whitePawns = (1ULL << 24) | (1ULL << 20);  // White pawns blocking each other
        testWhiteMover("Blocked by friendly pawn", 1ULL << 20);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 25;  // White pawn at position 25
        boardAfterMove.blackPawns = 1ULL << 21;  // Blocking one diagonal
        testWhiteMover("Partially blocked pawn", 1ULL << 25);  // Can still move in other diagonal
    }

    // King movement tests
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 17;  // White king at central position
        boardAfterMove.kings = 1ULL << 17;       // Mark as king
        testWhiteMover("King in center", 1ULL << 17);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 28;  // White king at edge
        boardAfterMove.kings = 1ULL << 28;       // Mark as king
        testWhiteMover("King at edge", 1ULL << 28);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 17;  // White king
        boardAfterMove.kings = 1ULL << 17;       // Mark as king
        boardAfterMove.blackPawns = (1ULL << 13) | (1ULL << 12);  // Blocking some directions
        testWhiteMover("King partially blocked", 1ULL << 17);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 17;  // White king
        boardAfterMove.kings = 1ULL << 17;       // Mark as king
        boardAfterMove.blackPawns = (1ULL << 13) | (1ULL << 12) | (1ULL << 21) | (1ULL << 20);
        testWhiteMover("King fully blocked", 0);
    }

    // Mixed pieces tests
    {
        setUp();
        boardAfterMove.whitePawns = (1ULL << 24) | (1ULL << 17);  // Pawn and king
        boardAfterMove.kings = 1ULL << 17;       // Mark one as king
        testWhiteMover("Pawn and king can move", (1ULL << 24) | (1ULL << 17));
    }

    // Special edge cases
    {
        setUp();
        boardAfterMove.whitePawns = (1ULL << 24) | (1ULL << 25) | (1ULL << 26) | (1ULL << 27);  // Full row
        testWhiteMover("Full row of pawns", (1ULL << 24) | (1ULL << 25) | (1ULL << 26) | (1ULL << 27));
    }
    {
        setUp();
        boardAfterMove.whitePawns = (1ULL << 28) | (1ULL << 31);  // Both corner pieces
        boardAfterMove.kings = (1ULL << 28) | (1ULL << 31);       // Both are kings
        testWhiteMover("Both corners kings", (1ULL << 28) | (1ULL << 31));
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 25;  // White pawn
        boardAfterMove.blackPawns = (1ULL << 20) | (1ULL << 21);  // Both diagonals blocked
        testWhiteMover("Completely blocked pawn", 0);
    }

    // Black piece movement tests
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 8;  // Black pawn at position 8
        testBlackMover("Black basic move right diagonal", 1ULL << 8);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 9;  // Black pawn at position 9
        testBlackMover("Black basic move both diagonals", 1ULL << 9);
    }

    // Black edge positions tests
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 4;  // Black pawn at left edge
        testBlackMover("Black left edge pawn movement", 1ULL << 4);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 7;  // Black pawn at right edge
        testBlackMover("Black right edge pawn movement", 1ULL << 7);
    }

    // Black multiple pieces movement
    {
        setUp();
        boardAfterMove.blackPawns = (1ULL << 8) | (1ULL << 10);  // Black pawns at 8 and 10
        testBlackMover("Two black pawns can move", (1ULL << 8) | (1ULL << 10));
    }
    {
        setUp();
        boardAfterMove.blackPawns = (1ULL << 8) | (1ULL << 9) | (1ULL << 10);  // Three black pawns
        testBlackMover("Three black pawns can move", (1ULL << 8) | (1ULL << 9) | (1ULL << 10));
    }

    // Black blocking tests
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 8;  // Black pawn at position 8
        boardAfterMove.whitePawns = 1ULL << 12;  // Blocking white pawn
        testBlackMover("Black pawn blocked by white pawn", 0);
    }
    {
        setUp();
        boardAfterMove.blackPawns = (1ULL << 8) | (1ULL << 12);  // Black pawns blocking each other
        testBlackMover("Black pawn blocked by friendly pawn", 1ULL << 12);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 9;  // Black pawn at position 9
        boardAfterMove.whitePawns = 1ULL << 13;  // Blocking one diagonal
        testBlackMover("Black pawn partially blocked", 1ULL << 9);  // Can still move in other diagonal
    }

    // Black king movement tests
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 17;  // Black king at central position
        boardAfterMove.kings = 1ULL << 17;       // Mark as king
        testBlackMover("Black king in center", 1ULL << 17);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 4;  // Black king at edge
        boardAfterMove.kings = 1ULL << 4;       // Mark as king
        testBlackMover("Black king at edge", 1ULL << 4);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 17;  // Black king
        boardAfterMove.kings = 1ULL << 17;       // Mark as king
        boardAfterMove.whitePawns = (1ULL << 20) | (1ULL << 21);  // Blocking some directions
        testBlackMover("Black king partially blocked", 1ULL << 17);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 17;  // Black king
        boardAfterMove.kings = 1ULL << 17;       // Mark as king
        boardAfterMove.whitePawns = (1ULL << 13) | (1ULL << 12) | (1ULL << 21) | (1ULL << 20);
        testBlackMover("Black king fully blocked", 0);
    }

    // Black mixed pieces tests
    {
        setUp();
        boardAfterMove.blackPawns = (1ULL << 8) | (1ULL << 17);  // Pawn and king
        boardAfterMove.kings = 1ULL << 17;       // Mark one as king
        testBlackMover("Black pawn and king can move", (1ULL << 8) | (1ULL << 17));
    }

    // Black special edge cases
    {
        setUp();
        boardAfterMove.blackPawns = (1ULL << 8) | (1ULL << 9) | (1ULL << 10) | (1ULL << 11);  // Full row
        testBlackMover("Full row of black pawns", (1ULL << 8) | (1ULL << 9) | (1ULL << 10) | (1ULL << 11));
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 9;  // Black pawn
        boardAfterMove.whitePawns = (1ULL << 13) | (1ULL << 12);  // Both diagonals blocked
        testBlackMover("Black completely blocked pawn", 0);
    }

    // Interaction tests between black and white pieces
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 8;
        boardAfterMove.whitePawns = 1ULL << 24;
        testBlackMover("Black and white pawns on board - black move", 1ULL << 8);
        testWhiteMover("Black and white pawns on board - white move", 1ULL << 24);
    }

    printSummary("Movers");
}