#include "MoversTest.h"
#include "MoveGenerationGpu.cuh"

bool MoversTest::testMover(const char* testName, UINT expectedMovers, PieceColor color = PieceColor::White)
{
    UINT actualMovers = moveGen.getAllMovers(boardAfterMove, color);
    totalTests++;
    if (verifyTest(testName, expectedMovers, actualMovers)) {
        passedTests++;
        return true;
    }
    return false;
}

bool MoversTest::testMoverGpu(const char* testName, UINT expectedMovers, PieceColor color = PieceColor::White)
{
    UINT actualMovers = getAllMovers(boardAfterMove, color);
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
        testMover("Basic move right diagonal", 1ULL << 24);
        testMoverGpu("Basic move right diagonal", 1ULL << 24);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 27;  // White pawn at position 27
        testMover("Basic move left diagonal", 1ULL << 27);
        testMoverGpu("Basic move left diagonal", 1ULL << 27);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 25;  // White pawn at position 25
        testMover("Basic move both diagonals", 1ULL << 25);
        testMoverGpu("Basic move both diagonals", 1ULL << 25);
    }

    // Edge positions tests
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 28;  // White pawn at left edge
        testMover("Left edge pawn movement", 1ULL << 28);
        testMoverGpu("Left edge pawn movement", 1ULL << 28);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 31;  // White pawn at right edge
        testMover("Right edge pawn movement", 1ULL << 31);
        testMoverGpu("Right edge pawn movement", 1ULL << 31);
    }

    // Multiple pieces movement
    {
        setUp();
        boardAfterMove.whitePawns = (1ULL << 24) | (1ULL << 26);  // White pawns at 24 and 26
        testMover("Two pawns can move", (1ULL << 24) | (1ULL << 26));
        testMoverGpu("Two pawns can move", (1ULL << 24) | (1ULL << 26));
    }
    {
        setUp();
        boardAfterMove.whitePawns = (1ULL << 24) | (1ULL << 26) | (1ULL << 28);  // Three pawns
        testMover("Three pawns can move", (1ULL << 24) | (1ULL << 26) | (1ULL << 28));
        testMoverGpu("Three pawns can move", (1ULL << 24) | (1ULL << 26) | (1ULL << 28));
    }

    // Blocking tests
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 24;  // White pawn at position 24
        boardAfterMove.blackPawns = 1ULL << 20;  // Blocking black pawn
        testMover("Blocked by black pawn", 0);
        testMoverGpu("Blocked by black pawn", 0);
    }
    {
        setUp();
        boardAfterMove.whitePawns = (1ULL << 24) | (1ULL << 20);  // White pawns blocking each other
        testMover("Blocked by friendly pawn", 1ULL << 20);
        testMoverGpu("Blocked by friendly pawn", 1ULL << 20);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 25;  // White pawn at position 25
        boardAfterMove.blackPawns = 1ULL << 21;  // Blocking one diagonal
        testMover("Partially blocked pawn", 1ULL << 25);  // Can still move in other diagonal
        testMoverGpu("Partially blocked pawn", 1ULL << 25);  // Can still move in other diagonal
    }

    // King movement tests
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 17;  // White king at central position
        boardAfterMove.kings = 1ULL << 17;       // Mark as king
        testMover("King in center", 1ULL << 17);
        testMoverGpu("King in center", 1ULL << 17);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 28;  // White king at edge
        boardAfterMove.kings = 1ULL << 28;       // Mark as king
        testMover("King at edge", 1ULL << 28);
        testMoverGpu("King at edge", 1ULL << 28);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 17;  // White king
        boardAfterMove.kings = 1ULL << 17;       // Mark as king
        boardAfterMove.blackPawns = (1ULL << 13) | (1ULL << 12);  // Blocking some directions
        testMover("King partially blocked", 1ULL << 17);
        testMoverGpu("King partially blocked", 1ULL << 17);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 17;  // White king
        boardAfterMove.kings = 1ULL << 17;       // Mark as king
        boardAfterMove.blackPawns = (1ULL << 13) | (1ULL << 12) | (1ULL << 21) | (1ULL << 20);
        testMover("King fully blocked", 0);
        testMoverGpu("King fully blocked", 0);
    }

    // Mixed pieces tests
    {
        setUp();
        boardAfterMove.whitePawns = (1ULL << 24) | (1ULL << 17);  // Pawn and king
        boardAfterMove.kings = 1ULL << 17;       // Mark one as king
        testMover("Pawn and king can move", (1ULL << 24) | (1ULL << 17));
        testMoverGpu("Pawn and king can move", (1ULL << 24) | (1ULL << 17));
    }

    // Special edge cases
    {
        setUp();
        boardAfterMove.whitePawns = (1ULL << 24) | (1ULL << 25) | (1ULL << 26) | (1ULL << 27);  // Full row
        testMover("Full row of pawns", (1ULL << 24) | (1ULL << 25) | (1ULL << 26) | (1ULL << 27));
        testMoverGpu("Full row of pawns", (1ULL << 24) | (1ULL << 25) | (1ULL << 26) | (1ULL << 27));
    }
    {
        setUp();
        boardAfterMove.whitePawns = (1ULL << 28) | (1ULL << 31);  // Both corner pieces
        boardAfterMove.kings = (1ULL << 28) | (1ULL << 31);       // Both are kings
        testMover("Both corners kings", (1ULL << 28) | (1ULL << 31));
        testMoverGpu("Both corners kings", (1ULL << 28) | (1ULL << 31));
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 25;  // White pawn
        boardAfterMove.blackPawns = (1ULL << 20) | (1ULL << 21);  // Both diagonals blocked
        testMover("Completely blocked pawn", 0);
        testMoverGpu("Completely blocked pawn", 0);
    }

    // Black piece movement tests
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 8;  // Black pawn at position 8
        testMover("Black basic move right diagonal", 1ULL << 8, PieceColor::Black);
        testMoverGpu("Black basic move right diagonal", 1ULL << 8, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 9;  // Black pawn at position 9
        testMover("Black basic move both diagonals", 1ULL << 9, PieceColor::Black);
        testMoverGpu("Black basic move both diagonals", 1ULL << 9, PieceColor::Black);
    }

    // Black edge positions tests
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 4;  // Black pawn at left edge
        testMover("Black left edge pawn movement", 1ULL << 4, PieceColor::Black);
        testMoverGpu("Black left edge pawn movement", 1ULL << 4, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 7;  // Black pawn at right edge
        testMover("Black right edge pawn movement", 1ULL << 7, PieceColor::Black);
        testMoverGpu("Black right edge pawn movement", 1ULL << 7, PieceColor::Black);
    }

    // Black multiple pieces movement
    {
        setUp();
        boardAfterMove.blackPawns = (1ULL << 8) | (1ULL << 10);  // Black pawns at 8 and 10
        testMover("Two black pawns can move", (1ULL << 8) | (1ULL << 10), PieceColor::Black);
        testMoverGpu("Two black pawns can move", (1ULL << 8) | (1ULL << 10), PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = (1ULL << 8) | (1ULL << 9) | (1ULL << 10);  // Three black pawns
        testMover("Three black pawns can move", (1ULL << 8) | (1ULL << 9) | (1ULL << 10), PieceColor::Black);
        testMoverGpu("Three black pawns can move", (1ULL << 8) | (1ULL << 9) | (1ULL << 10), PieceColor::Black);
    }

    // Black blocking tests
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 8;  // Black pawn at position 8
        boardAfterMove.whitePawns = 1ULL << 12;  // Blocking white pawn
        testMover("Black pawn blocked by white pawn", 0, PieceColor::Black);
        testMoverGpu("Black pawn blocked by white pawn", 0, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = (1ULL << 8) | (1ULL << 12);  // Black pawns blocking each other
        testMover("Black pawn blocked by friendly pawn", 1ULL << 12, PieceColor::Black);
        testMoverGpu("Black pawn blocked by friendly pawn", 1ULL << 12, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 9;  // Black pawn at position 9
        boardAfterMove.whitePawns = 1ULL << 13;  // Blocking one diagonal
        testMover("Black pawn partially blocked", 1ULL << 9, PieceColor::Black);  // Can still move in other diagonal
        testMoverGpu("Black pawn partially blocked", 1ULL << 9, PieceColor::Black);  // Can still move in other diagonal
    }

    // Black king movement tests
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 17;  // Black king at central position
        boardAfterMove.kings = 1ULL << 17;       // Mark as king
        testMover("Black king in center", 1ULL << 17, PieceColor::Black);
        testMoverGpu("Black king in center", 1ULL << 17, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 4;  // Black king at edge
        boardAfterMove.kings = 1ULL << 4;       // Mark as king
        testMover("Black king at edge", 1ULL << 4, PieceColor::Black);
        testMoverGpu("Black king at edge", 1ULL << 4, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 17;  // Black king
        boardAfterMove.kings = 1ULL << 17;       // Mark as king
        boardAfterMove.whitePawns = (1ULL << 20) | (1ULL << 21);  // Blocking some directions
        testMover("Black king partially blocked", 1ULL << 17, PieceColor::Black);
        testMoverGpu("Black king partially blocked", 1ULL << 17, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 17;  // Black king
        boardAfterMove.kings = 1ULL << 17;       // Mark as king
        boardAfterMove.whitePawns = (1ULL << 13) | (1ULL << 12) | (1ULL << 21) | (1ULL << 20);
        testMover("Black king fully blocked", 0, PieceColor::Black);
        testMoverGpu("Black king fully blocked", 0, PieceColor::Black);
    }

    // Black mixed pieces tests
    {
        setUp();
        boardAfterMove.blackPawns = (1ULL << 8) | (1ULL << 17);  // Pawn and king
        boardAfterMove.kings = 1ULL << 17;       // Mark one as king
        testMover("Black pawn and king can move", (1ULL << 8) | (1ULL << 17), PieceColor::Black);
        testMoverGpu("Black pawn and king can move", (1ULL << 8) | (1ULL << 17), PieceColor::Black);
    }

    // Black special edge cases
    {
        setUp();
        boardAfterMove.blackPawns = (1ULL << 8) | (1ULL << 9) | (1ULL << 10) | (1ULL << 11);  // Full row
        testMover("Full row of black pawns", (1ULL << 8) | (1ULL << 9) | (1ULL << 10) | (1ULL << 11), PieceColor::Black);
        testMoverGpu("Full row of black pawns", (1ULL << 8) | (1ULL << 9) | (1ULL << 10) | (1ULL << 11), PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 9;  // Black pawn
        boardAfterMove.whitePawns = (1ULL << 13) | (1ULL << 12);  // Both diagonals blocked
        testMover("Black completely blocked pawn", 0, PieceColor::Black);
        testMoverGpu("Black completely blocked pawn", 0, PieceColor::Black);
    }

    // Interaction tests between black and white pieces
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 8;
        boardAfterMove.whitePawns = 1ULL << 24;
        testMover("Black and white pawns on board - black move", 1ULL << 8, PieceColor::Black);
        testMoverGpu("Black and white pawns on board - black move", 1ULL << 8, PieceColor::Black);
        testMover("Black and white pawns on board - white move", 1ULL << 24);
        testMoverGpu("Black and white pawns on board - white move", 1ULL << 24);
    }

    printSummary("Movers");
}