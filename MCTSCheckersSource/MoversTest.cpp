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

    // Test 1: Simple move down-right
    {
        setUp();
        board.whitePawns = 1ULL << 24;  // White pawn at position 24
        testMover("Simple move down-right", 1ULL << 24);
    }

    // Test 2: Simple move down-left
    {
        setUp();
        board.whitePawns = 1ULL << 27;  // White pawn at position 27
        testMover("Simple move down-left", 1ULL << 27);
    }

    // Test 3: Multiple pieces can move
    {
        setUp();
        board.whitePawns = (1ULL << 24) | (1ULL << 26);  // White pawns at 24 and 26
        testMover("Multiple pieces can move", (1ULL << 24) | (1ULL << 26));
    }

    // Test 4: No moves available (blocked)
    {
        setUp();
        board.whitePawns = 1ULL << 24;  // White pawn at position 24
        board.blackPawns = 1ULL << 20;  // Blocking black pawn
        testMover("No moves available (blocked)", 0);
    }

    // Test 5: King moves
    {
        setUp();
        board.whitePawns = 1ULL << 24;  // White king at position 24
        board.kings = 1ULL << 24;       // Mark as king
        testMover("King moves all directions", 1ULL << 24);
    }

    // Test 7: King multiple directions
    {
        setUp();
        board.whitePawns = 1ULL << 24;  // White king at position 24
        board.kings = 1ULL << 24;       // Mark as king
		board.blackPawns = 1ULL << 20;  // Blocking black pawn
        testMover("King multiple directions", 1ULL << 24);
    }

    printSummary("Movers");
}
