#include "JumpersTest.h"

bool JumpersTest::testJumper(const char* testName, UINT expectedJumpers)
{
    UINT actualJumpers = moveGen.getJumpers(board, PieceColor::White);
    totalTests++;
    if (verifyTest(testName, expectedJumpers, actualJumpers)) {
        passedTests++;
        return true;
    }
    return false;
}

void JumpersTest::runAllTests()
{

    // Test 1: Simple forward capture down-right
    {
        setUp();
        board.whitePawns = 1ULL << 25;  // White pawn at position 25
        board.blackPawns = 1ULL << 21;  // Black pawn at position 21
        testJumper("Simple forward capture down-right", 1ULL << 25);
    }

    // Test 2: Simple forward capture down-left
    {
        setUp();
        board.whitePawns = 1ULL << 26;  // White pawn at position 26
        board.blackPawns = 1ULL << 21;  // Black pawn at position 21
        testJumper("Simple forward capture down-left", 1ULL << 26);
    }

    // Test 3: Multiple white pieces can capture
    {
        setUp();
        board.whitePawns = (1ULL << 29) | (1ULL << 28);  // White pawns at 29 and 28
        board.blackPawns = 1ULL << 25;  // Black pawn at position 25
        testJumper("Multiple white pieces can capture", (1ULL << 29) | (1ULL << 28));
    }

    // Test 4: No captures available
    {
        setUp();
        board.whitePawns = 1ULL << 24;  // White pawn at position 24
        board.blackPawns = 1ULL << 21;  // Black pawn not in capture position
        testJumper("No captures available", 0);
    }

    // Test 5: Capture blocked by another piece
    {
        setUp();
        board.whitePawns = 1ULL << 29;  // White pawn at position 29
        board.blackPawns = 1ULL << 25;  // Black pawn at position 25
        board.whitePawns |= 1ULL << 20; // Blocking white pawn at position 20
        testJumper("Capture blocked by another piece", 0);
    }

    // Test 6: Edge case - white pawn at edge
    {
        setUp();
        board.whitePawns = 1ULL << 28;   // White pawn at left edge
        board.blackPawns = 1ULL << 24;  // Black pawn
        testJumper("Edge case - left edge", 0);
    }

    // Test 7: Multiple capture directions available
    {
        setUp();
        board.whitePawns = 1ULL << 26;  // White pawn at position 26
        board.blackPawns = (1ULL << 21) | (1ULL << 22);  // Black pawns at 21 and 22
        testJumper("Multiple capture directions", 1ULL << 26);
    }

    // TODO: Add testing of king captures

    printf("\nTest Summary: %d/%d tests passed\n", passedTests, totalTests);
}
