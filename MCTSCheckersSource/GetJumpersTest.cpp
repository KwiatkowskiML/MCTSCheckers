#include "GetJumpersTest.h"

void GetJumpersTest::setUp()
{
    board.whitePawns = 0;
    board.blackPawns = 0;
    board.kings = 0;
}

bool GetJumpersTest::testCase(const char* testName, UINT expectedJumpers)
{
    UINT actualJumpers = moveGen.getJumpers(board, PieceColor::White);
    bool passed = (actualJumpers == expectedJumpers);

    printf("Test %s: %s\n", testName, passed ? "PASSED" : "FAILED");
    if (!passed) {
        printf("Expected: %u\n", expectedJumpers);
        printf("Actual: %u\n", actualJumpers);
    }
    return passed;
}

void GetJumpersTest::runAllTests()
{
    int passedTests = 0;
    int totalTests = 0;

    // Test 1: Simple forward capture down-right
    {
        setUp();
        board.whitePawns = 1ULL << 25;  // White pawn at position 25
        board.blackPawns = 1ULL << 21;  // Black pawn at position 21
        passedTests += testCase("Simple forward capture down-right", 1ULL << 25);
        totalTests++;
    }

    // Test 2: Simple forward capture down-left
    {
        setUp();
        board.whitePawns = 1ULL << 26;  // White pawn at position 26
        board.blackPawns = 1ULL << 21;  // Black pawn at position 21
        passedTests += testCase("Simple forward capture down-left", 1ULL << 26);
        totalTests++;
    }

    // Test 3: Multiple white pieces can capture
    {
        setUp();
        board.whitePawns = (1ULL << 29) | (1ULL << 28);  // White pawns at 29 and 28
        board.blackPawns = 1ULL << 25;  // Black pawn at position 25
        passedTests += testCase("Multiple white pieces can capture", (1ULL << 29) | (1ULL << 28));
        totalTests++;
    }

    // Test 4: No captures available
    {
        setUp();
        board.whitePawns = 1ULL << 24;  // White pawn at position 24
        board.blackPawns = 1ULL << 21;  // Black pawn not in capture position
        passedTests += testCase("No captures available", 0);
        totalTests++;
    }

    // Test 5: Capture blocked by another piece
    {
        setUp();
        board.whitePawns = 1ULL << 29;  // White pawn at position 29
        board.blackPawns = 1ULL << 25;  // Black pawn at position 25
        board.whitePawns |= 1ULL << 20; // Blocking white pawn at position 20
        passedTests += testCase("Capture blocked by another piece", 0);
        totalTests++;
    }

    // Test 6: Edge case - white pawn at edge
    {
        setUp();
        board.whitePawns = 1ULL << 28;   // White pawn at left edge
        board.blackPawns = 1ULL << 24;  // Black pawn
        passedTests += testCase("Edge case - left edge", 0);
        totalTests++;
    }

    // Test 7: Multiple capture directions available
    {
        setUp();
        board.whitePawns = 1ULL << 26;  // White pawn at position 26
        board.blackPawns = (1ULL << 21) | (1ULL << 22);  // Black pawns at 21 and 22
        passedTests += testCase("Multiple capture directions", 1ULL << 26);
        totalTests++;
    }

    // TODO: Add testing of king captures

    printf("\nTest Summary: %d/%d tests passed\n", passedTests, totalTests);
}
