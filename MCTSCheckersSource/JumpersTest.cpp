#include "JumpersTest.h"

bool JumpersTest::testJumper(const char* testName, UINT expectedJumpers)
{
    UINT actualJumpers = moveGen.getAllJumpers(board, PieceColor::White);
    totalTests++;
    if (verifyTest(testName, expectedJumpers, actualJumpers)) {
        passedTests++;
        return true;
    }
    return false;
}

void JumpersTest::runAllTests()
{
    // Basic single captures
    {
        setUp();
        board.whitePawns = 1ULL << 25;  // White pawn at position 25
        board.blackPawns = 1ULL << 21;  // Black pawn at position 21
        testJumper("Basic capture right diagonal", 1ULL << 25);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 26;  // White pawn at position 26
        board.blackPawns = 1ULL << 21;  // Black pawn at position 21
        testJumper("Basic capture left diagonal", 1ULL << 26);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 18;  // White pawn at position 18
        board.blackPawns = 1ULL << 22;  // Black pawn at position 22
        testJumper("Basic capture backwards", 1ULL << 18);
    }

    // Multiple capture opportunities
    {
        setUp();
        board.whitePawns = 1ULL << 26;  // White pawn at position 26
        board.blackPawns = (1ULL << 21) | (1ULL << 22);  // Black pawns at 21 and 22
        testJumper("Multiple capture directions available", 1ULL << 26);
    }
    {
        setUp();
        board.whitePawns = (1ULL << 29) | (1ULL << 28);  // White pawns at 29 and 28
        board.blackPawns = 1ULL << 25;  // Black pawn at position 25
        testJumper("Multiple pieces can capture same target", (1ULL << 29) | (1ULL << 28));
    }

    // Chain capture opportunities
    {
        setUp();
        board.whitePawns = 1ULL << 29;  // White pawn starting position
        board.blackPawns = (1ULL << 25) | (1ULL << 17);  // Black pawns in chain
        testJumper("Chain capture opportunity", 1ULL << 29);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 26;  // White pawn
        board.blackPawns = (1ULL << 21) | (1ULL << 14) | (1ULL << 13);  // Multiple black pawns in chain
        testJumper("Long chain capture opportunity", 1ULL << 26);
    }

    // Edge cases and board boundaries
    {
        setUp();
        board.whitePawns = 1ULL << 28;  // White pawn at left edge
        board.blackPawns = 1ULL << 24;  // Black pawn
        testJumper("Left edge - no capture possible", 0);
    }

    // Blocked captures
    {
        setUp();
        board.whitePawns = 1ULL << 29;  // White pawn
        board.blackPawns = 1ULL << 25;  // Black pawn to capture
        board.whitePawns |= 1ULL << 20; // Blocking white pawn
        testJumper("Capture blocked by friendly piece", 0);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 29;  // White pawn
        board.blackPawns = 1ULL << 25 | 1ULL << 20;  // Black pawn to capture + blocking
        testJumper("Capture blocked by enemy piece", 0);
    }

    // King captures
    {
        setUp();
        board.whitePawns = 1ULL << 17;  // White king at center
        board.kings = 1ULL << 17;       // Mark as king
        board.blackPawns = 1ULL << 13;  // Black pawn to capture
        testJumper("King basic capture", 1ULL << 17);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 17;  // White king
        board.kings = 1ULL << 17;
        board.blackPawns = (1ULL << 13) | (1ULL << 21);  // Multiple capture directions
        testJumper("King multiple capture directions", 1ULL << 17);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 18;  // White king at edge
        board.kings = 1ULL << 18;
        board.blackPawns = 1ULL << 25;  // Black pawn
        testJumper("King long distance capture backwards", 1ULL << 18);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 18;  // White king at edge
        board.kings = 1ULL << 18;
        board.blackPawns = 1ULL << 4;  // Black pawn
        testJumper("King long distance capture forwards", 1ULL << 18);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 18;  // White king at edge
        board.kings = 1ULL << 18;
        board.blackPawns = (1ULL << 4) | 1ULL;  // Black pawn
        testJumper("King long distance capture blocked", 0);
    }

    // Special edge cases
    {
        setUp();
        board.whitePawns = 1ULL << 25;  // White piece
        board.blackPawns = 0xFFFFFFFF & ~(1ULL << 18) & ~(1ULL << 25);  // All positions filled except capture landing
        testJumper("Only one valid capture path", 1ULL << 25);
    }

    printSummary("Jumpers");
}