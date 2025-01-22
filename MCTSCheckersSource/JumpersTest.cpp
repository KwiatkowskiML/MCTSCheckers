#include "JumpersTest.h"

bool JumpersTest::testJumper(const char* testName, UINT expectedJumpers, PieceColor playerColor)
{
    UINT actualJumpers = moveGen.getAllJumpers(boardAfterMove, playerColor);
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
        boardAfterMove.whitePawns = 1ULL << 25;  // White pawn at position 25
        boardAfterMove.blackPawns = 1ULL << 21;  // Black pawn at position 21
        testJumper("Basic capture right diagonal", 1ULL << 25);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 26;  // White pawn at position 26
        boardAfterMove.blackPawns = 1ULL << 21;  // Black pawn at position 21
        testJumper("Basic capture left diagonal", 1ULL << 26);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 18;  // White pawn at position 18
        boardAfterMove.blackPawns = 1ULL << 22;  // Black pawn at position 22
        testJumper("Basic capture backwards", 1ULL << 18);
    }

    // Multiple capture opportunities
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 26;  // White pawn at position 26
        boardAfterMove.blackPawns = (1ULL << 21) | (1ULL << 22);  // Black pawns at 21 and 22
        testJumper("Multiple capture directions available", 1ULL << 26);
    }
    {
        setUp();
        boardAfterMove.whitePawns = (1ULL << 29) | (1ULL << 28);  // White pawns at 29 and 28
        boardAfterMove.blackPawns = 1ULL << 25;  // Black pawn at position 25
        testJumper("Multiple pieces can capture same target", (1ULL << 29) | (1ULL << 28));
    }

    // Chain capture opportunities
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 29;  // White pawn starting position
        boardAfterMove.blackPawns = (1ULL << 25) | (1ULL << 17);  // Black pawns in chain
        testJumper("Chain capture opportunity", 1ULL << 29);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 26;  // White pawn
        boardAfterMove.blackPawns = (1ULL << 21) | (1ULL << 14) | (1ULL << 13);  // Multiple black pawns in chain
        testJumper("Long chain capture opportunity", 1ULL << 26);
    }

    // Edge cases and board boundaries
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 28;  // White pawn at left edge
        boardAfterMove.blackPawns = 1ULL << 24;  // Black pawn
        testJumper("Left edge - no capture possible", 0);
    }

    // Blocked captures
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 29;  // White pawn
        boardAfterMove.blackPawns = 1ULL << 25;  // Black pawn to capture
        boardAfterMove.whitePawns |= 1ULL << 20; // Blocking white pawn
        testJumper("Capture blocked by friendly piece", 0);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 29;  // White pawn
        boardAfterMove.blackPawns = 1ULL << 25 | 1ULL << 20;  // Black pawn to capture + blocking
        testJumper("Capture blocked by enemy piece", 0);
    }

    // King captures
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 17;  // White king at center
        boardAfterMove.kings = 1ULL << 17;       // Mark as king
        boardAfterMove.blackPawns = 1ULL << 13;  // Black pawn to capture
        testJumper("King basic capture", 1ULL << 17);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 17;  // White king
        boardAfterMove.kings = 1ULL << 17;
        boardAfterMove.blackPawns = (1ULL << 13) | (1ULL << 21);  // Multiple capture directions
        testJumper("King multiple capture directions", 1ULL << 17);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 18;  // White king at edge
        boardAfterMove.kings = 1ULL << 18;
        boardAfterMove.blackPawns = 1ULL << 25;  // Black pawn
        testJumper("King long distance capture backwards", 1ULL << 18);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 18;  // White king at edge
        boardAfterMove.kings = 1ULL << 18;
        boardAfterMove.blackPawns = 1ULL << 4;  // Black pawn
        testJumper("King long distance capture forwards", 1ULL << 18);
    }
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 18;  // White king at edge
        boardAfterMove.kings = 1ULL << 18;
        boardAfterMove.blackPawns = (1ULL << 4) | 1ULL;  // Black pawn
        testJumper("King long distance capture blocked", 0);
    }

    // Special edge cases
    {
        setUp();
        boardAfterMove.whitePawns = 1ULL << 25;  // White piece
        boardAfterMove.blackPawns = 0xFFFFFFFF & ~(1ULL << 18) & ~(1ULL << 25);  // All positions filled except capture landing
        testJumper("Only one valid capture path", 1ULL << 25);
    }

    // Basic black single captures
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 9;   // Black pawn at position 9
        boardAfterMove.whitePawns = 1ULL << 13;  // White pawn at position 13
        testJumper("Black basic capture right diagonal", 1ULL << 9, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 10;  // Black pawn at position 10
        boardAfterMove.whitePawns = 1ULL << 13;  // White pawn at position 13
        testJumper("Black basic capture left diagonal", 1ULL << 10, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 14;  // Black pawn at position 14
        boardAfterMove.whitePawns = 1ULL << 10;  // White pawn at position 10
        testJumper("Black basic capture backwards", 1ULL << 14, PieceColor::Black);
    }

    // Multiple black capture opportunities
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 10;  // Black pawn at position 10
        boardAfterMove.whitePawns = (1ULL << 13) | (1ULL << 14);  // White pawns at 13 and 14
        testJumper("Black multiple capture directions available", 1ULL << 10, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = (1ULL << 5) | (1ULL << 4);  // Black pawns at 5 and 4
        boardAfterMove.whitePawns = 1ULL << 9;  // White pawn at position 9
        testJumper("Black multiple pieces can capture same target", (1ULL << 5) | (1ULL << 4), PieceColor::Black);
    }

    // Black chain capture opportunities
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 5;  // Black pawn starting position
        boardAfterMove.whitePawns = (1ULL << 9) | (1ULL << 17);  // White pawns in chain
        testJumper("Black chain capture opportunity", 1ULL << 5, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 10;  // Black pawn
        boardAfterMove.whitePawns = (1ULL << 13) | (1ULL << 20) | (1ULL << 21);  // Multiple white pawns in chain
        testJumper("Black long chain capture opportunity", 1ULL << 10, PieceColor::Black);
    }

    // Black edge cases and board boundaries
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 4;  // Black pawn at left edge
        boardAfterMove.whitePawns = 1ULL << 8;  // White pawn
        testJumper("Black left edge - no capture possible", 0, PieceColor::Black);
    }

    // Blocked black captures
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 5;   // Black pawn
        boardAfterMove.whitePawns = 1ULL << 9;   // White pawn to capture
        boardAfterMove.blackPawns |= 1ULL << 12; // Blocking black pawn
        testJumper("Black capture blocked by friendly piece", 0, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 5;   // Black pawn
        boardAfterMove.whitePawns = 1ULL << 9 | 1ULL << 12;  // White pawn to capture + blocking
        testJumper("Black capture blocked by enemy piece", 0, PieceColor::Black);
    }

    // Black king captures
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 17;  // Black king at center
        boardAfterMove.kings = 1ULL << 17;       // Mark as king
        boardAfterMove.whitePawns = 1ULL << 21;  // White pawn to capture
        testJumper("Black king basic capture", 1ULL << 17, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 17;  // Black king
        boardAfterMove.kings = 1ULL << 17;
        boardAfterMove.whitePawns = (1ULL << 21) | (1ULL << 13);  // Multiple capture directions
        testJumper("Black king multiple capture directions", 1ULL << 17, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 11;  // Black king at edge
        boardAfterMove.kings = 1ULL << 11;
        boardAfterMove.whitePawns = 1ULL << 21;  // White pawn
        testJumper("Black king long distance capture forwards", 1ULL << 11, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 21;  // Black king at edge
        boardAfterMove.kings = 1ULL << 21;
        boardAfterMove.whitePawns = 1ULL << 11;  // White pawn
        testJumper("Black king long distance capture backwards", 1ULL << 21, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 11;  // Black king at edge
        boardAfterMove.kings = 1ULL << 11;
        boardAfterMove.whitePawns = 1ULL << 28;  // White pawn
        testJumper("Black king long distance capture not possible", 0, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 11;  // Black king at edge
        boardAfterMove.kings = 1ULL << 11;
        boardAfterMove.whitePawns = (1ULL << 25) | (1ULL << 21);  // White pawn
        testJumper("Black king long distance capture blocked by white", 0, PieceColor::Black);
    }
    {
        setUp();
        boardAfterMove.blackPawns = (1ULL << 11) | (1ULL << 25);  // Black king at edge
        boardAfterMove.kings = 1ULL << 11;
        boardAfterMove.whitePawns =  1ULL << 21;  // White pawn
        testJumper("Black king long distance capture blocked by black", 1ULL << 25, PieceColor::Black);
    }

    // Black special edge cases
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 9;   // Black piece
        boardAfterMove.whitePawns = 0xFFFFFFFF & ~(1ULL << 16) & ~(1ULL << 9);  // All positions filled except capture landing
        testJumper("Black only one valid capture path", 1ULL << 9, PieceColor::Black);
    }

    // Interaction tests between black and white jumpers
    {
        setUp();
        boardAfterMove.blackPawns = 1ULL << 9;
        boardAfterMove.whitePawns = 1ULL << 25;
        boardAfterMove.whitePawns |= 1ULL << 13;  // White pawn to be captured by black
        boardAfterMove.blackPawns |= 1ULL << 21;  // Black pawn to be captured by white
        testJumper("Black jumper in mixed position", (1ULL << 9) | (1ULL << 21), PieceColor::Black);
        testJumper("White jumper in mixed position", (1ULL << 25) | (1ULL << 13));
    }

    printSummary("Jumpers");
}
