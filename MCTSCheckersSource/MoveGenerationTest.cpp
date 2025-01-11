#include "MoveGenerationTest.h"
#include "Move2.h"

bool MoveGenerationTest::verifyMoveList(const char* testName, const MoveList& expected, const MoveList& actual)
{
    bool passed = true;

    if (expected.size() != actual.size()) {
        printf("Test %s: FAILED\n", testName);
        printf("Expected %zu moves, got %zu moves\n", expected.size(), actual.size());
        passed = false;
    }
    else {
        // Check if all expected moves are present
        for (const auto& expectedMove : expected) {
            bool moveFound = false;
            for (const auto& actualMove : actual) {
                if (expectedMove == actualMove) {
                    moveFound = true;
                    break;
                }
            }
            if (!moveFound) {
                printf("Test %s: FAILED\n", testName);
				std::cout << "Missing move: " << expectedMove.toString() << std::endl;
                passed = false;
                break;
            }
        }
    }

    if (passed) {
        printf("Test %s: PASSED\n", testName);
        passedTests++;
    }
    totalTests++;

    return passed;
}

void MoveGenerationTest::runAllTests()
{
    printf("\nRunning Move Generation Tests...\n");

    // Basic Pawn Moves
    {
        setUp();
        board.whitePawns = 1ULL << 25;  // White pawn at position 25
        MoveList expected = {
            Move2(1ULL << 25, 1ULL << 20),  // Down-right move
            Move2(1ULL << 25, 1ULL << 21)   // Down-left move
        };
        verifyMoveList("Basic pawn moves - center position", expected,
            moveGen.generateMoves(board, PieceColor::White));
    }
    {
        setUp();
        board.whitePawns = 1ULL << 7;  // White pawn at left edge
        MoveList expected = {
            Move2(1ULL << 7, 1ULL << 3)  // Only down-right possible
        };
        verifyMoveList("Basic pawn moves - left edge", expected,
            moveGen.generateMoves(board, PieceColor::White));
    }
    {
        setUp();
        board.whitePawns = 1ULL << 31;  // White pawn at right edge
        MoveList expected = {
            Move2(1ULL << 31, 1ULL << 27)  // Only down-left possible
        };
        verifyMoveList("Basic pawn moves - right edge", expected,
            moveGen.generateMoves(board, PieceColor::White));
    }

    // Single Captures
    {
        setUp();
        board.whitePawns = 1ULL << 25;   // White pawn
        board.blackPawns = 1ULL << 21;   // Black pawn to capture
        MoveList expected = {
            Move2(1ULL << 25, 1ULL << 18, 1ULL << 21)  // Capture move
        };
        verifyMoveList("Single capture - right diagonal", expected,
            moveGen.generateMoves(board, PieceColor::White));
    }
    {
        setUp();
        board.whitePawns = 1ULL << 26;   // White pawn
        board.blackPawns = 1ULL << 21;   // Black pawn to capture
        MoveList expected = {
            Move2(1ULL << 26, 1ULL << 17, 1ULL << 21)  // Capture move
        };
        verifyMoveList("Single capture - left diagonal", expected,
            moveGen.generateMoves(board, PieceColor::White));
    }

    // Chain Captures
    {
        setUp();
        board.whitePawns = 1ULL << 29;    // White pawn
        board.blackPawns = (1ULL << 25) | (1ULL << 17);  // Chain of black pawns
        MoveList expected = {
            Move2(1ULL << 29, 1ULL << 13, (1ULL << 25) | (1ULL << 17))  // Double capture
        };
        verifyMoveList("Chain capture - double", expected,
            moveGen.generateMoves(board, PieceColor::White));
    }
    {
        setUp();
        board.whitePawns = 1ULL << 29;    // White pawn
        board.blackPawns = (1ULL << 25) | (1ULL << 17) | (1ULL << 10);  // Triple chain
        MoveList expected = {
            Move2(1ULL << 29, 1ULL << 6, (1ULL << 25) | (1ULL << 17) | (1ULL << 10))  // Triple capture
        };
        verifyMoveList("Chain capture - triple", expected,
            moveGen.generateMoves(board, PieceColor::White));
    }
    {
        setUp();
        board.whitePawns = 1ULL << 21;    // White pawn
        board.blackPawns = (1ULL << 25) | (1ULL << 18) | (1ULL << 19) | (1ULL << 10) ;  // Triple chain
        MoveList expected = {
            Move2(1ULL << 21, 1ULL << 28, (1ULL << 25)),
            Move2(1ULL << 21, 1ULL << 23, (1ULL << 18) | (1ULL << 19)),
            Move2(1ULL << 21, 1ULL << 5, (1ULL << 18) | (1ULL << 10)),
        };
        verifyMoveList("Chain capture - multiple", expected,
            moveGen.generateMoves(board, PieceColor::White));
    }

    // Test 4: King basic moves
    {
        setUp();
        board.whitePawns = 1ULL << 21;  // White king
        board.kings = 1ULL << 21;       // Mark as king
        MoveList expected = {
            Move2(1ULL << 21, 1ULL << 25),  
            Move2(1ULL << 21, 1ULL << 28),  
            Move2(1ULL << 21, 1ULL << 26),  
            Move2(1ULL << 21, 1ULL << 30),   
            Move2(1ULL << 21, 1ULL << 17),   
            Move2(1ULL << 21, 1ULL << 12),   
            Move2(1ULL << 21, 1ULL << 8),   
            Move2(1ULL << 21, 1ULL << 18),   
            Move2(1ULL << 21, 1ULL << 14),   
            Move2(1ULL << 21, 1ULL << 11),   
            Move2(1ULL << 21, 1ULL << 7)   
        };
        MoveList actual = moveGen.generateMoves(board, PieceColor::White);
        verifyMoveList("King basic moves", expected, actual);
    }

    // Test 5: King single capture
    {
        setUp();
        board.whitePawns = 1ULL << 18;  // White king
        board.kings = 1ULL << 18;       // Mark as king
        board.blackPawns = (1ULL << 22) | (1ULL << 9);  // Black pawn to capture
        MoveList expected = {
            Move2(1ULL << 18, 1ULL << 4, (1ULL << 22) | (1ULL << 9)),
            Move2(1ULL << 18, 1ULL, (1ULL << 22) | (1ULL << 9)),
        };
        MoveList actual = moveGen.generateMoves(board, PieceColor::White);
        verifyMoveList("King single captures", expected, actual);
    }
    {
        setUp();
        board.whitePawns = 1ULL << 28;  // White king
        board.kings = 1ULL << 28;       // Mark as king
        board.blackPawns = (1ULL << 21) | (1ULL << 13) | (1ULL << 10) | (1ULL << 27) | (1ULL << 11);  // Black pawn to capture
        MoveList expected = {
            Move2(1ULL << 28, 1ULL << 9, (1ULL << 21) | (1ULL << 13)),
            Move2(1ULL << 28, 1ULL << 5, (1ULL << 21) | (1ULL << 10)),
            Move2(1ULL << 28, 1ULL << 31, (1ULL << 21) | (1ULL << 27)),
            Move2(1ULL << 28, 1ULL << 7, (1ULL << 21) | (1ULL << 11)),
        };
        MoveList actual = moveGen.generateMoves(board, PieceColor::White);
        verifyMoveList("King chain of captures", expected, actual);
    }

    // Test 6: Multiple pieces with moves
    {
        setUp();
        board.whitePawns = (1ULL << 24) | (1ULL << 26);  // Two white pawns
        MoveList expected = {
            Move2(1ULL << 24, 1ULL << 20),  // First pawn moves
            Move2(1ULL << 26, 1ULL << 22),  // Second pawn moves
            Move2(1ULL << 26, 1ULL << 21)   // Second pawn moves
        };
        MoveList actual = moveGen.generateMoves(board, PieceColor::White);
        verifyMoveList("Multiple pieces with moves", expected, actual);
    }

    // Test 7: Crowning move
    {
        // TODO: look into it
        setUp();
        board.whitePawns = 1ULL << 4;   // White pawn about to crown
        MoveList expected = {
            Move2(1ULL << 4, 1ULL << 0),  // Crowning move
            Move2(1ULL << 4, 1ULL << 1)   // Crowning move
        };
        MoveList actual = moveGen.generateMoves(board, PieceColor::White);
        verifyMoveList("Crowning move", expected, actual);
    }

    printSummary("Move Generation");
}
