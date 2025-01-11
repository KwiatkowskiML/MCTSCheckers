#include "MoveGenerationTest.h"

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
                if (expectedMove.source == actualMove.source &&
                    expectedMove.destination == actualMove.destination &&
                    expectedMove.captured == actualMove.captured) {
                    moveFound = true;
                    break;
                }
            }
            if (!moveFound) {
                printf("Test %s: FAILED\n", testName);
                printf("Missing move: source=%u, destination=%u, captured=%u\n",
                    expectedMove.source, expectedMove.destination, expectedMove.captured);
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

    // Test 1: Basic moves - no captures available
    {
        setUp();
        board.whitePawns = 1ULL << 25;  // White pawn at position 25
        MoveList expected = {
            Move(1ULL << 25, 1ULL << 20),  // Down-right move
            Move(1ULL << 25, 1ULL << 21)   // Down-left move
        };
        MoveList actual = moveGen.generateMoves(board, PieceColor::White);
        verifyMoveList("Basic moves - no captures", expected, actual);
    }

    // Test 2: Simple capture
    {
        setUp();
        board.whitePawns = 1ULL << 25;   // White pawn
        board.blackPawns = 1ULL << 21;   // Black pawn to capture
        MoveList expected = {
            Move(1ULL << 25, 1ULL << 18, 1ULL << 21)  // Capture move
        };
        MoveList actual = moveGen.generateMoves(board, PieceColor::White);
        verifyMoveList("Simple capture", expected, actual);
    }

    // Test 3: Multiple capture opportunities
    {
        setUp();
        board.whitePawns = 1ULL << 29;    // White pawn
        board.blackPawns = (1ULL << 25) | (1ULL << 17);  // Two black pawns to capture
        MoveList expected = {
            Move(1ULL << 29, 1ULL << 13, (1ULL << 25) | (1ULL << 17))  // Double capture
        };
        MoveList actual = moveGen.generateMoves(board, PieceColor::White);
        verifyMoveList("Multiple captures", expected, actual);
    }

    // Test 4: King basic moves
    {
        setUp();
        board.whitePawns = 1ULL << 21;  // White king
        board.kings = 1ULL << 21;       // Mark as king
        MoveList expected = {
            Move(1ULL << 21, 1ULL << 25, 0, true),  
            Move(1ULL << 21, 1ULL << 28, 0, true),  
            Move(1ULL << 21, 1ULL << 26, 0, true),  
            Move(1ULL << 21, 1ULL << 30, 0, true),   
            Move(1ULL << 21, 1ULL << 17, 0, true),   
            Move(1ULL << 21, 1ULL << 12, 0, true),   
            Move(1ULL << 21, 1ULL << 8, 0, true),   
            Move(1ULL << 21, 1ULL << 18, 0, true),   
            Move(1ULL << 21, 1ULL << 14, 0, true),   
            Move(1ULL << 21, 1ULL << 11, 0, true),   
            Move(1ULL << 21, 1ULL << 7, 0, true)   
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
            Move(1ULL << 18, 1ULL << 27, 1ULL << 22, true),
            Move(1ULL << 18, 1ULL << 31, 1ULL << 22, true),
            Move(1ULL << 18, 1ULL << 4, 1ULL << 9, true),
            Move(1ULL << 18, 1ULL, 1ULL << 9, true),
        };
        MoveList actual = moveGen.generateMoves(board, PieceColor::White);
        verifyMoveList("King single captures", expected, actual);
    }

    // Test 6: Multiple pieces with moves
    {
        setUp();
        board.whitePawns = (1ULL << 24) | (1ULL << 26);  // Two white pawns
        MoveList expected = {
            Move(1ULL << 24, 1ULL << 20),  // First pawn moves
            Move(1ULL << 26, 1ULL << 22),  // Second pawn moves
            Move(1ULL << 26, 1ULL << 21)   // Second pawn moves
        };
        MoveList actual = moveGen.generateMoves(board, PieceColor::White);
        verifyMoveList("Multiple pieces with moves", expected, actual);
    }

    // Test 7: Crowning move
    {
        setUp();
        board.whitePawns = 1ULL << 4;   // White pawn about to crown
        MoveList expected = {
            Move(1ULL << 4, 1ULL << 0, 0, true),  // Crowning move
            Move(1ULL << 4, 1ULL << 1, 0, true)   // Crowning move
        };
        MoveList actual = moveGen.generateMoves(board, PieceColor::White);
        verifyMoveList("Crowning move", expected, actual);
    }

    printSummary("Move Generation");
}
