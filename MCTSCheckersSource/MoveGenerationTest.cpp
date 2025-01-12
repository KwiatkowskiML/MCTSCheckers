#include "MoveGenerationTest.h"
#include "Move.h"
#include "Board.h"

bool MoveGenerationTest::verifyMoveList(const char* testName, const MoveList& expected, const MoveList& actual)
{
    bool passed = true;

    if (expected.size() != actual.size()) {
        printf("Test %s: FAILED\n", testName);
        printf("Expected %zu moves, got %zu moves\n", expected.size(), actual.size());
        passed = false;
    }
    else {
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
    else
    {
		printf("Expected moves: ");
		for (const auto& move : expected)
			std::cout << move.toString() << std::endl;
		printf("Actual moves: ");
		for (const auto& move : actual)
			std::cout << move.toString() << std::endl;
    }
    totalTests++;

    return passed;
}

void MoveGenerationTest::testBasicPawnMovesCenter()
{
    setUp();
    board.whitePawns = 1ULL << 25;
    MoveList expected = {
        Move(1ULL << 25, 1ULL << 20),
        Move(1ULL << 25, 1ULL << 21)
    };
    verifyMoveList("Basic pawn moves - center position", expected, moveGen.generateMoves(board, PieceColor::White));
}

void MoveGenerationTest::testBasicPawnMovesLeftEdge()
{
    setUp();
    board.whitePawns = 1ULL << 7;
    MoveList expected = {
        Move(1ULL << 7, 1ULL << 3)
    };
    verifyMoveList("Basic pawn moves - left edge", expected, moveGen.generateMoves(board, PieceColor::White));
}

void MoveGenerationTest::testBasicPawnMovesRightEdge()
{
    setUp();
    board.whitePawns = 1ULL << 31;
    MoveList expected = {
        Move(1ULL << 31, 1ULL << 27)
    };
    verifyMoveList("Basic pawn moves - right edge", expected, moveGen.generateMoves(board, PieceColor::White));
}

void MoveGenerationTest::testSingleCaptureRightDiagonal()
{
    setUp();
    board.whitePawns = 1ULL << 25;
    board.blackPawns = 1ULL << 21;
    MoveList expected = {
        Move(1ULL << 25, 1ULL << 18, 1ULL << 21)
    };
    verifyMoveList("Single capture - right diagonal", expected, moveGen.generateMoves(board, PieceColor::White));
}

void MoveGenerationTest::testSingleCaptureLeftDiagonal()
{
    setUp();
    board.whitePawns = 1ULL << 26;
    board.blackPawns = 1ULL << 21;
    MoveList expected = {
        Move(1ULL << 26, 1ULL << 17, 1ULL << 21)
    };
    verifyMoveList("Single capture - left diagonal", expected, moveGen.generateMoves(board, PieceColor::White));
}

void MoveGenerationTest::testChainCaptureDouble()
{
    setUp();
    board.whitePawns = 1ULL << 29;
    board.blackPawns = (1ULL << 25) | (1ULL << 17);
    MoveList expected = {
        Move({1ULL << 29, 1ULL << 20, 1ULL << 13}, (1ULL << 25) | (1ULL << 17))
    };
    verifyMoveList("Chain capture - double", expected, moveGen.generateMoves(board, PieceColor::White));
}

void MoveGenerationTest::testChainCaptureTriple()
{
    setUp();
    board.whitePawns = 1ULL << 29;
    board.blackPawns = (1ULL << 25) | (1ULL << 17) | (1ULL << 10);
    MoveList expected = {
        Move({1ULL << 29, 1ULL << 20, 1ULL << 13, 1ULL << 6}, (1ULL << 25) | (1ULL << 17) | (1ULL << 10))
    };
    verifyMoveList("Chain capture - triple", expected, moveGen.generateMoves(board, PieceColor::White));
}

void MoveGenerationTest::testChainCaptureMultiple()
{
    setUp();
    board.whitePawns = 0x90000000;
    board.blackPawns = (1ULL << 27) | (1ULL << 18) | (1ULL << 19) | (1ULL << 9) | (1ULL << 10) | (1ULL << 11);
    MoveList expected = {
        Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL << 4}, (1ULL << 27) | (1ULL << 18) | (1ULL << 9)),
		Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL << 6, 1ULL << 15, 1ULL << 22}, (1ULL << 27) | (1ULL << 18) | (1ULL << 10) | (1ULL << 11) | (1ULL << 19)),
		Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 13, 1ULL << 4}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 9)),
		Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 13, 1ULL << 22}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 18)),
    };
    verifyMoveList("Chain capture - multiple", expected, moveGen.generateMoves(board, PieceColor::White));
}

void MoveGenerationTest::testKingBasicMoves()
{
    setUp();
    board.whitePawns = 1ULL << 21;
    board.kings = 1ULL << 21;
    MoveList expected = {
        Move(1ULL << 21, 1ULL << 25),
        Move(1ULL << 21, 1ULL << 28),
        Move(1ULL << 21, 1ULL << 26),
        Move(1ULL << 21, 1ULL << 30),
        Move(1ULL << 21, 1ULL << 17),
        Move(1ULL << 21, 1ULL << 12),
        Move(1ULL << 21, 1ULL << 8),
        Move(1ULL << 21, 1ULL << 18),
        Move(1ULL << 21, 1ULL << 14),
        Move(1ULL << 21, 1ULL << 11),
        Move(1ULL << 21, 1ULL << 7)
    };
    verifyMoveList("King basic moves", expected, moveGen.generateMoves(board, PieceColor::White));
}

void MoveGenerationTest::testCrowningMove()
{
    setUp();
    board.whitePawns = 1ULL << 4;
    MoveList expected = {
        Move(1ULL << 4, 1ULL << 0),
        Move(1ULL << 4, 1ULL << 1)
    };
    verifyMoveList("Crowning move", expected, moveGen.generateMoves(board, PieceColor::White));
}

void MoveGenerationTest::runAllTests()
{
    testBasicPawnMovesCenter();
    testBasicPawnMovesLeftEdge();
    testBasicPawnMovesRightEdge();
    testSingleCaptureRightDiagonal();
    testSingleCaptureLeftDiagonal();
    testChainCaptureDouble();
    testChainCaptureTriple();
    testKingBasicMoves();
    testCrowningMove();

	testChainCaptureMultiple();

    printSummary("Move Generation");
}
