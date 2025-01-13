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
    verifyMoveList("Basic pawn moves - center position (White)", expected, moveGen.generateMoves(board, PieceColor::White));

    setUp();
    board.blackPawns = 1ULL << 9;  // Mirrored position for black
    MoveList expectedBlack = {
        Move(1ULL << 9, 1ULL << 13, 0, PieceColor::Black),
        Move(1ULL << 9, 1ULL << 12, 0, PieceColor::Black)
    };
    verifyMoveList("Basic pawn moves - center position (Black)", expectedBlack, moveGen.generateMoves(board, PieceColor::Black));
}

void MoveGenerationTest::testBasicPawnMovesLeftEdge()
{
    setUp();
    board.whitePawns = 1ULL << 7;
    MoveList expected = {
        Move(1ULL << 7, 1ULL << 3)
    };
    verifyMoveList("Basic pawn moves - left edge (White)", expected, moveGen.generateMoves(board, PieceColor::White));

    setUp();
    board.blackPawns = 1ULL << 7;  // Mirrored position for black
    MoveList expectedBlack = {
        Move(1ULL << 7, 1ULL << 11, 0, PieceColor::Black)
    };
    verifyMoveList("Basic pawn moves - left edge (Black)", expectedBlack, moveGen.generateMoves(board, PieceColor::Black));
}

void MoveGenerationTest::testBasicPawnMovesRightEdge()
{
    setUp();
    board.whitePawns = 1ULL << 31;
    MoveList expected = {
        Move(1ULL << 31, 1ULL << 27)
    };
    verifyMoveList("Basic pawn moves - right edge (White)", expected, moveGen.generateMoves(board, PieceColor::White));

    setUp();
    board.blackPawns = 1ULL;  // Mirrored position for black
    MoveList expectedBlack = {
        Move(1ULL, 1ULL << 4, 0, PieceColor::Black)
    };
    verifyMoveList("Basic pawn moves - right edge (Black)", expectedBlack, moveGen.generateMoves(board, PieceColor::Black));
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

    setUp();
    board.blackPawns = 1ULL << 9;
    board.whitePawns = 1ULL << 13;
    MoveList expectedBlack = {
        Move(1ULL << 9, 1ULL << 18, 1ULL << 13, PieceColor::Black)
    };
    verifyMoveList("Single capture - right diagonal (Black)", expectedBlack, moveGen.generateMoves(board, PieceColor::Black));

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

    setUp();
    board.blackPawns = 1ULL << 10;
    board.whitePawns = 1ULL << 13;
    MoveList expectedBlack = {
        Move(1ULL << 10, 1ULL << 17, 1ULL << 13, PieceColor::Black)
    };
    verifyMoveList("Single capture - left diagonal (Black)", expectedBlack, moveGen.generateMoves(board, PieceColor::Black));
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

    
    setUp();
    board.blackPawns = 1ULL << 5;
    board.whitePawns = (1ULL << 9) | (1ULL << 17);
    MoveList expectedBlack = {
        Move({1ULL << 5, 1ULL << 12, 1ULL << 21}, (1ULL << 9) | (1ULL << 17), PieceColor::Black)
    };
    verifyMoveList("Chain capture - double (Black)", expectedBlack, moveGen.generateMoves(board, PieceColor::Black));

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

    setUp();
    board.blackPawns = 1ULL << 5;
    board.whitePawns = (1ULL << 9) | (1ULL << 17) | (1ULL << 26);
    MoveList expectedBlack = {
        Move({1ULL << 5, 1ULL << 12, 1ULL << 21, 1ULL << 30}, (1ULL << 9) | (1ULL << 17) | (1ULL << 26), PieceColor::Black)
    };
    verifyMoveList("Chain capture - triple (Black)", expectedBlack, moveGen.generateMoves(board, PieceColor::Black));

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

    setUp();
    board.blackPawns = 1ULL;  // Mirrored position
    board.whitePawns = (1ULL << 4) | (1ULL << 13) | (1ULL << 12) | (1ULL << 22) | (1ULL << 21) | (1ULL << 20);
    MoveList expectedBlack = {
        Move({1ULL, 1ULL << 9, 1ULL << 18, 1ULL << 27}, (1ULL << 4) | (1ULL << 13) | (1ULL << 22), PieceColor::Black),
        Move({1ULL, 1ULL << 9, 1ULL << 18, 1ULL << 25, 1ULL << 16, 1ULL << 9}, (1ULL << 4) | (1ULL << 13) | (1ULL << 21) | (1ULL << 20) | (1ULL << 12), PieceColor::Black),
        Move({1ULL, 1ULL << 9, 1ULL << 16, 1ULL << 25, 1ULL << 18, 1ULL << 27}, (1ULL << 4) | (1ULL << 12) | (1ULL << 20) | (1ULL << 21) | (1ULL << 22), PieceColor::Black),
        Move({1ULL, 1ULL << 9, 1ULL << 16, 1ULL << 25, 1ULL << 18, 1ULL << 9}, (1ULL << 4) | (1ULL << 12) | (1ULL << 20) | (1ULL << 21) | (1ULL << 13), PieceColor::Black),
    };
    verifyMoveList("Chain capture - multiple (Black)", expectedBlack, moveGen.generateMoves(board, PieceColor::Black));
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

    setUp();
    board.blackPawns = 1ULL << 21;
    board.kings = 1ULL << 21;
    MoveList expectedBlack = {
        Move(1ULL << 21, 1ULL << 25, 0, PieceColor::Black),
        Move(1ULL << 21, 1ULL << 28, 0, PieceColor::Black),
        Move(1ULL << 21, 1ULL << 26, 0, PieceColor::Black),
        Move(1ULL << 21, 1ULL << 30, 0, PieceColor::Black),
        Move(1ULL << 21, 1ULL << 17, 0, PieceColor::Black),
        Move(1ULL << 21, 1ULL << 12, 0, PieceColor::Black),
        Move(1ULL << 21, 1ULL << 8, 0, PieceColor::Black),
        Move(1ULL << 21, 1ULL << 18, 0, PieceColor::Black),
        Move(1ULL << 21, 1ULL << 14, 0, PieceColor::Black),
        Move(1ULL << 21, 1ULL << 11, 0, PieceColor::Black),
        Move(1ULL << 21, 1ULL << 7, 0, PieceColor::Black)
    };
    verifyMoveList("King basic moves (black)", expectedBlack, moveGen.generateMoves(board, PieceColor::Black));
}

void MoveGenerationTest::testKingCapturingMoves()
{
    setUp();
    board.whitePawns = 1ULL << 25;
    board.kings = 1ULL << 25;
    board.blackPawns = 1ULL << 18;
    MoveList expected = {
        Move(1ULL << 25, 1ULL << 14, 1ULL << 18),
        Move(1ULL << 25, 1ULL << 11, 1ULL << 18),
        Move(1ULL << 25, 1ULL << 7, 1ULL << 18)
    };
    verifyMoveList("King capturing moves", expected, moveGen.generateMoves(board, PieceColor::White));

    setUp();
    board.blackPawns = 1ULL << 25;
    board.kings = 1ULL << 25;
    board.whitePawns = 1ULL << 18;
    MoveList expectedBlack = {
        Move(1ULL << 25, 1ULL << 14, 1ULL << 18, PieceColor::Black),
        Move(1ULL << 25, 1ULL << 11, 1ULL << 18, PieceColor::Black),
        Move(1ULL << 25, 1ULL << 7, 1ULL << 18, PieceColor::Black)
    };
    verifyMoveList("King capturing moves (black)", expectedBlack, moveGen.generateMoves(board, PieceColor::Black));
}

void MoveGenerationTest::testKingCapturingMoves2()
{
    setUp();
    board.whitePawns = 1ULL << 28;
    board.kings = 1ULL << 28;
    board.blackPawns = (1ULL << 21) | (1ULL << 13) | (1ULL << 10);
    MoveList expected = {
        Move({1ULL << 28, 1ULL << 18, 1ULL << 9}, (1ULL << 21) | (1ULL << 13)),
        Move({1ULL << 28, 1ULL << 18, 1ULL << 4}, (1ULL << 21) | (1ULL << 13)),
        Move({1ULL << 28, 1ULL << 18, 1ULL}, (1ULL << 21) | (1ULL << 13)),
        Move({1ULL << 28, 1ULL << 14, 1ULL << 5}, (1ULL << 21) | (1ULL << 10)),
        Move({1ULL << 28, 1ULL << 14, 1ULL << 1}, (1ULL << 21) | (1ULL << 10)),
        Move(1ULL << 28, 1ULL << 11, 1ULL << 21),
        Move(1ULL << 28, 1ULL << 7, 1ULL << 21)
    };
    verifyMoveList("King capturing moves", expected, moveGen.generateMoves(board, PieceColor::White));

    setUp();
    board.blackPawns = 1ULL << 28;
    board.kings = 1ULL << 28;
    board.whitePawns = (1ULL << 21) | (1ULL << 13) | (1ULL << 10);
    MoveList expectedBlack = {
        Move({1ULL << 28, 1ULL << 18, 1ULL << 9}, (1ULL << 21) | (1ULL << 13), PieceColor::Black),
        Move({1ULL << 28, 1ULL << 18, 1ULL << 4}, (1ULL << 21) | (1ULL << 13), PieceColor::Black),
        Move({1ULL << 28, 1ULL << 18, 1ULL}, (1ULL << 21) | (1ULL << 13), PieceColor::Black),
        Move({1ULL << 28, 1ULL << 14, 1ULL << 5}, (1ULL << 21) | (1ULL << 10), PieceColor::Black),
        Move({1ULL << 28, 1ULL << 14, 1ULL << 1}, (1ULL << 21) | (1ULL << 10), PieceColor::Black),
        Move(1ULL << 28, 1ULL << 11, 1ULL << 21, PieceColor::Black),
        Move(1ULL << 28, 1ULL << 7, 1ULL << 21, PieceColor::Black)
    };
    verifyMoveList("King capturing moves (black)", expectedBlack, moveGen.generateMoves(board, PieceColor::Black));
}

void MoveGenerationTest::testKingCapturingMoves3()
{
    setUp();
    board.whitePawns = 1ULL << 31;
    board.kings = 1ULL << 31;
    board.blackPawns = (1ULL << 27) | (1ULL << 18) | (1ULL << 19) | (1ULL << 9) | (1ULL << 10) | (1ULL << 11);
    MoveList expected = {
        Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL << 4}, (1ULL << 27) | (1ULL << 18) | (1ULL << 9)),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL}, (1ULL << 27) | (1ULL << 18) | (1ULL << 9)),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL << 3}, (1ULL << 27) | (1ULL << 18) | (1ULL << 10)),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 17}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10)),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 20}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10)),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 24}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10)),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 2, 1ULL << 12}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 9)),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 2, 1ULL << 16}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 9)),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 13, 1ULL << 22}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 18)),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 13, 1ULL << 4}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 9)),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 13, 1ULL}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 9)),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL << 6, 1ULL << 15, 1ULL << 22}, (1ULL << 27) | (1ULL << 18) | (1ULL << 10) | (1ULL << 11) | (1ULL << 19)),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL << 6, 1ULL << 15, 1ULL << 26}, (1ULL << 27) | (1ULL << 18) | (1ULL << 10) | (1ULL << 11) | (1ULL << 19)),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL << 6, 1ULL << 15, 1ULL << 29}, (1ULL << 27) | (1ULL << 18) | (1ULL << 10) | (1ULL << 11) | (1ULL << 19)),
        
    };
    verifyMoveList("Chain capture - multiple", expected, moveGen.generateMoves(board, PieceColor::White));

    setUp();
    board.blackPawns = 1ULL << 31;
    board.kings = 1ULL << 31;
    board.whitePawns = (1ULL << 27) | (1ULL << 18) | (1ULL << 19) | (1ULL << 9) | (1ULL << 10) | (1ULL << 11);
    MoveList expectedBlack = {
        Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL << 4}, (1ULL << 27) | (1ULL << 18) | (1ULL << 9), PieceColor::Black),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL}, (1ULL << 27) | (1ULL << 18) | (1ULL << 9), PieceColor::Black),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL << 3}, (1ULL << 27) | (1ULL << 18) | (1ULL << 10), PieceColor::Black),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 17}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10), PieceColor::Black),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 20}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10), PieceColor::Black),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 24}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10), PieceColor::Black),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 2, 1ULL << 12}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 9), PieceColor::Black),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 2, 1ULL << 16}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 9), PieceColor::Black),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 13, 1ULL << 22}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 18), PieceColor::Black),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 13, 1ULL << 4}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 9), PieceColor::Black),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 13, 1ULL}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 9), PieceColor::Black),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL << 6, 1ULL << 15, 1ULL << 22}, (1ULL << 27) | (1ULL << 18) | (1ULL << 10) | (1ULL << 11) | (1ULL << 19), PieceColor::Black),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL << 6, 1ULL << 15, 1ULL << 26}, (1ULL << 27) | (1ULL << 18) | (1ULL << 10) | (1ULL << 11) | (1ULL << 19), PieceColor::Black),
        Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL << 6, 1ULL << 15, 1ULL << 29}, (1ULL << 27) | (1ULL << 18) | (1ULL << 10) | (1ULL << 11) | (1ULL << 19), PieceColor::Black),

    };
    verifyMoveList("Chain capture - multiple (black)", expectedBlack, moveGen.generateMoves(board, PieceColor::Black));
}

void MoveGenerationTest::testKingCapturingMovesEdgeCase()
{
    setUp();
    board.whitePawns = 1ULL << 21;
    board.kings = 1ULL << 21;
    board.blackPawns = (1ULL << 17) | (1ULL << 26);
    MoveList expected = {
        Move(1ULL << 21, 1ULL << 12, 1ULL << 17),
        Move(1ULL << 21, 1ULL << 8, 1ULL << 17),
        Move(1ULL << 21, 1ULL << 30, 1ULL << 26)
    };
    verifyMoveList("King capturing moves", expected, moveGen.generateMoves(board, PieceColor::White));
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

void MoveGenerationTest::assertFailedTest()
{
    setUp();
    board.whitePawns = (1ULL << 30) | (1ULL << 31) | (1ULL << 26) | (1ULL << 20) | (1ULL << 23) | (1ULL << 17) | (1ULL << 18) | (1ULL << 19);
    board.blackPawns = (1ULL) | (1ULL << 2) | (1ULL << 6) | (1ULL << 7) | (1ULL << 9) | (1ULL << 10) | (1ULL << 11) | (1ULL << 13) | (1ULL << 14);
	Board newBoard(board.whitePawns, board.blackPawns, board.kings);
	newBoard.printBoard();
    MoveList expected = {
        Move(1ULL << 21, 1ULL << 12, 1ULL << 17),
        Move(1ULL << 21, 1ULL << 8, 1ULL << 17),
        Move(1ULL << 21, 1ULL << 30, 1ULL << 26)
    };
    verifyMoveList("King capturing moves", expected, moveGen.generateMoves(board, PieceColor::Black));
}

void MoveGenerationTest::runAllTests()
{
 //   testBasicPawnMovesCenter();
 //   testBasicPawnMovesLeftEdge();
 //   testBasicPawnMovesRightEdge();
 //   testSingleCaptureRightDiagonal();
 //   testSingleCaptureLeftDiagonal();
 //   testChainCaptureDouble();
 //   testChainCaptureTriple();
 //   testChainCaptureMultiple();
 //   testKingBasicMoves();
	//testKingCapturingMoves();
	//testKingCapturingMoves2();
 //   testKingCapturingMovesEdgeCase();
 //   testKingCapturingMoves3();
 //   testCrowningMove();

	assertFailedTest();


    printSummary("Move Generation");
}
