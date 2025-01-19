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

bool MoveGenerationTest::verifyMoveList2(const char* testName, const std::vector<Move2>& expected, Queue<Move2>* actual)
{
    bool passed = true;

    if (expected.size() != actual->length()) {
        printf("Test %s: FAILED\n", testName);
        printf("Expected %zu moves, got %zu moves\n", expected.size(), actual->length());
        passed = false;
    }
    else {
		while (!actual->empty()) {
			Move2 actualMove = actual->front();
			actual->pop();
			bool moveFound = false;
			for (const auto& expectedMove : expected) {
				if (expectedMove == actualMove) {
					moveFound = true;
                    std::cout << "Found move: " << actualMove.toString() << std::endl;
					break;
				}
			}
			if (!moveFound) {
				printf("Test %s: FAILED\n", testName);
				std::cout << "Missing move: " << actualMove.toString() << std::endl;
				passed = false;
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

void MoveGenerationTest::testBasicPawnMovesCenter()
{
    setUp();
    boardAfterMove.whitePawns = 1ULL << 25;
    MoveList expected = {
        Move(1ULL << 25, 1ULL << 20),
        Move(1ULL << 25, 1ULL << 21)
    };
    verifyMoveList("Basic pawn moves - center position (White)", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));

    std::vector<Move2> expected2 = {
        Move2(1ULL << 25, 1ULL << 20, PieceColor::White),
        Move2(1ULL << 25, 1ULL << 21, PieceColor::White)
    };
	moveQueue.clear();
	moveGen.generateMovesGpu(boardAfterMove, PieceColor::White, &moveQueue);
    verifyMoveList2("Basic pawn moves - center position (White)", expected2, &moveQueue);

    setUp();
    boardAfterMove.blackPawns = 1ULL << 9;  // Mirrored position for black
    MoveList expectedBlack = {
        Move(1ULL << 9, 1ULL << 13, 0, PieceColor::Black),
        Move(1ULL << 9, 1ULL << 12, 0, PieceColor::Black)
    };
    verifyMoveList("Basic pawn moves - center position (Black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));
}

void MoveGenerationTest::testBasicPawnMovesLeftEdge()
{
    setUp();
    boardAfterMove.whitePawns = 1ULL << 7;
    MoveList expected = {
        Move(1ULL << 7, 1ULL << 3)
    };
    verifyMoveList("Basic pawn moves - left edge (White)", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));

    setUp();
    boardAfterMove.blackPawns = 1ULL << 7;  // Mirrored position for black
    MoveList expectedBlack = {
        Move(1ULL << 7, 1ULL << 11, 0, PieceColor::Black)
    };
    verifyMoveList("Basic pawn moves - left edge (Black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));
}

void MoveGenerationTest::testBasicPawnMovesRightEdge()
{
    setUp();
    boardAfterMove.whitePawns = 1ULL << 31;
    MoveList expected = {
        Move(1ULL << 31, 1ULL << 27)
    };
    verifyMoveList("Basic pawn moves - right edge (White)", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));

    setUp();
    boardAfterMove.blackPawns = 1ULL;  // Mirrored position for black
    MoveList expectedBlack = {
        Move(1ULL, 1ULL << 4, 0, PieceColor::Black)
    };
    verifyMoveList("Basic pawn moves - right edge (Black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));
}

void MoveGenerationTest::testSingleCaptureRightDiagonal()
{
    setUp();
    boardAfterMove.whitePawns = 1ULL << 25;
    boardAfterMove.blackPawns = 1ULL << 21;
    MoveList expected = {
        Move(1ULL << 25, 1ULL << 18, 1ULL << 21)
    };
    verifyMoveList("Single capture - right diagonal", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));

    setUp();
    boardAfterMove.blackPawns = 1ULL << 9;
    boardAfterMove.whitePawns = 1ULL << 13;
    MoveList expectedBlack = {
        Move(1ULL << 9, 1ULL << 18, 1ULL << 13, PieceColor::Black)
    };
    verifyMoveList("Single capture - right diagonal (Black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));

}

void MoveGenerationTest::testSingleCaptureLeftDiagonal()
{
    setUp();
    boardAfterMove.whitePawns = 1ULL << 26;
    boardAfterMove.blackPawns = 1ULL << 21;
    MoveList expected = {
        Move(1ULL << 26, 1ULL << 17, 1ULL << 21)
    };
    verifyMoveList("Single capture - left diagonal", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));

    setUp();
    boardAfterMove.blackPawns = 1ULL << 10;
    boardAfterMove.whitePawns = 1ULL << 13;
    MoveList expectedBlack = {
        Move(1ULL << 10, 1ULL << 17, 1ULL << 13, PieceColor::Black)
    };
    verifyMoveList("Single capture - left diagonal (Black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));
}

void MoveGenerationTest::testChainCaptureDouble()
{
    setUp();
    boardAfterMove.whitePawns = 1ULL << 29;
    boardAfterMove.blackPawns = (1ULL << 25) | (1ULL << 17);
    MoveList expected = {
        Move({1ULL << 29, 1ULL << 20, 1ULL << 13}, (1ULL << 25) | (1ULL << 17))
    };
    verifyMoveList("Chain capture - double", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));

    
    setUp();
    boardAfterMove.blackPawns = 1ULL << 5;
    boardAfterMove.whitePawns = (1ULL << 9) | (1ULL << 17);
    MoveList expectedBlack = {
        Move({1ULL << 5, 1ULL << 12, 1ULL << 21}, (1ULL << 9) | (1ULL << 17), PieceColor::Black)
    };
    verifyMoveList("Chain capture - double (Black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));

}

void MoveGenerationTest::testChainCaptureTriple()
{
    setUp();
    boardAfterMove.whitePawns = 1ULL << 29;
    boardAfterMove.blackPawns = (1ULL << 25) | (1ULL << 17) | (1ULL << 10);
    MoveList expected = {
        Move({1ULL << 29, 1ULL << 20, 1ULL << 13, 1ULL << 6}, (1ULL << 25) | (1ULL << 17) | (1ULL << 10))
    };
    verifyMoveList("Chain capture - triple", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));

    setUp();
    boardAfterMove.blackPawns = 1ULL << 5;
    boardAfterMove.whitePawns = (1ULL << 9) | (1ULL << 17) | (1ULL << 26);
    MoveList expectedBlack = {
        Move({1ULL << 5, 1ULL << 12, 1ULL << 21, 1ULL << 30}, (1ULL << 9) | (1ULL << 17) | (1ULL << 26), PieceColor::Black)
    };
    verifyMoveList("Chain capture - triple (Black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));

}

void MoveGenerationTest::testChainCaptureMultiple()
{
    setUp();
    boardAfterMove.whitePawns = 0x90000000;
    boardAfterMove.blackPawns = (1ULL << 27) | (1ULL << 18) | (1ULL << 19) | (1ULL << 9) | (1ULL << 10) | (1ULL << 11);
    MoveList expected = {
        Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL << 4}, (1ULL << 27) | (1ULL << 18) | (1ULL << 9)),
		Move({1ULL << 31, 1ULL << 22, 1ULL << 13, 1ULL << 6, 1ULL << 15, 1ULL << 22}, (1ULL << 27) | (1ULL << 18) | (1ULL << 10) | (1ULL << 11) | (1ULL << 19)),
		Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 13, 1ULL << 4}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 9)),
		Move({1ULL << 31, 1ULL << 22, 1ULL << 15, 1ULL << 6, 1ULL << 13, 1ULL << 22}, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 18)),
    };
    verifyMoveList("Chain capture - multiple", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));

    setUp();
    boardAfterMove.blackPawns = 1ULL;  // Mirrored position
    boardAfterMove.whitePawns = (1ULL << 4) | (1ULL << 13) | (1ULL << 12) | (1ULL << 22) | (1ULL << 21) | (1ULL << 20);
    MoveList expectedBlack = {
        Move({1ULL, 1ULL << 9, 1ULL << 18, 1ULL << 27}, (1ULL << 4) | (1ULL << 13) | (1ULL << 22), PieceColor::Black),
        Move({1ULL, 1ULL << 9, 1ULL << 18, 1ULL << 25, 1ULL << 16, 1ULL << 9}, (1ULL << 4) | (1ULL << 13) | (1ULL << 21) | (1ULL << 20) | (1ULL << 12), PieceColor::Black),
        Move({1ULL, 1ULL << 9, 1ULL << 16, 1ULL << 25, 1ULL << 18, 1ULL << 27}, (1ULL << 4) | (1ULL << 12) | (1ULL << 20) | (1ULL << 21) | (1ULL << 22), PieceColor::Black),
        Move({1ULL, 1ULL << 9, 1ULL << 16, 1ULL << 25, 1ULL << 18, 1ULL << 9}, (1ULL << 4) | (1ULL << 12) | (1ULL << 20) | (1ULL << 21) | (1ULL << 13), PieceColor::Black),
    };
    verifyMoveList("Chain capture - multiple (Black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));
}

void MoveGenerationTest::testKingBasicMoves()
{
    setUp();
    boardAfterMove.whitePawns = 1ULL << 21;
    boardAfterMove.kings = 1ULL << 21;
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
    verifyMoveList("King basic moves", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));

    setUp();
    boardAfterMove.blackPawns = 1ULL << 21;
    boardAfterMove.kings = 1ULL << 21;
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
    verifyMoveList("King basic moves (black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));
}

void MoveGenerationTest::testKingCapturingMoves()
{
    setUp();
    boardAfterMove.whitePawns = 1ULL << 25;
    boardAfterMove.kings = 1ULL << 25;
    boardAfterMove.blackPawns = 1ULL << 18;
    MoveList expected = {
        Move(1ULL << 25, 1ULL << 14, 1ULL << 18),
        Move(1ULL << 25, 1ULL << 11, 1ULL << 18),
        Move(1ULL << 25, 1ULL << 7, 1ULL << 18)
    };
    verifyMoveList("King capturing moves", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));

    setUp();
    boardAfterMove.blackPawns = 1ULL << 25;
    boardAfterMove.kings = 1ULL << 25;
    boardAfterMove.whitePawns = 1ULL << 18;
    MoveList expectedBlack = {
        Move(1ULL << 25, 1ULL << 14, 1ULL << 18, PieceColor::Black),
        Move(1ULL << 25, 1ULL << 11, 1ULL << 18, PieceColor::Black),
        Move(1ULL << 25, 1ULL << 7, 1ULL << 18, PieceColor::Black)
    };
    verifyMoveList("King capturing moves (black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));
}

void MoveGenerationTest::testKingCapturingMoves2()
{
    setUp();
    boardAfterMove.whitePawns = 1ULL << 28;
    boardAfterMove.kings = 1ULL << 28;
    boardAfterMove.blackPawns = (1ULL << 21) | (1ULL << 13) | (1ULL << 10);
    MoveList expected = {
        Move({1ULL << 28, 1ULL << 18, 1ULL << 9}, (1ULL << 21) | (1ULL << 13)),
        Move({1ULL << 28, 1ULL << 18, 1ULL << 4}, (1ULL << 21) | (1ULL << 13)),
        Move({1ULL << 28, 1ULL << 18, 1ULL}, (1ULL << 21) | (1ULL << 13)),
        Move({1ULL << 28, 1ULL << 14, 1ULL << 5}, (1ULL << 21) | (1ULL << 10)),
        Move({1ULL << 28, 1ULL << 14, 1ULL << 1}, (1ULL << 21) | (1ULL << 10)),
        Move(1ULL << 28, 1ULL << 11, 1ULL << 21),
        Move(1ULL << 28, 1ULL << 7, 1ULL << 21)
    };
    verifyMoveList("King capturing moves", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));

    setUp();
    boardAfterMove.blackPawns = 1ULL << 28;
    boardAfterMove.kings = 1ULL << 28;
    boardAfterMove.whitePawns = (1ULL << 21) | (1ULL << 13) | (1ULL << 10);
    MoveList expectedBlack = {
        Move({1ULL << 28, 1ULL << 18, 1ULL << 9}, (1ULL << 21) | (1ULL << 13), PieceColor::Black),
        Move({1ULL << 28, 1ULL << 18, 1ULL << 4}, (1ULL << 21) | (1ULL << 13), PieceColor::Black),
        Move({1ULL << 28, 1ULL << 18, 1ULL}, (1ULL << 21) | (1ULL << 13), PieceColor::Black),
        Move({1ULL << 28, 1ULL << 14, 1ULL << 5}, (1ULL << 21) | (1ULL << 10), PieceColor::Black),
        Move({1ULL << 28, 1ULL << 14, 1ULL << 1}, (1ULL << 21) | (1ULL << 10), PieceColor::Black),
        Move(1ULL << 28, 1ULL << 11, 1ULL << 21, PieceColor::Black),
        Move(1ULL << 28, 1ULL << 7, 1ULL << 21, PieceColor::Black)
    };
    verifyMoveList("King capturing moves (black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));
}

void MoveGenerationTest::testKingCapturingMoves3()
{
    setUp();
    boardAfterMove.whitePawns = 1ULL << 31;
    boardAfterMove.kings = 1ULL << 31;
    boardAfterMove.blackPawns = (1ULL << 27) | (1ULL << 18) | (1ULL << 19) | (1ULL << 9) | (1ULL << 10) | (1ULL << 11);
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
    verifyMoveList("Chain capture - multiple", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));

    setUp();
    boardAfterMove.blackPawns = 1ULL << 31;
    boardAfterMove.kings = 1ULL << 31;
    boardAfterMove.whitePawns = (1ULL << 27) | (1ULL << 18) | (1ULL << 19) | (1ULL << 9) | (1ULL << 10) | (1ULL << 11);
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
    verifyMoveList("Chain capture - multiple (black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));
}

void MoveGenerationTest::testKingCapturingMovesEdgeCase()
{
    setUp();
    boardAfterMove.whitePawns = 1ULL << 21;
    boardAfterMove.kings = 1ULL << 21;
    boardAfterMove.blackPawns = (1ULL << 17) | (1ULL << 26);
    MoveList expected = {
        Move(1ULL << 21, 1ULL << 12, 1ULL << 17),
        Move(1ULL << 21, 1ULL << 8, 1ULL << 17),
        Move(1ULL << 21, 1ULL << 30, 1ULL << 26)
    };
    verifyMoveList("King capturing moves", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));
}

void MoveGenerationTest::testCrowningMove()
{
    setUp();
    boardAfterMove.whitePawns = 1ULL << 4;
    MoveList expected = {
        Move(1ULL << 4, 1ULL << 0),
        Move(1ULL << 4, 1ULL << 1)
    };
    verifyMoveList("Crowning move", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));
}

void MoveGenerationTest::assertFailedTest()
{
    setUp();
    boardAfterMove.whitePawns = (1ULL << 30) | (1ULL << 31) | (1ULL << 26) | (1ULL << 20) | (1ULL << 23) | (1ULL << 17) | (1ULL << 18) | (1ULL << 19);
    boardAfterMove.blackPawns = (1ULL) | (1ULL << 2) | (1ULL << 6) | (1ULL << 7) | (1ULL << 9) | (1ULL << 10) | (1ULL << 11) | (1ULL << 13) | (1ULL << 14);
	Board newBoard(boardAfterMove.whitePawns, boardAfterMove.blackPawns, boardAfterMove.kings);
	newBoard.toString();
    MoveList expected = {
        Move(1ULL << 21, 1ULL << 12, 1ULL << 17),
        Move(1ULL << 21, 1ULL << 8, 1ULL << 17),
        Move(1ULL << 21, 1ULL << 30, 1ULL << 26)
    };
    verifyMoveList("King capturing moves", expected, moveGen.generateMoves(boardAfterMove, PieceColor::Black));
}

void MoveGenerationTest::assertFailedTest2()
{
	setUp();
	boardAfterMove.whitePawns = (1ULL << 3);
	boardAfterMove.kings = (1ULL << 3) | (1ULL << 13);
	boardAfterMove.blackPawns = (1ULL << 13);
	MoveList expected = {
		Move(1ULL << 21, 1ULL << 12, 1ULL << 17),
		Move(1ULL << 21, 1ULL << 8, 1ULL << 17),
		Move(1ULL << 21, 1ULL << 30, 1ULL << 26)
	};
	verifyMoveList("assertion failed scenario", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));
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
    testChainCaptureMultiple();
    testKingBasicMoves();
	testKingCapturingMoves();
	testKingCapturingMoves2();
    testKingCapturingMovesEdgeCase();
    testKingCapturingMoves3();
    testCrowningMove();

	assertFailedTest();
	// assertFailedTest2();

    printSummary("Move Generation");
}
