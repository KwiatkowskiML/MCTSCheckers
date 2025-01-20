#include "MoveGenerationTest.h"
#include "Move.h"
#include "Board.h"
#include <fstream>
#include <sstream>
#include <random>

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

        while (!actual->empty())
        {
            Move2 actualMove = actual->front();
            actual->pop();
            std::cout << "Actual move: " << actualMove.toString() << std::endl;
        }

        for (const auto& expectedMove : expected) {
            std::cout << "Expected move: " << expectedMove.toString() << std::endl;
        }

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

    std::vector<Move2> expectedBlack2 = {
        Move2(1ULL << 9, 1ULL << 13, PieceColor::Black),
        Move2(1ULL << 9, 1ULL << 12, PieceColor::Black)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::Black, &moveQueue);
    verifyMoveList2("Basic pawn moves - center position (Black)", expectedBlack2, &moveQueue);
}

void MoveGenerationTest::testBasicPawnMovesLeftEdge()
{
    setUp();
    boardAfterMove.whitePawns = 1ULL << 7;
    MoveList expected = {
        Move(1ULL << 7, 1ULL << 3)
    };
    verifyMoveList("Basic pawn moves - left edge (White)", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));

    std::vector<Move2> expected2 = {
        Move2(1ULL << 7, 1ULL << 3, PieceColor::White)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::White, &moveQueue);
    verifyMoveList2("Basic pawn moves - left edge (White)", expected2, &moveQueue);

    setUp();
    boardAfterMove.blackPawns = 1ULL << 7;  // Mirrored position for black
    MoveList expectedBlack = {
        Move(1ULL << 7, 1ULL << 11, 0, PieceColor::Black)
    };
    verifyMoveList("Basic pawn moves - left edge (Black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));

    std::vector<Move2> expectedBlack2 = {
       Move2(1ULL << 7, 1ULL << 11, PieceColor::Black)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::Black, &moveQueue);
    verifyMoveList2("Basic pawn moves - left edge (Black)", expectedBlack2, &moveQueue);
}

void MoveGenerationTest::testBasicPawnMovesRightEdge()
{
    setUp();
    boardAfterMove.whitePawns = 1ULL << 31;
    MoveList expected = {
        Move(1ULL << 31, 1ULL << 27)
    };
    verifyMoveList("Basic pawn moves - right edge (White)", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));

    std::vector<Move2> expected2 = {
        Move2(1ULL << 31, 1ULL << 27, PieceColor::White)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::White, &moveQueue);
    verifyMoveList2("Basic pawn moves - right edge (White)", expected2, &moveQueue);

    setUp();
    boardAfterMove.blackPawns = 1ULL;  // Mirrored position for black
    MoveList expectedBlack = {
        Move(1ULL, 1ULL << 4, 0, PieceColor::Black)
    };
    verifyMoveList("Basic pawn moves - right edge (Black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));

    std::vector<Move2> expectedBlack2 = {
       Move2(1ULL, 1ULL << 4, PieceColor::Black)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::Black, &moveQueue);
    verifyMoveList2("Basic pawn moves - right edge (Black)", expectedBlack2, &moveQueue);
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

    std::vector<Move2> expected2 = {
       Move2(1ULL << 25, 1ULL << 18, PieceColor::White, 1ULL << 21)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::White, &moveQueue);
    verifyMoveList2("Single capture - right diagonal", expected2, &moveQueue);

    setUp();
    boardAfterMove.blackPawns = 1ULL << 9;
    boardAfterMove.whitePawns = 1ULL << 13;
    MoveList expectedBlack = {
        Move(1ULL << 9, 1ULL << 18, 1ULL << 13, PieceColor::Black)
    };
    verifyMoveList("Single capture - right diagonal (Black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));

    std::vector<Move2> expectedBlack2 = {
        Move2(1ULL << 9, 1ULL << 18, PieceColor::Black, 1ULL << 13)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::Black, &moveQueue);
    verifyMoveList2("Single capture - right diagonal (Black)", expectedBlack2, &moveQueue);

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

    std::vector<Move2> expected2 = {
      Move2(1ULL << 26, 1ULL << 17, PieceColor::White, 1ULL << 21)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::White, &moveQueue);
    verifyMoveList2("Single capture - left diagonal", expected2, &moveQueue);

    setUp();
    boardAfterMove.blackPawns = 1ULL << 10;
    boardAfterMove.whitePawns = 1ULL << 13;
    MoveList expectedBlack = {
        Move(1ULL << 10, 1ULL << 17, 1ULL << 13, PieceColor::Black)
    };
    verifyMoveList("Single capture - left diagonal (Black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));

    std::vector<Move2> expectedBlack2 = {
        Move2(1ULL << 10, 1ULL << 17, PieceColor::Black, 1ULL << 13)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::Black, &moveQueue);
    verifyMoveList2("Single capture - left diagonal (Black)", expectedBlack2, &moveQueue);
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

    std::vector<Move2> expected2 = {
     Move2(1ULL << 29, 1ULL << 13, PieceColor::White, (1ULL << 25) | (1ULL << 17))
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::White, &moveQueue);
    verifyMoveList2("Chain capture - double", expected2, &moveQueue);

    setUp();
    boardAfterMove.blackPawns = 1ULL << 5;
    boardAfterMove.whitePawns = (1ULL << 9) | (1ULL << 17);
    MoveList expectedBlack = {
        Move({1ULL << 5, 1ULL << 12, 1ULL << 21}, (1ULL << 9) | (1ULL << 17), PieceColor::Black)
    };
    verifyMoveList("Chain capture - double (Black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));

    std::vector<Move2> expectedBlack2 = {
        Move2(1ULL << 5 , 1ULL << 21, PieceColor::Black, (1ULL << 9) | (1ULL << 17))
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::Black, &moveQueue);
    verifyMoveList2("Chain capture - double (Black)", expectedBlack2, &moveQueue);
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

    std::vector<Move2> expected2 = {
     Move2(1ULL << 29, 1ULL << 6, PieceColor::White, (1ULL << 25) | (1ULL << 17) | (1ULL << 10))
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::White, &moveQueue);
    verifyMoveList2("Chain capture - triple", expected2, &moveQueue);

    setUp();
    boardAfterMove.blackPawns = 1ULL << 5;
    boardAfterMove.whitePawns = (1ULL << 9) | (1ULL << 17) | (1ULL << 26);
    MoveList expectedBlack = {
        Move({1ULL << 5, 1ULL << 12, 1ULL << 21, 1ULL << 30}, (1ULL << 9) | (1ULL << 17) | (1ULL << 26), PieceColor::Black)
    };
    verifyMoveList("Chain capture - triple (Black)", expectedBlack, moveGen.generateMoves(boardAfterMove, PieceColor::Black));

    std::vector<Move2> expectedBlack2 = {
       Move2(1ULL << 5, 1ULL << 30, PieceColor::Black, (1ULL << 9) | (1ULL << 17) | (1ULL << 26))
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::Black, &moveQueue);
    verifyMoveList2("Chain capture - triple (Black)", expectedBlack2, &moveQueue);
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

    std::vector<Move2> expected2 = {
        Move2(1ULL << 31, 1ULL << 4, PieceColor::White, (1ULL << 27) | (1ULL << 18) | (1ULL << 9)),
        Move2(1ULL << 31, 1ULL << 22, PieceColor::White, (1ULL << 27) | (1ULL << 18) | (1ULL << 10) | (1ULL << 11) | (1ULL << 19)),
        Move2(1ULL << 31, 1ULL << 4, PieceColor::White, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 9)),
        Move2(1ULL << 31, 1ULL << 22, PieceColor::White, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 18)),
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::White, &moveQueue);
    verifyMoveList2("Chain capture - multiple", expected2, &moveQueue);

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

    std::vector<Move2> expectedBlack2 = {
       Move2(1ULL, 1ULL << 27, PieceColor::Black, (1ULL << 4) | (1ULL << 13) | (1ULL << 22)),
        Move2(1ULL, 1ULL << 9, PieceColor::Black, (1ULL << 4) | (1ULL << 13) | (1ULL << 21) | (1ULL << 20) | (1ULL << 12)),
        Move2(1ULL, 1ULL << 27, PieceColor::Black, (1ULL << 4) | (1ULL << 12) | (1ULL << 20) | (1ULL << 21) | (1ULL << 22)),
        Move2(1ULL, 1ULL << 9, PieceColor::Black, (1ULL << 4) | (1ULL << 12) | (1ULL << 20) | (1ULL << 21) | (1ULL << 13)),
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::Black, &moveQueue);
    verifyMoveList2("Chain capture - multiple (Black)", expectedBlack2, &moveQueue);
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

    std::vector<Move2> expected2 = {
       Move2(1ULL << 21, 1ULL << 25, PieceColor::White),
        Move2(1ULL << 21, 1ULL << 28, PieceColor::White),
        Move2(1ULL << 21, 1ULL << 26, PieceColor::White),
        Move2(1ULL << 21, 1ULL << 30, PieceColor::White),
        Move2(1ULL << 21, 1ULL << 17, PieceColor::White),
        Move2(1ULL << 21, 1ULL << 12, PieceColor::White),
        Move2(1ULL << 21, 1ULL << 8, PieceColor::White),
        Move2(1ULL << 21, 1ULL << 18, PieceColor::White),
        Move2(1ULL << 21, 1ULL << 14, PieceColor::White),
        Move2(1ULL << 21, 1ULL << 11, PieceColor::White),
        Move2(1ULL << 21, 1ULL << 7, PieceColor::White)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::White, &moveQueue);
    verifyMoveList2("King basic moves", expected2, &moveQueue);

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

    std::vector<Move2> expectedBlack2 = {
       Move2(1ULL << 21, 1ULL << 25, PieceColor::Black),
        Move2(1ULL << 21, 1ULL << 28, PieceColor::Black),
        Move2(1ULL << 21, 1ULL << 26, PieceColor::Black),
        Move2(1ULL << 21, 1ULL << 30, PieceColor::Black),
        Move2(1ULL << 21, 1ULL << 17, PieceColor::Black),
        Move2(1ULL << 21, 1ULL << 12, PieceColor::Black),
        Move2(1ULL << 21, 1ULL << 8, PieceColor::Black),
        Move2(1ULL << 21, 1ULL << 18, PieceColor::Black),
        Move2(1ULL << 21, 1ULL << 14, PieceColor::Black),
        Move2(1ULL << 21, 1ULL << 11, PieceColor::Black),
        Move2(1ULL << 21, 1ULL << 7, PieceColor::Black)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::Black, &moveQueue);
    verifyMoveList2("King basic moves", expectedBlack2, &moveQueue);
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

    std::vector<Move2> expected2 = {
         Move2(1ULL << 25, 1ULL << 14, PieceColor::White, 1ULL << 18),
        Move2(1ULL << 25, 1ULL << 11, PieceColor::White, 1ULL << 18),
        Move2(1ULL << 25, 1ULL << 7, PieceColor::White, 1ULL << 18)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::White, &moveQueue);
    verifyMoveList2("King capturing moves", expected2, &moveQueue);

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

    std::vector<Move2> expectedBlack2 = {
        Move2(1ULL << 25, 1ULL << 14, PieceColor::Black, 1ULL << 18),
        Move2(1ULL << 25, 1ULL << 11, PieceColor::Black, 1ULL << 18),
        Move2(1ULL << 25, 1ULL << 7, PieceColor::Black, 1ULL << 18)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::Black, &moveQueue);
    verifyMoveList2("King capturing moves (black)", expectedBlack2, &moveQueue);
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

    std::vector<Move2> expected2 = {
        Move2(1ULL << 28, 1ULL << 9, PieceColor::White, (1ULL << 21) | (1ULL << 13)),
        Move2(1ULL << 28, 1ULL << 4, PieceColor::White, (1ULL << 21) | (1ULL << 13)),
        Move2(1ULL << 28, 1ULL, PieceColor::White, (1ULL << 21) | (1ULL << 13)),
        Move2(1ULL << 28, 1ULL << 5, PieceColor::White, (1ULL << 21) | (1ULL << 10)),
        Move2(1ULL << 28, 1ULL << 1, PieceColor::White, (1ULL << 21) | (1ULL << 10)),
        Move2(1ULL << 28, 1ULL << 11, PieceColor::White, 1ULL << 21),
        Move2(1ULL << 28, 1ULL << 7, PieceColor::White, 1ULL << 21)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::White, &moveQueue);
    verifyMoveList2("King capturing moves", expected2, &moveQueue);

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

    std::vector<Move2> expectedBlack2 = {
        Move2(1ULL << 28, 1ULL << 9, PieceColor::Black, (1ULL << 21) | (1ULL << 13)),
        Move2(1ULL << 28, 1ULL << 4, PieceColor::Black, (1ULL << 21) | (1ULL << 13)),
        Move2(1ULL << 28, 1ULL, PieceColor::Black, (1ULL << 21) | (1ULL << 13)),
        Move2(1ULL << 28, 1ULL << 5, PieceColor::Black, (1ULL << 21) | (1ULL << 10)),
        Move2(1ULL << 28, 1ULL << 1, PieceColor::Black, (1ULL << 21) | (1ULL << 10)),
        Move2(1ULL << 28, 1ULL << 11, PieceColor::Black, 1ULL << 21),
        Move2(1ULL << 28, 1ULL << 7, PieceColor::Black, 1ULL << 21)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::Black, &moveQueue);
    verifyMoveList2("King capturing moves (black)", expectedBlack2, &moveQueue);
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

    std::vector<Move2> expected2 = {
        Move2(1ULL << 31, 1ULL << 4, PieceColor::White, (1ULL << 27) | (1ULL << 18) | (1ULL << 9)),
        Move2(1ULL << 31, 1ULL, PieceColor::White,(1ULL << 27) | (1ULL << 18) | (1ULL << 9)),
        Move2(1ULL << 31, 1ULL << 3, PieceColor::White,(1ULL << 27) | (1ULL << 18) | (1ULL << 10)),
        Move2(1ULL << 31, 1ULL << 17, PieceColor::White,(1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10)),
        Move2(1ULL << 31, 1ULL << 20, PieceColor::White,(1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10)),
        Move2(1ULL << 31, 1ULL << 24, PieceColor::White,(1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10)),
        Move2(1ULL << 31, 1ULL << 12, PieceColor::White,(1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 9)),
        Move2(1ULL << 31, 1ULL << 16, PieceColor::White,(1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 9)),
        Move2(1ULL << 31, 1ULL << 22, PieceColor::White,(1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 18)),
        Move2(1ULL << 31, 1ULL << 4, PieceColor::White,(1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 9)),
        Move2(1ULL << 31, 1ULL, PieceColor::White,(1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 9)),
        Move2(1ULL << 31, 1ULL << 22, PieceColor::White,(1ULL << 27) | (1ULL << 18) | (1ULL << 10) | (1ULL << 11) | (1ULL << 19)),
        Move2(1ULL << 31, 1ULL << 26, PieceColor::White,(1ULL << 27) | (1ULL << 18) | (1ULL << 10) | (1ULL << 11) | (1ULL << 19)),
        Move2(1ULL << 31, 1ULL << 29, PieceColor::White,(1ULL << 27) | (1ULL << 18) | (1ULL << 10) | (1ULL << 11) | (1ULL << 19)),
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::White, &moveQueue);
    verifyMoveList2("Chain capture - multiple", expected2, &moveQueue);

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

    std::vector<Move2> expectedBlack2 = {
       Move2(1ULL << 31, 1ULL << 4, PieceColor::Black, (1ULL << 27) | (1ULL << 18) | (1ULL << 9)),
        Move2(1ULL << 31, 1ULL, PieceColor::Black, (1ULL << 27) | (1ULL << 18) | (1ULL << 9)),
        Move2(1ULL << 31, 1ULL << 3, PieceColor::Black, (1ULL << 27) | (1ULL << 18) | (1ULL << 10)),
        Move2(1ULL << 31, 1ULL << 17, PieceColor::Black, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10)),
        Move2(1ULL << 31, 1ULL << 20, PieceColor::Black, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10)),
        Move2(1ULL << 31, 1ULL << 24, PieceColor::Black, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10)),
        Move2(1ULL << 31, 1ULL << 12, PieceColor::Black, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 9)),
        Move2(1ULL << 31, 1ULL << 16, PieceColor::Black, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 9)),
        Move2(1ULL << 31, 1ULL << 22, PieceColor::Black, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 18)),
        Move2(1ULL << 31, 1ULL << 4, PieceColor::Black, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 9)),
        Move2(1ULL << 31, 1ULL, PieceColor::Black, (1ULL << 27) | (1ULL << 19) | (1ULL << 11) | (1ULL << 10) | (1ULL << 9)),
        Move2(1ULL << 31, 1ULL << 22, PieceColor::Black, (1ULL << 27) | (1ULL << 18) | (1ULL << 10) | (1ULL << 11) | (1ULL << 19)),
        Move2(1ULL << 31, 1ULL << 26, PieceColor::Black, (1ULL << 27) | (1ULL << 18) | (1ULL << 10) | (1ULL << 11) | (1ULL << 19)),
        Move2(1ULL << 31, 1ULL << 29, PieceColor::Black, (1ULL << 27) | (1ULL << 18) | (1ULL << 10) | (1ULL << 11) | (1ULL << 19)),
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::Black, &moveQueue);
    verifyMoveList2("Chain capture - multiple (black", expectedBlack2, &moveQueue);
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

    std::vector<Move2> expected2 = {
        Move2(1ULL << 21, 1ULL << 12, PieceColor::White, 1ULL << 17),
        Move2(1ULL << 21, 1ULL << 8, PieceColor::White, 1ULL << 17),
        Move2(1ULL << 21, 1ULL << 30, PieceColor::White, 1ULL << 26)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::White, &moveQueue);
    verifyMoveList2("King capturing moves", expected2, &moveQueue);
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
    MoveList expected = {
        Move({1ULL << 14, 1ull << 21, 1ULL << 12}, 1ULL << 17 | 1ULL << 18, PieceColor::Black),
        Move({1ULL << 13, 1ull << 22, 1ULL << 29}, 1ULL << 18 | 1ULL << 26, PieceColor::Black),
        Move({1ULL << 13, 1ull << 22, 1ULL << 15}, 1ULL << 18 | 1ULL << 19, PieceColor::Black)
    };
    verifyMoveList("King capturing moves", expected, moveGen.generateMoves(boardAfterMove, PieceColor::Black));

    std::vector<Move2> expected2 = {
        Move2(1ULL << 14, 1ULL << 12, PieceColor::Black, 1ULL << 17 | 1ULL << 18),
        Move2(1ULL << 13, 1ULL << 29, PieceColor::Black, 1ULL << 18 | 1ULL << 26),
        Move2(1ULL << 13, 1ULL << 15, PieceColor::Black, 1ULL << 18 | 1ULL << 19)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::Black, &moveQueue);
    verifyMoveList2("King capturing moves", expected2, &moveQueue);
}

void MoveGenerationTest::assertFailedTest2()
{
	setUp();
	boardAfterMove.whitePawns = (1ULL << 3);
	boardAfterMove.kings = (1ULL << 3) | (1ULL << 13);
	boardAfterMove.blackPawns = (1ULL << 13);
	MoveList expected = {
		Move(1ULL << 3, 1ULL << 17, 1ULL << 13),
		Move(1ULL << 3, 1ULL << 20, 1ULL << 13),
		Move(1ULL << 3, 1ULL << 24, 1ULL << 13)
	};
	verifyMoveList("assertion failed scenario", expected, moveGen.generateMoves(boardAfterMove, PieceColor::White));

    std::vector<Move2> expected2 = {
       Move2(1ULL << 3, 1ULL << 17, PieceColor::White, 1ULL << 13),
        Move2(1ULL << 3, 1ULL << 20, PieceColor::White, 1ULL << 13),
        Move2(1ULL << 3, 1ULL << 24, PieceColor::White, 1ULL << 13)
    };
    moveQueue.clear();
    moveGen.generateMovesGpu(boardAfterMove, PieceColor::White, &moveQueue);
    verifyMoveList2("King capturing moves", expected2, &moveQueue);
}

void MoveGenerationTest::simulationMoveGenerationTest()
{
    PieceColor currentMoveColor = PieceColor::White;
    Board newBoard = Board(INIT_WHITE_PAWNS, INIT_BLACK_PAWNS, 0);

    int noCaptureMoves = 0;
    std::ofstream debugLog(SIMULATION_LOG);
    if (!debugLog.is_open()) {
        throw std::runtime_error("Failed to open log file for writing.");
    }

    int result = DRAW;

    // Queue for gpu generated moves
	Move2 movesArray[QUEUE_SIZE];
	Queue<Move2> moveQueue = Queue<Move2>(movesArray, QUEUE_SIZE);
	bool areMovesTheSame = true;

    while (true)
    {
        MoveList moves = newBoard.getAvailableMoves(currentMoveColor);
		moveQueue.clear();
		MoveGenerator::generateMovesGpu(newBoard._pieces, currentMoveColor, &moveQueue);

        // Validation
		int queueLength = moveQueue.length();
		areMovesTheSame = true;
        

        if (queueLength != moves.size()) {
			std::cout << "Error: Move list size mismatch. cpu: " << moves.size() << ", gpu: " << queueLength << std::endl;
			areMovesTheSame = false;
        }

        for (int i = 0; i < queueLength; i++) {
			Move2 gpuMove = moveQueue[i];
            bool foundMove = false;
            for(Move cpuMove: moves) {
                if (cpuMove.getSource() == gpuMove.src &&
                    cpuMove.getDestination() == gpuMove.dst &&
                    cpuMove.getCaptured() == gpuMove.captured)
					foundMove = true;
            }
            if (!foundMove) {
				std::cout << "Error: Move not found in cpu list: " << gpuMove.toString() << std::endl;
				areMovesTheSame = false;
            }
        }

        for (int i = 0; i < queueLength; i++) {
            Move2 gpuMove = moveQueue[i];
            debugLog << "gpuMove:" << gpuMove.toString() << std::endl;
        }

        for (Move cpuMove : moves) {
            debugLog << "cpuMove:" << cpuMove.toString() << std::endl;
        }

        if (!areMovesTheSame)
        {   
            break;
        }      

        // No moves available - game is over
        if (moves.empty()) {
            result = currentMoveColor == PieceColor::White ? BLACK_WIN : WHITE_WIN;
            break;
        }

        // Check if the no capture moves limit has beeen exceeded
        if (noCaptureMoves >= MAX_NO_CAPTURE_MOVES) {
            result = DRAW;
            break;
        }

        // Random number generation
        std::random_device rd; // Seed
        std::mt19937 gen(rd()); // Mersenne Twister engine
        std::uniform_int_distribution<> dist(0, moves.size() - 1);

        // Select a random move
        int randomIndex = dist(gen);
        Move randomMove = moves[randomIndex];

        // Check if the move is a capture
        if (!randomMove.isCapture() && (randomMove.getSource() & newBoard.getKings()) > 0) {
            noCaptureMoves++;
        }
        else {
            noCaptureMoves = 0;
        }

        newBoard = newBoard.getBoardAfterMove(randomMove);
        currentMoveColor = getEnemyColor(currentMoveColor);

        debugLog << "Chosen move: " << randomMove.toString() << std::endl;
        debugLog << "Updated board state:\n" << newBoard.toString() << std::endl;
        debugLog << "Kings: \n" << std::hex << newBoard.getKings() << std::endl;
    }

	if (areMovesTheSame) {
        std::cout << "Simulation test: PASSED" << std::endl;
	}
    else
    {
		std::cout << "Simulation test: FAILED" << std::endl;
    }
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
    testKingCapturingMoves3();
    testCrowningMove();
    testKingCapturingMovesEdgeCase();

	assertFailedTest();
	assertFailedTest2();

	simulationMoveGenerationTest();

    printSummary("Move Generation");
}
