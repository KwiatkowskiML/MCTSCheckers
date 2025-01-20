#ifndef MOVE_GENERATION_TEST_H
#define MOVE_GENERATION_TEST_H

#include "CheckersTest.h"

class MoveGenerationTest : public CheckersTest {
private:
	Move2 moveArray[QUEUE_SIZE];
	Queue<Move2> moveQueue = Queue<Move2>(moveArray, QUEUE_SIZE);

    bool verifyMoveList(const char* testName, const MoveList& expected, const MoveList& actual);
    bool verifyMoveList2(const char* testName, const std::vector<Move2>& expected, Queue<Move2>* actual);

    void testBasicPawnMovesCenter();
    void testBasicPawnMovesLeftEdge();
    void testBasicPawnMovesRightEdge();
    void testSingleCaptureRightDiagonal();
    void testSingleCaptureLeftDiagonal();
    void testChainCaptureDouble();
    void testChainCaptureTriple();
    void testChainCaptureMultiple();
    void testKingBasicMoves();
    void testKingCapturingMoves();
    void testKingCapturingMoves2();
    void testKingCapturingMoves3();
    void testKingCapturingMovesEdgeCase();
    void testCrowningMove();
    void assertFailedTest();
    void assertFailedTest2();

    void simulationMoveGenerationTest();

public:
    void runAllTests() override;
};

#endif // MOVE_GENERATION_TEST_H
