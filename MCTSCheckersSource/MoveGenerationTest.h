#ifndef MOVE_GENERATION_TEST_H
#define MOVE_GENERATION_TEST_H

#include "CheckersTest.h"

class MoveGenerationTest : public CheckersTest {
private:
    bool verifyMoveList(const char* testName, const MoveList& expected, const MoveList& actual);

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

public:
    void runAllTests() override;
};

#endif // MOVE_GENERATION_TEST_H
