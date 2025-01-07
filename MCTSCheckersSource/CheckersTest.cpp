#include "CheckersTest.h"
#include "TestUtils.h"

void CheckersTest::setUp()
{
    board.whitePawns = 0;
    board.blackPawns = 0;
    board.kings = 0;
}

bool CheckersTest::verifyTest(const char* testName, UINT expected, UINT actual)
{
    bool passed = (actual == expected);
    TestUtils::printTestResult(testName, passed, expected, actual);
    return passed;
}

void CheckersTest::printSummary(const char* testType)
{
    printf("\n%s Test Summary: %d/%d tests passed\n\n",
        testType, passedTests, totalTests);
}
