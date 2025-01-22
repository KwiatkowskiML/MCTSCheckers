#include "../includes/CheckersTest.h"
#include "../includes/TestUtils.h"

void CheckersTest::setUp()
{
    boardAfterMove.whitePawns = 0;
    boardAfterMove.blackPawns = 0;
    boardAfterMove.kings = 0;
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
