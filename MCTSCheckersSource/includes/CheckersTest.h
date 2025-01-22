#pragma once
#include "BitBoard.h"
#include "MoveGenerator.h"

class CheckersTest {
protected:
    BitBoard boardAfterMove;
    MoveGenerator moveGen;
    int passedTests = 0;
    int totalTests = 0;

    void setUp();
    bool verifyTest(const char* testName, UINT expected, UINT actual);

public:
    virtual void runAllTests() = 0;
    void printSummary(const char* testType);
};