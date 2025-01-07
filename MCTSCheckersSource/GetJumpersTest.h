#pragma once
#include "BitBoard.h"
#include "MoveGenerator.h"

class GetJumpersTest {
private:
    BitBoard board;
    MoveGenerator moveGen;

    void setUp();
    bool testCase(const char* testName, UINT expectedJumpers);

public:
    void runAllTests();
};