#pragma once
#include "BitBoard.h"
#include "MoveGenerator.h"
#include "CheckersTest.h"

class JumpersTest : public CheckersTest {
private:
    bool testJumper(const char* testName, UINT expectedJumpers);

public:
    void runAllTests() override;
};