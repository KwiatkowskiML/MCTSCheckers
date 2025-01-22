#pragma once
#include "BitBoard.h"
#include "CheckersTest.h"

class JumpersTest : public CheckersTest {
private:
    bool testJumper(const char* testName, UINT expectedJumpers, PieceColor playerColor = PieceColor::White);

public:
    void runAllTests() override;
};