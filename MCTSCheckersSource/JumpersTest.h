#pragma once
#include "BitBoard.h"
#include "MoveGenerator.h"
#include "CheckersTest.h"

class JumpersTest : public CheckersTest {
private:
    bool testJumper(const char* testName, UINT expectedJumpers, PieceColor color = PieceColor::White);

public:
    void runAllTests() override;
};