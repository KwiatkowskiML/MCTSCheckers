#pragma once
#include "Types.h"
#include "CheckersTest.h"

class MoversTest : public CheckersTest {
private:
    bool testMover(const char* testName, UINT expectedMovers, PieceColor color);
	bool testMoverGpu(const char* testName, UINT expectedMovers, PieceColor color);

public:
    void runAllTests() override;
};
