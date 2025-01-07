#pragma once
#include "Types.h"
#include "CheckersTest.h"

class MoversTest : public CheckersTest {
private:
    bool testMover(const char* testName, UINT expectedMovers);

public:
    void runAllTests() override;
};
