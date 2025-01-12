#pragma once
#include "Types.h"
#include "CheckersTest.h"

class MoversTest : public CheckersTest {
private:
    bool testWhiteMover(const char* testName, UINT expectedMovers);
    bool testBlackMover(const char* testName, UINT expectedMovers);

public:
    void runAllTests() override;
};
