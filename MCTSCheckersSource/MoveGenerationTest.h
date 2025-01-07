#include "Move.h"
#include "CheckersTest.h"

class MoveGenerationTest : public CheckersTest {
private:
    bool verifyMoveList(const char* testName, const MoveList& expected, const MoveList& actual);

public:
    void runAllTests() override;
};