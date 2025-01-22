#pragma once
#include "JumpersTest.h"
#include "MoversTest.h"
#include "MoveGenerationTest.h"

class CheckersTestSuite {
public:
    static void runAll() {
        printf("=== Starting Checkers Test Suite ===\n\n");

        MoversTest moversTest;
        printf("Running Movers Tests...\n");
        moversTest.runAllTests();

        JumpersTest jumpersTest;
        printf("Running Jumpers Tests...\n");
        jumpersTest.runAllTests();

		MoveGenerationTest moveGenTest;
		printf("Running Move Generation Tests...\n");
		moveGenTest.runAllTests();

        printf("=== Checkers Test Suite Completed ===\n");
    }
};
