#pragma once
#include <stdio.h>
#include "Types.h"
#include "Board2.h"

class TestUtils
{
public:
    static void printTestResult(const char* testName, bool passed, UINT expected, UINT actual) {
        printf("Test %s: %s\n", testName, passed ? "PASSED" : "FAILED");
        if (!passed) {
            printf("Expected: %u\n", expected);
            Board2::printBitboard(expected);
            printf("Actual: %u\n", actual);
            Board2::printBitboard(actual);
        }
    }
};

