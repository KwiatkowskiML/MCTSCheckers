#pragma once
#include <gtest/gtest.h>
#include "MoveGenerator.h"

class JumperGeneratorTest : public ::testing::Test {
protected:
    MoveGenerator moveGenerator;
    BitBoard board;

    void SetUp() override {
        board.whitePawns = 0;
        board.blackPawns = 0;
        board.kings = 0;
    }
};

