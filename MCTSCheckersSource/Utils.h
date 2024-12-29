#pragma once
#include <iostream>
#include <cstdint>
#include <queue>

#define UINT uint32_t

#define MOVES_UP_RIGHT_AVAILABLE 0x00707070 // +5
#define MOVES_UP_LEFT_AVAILABLE 0x0E0E0E0E // +3
#define MOVES_DOWN_RIGHT_AVAILABLE 0x70707070 // -3
#define MOVES_DOWN_LEFT_AVAILABLE 0x0E0E0E00 // -5

#define UP_RIGHT_SHIFT 5
#define UP_LEFT_SHIFT 3
#define DOWN_RIGHT_SHIFT 3
#define DOWN_LEFT_SHIFT 5

#define BASE_DIAGONAL_SHIFT 4

#define WHITE_CROWNING 0x0000000F
#define BLACK_CROWNING 0xF0000000