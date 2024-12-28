#pragma once
#include <iostream>
#include <cstdint>
#include <queue>

#define UINT uint32_t

#define MOVES_RIGHT_UP_AVAILABLE 0x0E0E0E00 // +5
#define MOVES_LEFT_UP_AVAILABLE 0x70707070 // +3
#define MOVES_RIGHT_DOWN_AVAILABLE 0x0E0E0E0E // -3
#define MOVES_DOWN_LEFT_AVAILABLE 0x00707070 // -5

#define RIGHT_UP_SHIFT 5
#define LEFT_UP_SHIFT 3
#define RIGHT_DOWN_SHIFT 3
#define LEFT_DOWN_SHIFT 5

#define BASE_DIAGONAL_SHIFT 4