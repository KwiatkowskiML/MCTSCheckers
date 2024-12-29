#pragma once
#include <iostream>
#include <cstdint>
#include <queue>

#define UINT uint32_t

#define MOVES_UP_RIGHT_AVAILABLE 0x00E0E0E0 // +5
#define MOVES_UP_LEFT_AVAILABLE 0x07070707 // +3
#define MOVES_DOWN_RIGHT_AVAILABLE 0xE0E0E0E0 // -3
#define MOVES_DOWN_LEFT_AVAILABLE 0x07070700 // -5

#define RIGHT_UP_SHIFT 5
#define LEFT_UP_SHIFT 3
#define RIGHT_DOWN_SHIFT 3
#define LEFT_DOWN_SHIFT 5

#define BASE_DIAGONAL_SHIFT 4