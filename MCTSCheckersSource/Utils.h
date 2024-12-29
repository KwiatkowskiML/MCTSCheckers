#pragma once
#include <iostream>
#include <cstdint>
#include <queue>

#define UINT uint32_t

#define MOVES_UP_RIGHT_AVAILABLE 0x00E0E0E0 // +5
#define MOVES_UP_LEFT_AVAILABLE 0x07070707 // +3
#define MOVES_DOWN_RIGHT_AVAILABLE 0xE0E0E0E0 // -3
#define MOVES_DOWN_LEFT_AVAILABLE 0x07070700 // -5

#define UP_RIGHT_SHIFT 5
#define UP_LEFT_SHIFT 3
#define DOWN_RIGHT_SHIFT 3
#define DOWN_LEFT_SHIFT 5

#define BASE_DIAGONAL_SHIFT 4