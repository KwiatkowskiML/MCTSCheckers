#pragma once
#include <iostream>
#include <cstdint>
#include <queue>

#define UINT uint32_t

#define MASK_L5 0x00707070 // +5
#define MASK_L3 0x0E0E0E0E // +3
#define MASK_R3 0x70707070 // -3
#define MASK_R5 0x0E0E0E00 // -5

#define SHIFT_BASE 4
#define SHIFT_L5 5
#define SHIFT_L3 3
#define SHIFT_R3 3
#define SHIFT_R5 5

#define WHITE_CROWNING 0x0000000F
#define BLACK_CROWNING 0xF0000000
#define INIT_WHITE_PAWNS 0xFFF00000
#define INIT_BLACK_PAWNS 0x00000FFF

#define LOOSE -1
#define DRAW 0
#define WIN 1

#define WHITE_WIN 10
#define BLACK_WIN 11

#define MAX_NO_CAPTURE_MOVES 10
#define DEFAULT_TIME_LIMIT 3000 // in miliseconds

#define C_UCT 2 // Exploration parameter

#define GAME_LOG_FILE "game.log"
#define SIMULATION_LOG "simulation.log"
#define TREE_VISUALIZATION_FILE "C:\\Users\\Kmicha\\studia\\sem5\\gpu\\p2\\MCTSCheckers\\tree.dot"
#define TREE_VISUALIZATION_PREFIX "C:\\Users\\Kmicha\\studia\\sem5\\gpu\\p2\\MCTSCheckers\\tree_logs\\tree_iteration_"
#define TREE_GAME_LOG_PREFIX "C:\\Users\\Kmicha\\studia\\sem5\\gpu\\p2\\MCTSCheckers\\game_logs\\tree_iteration_"

#define NUMBER_OF_BLOCKS 1
#define THREADS_PER_BLOCK 100

#define QUEUE_SIZE 96