#pragma once
#include "Types.h"
#include "PieceColor.h"
#include "cuda_runtime.h"
#include "stdio.h"

struct BitBoard {
    UINT whitePawns;
    UINT blackPawns;
    UINT kings;

    __device__ __host__ BitBoard(UINT white = 0, UINT black = 0, UINT k = 0)
        : whitePawns(white), blackPawns(black), kings(k) {
    }

	// Getters
    __device__ __host__ UINT getAllPieces() const { return whitePawns | blackPawns; }
    __device__ __host__ UINT getEmptyFields() const { return ~getAllPieces(); }
    __device__ __host__ UINT getEnemyPieces(PieceColor playerColor) const { return static_cast<UINT>(playerColor)^1; }
    __device__ __host__ UINT getPieces(PieceColor playerColor) const { return playerColor == PieceColor::White ? whitePawns : blackPawns; }

    // Useless
    __device__ __host__ void print() {
        printf("     A   B   C   D   E   F   G   H\n");
        printf("   +---+---+---+---+---+---+---+---+\n");

        for (int row = 0; row < 8; row++) {
            printf(" %d |", 8 - row);

            for (int col = 0; col < 8; col++) {
                // Only dark squares can have pieces
                int isDarkSquare = (row + col) % 2 != 0;

                if (!isDarkSquare) {
                    printf("   |");  // Light square - always empty
                }
                else {
                    // Calculate bit position for dark squares (bottom to top, left to right)
                    int darkSquareNumber = (7 - row) * 4 + (col / 2);
                    unsigned int mask = 1u << darkSquareNumber;

                    // Check if square has a piece
                    if (whitePawns & mask) {
                        printf(" %c |", (kings & mask) ? 'W' : 'w');
                    }
                    else if (blackPawns & mask) {
                        printf(" %c |", (kings & mask) ? 'B' : 'b');
                    }
                    else {
                        printf("   |");  // Empty dark square
                    }
                }
            }

            printf(" %d\n", 8 - row);
            printf("   +---+---+---+---+---+---+---+---+\n");  // Horizontal separator
        }

        printf("     A   B   C   D   E   F   G   H\n\n");
    }
};