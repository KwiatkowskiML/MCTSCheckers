#pragma once

enum class PieceColor {
    White,
    Black
};

constexpr PieceColor getEnemyColor(PieceColor playerColor)
{
    return playerColor == PieceColor::White ? PieceColor::Black : PieceColor::White;
}