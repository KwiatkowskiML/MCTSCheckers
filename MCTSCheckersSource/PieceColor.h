#pragma once

enum class PieceColor {
    White,
    Black
};

constexpr PieceColor getEnemyColor(PieceColor color)
{
    return color == PieceColor::White ? PieceColor::Black : PieceColor::White;
}