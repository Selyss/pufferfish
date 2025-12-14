/*
 * Pufferfish Chess Engine
 * Attack detection
 */

#ifndef PUFFERFISH_ATTACK_H
#define PUFFERFISH_ATTACK_H

#include "types.h"
#include "position.h"

namespace pufferfish
{

    // =============================================================================
    // Attack detection
    // =============================================================================

    // Check if a square is attacked by pieces of the given color
    bool is_square_attacked(const Position &pos, Square sq, Color by);

    // Check if the given side's king is in check
    inline bool in_check(const Position &pos, Color side)
    {
        return is_square_attacked(pos, pos.king_square(side), ~side);
    }

    // Check if side to move is in check
    inline bool in_check(const Position &pos)
    {
        return in_check(pos, pos.side_to_move());
    }

} // namespace pufferfish

#endif // PUFFERFISH_ATTACK_H
