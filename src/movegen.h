/*
 * Pufferfish Chess Engine
 * Move generation
 */

#ifndef PUFFERFISH_MOVEGEN_H
#define PUFFERFISH_MOVEGEN_H

#include "position.h"
#include "move.h"
#include <vector>

namespace pufferfish
{

    // Generate all pseudo-legal moves (may leave king in check)
    void generate_pseudo_moves(const Position &pos, std::vector<Move> &moves);

    // Generate all legal moves (filtered for king safety)
    void generate_legal_moves(Position &pos, std::vector<Move> &moves);

    // Generate only capture moves (for quiescence search later)
    void generate_captures(const Position &pos, std::vector<Move> &moves);

} // namespace pufferfish

#endif // PUFFERFISH_MOVEGEN_H
