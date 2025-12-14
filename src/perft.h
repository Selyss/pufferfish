/*
 * Pufferfish Chess Engine
 * Perft and perft divide for validation
 */

#ifndef PUFFERFISH_PERFT_H
#define PUFFERFISH_PERFT_H

#include "position.h"
#include <cstdint>
#include <iostream>

namespace pufferfish
{

    // Count leaf nodes at given depth
    uint64_t perft(Position &pos, int depth);

    // Perft divide: show node counts per root move (for debugging)
    void perft_divide(Position &pos, int depth, std::ostream &os = std::cout);

    // Run perft test suite and return true if all pass
    bool run_perft_suite(std::ostream &os = std::cout);

} // namespace pufferfish

#endif // PUFFERFISH_PERFT_H
