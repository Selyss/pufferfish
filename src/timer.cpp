/*
 * Pufferfish Chess Engine
 * Timer implementation
 */

#include "timer.h"
#include <algorithm>

namespace pufferfish
{

    // =============================================================================
    // Timer Implementation
    // =============================================================================

    Timer::Timer()
        : start_time_(std::chrono::steady_clock::now())
    {
    }

    void Timer::start()
    {
        start_time_ = std::chrono::steady_clock::now();
    }

    uint64_t Timer::elapsed_ms() const
    {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
        return elapsed.count();
    }

    void Timer::reset()
    {
        start_time_ = std::chrono::steady_clock::now();
    }

    bool Timer::time_exceeded(uint64_t limit_ms) const
    {
        return elapsed_ms() >= limit_ms;
    }

    uint64_t Timer::remaining_ms(uint64_t limit_ms) const
    {
        uint64_t elapsed = elapsed_ms();
        if (elapsed >= limit_ms)
            return 0;
        return limit_ms - elapsed;
    }

    // =============================================================================
    // SearchTimeManager Implementation
    // =============================================================================

    uint64_t SearchTimeManager::get_time_budget() const
    {
        switch (mode)
        {
        case FIXED_DEPTH:
            return UINT64_MAX; // No time limit for depth-based search
        case FIXED_TIME:
            return time_ms;
        case TIME_PER_MOVE:
            // Allocate 1/moves_remaining of remaining clock
            // For now, simple heuristic: allocate a fraction of time
            if (moves_remaining == 0)
                return 1000; // Fallback: 1 second
            return std::max(1ULL, (1000ULL / static_cast<uint64_t>(moves_remaining)));
        case INFINITE:
            return UINT64_MAX;
        default:
            return 1000;
        }
    }

    std::string SearchTimeManager::description() const
    {
        switch (mode)
        {
        case FIXED_DEPTH:
            return "Fixed depth: " + std::to_string(depth);
        case FIXED_TIME:
            return "Fixed time: " + std::to_string(time_ms) + "ms";
        case TIME_PER_MOVE:
            return "Time per move: " + std::to_string(moves_remaining) + " moves remaining";
        case INFINITE:
            return "Infinite (no time limit)";
        default:
            return "Unknown mode";
        }
    }

} // namespace pufferfish
