/*
 * Pufferfish Chess Engine
 * Timer utility for time management and search time limits
 */

#ifndef PUFFERFISH_TIMER_H
#define PUFFERFISH_TIMER_H

#include <chrono>
#include <cstdint>

namespace pufferfish
{

    // =============================================================================
    // Timer class
    // =============================================================================
    // Tracks elapsed time for search time management.

    class Timer
    {
    public:
        Timer();

        // Start the timer
        void start();

        // Elapsed time in milliseconds
        uint64_t elapsed_ms() const;

        // Reset elapsed time
        void reset();

        // Check if time limit (in ms) has been exceeded
        bool time_exceeded(uint64_t limit_ms) const;

        // Get remaining time before limit (in ms)
        // Returns 0 if limit exceeded
        uint64_t remaining_ms(uint64_t limit_ms) const;

    private:
        std::chrono::steady_clock::time_point start_time_;
    };

    // =============================================================================
    // SearchTimeManager
    // =============================================================================
    // Manages time allocation for search in different modes.

    struct SearchTimeManager
    {
        // Time limit modes
        enum Mode
        {
            FIXED_DEPTH,   // Search to fixed depth (no time limit)
            FIXED_TIME,    // Search for fixed amount of time
            TIME_PER_MOVE, // Allocate time based on clock
            INFINITE       // Search indefinitely (for testing)
        };

        Mode mode = FIXED_DEPTH;
        int depth = 4;                 // For FIXED_DEPTH mode
        uint64_t time_ms = 0;          // For FIXED_TIME mode
        uint64_t moves_remaining = 50; // For TIME_PER_MOVE mode (assume game continues)

        // Calculate time budget for this move (in ms)
        uint64_t get_time_budget() const;

        // Get description of current settings
        std::string description() const;
    };

} // namespace pufferfish

#endif // PUFFERFISH_TIMER_H
