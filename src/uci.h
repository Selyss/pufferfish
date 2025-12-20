/*
 * Pufferfish Chess Engine
 * UCI (Universal Chess Interface) Protocol Handler
 */

#ifndef PUFFERFISH_UCI_H
#define PUFFERFISH_UCI_H

#include "position.h"
#include "search.h"
#include "timer.h"
#include "move.h"
#include <string>
#include <vector>
#include <memory>

namespace pufferfish
{

    // =============================================================================
    // UCI Command Parser and Handler
    // =============================================================================
    // Implements the UCI protocol for communicating with GUI/analysis tools.
    //
    // Supported commands:
    //   - uci: Identify engine
    //   - isready: Check if ready
    //   - position: Set up position (FEN or startpos + moves)
    //   - go: Start search with time/depth constraints
    //   - stop: Halt search
    //   - quit: Exit program

    class UCIHandler
    {
    public:
        // Constructor
        UCIHandler();

        // Main command loop - reads from stdin and processes commands
        void run();

        // Parse and execute a single command
        bool process_command(const std::string &line);

        // Get current position
        const Position &position() const { return position_; }

        // Get current search engine
        const Search &search() const { return search_; }

        // Check if should continue running
        bool is_running() const { return running_; }

    private:
        Position position_;
        Search search_;
        bool running_ = true;

        // Command handlers
        void handle_uci();
        void handle_isready();
        void handle_position(const std::string &args);
        void handle_go(const std::string &args);
        void handle_stop();
        void handle_quit();

        // Helper: Parse position command arguments
        struct PositionArgs
        {
            bool use_fen = false;
            std::string fen;
            std::vector<std::string> moves;
        };
        PositionArgs parse_position_args(const std::string &args);

        // Helper: Parse go command arguments
        struct GoArgs
        {
            SearchTimeManager::Mode mode = SearchTimeManager::FIXED_DEPTH;
            int depth = 4;
            uint64_t wtime = 0;    // White time in ms
            uint64_t btime = 0;    // Black time in ms
            uint64_t winc = 0;     // White increment
            uint64_t binc = 0;     // Black increment
            uint64_t movetime = 0; // Fixed move time
            uint64_t movestogo = 0;
        };
        GoArgs parse_go_args(const std::string &args);

        // Helper: Get time budget based on go command arguments
        SearchTimeManager get_time_manager(const GoArgs &args);

        // Helper: Split string by spaces
        std::vector<std::string> split(const std::string &s);
    };

} // namespace pufferfish

#endif // PUFFERFISH_UCI_H
