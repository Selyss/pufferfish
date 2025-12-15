/*
 * Pufferfish Chess Engine
 * Search - Alpha-beta pruning with transposition table
 */

#ifndef PUFFERFISH_SEARCH_H
#define PUFFERFISH_SEARCH_H

#include "types.h"
#include "move.h"
#include "position.h"
#include "transposition_table.h"
#include "timer.h"
#include "nnue.h"
#include <cstdint>
#include <vector>
#include <memory>

namespace pufferfish
{

    // =============================================================================
    // Search Constants
    // =============================================================================

    // Score boundaries for mating positions
    constexpr int MATE_SCORE = 30000;
    constexpr int MATE_IN_1 = MATE_SCORE - 1;
    constexpr int MATED_IN_1 = -MATE_IN_1;

    // Infinity values for alpha-beta
    constexpr int INF = 32767;
    constexpr int NEG_INF = -32767;

    // =============================================================================
    // Search Statistics
    // =============================================================================

    struct SearchStats
    {
        uint64_t nodes_searched = 0;
        uint64_t leaf_nodes = 0;
        uint64_t tt_hits = 0;
        uint64_t tt_probes = 0;
        uint64_t beta_cutoffs = 0;
        int max_depth = 0;

        double tt_hit_rate() const
        {
            return tt_probes > 0 ? (100.0 * tt_hits / tt_probes) : 0.0;
        }

        void reset()
        {
            nodes_searched = 0;
            leaf_nodes = 0;
            tt_hits = 0;
            tt_probes = 0;
            beta_cutoffs = 0;
            max_depth = 0;
        }
    };

    // =============================================================================
    // Search Engine
    // =============================================================================
    // Implements alpha-beta pruning with transposition table integration.
    //
    // Features:
    //   - Alpha-beta pruning
    //   - Transposition table caching
    //   - Move ordering hints from TT
    //   - Simple material-based evaluation
    //   - Mate detection

    class Search
    {
    public:
        // Constructor with optional TT size in MB
        explicit Search(size_t tt_size_mb = 16);

        // Search for best move at given depth
        Move find_best_move(Position &pos, int depth);

        // Iterative deepening search with time management
        // Returns best move found within time limit
        Move find_best_move_iterative(Position &pos, const SearchTimeManager &time_mgr);

        // Alpha-beta search (internal)
        int alpha_beta(Position &pos, int depth, int alpha, int beta);

        // Simple evaluation function (uses NNUE if available)
        int evaluate(const Position &pos);

        // Mate detection
        static bool is_mating_score(int score);
        static int mates_in(int score);

        // Clear transposition table for new search
        void clear();

        // Statistics access
        const SearchStats &stats() const { return stats_; }

        // Direct access to TT (for debugging/testing)
        TranspositionTable &tt() { return tt_; }

        // Check if NNUE is loaded and ready
        bool nnue_ready() const { return nnue_ && nnue_->is_ready(); }

    private:
        TranspositionTable tt_;
        SearchStats stats_;
        int nodes_at_depth_ = 0;
        Timer timer_;                         // For time management
        std::unique_ptr<NNUEEvaluator> nnue_; // Neural network evaluator

        // Quiet search for tactical positions (future optimization)
        int quiesce(Position &pos, int alpha, int beta);
    };

} // namespace pufferfish

#endif // PUFFERFISH_SEARCH_H
