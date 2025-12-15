/*
 * Pufferfish Chess Engine
 * Transposition Table for search
 */

#ifndef PUFFERFISH_TRANSPOSITION_TABLE_H
#define PUFFERFISH_TRANSPOSITION_TABLE_H

#include "types.h"
#include "move.h"
#include <cstdint>
#include <vector>
#include <cstring>

namespace pufferfish
{

    // =============================================================================
    // Transposition Table Entry
    // =============================================================================
    // Stores a cached position evaluation from search.
    //
    // Fields:
    //   - hash: Lower 32 bits of Zobrist key (for collision detection)
    //   - depth: Search depth at which this position was evaluated
    //   - score: Evaluation score from search (or bounds)
    //   - bound_type: EXACT, LOWER_BOUND, or UPPER_BOUND
    //   - best_move: Best move found in this position (for move ordering)

    enum BoundType : uint8_t
    {
        BOUND_EXACT = 0,
        BOUND_LOWER = 1,  // score >= value (alpha cutoff)
        BOUND_UPPER = 2   // score <= value (beta cutoff)
    };

    struct TTEntry
    {
        uint32_t hash;        // Lower 32 bits of Zobrist key
        int16_t score;        // Score in centipawns
        uint8_t depth;        // Search depth
        uint8_t bound_type;   // EXACT, LOWER, or UPPER
        Move best_move;       // Best move found

        // Check if this entry matches the given Zobrist key
        bool matches(uint64_t zobrist_key) const
        {
            return hash == (zobrist_key & 0xFFFFFFFFULL);
        }

        // Check if entry is valid for this depth
        bool is_valid_for_depth(int search_depth) const
        {
            return depth >= search_depth;
        }
    };

    // =============================================================================
    // Transposition Table
    // =============================================================================
    // Hash table for caching search results. Uses direct hashing (no chaining)
    // with Zobrist keys.
    //
    // Size: Configurable, typically 64 MB to 1 GB
    // Default: ~1 million entries * 16 bytes = 16 MB

    class TranspositionTable
    {
    public:
        // Constructor: size_mb is the target size in megabytes
        explicit TranspositionTable(size_t size_mb = 16);

        // Destructor
        ~TranspositionTable();

        // Store a position evaluation in the TT
        void store(uint64_t zobrist_key, int score, int depth, BoundType bound, Move best_move);

        // Retrieve a position evaluation from the TT
        // Returns nullptr if not found or not valid for current depth
        const TTEntry* lookup(uint64_t zobrist_key, int current_depth) const;

        // Clear the entire transposition table
        void clear();

        // Statistics
        uint64_t entries_stored() const { return entries_stored_; }
        uint64_t entries_probed() const { return entries_probed_; }
        uint64_t entries_hit() const { return entries_hit_; }
        uint64_t entries_collision() const { return entries_collision_; }

        double hit_rate() const
        {
            return entries_probed_ > 0 ? (100.0 * entries_hit_ / entries_probed_) : 0.0;
        }

        // Capacity
        size_t capacity() const { return table_.size(); }
        size_t size_bytes() const { return table_.size() * sizeof(TTEntry); }

    private:
        std::vector<TTEntry> table_;
        mutable uint64_t entries_stored_;
        mutable uint64_t entries_probed_;
        mutable uint64_t entries_hit_;
        mutable uint64_t entries_collision_;

        // Hash function: use lower bits of Zobrist key
        size_t hash_index(uint64_t zobrist_key) const
        {
            return (zobrist_key & 0xFFFFFFFFULL) % table_.size();
        }
    };

} // namespace pufferfish

#endif // PUFFERFISH_TRANSPOSITION_TABLE_H
