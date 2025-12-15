/*
 * Pufferfish Chess Engine
 * Transposition Table implementation
 */

#include "transposition_table.h"
#include <cstring>

namespace pufferfish
{

    TranspositionTable::TranspositionTable(size_t size_mb)
        : entries_stored_(0), entries_probed_(0), entries_hit_(0), entries_collision_(0)
    {
        // Calculate number of entries (each entry is 16 bytes)
        size_t num_entries = (size_mb * 1024 * 1024) / sizeof(TTEntry);

        // Align to nearest power of 2 for efficient hashing
        if (num_entries < 8)
            num_entries = 8;

        // Round down to nearest power of 2
        size_t power_of_2 = 1;
        while (power_of_2 * 2 <= num_entries)
            power_of_2 *= 2;

        table_.resize(power_of_2);

        // Initialize all entries to zero (value-initialize)
        for (auto& entry : table_)
        {
            entry.hash = 0;
            entry.score = 0;
            entry.depth = 0;
            entry.bound_type = BOUND_EXACT;
            entry.best_move = Move();
        }
    }

    TranspositionTable::~TranspositionTable() = default;

    void TranspositionTable::store(uint64_t zobrist_key, int score, int depth, BoundType bound, Move best_move)
    {
        size_t idx = hash_index(zobrist_key);
        TTEntry& entry = table_[idx];

        // Overwrite if:
        // 1. Slot is empty (hash == 0)
        // 2. This position (same hash) and we have a deeper search
        // 3. Always overwrite shallow searches with deeper ones (replacement strategy)
        bool should_store = (entry.hash == 0) ||
                            (entry.hash == (zobrist_key & 0xFFFFFFFFULL) && depth > entry.depth) ||
                            (depth > entry.depth);  // Always prefer deeper searches

        if (should_store)
        {
            if (entry.hash != 0 && entry.hash != (zobrist_key & 0xFFFFFFFFULL))
                ++entries_collision_;

            entry.hash = zobrist_key & 0xFFFFFFFFULL;
            entry.score = score;
            entry.depth = depth;
            entry.bound_type = bound;
            entry.best_move = best_move;

            ++entries_stored_;
        }
    }

    const TTEntry* TranspositionTable::lookup(uint64_t zobrist_key, int current_depth) const
    {
        ++entries_probed_;

        size_t idx = hash_index(zobrist_key);
        const TTEntry& entry = table_[idx];

        if (entry.hash == 0)
            return nullptr;  // Empty slot

        if (!entry.matches(zobrist_key))
            return nullptr;  // Hash collision with different position

        if (!entry.is_valid_for_depth(current_depth))
            return nullptr;  // Entry is from shallower search

        ++entries_hit_;
        return &entry;
    }

    void TranspositionTable::clear()
    {
        for (auto& entry : table_)
        {
            entry.hash = 0;
            entry.score = 0;
            entry.depth = 0;
            entry.bound_type = BOUND_EXACT;
            entry.best_move = Move();
        }
        entries_stored_ = 0;
        entries_probed_ = 0;
        entries_hit_ = 0;
        entries_collision_ = 0;
    }

} // namespace pufferfish
