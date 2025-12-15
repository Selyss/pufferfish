/*
 * Pufferfish Chess Engine
 * Transposition Table Unit Tests
 */

#include "../src/transposition_table.h"
#include "../src/move.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

using namespace pufferfish;

// Test 1: Basic store and lookup
bool test_basic_store_lookup()
{
    std::cout << "\n=== Test 1: Basic Store/Lookup ===\n";

    TranspositionTable tt(1);  // 1 MB TT

    uint64_t key = 0x123456789ABCDEF0ULL;
    Move m = Move(SQ_E2, SQ_E4, MOVE_DOUBLE_PUSH);

    // Initially, lookup should fail
    if (tt.lookup(key, 10) != nullptr)
    {
        std::cout << "[FAIL] Expected empty lookup\n";
        return false;
    }

    // Store an entry
    tt.store(key, 50, 10, BOUND_EXACT, m);

    // Lookup should now succeed
    const TTEntry* entry = tt.lookup(key, 10);
    if (entry == nullptr)
    {
        std::cout << "[FAIL] Expected entry after store\n";
        return false;
    }

    // Verify entry contents
    if (entry->score != 50 || entry->depth != 10 || entry->bound_type != BOUND_EXACT)
    {
        std::cout << "[FAIL] Entry contents mismatch\n";
        return false;
    }

    std::cout << "[PASS] Basic store/lookup works\n";
    return true;
}

// Test 2: Depth-aware lookup (shallow entries should not be used for deeper searches)
bool test_depth_filtering()
{
    std::cout << "\n=== Test 2: Depth Filtering ===\n";

    TranspositionTable tt(1);
    uint64_t key = 0x123456789ABCDEF0ULL;
    Move m = Move(SQ_E2, SQ_E4, MOVE_DOUBLE_PUSH);

    // Store at depth 5
    tt.store(key, 50, 5, BOUND_EXACT, m);

    // Lookup at depth 10 should fail (need deeper search result)
    if (tt.lookup(key, 10) != nullptr)
    {
        std::cout << "[FAIL] Shallow entry should not be used at deeper search\n";
        return false;
    }

    // Lookup at depth 5 should succeed
    if (tt.lookup(key, 5) == nullptr)
    {
        std::cout << "[FAIL] Entry should be found at same depth\n";
        return false;
    }

    // Lookup at depth 3 should succeed (shallow search can use deep results)
    if (tt.lookup(key, 3) == nullptr)
    {
        std::cout << "[FAIL] Shallow search should use deep results\n";
        return false;
    }

    std::cout << "[PASS] Depth filtering works correctly\n";
    return true;
}

// Test 3: Replacement strategy (deeper searches overwrite shallower ones)
bool test_replacement_strategy()
{
    std::cout << "\n=== Test 3: Replacement Strategy ===\n";

    TranspositionTable tt(1);
    uint64_t key = 0x123456789ABCDEF0ULL;
    Move m1 = Move(SQ_E2, SQ_E4, MOVE_DOUBLE_PUSH);
    Move m2 = Move(SQ_D2, SQ_D4, MOVE_DOUBLE_PUSH);

    // Store shallow search result
    tt.store(key, 50, 5, BOUND_EXACT, m1);
    const TTEntry* entry1 = tt.lookup(key, 5);
    if (entry1 == nullptr || entry1->best_move != m1)
    {
        std::cout << "[FAIL] Initial store failed\n";
        return false;
    }

    // Store deeper search result (should replace)
    tt.store(key, 75, 10, BOUND_EXACT, m2);
    const TTEntry* entry2 = tt.lookup(key, 10);
    if (entry2 == nullptr || entry2->best_move != m2 || entry2->score != 75)
    {
        std::cout << "[FAIL] Deeper search did not replace shallow search\n";
        return false;
    }

    std::cout << "[PASS] Replacement strategy works\n";
    return true;
}

// Test 4: Different bound types (EXACT, LOWER, UPPER)
bool test_bound_types()
{
    std::cout << "\n=== Test 4: Bound Types ===\n";

    TranspositionTable tt(1);
    Move m = Move(SQ_E2, SQ_E4, MOVE_DOUBLE_PUSH);

    std::vector<std::pair<uint64_t, BoundType>> tests = {
        {0x1111111111111111ULL, BOUND_EXACT},
        {0x2222222222222222ULL, BOUND_LOWER},
        {0x3333333333333333ULL, BOUND_UPPER},
    };

    for (const auto& [key, bound] : tests)
    {
        tt.store(key, 100, 10, bound, m);
        const TTEntry* entry = tt.lookup(key, 10);

        if (entry == nullptr || entry->bound_type != bound)
        {
            std::cout << "[FAIL] Bound type mismatch for bound " << (int)bound << "\n";
            return false;
        }
    }

    std::cout << "[PASS] All bound types stored correctly\n";
    return true;
}

// Test 5: Hash collisions and statistics
bool test_collision_handling()
{
    std::cout << "\n=== Test 5: Collision Handling ===\n";

    TranspositionTable tt(1);  // Small TT to force collisions
    Move m = Move(SQ_E2, SQ_E4, MOVE_DOUBLE_PUSH);

    // Store multiple entries that might collide
    for (int i = 0; i < 100; ++i)
    {
        uint64_t key = 0x1234567890ABCDEFULL + i;
        tt.store(key, 50 + i, 10, BOUND_EXACT, m);
    }

    // Verify some entries are still accessible
    const TTEntry* entry = tt.lookup(0x1234567890ABCDEFULL, 10);
    if (entry == nullptr)
    {
        std::cout << "[FAIL] Entry lost due to collision handling\n";
        return false;
    }

    std::cout << "[PASS] Collision handling works\n";
    std::cout << "       Entries stored: " << tt.entries_stored() << "\n";
    std::cout << "       Entries probed: " << tt.entries_probed() << "\n";
    std::cout << "       Entries hit: " << tt.entries_hit() << "\n";
    std::cout << "       Hit rate: " << std::fixed << std::setprecision(1) << tt.hit_rate() << "%\n";
    std::cout << "       Collisions detected: " << tt.entries_collision() << "\n";

    return true;
}

// Test 6: Clear operation
bool test_clear()
{
    std::cout << "\n=== Test 6: Clear Operation ===\n";

    TranspositionTable tt(1);
    Move m = Move(SQ_E2, SQ_E4, MOVE_DOUBLE_PUSH);

    // Store some entries
    for (int i = 0; i < 10; ++i)
    {
        tt.store(0x1000000000000000ULL + i, 50, 10, BOUND_EXACT, m);
    }

    uint64_t initial_stored = tt.entries_stored();

    // Clear and verify
    tt.clear();

    if (tt.entries_stored() != 0 || tt.entries_collision() != 0)
    {
        std::cout << "[FAIL] Storage statistics not reset after clear\n";
        return false;
    }

    // Lookup should fail after clear (no need to check probed stats since they track probe calls)
    if (tt.lookup(0x1000000000000000ULL, 10) != nullptr)
    {
        std::cout << "[FAIL] Entry found after clear\n";
        return false;
    }

    std::cout << "[PASS] Clear works correctly\n";
    std::cout << "       Cleared " << initial_stored << " entries\n";

    return true;
}

// Test 7: Performance with many entries
bool test_performance()
{
    std::cout << "\n=== Test 7: Performance Test ===\n";

    TranspositionTable tt(64);  // 64 MB
    Move m = Move(SQ_E2, SQ_E4, MOVE_DOUBLE_PUSH);

    std::mt19937_64 rng(12345);

    // Store 100k random entries
    const int NUM_ENTRIES = 100000;
    std::vector<uint64_t> keys;

    for (int i = 0; i < NUM_ENTRIES; ++i)
    {
        uint64_t key = rng();
        keys.push_back(key);
        tt.store(key, 50 + (i % 100), 10, BOUND_EXACT, m);
    }

    // Lookup all stored entries
    int found = 0;
    for (uint64_t key : keys)
    {
        if (tt.lookup(key, 10) != nullptr)
            ++found;
    }

    double hit_rate = (100.0 * found) / keys.size();

    std::cout << "[PASS] Performance test completed\n";
    std::cout << "       Stored: " << NUM_ENTRIES << " entries\n";
    std::cout << "       Found: " << found << " entries\n";
    std::cout << "       Hit rate: " << std::fixed << std::setprecision(1) << hit_rate << "%\n";
    std::cout << "       Capacity: " << (tt.size_bytes() / 1024 / 1024) << " MB\n";
    std::cout << "       TT Hit rate in probes: " << std::fixed << std::setprecision(1) << tt.hit_rate() << "%\n";

    if (hit_rate < 90.0)
    {
        std::cout << "[WARN] Hit rate lower than expected\n";
    }

    return true;
}

int main()
{
    std::cout << "========================================\n"
              << "  Transposition Table Unit Tests\n"
              << "========================================\n";

    bool test1 = test_basic_store_lookup();
    bool test2 = test_depth_filtering();
    bool test3 = test_replacement_strategy();
    bool test4 = test_bound_types();
    bool test5 = test_collision_handling();
    bool test6 = test_clear();
    bool test7 = test_performance();

    bool all_pass = test1 && test2 && test3 && test4 && test5 && test6 && test7;

    std::cout << "\n========================================\n"
              << "Results: " << (all_pass ? "ALL PASSED" : "SOME FAILED") << "\n"
              << "========================================\n";

    return all_pass ? 0 : 1;
}
