// Zobrist hashing verification test harness
#include "../src/position.h"
#include "../src/zobrist.h"
#include "../src/movegen.h"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>

using namespace pufferfish;

struct ZobristTest
{
    const char *name;
    const char *fen;
    // Keys that should be consistent across runs (same FEN => same key)
};

// Test 1: Determinism - same FEN always produces same key
bool test_zobrist_determinism()
{
    std::cout << "\n=== Test 1: Zobrist Determinism ===\n";

    std::vector<ZobristTest> tests = {
        {"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
        {"empty board", "8/8/8/8/8/8/8/8 w - - 0 1"},
        {"with ep square", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e3 0 1"},
        {"limited castling", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1"},
    };

    bool all_pass = true;
    for (const auto &test : tests)
    {
        Position pos1, pos2;
        pos1.set_fen(test.fen);
        pos2.set_fen(test.fen);

        uint64_t key1 = pos1.zobrist_key();
        uint64_t key2 = pos2.zobrist_key();

        bool match = (key1 == key2);
        std::cout << (match ? "[PASS] " : "[FAIL] ")
                  << std::left << std::setw(20) << test.name
                  << " 0x" << std::hex << std::setw(16) << key1 << std::dec << "\n";

        if (!match)
            all_pass = false;
    }
    return all_pass;
}

// Test 2: Make/unmake preserves key
bool test_zobrist_make_unmake()
{
    std::cout << "\n=== Test 2: Zobrist Make/Unmake Consistency ===\n";

    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Generate and play first move
    std::vector<Move> moves;
    generate_legal_moves(pos, moves);

    if (moves.empty())
    {
        std::cout << "[FAIL] No legal moves in startpos\n";
        return false;
    }

    bool all_pass = true;
    for (int i = 0; i < std::min(5, (int)moves.size()); ++i)
    {
        const Move &m = moves[i];
        Undo u;

        uint64_t before = pos.zobrist_key();
        pos.make_move(m, u);
        uint64_t after_make = pos.zobrist_key();
        pos.unmake_move(m, u);
        uint64_t after_unmake = pos.zobrist_key();

        bool key_restored = (before == after_unmake);
        std::cout << (key_restored ? "[PASS] " : "[FAIL] ")
                  << "Move " << (i + 1) << " (0x" << std::hex << before << " -> 0x"
                  << after_make << " -> 0x" << after_unmake << std::dec << ")\n";

        if (!key_restored)
            all_pass = false;
    }

    return all_pass;
}

// Test 3: Verify different positions have different keys
bool test_zobrist_uniqueness()
{
    std::cout << "\n=== Test 3: Zobrist Uniqueness (Different Positions) ===\n";

    std::vector<const char *> positions = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",    // black to move
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",      // limited castling
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", // after 1.e4
    };

    bool all_pass = true;
    for (size_t i = 0; i < positions.size(); ++i)
    {
        for (size_t j = i + 1; j < positions.size(); ++j)
        {
            Position pos1, pos2;
            pos1.set_fen(positions[i]);
            pos2.set_fen(positions[j]);

            uint64_t key1 = pos1.zobrist_key();
            uint64_t key2 = pos2.zobrist_key();

            bool different = (key1 != key2);
            std::cout << (different ? "[PASS] " : "[FAIL] ")
                      << "Pos " << (i + 1) << " vs Pos " << (j + 1) << " are "
                      << (different ? "different" : "SAME") << "\n";

            if (!different)
                all_pass = false;
        }
    }

    return all_pass;
}

int main()
{
    std::cout << "========================================\n"
              << "  Pufferfish Zobrist Hashing Tests\n"
              << "========================================\n";

    bool test1 = test_zobrist_determinism();
    bool test2 = test_zobrist_make_unmake();
    bool test3 = test_zobrist_uniqueness();

    std::cout << "\n========================================\n"
              << "Results: "
              << (test1 && test2 && test3 ? "ALL PASSED" : "SOME FAILED")
              << "\n========================================\n";

    return (test1 && test2 && test3) ? 0 : 1;
}
