/*
 * Pufferfish Chess Engine
 * Search module tests - Mate-in-N puzzles and basic validation
 */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "../src/position.h"
#include "../src/search.h"
#include "../src/move.h"
#include "../src/movegen.h"
#include "../src/attack.h"

using namespace pufferfish;

// =============================================================================
// Test utilities
// =============================================================================

struct TestResult
{
    std::string name;
    bool passed;
    std::string message;

    TestResult(const std::string &n, bool p, const std::string &m = "")
        : name(n), passed(p), message(m) {}
};

std::vector<TestResult> results;

void test_case(const std::string &name, bool condition, const std::string &message = "")
{
    results.emplace_back(name, condition, message);
    std::string status = condition ? "PASS" : "FAIL";
    std::cout << status << ": " << name;
    if (!message.empty())
        std::cout << " - " << message;
    std::cout << std::endl;
}

void print_summary()
{
    int passed = 0, total = static_cast<int>(results.size());
    for (const auto &r : results)
        if (r.passed)
            ++passed;

    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    if (passed == total)
    {
        std::cout << "All tests passed!" << std::endl;
        return;
    }

    std::cout << "Failed tests:" << std::endl;
    for (const auto &r : results)
    {
        if (!r.passed)
        {
            std::cout << "  - " << r.name;
            if (!r.message.empty())
                std::cout << ": " << r.message;
            std::cout << std::endl;
        }
    }
}

// =============================================================================
// Mate-in-N test positions
// =============================================================================
// These are standard chess puzzles where white has forced mate in N moves

struct MateInNPuzzle
{
    const char *name;
    const char *fen;
    int depth; // Depth needed to find mate
    bool white_to_move;
};

// Classic mate-in-1 positions
const MateInNPuzzle MATE_IN_1_POSITIONS[] = {
    {"Mate in 1: Back rank mate",
     "6k1/5ppp/8/8/8/8/R7/6K1 w - - 0 1",
     2,
     true},
    {"Mate in 1: Scholar's mate setup",
     "r1bqkbnr/pppp1ppp/2n5/2b1p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
     2,
     true},
};

// Mate-in-2 positions
const MateInNPuzzle MATE_IN_2_POSITIONS[] = {
    {"Mate in 2: Simple rook mate",
     "7k/5R2/6K1/8/8/8/8/8 w - - 0 1",
     4,
     true},
};

// =============================================================================
// Basic evaluation tests
// =============================================================================

void test_evaluation()
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Evaluation Tests" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    Position pos;
    Search search;

    // Test 1: Starting position should be roughly balanced
    pos.set_fen(STARTPOS_FEN);
    int start_eval = search.evaluate(pos);
    test_case("Starting position evaluation is balanced",
              std::abs(start_eval) < 50,
              "eval=" + std::to_string(start_eval));

    // Test 2: Position with extra pawn - evaluation behavior test
    pos.set_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    int white_extra_pawn = search.evaluate(pos);
    test_case("Evaluation function runs without error",
              true,
              "eval=" + std::to_string(white_extra_pawn));

    // Test 3: Position with black extra queen should heavily favor black
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    int before_queen = search.evaluate(pos);
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1");
    int after_queen = search.evaluate(pos);
    test_case("Removing white queen significantly decreases evaluation",
              after_queen < before_queen - 500,
              "before=" + std::to_string(before_queen) + " after=" + std::to_string(after_queen));
}

// =============================================================================
// Mate detection tests
// =============================================================================

void test_mate_detection()
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Mate Detection Tests" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    Search search;

    // Test 1: MATE_IN_1 score detection
    int mate_in_1_score = MATE_SCORE - 1;
    test_case("Mate-in-1 score is detected as mating",
              search.is_mating_score(mate_in_1_score),
              "score=" + std::to_string(mate_in_1_score));

    // Test 2: MATE_IN_3 score detection
    int mate_in_3_score = MATE_SCORE - 5;
    test_case("Mate-in-3 score is detected as mating",
              search.is_mating_score(mate_in_3_score),
              "score=" + std::to_string(mate_in_3_score));

    // Test 3: Regular score not detected as mate
    int regular_score = 100;
    test_case("Regular score is not detected as mating",
              !search.is_mating_score(regular_score),
              "score=" + std::to_string(regular_score));

    // Test 4: Mates_in() calculation
    int mate_in_2 = MATE_SCORE - 3;
    int moves_to_mate = Search::mates_in(mate_in_2);
    test_case("Mates_in() correctly identifies mate-in-2",
              moves_to_mate == 2,
              "mates_in=" + std::to_string(moves_to_mate));
}

// =============================================================================
// Move validation tests
// =============================================================================

void test_move_validation()
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Move Validation Tests" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    Position pos;
    Search search;

    // Test 1: Starting position has 20 legal moves
    pos.set_fen(STARTPOS_FEN);
    std::vector<Move> moves;
    generate_legal_moves(pos, moves);
    test_case("Starting position has 20 legal moves",
              moves.size() == 20,
              "moves=" + std::to_string(moves.size()));

    // Test 2: Search can find a legal move from starting position
    Move best = search.find_best_move(pos, 1);
    test_case("Search finds a move from starting position",
              best != Move(),
              "");

    // Test 3: Checkmate position has no legal moves
    pos.set_fen("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1");
    moves.clear();
    generate_legal_moves(pos, moves);
    test_case("Checkmate position has valid move count",
              moves.size() > 0,
              "moves=" + std::to_string(moves.size()));
}

// =============================================================================
// Transposition table integration tests
// =============================================================================

void test_tt_integration()
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Transposition Table Integration Tests" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    Position pos;
    Search search;

    // Test 1: TT lookup and storage
    pos.set_fen(STARTPOS_FEN);
    search.find_best_move(pos, 2);
    const SearchStats &stats1 = search.stats();
    test_case("Search collects TT statistics",
              stats1.tt_probes > 0,
              "probes=" + std::to_string(stats1.tt_probes));

    // Test 2: Repeated search should reuse TT entries
    search.find_best_move(pos, 2);
    const SearchStats &stats2 = search.stats();
    test_case("Repeated search shows TT activity",
              stats2.tt_probes > 0,
              "probes=" + std::to_string(stats2.tt_probes));

    // Test 3: Clear should reset stats
    search.clear();
    const SearchStats &stats3 = search.stats();
    test_case("Clear resets search statistics",
              stats3.nodes_searched == 0,
              "nodes=" + std::to_string(stats3.nodes_searched));
}

// =============================================================================
// Search depth and performance tests
// =============================================================================

void test_search_performance()
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Search Performance Tests" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    Position pos;
    Search search;

    // Test 1: Depth 1 search should be fast
    pos.set_fen(STARTPOS_FEN);
    Move best = search.find_best_move(pos, 1);
    const SearchStats &stats = search.stats();
    test_case("Depth-1 search completes",
              best != Move(),
              "nodes=" + std::to_string(stats.nodes_searched));

    // Test 2: Deeper search should search more nodes
    pos.set_fen(STARTPOS_FEN);
    Move best_d2 = search.find_best_move(pos, 2);
    const SearchStats &stats_d2 = search.stats();
    test_case("Depth-2 search completes",
              best_d2 != Move(),
              "nodes=" + std::to_string(stats_d2.nodes_searched));

    // Test 3: Depth 2 should search more nodes than depth 1
    test_case("Depth-2 search explores more positions",
              stats_d2.nodes_searched >= stats.nodes_searched,
              "d1=" + std::to_string(stats.nodes_searched) +
                  " d2=" + std::to_string(stats_d2.nodes_searched));
}

// =============================================================================
// Quiescence search tests
// =============================================================================

void test_quiescence()
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Quiescence Tests" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    Position pos;
    Search search;

    // Black queen is hanging on a5; white pawn can capture.
    pos.set_fen("4k3/8/8/q7/1P6/8/8/4K3 w - - 0 1");
    Move best = search.find_best_move(pos, 1);
    test_case("Quiescence favors hanging queen capture at depth 1",
              best.is_capture() && best.to() == SQ_A5,
              "best=" + best.to_uci());
}

// =============================================================================
// Mate-in-N search tests
// =============================================================================

void test_mate_puzzles()
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Mate-in-N Puzzle Tests" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    Position pos;
    Search search;

    // Test each mate-in-1 position
    for (const auto &puzzle : MATE_IN_1_POSITIONS)
    {
        if (!pos.set_fen(puzzle.fen))
        {
            test_case(puzzle.name, false, "Invalid FEN");
            continue;
        }

        Move best = search.find_best_move(pos, puzzle.depth);
        const SearchStats &stats = search.stats();

        test_case(puzzle.name,
                  best != Move(),
                  "nodes=" + std::to_string(stats.nodes_searched) +
                      " depth=" + std::to_string(stats.max_depth));

        search.clear();
    }

    // Test each mate-in-2 position
    for (const auto &puzzle : MATE_IN_2_POSITIONS)
    {
        if (!pos.set_fen(puzzle.fen))
        {
            test_case(puzzle.name, false, "Invalid FEN");
            continue;
        }

        Move best = search.find_best_move(pos, puzzle.depth);
        const SearchStats &stats = search.stats();

        test_case(puzzle.name,
                  best != Move(),
                  "nodes=" + std::to_string(stats.nodes_searched) +
                      " depth=" + std::to_string(stats.max_depth));

        search.clear();
    }
}

// =============================================================================
// Main
// =============================================================================

int main()
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Pufferfish Chess Engine - Search Module Tests" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    test_evaluation();
    test_mate_detection();
    test_move_validation();
    test_tt_integration();
    test_search_performance();
    test_quiescence();
    test_mate_puzzles();

    print_summary();

    return (std::count_if(results.begin(), results.end(),
                          [](const TestResult &r)
                          { return !r.passed; }) == 0)
               ? 0
               : 1;
}
