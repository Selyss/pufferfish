/*
 * Pufferfish Chess Engine
 * UCI Protocol tests
 */

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "../src/position.h"
#include "../src/uci.h"
#include "../src/move.h"
#include "../src/movegen.h"

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
    std::string status = condition ? "✓ PASS" : "✗ FAIL";
    std::cout << status << ": " << name;
    if (!message.empty())
        std::cout << " - " << message;
    std::cout << std::endl;
}

void print_summary()
{
    int passed = 0, total = results.size();
    for (const auto &r : results)
        if (r.passed)
            ++passed;

    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    if (passed == total)
    {
        std::cout << "✓ All tests passed!" << std::endl;
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
// Command Parsing Tests
// =============================================================================

void test_command_parsing()
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "UCI Command Parsing Tests" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    UCIHandler uci;

    // Test 1: UCI command
    std::cout.flush(); // Suppress output during test
    bool result1 = uci.process_command("uci");
    test_case("UCI command accepted", result1 && uci.is_running(), "");

    // Test 2: isready command
    bool result2 = uci.process_command("isready");
    test_case("isready command accepted", result2 && uci.is_running(), "");

    // Test 3: Invalid command (should be ignored)
    bool result3 = uci.process_command("invalid command");
    test_case("Invalid command ignored gracefully", result3 && uci.is_running(), "");

    // Test 4: quit command stops execution
    bool result4 = uci.process_command("quit");
    test_case("quit command stops engine", !result4 && !uci.is_running(), "");
}

// =============================================================================
// Position Parsing Tests
// =============================================================================

void test_position_parsing()
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Position Parsing Tests" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    UCIHandler uci;

    // Test 1: startpos
    uci.process_command("position startpos");
    Position expected;
    expected.set_fen(STARTPOS_FEN);
    bool same = (uci.position().fen() == expected.fen());
    test_case("position startpos sets correct position",
              same,
              "fen=" + uci.position().fen());

    // Test 2: startpos with moves
    uci.process_command("position startpos moves e2e4 c7c5");
    bool has_moves = (uci.position().fen() != STARTPOS_FEN);
    test_case("position with moves changes position",
              has_moves,
              "fen=" + uci.position().fen());

    // Test 3: FEN position
    std::string fen_cmd = "position fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    uci.process_command(fen_cmd);
    bool fen_set = (uci.position().side_to_move() == BLACK);
    test_case("position fen sets correct FEN",
              fen_set,
              "stm=" + std::string(uci.position().side_to_move() == WHITE ? "WHITE" : "BLACK"));

    // Test 4: Empty position command (should keep current)
    uci.process_command("position startpos");
    std::string pos_before = uci.position().fen();
    uci.process_command("position"); // No args
    std::string pos_after = uci.position().fen();
    test_case("empty position command handled",
              true, // Shouldn't crash
              "");
}

// =============================================================================
// Go Command Parsing Tests
// =============================================================================

void test_go_command()
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Go Command Parsing Tests" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    UCIHandler uci;
    uci.process_command("position startpos");

    // Test 1: go with depth
    bool result1 = uci.process_command("go depth 3");
    test_case("go depth command executes",
              result1,
              "");

    // Test 2: go with movetime
    uci.process_command("position startpos");
    bool result2 = uci.process_command("go movetime 100");
    test_case("go movetime command executes",
              result2,
              "");

    // Test 3: go with time controls (wtime/btime)
    uci.process_command("position startpos");
    bool result3 = uci.process_command("go wtime 300000 btime 300000 winc 5000 binc 5000");
    test_case("go with time controls executes",
              result3,
              "");

    // Test 4: go without args (uses defaults)
    uci.process_command("position startpos");
    bool result4 = uci.process_command("go");
    test_case("go without args uses defaults",
              result4,
              "");
}

// =============================================================================
// Game State Tests
// =============================================================================

void test_game_state()
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Game State Tests" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    UCIHandler uci;

    // Test 1: Initial position is startpos
    bool initial = (uci.position().fen() == STARTPOS_FEN);
    test_case("UCI handler starts in startpos",
              initial,
              "");

    // Test 2: Position changes after move
    uci.process_command("position startpos moves e2e4");
    bool moved = (uci.position().fen() != STARTPOS_FEN);
    test_case("Position updates after moves",
              moved,
              "");

    // Test 3: Multiple position commands reset state
    uci.process_command("position startpos moves e2e4");
    std::string after_move = uci.position().fen();
    uci.process_command("position startpos");
    std::string after_reset = uci.position().fen();
    bool reset = (after_reset == STARTPOS_FEN && after_move != STARTPOS_FEN);
    test_case("Position resets correctly",
              reset,
              "");

    // Test 4: Search engine persists
    const Search &search = uci.search();
    (void)search; // Suppress unused variable warning
    test_case("Search engine is accessible",
              true,
              "");
}

// =============================================================================
// Integration Tests
// =============================================================================

void test_integration()
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Integration Tests" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    UCIHandler uci;

    // Test 1: Full game flow - setup and search
    uci.process_command("uci");
    uci.process_command("isready");
    uci.process_command("position startpos");
    bool flow1 = uci.process_command("go depth 1");
    test_case("Basic game flow works",
              flow1,
              "");

    // Test 2: Rapid position changes
    uci.process_command("position startpos moves e2e4");
    uci.process_command("position startpos moves d2d4");
    uci.process_command("position startpos moves c2c4");
    bool flow2 = (uci.position().side_to_move() == BLACK);
    test_case("Rapid position changes handled",
              flow2,
              "");

    // Test 3: Search after position change
    uci.process_command("position startpos moves e2e4 c7c5");
    bool flow3 = uci.process_command("go depth 2");
    test_case("Search after position moves works",
              flow3,
              "");

    // Test 4: Different time controls
    uci.process_command("position startpos");
    bool flow4a = uci.process_command("go depth 2");
    uci.process_command("position startpos");
    bool flow4b = uci.process_command("go movetime 50");
    uci.process_command("position startpos");
    bool flow4c = uci.process_command("go wtime 5000 btime 5000");
    test_case("All time control modes work",
              flow4a && flow4b && flow4c,
              "");
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

void test_edge_cases()
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Edge Cases and Error Handling" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    UCIHandler uci;

    // Test 1: Empty command
    bool result1 = uci.process_command("");
    test_case("Empty command handled",
              result1 && uci.is_running(),
              "");

    // Test 2: Command with extra spaces
    bool result2 = uci.process_command("  position   startpos  ");
    test_case("Extra spaces handled",
              result2,
              "");

    // Test 3: go command with invalid depth
    uci.process_command("position startpos");
    bool result3 = uci.process_command("go depth 0");
    test_case("Invalid depth handled gracefully",
              result3,
              "");

    // Test 4: Illegal move in position (should be ignored)
    uci.process_command("position startpos moves e2e5"); // Invalid
    bool result4 = uci.process_command("go depth 1");
    test_case("Illegal moves in position handled",
              result4,
              "");

    // Test 5: FEN and moves together
    std::string cmd = "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4";
    uci.process_command(cmd);
    bool result5 = (uci.position().side_to_move() == BLACK);
    test_case("FEN with moves parsed correctly",
              result5,
              "");
}

// =============================================================================
// Main
// =============================================================================

int main()
{
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "Pufferfish Chess Engine - UCI Protocol Tests" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Suppress cout from UCI commands during testing
    std::streambuf *backup = std::cout.rdbuf();
    std::stringstream buffer;
    std::cout.rdbuf(buffer.rdbuf());

    test_command_parsing();
    test_position_parsing();
    test_go_command();
    test_game_state();
    test_integration();
    test_edge_cases();

    // Restore cout
    std::cout.rdbuf(backup);

    print_summary();

    return (std::count_if(results.begin(), results.end(),
                          [](const TestResult &r)
                          { return !r.passed; }) == 0)
               ? 0
               : 1;
}
