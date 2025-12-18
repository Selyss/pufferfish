// FEN validity test harness (Position::set_fen correctness)
#include "../src/position.h"
#include <iostream>
#include <string>

using namespace pufferfish;

static bool expect_set_fen(const char *name, const std::string &fen, bool expected_ok)
{
    Position pos;
    bool ok = pos.set_fen(fen);

    bool pass = (ok == expected_ok);
    std::cout << (pass ? "[PASS] " : "[FAIL] ") << name << " => set_fen() returned "
              << (ok ? "true" : "false") << "\n";

    if (!pass)
    {
        std::cout << "        FEN: " << fen << "\n";
        std::cout << "        Expected: " << (expected_ok ? "true" : "false") << "\n";
    }

    return pass;
}

int main()
{
    std::cout << "========================================\n"
              << "  Pufferfish FEN Validity Tests\n"
              << "========================================\n";

    bool all_pass = true;

    // Valid minimal position (kings only)
    all_pass &= expect_set_fen("kings only", "4k3/8/8/8/8/8/8/4K3 w - - 0 1", true);

    // Invalid: missing one or both kings
    all_pass &= expect_set_fen("missing both kings", "8/8/8/8/8/8/8/8 w - - 0 1", false);
    all_pass &= expect_set_fen("missing black king", "8/8/8/8/8/8/8/4K3 w - - 0 1", false);
    all_pass &= expect_set_fen("missing white king", "4k3/8/8/8/8/8/8/8 w - - 0 1", false);

    // Invalid: en passant square must be on rank 6 for white-to-move and rank 3 for black-to-move
    all_pass &= expect_set_fen("invalid ep rank (white)", "4k3/8/8/8/8/8/8/4K3 w - e3 0 1", false);
    all_pass &= expect_set_fen("invalid ep rank (black)", "4k3/8/8/8/8/8/8/4K3 b - e6 0 1", false);

    std::cout << "========================================\n"
              << "Results: " << (all_pass ? "ALL PASSED" : "SOME FAILED") << "\n"
              << "========================================\n";

    return all_pass ? 0 : 1;
}

