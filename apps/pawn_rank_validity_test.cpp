// Pawn rank validity test harness
// Confirms whether Position::set_fen rejects pawns on rank 1/8.
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
              << "  Pawn Rank Validity Tests\n"
              << "========================================\n";

    bool all_pass = true;

    // Control: valid minimal position
    all_pass &= expect_set_fen("kings only (control)", "4k3/8/8/8/8/8/8/4K3 w - - 0 1", true);

    // These should be invalid in strict chess position validation:
    // pawns cannot exist on rank 8 or rank 1.
    all_pass &= expect_set_fen("white pawn on a8", "P3k3/8/8/8/8/8/8/4K3 w - - 0 1", false);
    all_pass &= expect_set_fen("black pawn on h1", "4k3/8/8/8/8/8/8/4K2p w - - 0 1", false);
    all_pass &= expect_set_fen("white pawn on e1", "4k3/8/8/8/8/8/8/4P3 w - - 0 1", false);
    all_pass &= expect_set_fen("black pawn on e8", "4p3/8/8/8/8/8/8/4K2k w - - 0 1", false);

    std::cout << "========================================\n"
              << "Results: " << (all_pass ? "ALL PASSED" : "SOME FAILED") << "\n"
              << "========================================\n";

    return all_pass ? 0 : 1;
}

