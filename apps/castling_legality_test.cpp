// Castling legality test harness
// Confirms whether move generation requires king/rook presence for castling rights.
#include "../src/position.h"
#include "../src/movegen.h"
#include <iostream>
#include <string>
#include <vector>

using namespace pufferfish;

static bool contains_uci(const std::vector<Move> &moves, const char *uci)
{
    for (const auto &m : moves)
    {
        if (m.to_uci() == uci)
            return true;
    }
    return false;
}

static bool expect_no_castle(const char *name, const std::string &fen, const char *castle_uci)
{
    Position pos;
    if (!pos.set_fen(fen))
    {
        std::cout << "[FAIL] " << name << " => set_fen() failed\n";
        std::cout << "        FEN: " << fen << "\n";
        return false;
    }

    std::vector<Move> moves;
    generate_legal_moves(pos, moves);

    bool has_castle = contains_uci(moves, castle_uci);
    bool pass = !has_castle;

    std::cout << (pass ? "[PASS] " : "[FAIL] ") << name
              << " => " << (has_castle ? "generated " : "did not generate ")
              << castle_uci << "\n";
    if (!pass)
    {
        std::cout << "        FEN: " << fen << "\n";
    }
    return pass;
}

int main()
{
    std::cout << "========================================\n"
              << "  Castling Legality Tests\n"
              << "========================================\n";

    bool all_pass = true;

    // 1) Rights claim castling, but rook is missing (should never allow castling).
    all_pass &= expect_no_castle("white O-O without rook", "4k3/8/8/8/8/8/8/4K3 w K - 0 1", "e1g1");
    all_pass &= expect_no_castle("white O-O-O without rook", "4k3/8/8/8/8/8/8/4K3 w Q - 0 1", "e1c1");
    all_pass &= expect_no_castle("black O-O without rook", "4k3/8/8/8/8/8/8/4K3 b k - 0 1", "e8g8");
    all_pass &= expect_no_castle("black O-O-O without rook", "4k3/8/8/8/8/8/8/4K3 b q - 0 1", "e8c8");

    // 2) Rights claim castling, but king is not on the home square (should never allow castling).
    all_pass &= expect_no_castle("white O-O with king on d1", "4k3/8/8/8/8/8/8/3K4 w K - 0 1", "e1g1");
    all_pass &= expect_no_castle("black O-O with king on d8", "3k4/8/8/8/8/8/8/4K3 b k - 0 1", "e8g8");

    std::cout << "========================================\n"
              << "Results: " << (all_pass ? "ALL PASSED" : "SOME FAILED") << "\n"
              << "========================================\n";

    return all_pass ? 0 : 1;
}

