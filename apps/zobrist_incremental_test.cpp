// Zobrist incremental update helper verification test harness
//
// This compares Zobrist "full recompute" against the incremental helpers in
// src/zobrist.h on a controlled, deterministic mini-position representation.
#include "../src/zobrist.h"
#include "../src/types.h"
#include <array>
#include <cstdint>
#include <iostream>

using namespace pufferfish;

struct PieceOnSquare
{
    Piece piece;
    Square square;
};

static int ref_piece_index(Piece p)
{
    if (p >= W_PAWN && p <= W_KING)
        return int(p) - int(W_PAWN);
    if (p >= B_PAWN && p <= B_KING)
        return 6 + (int(p) - int(B_PAWN));
    return -1;
}

template <size_t N>
static uint64_t ref_compute_key(const std::array<PieceOnSquare, N> &pieces,
                                Color stm,
                                CastlingRights castling,
                                Square ep_sq)
{
    const Zobrist &z = zobrist();
    uint64_t key = 0;

    for (const auto &ps : pieces)
    {
        int idx = ref_piece_index(ps.piece);
        if (idx < 0 || idx >= 12)
        {
            // Invalid piece encoding for reference calculation.
            // Return a sentinel to make failures obvious.
            return 0xDEADBEEFDEADBEEFull;
        }
        key ^= z.piece[idx][ps.square];
    }

    key ^= z.castling[castling];
    key ^= z.side_to_move[stm];

    if (ep_sq != SQ_NONE)
        key ^= z.en_passant[file_of(ep_sq)];

    return key;
}

static bool expect_equal(const char *name, uint64_t got, uint64_t expected)
{
    bool pass = (got == expected);
    std::cout << (pass ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!pass)
    {
        std::cout << "        got:      0x" << std::hex << got << std::dec << "\n";
        std::cout << "        expected: 0x" << std::hex << expected << std::dec << "\n";
    }
    return pass;
}

int main()
{
    std::cout << "========================================\n"
              << "  Pufferfish Zobrist Incremental Tests\n"
              << "========================================\n";

    // Deterministic mini-position:
    //   White: King e1, Pawn e2
    //   Black: King e8
    // This is not intended to be a fully legal chess position; it's a stable
    // state vector for hashing tests.
    const std::array<PieceOnSquare, 3> pieces_base = {{
        {W_KING, SQ_E1},
        {W_PAWN, SQ_E2},
        {B_KING, SQ_E8},
    }};

    const CastlingRights castling_base = ANY_CASTLING;
    const Square ep_base = SQ_NONE;
    const Color stm_base = WHITE;

    const uint64_t key0 = ref_compute_key(pieces_base, stm_base, castling_base, ep_base);

    bool all_pass = true;

    // 1) Side to move update
    {
        uint64_t inc = update_side_to_move(key0);
        uint64_t ref = ref_compute_key(pieces_base, BLACK, castling_base, ep_base);
        all_pass &= expect_equal("update_side_to_move matches reference", inc, ref);
    }

    // 2) En passant update (none -> e3)
    {
        uint64_t inc = update_en_passant(key0, SQ_NONE, SQ_E3);
        uint64_t ref = ref_compute_key(pieces_base, stm_base, castling_base, SQ_E3);
        all_pass &= expect_equal("update_en_passant matches reference", inc, ref);
    }

    // 3) Castling update (KQkq -> -)
    {
        uint64_t inc = update_castling(key0, castling_base, NO_CASTLING);
        uint64_t ref = ref_compute_key(pieces_base, stm_base, NO_CASTLING, ep_base);
        all_pass &= expect_equal("update_castling matches reference", inc, ref);
    }

    // 4) Piece move update (pawn e2 -> e4)
    // This is the most important drift test: incremental piece move should match
    // full recompute after the move is applied.
    {
        uint64_t inc = update_piece_move(key0, W_PAWN, SQ_E2, SQ_E4);

        const std::array<PieceOnSquare, 3> pieces_after = {{
            {W_KING, SQ_E1},
            {W_PAWN, SQ_E4},
            {B_KING, SQ_E8},
        }};
        uint64_t ref = ref_compute_key(pieces_after, stm_base, castling_base, ep_base);
        all_pass &= expect_equal("update_piece_move matches reference", inc, ref);
    }

    std::cout << "========================================\n"
              << "Results: " << (all_pass ? "ALL PASSED" : "SOME FAILED") << "\n"
              << "========================================\n";

    return all_pass ? 0 : 1;
}

