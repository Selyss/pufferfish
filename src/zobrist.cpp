/*
 * Pufferfish Chess Engine
 * Zobrist hashing implementation
 */

#include "zobrist.h"
#include "position.h"
#include "types.h"
#include <random>
#include <cstring>

namespace pufferfish
{

    // =============================================================================
    // Zobrist table initialization
    // =============================================================================

    static Zobrist init_zobrist_table()
    {
        Zobrist z;
        std::mt19937_64 rng(12345); // Fixed seed for reproducibility

        // Initialize piece keys
        for (int piece = 0; piece < 12; ++piece)
        {
            for (int sq = 0; sq < 64; ++sq)
            {
                z.piece[piece][sq] = rng();
            }
        }

        // Initialize castling keys
        for (int i = 0; i < 16; ++i)
        {
            z.castling[i] = rng();
        }

        // Initialize side to move keys
        z.side_to_move[WHITE] = rng();
        z.side_to_move[BLACK] = rng();

        // Initialize en passant keys
        for (int file = 0; file < 8; ++file)
        {
            z.en_passant[file] = rng();
        }

        return z;
    }

    // =============================================================================
    // Singleton accessor
    // =============================================================================

    const Zobrist &zobrist()
    {
        static const Zobrist z = init_zobrist_table();
        return z;
    }

    // =============================================================================
    // Zobrist key computation
    // =============================================================================

    uint64_t compute_zobrist_key(const Position &pos)
    {
        uint64_t key = 0;
        const Zobrist &z = zobrist();

        // XOR in all pieces
        for (Square sq = SQ_A1; sq <= SQ_H8; ++sq)
        {
            Piece p = pos.piece_on(sq);
            if (p != NO_PIECE)
            {
                // Map piece to zobrist index (0-11)
                // Pieces: WHITE_PAWN=0, WHITE_KNIGHT=1, ..., WHITE_KING=5, BLACK_PAWN=6, ...
                key ^= z.piece[p][sq];
            }
        }

        // XOR in castling rights
        key ^= z.castling[pos.castling_rights()];

        // XOR in side to move
        key ^= z.side_to_move[pos.side_to_move()];

        // XOR in en passant if set
        Square ep_sq = pos.ep_square();
        if (ep_sq != SQ_NONE)
        {
            int file = file_of(ep_sq);
            key ^= z.en_passant[file];
        }

        return key;
    }

} // namespace pufferfish
