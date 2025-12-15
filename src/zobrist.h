/*
 * Pufferfish Chess Engine
 * Zobrist hashing for position representation
 */

#ifndef PUFFERFISH_ZOBRIST_H
#define PUFFERFISH_ZOBRIST_H

#include "types.h"
#include <cstdint>

namespace pufferfish
{

    class Position;

    // =============================================================================
    // Zobrist Hashing
    // =============================================================================
    // Provides random 64-bit keys for:
    //   - Each piece type on each square
    //   - Each castling right configuration
    //   - Side to move (white/black)
    //   - Each en passant file
    //
    // A position's hash is the XOR of all applicable keys, updated incrementally
    // by make/unmake to support efficient transposition tables and repetition detection.

    struct Zobrist
    {
        // Piece keys: indexed as [piece_type][square]
        // piece_type 0 = pawn, 1 = knight, ..., 5 = king (6 types, white/black separate)
        uint64_t piece[12][64];

        // Castling rights keys: indexed as [castling_bitmask]
        uint64_t castling[16];

        // Side to move: white = 0, black = 1
        uint64_t side_to_move[2];

        // En passant keys: indexed as [file], file 0..7
        // Set when an en passant square is available on a particular file
        uint64_t en_passant[8];
    };

    // Global Zobrist table accessor (singleton)
    const Zobrist &zobrist();

    // Compute Zobrist key for a given position (full recompute)
    uint64_t compute_zobrist_key(const Position &pos);

    // Incremental update helpers (for later optimization)
    // These allow efficient key updates during search without full recomputation

    // Update key when a piece moves from 'from' to 'to'
    inline uint64_t update_piece_move(uint64_t key, Piece moving, Square from, Square to)
    {
        const Zobrist &z = zobrist();
        key ^= z.piece[moving][from]; // Remove from source
        key ^= z.piece[moving][to];   // Add to destination
        return key;
    }

    // Update key when a piece is placed
    inline uint64_t update_piece_placed(uint64_t key, Piece p, Square sq)
    {
        return key ^ zobrist().piece[p][sq];
    }

    // Update key when a piece is removed
    inline uint64_t update_piece_removed(uint64_t key, Piece p, Square sq)
    {
        return key ^ zobrist().piece[p][sq];
    }

    // Update key for castling rights change
    inline uint64_t update_castling(uint64_t key, CastlingRights old_rights, CastlingRights new_rights)
    {
        const Zobrist &z = zobrist();
        key ^= z.castling[old_rights];
        key ^= z.castling[new_rights];
        return key;
    }

    // Update key for en passant square change
    inline uint64_t update_en_passant(uint64_t key, Square old_ep, Square new_ep)
    {
        const Zobrist &z = zobrist();
        if (old_ep != SQ_NONE)
        {
            key ^= z.en_passant[file_of(old_ep)];
        }
        if (new_ep != SQ_NONE)
        {
            key ^= z.en_passant[file_of(new_ep)];
        }
        return key;
    }

    // Update key for side to move change
    inline uint64_t update_side_to_move(uint64_t key)
    {
        return key ^ zobrist().side_to_move[WHITE] ^ zobrist().side_to_move[BLACK];
    }

} // namespace pufferfish

#endif // PUFFERFISH_ZOBRIST_H
