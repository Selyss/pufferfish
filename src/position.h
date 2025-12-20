/*
 * Pufferfish Chess Engine
 * Position representation and state management
 */

#ifndef PUFFERFISH_POSITION_H
#define PUFFERFISH_POSITION_H

#include "types.h"
#include "move.h"
#include "nnue_defs.h"
#include <array>
#include <string>
#include <iostream>

namespace pufferfish
{

    // Starting position FEN
    constexpr const char *STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    // =============================================================================
    // Position class
    // =============================================================================
    // Complete representation of a chess position with all state needed for
    // legal move generation and make/unmake operations.

    class Position
    {
    public:
        // -------------------------------------------------------------------------
        // Constructors and setup
        // -------------------------------------------------------------------------
        Position();

        // Set position from FEN string. Returns false if FEN is invalid.
        bool set_fen(const std::string &fen);

        // Get FEN string for current position
        std::string fen() const;

        // Reset to starting position
        void reset();

        // Clear the board completely
        void clear();

        // -------------------------------------------------------------------------
        // Board access
        // -------------------------------------------------------------------------
        Piece piece_on(Square sq) const { return board_[sq]; }
        bool is_empty(Square sq) const { return board_[sq] == NO_PIECE; }

        // -------------------------------------------------------------------------
        // State access
        // -------------------------------------------------------------------------
        Color side_to_move() const { return stm_; }
        CastlingRights castling_rights() const { return castling_; }
        Square ep_square() const { return ep_sq_; }
        int halfmove_clock() const { return halfmove_clock_; }
        int fullmove_number() const { return fullmove_; }
        Square king_square(Color c) const { return king_sq_[c]; }

        // -------------------------------------------------------------------------
        // Move operations (to be implemented in later stages)
        // -------------------------------------------------------------------------
        void make_move(const Move &m, Undo &undo);
        void unmake_move(const Move &m, const Undo &undo);

        // -------------------------------------------------------------------------
        // Debug and display
        // -------------------------------------------------------------------------
        void print(std::ostream &os = std::cout) const;

        // Verify internal consistency (for debugging)
        bool is_valid() const;

        // -------------------------------------------------------------------------
        // Comparison (for testing make/unmake)
        // -------------------------------------------------------------------------
        bool operator==(const Position &other) const;
        bool operator!=(const Position &other) const { return !(*this == other); }

        // -------------------------------------------------------------------------
        // Zobrist hashing
        // -------------------------------------------------------------------------
        uint64_t zobrist_key() const { return zobrist_key_; }

    private:
        friend class NNUEEvaluator;

        // -------------------------------------------------------------------------
        // Internal helpers
        // -------------------------------------------------------------------------
        void put_piece(Piece p, Square sq);
        void remove_piece(Square sq);
        void move_piece(Square from, Square to);

        // -------------------------------------------------------------------------
        // State
        // -------------------------------------------------------------------------
        Piece board_[64];         // Mailbox board representation
        Color stm_;               // Side to move
        CastlingRights castling_; // Castling rights bitmask
        Square ep_sq_;            // En passant target square (or SQ_NONE)
        int halfmove_clock_;      // Halfmove clock for 50-move rule
        int fullmove_;            // Fullmove number
        Square king_sq_[2];       // King squares indexed by color
        uint64_t zobrist_key_;    // Zobrist hash of the position
        std::array<float, NNUE_FEATURE_DIM> nnue_features_;
        bool nnue_features_valid_;
    };

    // Convenience output operator
    inline std::ostream &operator<<(std::ostream &os, const Position &pos)
    {
        pos.print(os);
        return os;
    }

} // namespace pufferfish

#endif // PUFFERFISH_POSITION_H
