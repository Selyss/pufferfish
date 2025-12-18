/*
 * Pufferfish Chess Engine
 * Move representation
 */

#ifndef PUFFERFISH_MOVE_H
#define PUFFERFISH_MOVE_H

#include "types.h"
#include "nnue_defs.h"
#include <array>
#include <string>

namespace pufferfish
{

    // =============================================================================
    // Move flags
    // =============================================================================
    // Each move carries flags indicating its type for proper handling in make/unmake

    enum MoveFlag : int
    {
        MOVE_NONE = 0,
        MOVE_QUIET = 1,            // Normal non-capture move
        MOVE_CAPTURE = 2,          // Normal capture
        MOVE_DOUBLE_PUSH = 3,      // Pawn double push (sets ep square)
        MOVE_EN_PASSANT = 4,       // En passant capture
        MOVE_CASTLE_K = 5,         // Kingside castling
        MOVE_CASTLE_Q = 6,         // Queenside castling
        MOVE_PROMO_N = 7,          // Promotion to knight
        MOVE_PROMO_B = 8,          // Promotion to bishop
        MOVE_PROMO_R = 9,          // Promotion to rook
        MOVE_PROMO_Q = 10,         // Promotion to queen
        MOVE_PROMO_CAPTURE_N = 11, // Capture + promotion to knight
        MOVE_PROMO_CAPTURE_B = 12, // Capture + promotion to bishop
        MOVE_PROMO_CAPTURE_R = 13, // Capture + promotion to rook
        MOVE_PROMO_CAPTURE_Q = 14  // Capture + promotion to queen
    };

    // =============================================================================
    // Move representation
    // =============================================================================
    // Compact 16-bit move encoding:
    //   Bits 0-5:   from square (0-63)
    //   Bits 6-11:  to square (0-63)
    //   Bits 12-15: flags

    class Move
    {
    public:
        constexpr Move() : data_(0) {}
        constexpr Move(Square from, Square to, MoveFlag flag = MOVE_QUIET)
            : data_((flag << 12) | (to << 6) | from) {}

        constexpr Square from() const { return Square(data_ & 0x3F); }
        constexpr Square to() const { return Square((data_ >> 6) & 0x3F); }
        constexpr MoveFlag flag() const { return MoveFlag((data_ >> 12) & 0xF); }

        constexpr bool is_none() const { return data_ == 0; }
        constexpr bool is_capture() const
        {
            MoveFlag f = flag();
            return f == MOVE_CAPTURE || f == MOVE_EN_PASSANT ||
                   (f >= MOVE_PROMO_CAPTURE_N && f <= MOVE_PROMO_CAPTURE_Q);
        }
        constexpr bool is_promotion() const
        {
            return flag() >= MOVE_PROMO_N;
        }
        constexpr bool is_castle() const
        {
            return flag() == MOVE_CASTLE_K || flag() == MOVE_CASTLE_Q;
        }
        constexpr bool is_en_passant() const
        {
            return flag() == MOVE_EN_PASSANT;
        }
        constexpr bool is_double_push() const
        {
            return flag() == MOVE_DOUBLE_PUSH;
        }

        // Get promotion piece type (only valid if is_promotion() is true)
        constexpr PieceType promotion_type() const
        {
            MoveFlag f = flag();
            if (f == MOVE_PROMO_N || f == MOVE_PROMO_CAPTURE_N)
                return KNIGHT;
            if (f == MOVE_PROMO_B || f == MOVE_PROMO_CAPTURE_B)
                return BISHOP;
            if (f == MOVE_PROMO_R || f == MOVE_PROMO_CAPTURE_R)
                return ROOK;
            if (f == MOVE_PROMO_Q || f == MOVE_PROMO_CAPTURE_Q)
                return QUEEN;
            return NO_PIECE_TYPE;
        }

        // UCI format: e2e4, e7e8q (promotion), e1g1 (castle)
        std::string to_uci() const
        {
            if (is_none())
                return "0000";
            std::string s = square_to_string(from()) + square_to_string(to());
            if (is_promotion())
            {
                s += piece_type_to_char(promotion_type());
            }
            return s;
        }

        constexpr uint16_t raw() const { return data_; }

        constexpr bool operator==(const Move &other) const { return data_ == other.data_; }
        constexpr bool operator!=(const Move &other) const { return data_ != other.data_; }

    private:
        uint16_t data_;
    };

    constexpr Move MOVE_NULL = Move();

    // =============================================================================
    // Undo information for unmake_move
    // =============================================================================
    // Stores all state needed to restore position after unmake

    struct Undo
    {
        Piece captured;          // Captured piece (or NO_PIECE)
        Square captured_sq;      // Square of captured piece (for en passant)
        CastlingRights castling; // Previous castling rights
        Square ep_square;        // Previous en passant square
        int halfmove_clock;      // Previous halfmove clock
        Square king_sq[2];       // Previous king squares [WHITE, BLACK]
        std::array<int32_t, NNUE_ACC_UNITS> nnue_acc_friendly;
        std::array<int32_t, NNUE_ACC_UNITS> nnue_acc_enemy;
        bool nnue_acc_valid;

        Undo() : captured(NO_PIECE), captured_sq(SQ_NONE),
                 castling(NO_CASTLING), ep_square(SQ_NONE), halfmove_clock(0),
                 nnue_acc_friendly{}, nnue_acc_enemy{}, nnue_acc_valid(false)
        {
            king_sq[WHITE] = SQ_NONE;
            king_sq[BLACK] = SQ_NONE;
        }
    };

} // namespace pufferfish

#endif // PUFFERFISH_MOVE_H
