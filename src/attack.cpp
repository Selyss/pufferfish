/*
 * Pufferfish Chess Engine
 * Attack detection implementation
 */

#include "attack.h"

namespace pufferfish
{

    // =============================================================================
    // Attack lookup tables
    // =============================================================================

    // Knight attack offsets
    static constexpr int KNIGHT_OFFSETS[] = {-17, -15, -10, -6, 6, 10, 15, 17};

    // King attack offsets
    static constexpr int KING_OFFSETS[] = {-9, -8, -7, -1, 1, 7, 8, 9};

    // =============================================================================
    // Slider attack helpers
    // =============================================================================

    // Check if moving in direction `dir` from square `from` stays on board
    static inline bool is_valid_step(Square from, int dir)
    {
        Square to = Square(from + dir);
        if (to < 0 || to >= 64)
            return false;

        // Check for wrap-around
        int from_file = file_of(from);
        int to_file = file_of(to);
        int file_diff = to_file - from_file;

        // For diagonal/horizontal moves, file diff should be -1, 0, or 1
        if (file_diff < -1 || file_diff > 1)
            return false;

        return true;
    }

    // Check if there's a slider attack from direction
    static bool slider_attacks(const Position &pos, Square sq, Color by,
                               const int *dirs, int num_dirs, PieceType slider_type)
    {
        Piece queen = make_piece(by, QUEEN);
        Piece slider = make_piece(by, slider_type);

        for (int i = 0; i < num_dirs; ++i)
        {
            int dir = dirs[i];
            Square s = sq;

            while (is_valid_step(s, dir))
            {
                s = Square(s + dir);
                Piece p = pos.piece_on(s);

                if (p != NO_PIECE)
                {
                    // Found a piece
                    if (p == queen || p == slider)
                        return true;
                    break; // Blocked
                }
            }
        }

        return false;
    }

    // =============================================================================
    // Main attack detection
    // =============================================================================

    bool is_square_attacked(const Position &pos, Square sq, Color by)
    {
        // Pawn attacks
        {
            int pawn_dir = (by == WHITE) ? SOUTH : NORTH; // Direction FROM pawn TO target
            Piece pawn = make_piece(by, PAWN);

            // Check left and right pawn attack squares
            int left_file = file_of(sq) - 1;
            int right_file = file_of(sq) + 1;
            int pawn_rank = rank_of(sq) - pawn_dir / 8; // Rank where attacking pawn would be

            if (pawn_rank >= 0 && pawn_rank <= 7)
            {
                if (left_file >= 0)
                {
                    Square pawn_sq = make_square(left_file, pawn_rank);
                    if (pos.piece_on(pawn_sq) == pawn)
                        return true;
                }
                if (right_file <= 7)
                {
                    Square pawn_sq = make_square(right_file, pawn_rank);
                    if (pos.piece_on(pawn_sq) == pawn)
                        return true;
                }
            }
        }

        // Knight attacks
        {
            Piece knight = make_piece(by, KNIGHT);
            for (int offset : KNIGHT_OFFSETS)
            {
                Square s = Square(sq + offset);
                if (!is_valid_square(s))
                    continue;

                // Check for wrap-around (knight moves at most 2 files)
                int file_diff = file_of(s) - file_of(sq);
                if (file_diff < -2 || file_diff > 2)
                    continue;

                if (pos.piece_on(s) == knight)
                    return true;
            }
        }

        // King attacks
        {
            Piece king = make_piece(by, KING);
            for (int offset : KING_OFFSETS)
            {
                Square s = Square(sq + offset);
                if (!is_valid_square(s))
                    continue;

                // Check for wrap-around
                int file_diff = file_of(s) - file_of(sq);
                if (file_diff < -1 || file_diff > 1)
                    continue;

                if (pos.piece_on(s) == king)
                    return true;
            }
        }

        // Bishop/Queen attacks (diagonals)
        {
            static constexpr int BISHOP_DIRS[] = {-9, -7, 7, 9};
            if (slider_attacks(pos, sq, by, BISHOP_DIRS, 4, BISHOP))
                return true;
        }

        // Rook/Queen attacks (orthogonals)
        {
            static constexpr int ROOK_DIRS[] = {-8, -1, 1, 8};
            if (slider_attacks(pos, sq, by, ROOK_DIRS, 4, ROOK))
                return true;
        }

        return false;
    }

} // namespace pufferfish
