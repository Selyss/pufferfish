/*
 * Pufferfish Chess Engine
 * Move generation implementation
 */

#include "movegen.h"
#include "attack.h"

namespace pufferfish
{

    // =============================================================================
    // Move generation helpers
    // =============================================================================

    // Add a pawn move, handling promotions
    static inline void add_pawn_move(std::vector<Move> &moves, Square from, Square to,
                                     bool is_capture)
    {
        int to_rank = rank_of(to);

        // Check for promotion
        if (to_rank == RANK_8 || to_rank == RANK_1)
        {
            if (is_capture)
            {
                moves.emplace_back(from, to, MOVE_PROMO_CAPTURE_Q);
                moves.emplace_back(from, to, MOVE_PROMO_CAPTURE_R);
                moves.emplace_back(from, to, MOVE_PROMO_CAPTURE_B);
                moves.emplace_back(from, to, MOVE_PROMO_CAPTURE_N);
            }
            else
            {
                moves.emplace_back(from, to, MOVE_PROMO_Q);
                moves.emplace_back(from, to, MOVE_PROMO_R);
                moves.emplace_back(from, to, MOVE_PROMO_B);
                moves.emplace_back(from, to, MOVE_PROMO_N);
            }
        }
        else
        {
            moves.emplace_back(from, to, is_capture ? MOVE_CAPTURE : MOVE_QUIET);
        }
    }

    // Check if a square is valid for moving to (on board, correct wrap detection)
    static inline bool is_valid_move_to(Square from, Square to, int expected_file_diff)
    {
        if (to < 0 || to >= 64)
            return false;
        int actual_file_diff = file_of(to) - file_of(from);
        return actual_file_diff == expected_file_diff;
    }

    // =============================================================================
    // Piece move generation
    // =============================================================================

    static void generate_pawn_moves(const Position &pos, std::vector<Move> &moves)
    {
        Color us = pos.side_to_move();
        Color them = ~us;
        Piece our_pawn = make_piece(us, PAWN);
        int push_dir = pawn_push(us);
        int start_rank = pawn_start_rank(us);
        Square ep_sq = pos.ep_square();

        for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(sq + 1))
        {
            if (pos.piece_on(sq) != our_pawn)
                continue;

            int rank = rank_of(sq);
            int file = file_of(sq);

            // Single push
            Square push1 = Square(sq + push_dir);
            if (pos.is_empty(push1))
            {
                add_pawn_move(moves, sq, push1, false);

                // Double push from starting rank
                if (rank == start_rank)
                {
                    Square push2 = Square(sq + 2 * push_dir);
                    if (pos.is_empty(push2))
                    {
                        moves.emplace_back(sq, push2, MOVE_DOUBLE_PUSH);
                    }
                }
            }

            // Captures (left and right)
            for (int file_delta : {-1, 1})
            {
                int new_file = file + file_delta;
                if (new_file < 0 || new_file > 7)
                    continue;

                Square cap_sq = Square(sq + push_dir + file_delta);

                // Normal capture
                Piece target = pos.piece_on(cap_sq);
                if (target != NO_PIECE && color_of(target) == them)
                {
                    add_pawn_move(moves, sq, cap_sq, true);
                }

                // En passant capture
                if (cap_sq == ep_sq)
                {
                    moves.emplace_back(sq, cap_sq, MOVE_EN_PASSANT);
                }
            }
        }
    }

    static void generate_knight_moves(const Position &pos, std::vector<Move> &moves)
    {
        Color us = pos.side_to_move();
        Piece our_knight = make_piece(us, KNIGHT);

        // Knight move offsets with their file differences
        static constexpr struct
        {
            int offset;
            int file_diff;
        } KNIGHT_MOVES[] = {
            {-17, -1}, {-15, 1}, {-10, -2}, {-6, 2}, {6, -2}, {10, 2}, {15, -1}, {17, 1}};

        for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(sq + 1))
        {
            if (pos.piece_on(sq) != our_knight)
                continue;

            for (const auto &nm : KNIGHT_MOVES)
            {
                Square to = Square(sq + nm.offset);
                if (!is_valid_move_to(sq, to, nm.file_diff))
                    continue;

                Piece target = pos.piece_on(to);
                if (target == NO_PIECE)
                {
                    moves.emplace_back(sq, to, MOVE_QUIET);
                }
                else if (color_of(target) != us)
                {
                    moves.emplace_back(sq, to, MOVE_CAPTURE);
                }
            }
        }
    }

    static void generate_king_moves(const Position &pos, std::vector<Move> &moves)
    {
        Color us = pos.side_to_move();
        Square king_sq = pos.king_square(us);

        // King move offsets with file differences
        static constexpr struct
        {
            int offset;
            int file_diff;
        } KING_MOVES[] = {
            {-9, -1}, {-8, 0}, {-7, 1}, {-1, -1}, {1, 1}, {7, -1}, {8, 0}, {9, 1}};

        for (const auto &km : KING_MOVES)
        {
            Square to = Square(king_sq + km.offset);
            if (!is_valid_move_to(king_sq, to, km.file_diff))
                continue;

            Piece target = pos.piece_on(to);
            if (target == NO_PIECE)
            {
                moves.emplace_back(king_sq, to, MOVE_QUIET);
            }
            else if (color_of(target) != us)
            {
                moves.emplace_back(king_sq, to, MOVE_CAPTURE);
            }
        }

        // Castling
        CastlingRights rights = pos.castling_rights();
        Color them = ~us;

        if (us == WHITE)
        {
            // White kingside
            if ((rights & WHITE_OO) &&
                king_sq == SQ_E1 && pos.piece_on(SQ_H1) == W_ROOK &&
                pos.is_empty(SQ_F1) && pos.is_empty(SQ_G1) &&
                !is_square_attacked(pos, SQ_E1, them) &&
                !is_square_attacked(pos, SQ_F1, them) &&
                !is_square_attacked(pos, SQ_G1, them))
            {
                moves.emplace_back(SQ_E1, SQ_G1, MOVE_CASTLE_K);
            }
            // White queenside
            if ((rights & WHITE_OOO) &&
                king_sq == SQ_E1 && pos.piece_on(SQ_A1) == W_ROOK &&
                pos.is_empty(SQ_D1) && pos.is_empty(SQ_C1) && pos.is_empty(SQ_B1) &&
                !is_square_attacked(pos, SQ_E1, them) &&
                !is_square_attacked(pos, SQ_D1, them) &&
                !is_square_attacked(pos, SQ_C1, them))
            {
                moves.emplace_back(SQ_E1, SQ_C1, MOVE_CASTLE_Q);
            }
        }
        else
        {
            // Black kingside
            if ((rights & BLACK_OO) &&
                king_sq == SQ_E8 && pos.piece_on(SQ_H8) == B_ROOK &&
                pos.is_empty(SQ_F8) && pos.is_empty(SQ_G8) &&
                !is_square_attacked(pos, SQ_E8, them) &&
                !is_square_attacked(pos, SQ_F8, them) &&
                !is_square_attacked(pos, SQ_G8, them))
            {
                moves.emplace_back(SQ_E8, SQ_G8, MOVE_CASTLE_K);
            }
            // Black queenside
            if ((rights & BLACK_OOO) &&
                king_sq == SQ_E8 && pos.piece_on(SQ_A8) == B_ROOK &&
                pos.is_empty(SQ_D8) && pos.is_empty(SQ_C8) && pos.is_empty(SQ_B8) &&
                !is_square_attacked(pos, SQ_E8, them) &&
                !is_square_attacked(pos, SQ_D8, them) &&
                !is_square_attacked(pos, SQ_C8, them))
            {
                moves.emplace_back(SQ_E8, SQ_C8, MOVE_CASTLE_Q);
            }
        }
    }

    static void generate_slider_moves(const Position &pos, std::vector<Move> &moves,
                                      PieceType pt, const int *dirs, int num_dirs)
    {
        Color us = pos.side_to_move();
        Piece our_piece = make_piece(us, pt);

        for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(sq + 1))
        {
            Piece p = pos.piece_on(sq);
            if (p != our_piece)
                continue;

            for (int i = 0; i < num_dirs; ++i)
            {
                int dir = dirs[i];
                Square to = sq;

                while (true)
                {
                    int from_file = file_of(to);
                    to = Square(to + dir);

                    if (to < 0 || to >= 64)
                        break;

                    // Check for wrap-around
                    int to_file = file_of(to);
                    int file_diff = to_file - from_file;
                    if (file_diff < -1 || file_diff > 1)
                        break;

                    Piece target = pos.piece_on(to);
                    if (target == NO_PIECE)
                    {
                        moves.emplace_back(sq, to, MOVE_QUIET);
                    }
                    else
                    {
                        if (color_of(target) != us)
                        {
                            moves.emplace_back(sq, to, MOVE_CAPTURE);
                        }
                        break; // Blocked
                    }
                }
            }
        }
    }

    static void generate_bishop_moves(const Position &pos, std::vector<Move> &moves)
    {
        static constexpr int BISHOP_DIRS[] = {-9, -7, 7, 9};
        generate_slider_moves(pos, moves, BISHOP, BISHOP_DIRS, 4);
    }

    static void generate_rook_moves(const Position &pos, std::vector<Move> &moves)
    {
        static constexpr int ROOK_DIRS[] = {-8, -1, 1, 8};
        generate_slider_moves(pos, moves, ROOK, ROOK_DIRS, 4);
    }

    static void generate_queen_moves(const Position &pos, std::vector<Move> &moves)
    {
        static constexpr int QUEEN_DIRS[] = {-9, -8, -7, -1, 1, 7, 8, 9};
        generate_slider_moves(pos, moves, QUEEN, QUEEN_DIRS, 8);
    }

    // =============================================================================
    // Public interface
    // =============================================================================

    void generate_pseudo_moves(const Position &pos, std::vector<Move> &moves)
    {
        moves.clear();
        moves.reserve(256); // Preallocate for performance

        generate_pawn_moves(pos, moves);
        generate_knight_moves(pos, moves);
        generate_bishop_moves(pos, moves);
        generate_rook_moves(pos, moves);
        generate_queen_moves(pos, moves);
        generate_king_moves(pos, moves);
    }

    void generate_legal_moves(Position &pos, std::vector<Move> &moves)
    {
        std::vector<Move> pseudo;
        generate_pseudo_moves(pos, pseudo);

        moves.clear();
        moves.reserve(pseudo.size());

        Color us = pos.side_to_move();
        Undo undo;

        for (const Move &m : pseudo)
        {
            pos.make_move(m, undo);

            // Check if the move left us in check (illegal)
            if (!is_square_attacked(pos, pos.king_square(us), ~us))
            {
                moves.push_back(m);
            }

            pos.unmake_move(m, undo);
        }
    }

    void generate_captures(Position &pos, std::vector<Move> &moves)
    {
        std::vector<Move> pseudo;
        generate_pseudo_moves(pos, pseudo);

        moves.clear();
        moves.reserve(pseudo.size());

        Color us = pos.side_to_move();
        Undo undo;

        for (const Move &m : pseudo)
        {
            if (!m.is_capture())
                continue;

            pos.make_move(m, undo);
            if (!is_square_attacked(pos, pos.king_square(us), ~us))
            {
                moves.push_back(m);
            }
            pos.unmake_move(m, undo);
        }
    }

} // namespace pufferfish
