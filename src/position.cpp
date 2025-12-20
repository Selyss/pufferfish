/*
 * Pufferfish Chess Engine
 * Position implementation
 */

#include "position.h"
#include "zobrist.h"
#include <sstream>
#include <iostream>
#include <cstring>

namespace pufferfish
{

    // =============================================================================
    // Construction and setup
    // =============================================================================

    Position::Position()
    {
        reset();
        zobrist_key_ = compute_zobrist_key(*this);
    }

    void Position::clear()
    {
        std::memset(board_, 0, sizeof(board_));
        stm_ = WHITE;
        castling_ = NO_CASTLING;
        ep_sq_ = SQ_NONE;
        halfmove_clock_ = 0;
        fullmove_ = 1;
        king_sq_[WHITE] = SQ_NONE;
        king_sq_[BLACK] = SQ_NONE;
        zobrist_key_ = 0;
        nnue_features_.fill(0.0f);
        nnue_features_valid_ = false;
    }

    void Position::reset()
    {
        set_fen(STARTPOS_FEN);
    }

    // =============================================================================
    // FEN parsing
    // =============================================================================

    bool Position::set_fen(const std::string &fen)
    {
        clear();

        std::istringstream ss(fen);
        std::string token;

        // 1. Piece placement
        if (!(ss >> token))
            return false;

        Square sq = SQ_A8; // Start at top-left (a8)
        for (char c : token)
        {
            if (c == '/')
            {
                // Move to next rank (go down 2 ranks, since we'll increment file)
                sq = Square(sq - 16);
            }
            else if (c >= '1' && c <= '8')
            {
                // Empty squares
                sq = Square(sq + (c - '0'));
            }
            else
            {
                // Piece
                Piece p = char_to_piece(c);
                if (p == NO_PIECE)
                    return false;
                if (!is_valid_square(sq))
                    return false;

                put_piece(p, sq);

                // Track king positions
                if (type_of(p) == KING)
                {
                    king_sq_[color_of(p)] = sq;
                }

                sq = Square(sq + 1);
            }
        }

        // 2. Side to move
        if (!(ss >> token))
            return false;
        if (token == "w")
            stm_ = WHITE;
        else if (token == "b")
            stm_ = BLACK;
        else
            return false;

        // 3. Castling rights
        if (!(ss >> token))
            return false;
        castling_ = NO_CASTLING;
        if (token != "-")
        {
            for (char c : token)
            {
                switch (c)
                {
                case 'K':
                    castling_ |= WHITE_OO;
                    break;
                case 'Q':
                    castling_ |= WHITE_OOO;
                    break;
                case 'k':
                    castling_ |= BLACK_OO;
                    break;
                case 'q':
                    castling_ |= BLACK_OOO;
                    break;
                default:
                    return false;
                }
            }
        }

        // 4. En passant square
        if (!(ss >> token))
            return false;
        ep_sq_ = string_to_square(token);

        // 5. Halfmove clock (optional)
        if (ss >> token)
        {
            halfmove_clock_ = std::stoi(token);
        }

        // 6. Fullmove number (optional)
        if (ss >> token)
        {
            fullmove_ = std::stoi(token);
        }

        bool valid = is_valid();
        zobrist_key_ = compute_zobrist_key(*this);
        nnue_features_valid_ = false;
        return valid;
    }

    // =============================================================================
    // FEN generation
    // =============================================================================

    std::string Position::fen() const
    {
        std::ostringstream ss;

        // 1. Piece placement
        for (int rank = RANK_8; rank >= RANK_1; --rank)
        {
            int empty = 0;
            for (int file = FILE_A; file <= FILE_H; ++file)
            {
                Square sq = make_square(file, rank);
                Piece p = board_[sq];
                if (p == NO_PIECE)
                {
                    ++empty;
                }
                else
                {
                    if (empty > 0)
                    {
                        ss << empty;
                        empty = 0;
                    }
                    ss << piece_to_char(p);
                }
            }
            if (empty > 0)
                ss << empty;
            if (rank > RANK_1)
                ss << '/';
        }

        // 2. Side to move
        ss << ' ' << (stm_ == WHITE ? 'w' : 'b');

        // 3. Castling rights
        ss << ' ' << castling_to_string(castling_);

        // 4. En passant square
        ss << ' ' << square_to_string(ep_sq_);

        // 5. Halfmove clock
        ss << ' ' << halfmove_clock_;

        // 6. Fullmove number
        ss << ' ' << fullmove_;

        return ss.str();
    }

    // =============================================================================
    // Board manipulation helpers
    // =============================================================================

    void Position::put_piece(Piece p, Square sq)
    {
        assert(is_valid_square(sq));
        assert(p != NO_PIECE);
        board_[sq] = p;
        if (type_of(p) == KING)
        {
            king_sq_[color_of(p)] = sq;
        }
    }

    void Position::remove_piece(Square sq)
    {
        assert(is_valid_square(sq));
        board_[sq] = NO_PIECE;
    }

    void Position::move_piece(Square from, Square to)
    {
        assert(is_valid_square(from) && is_valid_square(to));
        Piece p = board_[from];
        board_[from] = NO_PIECE;
        board_[to] = p;
        if (type_of(p) == KING)
        {
            king_sq_[color_of(p)] = to;
        }
    }

    // =============================================================================
    // Make/Unmake (stub implementations for now)
    // =============================================================================

    void Position::make_move(const Move &m, Undo &undo)
    {
        // Store state for undo
        undo.captured = NO_PIECE;
        undo.captured_sq = SQ_NONE;
        undo.castling = castling_;
        undo.ep_square = ep_sq_;
        undo.halfmove_clock = halfmove_clock_;
        undo.king_sq[WHITE] = king_sq_[WHITE];
        undo.king_sq[BLACK] = king_sq_[BLACK];
        undo.nnue_features = nnue_features_;
        undo.nnue_features_valid = nnue_features_valid_;

        Square from = m.from();
        Square to = m.to();
        Piece moving = board_[from];
        Piece captured = board_[to];
        MoveFlag flag = m.flag();
        Color us = stm_;

        // Clear ep square (will be set if double push)
        ep_sq_ = SQ_NONE;

        // Handle captures
        if (m.is_capture())
        {
            if (flag == MOVE_EN_PASSANT)
            {
                // En passant: captured pawn is on different square
                Square cap_sq = make_square(file_of(to), rank_of(from));
                undo.captured = board_[cap_sq];
                undo.captured_sq = cap_sq;
                remove_piece(cap_sq);
            }
            else
            {
                undo.captured = captured;
                undo.captured_sq = to;
            }
        }

        // Move the piece
        if (m.is_promotion())
        {
            // Remove pawn and add promoted piece
            remove_piece(from);
            put_piece(make_piece(us, m.promotion_type()), to);
        }
        else if (m.is_castle())
        {
            // Move king and rook
            move_piece(from, to);

            // Move the rook
            if (flag == MOVE_CASTLE_K)
            {
                // Kingside
                Square rook_from = (us == WHITE) ? SQ_H1 : SQ_H8;
                Square rook_to = (us == WHITE) ? SQ_F1 : SQ_F8;
                move_piece(rook_from, rook_to);
            }
            else
            {
                // Queenside
                Square rook_from = (us == WHITE) ? SQ_A1 : SQ_A8;
                Square rook_to = (us == WHITE) ? SQ_D1 : SQ_D8;
                move_piece(rook_from, rook_to);
            }
        }
        else
        {
            move_piece(from, to);
        }

        // Handle double pawn push (set ep square)
        if (flag == MOVE_DOUBLE_PUSH)
        {
            ep_sq_ = Square(from + pawn_push(us));
        }

        // Update castling rights
        castling_ &= ~CASTLING_RIGHTS_MASK[from];
        castling_ &= ~CASTLING_RIGHTS_MASK[to];

        // Update halfmove clock
        if (type_of(moving) == PAWN || m.is_capture())
        {
            halfmove_clock_ = 0;
        }
        else
        {
            ++halfmove_clock_;
        }

        // Update fullmove number
        if (stm_ == BLACK)
        {
            ++fullmove_;
        }

        // Switch side to move and update Zobrist key
        stm_ = ~stm_;
        zobrist_key_ = compute_zobrist_key(*this);
        // NNUE features updated by evaluator; invalidate after mutation
        nnue_features_valid_ = false;
    }

    void Position::unmake_move(const Move &m, const Undo &undo)
    {
        // Switch side back
        stm_ = ~stm_;
        Color us = stm_;

        Square from = m.from();
        Square to = m.to();
        MoveFlag flag = m.flag();

        // Undo the move
        if (m.is_promotion())
        {
            // Remove promoted piece and restore pawn
            remove_piece(to);
            put_piece(make_piece(us, PAWN), from);
        }
        else if (m.is_castle())
        {
            // Move king back
            move_piece(to, from);

            // Move rook back
            if (flag == MOVE_CASTLE_K)
            {
                Square rook_from = (us == WHITE) ? SQ_H1 : SQ_H8;
                Square rook_to = (us == WHITE) ? SQ_F1 : SQ_F8;
                move_piece(rook_to, rook_from);
            }
            else
            {
                Square rook_from = (us == WHITE) ? SQ_A1 : SQ_A8;
                Square rook_to = (us == WHITE) ? SQ_D1 : SQ_D8;
                move_piece(rook_to, rook_from);
            }
        }
        else
        {
            move_piece(to, from);
        }

        // Restore captured piece
        if (undo.captured != NO_PIECE)
        {
            put_piece(undo.captured, undo.captured_sq);
        }

        // Restore state
        castling_ = undo.castling;
        ep_sq_ = undo.ep_square;
        halfmove_clock_ = undo.halfmove_clock;
        king_sq_[WHITE] = undo.king_sq[WHITE];
        king_sq_[BLACK] = undo.king_sq[BLACK];

        // Restore fullmove if needed
        if (stm_ == BLACK)
        {
            --fullmove_;
        }

        // Recompute Zobrist key
        zobrist_key_ = compute_zobrist_key(*this);
        nnue_features_ = undo.nnue_features;
        nnue_features_valid_ = undo.nnue_features_valid;
    }

    // =============================================================================
    // Display
    // =============================================================================

    void Position::print(std::ostream &os) const
    {
        os << "\n +---+---+---+---+---+---+---+---+\n";
        for (int rank = RANK_8; rank >= RANK_1; --rank)
        {
            os << " |";
            for (int file = FILE_A; file <= FILE_H; ++file)
            {
                Square sq = make_square(file, rank);
                Piece p = board_[sq];
                os << ' ' << (p == NO_PIECE ? '.' : piece_to_char(p)) << " |";
            }
            os << " " << (rank + 1) << "\n +---+---+---+---+---+---+---+---+\n";
        }
        os << "   a   b   c   d   e   f   g   h\n\n";

        os << "FEN: " << fen() << "\n";
        os << "Side to move: " << (stm_ == WHITE ? "White" : "Black") << "\n";
        os << "Castling: " << castling_to_string(castling_) << "\n";
        os << "En passant: " << square_to_string(ep_sq_) << "\n";
        os << "Halfmove clock: " << halfmove_clock_ << "\n";
        os << "Fullmove: " << fullmove_ << "\n";
    }

    // =============================================================================
    // Validation
    // =============================================================================

    bool Position::is_valid() const
    {
        // Check king count and positions
        int white_kings = 0, black_kings = 0;
        Square wk_sq = SQ_NONE, bk_sq = SQ_NONE;

        for (Square sq = SQ_A1; sq <= SQ_H8; sq = Square(sq + 1))
        {
            Piece p = board_[sq];
            if (p == W_KING)
            {
                ++white_kings;
                wk_sq = sq;
            }
            if (p == B_KING)
            {
                ++black_kings;
                bk_sq = sq;
            }
        }

        if (white_kings != 1 || black_kings != 1)
            return false;
        if (king_sq_[WHITE] != wk_sq || king_sq_[BLACK] != bk_sq)
            return false;

        // Pawns cannot exist on rank 1 or rank 8 in a valid position.
        for (int file = FILE_A; file <= FILE_H; ++file)
        {
            const Square sq1 = make_square(file, RANK_1);
            const Square sq8 = make_square(file, RANK_8);
            if (type_of(board_[sq1]) == PAWN || type_of(board_[sq8]) == PAWN)
                return false;
        }

        // Check en passant square validity
        if (ep_sq_ != SQ_NONE)
        {
            int ep_rank = rank_of(ep_sq_);
            if (stm_ == WHITE && ep_rank != RANK_6)
                return false;
            if (stm_ == BLACK && ep_rank != RANK_3)
                return false;
        }

        return true;
    }

    // =============================================================================
    // Comparison
    // =============================================================================

    bool Position::operator==(const Position &other) const
    {
        for (int i = 0; i < 64; ++i)
        {
            if (board_[i] != other.board_[i])
                return false;
        }
        return stm_ == other.stm_ &&
               castling_ == other.castling_ &&
               ep_sq_ == other.ep_sq_ &&
               halfmove_clock_ == other.halfmove_clock_ &&
               king_sq_[WHITE] == other.king_sq_[WHITE] &&
               king_sq_[BLACK] == other.king_sq_[BLACK];
    }

} // namespace pufferfish
