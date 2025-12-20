/*
 * Pufferfish Chess Engine
 * Search implementation - Alpha-beta pruning with TT
 */

#include "search.h"
#include "movegen.h"
#include "position.h"
#include "attack.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace pufferfish
{
    namespace
    {
        int piece_value(PieceType pt)
        {
            switch (pt)
            {
            case PAWN:
                return 100;
            case KNIGHT:
                return 320;
            case BISHOP:
                return 330;
            case ROOK:
                return 500;
            case QUEEN:
                return 900;
            case KING:
                return 20000;
            default:
                return 0;
            }
        }

        int capture_mvv_lva(const Position &pos, const Move &m)
        {
            if (!m.is_capture())
                return 0;

            PieceType attacker = type_of(pos.piece_on(m.from()));
            PieceType victim = PAWN;
            if (m.flag() != MOVE_EN_PASSANT)
            {
                Piece captured = pos.piece_on(m.to());
                if (captured != NO_PIECE)
                {
                    victim = type_of(captured);
                }
            }
            return piece_value(victim) * 10 - piece_value(attacker);
        }
    }

    Search::Search(size_t tt_size_mb)
        : tt_(tt_size_mb), nnue_(std::make_unique<NNUEEvaluator>())
    {
        stats_.reset();

        // Try to load NNUE weights from multiple possible locations
        const std::vector<std::string> nnue_paths = {
            "models/nnue_residual.bin",       // Preferred export name
            "models/nnue_weights.bin",        // Legacy name
            "../models/nnue_residual.bin",    // One level up
            "../models/nnue_weights.bin",
            "../../models/nnue_residual.bin", // Two levels up
            "../../models/nnue_weights.bin",
            "./nnue_residual.bin",            // Current dir
            "./nnue_weights.bin",
        };

        for (const auto &path : nnue_paths)
        {
            if (std::filesystem::exists(path))
            {
                if (nnue_->load(path))
                {
                    break; // Successfully loaded
                }
            }
        }
    }

    // =============================================================================
    // Simple Evaluation Function
    // =============================================================================
    // Evaluates a position from the perspective of the side to move.
    // Positive = side to move is better
    // Negative = opponent is better

    int Search::evaluate(Position &pos)
    {
        // Use NNUE if available
        if (nnue_ && nnue_->is_ready())
        {
            return nnue_->evaluate(pos);
        }

        // Fallback to material counting
        int score = 0;

        // Material counting (in centipawns)
        constexpr int PAWN_VALUE = 100;
        constexpr int KNIGHT_VALUE = 320;
        constexpr int BISHOP_VALUE = 330;
        constexpr int ROOK_VALUE = 500;
        constexpr int QUEEN_VALUE = 900;
        constexpr int KING_VALUE = 0; // King cannot be captured, not counted

        // Count pieces and add to score
        for (Square sq = SQ_A1; sq <= SQ_H8; ++sq)
        {
            Piece p = pos.piece_on(sq);
            if (p == NO_PIECE)
                continue;

            int value = 0;
            PieceType pt = type_of(p);
            switch (pt)
            {
            case PAWN:
                value = PAWN_VALUE;
                break;
            case KNIGHT:
                value = KNIGHT_VALUE;
                break;
            case BISHOP:
                value = BISHOP_VALUE;
                break;
            case ROOK:
                value = ROOK_VALUE;
                break;
            case QUEEN:
                value = QUEEN_VALUE;
                break;
            case KING:
                value = KING_VALUE;
                break;
            default:
                continue;
            }

            if (color_of(p) == pos.side_to_move())
                score += value;
            else
                score -= value;
        }

        return score;
    }

    // =============================================================================
    // Mate Detection
    // =============================================================================

    bool Search::is_mating_score(int score)
    {
        return std::abs(score) > MATE_SCORE - 500;
    }

    int Search::mates_in(int score)
    {
        if (score > 0)
            return (MATE_SCORE - score + 1) / 2;
        else
            return -(MATE_SCORE + score + 1) / 2;
    }

    // =============================================================================
    // Alpha-Beta Search
    // =============================================================================

    int Search::alpha_beta(Position &pos, int depth, int alpha, int beta)
    {
        if (time_limit_ms_ > 0 && timer_.time_exceeded(time_limit_ms_))
        {
            time_stop_ = true;
            return evaluate(pos);
        }

        ++stats_.nodes_searched;

        // Transposition table lookup
        uint64_t zobrist = pos.zobrist_key();
        const TTEntry *tt_entry = tt_.lookup(zobrist, depth);
        Move tt_move;

        if (tt_entry != nullptr)
        {
            ++stats_.tt_hits;
            tt_move = tt_entry->best_move;
            // Use TT score if it's valid for this node
            int tt_score = static_cast<int>(tt_entry->score);
            if (tt_entry->bound_type == BOUND_EXACT)
                return tt_score;
            else if (tt_entry->bound_type == BOUND_LOWER)
                alpha = std::max(alpha, tt_score);
            else if (tt_entry->bound_type == BOUND_UPPER)
                beta = std::min(beta, tt_score);

            if (alpha >= beta)
                return tt_score; // Beta cutoff
        }

        ++stats_.tt_probes;

        // Terminal node: quiescence search
        if (depth == 0)
        {
            ++stats_.leaf_nodes;
            stats_.max_depth = std::max(stats_.max_depth, 0);
            return quiesce(pos, alpha, beta);
        }

        // Null-move pruning (skip in check, and only at reasonable depth).
        if (depth >= 3 && !in_check(pos))
        {
            Undo null_undo;
            pos.make_null_move(null_undo);
            if (nnue_ && nnue_->is_ready())
            {
                nnue_->refresh_accumulator(pos);
            }
            int score = -alpha_beta(pos, depth - 1 - 2, -beta, -beta + 1);
            pos.unmake_null_move(null_undo);

            if (score >= beta)
            {
                return score;
            }
        }

        // Generate legal moves
        std::vector<Move> moves;
        generate_legal_moves(pos, moves);

        // Check for mate or stalemate
        if (moves.empty())
        {
            if (in_check(pos))
            {
                // Checkmate - mating score based on depth
                int score = MATED_IN_1 + depth;
                tt_.store(zobrist, score, depth, BOUND_EXACT, Move());
                return score;
            }
            else
            {
                // Stalemate - draw
                int score = 0;
                tt_.store(zobrist, score, depth, BOUND_EXACT, Move());
                return score;
            }
        }

        int best_score = NEG_INF;
        Move best_move;
        BoundType bound_type = BOUND_UPPER;

        // Move ordering: TT move first, then captures/promotions.
        auto move_score = [&](const Move &m) -> int {
            if (!tt_move.is_none() && m == tt_move)
                return 1000000;
            int score = 0;
            if (m.is_capture())
                score += 100000 + capture_mvv_lva(pos, m);
            if (m.is_promotion())
                score += 50000;
            if (m.is_castle())
                score += 1000;
            return score;
        };

        std::stable_sort(moves.begin(), moves.end(),
                         [&](const Move &a, const Move &b)
                         { return move_score(a) > move_score(b); });

        // Try each move
        for (const Move &m : moves)
        {
            if (time_stop_)
                break;

            Undo undo;
            pos.make_move(m, undo);
            if (nnue_ && nnue_->is_ready())
            {
                nnue_->update_after_move(pos, m, undo);
            }
            int score = -alpha_beta(pos, depth - 1, -beta, -alpha);
            pos.unmake_move(m, undo);
            if (time_stop_)
                break;

            if (score > best_score)
            {
                best_score = score;
                best_move = m;

                if (score > alpha)
                {
                    alpha = score;
                    bound_type = BOUND_EXACT;

                    if (alpha >= beta)
                    {
                        ++stats_.beta_cutoffs;
                        bound_type = BOUND_LOWER;
                        break; // Beta cutoff
                    }
                }
            }
        }

        // Store in transposition table
        tt_.store(zobrist, best_score, depth, bound_type, best_move);

        stats_.max_depth = std::max(stats_.max_depth, depth);
        return best_score;
    }

    // =============================================================================
    // Public API
    // =============================================================================

    Move Search::find_best_move(Position &pos, int depth)
    {
        time_limit_ms_ = 0;
        time_stop_ = false;
        return find_best_move_internal(pos, depth);
    }

    Move Search::find_best_move_internal(Position &pos, int depth)
    {
        if (depth < 1)
            return Move();

        stats_.reset();
        tt_.clear();
        time_stop_ = false;
        if (nnue_ && nnue_->is_ready())
        {
            nnue_->refresh_accumulator(pos);
        }

        // Root node search with move tracking
        std::vector<Move> moves;
        generate_legal_moves(pos, moves);

        if (moves.empty())
            return Move(); // No legal moves

        // Move ordering: prefer TT move and captures at root.
        Move tt_move;
        if (const TTEntry *tt_entry = tt_.lookup(pos.zobrist_key(), depth))
            tt_move = tt_entry->best_move;

        auto move_score = [&](const Move &m) -> int {
            if (!tt_move.is_none() && m == tt_move)
                return 1000000;
            int score = 0;
            if (m.is_capture())
                score += 100000 + capture_mvv_lva(pos, m);
            if (m.is_promotion())
                score += 50000;
            if (m.is_castle())
                score += 1000;
            return score;
        };

        std::stable_sort(moves.begin(), moves.end(),
                         [&](const Move &a, const Move &b)
                         { return move_score(a) > move_score(b); });

        Move best_move = moves[0];
        int best_score = NEG_INF;

        for (const Move &m : moves)
        {
            if (time_stop_)
                break;

            Undo undo;
            pos.make_move(m, undo);
            if (nnue_ && nnue_->is_ready())
            {
                nnue_->update_after_move(pos, m, undo);
            }
            int score = -alpha_beta(pos, depth - 1, -INF, INF);
            pos.unmake_move(m, undo);
            if (time_stop_)
                break;

            if (score > best_score)
            {
                best_score = score;
                best_move = m;
            }
        }

        return best_move;
    }

    // =============================================================================
    // Iterative Deepening Search
    // =============================================================================

    Move Search::find_best_move_iterative(Position &pos, const SearchTimeManager &time_mgr)
    {
        timer_.reset();
        uint64_t time_limit = time_mgr.get_time_budget();
        time_limit_ms_ = time_limit;
        time_stop_ = false;
        Move best_move;
        if (nnue_ && nnue_->is_ready())
        {
            nnue_->refresh_accumulator(pos);
        }

        // Handle FIXED_DEPTH mode
        if (time_mgr.mode == SearchTimeManager::FIXED_DEPTH)
        {
            return find_best_move(pos, time_mgr.depth);
        }

        // Iterative deepening loop
        for (int depth = 1; depth <= 256; ++depth)
        {
            // Check time before searching at this depth
            if (timer_.time_exceeded(time_limit) && depth > 1)
                break; // Return result from previous depth

            // Search at this depth
            Move move_at_depth = find_best_move_internal(pos, depth);

            if (move_at_depth == Move())
                break; // No legal moves

            best_move = move_at_depth;
            if (time_stop_)
                break;

            // Output UCI info for this depth
            uint64_t elapsed = timer_.elapsed_ms();
            std::cout << "info depth " << depth
                      << " nodes " << stats_.nodes_searched
                      << " time " << elapsed
                      << " nps " << (elapsed > 0 ? (1000 * stats_.nodes_searched / elapsed) : 0)
                      << std::endl;

            // Check time after search completes
            if (timer_.time_exceeded(time_limit))
                break;

            // Safety limit: don't go deeper than mate
            if (stats_.max_depth >= 200)
                break;
        }

        return best_move;
    }

    // =============================================================================
    // Private Helper: Clear for search
    // =============================================================================

    void Search::clear()
    {
        tt_.clear();
        stats_.reset();
        timer_.reset();
    }

    // =============================================================================
    // Quiescence Search (stub for future)
    // =============================================================================
    // For now, just return static evaluation.
    // Later: explore captures and checks to avoid horizon effect.

    int Search::quiesce(Position &pos, int alpha, int beta)
    {
        if (time_limit_ms_ > 0 && timer_.time_exceeded(time_limit_ms_))
        {
            time_stop_ = true;
            return evaluate(pos);
        }

        int stand_pat = evaluate(pos);
        if (stand_pat >= beta)
            return beta;
        if (stand_pat > alpha)
            alpha = stand_pat;

        std::vector<Move> captures;
        generate_captures(pos, captures);

        auto capture_score = [&](const Move &m) -> int
        { return 100000 + capture_mvv_lva(pos, m); };

        std::stable_sort(captures.begin(), captures.end(),
                         [&](const Move &a, const Move &b)
                         { return capture_score(a) > capture_score(b); });

        for (const Move &m : captures)
        {
            if (time_stop_)
                break;

            Undo undo;
            pos.make_move(m, undo);
            if (nnue_ && nnue_->is_ready())
            {
                nnue_->update_after_move(pos, m, undo);
            }
            int score = -quiesce(pos, -beta, -alpha);
            pos.unmake_move(m, undo);
            if (time_stop_)
                break;

            if (score >= beta)
                return beta;
            if (score > alpha)
                alpha = score;
        }

        return alpha;
    }

} // namespace pufferfish
