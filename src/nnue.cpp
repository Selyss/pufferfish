/*
 * Pufferfish Chess Engine
 * NNUE Implementation - Quantized accumulator (train_modal.py architecture)
 */

#include "nnue.h"
#include "types.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>

namespace pufferfish
{
    NNUEEvaluator::NNUEEvaluator() : ready_(false) {}

    static bool read_exact(std::ifstream &file, void *buffer, size_t bytes)
    {
        file.read(reinterpret_cast<char *>(buffer), static_cast<std::streamsize>(bytes));
        return !file.fail();
    }

    int32_t NNUEEvaluator::clamp_int32(int64_t value)
    {
        if (value > std::numeric_limits<int32_t>::max())
            return std::numeric_limits<int32_t>::max();
        if (value < std::numeric_limits<int32_t>::min())
            return std::numeric_limits<int32_t>::min();
        return static_cast<int32_t>(value);
    }

    bool NNUEEvaluator::load(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "[NNUE] Error: Could not open file: " << filename << std::endl;
            return false;
        }

        Weights w;
        int32_t header[4] = {0, 0, 0, 0};
        if (!read_exact(file, header, sizeof(header)))
        {
            std::cerr << "[NNUE] Error: Failed to read header" << std::endl;
            return false;
        }

        w.feature_dim = header[0];
        w.acc_units = header[1];
        w.hidden1 = header[2];
        w.hidden2 = header[3];

        if (w.feature_dim != NNUE_FEATURE_DIM || w.acc_units != NNUE_ACC_UNITS ||
            w.hidden1 != NNUE_HIDDEN1 || w.hidden2 != NNUE_HIDDEN2)
        {
            std::cerr << "[NNUE] Error: Model dimensions mismatch. "
                      << "Expected (" << NNUE_FEATURE_DIM << "," << NNUE_ACC_UNITS << ","
                      << NNUE_HIDDEN1 << "," << NNUE_HIDDEN2 << ") but got ("
                      << w.feature_dim << "," << w.acc_units << ","
                      << w.hidden1 << "," << w.hidden2 << ")" << std::endl;
            return false;
        }

        w.acc_f_bias.resize(w.acc_units);
        w.acc_e_bias.resize(w.acc_units);
        if (!read_exact(file, w.acc_f_bias.data(), w.acc_units * sizeof(int32_t)) ||
            !read_exact(file, w.acc_e_bias.data(), w.acc_units * sizeof(int32_t)))
        {
            std::cerr << "[NNUE] Error: Failed to read accumulator biases" << std::endl;
            return false;
        }

        const size_t acc_weights_size = static_cast<size_t>(w.feature_dim) * w.acc_units;
        w.acc_f_weights.resize(acc_weights_size);
        w.acc_e_weights.resize(acc_weights_size);

        for (int f = 0; f < w.feature_dim; ++f)
        {
            int16_t *f_base = w.acc_f_weights.data() + static_cast<size_t>(f) * w.acc_units;
            int16_t *e_base = w.acc_e_weights.data() + static_cast<size_t>(f) * w.acc_units;
            if (!read_exact(file, f_base, w.acc_units * sizeof(int16_t)) ||
                !read_exact(file, e_base, w.acc_units * sizeof(int16_t)))
            {
                std::cerr << "[NNUE] Error: Failed to read accumulator weights" << std::endl;
                return false;
            }
        }

        w.fc1_bias.resize(w.hidden1);
        w.fc1_weights.resize(static_cast<size_t>(w.hidden1) * (2 * w.acc_units));
        if (!read_exact(file, w.fc1_bias.data(), w.hidden1 * sizeof(int32_t)) ||
            !read_exact(file, w.fc1_weights.data(), w.fc1_weights.size() * sizeof(int16_t)))
        {
            std::cerr << "[NNUE] Error: Failed to read fc1 weights" << std::endl;
            return false;
        }

        w.fc2_bias.resize(w.hidden2);
        w.fc2_weights.resize(static_cast<size_t>(w.hidden2) * w.hidden1);
        if (!read_exact(file, w.fc2_bias.data(), w.hidden2 * sizeof(int32_t)) ||
            !read_exact(file, w.fc2_weights.data(), w.fc2_weights.size() * sizeof(int16_t)))
        {
            std::cerr << "[NNUE] Error: Failed to read fc2 weights" << std::endl;
            return false;
        }

        if (!read_exact(file, &w.out_bias, sizeof(int32_t)))
        {
            std::cerr << "[NNUE] Error: Failed to read output bias" << std::endl;
            return false;
        }

        w.out_weights.resize(w.hidden2);
        if (!read_exact(file, w.out_weights.data(), w.hidden2 * sizeof(int16_t)))
        {
            std::cerr << "[NNUE] Error: Failed to read output weights" << std::endl;
            return false;
        }

        weights_ = std::move(w);
        ready_ = true;

        std::cout << "[NNUE] Loaded model: features=" << weights_.feature_dim
                  << ", acc_units=" << weights_.acc_units
                  << ", hidden1=" << weights_.hidden1
                  << ", hidden2=" << weights_.hidden2 << std::endl;
        return true;
    }

    int NNUEEvaluator::piece_offset(Piece p)
    {
        if (p == NO_PIECE)
            return -1;
        PieceType pt = type_of(p);
        if (pt == NO_PIECE_TYPE)
            return -1;
        int offset = (static_cast<int>(pt) - 1);
        if (color_of(p) == BLACK)
            offset += 6;
        return (offset >= 0 && offset < 12) ? offset : -1;
    }

    int NNUEEvaluator::feature_index(Piece p, Square sq)
    {
        if (!is_valid_square(sq))
            return -1;
        int offset = piece_offset(p);
        if (offset < 0)
            return -1;
        return static_cast<int>(sq) * 12 + offset;
    }

    void NNUEEvaluator::apply_feature_delta(Position &pos, int feature_idx, int delta) const
    {
        if (feature_idx < 0 || feature_idx >= weights_.feature_dim)
            return;

        int32_t *acc_f = pos.nnue_acc_friendly_.data();
        int32_t *acc_e = pos.nnue_acc_enemy_.data();

        const int16_t *w_f = weights_.acc_f_weights.data() +
                             static_cast<size_t>(feature_idx) * weights_.acc_units;
        const int16_t *w_e = weights_.acc_e_weights.data() +
                             static_cast<size_t>(feature_idx) * weights_.acc_units;

        for (int i = 0; i < weights_.acc_units; ++i)
        {
            acc_f[i] += static_cast<int32_t>(delta) * w_f[i];
            acc_e[i] += static_cast<int32_t>(delta) * w_e[i];
        }
    }

    void NNUEEvaluator::refresh_accumulator(Position &pos) const
    {
        if (!ready_)
            return;

        std::copy(weights_.acc_f_bias.begin(), weights_.acc_f_bias.end(),
                  pos.nnue_acc_friendly_.begin());
        std::copy(weights_.acc_e_bias.begin(), weights_.acc_e_bias.end(),
                  pos.nnue_acc_enemy_.begin());

        for (Square sq = SQ_A1; sq <= SQ_H8; ++sq)
        {
            Piece p = pos.piece_on(sq);
            if (p == NO_PIECE)
                continue;
            int idx = feature_index(p, sq);
            apply_feature_delta(pos, idx, 1);
        }

        pos.nnue_acc_valid_ = true;
    }

    void NNUEEvaluator::update_after_move(Position &pos, const Move &m, const Undo &undo) const
    {
        if (!ready_)
            return;

        if (!undo.nnue_acc_valid)
        {
            refresh_accumulator(pos);
            return;
        }

        Color us = ~pos.side_to_move();
        Square from = m.from();
        Square to = m.to();

        Piece moving_piece = NO_PIECE;
        if (m.is_promotion())
        {
            moving_piece = make_piece(us, PAWN);
        }
        else if (m.is_castle())
        {
            moving_piece = make_piece(us, KING);
        }
        else
        {
            moving_piece = pos.piece_on(to);
        }

        apply_feature_delta(pos, feature_index(moving_piece, from), -1);

        if (m.is_capture() && undo.captured != NO_PIECE)
        {
            apply_feature_delta(pos, feature_index(undo.captured, undo.captured_sq), -1);
        }

        Piece placed_piece = moving_piece;
        if (m.is_promotion())
        {
            placed_piece = make_piece(us, m.promotion_type());
        }

        apply_feature_delta(pos, feature_index(placed_piece, to), 1);

        if (m.is_castle())
        {
            Square rook_from = SQ_NONE;
            Square rook_to = SQ_NONE;
            if (m.flag() == MOVE_CASTLE_K)
            {
                rook_from = (us == WHITE) ? SQ_H1 : SQ_H8;
                rook_to = (us == WHITE) ? SQ_F1 : SQ_F8;
            }
            else
            {
                rook_from = (us == WHITE) ? SQ_A1 : SQ_A8;
                rook_to = (us == WHITE) ? SQ_D1 : SQ_D8;
            }

            Piece rook = make_piece(us, ROOK);
            apply_feature_delta(pos, feature_index(rook, rook_from), -1);
            apply_feature_delta(pos, feature_index(rook, rook_to), 1);
        }

        pos.nnue_acc_valid_ = true;
    }

    int NNUEEvaluator::evaluate(Position &pos) const
    {
        if (!ready_)
            return 0;

        if (!pos.nnue_acc_valid_)
        {
            refresh_accumulator(pos);
        }

        std::vector<int32_t> acc_f(weights_.acc_units);
        std::vector<int32_t> acc_e(weights_.acc_units);

        for (int i = 0; i < weights_.acc_units; ++i)
        {
            acc_f[i] = relu(pos.nnue_acc_friendly_[i]);
            acc_e[i] = relu(pos.nnue_acc_enemy_[i]);
        }

        std::vector<int32_t> fc1_out(weights_.hidden1);
        const int in1 = 2 * weights_.acc_units;

        for (int i = 0; i < weights_.hidden1; ++i)
        {
            int64_t sum = weights_.fc1_bias[i];
            const int16_t *w_row = weights_.fc1_weights.data() + static_cast<size_t>(i) * in1;
            for (int j = 0; j < weights_.acc_units; ++j)
            {
                sum += static_cast<int64_t>(w_row[j]) * acc_f[j];
            }
            for (int j = 0; j < weights_.acc_units; ++j)
            {
                sum += static_cast<int64_t>(w_row[j + weights_.acc_units]) * acc_e[j];
            }
            fc1_out[i] = relu(clamp_int32(sum));
        }

        std::vector<int32_t> fc2_out(weights_.hidden2);
        for (int i = 0; i < weights_.hidden2; ++i)
        {
            int64_t sum = weights_.fc2_bias[i];
            const int16_t *w_row = weights_.fc2_weights.data() + static_cast<size_t>(i) * weights_.hidden1;
            for (int j = 0; j < weights_.hidden1; ++j)
            {
                sum += static_cast<int64_t>(w_row[j]) * fc1_out[j];
            }
            fc2_out[i] = relu(clamp_int32(sum));
        }

        int64_t out_sum = weights_.out_bias;
        for (int i = 0; i < weights_.hidden2; ++i)
        {
            out_sum += static_cast<int64_t>(weights_.out_weights[i]) * fc2_out[i];
        }

        int32_t score = clamp_int32(out_sum);

        if (pos.side_to_move() == BLACK)
        {
            score = -score;
        }

        return score;
    }

} // namespace pufferfish
