/*
 * Pufferfish Chess Engine
 * NNUE Evaluation - Quantized accumulator (train_modal.py architecture)
 */

#ifndef PUFFERFISH_NNUE_H
#define PUFFERFISH_NNUE_H

#include "position.h"
#include "nnue_defs.h"
#include <cstdint>
#include <string>
#include <vector>

namespace pufferfish
{
    class NNUEEvaluator
    {
    public:
        NNUEEvaluator();

        // Load NNUE weights from binary file (export_int16.py format).
        bool load(const std::string &filename);

        // Evaluate a position using the neural network.
        // Returns score in centipawns (positive = advantage for side to move).
        int evaluate(Position &pos) const;

        // Update accumulator after a move is made (incremental update).
        void update_after_move(Position &pos, const Move &m, const Undo &undo) const;

        // Recompute accumulator from scratch for the current position.
        void refresh_accumulator(Position &pos) const;

        bool is_ready() const { return ready_; }

    private:
        struct Weights
        {
            int feature_dim = 0;
            int acc_units = 0;
            int hidden1 = 0;
            int hidden2 = 0;

            std::vector<int32_t> acc_f_bias;
            std::vector<int32_t> acc_e_bias;
            std::vector<int16_t> acc_f_weights; // [feature_dim][acc_units]
            std::vector<int16_t> acc_e_weights; // [feature_dim][acc_units]

            std::vector<int32_t> fc1_bias;
            std::vector<int16_t> fc1_weights; // [hidden1][2*acc_units]
            std::vector<int32_t> fc2_bias;
            std::vector<int16_t> fc2_weights; // [hidden2][hidden1]

            int32_t out_bias = 0;
            std::vector<int16_t> out_weights; // [hidden2]
        };

        Weights weights_;
        bool ready_;

        static int piece_offset(Piece p);
        static int feature_index(Piece p, Square sq);
        void apply_feature_delta(Position &pos, int feature_idx, int delta) const;
        static int32_t relu(int32_t x) { return x > 0 ? x : 0; }
        static int32_t clamp_int32(int64_t value);
    };

} // namespace pufferfish

#endif // PUFFERFISH_NNUE_H
