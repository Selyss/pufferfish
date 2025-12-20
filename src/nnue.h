/*
 * Pufferfish Chess Engine
 * NNUE Evaluation - Residual NNUE loader (export_int16.py format)
 */

#ifndef PUFFERFISH_NNUE_H
#define PUFFERFISH_NNUE_H

#include "position.h"
#include <cstdint>
#include <string>
#include <vector>

namespace pufferfish
{
    class NNUEEvaluator
    {
    public:
        NNUEEvaluator();

        // Load NNUE weights from binary file (training/export_int16.py format).
        bool load(const std::string &filename);

        // Evaluate a position using the neural network.
        // Returns score in centipawns (positive = advantage for side to move).
        int evaluate(Position &pos) const;

        // Incremental hooks (no-op for residual NNUE; evaluation recomputes features).
        void update_after_move(Position &pos, const Move &m, const Undo &undo) const;
        void refresh_accumulator(Position &pos) const;

        bool is_ready() const { return ready_; }

    private:
        enum LayerType
        {
            LAYER_LINEAR = 1,
            LAYER_LAYERNORM = 2,
            LAYER_RESIDUAL = 3,
            LAYER_COMPACT_RESIDUAL = 4
        };

        struct Layer
        {
            LayerType type = LAYER_LINEAR;
            int in = 0;
            int out = 0;
            int dim = 0;
            float eps = 0.0f;
            bool has_norm = false;
            std::vector<float> weights;
            std::vector<float> bias;
            std::vector<float> w1, b1, w2, b2;
            std::vector<float> norm_w, norm_b;
        };

        bool ready_ = false;
        int input_dim_ = 0;
        std::vector<Layer> layers_;

        struct Scratch
        {
            std::vector<float> features;
            std::vector<float> buf_a;
            std::vector<float> buf_b;
            std::vector<float> tmp;
        };
        mutable Scratch scratch_;

        // Feature encoding (matches dataset.FenFeatureEncoder)
        void encode_features(const Position &pos, std::array<float, NNUE_FEATURE_DIM> &features) const;

        // Helpers
        static float relu(float x) { return x > 0.0f ? x : 0.0f; }
        static void layernorm(std::vector<float> &x, const std::vector<float> &w,
                              const std::vector<float> &b, float eps);
        static float dot_row(const float *row, const std::vector<float> &x, int n);
        static int piece_offset(Piece p);
    };

} // namespace pufferfish

#endif // PUFFERFISH_NNUE_H
