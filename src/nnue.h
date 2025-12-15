/*
 * Pufferfish Chess Engine
 * NNUE Evaluation - Neural Network Utility Evaluation
 *
 * Architecture: Residual NNUE with 795 input features
 * Supports: LINEAR, LAYERNORM, and RESIDUAL layer types from export_int16.py binary format
 */

#ifndef PUFFERFISH_NNUE_H
#define PUFFERFISH_NNUE_H

#include "position.h"
#include <vector>
#include <cstdint>
#include <string>
#include <memory>

namespace pufferfish
{

    // =============================================================================
    // NNUE Evaluator
    // =============================================================================
    // Loads and uses a residual neural network for position evaluation
    // Model format: Binary NNUE weights file from export_int16.py

    class NNUEEvaluator
    {
    public:
        NNUEEvaluator();
        ~NNUEEvaluator();

        // Load NNUE weights from binary file (residual-nnue-v1 format)
        // Returns true if successful
        bool load(const std::string &filename);

        // Evaluate a position using the neural network
        // Returns score in centipawns (positive = advantage for side to move)
        int evaluate(const Position &pos);

        // Check if NNUE is ready to use
        bool is_ready() const { return ready_; }

    private:
        // Feature encoding: 795 dims total
        // - 768: Piece placement (12 piece types × 64 squares × 2 perspectives)
        // - 1: Material balance
        // - 1: Game phase
        // - 1: Side to move
        // - 4: Castling rights (white/black kingside/queenside)
        // - 8: En passant file (8 possible files)
        static constexpr int INPUT_DIM = 795;

        // Layer types in binary format
        enum class LayerType : uint32_t
        {
            LINEAR = 1,
            LAYERNORM = 2,
            RESIDUAL = 3
        };

        // Generic layer interface
        struct Layer
        {
            virtual ~Layer() = default;
            virtual void forward(const std::vector<float> &input, std::vector<float> &output) = 0;
            virtual int output_dim() const = 0;
        };

        // Linear layer: output = input @ W + b, optionally with ReLU
        struct LinearLayer : Layer
        {
            std::vector<float> weights; // [out_dim][in_dim]
            std::vector<float> bias;    // [out_dim]
            int in_dim, out_dim;
            bool apply_relu = false;

            void forward(const std::vector<float> &input, std::vector<float> &output) override;
            int output_dim() const override { return out_dim; }
        };

        // Layer normalization
        struct LayerNormLayer : Layer
        {
            std::vector<float> weight; // γ [size]
            std::vector<float> bias;   // β [size]
            float eps;
            int size;

            void forward(const std::vector<float> &input, std::vector<float> &output) override;
            int output_dim() const override { return size; }
        };

        // Residual block: residual + ReLU(lin2(ReLU(lin1(x)))) → ReLU(norm(...))
        struct ResidualBlock : Layer
        {
            std::vector<float> lin1_weight, lin1_bias;
            std::vector<float> lin2_weight, lin2_bias;
            std::vector<float> norm_weight, norm_bias;
            float norm_eps;
            int dim;

            void forward(const std::vector<float> &input, std::vector<float> &output) override;
            int output_dim() const override { return dim; }
        };

        // Network structure
        std::vector<std::unique_ptr<Layer>> layers_;
        int current_input_dim_;
        bool ready_;

        // Feature encoding helpers
        void encode_position(const Position &pos, std::vector<float> &features);
        void add_piece_features(const Position &pos, std::vector<float> &features);
        void add_board_state_features(const Position &pos, std::vector<float> &features);

        // Forward pass computation
        void forward(const std::vector<float> &input, std::vector<float> &output);
    };

} // namespace pufferfish

#endif // PUFFERFISH_NNUE_H
