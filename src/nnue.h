/*
 * Pufferfish Chess Engine
 * NNUE Evaluation - Neural Network Utility Evaluation
 */

#ifndef PUFFERFISH_NNUE_H
#define PUFFERFISH_NNUE_H

#include "position.h"
#include <vector>
#include <cstdint>
#include <string>

namespace pufferfish
{

    // =============================================================================
    // NNUE Evaluator
    // =============================================================================
    // Loads and uses a neural network for position evaluation
    // Model format: binary NNUE weights file

    class NNUEEvaluator
    {
    public:
        NNUEEvaluator();
        ~NNUEEvaluator();

        // Load NNUE weights from binary file
        // Returns true if successful
        bool load(const std::string &filename);

        // Evaluate a position using the neural network
        // Returns score in centipawns (positive = advantage for side to move)
        int evaluate(const Position &pos);

        // Check if NNUE is ready to use
        bool is_ready() const { return ready_; }

    private:
        // Feature encoding: 768 dims (12 piece types × 64 squares × 2 perspectives)
        static constexpr int INPUT_DIM = 768;

        // Layer dimensions
        std::vector<int> layer_dims_;

        // Network weights and biases
        struct Layer
        {
            std::vector<float> weights;
            std::vector<float> bias;
        };
        std::vector<Layer> layers_;

        bool ready_;

        // Feature encoding helpers
        int nnue_feature_index(Piece piece, Square sq, bool from_white);
        void encode_position(const Position &pos, std::vector<float> &features);

        // Forward pass computation
        void forward(const std::vector<float> &input, std::vector<float> &output);
    };

} // namespace pufferfish

#endif // PUFFERFISH_NNUE_H
