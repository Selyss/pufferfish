/*
 * Pufferfish Chess Engine
 * NNUE Implementation
 */

#include "nnue.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace pufferfish
{

    NNUEEvaluator::NNUEEvaluator() : ready_(false) {}

    NNUEEvaluator::~NNUEEvaluator() {}

    bool NNUEEvaluator::load(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }

        // Read header
        char magic[4];
        file.read(magic, 4);
        if (std::strncmp(magic, "NNUE", 4) != 0)
        {
            return false;
        }

        uint32_t version;
        file.read(reinterpret_cast<char *>(&version), sizeof(uint32_t));

        uint32_t input_dim;
        file.read(reinterpret_cast<char *>(&input_dim), sizeof(uint32_t));
        if (input_dim != INPUT_DIM)
        {
            return false;
        }

        uint32_t num_layers;
        file.read(reinterpret_cast<char *>(&num_layers), sizeof(uint32_t));

        layers_.clear();
        layer_dims_.clear();
        layer_dims_.push_back(INPUT_DIM);

        // Read layers
        for (uint32_t i = 0; i < num_layers; i++)
        {
            uint32_t rows, cols;
            file.read(reinterpret_cast<char *>(&rows), sizeof(uint32_t));
            file.read(reinterpret_cast<char *>(&cols), sizeof(uint32_t));

            Layer layer;

            // Read weights
            uint32_t weight_count = rows * cols;
            layer.weights.resize(weight_count);
            file.read(reinterpret_cast<char *>(layer.weights.data()), weight_count * sizeof(float));

            // Read bias
            uint32_t bias_size;
            file.read(reinterpret_cast<char *>(&bias_size), sizeof(uint32_t));
            layer.bias.resize(bias_size);
            file.read(reinterpret_cast<char *>(layer.bias.data()), bias_size * sizeof(float));

            layers_.push_back(layer);
            layer_dims_.push_back(rows);
        }

        file.close();
        ready_ = true;
        return true;
    }

    int NNUEEvaluator::nnue_feature_index(Piece piece, Square sq, bool from_white)
    {
        // Feature index: piece type (0-11) * 64 + square (0-63)
        // Pieces: wP=0, wN=1, wB=2, wR=3, wQ=4, wK=5, bP=6, bN=7, bB=8, bR=9, bQ=10, bK=11

        if (piece == NO_PIECE)
            return -1;

        int piece_idx = static_cast<int>(piece) - 1; // Convert to 0-11
        int sq_idx = static_cast<int>(sq);

        if (!from_white)
        {
            // Flip square for black perspective
            sq_idx = 63 - sq_idx;
            // Adjust piece index for black perspective
            if (piece_idx >= 6)
                piece_idx -= 6;
            else
                piece_idx += 6;
        }

        return piece_idx * 64 + sq_idx;
    }

    void NNUEEvaluator::encode_position(const Position &pos, std::vector<float> &features)
    {
        features.assign(INPUT_DIM, 0.0f);

        // Iterate through all squares
        for (int sq = 0; sq < 64; sq++)
        {
            Piece piece = pos.piece_on(static_cast<Square>(sq));
            if (piece != NO_PIECE)
            {
                // Add features for white perspective
                int idx_white = nnue_feature_index(piece, static_cast<Square>(sq), true);
                if (idx_white >= 0)
                    features[idx_white] = 1.0f;

                // Add features for black perspective
                int idx_black = nnue_feature_index(piece, static_cast<Square>(sq), false);
                if (idx_black >= 0)
                    features[INPUT_DIM / 2 + idx_black] = 1.0f;
            }
        }
    }

    void NNUEEvaluator::forward(const std::vector<float> &input, std::vector<float> &output)
    {
        std::vector<float> current = input;

        for (size_t layer_idx = 0; layer_idx < layers_.size(); layer_idx++)
        {
            const Layer &layer = layers_[layer_idx];
            int in_size = layer_dims_[layer_idx];
            int out_size = layer_dims_[layer_idx + 1];

            std::vector<float> next(out_size);

            // Matrix multiplication: next = current * weights + bias
            for (int i = 0; i < out_size; i++)
            {
                float sum = layer.bias[i];
                for (int j = 0; j < in_size; j++)
                {
                    sum += current[j] * layer.weights[j * out_size + i];
                }

                // ReLU activation (except last layer)
                if (layer_idx < layers_.size() - 1)
                {
                    next[i] = std::max(0.0f, sum);
                }
                else
                {
                    next[i] = sum; // Linear output layer
                }
            }

            current = next;
        }

        output = current;
    }

    int NNUEEvaluator::evaluate(const Position &pos)
    {
        if (!ready_)
            return 0;

        std::vector<float> features;
        encode_position(pos, features);

        std::vector<float> output;
        forward(features, output);

        if (output.empty())
            return 0;

        // Convert to centipawns
        float raw_score = output[0];
        int cp_score = static_cast<int>(raw_score * 100.0f);

        // Flip perspective if black to move
        if (pos.side_to_move() == BLACK)
            cp_score = -cp_score;

        return cp_score;
    }

} // namespace pufferfish
