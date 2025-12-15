/*
 * Pufferfish Chess Engine
 * NNUE Implementation - Residual Neural Network Utility Evaluation
 *
 * Loads binary NNUE models exported from PyTorch via export_int16.py
 * Supports LINEAR, LAYERNORM, and RESIDUAL layer types
 */

#include "nnue.h"
#include "types.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

using pufferfish::file_of;

namespace pufferfish
{

    // =============================================================================
    // Linear Layer Implementation
    // =============================================================================

    void NNUEEvaluator::LinearLayer::forward(const std::vector<float> &input, std::vector<float> &output)
    {
        output.assign(out_dim, 0.0f);

        // Matrix multiply: output = input @ weights^T + bias
        for (int i = 0; i < out_dim; i++)
        {
            float sum = bias[i];
            for (int j = 0; j < in_dim; j++)
            {
                sum += input[j] * weights[i * in_dim + j];
            }

            if (apply_relu)
            {
                output[i] = std::max(0.0f, sum);
            }
            else
            {
                output[i] = sum;
            }
        }
    }

    // =============================================================================
    // LayerNorm Implementation
    // =============================================================================

    void NNUEEvaluator::LayerNormLayer::forward(const std::vector<float> &input, std::vector<float> &output)
    {
        output.assign(size, 0.0f);

        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < size; i++)
        {
            mean += input[i];
        }
        mean /= size;

        // Compute variance
        float variance = 0.0f;
        for (int i = 0; i < size; i++)
        {
            float diff = input[i] - mean;
            variance += diff * diff;
        }
        variance /= size;

        // Normalize and apply affine transform: γ * (x - μ) / sqrt(σ² + ε) + β
        float inv_std = 1.0f / std::sqrt(variance + eps);
        for (int i = 0; i < size; i++)
        {
            output[i] = weight[i] * (input[i] - mean) * inv_std + bias[i];
        }
    }

    // =============================================================================
    // Residual Block Implementation
    // =============================================================================

    void NNUEEvaluator::ResidualBlock::forward(const std::vector<float> &input, std::vector<float> &output)
    {
        output.assign(dim, 0.0f);
        std::vector<float> temp(dim);

        // Apply first linear layer with ReLU: y = ReLU(lin1(x))
        for (int i = 0; i < dim; i++)
        {
            float sum = lin1_bias[i];
            for (int j = 0; j < dim; j++)
            {
                sum += input[j] * lin1_weight[i * dim + j];
            }
            temp[i] = std::max(0.0f, sum);
        }

        // Apply second linear layer without activation: y = lin2(y)
        std::vector<float> lin2_out(dim);
        for (int i = 0; i < dim; i++)
        {
            float sum = lin2_bias[i];
            for (int j = 0; j < dim; j++)
            {
                sum += temp[j] * lin2_weight[i * dim + j];
            }
            lin2_out[i] = sum;
        }

        // Add residual connection: residual + lin2_out
        for (int i = 0; i < dim; i++)
        {
            temp[i] = input[i] + lin2_out[i];
        }

        // Apply layer norm
        float mean = 0.0f;
        for (int i = 0; i < dim; i++)
        {
            mean += temp[i];
        }
        mean /= dim;

        float variance = 0.0f;
        for (int i = 0; i < dim; i++)
        {
            float diff = temp[i] - mean;
            variance += diff * diff;
        }
        variance /= dim;

        float inv_std = 1.0f / std::sqrt(variance + norm_eps);
        std::vector<float> norm_out(dim);
        for (int i = 0; i < dim; i++)
        {
            norm_out[i] = norm_weight[i] * (temp[i] - mean) * inv_std + norm_bias[i];
        }

        // Apply final ReLU: output = ReLU(norm(residual + lin2))
        for (int i = 0; i < dim; i++)
        {
            output[i] = std::max(0.0f, norm_out[i]);
        }
    }

    // =============================================================================
    // Simple JSON Parser for NNUE header
    // =============================================================================

    static int parse_json_int(const std::string &json, const std::string &key)
    {
        std::string search = "\"" + key + "\"";
        size_t pos = json.find(search);
        if (pos == std::string::npos)
            return -1;

        pos += search.length();
        // Skip to colon and whitespace
        while (pos < json.length() && json[pos] != ':')
            pos++;
        if (pos >= json.length())
            return -1;
        pos++; // Skip colon

        // Skip whitespace
        while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t'))
            pos++;

        // Parse integer (may be negative, may be part of array)
        size_t end = pos;
        if (end < json.length() && json[end] == '-')
            end++; // Handle negative numbers
        while (end < json.length() && std::isdigit(json[end]))
            end++;

        if (end == pos || (json[pos] == '-' && end == pos + 1))
            return -1;

        return std::stoi(json.substr(pos, end - pos));
    }

    static std::string parse_json_string(const std::string &json, const std::string &key)
    {
        std::string search = "\"" + key + "\"";
        size_t pos = json.find(search);
        if (pos == std::string::npos)
            return "";

        pos += search.length();
        // Skip to colon
        while (pos < json.length() && json[pos] != ':')
            pos++;
        if (pos >= json.length())
            return "";
        pos++; // Skip colon

        // Skip whitespace
        while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t'))
            pos++;

        // Expect opening quote
        if (pos >= json.length() || json[pos] != '"')
            return "";
        pos++;

        // Find closing quote
        size_t end = json.find("\"", pos);
        if (end == std::string::npos)
            return "";

        return json.substr(pos, end - pos);
    }

    // =============================================================================
    // NNUEEvaluator - Public Interface
    // =============================================================================

    NNUEEvaluator::NNUEEvaluator() : current_input_dim_(INPUT_DIM), ready_(false) {}

    NNUEEvaluator::~NNUEEvaluator() {}

    bool NNUEEvaluator::load(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "[NNUE] Error: Could not open file: " << filename << std::endl;
            return false;
        }

        try
        {
            // Read JSON header length
            uint32_t json_len = 0;
            file.read(reinterpret_cast<char *>(&json_len), sizeof(uint32_t));
            if (file.fail() || json_len == 0 || json_len > 10000)
            {
                std::cerr << "[NNUE] Error: Invalid JSON header length: " << json_len << std::endl;
                return false;
            }

            // Read JSON metadata
            std::vector<char> json_buffer(json_len);
            file.read(json_buffer.data(), json_len);
            if (file.fail())
            {
                std::cerr << "[NNUE] Error: Failed to read JSON header" << std::endl;
                return false;
            }

            std::string json_str(json_buffer.begin(), json_buffer.end());

            // Parse JSON manually (simple key-value extraction)
            std::string format = parse_json_string(json_str, "format");
            int input_dim = parse_json_int(json_str, "input_dim");
            int layer_count = parse_json_int(json_str, "layer_count");

            // Validate format
            if (format != "residual-nnue-v1")
            {
                std::cerr << "[NNUE] Error: Unsupported format: " << format << std::endl;
                return false;
            }

            if (input_dim < 0)
                input_dim = 795; // Default

            if (layer_count <= 0)
            {
                std::cerr << "[NNUE] Error: Invalid layer count: " << layer_count << std::endl;
                return false;
            }

            if (input_dim != INPUT_DIM)
            {
                std::cerr << "[NNUE] Warning: Expected input_dim=" << INPUT_DIM
                          << " but file has " << input_dim << std::endl;
            }

            std::cout << "[NNUE] Loading model: format=" << format
                      << ", input_dim=" << input_dim
                      << ", layers=" << layer_count << std::endl;

            // Read layer count from file
            uint32_t file_layer_count = 0;
            file.read(reinterpret_cast<char *>(&file_layer_count), sizeof(uint32_t));
            if (file.fail())
            {
                std::cerr << "[NNUE] Error: Failed to read layer count" << std::endl;
                return false;
            }

            if (static_cast<int>(file_layer_count) != layer_count)
            {
                std::cerr << "[NNUE] Warning: Metadata says " << layer_count
                          << " layers but file has " << file_layer_count << std::endl;
            }

            layers_.clear();

            // Read each layer
            for (uint32_t layer_idx = 0; layer_idx < file_layer_count; layer_idx++)
            {
                uint32_t type_id = 0;
                file.read(reinterpret_cast<char *>(&type_id), sizeof(uint32_t));
                if (file.fail())
                {
                    std::cerr << "[NNUE] Error: Failed to read layer type at layer " << layer_idx << std::endl;
                    return false;
                }

                LayerType type = static_cast<LayerType>(type_id);

                if (type == LayerType::LINEAR)
                {
                    uint32_t out_dim = 0, in_dim = 0;
                    file.read(reinterpret_cast<char *>(&out_dim), sizeof(uint32_t));
                    file.read(reinterpret_cast<char *>(&in_dim), sizeof(uint32_t));

                    if (file.fail())
                    {
                        std::cerr << "[NNUE] Error: Failed to read LINEAR layer dimensions" << std::endl;
                        return false;
                    }

                    auto linear = std::make_unique<LinearLayer>();
                    linear->in_dim = in_dim;
                    linear->out_dim = out_dim;
                    linear->apply_relu = (layer_idx < file_layer_count - 1); // ReLU for all but last

                    // Read weights: [out_dim][in_dim]
                    linear->weights.resize(out_dim * in_dim);
                    file.read(reinterpret_cast<char *>(linear->weights.data()),
                              out_dim * in_dim * sizeof(float));

                    // Read bias: [out_dim]
                    linear->bias.resize(out_dim);
                    file.read(reinterpret_cast<char *>(linear->bias.data()),
                              out_dim * sizeof(float));

                    if (file.fail())
                    {
                        std::cerr << "[NNUE] Error: Failed to read LINEAR layer weights/bias" << std::endl;
                        return false;
                    }

                    std::cout << "[NNUE] Layer " << layer_idx << ": LINEAR "
                              << in_dim << " -> " << out_dim
                              << " (ReLU: " << linear->apply_relu << ")" << std::endl;

                    layers_.push_back(std::move(linear));
                    current_input_dim_ = out_dim;
                }
                else if (type == LayerType::LAYERNORM)
                {
                    uint32_t size = 0, padding = 0;
                    file.read(reinterpret_cast<char *>(&size), sizeof(uint32_t));
                    file.read(reinterpret_cast<char *>(&padding), sizeof(uint32_t));

                    auto layernorm = std::make_unique<LayerNormLayer>();
                    layernorm->size = size;

                    // Read weight (γ): [size]
                    layernorm->weight.resize(size);
                    file.read(reinterpret_cast<char *>(layernorm->weight.data()),
                              size * sizeof(float));

                    // Read bias (β): [size]
                    layernorm->bias.resize(size);
                    file.read(reinterpret_cast<char *>(layernorm->bias.data()),
                              size * sizeof(float));

                    // Read epsilon
                    file.read(reinterpret_cast<char *>(&layernorm->eps), sizeof(float));

                    if (file.fail())
                    {
                        std::cerr << "[NNUE] Error: Failed to read LAYERNORM layer" << std::endl;
                        return false;
                    }

                    std::cout << "[NNUE] Layer " << layer_idx << ": LAYERNORM " << size << std::endl;

                    layers_.push_back(std::move(layernorm));
                }
                else if (type == LayerType::RESIDUAL)
                {
                    uint32_t dim1 = 0, dim2 = 0;
                    file.read(reinterpret_cast<char *>(&dim1), sizeof(uint32_t));
                    file.read(reinterpret_cast<char *>(&dim2), sizeof(uint32_t));

                    if (dim1 != dim2)
                    {
                        std::cerr << "[NNUE] Error: RESIDUAL block dimensions mismatch: "
                                  << dim1 << " != " << dim2 << std::endl;
                        return false;
                    }

                    auto residual = std::make_unique<ResidualBlock>();
                    residual->dim = dim1;

                    // Read lin1 weights and bias
                    residual->lin1_weight.resize(dim1 * dim1);
                    file.read(reinterpret_cast<char *>(residual->lin1_weight.data()),
                              dim1 * dim1 * sizeof(float));

                    residual->lin1_bias.resize(dim1);
                    file.read(reinterpret_cast<char *>(residual->lin1_bias.data()),
                              dim1 * sizeof(float));

                    // Read lin2 weights and bias
                    residual->lin2_weight.resize(dim1 * dim1);
                    file.read(reinterpret_cast<char *>(residual->lin2_weight.data()),
                              dim1 * dim1 * sizeof(float));

                    residual->lin2_bias.resize(dim1);
                    file.read(reinterpret_cast<char *>(residual->lin2_bias.data()),
                              dim1 * sizeof(float));

                    // Read norm weights and bias
                    residual->norm_weight.resize(dim1);
                    file.read(reinterpret_cast<char *>(residual->norm_weight.data()),
                              dim1 * sizeof(float));

                    residual->norm_bias.resize(dim1);
                    file.read(reinterpret_cast<char *>(residual->norm_bias.data()),
                              dim1 * sizeof(float));

                    // Read norm epsilon
                    file.read(reinterpret_cast<char *>(&residual->norm_eps), sizeof(float));

                    if (file.fail())
                    {
                        std::cerr << "[NNUE] Error: Failed to read RESIDUAL block" << std::endl;
                        return false;
                    }

                    std::cout << "[NNUE] Layer " << layer_idx << ": RESIDUAL " << dim1 << "x" << dim1 << std::endl;

                    layers_.push_back(std::move(residual));
                }
                else
                {
                    std::cerr << "[NNUE] Error: Unknown layer type: " << type_id << std::endl;
                    return false;
                }
            }

            file.close();
            ready_ = true;

            std::cout << "[NNUE] Successfully loaded " << layers_.size() << " layers" << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[NNUE] Error: Exception while loading: " << e.what() << std::endl;
            return false;
        }
    }

    // =============================================================================
    // Feature Encoding
    // =============================================================================

    void NNUEEvaluator::add_piece_features(const Position &pos, std::vector<float> &features)
    {
        // Piece placement features: 12 piece types × 64 squares
        // Index: piece_type * 64 + square
        // Piece types: wP=0, wN=1, wB=2, wR=3, wQ=4, wK=5, bP=6, bN=7, bB=8, bR=9, bQ=10, bK=11

        for (Square sq = 0; sq < 64; sq++)
        {
            Piece piece = pos.piece_on(sq);
            if (piece == NO_PIECE)
                continue;

            int piece_idx = static_cast<int>(piece) - 1; // 0-11
            int feature_idx = piece_idx * 64 + sq;

            if (feature_idx >= 0 && feature_idx < 768)
            {
                features[feature_idx] = 1.0f;
            }
        }
    }

    void NNUEEvaluator::add_board_state_features(const Position &pos, std::vector<float> &features)
    {
        // Features 768+ (27 additional features for total of 795)
        // [768] material, [769] phase, [770] side, [771-774] castling (4), [775-782] ep_file (8), [783-794] reserved (12)

        // Feature 768: Material balance
        float material = 0.0f;
        for (Square sq = 0; sq < 64; sq++)
        {
            Piece piece = pos.piece_on(sq);
            if (piece == NO_PIECE)
                continue;

            int piece_type = piece & 7;              // 1-6 for pawn-king
            float values[] = {0, 1, 3, 3, 5, 9, 0};  // pawn, knight, bishop, rook, queen, king, dummy
            float sign = (piece & 8) ? -1.0f : 1.0f; // black pieces negative
            material += sign * values[piece_type];
        }
        features[768] = std::tanh(material / 40.0f);

        // Feature 769: Game phase
        int piece_count = 0;
        for (Square sq = 0; sq < 64; sq++)
        {
            if (pos.piece_on(sq) != NO_PIECE)
                piece_count++;
        }
        features[769] = 1.0f - (piece_count / 32.0f);

        // Feature 770: Side to move
        features[770] = (pos.side_to_move() == WHITE) ? 1.0f : 0.0f;

        // Features 771-774: Castling rights (4 features)
        CastlingRights cr = pos.castling_rights();
        features[771] = (cr & WHITE_OO) ? 1.0f : 0.0f;
        features[772] = (cr & WHITE_OOO) ? 1.0f : 0.0f;
        features[773] = (cr & BLACK_OO) ? 1.0f : 0.0f;
        features[774] = (cr & BLACK_OOO) ? 1.0f : 0.0f;

        // Features 775-782: En passant file (8 features) - all zeros by default
        // Set only if ep square is valid
        Square ep_sq = pos.ep_square();
        if (ep_sq != SQ_NONE)
        {
            int file = file_of(ep_sq); // Extract file (0-7)
            if (file >= 0 && file < 8)
            {
                features[775 + file] = 1.0f;
            }
        }

        // Features 783-794: reserved/unused (all zeros)
    }

    void NNUEEvaluator::encode_position(const Position &pos, std::vector<float> &features)
    {
        features.assign(INPUT_DIM, 0.0f);
        add_piece_features(pos, features);
        add_board_state_features(pos, features);
    }

    // =============================================================================
    // Forward Pass
    // =============================================================================

    void NNUEEvaluator::forward(const std::vector<float> &input, std::vector<float> &output)
    {
        std::vector<float> current = input;

        for (size_t i = 0; i < layers_.size(); i++)
        {
            std::vector<float> next;
            layers_[i]->forward(current, next);
            current = std::move(next);
        }

        output = current;
    }

    // =============================================================================
    // Evaluation
    // =============================================================================

    int NNUEEvaluator::evaluate(const Position &pos)
    {
        if (!ready_)
        {
            return 0;
        }

        std::vector<float> features;
        encode_position(pos, features);

        std::vector<float> output;
        forward(features, output);

        if (output.empty())
        {
            return 0;
        }

        // Network outputs a single float value
        float raw_score = output[0];

        // Convert to centipawns: multiply by 100 and ensure it's in reasonable range
        int cp_score = static_cast<int>(std::round(raw_score * 100.0f));

        // Clamp to reasonable range (±10000 cp = ±100 pawns)
        cp_score = std::max(-10000, std::min(10000, cp_score));

        // Flip perspective if black to move
        if (pos.side_to_move() == BLACK)
        {
            cp_score = -cp_score;
        }

        return cp_score;
    }

} // namespace pufferfish
