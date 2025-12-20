/*
 * Pufferfish Chess Engine
 * NNUE Evaluation - Residual NNUE loader (export_int16.py format)
 */

#include "nnue.h"
#include "types.h"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace pufferfish
{
    namespace
    {
        bool read_exact(std::ifstream &file, void *dst, size_t bytes)
        {
            return static_cast<bool>(file.read(reinterpret_cast<char *>(dst), bytes));
        }

        bool read_u32(std::ifstream &file, uint32_t &out)
        {
            return read_exact(file, &out, sizeof(out));
        }

        std::string read_json(std::ifstream &file)
        {
            uint32_t len = 0;
            if (!read_u32(file, len))
                return {};
            std::string json(len, '\0');
            if (!read_exact(file, json.data(), len))
                return {};
            return json;
        }

        bool parse_json_int(const std::string &json, const std::string &key, int &value)
        {
            const std::string pattern = "\"" + key + "\":";
            auto pos = json.find(pattern);
            if (pos == std::string::npos)
                return false;
            pos += pattern.size();
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t'))
                ++pos;
            size_t end = pos;
            while (end < json.size() && (json[end] == '-' || (json[end] >= '0' && json[end] <= '9')))
                ++end;
            if (end == pos)
                return false;
            value = std::stoi(json.substr(pos, end - pos));
            return true;
        }
    } // namespace

    NNUEEvaluator::NNUEEvaluator() = default;

    bool NNUEEvaluator::load(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "[NNUE] Error: Could not open file: " << filename << std::endl;
            return false;
        }

        std::string json = read_json(file);
        if (json.empty())
        {
            std::cerr << "[NNUE] Error: Failed to read JSON header" << std::endl;
            return false;
        }

        int input_dim = 0;
        int layer_count = 0;
        if (!parse_json_int(json, "input_dim", input_dim) ||
            !parse_json_int(json, "layer_count", layer_count))
        {
            std::cerr << "[NNUE] Error: Invalid JSON header: " << json << std::endl;
            return false;
        }

        uint32_t layer_count_file = 0;
        if (!read_u32(file, layer_count_file))
        {
            std::cerr << "[NNUE] Error: Failed to read layer count" << std::endl;
            return false;
        }

        if (layer_count_file != static_cast<uint32_t>(layer_count))
        {
            std::cerr << "[NNUE] Warning: Header layer_count (" << layer_count
                      << ") != file layer_count (" << layer_count_file << ")" << std::endl;
        }

        std::vector<Layer> layers;
        layers.reserve(layer_count_file);

        for (uint32_t i = 0; i < layer_count_file; ++i)
        {
            uint32_t type = 0;
            if (!read_u32(file, type))
            {
                std::cerr << "[NNUE] Error: Failed to read layer type" << std::endl;
                return false;
            }

            Layer layer;
            layer.type = static_cast<LayerType>(type);

            if (layer.type == LAYER_LINEAR)
            {
                uint32_t out = 0, in = 0;
                if (!read_u32(file, out) || !read_u32(file, in))
                {
                    std::cerr << "[NNUE] Error: Failed to read linear dimensions" << std::endl;
                    return false;
                }
                layer.in = static_cast<int>(in);
                layer.out = static_cast<int>(out);
                layer.weights.resize(static_cast<size_t>(out) * in);
                layer.bias.resize(out);

                if (!read_exact(file, layer.weights.data(), layer.weights.size() * sizeof(float)) ||
                    !read_exact(file, layer.bias.data(), layer.bias.size() * sizeof(float)))
                {
                    std::cerr << "[NNUE] Error: Failed to read linear weights" << std::endl;
                    return false;
                }
            }
            else if (layer.type == LAYER_LAYERNORM)
            {
                uint32_t size = 0, padding = 0;
                if (!read_u32(file, size) || !read_u32(file, padding))
                {
                    std::cerr << "[NNUE] Error: Failed to read layernorm header" << std::endl;
                    return false;
                }
                layer.out = static_cast<int>(size);
                layer.weights.resize(size);
                layer.bias.resize(size);
                if (!read_exact(file, layer.weights.data(), layer.weights.size() * sizeof(float)) ||
                    !read_exact(file, layer.bias.data(), layer.bias.size() * sizeof(float)) ||
                    !read_exact(file, &layer.eps, sizeof(float)))
                {
                    std::cerr << "[NNUE] Error: Failed to read layernorm weights" << std::endl;
                    return false;
                }
            }
            else if (layer.type == LAYER_RESIDUAL)
            {
                uint32_t dim_a = 0, dim_b = 0;
                if (!read_u32(file, dim_a) || !read_u32(file, dim_b))
                {
                    std::cerr << "[NNUE] Error: Failed to read residual header" << std::endl;
                    return false;
                }
                layer.dim = static_cast<int>(dim_a);
                layer.has_norm = true;

                size_t mat_size = static_cast<size_t>(layer.dim) * layer.dim;
                layer.w1.resize(mat_size);
                layer.b1.resize(layer.dim);
                layer.w2.resize(mat_size);
                layer.b2.resize(layer.dim);
                layer.norm_w.resize(layer.dim);
                layer.norm_b.resize(layer.dim);

                if (!read_exact(file, layer.w1.data(), layer.w1.size() * sizeof(float)) ||
                    !read_exact(file, layer.b1.data(), layer.b1.size() * sizeof(float)) ||
                    !read_exact(file, layer.w2.data(), layer.w2.size() * sizeof(float)) ||
                    !read_exact(file, layer.b2.data(), layer.b2.size() * sizeof(float)) ||
                    !read_exact(file, layer.norm_w.data(), layer.norm_w.size() * sizeof(float)) ||
                    !read_exact(file, layer.norm_b.data(), layer.norm_b.size() * sizeof(float)) ||
                    !read_exact(file, &layer.eps, sizeof(float)))
                {
                    std::cerr << "[NNUE] Error: Failed to read residual weights" << std::endl;
                    return false;
                }
            }
            else if (layer.type == LAYER_COMPACT_RESIDUAL)
            {
                uint32_t dim_a = 0, dim_b = 0;
                if (!read_u32(file, dim_a) || !read_u32(file, dim_b))
                {
                    std::cerr << "[NNUE] Error: Failed to read compact residual header" << std::endl;
                    return false;
                }
                layer.dim = static_cast<int>(dim_a);
                layer.has_norm = false;

                size_t mat_size = static_cast<size_t>(layer.dim) * layer.dim;
                layer.w1.resize(mat_size);
                layer.b1.resize(layer.dim);
                layer.w2.resize(mat_size);
                layer.b2.resize(layer.dim);

                if (!read_exact(file, layer.w1.data(), layer.w1.size() * sizeof(float)) ||
                    !read_exact(file, layer.b1.data(), layer.b1.size() * sizeof(float)) ||
                    !read_exact(file, layer.w2.data(), layer.w2.size() * sizeof(float)) ||
                    !read_exact(file, layer.b2.data(), layer.b2.size() * sizeof(float)))
                {
                    std::cerr << "[NNUE] Error: Failed to read compact residual weights" << std::endl;
                    return false;
                }
            }
            else
            {
                std::cerr << "[NNUE] Error: Unknown layer type " << type << std::endl;
                return false;
            }

            layers.push_back(std::move(layer));
        }

        input_dim_ = input_dim;
        layers_ = std::move(layers);
        ready_ = true;

        std::cout << "[NNUE] Loaded model: input_dim=" << input_dim_
                  << ", layers=" << layers_.size() << std::endl;
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

    void NNUEEvaluator::encode_features(const Position &pos, std::vector<float> &features) const
    {
        const int base_dim = 768;
        const int extra_dim = 27;
        const int expected_dim = base_dim + extra_dim;

        if (input_dim_ < expected_dim)
        {
            features.assign(static_cast<size_t>(input_dim_), 0.0f);
            return;
        }

        features.assign(static_cast<size_t>(input_dim_), 0.0f);

        for (Square sq = SQ_A1; sq <= SQ_H8; ++sq)
        {
            Piece p = pos.piece_on(sq);
            if (p == NO_PIECE)
                continue;
            int offset = piece_offset(p);
            if (offset < 0)
                continue;
            int idx = offset * 64 + static_cast<int>(sq);
            if (idx >= 0 && idx < base_dim)
                features[static_cast<size_t>(idx)] = 1.0f;
        }

        int idx = base_dim;

        // Side to move
        features[static_cast<size_t>(idx++)] = (pos.side_to_move() == WHITE) ? 1.0f : 0.0f;

        // Castling rights
        CastlingRights cr = pos.castling_rights();
        features[static_cast<size_t>(idx++)] = (cr & WHITE_OO) ? 1.0f : 0.0f;
        features[static_cast<size_t>(idx++)] = (cr & WHITE_OOO) ? 1.0f : 0.0f;
        features[static_cast<size_t>(idx++)] = (cr & BLACK_OO) ? 1.0f : 0.0f;
        features[static_cast<size_t>(idx++)] = (cr & BLACK_OOO) ? 1.0f : 0.0f;

        // En passant file
        Square ep = pos.ep_square();
        if (ep != SQ_NONE)
        {
            int file = file_of(ep);
            if (file >= 0 && file < 8)
                features[static_cast<size_t>(idx + file)] = 1.0f;
        }
        idx += 8;

        // Material balance and piece counts
        constexpr int PAWN_VALUE = 100;
        constexpr int KNIGHT_VALUE = 320;
        constexpr int BISHOP_VALUE = 330;
        constexpr int ROOK_VALUE = 500;
        constexpr int QUEEN_VALUE = 900;
        constexpr int KING_VALUE = 0;

        int material = 0;
        int piece_counts[2][6] = {};

        for (Square sq = SQ_A1; sq <= SQ_H8; ++sq)
        {
            Piece p = pos.piece_on(sq);
            if (p == NO_PIECE)
                continue;
            Color c = color_of(p);
            PieceType pt = type_of(p);
            int value = 0;
            switch (pt)
            {
            case PAWN:
                value = PAWN_VALUE;
                piece_counts[c][0]++;
                break;
            case KNIGHT:
                value = KNIGHT_VALUE;
                piece_counts[c][1]++;
                break;
            case BISHOP:
                value = BISHOP_VALUE;
                piece_counts[c][2]++;
                break;
            case ROOK:
                value = ROOK_VALUE;
                piece_counts[c][3]++;
                break;
            case QUEEN:
                value = QUEEN_VALUE;
                piece_counts[c][4]++;
                break;
            case KING:
                value = KING_VALUE;
                piece_counts[c][5]++;
                break;
            default:
                break;
            }
            material += (c == WHITE) ? value : -value;
        }

        features[static_cast<size_t>(idx++)] = static_cast<float>(material) / 2000.0f;

        // Piece counts (white then black, P N B R Q K)
        for (int color = 0; color < 2; ++color)
        {
            for (int pt = 0; pt < 6; ++pt)
            {
                features[static_cast<size_t>(idx++)] = piece_counts[color][pt] / 8.0f;
            }
        }

        int total_pieces = 0;
        for (int color = 0; color < 2; ++color)
        {
            for (int pt = 0; pt < 6; ++pt)
                total_pieces += piece_counts[color][pt];
        }
        features[static_cast<size_t>(idx++)] = total_pieces / 32.0f;
    }

    float NNUEEvaluator::dot_row(const float *row, const std::vector<float> &x, int n)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
            sum += row[i] * x[static_cast<size_t>(i)];
        return sum;
    }

    void NNUEEvaluator::layernorm(std::vector<float> &x, const std::vector<float> &w,
                                  const std::vector<float> &b, float eps)
    {
        float mean = 0.0f;
        for (float v : x)
            mean += v;
        mean /= static_cast<float>(x.size());

        float var = 0.0f;
        for (float v : x)
        {
            float d = v - mean;
            var += d * d;
        }
        var /= static_cast<float>(x.size());

        float inv = 1.0f / std::sqrt(var + eps);
        for (size_t i = 0; i < x.size(); ++i)
        {
            x[i] = w[i] * ((x[i] - mean) * inv) + b[i];
        }
    }

    int NNUEEvaluator::evaluate(Position &pos) const
    {
        if (!ready_ || layers_.empty())
            return 0;

        encode_features(pos, scratch_.features);
        std::vector<float> *x = &scratch_.features;
        std::vector<float> *y = &scratch_.buf_a;
        std::vector<float> *z = &scratch_.buf_b;

        for (size_t li = 0; li < layers_.size(); ++li)
        {
            const Layer &layer = layers_[li];

            if (layer.type == LAYER_LINEAR)
            {
                y->assign(static_cast<size_t>(layer.out), 0.0f);
                const int in = layer.in;
                for (int i = 0; i < layer.out; ++i)
                {
                    const float *row = layer.weights.data() + static_cast<size_t>(i) * in;
                    (*y)[static_cast<size_t>(i)] =
                        layer.bias[static_cast<size_t>(i)] + dot_row(row, *x, in);
                }

                const bool is_output = (li + 1 == layers_.size());
                if (!is_output)
                {
                    for (float &v : *y)
                        v = relu(v);
                }
                std::swap(x, y);
            }
            else if (layer.type == LAYER_LAYERNORM)
            {
                layernorm(*x, layer.weights, layer.bias, layer.eps);
            }
            else if (layer.type == LAYER_RESIDUAL || layer.type == LAYER_COMPACT_RESIDUAL)
            {
                const int dim = layer.dim;
                y->assign(static_cast<size_t>(dim), 0.0f);

                for (int i = 0; i < dim; ++i)
                {
                    const float *row = layer.w1.data() + static_cast<size_t>(i) * dim;
                    (*y)[static_cast<size_t>(i)] =
                        layer.b1[static_cast<size_t>(i)] + dot_row(row, *x, dim);
                    (*y)[static_cast<size_t>(i)] = relu((*y)[static_cast<size_t>(i)]);
                }

                z->assign(static_cast<size_t>(dim), 0.0f);
                for (int i = 0; i < dim; ++i)
                {
                    const float *row = layer.w2.data() + static_cast<size_t>(i) * dim;
                    (*z)[static_cast<size_t>(i)] =
                        layer.b2[static_cast<size_t>(i)] + dot_row(row, *y, dim);
                }

                for (int i = 0; i < dim; ++i)
                    (*z)[static_cast<size_t>(i)] =
                        (*x)[static_cast<size_t>(i)] + (*z)[static_cast<size_t>(i)];

                if (layer.type == LAYER_RESIDUAL && layer.has_norm)
                {
                    layernorm(*z, layer.norm_w, layer.norm_b, layer.eps);
                }

                for (float &v : *z)
                    v = relu(v);

                std::swap(x, z);
            }
        }

        if (x->empty())
            return 0;

        float score = (*x)[0];
        if (pos.side_to_move() == BLACK)
            score = -score;

        return static_cast<int>(std::lround(score * 100.0f));
    }

    void NNUEEvaluator::update_after_move(Position & /*pos*/, const Move & /*m*/, const Undo & /*undo*/) const
    {
        // No incremental accumulator for residual NNUE; evaluation recomputes features.
    }

    void NNUEEvaluator::refresh_accumulator(Position & /*pos*/) const
    {
        // No-op for residual NNUE.
    }

} // namespace pufferfish
