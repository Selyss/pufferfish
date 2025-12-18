/*
 * Pufferfish Chess Engine
 * NNUE shared constants (train_modal.py architecture)
 */

#ifndef PUFFERFISH_NNUE_DEFS_H
#define PUFFERFISH_NNUE_DEFS_H

namespace pufferfish
{
    constexpr int NNUE_FEATURE_DIM = 768;
    constexpr int NNUE_ACC_UNITS = 256;
    constexpr int NNUE_HIDDEN1 = 32;
    constexpr int NNUE_HIDDEN2 = 32;
} // namespace pufferfish

#endif // PUFFERFISH_NNUE_DEFS_H
