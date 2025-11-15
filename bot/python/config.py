# Shared constants between Python and C++.
# Keep these in sync with cpp/nnue_constants.h.

SQUARES = 64
PIECE_TYPES = 6      # P, N, B, R, Q, K per color
COLORS = 2          # white, black

# SimpleNNUE architecture: 795 -> 2048 -> 2048 -> 1024 -> 512 -> 256 -> 1
FEATURE_DIM = 795    # Enhanced feature set (piece-square + game state)
HIDDEN1 = 2048
HIDDEN2 = 2048
HIDDEN3 = 1024
HIDDEN4 = 512
HIDDEN5 = 256

DROPOUT_RATE = 0.05

# Quantization parameters (for export)
RELU_CLIP = 255
OUTPUT_SCALE_BITS = 5    # 2^5 = 32
SCALE1 = 32.0            # scale for first stage
SCALE2 = 2.0 ** OUTPUT_SCALE_BITS

