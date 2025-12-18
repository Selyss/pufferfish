"""
NNUE export configuration aligned with train_modal.py architecture.
"""

FEATURE_DIM = 768
ACC_UNITS = 256
HIDDEN1 = 32
HIDDEN2 = 32

# Quantization parameters. These must match the C++ inference expectations.
SCALE1 = 1.0
SCALE2 = 1.0
OUTPUT_SCALE_BITS = 0

# Placeholder for compatibility with exporter interface.
RELU_CLIP = 0
