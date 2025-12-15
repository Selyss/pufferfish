# NNUE Architecture Specification

## Overview
The NNUE (Neural Network Utility Evaluation) uses a residual block architecture exported from PyTorch checkpoints into a custom binary format.

## Model Architecture (from reference repo)

### Input Features (795 dimensions)
- **Piece Placement (768 dims)**: 12 piece types × 64 squares × 2 perspectives (white and black to move)
- **Material (1 dim)**: Total material balance (scaled by encoder.material_scale)
- **Phase (1 dim)**: Game phase based on piece count (scaled by encoder.phase_scale)
- **Side to Move (1 dim)**: Binary flag (1.0 for white, 0.0 for black)
- **Castling Rights (4 dims)**: White kingside, white queenside, black kingside, black queenside
- **En Passant (8 dims)**: 8 possible files where en passant is legal

### Network Layers
```
Input (795) 
  → Linear (2048)
  → ReLU + LayerNorm(2048)
  → 2x ResidualBlock(2048) [skip connection + ReLU + dropout + output]
  → Linear (2048)
  → ReLU + LayerNorm(2048)
  → 2x ResidualBlock(2048)
  → Linear (1024)
  → ReLU + LayerNorm(1024)
  → 2x ResidualBlock(1024)
  → Linear (512)
  → ReLU + LayerNorm(512)
  → 2x ResidualBlock(512)
  → Linear (256)
  → ReLU + LayerNorm(256)
  → 2x ResidualBlock(256)
  → Linear (1)  [output head]
  → Output score
```

### ResidualBlock Structure
```
ResidualBlock(dim):
  input → Linear(dim→dim) → ReLU → Dropout → Linear(dim→dim) → 
    + (skip) → ReLU → LayerNorm(dim) → output
```

### Layer Normalization
```
LayerNorm(x) = γ * (x - mean) / sqrt(variance + eps) + β
```

## Binary File Format

### Header Section
```
[4 bytes] JSON length (u32, little-endian)
[variable] JSON metadata: {"format":"residual-nnue-v1","input_dim":795,"layer_count":...}
[4 bytes] Layer count (u32, little-endian)
```

### Layer Section (repeated for each layer)
Each layer begins with type identifier:

#### Type 1: LINEAR Layer
```
[4 bytes] Type ID = 1 (u32)
[4 bytes] Output features (out) (u32)
[4 bytes] Input features (in) (u32)
[out*in*4 bytes] Weights in row-major format (float32)
[out*4 bytes] Bias vector (float32)
```

#### Type 2: LAYERNORM Layer
```
[4 bytes] Type ID = 2 (u32)
[4 bytes] Size (u32)
[4 bytes] Padding = 0 (u32)
[size*4 bytes] Weight γ (float32)
[size*4 bytes] Bias β (float32)
[4 bytes] Epsilon value (float32)
```

#### Type 3: RESIDUAL Block
```
[4 bytes] Type ID = 3 (u32)
[4 bytes] Dimension (u32)
[4 bytes] Dimension (u32)
[dim*4 bytes] lin1 weight (float32)
[dim*4 bytes] lin1 bias (float32)
[dim*4 bytes] lin2 weight (float32)
[dim*4 bytes] lin2 bias (float32)
[dim*4 bytes] norm weight γ (float32)
[dim*4 bytes] norm bias β (float32)
[4 bytes] norm epsilon (float32)
```

## Forward Pass Algorithm

### 1. Encode Position
Extract 795 features from chess board state as floats [0.0, 1.0].

### 2. Forward Through Layers
For each LINEAR layer:
```
output[i] = bias[i] + Σ(input[j] * weight[j, i])
```
Apply ReLU activation (except output layer):
```
output[i] = max(0, output[i])
```

For LAYERNORM:
```
mean = Σ(x[i]) / size
variance = Σ((x[i] - mean)²) / size
output[i] = γ[i] * (x[i] - mean) / sqrt(variance + eps) + β[i]
```

For RESIDUAL:
```
residual = input
y = ReLU(lin1(input))
y = dropout(y)
y = lin2(y)
output = ReLU(norm(residual + y))
```

### 3. Output
Final output is a single float value representing evaluation in centi-pawns.
- Positive = advantage for side to move
- Negative = disadvantage for side to move

## Known Configuration
- Input dim: 795
- Hidden dims: [2048, 2048, 1024, 512, 256]
- Residual repeats: [2, 2, 2, 2, 2]
- Activation: ReLU + LayerNorm in hidden layers, Linear in output
- Dropout rate: 0.05 (not needed for inference-only)
- Total layers exported: ~52 (5 stages × (1 linear + 1 layernorm + 2 residual) + 1 output)
