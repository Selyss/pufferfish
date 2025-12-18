# NNUE Architecture Specification (train_modal.py)

## Overview
This project uses a compact accumulator NNUE trained in `train_modal.py`. The model is exported with `export_int16.py` into a quantized int16 binary format and evaluated with incremental accumulator updates.

## Feature Encoding (768 dimensions)
Piece placement only:

- **Indexing:** `feature_idx = square * 12 + piece_offset`
- **Square:** 0..63 (a1=0, h8=63)
- **Piece offsets:**
  - 0..5: White pawn, knight, bishop, rook, queen, king
  - 6..11: Black pawn, knight, bishop, rook, queen, king
- **Value:** 1.0 if piece exists on that square, else 0.0

There are no side-to-move, castling, en passant, phase, or material features.

## Network Architecture
```
Input (768)
  -> acc_friendly: Linear(768 -> 256) + ReLU
  -> acc_enemy:    Linear(768 -> 256) + ReLU
  -> concat(512)
  -> fc1: Linear(512 -> 32) + ReLU
  -> fc2: Linear(32 -> 32) + ReLU
  -> fc_out: Linear(32 -> 1)
```

The model is trained to predict `cp_label` values. In the engine, the final score is interpreted as centipawns and flipped for black to move to maintain side-to-move perspective.

## Binary File Format (export_int16.py)
```
[4 bytes] FEATURE_DIM (int32)
[4 bytes] ACC_UNITS (int32)
[4 bytes] HIDDEN1 (int32)
[4 bytes] HIDDEN2 (int32)

[ACC_UNITS * 4] acc_friendly bias (int32)
[ACC_UNITS * 4] acc_enemy bias (int32)

For each feature f in [0..FEATURE_DIM-1]:
  [ACC_UNITS * 2 bytes] acc_friendly weights (int16)
  [ACC_UNITS * 2 bytes] acc_enemy weights (int16)

[HIDDEN1 * 4] fc1 bias (int32)
[HIDDEN1 * 2*ACC_UNITS] fc1 weights (int16)

[HIDDEN2 * 4] fc2 bias (int32)
[HIDDEN2 * HIDDEN1] fc2 weights (int16)

[4 bytes] out bias (int32)
[HIDDEN2 * 2 bytes] out weights (int16)
```

## Incremental Accumulator Updates
The engine maintains two int32 accumulators (friendly/enemy). On each move:
- Remove the feature for the moving piece at `from`
- Remove the feature for any captured piece
- Add the feature for the moving/promoted piece at `to`
- If castling, update the rook feature as well

Accumulator snapshots are stored in `Undo` for fast unmake.
