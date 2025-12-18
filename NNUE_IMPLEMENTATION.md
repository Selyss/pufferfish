# NNUE Implementation Status

## Status: Implemented (Accumulator + Incremental Updates)

The NNUE evaluation is implemented using the compact accumulator architecture from `train_modal.py` and the quantized int16 export format from `export_int16.py`.

## What Is Implemented

### Model Loader
- Parses the int16 binary format (feature dims, accumulator dims, FC layers).
- Validates dimensions against the training architecture.
- Stores weights in feature-major layout for fast accumulator updates.

### Feature Encoding
- 768-dim one-hot piece placement.
- Index: `square * 12 + piece_offset` with 0..5 white, 6..11 black.

### Accumulator + Incremental Updates
- Accumulators stored in `Position` as int32 arrays.
- `make_move` saves accumulator state into `Undo`.
- `NNUEEvaluator::update_after_move` applies deltas for moved/captured/promoted pieces.
- `unmake_move` restores accumulator state from `Undo`.
- `NNUEEvaluator::refresh_accumulator` can recompute from scratch for safety.

### Inference
```
acc_f = ReLU(acc_friendly)
acc_e = ReLU(acc_enemy)
fc1 = ReLU(W1 * [acc_f, acc_e] + b1)
fc2 = ReLU(W2 * fc1 + b2)
out = Wout * fc2 + bout
```
The output is interpreted as centipawns and flipped for black to move.

## Key Files
- `src/nnue.h` / `src/nnue.cpp`: loader + inference + incremental updates
- `src/position.h` / `src/position.cpp`: accumulator storage and undo snapshots
- `apps/nnue_loader_test.cpp`: loader + incremental update smoke test

## Notes
- The model assumes no extra features (stm, castling, ep, phase).
- The exported file must use the same dims as `train_modal.py`.
