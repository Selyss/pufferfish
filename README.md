# Pufferfish Chess Engine

## Project Status
Core rules, Zobrist hashing, transposition table, alpha-beta search, and NNUE evaluation (accumulator + incremental updates) are implemented.

## Architecture Overview
```
src/
  types.h                    # Core enums and helpers
  move.h                     # Move encoding + Undo state
  position.h/cpp             # Board state and make/unmake
  attack.h/cpp               # Attack detection
  movegen.h/cpp              # Pseudo-legal + legal move generation
  perft.h/cpp                # Perft + test suite
  zobrist.h/cpp              # Zobrist hashing
  transposition_table.h/cpp  # TT cache
  search.h/cpp               # Alpha-beta search
  nnue.h/cpp                 # NNUE loader + accumulator inference

apps/
  perft_main.cpp
  zobrist_test.cpp
  transposition_table_test.cpp
  search_test.cpp
  nnue_loader_test.cpp
```

## NNUE Model
- Architecture matches `train_modal.py` (768 features, 256 accumulators, 2x32 hidden layers).
- Quantized binary format is produced by `export_int16.py`.
- Default model filename: `models/nnue_weights.bin`.

## Building
```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Tests
```
./perft.exe
./zobrist_test.exe
./transposition_table_test.exe
./search_test.exe
./nnue_loader_test.exe
```

## Notes
- If `models/nnue_weights.bin` is not present, the engine falls back to material evaluation.
- See `NNUE_SPECIFICATION.md` and `NNUE_IMPLEMENTATION.md` for details.
