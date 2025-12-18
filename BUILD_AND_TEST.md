# Pufferfish Build & Test Guide

## Quick Start (CMake)
```bash
mkdir build
cd build
cmake ..
cmake --build .

# Run tests
./perft.exe
./zobrist_test.exe
./transposition_table_test.exe
./search_test.exe
./nnue_loader_test.exe
```

## NNUE Model Setup
- Place the exported model at `models/nnue_weights.bin`.
- The engine falls back to material evaluation if the file is missing.

## Manual g++ Build (Example)
```bash
g++ -std=c++20 -I./src -o apps/nnue_loader_test.exe \
    apps/nnue_loader_test.cpp src/nnue.cpp src/position.cpp src/zobrist.cpp \
    src/attack.cpp src/movegen.cpp src/perft.cpp
./apps/nnue_loader_test.exe
```

## Troubleshooting
- If NNUE fails to load, confirm the file format matches `export_int16.py`.
- If search tests fail with NNUE enabled, verify the model output scale.
