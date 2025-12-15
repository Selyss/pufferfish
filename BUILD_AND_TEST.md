# Pufferfish Build & Test Guide

## Quick Start

### Using CMake (Recommended)

```bash
cd c:\Users\jtred\code\pufferfish
mkdir build && cd build
cmake ..
cmake --build .

# Run tests
./perft.exe           # All 46 perft tests (should see 46 passed, 0 failed)
./zobrist_test.exe    # Zobrist unit tests (should see ALL PASSED)
```

### Using g++ Directly

```bash
cd c:\Users\jtred\code\pufferfish

# Build and run Zobrist tests
g++ -std=c++20 -I./src -o apps/zobrist_test.exe \
    apps/zobrist_test.cpp src/position.cpp src/zobrist.cpp \
    src/attack.cpp src/movegen.cpp src/perft.cpp
./apps/zobrist_test.exe

# Build and run Perft tests
g++ -std=c++20 -I./src -o apps/perft.exe \
    apps/perft_main.cpp src/position.cpp src/zobrist.cpp \
    src/attack.cpp src/movegen.cpp src/perft.cpp
./apps/perft.exe

# Specific perft commands
./apps/perft.exe 5                                  # Depth 5 on startpos
./apps/perft.exe 4 "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"  # Custom FEN
./apps/perft.exe divide 3 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Perft divide
```

## Expected Test Results

### Zobrist Tests (3 test groups)
```
=== Test 1: Zobrist Determinism ===
[PASS] startpos             0x6d5066f2ebb6cf2a
[PASS] empty board          0xa3208b89fb2bfce0
[PASS] with ep square       0xa518c1480be29f8
[PASS] limited castling     0xada0c685ffe55a4b

=== Test 2: Zobrist Make/Unmake Consistency ===
[PASS] Move 1-5 (keys restored after unmake)

=== Test 3: Zobrist Uniqueness ===
[PASS] Different positions have different keys

Results: ALL PASSED
```

### Perft Tests (46 comprehensive test cases)
```
Results: 46 passed, 0 failed

Includes:
- Starting position (d1-5)
- Kiwipete complex position (d1-4)
- Position 3-6 from CPW (various depths)
- Castling tests
- Promotion tests
- En passant tests
- Pin tests
- Stalemate tests
- King vs King endgame
```

## Performance Baseline

- **Perft d5**: ~540K nodes/sec
- **Move generation**: ~5M moves/sec
- **Build time**: ~2-3 seconds (g++) / ~5-10 seconds (CMake)

## File Structure

```
src/
├── types.h              # Core type definitions
├── move.h               # Move encoding
├── position.h/cpp       # Position class (FEN, make/unmake)
├── attack.h/cpp         # Attack detection
├── movegen.h/cpp        # Move generation
├── perft.h/cpp          # Perft & test suite
└── zobrist.h/cpp        # Zobrist hashing

apps/
├── perft_main.cpp       # Perft runner
└── zobrist_test.cpp     # Zobrist test harness

build/
├── CMakeFiles/
├── perft.exe
└── zobrist_test.exe
```

## Troubleshooting

### Build fails with "zobrist not found"
- Ensure `src/zobrist.h` and `src/zobrist.cpp` exist
- Verify CMakeLists.txt includes `src/zobrist.cpp` in the library target
- Run `cmake --clean` then `cmake ..` again

### Perft test fails (mismatch in node count)
- This indicates a move generation bug (not related to Zobrist)
- Run `perft divide <depth>` to find which move type is wrong
- Compare against known correct node counts in [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)

### Zobrist test fails (key mismatch)
- Use "Zobrist Determinism" test to verify same FEN produces same key
- Use "Make/Unmake Consistency" to check key restoration
- If different positions have same key, hash collision detected (extremely rare)

## Next Steps

After confirming all tests pass, proceed with:

1. **Search Implementation** (Phase 1)
   - Alpha-beta pruning
   - Transposition table using Zobrist keys
   - Move ordering heuristics

2. **NNUE Integration** (Phase 2)
   - Incremental accumulator updates in make/unmake
   - Neural network evaluation

3. **UCI Protocol** (Phase 3)
   - Input/output protocol for GUIs
   - Move validation and time management

4. **Lichess Integration** (Phase 4)
   - Online bot play
   - Repetition detection for draws

---

**Status**: Core rules + Zobrist complete; Ready for search
**Last Updated**: December 14, 2025
