# Pufferfish Chess Engine

## Project Status

**Current Stage:** Core Rules & Perft Validation

This stage implements the foundational "rules core" that all other components depend on:
- Position representation and FEN parsing
- Move generation (all piece types, special moves)
- Make/Unmake operations
- Attack detection
- Perft validation

## Architecture Overview

```
pufferfish/
├── src/
│   ├── types.h        # Core types: Square, Piece, Color, CastlingRights
│   ├── move.h         # Move representation (16-bit compact encoding)
│   ├── position.h/cpp # Board state, FEN parsing, make/unmake
│   ├── attack.h/cpp   # Square attack detection
│   ├── movegen.h/cpp  # Pseudo-legal and legal move generation
│   └── perft.h/cpp    # Perft, perft divide, test suite
├── apps/
│   └── perft_main.cpp # Command-line perft runner
└── CMakeLists.txt     # Build configuration
```

### Key Components

| Component | Description |
|-----------|-------------|
| **types.h** | Defines `Square` (0-63 indexing), `Piece`, `Color`, `CastlingRights` bitmask, and directional constants |
| **move.h** | 16-bit move encoding with `from`, `to`, and flags (capture, castle, promotion, en passant). Includes `Undo` struct for unmake |
| **position.cpp** | Mailbox board representation. Handles FEN parsing, piece placement, and complete make/unmake for all move types |
| **attack.cpp** | Detects if a square is attacked by pawns, knights, bishops, rooks, queens, or kings |
| **movegen.cpp** | Generates pseudo-legal moves (ignoring check), then filters to legal moves via make/in_check/unmake |
| **perft.cpp** | Node counting for validation, perft divide for debugging, built-in test suite with known positions |

## Building

### Prerequisites

- CMake 3.16+
- C++20 compatible compiler:
  - **Windows:** MSVC 2019+ or MinGW-w64 with GCC 10+
  - **Linux/macOS:** GCC 10+ or Clang 10+

### Build Commands

#### Windows (MSVC / Visual Studio)

```powershell
# From the pufferfish directory
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

The executable will be at `build\Release\perft.exe`

#### Windows (MinGW)

```powershell
mkdir build
cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

The executable will be at `build\perft.exe`

#### Linux / macOS

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The executable will be at `build/perft`

### Debug Build

For development with assertions enabled:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```

## Usage

### Run Full Test Suite

```bash
./perft
```

Runs all built-in perft tests (startpos, kiwipete, en passant, promotions, castling).

### Perft at Specific Depth

```bash
./perft 5                    # Perft on starting position
./perft 4 "r3k2r/..."        # Perft on custom FEN
```

### Perft Divide (Debugging)

When perft counts don't match, use divide to find which move causes the discrepancy:

```bash
./perft divide 3             # Divide on starting position
./perft divide 2 "<fen>"     # Divide on custom position
```

This prints node counts for each root move, helping isolate bugs to specific move types.

## Perft Reference Values

| Position | Depth | Nodes |
|----------|-------|-------|
| Starting | 1 | 20 |
| Starting | 2 | 400 |
| Starting | 3 | 8,902 |
| Starting | 4 | 197,281 |
| Starting | 5 | 4,865,609 |
| Kiwipete | 1 | 48 |
| Kiwipete | 2 | 2,039 |
| Kiwipete | 3 | 97,862 |
| Kiwipete | 4 | 4,085,603 |

## Design Principles

1. **Correctness First:** Perft validation before any optimization
2. **Clear Boundaries:** Each module has a single responsibility
3. **Make/Unmake Purity:** No hidden global state; positions restore exactly
4. **Incremental Development:** Each feature tested before moving on

## Roadmap

- [ ] Zobrist hashing and repetition detection
- [ ] NNUE evaluation backend
- [ ] Alpha-beta search with pruning
- [ ] UCI protocol support
- [ ] Lichess bot integration