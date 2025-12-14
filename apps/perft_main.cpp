/*
 * Pufferfish Chess Engine
 * Perft runner - validates move generation correctness
 *
 * Usage:
 *   perft                          - Run built-in test suite
 *   perft <depth>                  - Run perft on startpos at given depth
 *   perft <depth> "<fen>"          - Run perft on given FEN at given depth
 *   perft divide <depth>           - Run perft divide on startpos
 *   perft divide <depth> "<fen>"   - Run perft divide on given FEN
 */

#include "position.h"
#include "perft.h"
#include <iostream>
#include <string>
#include <chrono>
#include <cstring>

using namespace pufferfish;

void print_usage()
{
    std::cout << "Pufferfish Perft Runner\n"
              << "========================\n\n"
              << "Usage:\n"
              << "  perft                        - Run built-in test suite\n"
              << "  perft <depth>                - Perft on starting position\n"
              << "  perft <depth> \"<fen>\"        - Perft on custom FEN\n"
              << "  perft divide <depth>         - Perft divide on starting position\n"
              << "  perft divide <depth> \"<fen>\" - Perft divide on custom FEN\n"
              << "\nExamples:\n"
              << "  perft 5\n"
              << "  perft 4 \"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1\"\n"
              << "  perft divide 3\n";
}

int main(int argc, char *argv[])
{
    std::cout << "=================================\n";
    std::cout << " Pufferfish Chess Engine v0.1.0\n";
    std::cout << " Perft Validation Tool\n";
    std::cout << "=================================\n";

    // No arguments - run test suite
    if (argc == 1)
    {
        bool passed = run_perft_suite(std::cout);
        return passed ? 0 : 1;
    }

    // Check for divide mode
    bool divide_mode = false;
    int arg_offset = 1;

    if (std::strcmp(argv[1], "divide") == 0)
    {
        divide_mode = true;
        arg_offset = 2;
        if (argc < 3)
        {
            print_usage();
            return 1;
        }
    }

    if (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0)
    {
        print_usage();
        return 0;
    }

    // Parse depth
    int depth;
    try
    {
        depth = std::stoi(argv[arg_offset]);
    }
    catch (...)
    {
        std::cerr << "Error: Invalid depth '" << argv[arg_offset] << "'\n";
        return 1;
    }

    if (depth < 0 || depth > 10)
    {
        std::cerr << "Error: Depth must be between 0 and 10\n";
        return 1;
    }

    // Get FEN (default to startpos)
    std::string fen = STARTPOS_FEN;
    if (argc > arg_offset + 1)
    {
        fen = argv[arg_offset + 1];
    }

    // Set up position
    Position pos;
    if (!pos.set_fen(fen))
    {
        std::cerr << "Error: Invalid FEN string\n";
        return 1;
    }

    std::cout << "\n";
    pos.print(std::cout);
    std::cout << "\n";

    if (divide_mode)
    {
        perft_divide(pos, depth, std::cout);
    }
    else
    {
        std::cout << "Running perft at depth " << depth << "...\n\n";

        auto start = std::chrono::high_resolution_clock::now();
        uint64_t nodes = perft(pos, depth);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double seconds = duration.count() / 1000.0;
        double nps = seconds > 0 ? nodes / seconds : 0;

        std::cout << "Depth:   " << depth << "\n";
        std::cout << "Nodes:   " << nodes << "\n";
        std::cout << "Time:    " << std::fixed << std::setprecision(3) << seconds << " s\n";
        std::cout << "NPS:     " << std::fixed << std::setprecision(0) << nps << "\n";
    }

    return 0;
}
