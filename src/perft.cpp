/*
 * Pufferfish Chess Engine
 * Perft implementation
 */

#include "perft.h"
#include "movegen.h"
#include <vector>
#include <iomanip>
#include <chrono>

namespace pufferfish
{

    uint64_t perft(Position &pos, int depth)
    {
        if (depth == 0)
            return 1;

        std::vector<Move> moves;
        generate_legal_moves(pos, moves);

        if (depth == 1)
            return moves.size();

        uint64_t nodes = 0;
        Undo undo;

        for (const Move &m : moves)
        {
            pos.make_move(m, undo);
            nodes += perft(pos, depth - 1);
            pos.unmake_move(m, undo);
        }

        return nodes;
    }

    void perft_divide(Position &pos, int depth, std::ostream &os)
    {
        std::vector<Move> moves;
        generate_legal_moves(pos, moves);

        uint64_t total = 0;
        Undo undo;

        os << "\nPerft divide at depth " << depth << ":\n";
        os << std::string(30, '-') << "\n";

        for (const Move &m : moves)
        {
            pos.make_move(m, undo);
            uint64_t nodes = (depth > 1) ? perft(pos, depth - 1) : 1;
            pos.unmake_move(m, undo);

            os << std::left << std::setw(6) << m.to_uci() << ": " << nodes << "\n";
            total += nodes;
        }

        os << std::string(30, '-') << "\n";
        os << "Total: " << total << " nodes (" << moves.size() << " moves)\n";
    }

    // =============================================================================
    // Perft test suite
    // =============================================================================

    struct PerftTest
    {
        const char *name;
        const char *fen;
        int depth;
        uint64_t expected_nodes;
    };

    // Standard perft test positions
    static const PerftTest PERFT_TESTS[] = {
        // Starting position
        {"startpos d1", STARTPOS_FEN, 1, 20},
        {"startpos d2", STARTPOS_FEN, 2, 400},
        {"startpos d3", STARTPOS_FEN, 3, 8902},
        {"startpos d4", STARTPOS_FEN, 4, 197281},
        {"startpos d5", STARTPOS_FEN, 5, 4865609},

        // Kiwipete - complex middlegame position
        {"kiwipete d1", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48},
        {"kiwipete d2", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039},
        {"kiwipete d3", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862},
        {"kiwipete d4", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},

        // Position with en passant
        {"ep test d1", "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 1", 1, 31},
        {"ep test d2", "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 1", 2, 570},

        // Position with promotions
        {"promo test d1", "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", 1, 24},
        {"promo test d2", "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", 2, 496},
        {"promo test d3", "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", 3, 9483},

        // Castling test
        {"castle test d1", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 1, 26},
        {"castle test d2", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 2, 568},
        {"castle test d3", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 3, 13744},

        // Check evasion
        {"check test d1", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", 1, 20},
        {"check test d2", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", 2, 400},
    };

    bool run_perft_suite(std::ostream &os)
    {
        os << "\n=== Pufferfish Perft Test Suite ===\n\n";

        int passed = 0;
        int failed = 0;

        for (const auto &test : PERFT_TESTS)
        {
            Position pos;
            if (!pos.set_fen(test.fen))
            {
                os << "[ERROR] Failed to parse FEN: " << test.fen << "\n";
                ++failed;
                continue;
            }

            auto start = std::chrono::high_resolution_clock::now();
            uint64_t nodes = perft(pos, test.depth);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            double nps = duration.count() > 0 ? (nodes * 1000.0 / duration.count()) : 0;

            bool pass = (nodes == test.expected_nodes);

            os << (pass ? "[PASS] " : "[FAIL] ")
               << std::left << std::setw(16) << test.name
               << " depth " << test.depth
               << " nodes " << std::setw(10) << nodes;

            if (!pass)
            {
                os << " (expected " << test.expected_nodes << ")";
            }
            else
            {
                os << " " << std::fixed << std::setprecision(0) << nps << " nps";
            }

            os << "\n";

            if (pass)
                ++passed;
            else
                ++failed;
        }

        os << "\n"
           << std::string(50, '=') << "\n";
        os << "Results: " << passed << " passed, " << failed << " failed\n";

        return failed == 0;
    }

} // namespace pufferfish
