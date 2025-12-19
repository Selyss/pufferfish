// Root score dump tool
// Prints a score for every legal root move from a given position at a given depth.
//
// Usage:
//   root_score_dump <depth> ["<fen>"]
// Examples:
//   root_score_dump 4
//   root_score_dump 4 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

#include "../src/position.h"
#include "../src/movegen.h"
#include "../src/search.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

using namespace pufferfish;

static void usage()
{
    std::cout << "Usage: root_score_dump <depth> [\"<fen>\"]\n";
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        usage();
        return 2;
    }

    int depth = 0;
    try
    {
        depth = std::stoi(argv[1]);
    }
    catch (...)
    {
        usage();
        return 2;
    }

    std::string fen = STARTPOS_FEN;
    if (argc >= 3)
        fen = argv[2];

    Position pos;
    if (!pos.set_fen(fen))
    {
        std::cerr << "Error: set_fen() failed\n";
        std::cerr << "FEN: " << fen << "\n";
        return 1;
    }

    Search search;
    std::cout << "NNUE: " << (search.nnue_ready() ? "ready" : "not ready") << "\n";
    std::cout << "Depth: " << depth << "\n";
    std::cout << "FEN: " << pos.fen() << "\n\n";

    std::vector<Move> moves;
    generate_legal_moves(pos, moves);

    if (moves.empty())
    {
        std::cout << "No legal moves.\n";
        return 0;
    }

    struct ScoredMove
    {
        Move move;
        int score;
    };
    std::vector<ScoredMove> scored;
    scored.reserve(moves.size());

    // Root scoring: same as Search::find_best_move's root loop, but record every move's score.
    for (const Move &m : moves)
    {
        Undo undo;
        pos.make_move(m, undo);
        int score = -search.alpha_beta(pos, depth - 1, -INF, INF);
        pos.unmake_move(m, undo);
        scored.push_back({m, score});
    }

    // Print in generation order and also show the best by score.
    int best_score = scored[0].score;
    Move best_move = scored[0].move;
    for (const auto &sm : scored)
    {
        if (sm.score > best_score)
        {
            best_score = sm.score;
            best_move = sm.move;
        }
    }

    std::cout << "Generated moves: " << moves.size() << "\n";
    std::cout << "Best by score: " << best_move.to_uci() << " (" << best_score << ")\n\n";

    std::cout << std::left << std::setw(8) << "move" << "score\n";
    std::cout << std::string(18, '-') << "\n";
    for (const auto &sm : scored)
    {
        std::cout << std::left << std::setw(8) << sm.move.to_uci() << sm.score << "\n";
    }

    return 0;
}

