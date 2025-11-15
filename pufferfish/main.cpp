// Minimal runner: initialize engine, run a short search from startpos.

#include <iostream>
#include <sstream>
#include <string>

#include "engine/types.h"
#include "engine/bitboard.h"
#include "engine/position.h"
#include "engine/movegen.h"
#include "engine/tt.h"
#include "engine/nn_interface.h"
#include "engine/search.h"

using namespace pf;

struct MaterialEvaluator : NNEvaluator
{
    int evaluate(const Position &pos) override
    {
        static const int val[PIECE_NB] = {
            0,
            100, 320, 330, 500, 900, 0,
            -100, -320, -330, -500, -900, 0};
        int score = 0;
        for (int sq = 0; sq < 64; ++sq)
            score += val[pos.board[sq]];
        return (pos.side_to_move == WHITE ? score : -score);
    }
};

static std::string sq_to_str(int sq)
{
    const char files[] = "abcdefgh";
    std::string s;
    s += files[sq & 7];
    s += char('1' + (sq >> 3));
    return s;
}

static char promo_char_from_piece(int promoPiece)
{
    int typeIdx = 0;
    if (promoPiece >= W_PAWN && promoPiece <= W_KING)
        typeIdx = promoPiece - W_PAWN;
    else if (promoPiece >= B_PAWN && promoPiece <= B_KING)
        typeIdx = promoPiece - B_PAWN;
    else
        return '\0';
    switch (typeIdx)
    {
    case KNIGHT:
        return 'n';
    case BISHOP:
        return 'b';
    case ROOK:
        return 'r';
    case QUEEN:
        return 'q';
    default:
        return '\0';
    }
}

static std::string move_to_uci(Move m)
{
    int from = from_sq(m);
    int to = to_sq(m);
    std::string uci = sq_to_str(from) + sq_to_str(to);
    if (move_flags(m) & FLAG_PROMOTION)
    {
        char pc = promo_char_from_piece(promo_piece(m));
        if (pc)
            uci += pc;
    }
    return uci;
}

int main(int argc, char **argv)
{
    init_zobrist();
    init_bitboards();

    Position pos;
    pos.set_startpos();

    // Parse CLI args: --fen <6 tokens>, --depth N, --movetime ms
    std::string fen;
    int depth = 5;
    int movetime = 0;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--fen" && i + 6 <= argc)
        {
            std::ostringstream os;
            os << argv[i + 1] << ' ' << argv[i + 2] << ' ' << argv[i + 3]
               << ' ' << argv[i + 4] << ' ' << argv[i + 5] << ' ' << argv[i + 6];
            fen = os.str();
            i += 6;
        }
        else if (a == "--depth" && i + 1 < argc)
        {
            depth = std::max(1, std::atoi(argv[++i]));
        }
        else if (a == "--movetime" && i + 1 < argc)
        {
            movetime = std::max(0, std::atoi(argv[++i]));
        }
    }
    if (!fen.empty())
    {
        pos.set_fen(fen);
    }

    TranspositionTable tt;
    tt.resize(64); // 64 MB

    NNUEEvaluator nn;
    bool has_nn = nn.load("nnue_weights.bin");
    MaterialEvaluator mat;

    SearchContext ctx;
    ctx.tt = &tt;
    ctx.nn = has_nn ? static_cast<NNEvaluator *>(&nn) : static_cast<NNEvaluator *>(&mat);
    if (movetime > 0)
    {
        ctx.limits.time_ms = static_cast<std::uint64_t>(movetime);
        ctx.limits.depth = 0;
    }
    else
    {
        ctx.limits.depth = depth;
        ctx.limits.time_ms = 0;
    }

    SearchResult res = search(pos, ctx);

    std::cout << "bestmove " << move_to_uci(res.bestMove) << "\n";

    return 0;
}
