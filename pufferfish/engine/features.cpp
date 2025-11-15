#include "features.h"
#include <algorithm>
#include <iterator>

namespace pf
{

    void extract_features(const Position &pos, std::vector<int> &features)
    {
        features.clear();
        const bool stmWhite = (pos.side_to_move == WHITE);
        for (int sq = 0; sq < 64; ++sq)
        {
            Piece pc = pos.board[sq];
            if (pc == NO_PIECE)
                continue;
            bool isWhite = (pc <= W_KING);
            int typeIdx;
            switch (pc)
            {
            case W_PAWN:
            case B_PAWN:
                typeIdx = 0;
                break;
            case W_KNIGHT:
            case B_KNIGHT:
                typeIdx = 1;
                break;
            case W_BISHOP:
            case B_BISHOP:
                typeIdx = 2;
                break;
            case W_ROOK:
            case B_ROOK:
                typeIdx = 3;
                break;
            case W_QUEEN:
            case B_QUEEN:
                typeIdx = 4;
                break;
            case W_KING:
            case B_KING:
                typeIdx = 5;
                break;
            default:
                continue;
            }
            bool isFriendly = (stmWhite ? isWhite : !isWhite);
            int colorOffset = isFriendly ? 0 : 6; // 0..5 friendly, 6..11 enemy
            int feat = (colorOffset + typeIdx) * 64 + sq;
            if (feat >= 0 && feat < FEATURE_DIM)
                features.push_back(feat);
        }
    }

    void diff_features(const Position &before, const Position &after,
                       std::vector<int> &added, std::vector<int> &removed)
    {
        added.clear();
        removed.clear();
        // Naive diff via extraction; can be optimized later.
        std::vector<int> fb, fa;
        extract_features(before, fb);
        extract_features(after, fa);
        std::sort(fb.begin(), fb.end());
        std::sort(fa.begin(), fa.end());
        std::set_difference(fa.begin(), fa.end(), fb.begin(), fb.end(), std::back_inserter(added));
        std::set_difference(fb.begin(), fb.end(), fa.begin(), fa.end(), std::back_inserter(removed));
    }

    void extract_features_795(const Position &pos, std::vector<float> &out)
    {
        out.assign(795, 0.0f);

        // 0..767: absolute PSQ one-hot (white pieces first, then black)
        auto psq_index = [](bool isWhite, int typeIdx, int sq) -> int
        {
            int channelBase = isWhite ? 0 : 6;
            return (channelBase + typeIdx) * 64 + sq;
        };
        for (int sq = 0; sq < 64; ++sq)
        {
            Piece pc = pos.board[sq];
            if (pc == NO_PIECE)
                continue;
            bool isWhite = (pc <= W_KING);
            int typeIdx = 0;
            switch (pc)
            {
            case W_PAWN:
            case B_PAWN:
                typeIdx = 0;
                break;
            case W_KNIGHT:
            case B_KNIGHT:
                typeIdx = 1;
                break;
            case W_BISHOP:
            case B_BISHOP:
                typeIdx = 2;
                break;
            case W_ROOK:
            case B_ROOK:
                typeIdx = 3;
                break;
            case W_QUEEN:
            case B_QUEEN:
                typeIdx = 4;
                break;
            case W_KING:
            case B_KING:
                typeIdx = 5;
                break;
            default:
                continue;
            }
            int idx = psq_index(isWhite, typeIdx, sq);
            out[idx] = 1.0f;
        }

        int cursor = 768;
        // 768: side-to-move (1 for White, 0 for Black)
        out[cursor++] = (pos.side_to_move == WHITE) ? 1.0f : 0.0f;

        // 769..772: castling rights K,Q,k,q
        int cr = pos.castling_rights;
        out[cursor++] = (cr & 0b0001) ? 1.0f : 0.0f; // white kingside
        out[cursor++] = (cr & 0b0010) ? 1.0f : 0.0f; // white queenside
        out[cursor++] = (cr & 0b0100) ? 1.0f : 0.0f; // black kingside
        out[cursor++] = (cr & 0b1000) ? 1.0f : 0.0f; // black queenside

        // 773..780: en-passant file one-hot (a..h)
        if (pos.ep_square >= 0)
        {
            int file = pos.ep_square & 7;
            if (file >= 0 && file < 8)
                out[cursor + file] = 1.0f;
        }
        cursor += 8;

        auto count_piece = [&](Piece p) -> int
        {
            // Count bits in piece bitboard
            Bitboard bb = pos.pieceBB[p];
#ifdef _MSC_VER
            return (int)__popcnt64(bb);
#else
            return __builtin_popcountll(bb);
#endif
        };
        int wp = count_piece(W_PAWN), bp = count_piece(B_PAWN);
        int wn = count_piece(W_KNIGHT), bn = count_piece(B_KNIGHT);
        int wb = count_piece(W_BISHOP), bbp = count_piece(B_BISHOP);
        int wr = count_piece(W_ROOK), br = count_piece(B_ROOK);
        int wq = count_piece(W_QUEEN), bq = count_piece(B_QUEEN);
        int wk = count_piece(W_KING), bk = count_piece(B_KING);

        // 781: material balance scaled by 2000.0
        float material = float((wp - bp) * 100 + (wn - bn) * 320 + (wb - bbp) * 330 +
                               (wr - br) * 500 + (wq - bq) * 900);
        out[cursor++] = material / 2000.0f;

        // 782..793: per-piece counts scaled by 8.0 (white P..K then black P..K)
        const float countScale = 8.0f;
        int countsW[6] = {wp, wn, wb, wr, wq, wk};
        int countsB[6] = {bp, bn, bbp, br, bq, bk};
        for (int i = 0; i < 6; ++i)
            out[cursor++] = countsW[i] / countScale;
        for (int i = 0; i < 6; ++i)
            out[cursor++] = countsB[i] / countScale;

        // 794: phase indicator = total pieces / 32.0
        int totalPieces = wp + wn + wb + wr + wq + wk + bp + bn + bbp + br + bq + bk;
        out[cursor++] = totalPieces / 32.0f;
    }

} // namespace pf
