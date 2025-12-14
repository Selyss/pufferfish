/*
 * Pufferfish Chess Engine
 * Core type definitions
 */

#ifndef PUFFERFISH_TYPES_H
#define PUFFERFISH_TYPES_H

#include <cstdint>
#include <string>
#include <cassert>

namespace pufferfish
{

    // =============================================================================
    // Square representation
    // =============================================================================
    // Square indexing: 0 = a1, 7 = h1, 56 = a8, 63 = h8
    // Layout:
    //   56 57 58 59 60 61 62 63  (rank 8)
    //   48 49 50 51 52 53 54 55  (rank 7)
    //   ...
    //    8  9 10 11 12 13 14 15  (rank 2)
    //    0  1  2  3  4  5  6  7  (rank 1)
    //    a  b  c  d  e  f  g  h

    using Square = int;

    constexpr Square SQ_NONE = -1;

    constexpr Square SQ_A1 = 0, SQ_B1 = 1, SQ_C1 = 2, SQ_D1 = 3;
    constexpr Square SQ_E1 = 4, SQ_F1 = 5, SQ_G1 = 6, SQ_H1 = 7;
    constexpr Square SQ_A2 = 8, SQ_B2 = 9, SQ_C2 = 10, SQ_D2 = 11;
    constexpr Square SQ_E2 = 12, SQ_F2 = 13, SQ_G2 = 14, SQ_H2 = 15;
    constexpr Square SQ_A3 = 16, SQ_B3 = 17, SQ_C3 = 18, SQ_D3 = 19;
    constexpr Square SQ_E3 = 20, SQ_F3 = 21, SQ_G3 = 22, SQ_H3 = 23;
    constexpr Square SQ_A4 = 24, SQ_B4 = 25, SQ_C4 = 26, SQ_D4 = 27;
    constexpr Square SQ_E4 = 28, SQ_F4 = 29, SQ_G4 = 30, SQ_H4 = 31;
    constexpr Square SQ_A5 = 32, SQ_B5 = 33, SQ_C5 = 34, SQ_D5 = 35;
    constexpr Square SQ_E5 = 36, SQ_F5 = 37, SQ_G5 = 38, SQ_H5 = 39;
    constexpr Square SQ_A6 = 40, SQ_B6 = 41, SQ_C6 = 42, SQ_D6 = 43;
    constexpr Square SQ_E6 = 44, SQ_F6 = 45, SQ_G6 = 46, SQ_H6 = 47;
    constexpr Square SQ_A7 = 48, SQ_B7 = 49, SQ_C7 = 50, SQ_D7 = 51;
    constexpr Square SQ_E7 = 52, SQ_F7 = 53, SQ_G7 = 54, SQ_H7 = 55;
    constexpr Square SQ_A8 = 56, SQ_B8 = 57, SQ_C8 = 58, SQ_D8 = 59;
    constexpr Square SQ_E8 = 60, SQ_F8 = 61, SQ_G8 = 62, SQ_H8 = 63;

    constexpr int NUM_SQUARES = 64;

    // Square helper functions
    constexpr int file_of(Square sq) { return sq & 7; }
    constexpr int rank_of(Square sq) { return sq >> 3; }
    constexpr Square make_square(int file, int rank) { return (rank << 3) + file; }
    constexpr bool is_valid_square(Square sq) { return sq >= 0 && sq < 64; }

    inline std::string square_to_string(Square sq)
    {
        if (sq == SQ_NONE)
            return "-";
        assert(is_valid_square(sq));
        return std::string(1, 'a' + file_of(sq)) + std::string(1, '1' + rank_of(sq));
    }

    inline Square string_to_square(const std::string &s)
    {
        if (s == "-" || s.length() < 2)
            return SQ_NONE;
        int file = s[0] - 'a';
        int rank = s[1] - '1';
        if (file < 0 || file > 7 || rank < 0 || rank > 7)
            return SQ_NONE;
        return make_square(file, rank);
    }

    // File and rank constants
    constexpr int FILE_A = 0, FILE_B = 1, FILE_C = 2, FILE_D = 3;
    constexpr int FILE_E = 4, FILE_F = 5, FILE_G = 6, FILE_H = 7;
    constexpr int RANK_1 = 0, RANK_2 = 1, RANK_3 = 2, RANK_4 = 3;
    constexpr int RANK_5 = 4, RANK_6 = 5, RANK_7 = 6, RANK_8 = 7;

    // =============================================================================
    // Color representation
    // =============================================================================

    enum Color : int
    {
        WHITE = 0,
        BLACK = 1,
        COLOR_NONE = 2
    };

    constexpr int NUM_COLORS = 2;

    constexpr Color operator~(Color c) { return Color(c ^ 1); }

    inline std::string color_to_string(Color c)
    {
        return c == WHITE ? "white" : (c == BLACK ? "black" : "none");
    }

    // =============================================================================
    // Piece representation
    // =============================================================================
    // Piece encoding: 4 bits total
    //   Bit 3: color (0 = white, 1 = black)
    //   Bits 0-2: piece type

    enum PieceType : int
    {
        NO_PIECE_TYPE = 0,
        PAWN = 1,
        KNIGHT = 2,
        BISHOP = 3,
        ROOK = 4,
        QUEEN = 5,
        KING = 6,
        NUM_PIECE_TYPES = 7
    };

    enum Piece : int
    {
        NO_PIECE = 0,
        W_PAWN = 1,
        W_KNIGHT = 2,
        W_BISHOP = 3,
        W_ROOK = 4,
        W_QUEEN = 5,
        W_KING = 6,
        B_PAWN = 9,
        B_KNIGHT = 10,
        B_BISHOP = 11,
        B_ROOK = 12,
        B_QUEEN = 13,
        B_KING = 14,
        NUM_PIECES = 15
    };

    constexpr PieceType type_of(Piece p) { return PieceType(p & 7); }
    constexpr Color color_of(Piece p) { return Color((p >> 3) & 1); }
    constexpr Piece make_piece(Color c, PieceType pt) { return Piece((c << 3) | pt); }

    constexpr bool is_ok(Piece p) { return p != NO_PIECE; }

    inline char piece_to_char(Piece p)
    {
        constexpr char chars[] = ".PNBRQK..pnbrqk.";
        return chars[p];
    }

    inline Piece char_to_piece(char c)
    {
        switch (c)
        {
        case 'P':
            return W_PAWN;
        case 'N':
            return W_KNIGHT;
        case 'B':
            return W_BISHOP;
        case 'R':
            return W_ROOK;
        case 'Q':
            return W_QUEEN;
        case 'K':
            return W_KING;
        case 'p':
            return B_PAWN;
        case 'n':
            return B_KNIGHT;
        case 'b':
            return B_BISHOP;
        case 'r':
            return B_ROOK;
        case 'q':
            return B_QUEEN;
        case 'k':
            return B_KING;
        default:
            return NO_PIECE;
        }
    }

    inline char piece_type_to_char(PieceType pt)
    {
        constexpr char chars[] = ".pnbrqk";
        return chars[pt];
    }

    inline PieceType char_to_piece_type(char c)
    {
        switch (c)
        {
        case 'n':
        case 'N':
            return KNIGHT;
        case 'b':
        case 'B':
            return BISHOP;
        case 'r':
        case 'R':
            return ROOK;
        case 'q':
        case 'Q':
            return QUEEN;
        case 'k':
        case 'K':
            return KING;
        default:
            return NO_PIECE_TYPE;
        }
    }

    // =============================================================================
    // Castling rights
    // =============================================================================
    // Bitmask: 1 = white kingside, 2 = white queenside, 4 = black kingside, 8 = black queenside

    enum CastlingRights : int
    {
        NO_CASTLING = 0,
        WHITE_OO = 1,  // White kingside
        WHITE_OOO = 2, // White queenside
        BLACK_OO = 4,  // Black kingside
        BLACK_OOO = 8, // Black queenside

        WHITE_CASTLING = WHITE_OO | WHITE_OOO,
        BLACK_CASTLING = BLACK_OO | BLACK_OOO,
        ANY_CASTLING = WHITE_CASTLING | BLACK_CASTLING
    };

    constexpr CastlingRights operator|(CastlingRights a, CastlingRights b)
    {
        return CastlingRights(int(a) | int(b));
    }
    constexpr CastlingRights operator&(CastlingRights a, CastlingRights b)
    {
        return CastlingRights(int(a) & int(b));
    }
    constexpr CastlingRights operator~(CastlingRights c)
    {
        return CastlingRights(~int(c) & ANY_CASTLING);
    }
    inline CastlingRights &operator|=(CastlingRights &a, CastlingRights b)
    {
        return a = a | b;
    }
    inline CastlingRights &operator&=(CastlingRights &a, CastlingRights b)
    {
        return a = a & b;
    }

    inline std::string castling_to_string(CastlingRights cr)
    {
        if (cr == NO_CASTLING)
            return "-";
        std::string s;
        if (cr & WHITE_OO)
            s += 'K';
        if (cr & WHITE_OOO)
            s += 'Q';
        if (cr & BLACK_OO)
            s += 'k';
        if (cr & BLACK_OOO)
            s += 'q';
        return s;
    }

    // Castling rights lookup by square (for updating rights when pieces move)
    // Index by square, returns the rights to REMOVE when that square is involved
    constexpr CastlingRights CASTLING_RIGHTS_MASK[64] = {
        WHITE_OOO, NO_CASTLING, NO_CASTLING, NO_CASTLING, CastlingRights(WHITE_OO | WHITE_OOO), NO_CASTLING, NO_CASTLING, WHITE_OO,
        NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING,
        NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING,
        NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING,
        NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING,
        NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING,
        NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING, NO_CASTLING,
        BLACK_OOO, NO_CASTLING, NO_CASTLING, NO_CASTLING, CastlingRights(BLACK_OO | BLACK_OOO), NO_CASTLING, NO_CASTLING, BLACK_OO};

    // =============================================================================
    // Direction helpers
    // =============================================================================

    constexpr int NORTH = 8;
    constexpr int SOUTH = -8;
    constexpr int EAST = 1;
    constexpr int WEST = -1;
    constexpr int NORTH_EAST = 9;
    constexpr int NORTH_WEST = 7;
    constexpr int SOUTH_EAST = -7;
    constexpr int SOUTH_WEST = -9;

    // Pawn push direction by color
    constexpr int pawn_push(Color c) { return c == WHITE ? NORTH : SOUTH; }

    // Starting rank for pawns
    constexpr int relative_rank(Color c, int rank) { return c == WHITE ? rank : 7 - rank; }
    constexpr int pawn_start_rank(Color c) { return c == WHITE ? RANK_2 : RANK_7; }
    constexpr int pawn_promo_rank(Color c) { return c == WHITE ? RANK_8 : RANK_1; }

} // namespace pufferfish

#endif // PUFFERFISH_TYPES_H
