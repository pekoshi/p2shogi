#ifndef P2NNUE_THREAT_DIRECT15_H_INCLUDED
#define P2NNUE_THREAT_DIRECT15_H_INCLUDED

#if defined(EVAL_P2NNUE)

#include <cstdint>

namespace YaneuraOu::Eval::P2NNUE::Direct15 {

// Profile 15 is intentionally independent from the legacy Threat LUT.
// It adds King as a true tenth class without perturbing profiles 0/3/11-14.
constexpr int PROFILE_ID = 15;
constexpr int CLASS_NB = 10;
constexpr int KING_CLASS = 9;
constexpr int NUM_PATTERNS = 15; // 10 Black/base patterns + 5 White directional patterns.
constexpr int DIM = 65640;
constexpr int NUM_BUCKETS_REQUIRED = 8;

constexpr int ATTACKS_PER_COLOR[CLASS_NB] = {
    72, 324, 112, 328, 416, 816, 1296, 1104, 1552, 544,
};

constexpr int sum_attacks_per_color() {
    int sum = 0;
    for (int i = 0; i < CLASS_NB; ++i) sum += ATTACKS_PER_COLOR[i];
    return sum;
}
static_assert(sum_attacks_per_color() == 6564, "Profile 15 attack count");
static_assert(DIM == CLASS_NB * sum_attacks_per_color(), "Profile 15 direct dimension");

// PieceType -> Direct15 クラスの LUT。 旧実装は switch だったが、 評価 1 回あたり
// 盤上全駒分 (~40 回) 引かれるホットパスなので分岐なしの 16 バイト表にしてある。
// 値は旧 switch と完全一致 (NO_PIECE_TYPE / QUEEN は -1)。
constexpr std::int8_t CLASS_OF_PT[16] = {
    -1,          // NO_PIECE_TYPE
    0,           // PAWN
    1,           // LANCE
    2,           // KNIGHT
    3,           // SILVER
    5,           // BISHOP
    6,           // ROOK
    4,           // GOLD
    KING_CLASS,  // KING
    4,           // PRO_PAWN
    4,           // PRO_LANCE
    4,           // PRO_KNIGHT
    4,           // PRO_SILVER
    7,           // HORSE
    8,           // DRAGON
    -1,          // QUEEN (未使用)
};

// 飛び駒クラス (1=香, 5=角, 6=飛, 7=馬, 8=龍) のビットマスク。
// これ以外のクラスは空盤面利き == 実利きなので occ 依存の利き計算を省略できる。
constexpr unsigned SLIDER_CLASS_MASK =
    (1u << 1) | (1u << 5) | (1u << 6) | (1u << 7) | (1u << 8);

inline int class_from_piece_type(PieceType pt) {
    return CLASS_OF_PT[(int)pt & 15];
}

inline bool is_directional(int tc) { return tc <= 4; }
inline int attack_pattern_id(int tc, Color oriented) {
    return oriented == WHITE && is_directional(tc) ? CLASS_NB + tc : tc;
}

inline int attacks_empty_board(int tc, Color color, int from, std::uint8_t* targets) {
    int count = 0;
    const int file = from / 9;
    const int rank = from % 9;
    auto push = [&](int f, int r) {
        if (f >= 0 && f < 9 && r >= 0 && r < 9)
            targets[count++] = static_cast<std::uint8_t>(f * 9 + r);
    };
    const int fwd = color == BLACK ? -1 : 1;

    switch (tc) {
    case 0:
        push(file, rank + fwd);
        break;
    case 1:
        if (color == BLACK) for (int r = rank - 1; r >= 0; --r) push(file, r);
        else                for (int r = rank + 1; r < 9; ++r) push(file, r);
        break;
    case 2:
        push(file - 1, rank + 2 * fwd);
        push(file + 1, rank + 2 * fwd);
        break;
    case 3:
        push(file - 1, rank + fwd); push(file, rank + fwd); push(file + 1, rank + fwd);
        push(file - 1, rank - fwd); push(file + 1, rank - fwd);
        break;
    case 4:
        push(file - 1, rank + fwd); push(file, rank + fwd); push(file + 1, rank + fwd);
        push(file - 1, rank);       push(file + 1, rank);
        push(file, rank - fwd);
        break;
    case 5:
        for (int d = 0; d < 4; ++d) {
            const int df = d < 2 ? -1 : 1;
            const int dr = (d & 1) ? 1 : -1;
            for (int f = file + df, r = rank + dr; f >= 0 && f < 9 && r >= 0 && r < 9; f += df, r += dr)
                targets[count++] = static_cast<std::uint8_t>(f * 9 + r);
        }
        break;
    case 6:
        for (int d = 0; d < 4; ++d) {
            const int df = d == 0 ? -1 : d == 1 ? 1 : 0;
            const int dr = d == 2 ? -1 : d == 3 ? 1 : 0;
            for (int f = file + df, r = rank + dr; f >= 0 && f < 9 && r >= 0 && r < 9; f += df, r += dr)
                targets[count++] = static_cast<std::uint8_t>(f * 9 + r);
        }
        break;
    case 7:
        for (int d = 0; d < 4; ++d) {
            const int df = d < 2 ? -1 : 1;
            const int dr = (d & 1) ? 1 : -1;
            for (int f = file + df, r = rank + dr; f >= 0 && f < 9 && r >= 0 && r < 9; f += df, r += dr)
                targets[count++] = static_cast<std::uint8_t>(f * 9 + r);
        }
        push(file - 1, rank); push(file + 1, rank); push(file, rank - 1); push(file, rank + 1);
        break;
    case 8:
        for (int d = 0; d < 4; ++d) {
            const int df = d == 0 ? -1 : d == 1 ? 1 : 0;
            const int dr = d == 2 ? -1 : d == 3 ? 1 : 0;
            for (int f = file + df, r = rank + dr; f >= 0 && f < 9 && r >= 0 && r < 9; f += df, r += dr)
                targets[count++] = static_cast<std::uint8_t>(f * 9 + r);
        }
        push(file - 1, rank - 1); push(file - 1, rank + 1);
        push(file + 1, rank - 1); push(file + 1, rank + 1);
        break;
    case KING_CLASS:
        for (int df = -1; df <= 1; ++df)
            for (int dr = -1; dr <= 1; ++dr)
                if (df != 0 || dr != 0) push(file + df, rank + dr);
        break;
    default:
        break;
    }

    for (int i = 1; i < count; ++i) {
        const std::uint8_t key = targets[i];
        int j = i;
        while (j > 0 && targets[j - 1] > key) {
            targets[j] = targets[j - 1];
            --j;
        }
        targets[j] = key;
    }
    return count;
}

// 推論用の重み並べ替え (dc-minor) と、 その際に必要な末尾パディング。
//   ファイル上の索引 : pair_base[ac*10+dc] + from_offset + attack_order
//                    = class_base[ac] + dc * ATTACKS_PER_COLOR[ac] + r   (r = fo + ord)
//   推論用の索引     : class_base[ac] + r * CLASS_NB + dc
// 同一 attacker の複数 target がメモリ上で近接するため、 キャッシュライン数が減る。
// 値の集合は同一なので評価値は bit-exact。
// ord は攻撃対象外マスに対し 255 を返しうる (実際には到達しないが)、 その場合の
// 最大はみ出しは 255 * CLASS_NB = 2550 要素なので、 重み配列末尾に余白を確保して
// ホットループから範囲判定分岐を除去する。
constexpr int LOAD_PAD = 4096;

struct Tables {
    int pair_base[CLASS_NB * CLASS_NB];
    // class_base[ac] = pair_base[ac * CLASS_NB] (dc-minor 索引の基点)。
    int class_base[CLASS_NB];
    int from_offset[NUM_PATTERNS][81];
    std::uint8_t attack_order[NUM_PATTERNS][81][81];
    int dim = 0;
    bool built = false;

    void build_pattern(int tc, Color color, int pattern, std::uint8_t* targets) {
        int cumulative = 0;
        for (int from = 0; from < 81; ++from) {
            from_offset[pattern][from] = cumulative;
            const int count = attacks_empty_board(tc, color, from, targets);
            for (int order = 0; order < count; ++order)
                attack_order[pattern][from][targets[order]] = static_cast<std::uint8_t>(order);
            cumulative += count;
        }
    }

    void build() {
        if (built) return;
        int cumulative = 0;
        for (int ac = 0; ac < CLASS_NB; ++ac) {
            class_base[ac] = cumulative;
            for (int dc = 0; dc < CLASS_NB; ++dc) {
                pair_base[ac * CLASS_NB + dc] = cumulative;
                cumulative += ATTACKS_PER_COLOR[ac];
            }
        }
        dim = cumulative;

        for (int pattern = 0; pattern < NUM_PATTERNS; ++pattern)
            for (int from = 0; from < 81; ++from) {
                from_offset[pattern][from] = 0;
                for (int to = 0; to < 81; ++to) attack_order[pattern][from][to] = 255;
            }

        std::uint8_t targets[36];
        for (int tc = 0; tc < CLASS_NB; ++tc) {
            build_pattern(tc, BLACK, tc, targets);
            if (is_directional(tc)) build_pattern(tc, WHITE, CLASS_NB + tc, targets);
        }
        built = true;
    }

    inline int index(int ac, Color oriented, int dc, int from_normalized, int to_normalized) const {
        const int pattern = attack_pattern_id(ac, oriented);
        return pair_base[ac * CLASS_NB + dc]
             + from_offset[pattern][from_normalized]
             + static_cast<int>(attack_order[pattern][from_normalized][to_normalized]);
    }
};

} // namespace YaneuraOu::Eval::P2NNUE::Direct15

#endif // EVAL_P2NNUE
#endif // P2NNUE_THREAT_DIRECT15_H_INCLUDED
