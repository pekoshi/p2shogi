#ifndef P2VOLM_H_INCLUDED
#define P2VOLM_H_INCLUDED

#include "position.h"
#include "types.h"

namespace YaneuraOu {

// 2026-09-28 NALLOC: 局面の「揺れ」V = 両玉の 8 近傍のうち**駒が乗っていない升**への、
//   その玉の**敵**の利き数の和。 `board_effect` は do_move で差分維持されている (LONG_EFFECT_LIBRARY)。
//   測定 (runs/rescore/empty_split.py:47-66) と同じ定義。 ΔR² +3.475 で |静的評価 - 20万ノード探索| を当てる。
//   コスト = 8 byte load x 2 玉 + 加算 ≈ 40-60 cycle (1 node 3.0-3.8 us の 0.2% 未満)。
namespace P2Volm {
// 2026-11-05 (fable): LONG_EFFECT_LIBRARY が無いビルドでは attackers_to の popcount で同じ値を出す。
//   board_effect[c].e[sq] は「色 c の駒のうち sq に利いている枚数」(各駒 1 回・遮蔽込み) で、
//   attackers_to(c, sq, occ) の popcount と定義が同一。 razoring の門でしか呼ばれないので
//   (RAZV_VMIN=3 のとき depth<=razor 帯の一部ノード)、 都度計算で十分安い。
inline int value(const Position& pos) {
    int v = 0;
    for (Color kc : {BLACK, WHITE})
    {
        const Square ksq = pos.square<KING>(kc);
        const int    kf = (int) file_of(ksq), kr = (int) rank_of(ksq);
        const Color  en = ~kc;
        for (int df = -1; df <= 1; ++df)
            for (int dr = -1; dr <= 1; ++dr)
            {
                if (df == 0 && dr == 0) continue;
                const int f = kf + df, r = kr + dr;
                if (f < 0 || f >= 9 || r < 0 || r >= 9) continue;
                const Square sq = (File) f | (Rank) r;
                if (pos.piece_on(sq) != NO_PIECE) continue;   // 空升のみ
#if defined(LONG_EFFECT_LIBRARY)
                v += (int) pos.board_effect[en].effect(sq);
#else
                v += pos.attackers_to(en, sq).pop_count();
#endif
            }
    }
    return v;
}
// bin: 0 / 1-2 / 3-4 / 5-6 / >=7  (実分布 約 0.38 / 0.32 / 0.20 / 0.07 / 0.03)
inline int bin(int v) { return v <= 0 ? 0 : v <= 2 ? 1 : v <= 4 ? 2 : v <= 6 ? 3 : 4; }
}  // namespace P2Volm

}  // namespace YaneuraOu

#endif  // P2VOLM_H_INCLUDED
