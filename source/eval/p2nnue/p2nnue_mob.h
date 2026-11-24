#ifndef P2NNUE_MOB_H_INCLUDED
#define P2NNUE_MOB_H_INCLUDED

// ---------------------------------------------------------------------------
// B6 可動域 (mobility) — 反対称スカラー 1 本
// ---------------------------------------------------------------------------
// 2026-09-29。 手設計の特徴量 26 ブロックが全滅したあと、 screen の器を作り直して
// (594 局 → 33,096 局、 基準式に非線形 17 列) 裁き直したところ、 唯一生き残った量。
//
//   勝敗 ΔR² +0.038 (床 ±0.001 の 40 倍)、 較正 1 ΔR² = +19〜39pt → **+0.7〜1.5pt (5〜10 Elo)**
//   関門: 集合分割で再現 / 係数の符号安定 (mob_u 正・mob_t 負) / 手数帯で全帯正 /
//         **信号の 99% が反対称側** / 戦型 (net 既知) を足しても減らない
//
// 定義 (正典 = C:/project/p2shogi/runs/rescore/fs_feats.py:28-61, 158):
//
//   mob_c = Σ(色 c の全駒の疑似合法な移動先の数)
//         = Σ_sq board_effect[c].e[sq] ただし **色 c の駒が乗る升は数えない**
//
//   - step 駒も ray 駒も board_effect が正しく数えている (遮蔽込み)
//   - **敵駒の升は数える** (取れるので)、 **味方駒の升は数えない**
//   - 玉も含む (fs_feats の STEPS[8] = 玉、 C++ の calc_effect も kingEffect を含む)
//   - 🚨王手放置等の合法性は見ていない (疑似合法)。 正典も同じ
//
// B6 = mob(手番側) − mob(非手番側)   ← 反対称なので PSQT 頭で表現できる
//
// 🚨board_effect は do_move/undo_move で差分維持されている (position.cpp:1911 他) ので、
//   ここでは**読むだけ**。 再カウントも差分更新も要らない。
// ---------------------------------------------------------------------------

#include "../../bitboard.h"
#include "../../position.h"
#include "../../types.h"

#if !defined(LONG_EFFECT_LIBRARY)
#if defined(P2_MOB)
#error "B6 mobility は board_effect を使うので LONG_EFFECT_LIBRARY が必要"
#endif
#else  // LONG_EFFECT_LIBRARY: 本体 (2026-11-05 fable: LE 無しビルドでは丸ごと外す)

namespace YaneuraOu {
namespace Eval {
namespace P2NNUE {
namespace Mob {

// 参照実装 (スカラー)。 検証の正典として常に残す。
inline int mob_scalar(const Position& pos, Color c) {
	const auto& bd = pos.piece_array();
	const auto& eb = pos.board_effect[c];
	int s = 0;
	for (Square sq = SQ_ZERO; sq < SQ_NB; ++sq) {
		const Piece p = bd[sq];
		// 🚨色 c の駒が乗っている升は「移動先」にならないので数えない。
		//   NO_PIECE(0) に color_of を適用しないよう p の非ゼロを先に見る。
		if (p != NO_PIECE && color_of(p) == c)
			continue;
		s += (int) eb.e[sq];
	}
	return s;
}

// 速い版。 参照実装と**必ず同じ値**を返す (p2mob で 4,010 局面照合済み)。
//
//   mob(c) = Σ_全升 eb.e[sq] − Σ_自駒の升 eb.e[sq]
//
// 第 1 項は 81 バイトの総和 (SAD で一気に)、 第 2 項は自駒 bitboard を回すだけ
// (駒数は最大 ~20)。 参照実装の「81 升を舐めて分岐」より分岐が消える。
//
// 🚨e[] の 81 バイトより後ろ (padding) は読まない。 64..80 の 17 バイトは
//   スカラーで足す (padding の中身を仮定しないため)。
inline int mob_scalar_fast(const Position& pos, Color c) {
	const auto& eb = pos.board_effect[c];
	const std::uint8_t* p = reinterpret_cast<const std::uint8_t*>(&eb.e[0]);
	int s = 0;
#if defined(USE_AVX2)
	// sq 0..63 を 32 バイト x 2 で。 SAD は 8 バイトごとの和を 4 本の u64 に置く。
	const __m256i z = _mm256_setzero_si256();
	__m256i acc = _mm256_sad_epu8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)), z);
	acc = _mm256_add_epi64(acc,
	        _mm256_sad_epu8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32)), z));
	alignas(32) std::uint64_t t[4];
	_mm256_store_si256(reinterpret_cast<__m256i*>(t), acc);
	s = (int) (t[0] + t[1] + t[2] + t[3]);
	for (int i = 64; i < (int) SQ_NB; ++i)
		s += (int) p[i];
#else
	for (int i = 0; i < (int) SQ_NB; ++i)
		s += (int) p[i];
#endif
	// 自駒が乗っている升の分を引く (そこへは動けないので)。
	Bitboard bb = pos.pieces(c);
	while (bb) {
		const Square sq = bb.pop();
		s -= (int) eb.e[sq];
	}
	return s;
}

// 手番側 − 非手番側。 これが B6。
inline int mob_diff(const Position& pos) {
	const Color us = pos.side_to_move();
#if defined(P2_MOB_REF)
	// 🚨定義照合用。 p2mob は常にこちらと一致すること。
	return mob_scalar(pos, us) - mob_scalar(pos, ~us);
#else
	return mob_scalar_fast(pos, us) - mob_scalar_fast(pos, ~us);
#endif
}

}  // namespace Mob
}  // namespace P2NNUE
}  // namespace Eval
}  // namespace YaneuraOu
#endif  // LONG_EFFECT_LIBRARY

#endif  // P2NNUE_MOB_H_INCLUDED
