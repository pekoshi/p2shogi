#ifndef P2NNUE_AUX_H_INCLUDED
#define P2NNUE_AUX_H_INCLUDED

// p2shogi NNUE: AUX 入力(KZ + HANG + STATKD, 600 次元/persp)の発火 index 列挙。
// 移植元(source of truth): bullet-shogi-nodchip-v2
//   crates/bullet_lib/src/game/inputs/shogi_halfka_hm_aux.rs
//   公開関数 emit_aux_active_index(board, persp, f) / example shogi_aux_canonical。
//
// ★bit-exact 契約:
//   - Square: YaneuraOu Square(int 0..80) == bullet raw(file*9+rank, file 0=1筋 / rank 0=1段)。
//   - AttackMaps: atk[c][sq]=色 c の利き総数(slider は occupancy-aware で遮蔽駒マス自身も含む、
//     王は占有無関係の 8 近傍 Cheby1)。atkmin[c][sq]=攻撃駒の最小 VAL(なし=99)。
//     → YaneuraOu の effects_from(pc, sq, occ) が全駒(玉含む)でこの規約と一致する。
//   - aux には square index 無し・HM mirror/sq 正規化なし(純カウントの thermometer)。
//   - 発火は非対称: STM acc は own=STM、NSTM acc は own=NSTM で emit する。
//   検証: USI "p2aux [sfen]" と example shogi_aux_canonical の出力を照合。

#include "../../config.h"

#if defined(EVAL_P2NNUE)

#include <cstdint>
#include <cstring>
#include "../../types.h"
#include "../../position.h"
#include "../../bitboard.h"
#include "p2nnue_threat.h"

// P2_AUX_LE(利き付き aux)は board_effect 差分利きテーブルを読むため LONG_EFFECT 必須。
#if defined(P2_AUX_LE) && !defined(LONG_EFFECT_LIBRARY)
#error "P2_AUX_LE requires -DLONG_EFFECT_LIBRARY (board_effect diff-effect table)"
#endif

namespace YaneuraOu::Eval::P2NNUE::Aux {

// ---- 次元定数(契約 shogi_halfka_hm_aux.rs と一致)----
constexpr int AUX_NUM_CLASSES = 9;          // ThreatClass 数(King 除外)
constexpr int KZ_DIM     = 60;              // (own玉,敵玉) × (A8 8 + D8 7 + A24 8 + D24 7)
constexpr int HANG_DIM   = 108;             // (us,them) × (hang 9×3 + cheap 9×3)
constexpr int STATKD_DIM = 432;             // (us,them) × 9class × 4status × 3kdist × 2thermo
constexpr int AUX_DIM_PER_PERSP = KZ_DIM + HANG_DIM + STATKD_DIM; // 600
constexpr int KZ_OFFSET     = 0;
constexpr int HANG_OFFSET   = KZ_DIM;                 // 60
constexpr int STATKD_OFFSET = KZ_DIM + HANG_DIM;      // 168

// ---- 量子化 / section 定数 ----
constexpr int  AUX_QA = 127;                          // FT 入力なので QA=127
static const char AUX_ARCH_MARKER[] = "AUX=600,";     // loader 検出子
static const char AUX_SUB_MAGIC[8]  = {'P','2','A','U','X',0,0,0};
constexpr std::uint32_t AUX_SECTION_VERSION = 1;

// ---- bucket edges(契約と一致)----
constexpr int A8_E[8]  = {1, 2, 3, 4, 5, 6, 8, 11};
constexpr int D8_E[7]  = {1, 3, 5, 7, 9, 11, 14};
constexpr int A24_E[8] = {1, 2, 4, 6, 8, 11, 15, 20};
constexpr int D24_E[7] = {2, 5, 8, 11, 14, 18, 23};
constexpr int CNT3_E[3] = {1, 2, 3};
constexpr int CNT2_E[2] = {1, 2};

// 駒価値 VAL(PieceType enum 0..14 で index、King=99、成駒は全て 6)。
// 契約 shogi_halfka_hm_aux.rs VAL と完全一致。
constexpr int AUX_VAL[15] = {
	0,   // NO_PIECE_TYPE
	1,   // PAWN
	3,   // LANCE
	4,   // KNIGHT
	5,   // SILVER
	8,   // BISHOP
	10,  // ROOK
	6,   // GOLD
	99,  // KING
	6,   // PRO_PAWN
	6,   // PRO_LANCE
	6,   // PRO_KNIGHT
	6,   // PRO_SILVER
	10,  // HORSE
	12,  // DRAGON
};
inline int aux_val_of(PieceType pt) { return AUX_VAL[(int)pt]; }

// ---- square helpers(bullet raw == YaneuraOu Square int)----
inline int aux_file(int s) { return s / 9; }
inline int aux_rank(int s) { return s % 9; }
inline int aux_cheby(int a, int b) {
	int df = aux_file(a) - aux_file(b); if (df < 0) df = -df;
	int dr = aux_rank(a) - aux_rank(b); if (dr < 0) dr = -dr;
	return df > dr ? df : dr;
}

// thermometer: x >= edges[i] の bit を base+i として emit。
template <typename F>
inline void emit_thermo(int x, const int* edges, int n, int base, F&& f) {
	for (int i = 0; i < n; ++i)
		if (x >= edges[i]) f(base + i);
}

// ---- 利きマップ(契約 attack_maps と一致)----
// atk[c][81] = 色 c の駒が各マスに利かせている利き数。
// atkmin[c][81] = マスを攻撃する色 c の駒の最小 VAL(99=なし/王のみ)。
struct AttackMaps {
	std::uint16_t atk[2][81];
	std::int32_t  atkmin[2][81];

	explicit AttackMaps(const Position& pos) {
		std::memset(atk, 0, sizeof(atk));
		for (int c = 0; c < 2; ++c)
			for (int s = 0; s < 81; ++s) atkmin[c][s] = 99;
#if defined(P2_AUX_LE)
		// LONG_EFFECT 経路: 差分維持されている board_effect[c].e[sq] を直接読む。
		// これは色 c の利き総数 = fresh の effects_from 集計 atk[c][sq] と bit-exact
		// (YaneuraOu Square int == aux raw index。玉含む全駒の利き数)。
		// 毎 eval の全駒 effects_from ループ(3.5倍減速の真犯人)を排除する。
		// atkmin は board_effect に無い → 本経路では未計算(99 のまま)。HANG は
		// HANGC(count-only: 数的劣勢 a>d + 歩当たり)で代替するため atkmin を使わない。
		for (int c = 0; c < 2; ++c)
			for (int s = 0; s < 81; ++s)
				atk[c][s] = pos.board_effect[c].effect((Square)s);
#else
		const Bitboard occ = pos.pieces();
		Bitboard bb = occ;
		while (bb) {
			const Square from = bb.pop();
			const Piece  pc   = pos.piece_on(from);
			const int    ci   = (int)color_of(pc);
			const int    v    = aux_val_of(type_of(pc));
			// effects_from は玉=kingEffect(占有無関係 8 近傍)、slider=occupancy-aware
			// (遮蔽駒マス自身を含む)で、契約 attack_maps と bit-exact 一致。
			Bitboard e = effects_from(pc, from, occ);
			while (e) {
				const int ti = (int)e.pop();
				atk[ci][ti] += 1;
				if (v < atkmin[ci][ti]) atkmin[ci][ti] = v;
			}
		}
#endif
	}

	// king 周辺(Chebyshev 距離 <= radius)の atk[color] 総和。
	inline int zone_sum(Color color, int ksq, int radius) const {
		const int ci = (int)color;
		const int f0 = aux_file(ksq), r0 = aux_rank(ksq);
		int acc = 0;
		for (int df = -radius; df <= radius; ++df)
			for (int dr = -radius; dr <= radius; ++dr) {
				if (df == 0 && dr == 0) continue;
				int f = f0 + df, r = r0 + dr;
				if (f >= 0 && f < 9 && r >= 0 && r < 9)
					acc += (int)atk[ci][f * 9 + r];
			}
		return acc;
	}
};

// persp 視点(own = persp)の aux active feature を連結空間の local index
// ([0, AUX_DIM_PER_PERSP))で f(local) に渡す。契約 emit_aux_local と同一順序。
template <typename F>
inline void for_each_aux_feature(const Position& pos, const AttackMaps& maps, Color persp, F&& f) {
	const Color us   = persp;
	const Color them = ~persp;

	// ---- KZ: (own玉, 敵玉) × (A8, D8, A24, D24) ----
	const Color king_sides[2] = {us, them};
	for (int ki = 0; ki < 2; ++ki) {
		const Color king_side = king_sides[ki];
		const int   ksq = (int)pos.square<KING>(king_side);
		const Color att_side = ~king_side;
		const int a8  = maps.zone_sum(att_side, ksq, 1);
		const int d8  = maps.zone_sum(king_side, ksq, 1);
		const int a24 = maps.zone_sum(att_side, ksq, 2);
		const int d24 = maps.zone_sum(king_side, ksq, 2);
		const int base = KZ_OFFSET + ki * 30;
		emit_thermo(a8,  A8_E,  8, base,      f);
		emit_thermo(d8,  D8_E,  7, base + 8,  f);
		emit_thermo(a24, A24_E, 8, base + 15, f);
		emit_thermo(d24, D24_E, 7, base + 23, f);
	}

	// ---- HANG: (us, them) × (hang 9×3, cheap 9×3) ----
	const Color hang_sides[2] = {us, them};
	for (int si = 0; si < 2; ++si) {
		const Color side = hang_sides[si];
		const int   side_i = (int)side;
		const int   opp_i  = (int)(~side);
		int c_hang[AUX_NUM_CLASSES] = {0};
		int c_cheap[AUX_NUM_CLASSES] = {0};
#if defined(P2_AUX_LE)
		// HANGC(count-only, atkmin 不使用): board_effect には atkmin が無いため
		//   hang 半 → 数的劣勢 a>d(攻め手数 > 守り手数)、
		//   cheap半 → 歩当たり(敵歩の利きが乗る升)で代替する。
		//   敵歩の利きは pawnBbEffect の bitboard シフト 1 回で列挙(ループ無し)。
		//   ※net(aux768 sb488)は atkmin 版 HANG で訓練されているため eval 値は
		//     不一致だが、発火 index 空間(HANG_OFFSET 区間)は同一なので FT 加算は
		//     成立し NPS 計測が可能(列挙コストは weight 非依存で構造的)。
		const Bitboard opp_pawn_atk = pawnBbEffect(~side, pos.pieces(~side, PAWN));
#endif
		for (int ti = 0; ti < 81; ++ti) {
			const Piece pc = pos.piece_on((Square)ti);
			if (pc == NO_PIECE) continue;
			if (color_of(pc) != side) continue;
			const PieceType pt = type_of(pc);
			if (pt == KING) continue;
			const int tc = Threat::tc_from_pt(pt);
			if (tc < 0) continue;
#if defined(P2_AUX_LE)
			const int a = (int)maps.atk[opp_i][ti];
			const int d = (int)maps.atk[side_i][ti];
			if (a == 0) continue;
			if (a > d) c_hang[tc] += 1;
			if (opp_pawn_atk.test((Square)ti)) c_cheap[tc] += 1;
#else
			if (maps.atk[opp_i][ti] == 0) continue;
			if (maps.atk[side_i][ti] == 0) c_hang[tc] += 1;
			if (maps.atkmin[opp_i][ti] < aux_val_of(pt)) c_cheap[tc] += 1;
#endif
		}
		const int base = HANG_OFFSET + si * 54;
		for (int tc = 0; tc < AUX_NUM_CLASSES; ++tc)
			emit_thermo(c_hang[tc], CNT3_E, 3, base + tc * 3, f);
		for (int tc = 0; tc < AUX_NUM_CLASSES; ++tc)
			emit_thermo(c_cheap[tc], CNT3_E, 3, base + 27 + tc * 3, f);
	}

	// ---- STATKD: (side0=us, side1=them) × class × status × kdist × thermo2 ----
	int skd[2][AUX_NUM_CLASSES][4][3];
	std::memset(skd, 0, sizeof(skd));
	for (int ti = 0; ti < 81; ++ti) {
		const Piece pc = pos.piece_on((Square)ti);
		if (pc == NO_PIECE) continue;
		const PieceType pt = type_of(pc);
		if (pt == KING) continue;
		const int tc = Threat::tc_from_pt(pt);
		if (tc < 0) continue;
		const Color color = color_of(pc);
		const int side = (color == us) ? 0 : 1;
		const int a = (int)maps.atk[(int)(~color)][ti];
		const int d = (int)maps.atk[(int)color][ti];
		const int ksq_own = (int)pos.square<KING>(color);
		const int kd = aux_cheby(ti, ksq_own);
		const int kb = (kd <= 1) ? 0 : (kd == 2) ? 1 : 2;
		if (a >= 1) {
			skd[side][tc][0][kb] += 1;
			if (d == 0) skd[side][tc][1][kb] += 1;
		}
		if (d >= 1) skd[side][tc][2][kb] += 1;
		if (d >= 2) skd[side][tc][3][kb] += 1;
	}
	for (int side = 0; side < 2; ++side)
		for (int tc = 0; tc < AUX_NUM_CLASSES; ++tc)
			for (int st = 0; st < 4; ++st)
				for (int kb = 0; kb < 3; ++kb) {
					const int local = ((side * AUX_NUM_CLASSES + tc) * 4 + st) * 3 + kb;
					emit_thermo(skd[side][tc][st][kb], CNT2_E, 2, STATKD_OFFSET + local * 2, f);
				}
}

} // namespace YaneuraOu::Eval::P2NNUE::Aux

#endif // EVAL_P2NNUE
#endif // P2NNUE_AUX_H_INCLUDED
