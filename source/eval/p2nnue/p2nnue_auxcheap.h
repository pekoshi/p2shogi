#ifndef P2NNUE_AUXCHEAP_H_INCLUDED
#define P2NNUE_AUXCHEAP_H_INCLUDED

// p2shogi NNUE: 「安いスキャン」aux 候補 KZ2x + HG6 + OUT6 の board_effect(LONG_EFFECT
// 差分利き)からの SIMD 抽出。NPS ゲート計測専用(eval への寄与は PSQT-only 経路で、
// 重みは 0 = eval 不変・探索木不変)。
//
// ★契約元(source of truth):
//   scratchpad/cand_features4.py     (board_effect 互換 raw 量の定義)
//   scratchpad/cand_ridge5.py build() (KZ2x / HG6 / OUT6 の符号化)
//
// 依存: pos.board_effect[c].e[sq] (色 c の利き数, do_move 差分更新済) を読むだけ。
//       fresh AttackMaps 再構築(全駒 effects_from ループ = 1.43us)を行わない。
//
// レイアウト(persp p, us=p, them=~p):
//   KZ2x  (KZ_OFF=0):   king(own=us, enemy=them) × cross-cell(ta 6level × td 6level, 非発火除く35)
//   HG6   (HG_OFF=70):  side(us, them) × {light=hang(g0+g1) oh[1,2,3], hang(g2) oh[1,2,3], hang(g3) oh[1,2]}
//   OUT6  (OUT_OFF=86): side(us, them) × {light=out(g0+g1) oh[1,2,3,4], out(g2) oh[1,2,3], out(g3) oh[1,2]}
//   グループ: 0=歩, 1=香桂, 2=金銀成駒, 3=飛角馬龍
//   hang[g] = a>0 && d==0 の side 駒数, out[g] = a>d の side 駒数
//     (a=board_effect[~side], d=board_effect[side])。KZ の a8/d8/a24/d24 は玉 zone 利き和。

#include "../../config.h"

#if defined(EVAL_P2NNUE) && defined(P2_AUXCHEAP)

#include <cstdint>
#include "../../types.h"
#include "../../position.h"
#include "../../bitboard.h"
#include "../../extra/long_effect.h"

#if defined(USE_AVX2)
#include <immintrin.h>
#endif

namespace YaneuraOu::Eval::P2NNUE::AuxCheap {

// PieceType(0..14) → グループ。-1 = NO_PIECE_TYPE / KING(集計対象外)。
// PAWN=1,LANCE=2,KNIGHT=3,SILVER=4,BISHOP=5,ROOK=6,GOLD=7,KING=8,
// PRO_PAWN=9,PRO_LANCE=10,PRO_KNIGHT=11,PRO_SILVER=12,HORSE=13,DRAGON=14
constexpr int GRP[16] = {-1, 0, 1, 1, 2, 3, 3, 2, -1, 2, 2, 2, 2, 3, 3, -1};

// PieceType(0..15) → STATKD class tc(0..8)。-1 = NO_PIECE_TYPE / KING(集計対象外)。
//   0=P,1=L,2=N,3=S,4=GOLDS(金/成銀桂香/と),5=B,6=R,7=H,8=D。
//   (= build_raw の cls[] と同一分類、 per-square LUT スキャン用の直引き表)
constexpr int TC_OF[16] = {-1, 0, 1, 2, 3, 5, 6, 4, -1, 4, 4, 4, 4, 7, 8, -1};

// ---- 次元レイアウト ----
constexpr int KZ_DIM  = 70;  // 2 king × 35 cross-cell
constexpr int HG_DIM  = 16;  // 2 side × (3+3+2)
constexpr int OUT_DIM = 18;  // 2 side × (4+3+2)
constexpr int KZ_OFF  = 0;
constexpr int HG_OFF  = KZ_DIM;           // 70
constexpr int OUT_OFF = KZ_DIM + HG_DIM;  // 86

// ---- STATKD-kd2 C1(count-linear 版 status4→2, 2026-07-25)----
//   絶対色 cnt[c][tc][st][kb] を 1 回計算し、視点は side 添字の付け替えで両視点。
//   tc = 9 class {P,L,N,S,GOLDS,B,R,H,D}, st = 2 status {att, def1}(C1 削減:
//   旧 status4 {att,hang,def1,def2} から hang/def2 を drop)、
//   kb = 2 kdist {Chebyshev<=2, >=3}。各セルは **生カウント v** を持ち、 for_each では
//   発火 index を v 回 emit (multiplicity)、 推論では per-piece LUT 2 加算で等価計算する。
//   (C1: status4→2 で 次元 144→72、 aux ΔR²(C0-C1)=+0.0023 = aux benefit 98% 保持、
//    実 NPS ~+5% 見込み。 status4 count-linear は thermo2 288 から変更した中間版)。
constexpr int SKD_NTC = 9, SKD_NST = 2, SKD_NKB = 2;
constexpr int SKD_PER_SIDE = SKD_NTC * SKD_NST * SKD_NKB;   // 36 cells / side
constexpr int STATKD_OFF = KZ_DIM + HG_DIM + OUT_DIM;       // 104
constexpr int STATKD_DIM = 2 * SKD_PER_SIDE;               // 2 side × 36 = 72 (count-linear C1)

constexpr int DIM_PER_PERSP = STATKD_OFF + STATKD_DIM;      // 104 + 72 = 176

// KZ2x 合成量の量子化 edge(代表値。NPS ゲートは 0 重みのため厳密 bucket 不問)。
constexpr int TA_E[5] = {4, 8, 12, 18, 26};
constexpr int TD_E[5] = {4, 8, 12, 16, 22};
// one-hot(count) の edge。
constexpr int OH3_E[3] = {1, 2, 3};
constexpr int OH4_E[4] = {1, 2, 3, 4};
constexpr int CNT2_E[2] = {1, 2};

// ---- board_effect から 1 回だけ集計する raw 量(視点非依存)----
struct Raw {
	// KZ: 玉(色 kc)の zone 利き和。attacker = ~kc。
	int a8[2], d8[2], a24[2], d24[2], ge2[2];  // index = king color
	// HG/OUT: side(色)ごとのグループ別カウント。
	int hang[2][4];  // [side color][grp]
	int outc[2][4];  // [side color][grp]
	// STATKD-kd2: 絶対色カウント cnt[color][tc][st][kb]。
	int statkd[2][SKD_NTC][SKD_NST][SKD_NKB];
};

// 玉 zone 利き和(radius 1 / 2 の Chebyshev)の SIMD 化(fable5 D1、2026-07-25)。
//   per-ksq 事前計算マスク(0xFF=zone 内 / 0x00=zone 外・盤端・玉自身)を board_effect の
//   e[0..95](= 全 81 マス + padding2)と AND し、_mm256_sad_epu8 で u8 水平和する。
//   ★ 常に e[0] から 3×32B ロードするため、前方 padding(SQ_22=10B のみ)不足による
//     e[sq-20] の struct 外読み(窓ロード方式の front-OOB)が起きない。マスクが 81..95 を
//     0 にするので padding2 のゴミ値も混入しない。u8 の単純整数和ゆえ旧スカラー版と厳密一致。
struct KzZoneMask {
	alignas(32) std::uint8_t r1[81][96];  // radius 1(8 近傍)
	alignas(32) std::uint8_t r2[81][96];  // radius 2(24 近傍)
};
inline const KzZoneMask& kz_zone_mask() {
	static const KzZoneMask M = [] {
		KzZoneMask k{};  // 全 0 初期化(zone 外・cell 81..95 は 0 のまま)
		for (int ksq = 0; ksq < 81; ++ksq) {
			const int f0 = ksq / 9, r0 = ksq % 9;
			for (int radius = 1; radius <= 2; ++radius) {
				std::uint8_t* const mask = (radius == 1) ? k.r1[ksq] : k.r2[ksq];
				for (int df = -radius; df <= radius; ++df)
					for (int dr = -radius; dr <= radius; ++dr) {
						if (df == 0 && dr == 0) continue;  // 玉自身は除外
						const int f = f0 + df, r = r0 + dr;
						if (f >= 0 && f < 9 && r >= 0 && r < 9)
							mask[f * 9 + r] = 0xFF;  // 旧 zone_sum の境界チェックと同一集合
					}
			}
		}
		return k;
	}();
	return M;
}
// board_effect(bb)を mask96(96B, 各バイト 0x00/0xFF)で AND した u8 和を返す。
inline int masked_board_sum(const LongEffect::ByteBoard& bb, const std::uint8_t* mask96) {
#if defined(USE_AVX2)
	const std::uint8_t* const e = bb.e;
	const __m256i zero = _mm256_setzero_si256();
	__m256i acc = zero;
	for (int c = 0; c < 3; ++c) {
		const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(e + c * 32));
		const __m256i m = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask96 + c * 32));
		acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_and_si256(v, m), zero));
	}
	// 4 個の 64bit レーン部分和を水平加算。
	const __m128i lo = _mm256_castsi256_si128(acc);
	const __m128i hi = _mm256_extracti128_si256(acc, 1);
	__m128i s = _mm_add_epi64(lo, hi);
	s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
	return (int)_mm_cvtsi128_si64(s);
#else
	int acc = 0;
	for (int cell = 0; cell < 81; ++cell)
		if (mask96[cell]) acc += (int)bb.e[cell];
	return acc;
#endif
}

// 2 つの ByteBoard(opp, own)の 81 マス比較で Bitboard を生成。
//   HANG: opp.e[sq] > 0 && own.e[sq] == 0
//   OUT : opp.e[sq] > own.e[sq]
// SIMD(AVX2)で 32 マス単位に比較 → movemask → Bitboard レイアウトへ詰める。
template <bool IsHang>
inline Bitboard cmp_mask(const LongEffect::ByteBoard& opp, const LongEffect::ByteBoard& own) {
	std::uint64_t lo = 0, hi = 0;  // lo=sq0..63, hi=sq64..80
#if defined(USE_AVX2)
	const __m256i zero = _mm256_setzero_si256();
	for (int c = 0; c < 3; ++c) {
		const __m256i o = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(opp.e + c * 32));
		const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(own.e + c * 32));
		// 利き数は < 128 のため signed cmpgt/cmpeq で unsigned 比較と一致。
		__m256i m;
		if (IsHang) {
			const __m256i og0 = _mm256_cmpgt_epi8(o, zero);  // opp > 0
			const __m256i we0 = _mm256_cmpeq_epi8(w, zero);  // own == 0
			m = _mm256_and_si256(og0, we0);
		} else {
			m = _mm256_cmpgt_epi8(o, w);                     // opp > own
		}
		const std::uint32_t mask = (std::uint32_t)_mm256_movemask_epi8(m);
		if (c == 0)      lo |= (std::uint64_t)mask;
		else if (c == 1) lo |= ((std::uint64_t)mask) << 32;
		else             hi |= (std::uint64_t)mask;  // sq64..(95) の下位 17bit のみ有効
	}
#else
	for (int sq = 0; sq < 81; ++sq) {
		const int a = (int)opp.e[sq], d = (int)own.e[sq];
		const bool set = IsHang ? (a > 0 && d == 0) : (a > d);
		if (set) { if (sq < 64) lo |= (1ULL << sq); else hi |= (1ULL << (sq - 64)); }
	}
#endif
	// Bitboard レイアウト: p[0]=sq0..62, p[1]=sq63..80(bit sq-63)。
	const std::uint64_t p0 = lo & ((1ULL << 63) - 1);
	std::uint64_t p1 = (lo >> 63) | (hi << 1);
	p1 &= ((1ULL << 18) - 1);
	return Bitboard(p0, p1);
}

// ByteBoard の 81 マスを THR と比較(e[sq] > THR)し 81bit Bitboard 化(SIMD 1 パス)。
//   ge1[c] = bytes_gt<0>(be[c]) (e>=1), ge2[c] = bytes_gt<1>(be[c]) (e>=2)。
//   cmp_mask と同一の movemask → Bitboard(p0=sq0..62, p1=sq63..80)パッキング。
//   利き数 <= 38 のため signed cmpgt_epi8 で unsigned 比較と一致。ゴミ bit は p1 の
//   0x3FFFF マスクで除去(sq81..95 は無効)。
template <int THR>
inline Bitboard bytes_gt(const LongEffect::ByteBoard& b) {
	std::uint64_t lo = 0, hi = 0;
#if defined(USE_AVX2)
	const __m256i t = _mm256_set1_epi8((char)THR);
	for (int c = 0; c < 3; ++c) {
		const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b.e + c * 32));
		const __m256i m = _mm256_cmpgt_epi8(v, t);  // v > THR
		const std::uint32_t mask = (std::uint32_t)_mm256_movemask_epi8(m);
		if (c == 0)      lo |= (std::uint64_t)mask;
		else if (c == 1) lo |= ((std::uint64_t)mask) << 32;
		else             hi |= (std::uint64_t)mask;
	}
#else
	for (int sq = 0; sq < 81; ++sq)
		if ((int)b.e[sq] > THR) { if (sq < 64) lo |= (1ULL << sq); else hi |= (1ULL << (sq - 64)); }
#endif
	const std::uint64_t p0 = lo & ((1ULL << 63) - 1);
	std::uint64_t p1 = (lo >> 63) | (hi << 1);
	p1 &= ((1ULL << 18) - 1);
	return Bitboard(p0, p1);
}

// KD2_LUT[ksq][kb]: 自玉 ksq から Chebyshev 距離 <=2(kb=0)/ >=3(kb=1)のマス集合。
//   初回呼び出しで一度だけ構築(function-local static、C++11 でスレッド安全)。2.6KB。
struct Kd2Lut { Bitboard b[81][2]; };
inline const Kd2Lut& kd2_lut() {
	static const Kd2Lut L = [] {
		Kd2Lut k{};
		for (int ksq = 0; ksq < 81; ++ksq) {
			const int f0 = ksq / 9, r0 = ksq % 9;
			for (int sq = 0; sq < 81; ++sq) {
				const int df = sq / 9 - f0, dr = sq % 9 - r0;
				const int adf = df < 0 ? -df : df, adr = dr < 0 ? -dr : dr;
				const int kd = adf > adr ? adf : adr;
				const int kb = kd <= 2 ? 0 : 1;
				k.b[ksq][kb] |= Bitboard((Square)sq);
			}
		}
		return k;
	}();
	return L;
}

// board_effect から KZ / HG / OUT の raw 量のみ集計(推論ホットパス用)。
// STATKD は build_statkd / per-piece LUT で別途扱う(bytes_gt / 144 popcount を回避)。
inline void build_kzhgout(const Position& pos, Raw& r) {
	const LongEffect::ByteBoard* be = pos.board_effect;
	const KzZoneMask& zm = kz_zone_mask();
	// ---- KZ zone 量(玉ごと)----
	for (int kci = 0; kci < 2; ++kci) {
		const Color kc = (Color)kci;
		const int ksq = (int)pos.square<KING>(kc);
		const LongEffect::ByteBoard& att = be[(int)(~kc)];
		const LongEffect::ByteBoard& own = be[kci];
		r.a8[kci]  = masked_board_sum(att, zm.r1[ksq]);
		r.d8[kci]  = masked_board_sum(own, zm.r1[ksq]);
		r.a24[kci] = masked_board_sum(att, zm.r2[ksq]);
		r.d24[kci] = masked_board_sum(own, zm.r2[ksq]);
		// ge2 = zone8 で attacker 利き >= 2 のマス数(SIMD around8_greater_than_one)。
		const Effect8::Directions d2 = att.around8_greater_than_one((Square)ksq);
		const Effect8::Directions mk = Effect8::board_mask((Square)ksq);
		r.ge2[kci] = __builtin_popcount((unsigned)(d2 & mk) & 0xff);
	}
	// ---- HG / OUT グループカウント(side ごと)----
	for (int sci = 0; sci < 2; ++sci) {
		const Color s = (Color)sci;
		const Bitboard hb = cmp_mask<true >(be[(int)(~s)], be[sci]);
		const Bitboard ob = cmp_mask<false>(be[(int)(~s)], be[sci]);
		const Bitboard g0 = pos.pieces(s, PAWN);
		const Bitboard g1 = pos.pieces(s, LANCE, KNIGHT);
		const Bitboard g2 = pos.pieces(s, SILVER, GOLD, PRO_PAWN, PRO_LANCE, PRO_KNIGHT, PRO_SILVER);
		const Bitboard g3 = pos.pieces(s, BISHOP, ROOK, HORSE, DRAGON);
		r.hang[sci][0] = (hb & g0).pop_count();
		r.hang[sci][1] = (hb & g1).pop_count();
		r.hang[sci][2] = (hb & g2).pop_count();
		r.hang[sci][3] = (hb & g3).pop_count();
		r.outc[sci][0] = (ob & g0).pop_count();
		r.outc[sci][1] = (ob & g1).pop_count();
		r.outc[sci][2] = (ob & g2).pop_count();
		r.outc[sci][3] = (ob & g3).pop_count();
	}
}

// STATKD-kd2 の絶対色カウント r.statkd[c][tc][st][kb] を bitboard で集計。
// dump / for_each(検証)用。推論ホットパスは per-piece LUT スキャンを使い本関数を呼ばない。
inline void build_statkd(const Position& pos, Raw& r) {
	const LongEffect::ByteBoard* be = pos.board_effect;
	// ---- STATKD-kd2 C1(絶対色カウント status4→2, 契約 = statkd_bbverify.py bitboard版)----
	const Kd2Lut& lut = kd2_lut();
	// Phase A: ge1 Bitboard 化(色別, 2 パス)。C1 は att/def1 のみ使うため ge2 は不要
	// (旧 def2=ge2[c] を drop、 bytes_gt<1> 2 パスと hang の andnot を削減)。
	Bitboard ge1[2];
	for (int c = 0; c < 2; ++c)
		ge1[c] = bytes_gt<0>(be[c]);  // e >= 1
	for (int c = 0; c < 2; ++c) {
		const int opp = c ^ 1;
		// status[st'](C1): att=ge1[opp](a>=1), def1=ge1[c](d>=1)。hang/def2 は drop。
		Bitboard status[SKD_NST];
		status[0] = ge1[opp];  // att : a>=1
		status[1] = ge1[c];    // def1: d>=1
		const int ksq = (int)pos.square<KING>((Color)c);
		// class bitboard(自玉は class に含まれず自然に除外)。
		Bitboard cls[SKD_NTC];
		cls[0] = pos.pieces((Color)c, PAWN);
		cls[1] = pos.pieces((Color)c, LANCE);
		cls[2] = pos.pieces((Color)c, KNIGHT);
		cls[3] = pos.pieces((Color)c, SILVER);
		cls[4] = pos.pieces((Color)c, GOLD, PRO_PAWN, PRO_LANCE, PRO_KNIGHT, PRO_SILVER);
		cls[5] = pos.pieces((Color)c, BISHOP);
		cls[6] = pos.pieces((Color)c, ROOK);
		cls[7] = pos.pieces((Color)c, HORSE);
		cls[8] = pos.pieces((Color)c, DRAGON);
		for (int st = 0; st < SKD_NST; ++st)
			for (int kb = 0; kb < SKD_NKB; ++kb) {
				// Phase B: stkd = status[st] & KD2_LUT[ksq][kb]。
				const Bitboard stkd = status[st] & lut.b[ksq][kb];
				// Phase C: cnt = popcount(class & stkd)。
				for (int tc = 0; tc < SKD_NTC; ++tc)
					r.statkd[c][tc][st][kb] = (cls[tc] & stkd).pop_count();
			}
	}
}

// board_effect から Raw を全部集計(KZ/HG/OUT + STATKD)。dump / 検証用。
inline void build_raw(const Position& pos, Raw& r) {
	build_kzhgout(pos, r);
	build_statkd(pos, r);
}

// one-hot(level): level = count(v >= edges[i])。level>0 なら base+(level-1) を emit。
template <typename F>
inline void emit_oh(int v, const int* edges, int n, int base, F&& f) {
	int level = 0;
	for (int i = 0; i < n; ++i) level += (v >= edges[i]);
	if (level > 0) f(base + level - 1);
}

// persp 視点の KZ/HG/OUT 発火 local index のみ f(local) に渡す(推論ホットパス用)。
template <typename F>
inline void for_each_kzhgout(const Raw& r, Color persp, F&& f) {
	const int us = (int)persp, them = (int)(~persp);
	// ---- KZ2x: king(own=us, enemy=them)----
	const int kings[2] = {us, them};
	for (int ki = 0; ki < 2; ++ki) {
		const int kc = kings[ki];
		const int ta = 2 * r.a8[kc] + r.a24[kc] + 2 * r.ge2[kc];
		const int td = 2 * r.d8[kc] + r.d24[kc];
		int la = 0, lb = 0;
		for (int e : TA_E) la += (ta >= e);
		for (int e : TD_E) lb += (td >= e);
		const int cell = la * 6 + lb;  // 0..35
		if (cell > 0) f(KZ_OFF + ki * 35 + (cell - 1));
	}
	// ---- HG6: side(us, them)----
	const int sides[2] = {us, them};
	for (int si = 0; si < 2; ++si) {
		const int s = sides[si];
		const int base = HG_OFF + si * 8;
		emit_oh(r.hang[s][0] + r.hang[s][1], OH3_E, 3, base,     f);
		emit_oh(r.hang[s][2],                OH3_E, 3, base + 3, f);
		emit_oh(r.hang[s][3],                CNT2_E, 2, base + 6, f);
	}
	// ---- OUT6: side(us, them)----
	for (int si = 0; si < 2; ++si) {
		const int s = sides[si];
		const int base = OUT_OFF + si * 9;
		emit_oh(r.outc[s][0] + r.outc[s][1], OH4_E, 4, base,     f);
		emit_oh(r.outc[s][2],                OH3_E, 3, base + 4, f);
		emit_oh(r.outc[s][3],                CNT2_E, 2, base + 7, f);
	}
}

// persp 視点の全発火 local index を f(local) に渡す(dump / bit-exact 検証用)。
// STATKD は count-linear: 各 cell の生カウント v を multiplicity で emit(v 回)。
template <typename F>
inline void for_each(const Raw& r, Color persp, F&& f) {
	for_each_kzhgout(r, persp, f);
	const int us = (int)persp, them = (int)(~persp);
	const int sides[2] = {us, them};
	// ---- STATKD-kd2 C1: side(us, them) × 9tc × 2st'(att,def1) × 2kb × count-linear----
	//   絶対色カウント r.statkd を側添字 relabel(side0=us 色, side1=them 色)。
	for (int si = 0; si < 2; ++si) {
		const int c = sides[si];
		const int sbase = STATKD_OFF + si * SKD_PER_SIDE;
		for (int tc = 0; tc < SKD_NTC; ++tc)
			for (int st = 0; st < SKD_NST; ++st)
				for (int kb = 0; kb < SKD_NKB; ++kb) {
					const int v = r.statkd[c][tc][st][kb];
					const int cell = sbase + (tc * SKD_NST + st) * SKD_NKB + kb;
					for (int i = 0; i < v; ++i) f(cell);  // count-linear multiplicity
				}
	}
}

}  // namespace YaneuraOu::Eval::P2NNUE::AuxCheap

#endif  // EVAL_P2NNUE && P2_AUXCHEAP
#endif  // P2NNUE_AUXCHEAP_H_INCLUDED
