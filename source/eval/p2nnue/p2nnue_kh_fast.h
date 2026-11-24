// KH (H 部 = 浮き駒) の高速版。 2026-11-13
//   `-DP2_KH -DP2_KH_FAST` のときだけ p2nnue_kh.h から (namespace kh の中で) include される。
//   特徴の定義・重み・足し込む値は p2nnue_kh.h の apply() と**完全に同じ** (bit-exact)。 変えたのは計算の仕方だけ。
//
// ■ 何を変えたか
//   (1) 利きの和集合 any[c] を、 駒種ごと・駒ごとのループ (10 種 x 2 色 = 20 本の可変長ループ) ではなく
//       **盤全体の shift** で作る。 歩・桂・銀・金相当・玉・馬/龍の「1 歩の利き」は方向ごとに 1 回の shift。
//         前 = 歩 | 銀 | 金相当 | HDK / 斜め前 = 銀 | 金相当 | HDK / 横・後ろ = 金相当 | HDK / 斜め後ろ = 銀 | HDK
//       (HDK = 馬 龍 玉。 銀の方向 + 金の方向 = 8 方向なので、 両方に入れれば玉の 8 近傍になる = Apery の attackers_to と同じ手口)
//       ループが残るのは香・角/馬・飛/龍 (両色あわせて盤上に最大 4 + 2 + 2 枚) だけ。
//   (2) 浮き駒が 1 枚も無ければ何も書かずに false を返す (呼び出し側は threat acc をそのまま transform に渡す)。
//   (3) 特徴の index を表引き (駒 x 段区分) にし、 重みの行は AVX2 のレジスタ上で足してから 1 回だけ store。
//   (4) K 部に重みがある KHFile のときは従来の apply() に落とす (K 部は利きの「数」が要るので別物)。
//
// ■ bit-exact の根拠
//   any[c] は「c の駒の利きが 1 つ以上ある升」の集合で、 作り方 (駒ごとの OR / 方向ごとの shift) に依らない。
//   浮き駒の集合が同じなら特徴の多重集合が同じで、 i16 の wrapping 加算は順序に依らない。
//   🚨前提: 行き所のない駒 (先手の 1 段目の歩・香、 1-2 段目の桂 / 後手はその逆) が盤上に無いこと。
//     従来版は歩を `bb >> 1` で動かすので 1 段目の歩があると隣の筋の 9 段目に利きが漏れるが、 合法局面では起きない。
#pragma once

#if defined(P2_KH) && defined(P2_KH_FAST)

#if defined(USE_AVX2)
static_assert(THREAT_ACC_DIM % 16 == 0, "KH_FAST: threat acc の幅は 16 の倍数 (AVX2 の i16 x 16) であること");
#endif

namespace fast {

typedef unsigned __int128 u128;

// 盤 81 升を 1 本の 128 bit 整数に詰める。 bit 番号 = Square の値 (0..80)。
//   Bitboard は p[0] = 升 0..62 (bit 63 は未使用)、 p[1] = 升 63..80 なので、 p[1] を 63 bit 左へずらして繋ぐ。
static inline u128 to128(const Bitboard& b) {
	const u64 p0 = b.extract64<0>(), p1 = b.extract64<1>();
	return (u128) p0 | ((u128) p1 << 63);
}

static constexpr u128 make_rank(int r) {
	u128 x = 0;
	for (int f = 0; f < 9; ++f) x |= (u128) 1 << (f * 9 + r);
	return x;
}
static constexpr u128 ALL81 = (((u128) 1) << 81) - 1;
static constexpr u128 R1 = make_rank(0), R2 = make_rank(1), R8 = make_rank(7), R9 = make_rank(8);

// 段区分 (先手から見た段 / 3)。 後手視点は 2 - ZONE。
struct ZoneTable {
	std::uint8_t z[SQ_NB_PLUS1];
	constexpr ZoneTable() : z() { for (int s = 0; s < int(SQ_NB_PLUS1); ++s) z[s] = (std::uint8_t) ((s % 9) / 3); }
};
static constexpr ZoneTable ZONE{};

// g_row[視点 P][駒 * 4 + 先手から見た段区分] = g_w の行番号 (学習器の列番号)。 load のたびに fast_init() が作る。
//   玉と空升の欄は 0 (浮き駒の集合から玉を除いているので読まれない)。
alignas(64) static std::uint16_t g_row[COLOR_NB][PIECE_NB * 4];

static inline void init() {
	for (int p = 0; p < COLOR_NB; ++p)
		for (int pc = 0; pc < int(PIECE_NB); ++pc)
			for (int z = 0; z < 4; ++z) {
				int v = 0;
				const int pt = piece_index(type_of(Piece(pc)));
				if (pt >= 0 && z < 3) {
					const int side = (int(color_of(Piece(pc))) == p) ? 0 : 1;   // P の駒 = 0 / 相手の駒 = 1
					const int zone = (p == BLACK) ? z : 2 - z;                  // P から見た段 / 3
					v = g_kdims + side * 39 + pt * 3 + zone;
				}
				g_row[p][pc * 4 + z] = (std::uint16_t) v;
			}
}

// 色 C の駒の利きがある升 (1 枚目で止まる。 玉の利きも含む)。 = 従来版 effects_of<C, false> の any[C]。
template <Color C>
static inline u128 any_effect(const Position& pos, const Bitboard& occ) {
	const Bitboard us = pos.pieces(C);
	const u128 pawn = to128(pos.pieces(PAWN) & us);
	const u128 kn   = to128(pos.pieces(KNIGHT) & us);
	const u128 sv   = to128(pos.pieces(SILVER_HDK) & us);   // 銀 + 馬 龍 玉
	const u128 gd   = to128(pos.pieces(GOLDS_HDK) & us);    // 金相当 + 馬 龍 玉
	const u128 sg   = sv | gd;
	// 升 = 筋 * 9 + 段。 段を 1 つ上へ = >> 1 / 下へ = << 1、 筋を 1 つ動かす = >> 9 / << 9。
	//   段方向へ動かす前に端の段を落とす (落とさないと隣の筋へ回り込む)。 筋方向は 0 未満 / 81 以上へ出るだけなので最後に ALL81 で切る。
	u128 e;
	if (C == BLACK) {
		const u128 f  = (pawn | sg) & ~R1;   // 前へ動ける駒
		const u128 fd = sg & ~R1;            // 斜め前
		const u128 bk = gd & ~R9;            // 後ろ (金相当 + HDK)
		const u128 bd = sv & ~R9;            // 斜め後ろ (銀 + HDK)
		const u128 k2 = kn & ~(R1 | R2);     // 桂 (2 段上)
		e = (f >> 1) | (fd >> 10) | (fd << 8) | (gd >> 9) | (gd << 9) | (bk << 1) | (bd >> 8) | (bd << 10)
		  | (k2 >> 11) | (k2 << 7);
	} else {
		const u128 f  = (pawn | sg) & ~R9;
		const u128 fd = sg & ~R9;
		const u128 bk = gd & ~R1;
		const u128 bd = sv & ~R1;
		const u128 k2 = kn & ~(R8 | R9);
		e = (f << 1) | (fd << 10) | (fd >> 8) | (gd >> 9) | (gd << 9) | (bk >> 1) | (bd << 8) | (bd >> 10)
		  | (k2 << 11) | (k2 >> 7);
	}
	// 走り駒。 馬・龍の 8 近傍は上の HDK で済んでいるので、 ここは角・飛の利きだけ足す。
	//   香 = 縦の利き (rookFileEffect) のうち前方だけ。 lanceEffect<C> と同じ集合。
	for (Bitboard bb = pos.pieces(LANCE) & us; bb; ) {
		const Square s = bb.pop();
		e |= to128(rookFileEffect(s, occ) & lanceStepEffect<C>(s));
	}
	for (Bitboard bb = pos.pieces(BISHOP_HORSE) & us; bb; ) { const Square s = bb.pop(); e |= to128(bishopEffect(s, occ)); }
	for (Bitboard bb = pos.pieces(ROOK_DRAGON) & us;  bb; ) { const Square s = bb.pop(); e |= to128(rookEffect(s, occ)); }
	return e & ALL81;
}

// 浮き駒の集合 (両色)。 = 従来版の hang を両色で OR したもの (駒の色は piece_on で判る)。
static inline u128 hanging(const Position& pos) {
	const Bitboard occ = pos.pieces();
	const u128 eb = any_effect<BLACK>(pos, occ);
	const u128 ew = any_effect<WHITE>(pos, occ);
	const u128 pb = to128(pos.pieces(BLACK)), pw = to128(pos.pieces(WHITE));
	const u128 kg = to128(pos.pieces(KING));
	return ((pb & ew & ~eb) | (pw & eb & ~ew)) & ~kg;
}

} // namespace fast

// apply() の高速版。 浮き駒が無ければ **out_s / out_n を書かずに** false を返す
//   (呼び出し側は thr_s / thr_n をそのまま使う = apply() が書くはずだった値と同じ)。
//   true のとき out_s / out_n = threat acc (stm / nstm) + Σ KH 行 (apply() と同じ値)。 g_loaded のときだけ呼ぶこと。
static inline bool apply_fast(const Position& pos, Color stm,
                              const std::int16_t* thr_s, const std::int16_t* thr_n,
                              std::int16_t* out_s, std::int16_t* out_n) {
	if (g_k_on) {            // K 部つきの KHFile は従来の経路 (利きの数が要る)
		apply(pos, stm, thr_s, thr_n, out_s, out_n);
		return true;
	}
	if (!g_h_on) return false;   // 重みが全部 0 = 何も足さない (従来版も特徴を作らない)
	fast::u128 hang = fast::hanging(pos);
	if (hang == 0) return false;
	const std::uint16_t* const rs = fast::g_row[stm];
	const std::uint16_t* const rn = fast::g_row[~stm];
#if defined(USE_AVX2)
	constexpr int NV = THREAT_ACC_DIM / 16;
	__m256i as[NV], an[NV];
	// SAFETY (unaligned load / store と境界検査なしの読み書き):
	//   - thr_s / thr_n / out_s / out_n は呼び出し側が THREAT_ACC_DIM (= NV * 16) 個の i16 を確保している
	//     (threat_acc_ptr は StateInfo::p2_threat_acc[persp] の先頭、 out_* は evaluate の一時配列)。 loadu / storeu なので整列は不問。
	//   - g_w の行番号は fast::g_row の値で、 init() が [0, g_kdims + 78) (<= MAX_DIMS) の範囲にしか作らない。
	//     g_w は alignas(64) で 1 行 = THREAT_ACC_DIM * 2 byte。 THREAT_ACC_DIM が 16 の倍数なので各行の先頭は 32 byte 境界に載る
	//     (下は loadu を使うので、 幅を変えても整列の前提は壊れない)。
	//   - 不変条件: hang の bit は盤上の玉以外の駒の升だけ (sq < 81、 piece_on(sq) は NO_PIECE でも玉でもない)。
	for (int i = 0; i < NV; ++i) {
		as[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(thr_s + 16 * i));
		an[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(thr_n + 16 * i));
	}
	do {
		const u64 lo = (u64) hang, hi = (u64) (hang >> 64);
		const int sq = lo ? (int) LSB64(lo) : 64 + (int) LSB64(hi);
		hang &= hang - 1;
		const int k = int(pos.piece_on(Square(sq))) * 4 + fast::ZONE.z[sq];
		const std::int16_t* ws = g_w[rs[k]];
		const std::int16_t* wn = g_w[rn[k]];
		// i16 の wrapping 加算 (_mm256_add_epi16 は飽和しない = 従来版のスカラー加算と同じ)。
		for (int i = 0; i < NV; ++i) {
			as[i] = _mm256_add_epi16(as[i], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ws + 16 * i)));
			an[i] = _mm256_add_epi16(an[i], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wn + 16 * i)));
		}
	} while (hang);
	for (int i = 0; i < NV; ++i) {
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(out_s + 16 * i), as[i]);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(out_n + 16 * i), an[i]);
	}
#else
	std::memcpy(out_s, thr_s, sizeof(std::int16_t) * THREAT_ACC_DIM);
	std::memcpy(out_n, thr_n, sizeof(std::int16_t) * THREAT_ACC_DIM);
	do {
		const u64 lo = (u64) hang, hi = (u64) (hang >> 64);
		const int sq = lo ? (int) LSB64(lo) : 64 + (int) LSB64(hi);
		hang &= hang - 1;
		const int k = int(pos.piece_on(Square(sq))) * 4 + fast::ZONE.z[sq];
		const std::int16_t* ws = g_w[rs[k]];
		const std::int16_t* wn = g_w[rn[k]];
		for (int i = 0; i < THREAT_ACC_DIM; ++i) {
			out_s[i] = (std::int16_t) (std::uint16_t) ((std::uint16_t) out_s[i] + (std::uint16_t) ws[i]);
			out_n[i] = (std::int16_t) (std::uint16_t) ((std::uint16_t) out_n[i] + (std::uint16_t) wn[i]);
		}
	} while (hang);
#endif
	return true;
}

// eval コマンド (dump) 用の自己検査: 同じ局面で apply() と apply_fast() の出力を突き合わせる。
static inline bool fast_selfcheck(const Position& pos, Color stm, const std::int16_t* thr_s, const std::int16_t* thr_n) {
	alignas(32) std::int16_t a_s[THREAT_ACC_DIM], a_n[THREAT_ACC_DIM], b_s[THREAT_ACC_DIM], b_n[THREAT_ACC_DIM];
	apply(pos, stm, thr_s, thr_n, a_s, a_n);
	if (!apply_fast(pos, stm, thr_s, thr_n, b_s, b_n)) {
		std::memcpy(b_s, thr_s, sizeof(b_s));
		std::memcpy(b_n, thr_n, sizeof(b_n));
	}
	return std::memcmp(a_s, b_s, sizeof(a_s)) == 0 && std::memcmp(a_n, b_n, sizeof(a_n)) == 0;
}

#endif // P2_KH && P2_KH_FAST
