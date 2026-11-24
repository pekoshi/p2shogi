// KH ブロック (K 部 = 玉近傍の空升 x 敵の利き / H 部 = 浮き駒) — エンジン側。 2026-11-12
//   `-DP2_KH` のときだけ p2nnue.cpp から (namespace YaneuraOu::Eval::P2NNUE の中で) include される。
//   初版 (09:05) は H 部だけ。 17 時台に K 部 (kmode 1 / kmode 2) を追加 (旧版 = p2nnue_kh.h.bak_kh3_20261112)。
//
// ■ 学習器の正本
//   E:/shogi/bullet-shogi-nodchip-v2/crates/bullet_lib/src/game/inputs/shogi_kh.rs
//     (kh_features_both の kmode 1 / kh2_region / push_k2 / H 部)
//   E:/shogi/bullet-shogi-nodchip-v2/examples/shogi_layerstack.rs (kh_add_rows / --kh-dump)
//   Python: C:/project/p2shogi/runs/sibling/kh_verify.py (onehot / k2_feats) / sib_feats.py
//
// ■ 利き eff[c][sq]
//   色 c の駒 (玉を含む) が sq に利いている数。 slider は 1 枚目で止まる (x-ray なし)。
//   = popcount(Position::attackers_to(c, sq))。 ここでは駒ごとの利き Bitboard を
//   ビットスライスの加算器 (b0 / b1 / b2 / 飽和 sat) で数える (1 回の駒列挙で全升の数が出る)。
//   8 以上は sat に入る (K 部は min(e, 3) / min(zaE, 7) しか使わないので 8 で頭打ちで十分)。
//
// ■ 特徴 (視点 P ごと。 index は学習器と同じ = kh_ft_w の列番号)
//   kmode 1 (dims 142, K 部 [0, 64)):
//     side * 32 + dir * 4 + level
//       side 0 = P の玉 / 1 = P の相手の玉。 dir は **P から見た** 向き (P = 後手なら盤を 180 度回す)
//       level = min(その空升への、 その玉の敵の利き数, 3)。 盤外・駒のある升は出さない
//   kmode 2 (dims 798, K 部 [0, 720)):
//     玉ごとに **持ち主から見た** 向き。 持ち主から見た筋 index >= 5 (6-9 筋) なら左右反転 (df を反転)。
//     region = 筋区分 (端からの距離 1-2 / 3-4 / 5 筋) * 3 + 段区分 (9 段 / 8 段 / 7 段以上)
//     方向部 [0, 576): ((side * 9 + region) * 8 + dir) * 4 + level
//     集計部 [576, 720): 576 + (side * 9 + region) * 8 + min(zaE, 7)   (空升が 0 個でも 1 本立つ)
//   H 部 [k_dims, k_dims + 78): k_dims + side * 39 + pt * 3 + zone
//     浮き駒 = 玉以外の盤上の駒で、 敵の利きがあり味方の利きが無いもの。
//     side 0 = P の駒 / 1 = P の相手の駒、 pt = 歩 香 桂 銀 角 飛 金 = 0..6 / と 成香 成桂 成銀 馬 龍 = 7..12
//     zone = P から見た段 / 3
//   同じ index が複数立てば複数回足す (sparse matmul の重複と同じ)。 1 視点の上限 = 18 + 38 = 56 本。
//
// ■ 重み (KHFile)
//   `--kh-dump` の f32 LE、 [out=64 x dims]、 values[feature * out + row] (column-major)。
//   同名 + ".json" で形を検査する (out / dims / k_dims / h_dims / row_offset / layout、 kmode 2 は kmode と k_sub も)。
//   json に kmode が無ければ 1 (学習器は kmode 1 のとき鍵を書かない)。
//   量子化: round(QA_FT * w) を i16 で保持 (QA_FT = 254 = 学習器の threat 表と FT 本体の目盛り)。
//   K 部 / H 部の重みが全部 0 ならその部分は**特徴を計算しない** (H だけのファイルは従来と同じ計算量・同じ値)。
//
// ■ 足し込み
//   学習器の kh_add_rows (off = 0, W = 64): KH 出力の行 r < 32 は FT[r]、 r >= 32 は FT[512 + r - 32]。
//   エンジンの threat acc b[k] は k < THREAT_LO_DIM(32) が FT[k]、 k >= 32 が FT[FT_HALF + k - 32]。
//   -> W == THREAT_ACC_DIM (64) / off == 0 なら **KH 行 r = threat acc の k = r** で 1 対 1。
//   評価のたびに threat acc + Σ KH 行 を一時 buffer に作り transform に渡す (st->p2_threat_acc は書かない)。
#pragma once

#if defined(P2_KH)

static_assert(THREAT_ACC_DIM % 2 == 0, "KH: threat acc の幅は偶数であること");
static_assert(THREAT_LO_DIM == THREAT_ACC_DIM / 2, "KH: threat acc の前半/後半の分け方が kh_add_rows と食い違う");

namespace kh {

static constexpr int H_DIMS     = 78;
static constexpr int K1_DIMS    = 64;                  // kmode 1 の K 部
static constexpr int K2_REGIONS = 9;
static constexpr int K2_DIR     = 2 * K2_REGIONS * 8 * 4;   // 576
static constexpr int K2_DIMS    = K2_DIR + 2 * K2_REGIONS * 8; // 720
static constexpr int MAX_DIMS   = K2_DIMS + H_DIMS;    // 798
static constexpr int MAX_FEATS  = 64;                  // 1 視点の上限 56 (+余裕)
static constexpr int QA_KH      = 254;                 // = 学習器 QA_FT (threat 表と同じ目盛り)

// 量子化済み重み。 g_w[学習器の列番号][k] は threat acc の k 番目に足す値。
alignas(64) static std::int16_t g_w[MAX_DIMS][THREAT_ACC_DIM];
static bool g_loaded = false;
static int  g_kmode  = 1;      // json の kmode (1 / 2)
static int  g_kdims  = K1_DIMS; // H 部の開始 index (= K 部の次元)
static bool g_k_on   = false;  // K 部に非 0 の重みがある -> K 部を計算する
static bool g_h_on   = false;  // H 部に非 0 の重みがある -> H 部を計算する

// sib_feats.py の NEIGH と同じ並び (df = 筋の差 外側、 dr = 段の差 内側、 (0,0) を除く)。
static constexpr int NEIGH_DF[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
static constexpr int NEIGH_DR[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

// 駒種 -> 0..12 (玉 / 空は -1)。 学習器 piece_index と同じ。
static inline int piece_index(PieceType pt) {
	const int v = int(pt);
	if (v >= 1 && v <= 7)  return v - 1;
	if (v >= 9 && v <= 14) return v - 2;
	return -1;
}

// 視点 p の座標へ写す (後手は 180 度回転)。 自己逆写像。
static inline Square orient(Square sq, Color p) { return p == BLACK ? sq : Inv(sq); }

// 色ごとの利き。 COUNT = false なら any (利きがある升) だけ、 true なら数 (0..7、 8 以上は sat) も持つ。
struct Effects {
	Bitboard any[COLOR_NB];
	Bitboard b0[COLOR_NB], b1[COLOR_NB], b2[COLOR_NB], sat[COLOR_NB];

	// 色 c の sq への利きの数 (8 以上は 8)。 COUNT = true で作ったときだけ使える。
	int count(Color c, Square sq) const {
		if (sat[c].test(sq)) return 8;
		return int(b0[c].test(sq)) + 2 * int(b1[c].test(sq)) + 4 * int(b2[c].test(sq));
	}
};

template <Color C, bool COUNT>
static inline void effects_of(const Position& pos, const Bitboard& occ, Effects& e) {
	Bitboard b0(ZERO), b1(ZERO), b2(ZERO), sat(ZERO), any(ZERO);
	auto add = [&](const Bitboard& a) {
		if constexpr (COUNT) {
			// ビットスライスの 1 加算 (升ごとに独立な 3 bit カウンタ + 飽和フラグ)。
			const Bitboard c0 = b0 & a;
			b0 ^= a;
			const Bitboard c1 = b1 & c0;
			b1 ^= c0;
			const Bitboard c2 = b2 & c1;
			b2 ^= c1;
			sat |= c2;
		} else {
			any |= a;
		}
	};
	// 歩はまとめて shift (別々の歩の利き先は必ず別の升 = 和集合が数そのもの)。
	add(pawnBbEffect<C>(pos.pieces(C, PAWN)));
	for (Bitboard bb = pos.pieces(C, LANCE);  bb; ) { const Square s = bb.pop(); add(lanceEffect<C>(s, occ)); }
	for (Bitboard bb = pos.pieces(C, KNIGHT); bb; ) { const Square s = bb.pop(); add(knightEffect<C>(s)); }
	for (Bitboard bb = pos.pieces(C, SILVER); bb; ) { const Square s = bb.pop(); add(silverEffect<C>(s)); }
	for (Bitboard bb = pos.pieces(C, GOLDS);  bb; ) { const Square s = bb.pop(); add(goldEffect<C>(s)); }
	// 馬・龍は 1 枚の駒として 1 回だけ数える (角 + 玉に分けると斜め/縦横の隣を 2 回数えてしまう)。
	for (Bitboard bb = pos.pieces(C, BISHOP); bb; ) { const Square s = bb.pop(); add(bishopEffect(s, occ)); }
	for (Bitboard bb = pos.pieces(C, HORSE);  bb; ) { const Square s = bb.pop(); add(horseEffect(s, occ)); }
	for (Bitboard bb = pos.pieces(C, ROOK);   bb; ) { const Square s = bb.pop(); add(rookEffect(s, occ)); }
	for (Bitboard bb = pos.pieces(C, DRAGON); bb; ) { const Square s = bb.pop(); add(dragonEffect(s, occ)); }
	for (Bitboard bb = pos.pieces(C, KING);   bb; ) { const Square s = bb.pop(); add(kingEffect(s)); }
	if constexpr (COUNT) {
		e.b0[C] = b0; e.b1[C] = b1; e.b2[C] = b2; e.sat[C] = sat;
		e.any[C] = b0 | b1 | b2 | sat;
	} else {
		e.any[C] = any;
	}
}

// 両視点の特徴 (学習器の index) を書く。 fb = 先手視点 / fw = 後手視点。 本数は nb / nw に返す。
//   want_k / want_h で部分ごとに計算するかを選ぶ (重みが全部 0 の部分は飛ばしてよい = 結果が変わらない)。
static inline void features_both(const Position& pos, int kmode, bool want_k, bool want_h,
                                 int* fb, int& nb, int* fw, int& nw) {
	nb = nw = 0;
	if (!want_k && !want_h) return;
	const Bitboard occ = pos.pieces();
	Effects e;
	if (want_k) {
		effects_of<BLACK, true>(pos, occ, e);
		effects_of<WHITE, true>(pos, occ, e);
	} else {
		effects_of<BLACK, false>(pos, occ, e);
		effects_of<WHITE, false>(pos, occ, e);
	}
	const int kdims = (kmode == 2) ? K2_DIMS : K1_DIMS;
	int* const f[COLOR_NB] = { fb, fw };
	int*       n[COLOR_NB] = { &nb, &nw };

	if (want_k && kmode == 1) {
		// 向きは視点 P から見る (玉の持ち主ではない)。
		for (Color p : { BLACK, WHITE })
			for (Color kc : { BLACK, WHITE }) {
				const Square k = pos.square<KING>(kc);
				if (k >= SQ_NB) continue;
				const int    side  = (kc == p) ? 0 : 1;
				const Color  enemy = ~kc;
				const Square ko    = orient(k, p);
				const int    kf = int(ko) / 9, kr = int(ko) % 9;
				for (int d = 0; d < 8; ++d) {
					const int tf = kf + NEIGH_DF[d], tr = kr + NEIGH_DR[d];
					if (tf < 0 || tf > 8 || tr < 0 || tr > 8) continue;
					const Square t = orient(Square(tf * 9 + tr), p);
					if (occ.test(t)) continue;
					const int e_ = e.count(enemy, t);
					f[p][(*n[p])++] = side * 32 + d * 4 + (e_ < 3 ? e_ : 3);
				}
			}
	}
	if (want_k && kmode == 2) {
		// 玉ごとに持ち主から見た向き + 区分で 1 度だけ計算し、 両視点へ side をずらして書く。
		for (Color kc : { BLACK, WHITE }) {
			const Square k = pos.square<KING>(kc);
			if (k >= SQ_NB) continue;
			const Color  enemy = ~kc;
			const Square ko    = orient(k, kc);
			const int    kf = int(ko) / 9, kr = int(ko) % 9;
			const bool   mirror = kf >= 5;   // HalfKA_hm の is_hm_mirror と同じ境界
			const int    fold   = mirror ? 8 - kf : kf;
			const int    fbk    = fold <= 1 ? 0 : (fold <= 3 ? 1 : 2);
			const int    rbk    = kr == 8 ? 0 : (kr == 7 ? 1 : 2);
			const int    region = fbk * 3 + rbk;
			int loc[8], nl = 0, za = 0;
			for (int d = 0; d < 8; ++d) {
				const int df = mirror ? -NEIGH_DF[d] : NEIGH_DF[d];
				const int tf = kf + df, tr = kr + NEIGH_DR[d];
				if (tf < 0 || tf > 8 || tr < 0 || tr > 8) continue;
				const Square t = orient(Square(tf * 9 + tr), kc);
				if (occ.test(t)) continue;
				const int e_ = e.count(enemy, t);
				za += e_;
				loc[nl++] = (region * 8 + d) * 4 + (e_ < 3 ? e_ : 3);
			}
			const int agg = region * 8 + (za < 7 ? za : 7);
			for (Color p : { BLACK, WHITE }) {
				const int side = (kc == p) ? 0 : 1;
				for (int i = 0; i < nl; ++i) f[p][(*n[p])++] = side * (K2_REGIONS * 8 * 4) + loc[i];
				f[p][(*n[p])++] = K2_DIR + side * (K2_REGIONS * 8) + agg;
			}
		}
	}
	if (want_h) {
		for (Color c : { BLACK, WHITE }) {
			// 浮き駒 = c の玉以外の駒 & 敵の利きあり & 味方の利きなし。
			Bitboard hang = pos.pieces(c) & e.any[~c] & ~(e.any[c] | pos.pieces(KING));
			while (hang) {
				const Square sq = hang.pop();
				const int    pt = piece_index(type_of(pos.piece_on(sq)));
				if (pt < 0) continue;   // 玉は除いているので来ない
				const int r = int(sq) % 9;                 // RANK_1 = 0 (先手から見た 1 段目)
				const int side_b = (c == BLACK) ? 0 : 1;   // 先手視点
				fb[nb++] = kdims + side_b * 39 + pt * 3 + r / 3;
				fw[nw++] = kdims + (1 - side_b) * 39 + pt * 3 + (8 - r) / 3;
			}
		}
	}
}

#if defined(P2_KH_PROBE)
// 計測専用 (2026-11-13): KH の区間ごとの cycle (rdtsc)。 本番ビルドには 1 命令も入らない。 終了時に stderr へ出す。
//   feat = 利き + 浮き駒の列挙 / add = copy + 行の加算 (従来版 apply の中だけ) / all = 呼び出し全体 (従来版・高速版の両方)。
struct Probe {
	std::uint64_t n = 0, cyc_all = 0, cyc_feat = 0, cyc_add = 0, n_zero = 0, n_feat = 0;
	~Probe() {
		std::fprintf(stderr, "KHPROBE n=%llu cyc_all=%llu cyc_feat=%llu cyc_add=%llu n_zero=%llu n_feat=%llu\n",
			(unsigned long long) n, (unsigned long long) cyc_all, (unsigned long long) cyc_feat,
			(unsigned long long) cyc_add, (unsigned long long) n_zero, (unsigned long long) n_feat);
	}
};
static Probe g_probe;
#endif

// out_s / out_n = threat acc (stm / nstm) + Σ KH 行。 g_loaded のときだけ呼ぶこと。
static inline void apply(const Position& pos, Color stm,
                         const std::int16_t* thr_s, const std::int16_t* thr_n,
                         std::int16_t* out_s, std::int16_t* out_n) {
	int fb[MAX_FEATS], fw[MAX_FEATS], nb, nw;
#if defined(P2_KH_PROBE)
	const std::uint64_t kp_t0 = __rdtsc();
#endif
	features_both(pos, g_kmode, g_k_on, g_h_on, fb, nb, fw, nw);
#if defined(P2_KH_PROBE)
	const std::uint64_t kp_t1 = __rdtsc();
	g_probe.cyc_feat += kp_t1 - kp_t0; g_probe.n_zero += (nb + nw == 0); g_probe.n_feat += (std::uint64_t) nb;
#endif
	const int* fs = (stm == BLACK) ? fb : fw;
	const int* fn = (stm == BLACK) ? fw : fb;
	const int  ns = (stm == BLACK) ? nb : nw;
	const int  nn = (stm == BLACK) ? nw : nb;
	std::memcpy(out_s, thr_s, sizeof(std::int16_t) * THREAT_ACC_DIM);
	std::memcpy(out_n, thr_n, sizeof(std::int16_t) * THREAT_ACC_DIM);
	// i16 の wrapping 加算 (threat acc と FT acc の合算も i16 wrapping なので同じ扱い)。 足す順序は結果に影響しない。
	for (int i = 0; i < ns; ++i) {
		const std::int16_t* w = g_w[fs[i]];
		for (int k = 0; k < THREAT_ACC_DIM; ++k)
			out_s[k] = (std::int16_t) (std::uint16_t) ((std::uint16_t) out_s[k] + (std::uint16_t) w[k]);
	}
	for (int i = 0; i < nn; ++i) {
		const std::int16_t* w = g_w[fn[i]];
		for (int k = 0; k < THREAT_ACC_DIM; ++k)
			out_n[k] = (std::int16_t) (std::uint16_t) ((std::uint16_t) out_n[k] + (std::uint16_t) w[k]);
	}
#if defined(P2_KH_PROBE)
	g_probe.cyc_add += __rdtsc() - kp_t1;
#endif
}

#if defined(P2_KH_FAST)
#include "p2nnue_kh_fast.h"   // apply_fast / fast::init / fast_selfcheck (2026-11-13)
#endif

// json から `"key": <整数>` を取る。 無ければ false。
static inline bool json_int(const std::string& js, const char* key, long long& v) {
	const std::string pat = std::string("\"") + key + "\"";
	const size_t p = js.find(pat);
	if (p == std::string::npos) return false;
	size_t i = js.find(':', p + pat.size());
	if (i == std::string::npos) return false;
	++i;
	while (i < js.size() && (js[i] == ' ' || js[i] == '\t' || js[i] == '\r' || js[i] == '\n')) ++i;
	bool neg = false;
	if (i < js.size() && js[i] == '-') { neg = true; ++i; }
	if (i >= js.size() || js[i] < '0' || js[i] > '9') return false;
	long long x = 0;
	while (i < js.size() && js[i] >= '0' && js[i] <= '9') { x = x * 10 + (js[i] - '0'); ++i; }
	v = neg ? -x : x;
	return true;
}

// json の `"key": [ ... ]` の中の整数 (文字列の中は除く) を順に取る。 無ければ false。
static inline bool json_ints_in_array(const std::string& js, const char* key, std::vector<long long>& out) {
	out.clear();
	const std::string pat = std::string("\"") + key + "\"";
	const size_t p = js.find(pat);
	if (p == std::string::npos) return false;
	size_t i = js.find('[', p + pat.size());
	if (i == std::string::npos) return false;
	int depth = 0;
	bool in_str = false;
	for (; i < js.size(); ++i) {
		const char ch = js[i];
		if (in_str) {
			if (ch == '\\') ++i;
			else if (ch == '"') in_str = false;
			continue;
		}
		if (ch == '"') { in_str = true; continue; }
		if (ch == '[') { ++depth; continue; }
		if (ch == ']') { if (--depth == 0) return true; continue; }
		if (ch >= '0' && ch <= '9') {
			long long x = 0;
			while (i < js.size() && js[i] >= '0' && js[i] <= '9') { x = x * 10 + (js[i] - '0'); ++i; }
			--i;
			out.push_back(x);
		}
	}
	return false;
}

// 読めなければ理由を返す (空文字 = 成功)。
static inline std::string load_impl(const std::string& path) {
	if (!g_net || !g_net->loaded) return "net が読まれていない";
	if (!g_net->has_threat)       return "net に threat が無い (KH は threat acc に足すので threat net 専用)";
	if (g_net->has_aux)           return "aux 付き net は未対応 (aux 経路に KH を配線していない)";

	const std::string jpath = path + ".json";
	std::ifstream jf(jpath, std::ios::binary);
	if (!jf) return "json が開けない: " + jpath + " (学習器の --kh-dump は <path>.json を書く)";
	const std::string js((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());
	long long out = 0, dims = 0, kd = 0, hd = 0, off = 0, kmode = 1;
	if (!json_int(js, "out", out) || !json_int(js, "dims", dims) || !json_int(js, "k_dims", kd)
	    || !json_int(js, "h_dims", hd) || !json_int(js, "row_offset", off))
		return "json に out / dims / k_dims / h_dims / row_offset のどれかが無い: " + jpath;
	if (js.find("values[feature * out + row]") == std::string::npos)
		return "json の layout が column-major (values[feature * out + row]) でない";
	const bool has_kmode = json_int(js, "kmode", kmode);   // 学習器は kmode 1 のとき鍵を書かない
	if (js.find("\"kmode\"") != std::string::npos && !has_kmode)
		return "json の kmode が整数として読めない";
	if (kmode == 1) {
		if (dims != K1_DIMS + H_DIMS || kd != K1_DIMS || hd != H_DIMS)
			return "kmode 1 なのに json の dims/k_dims/h_dims が 142/64/78 でない";
	} else if (kmode == 2) {
		if (dims != K2_DIMS + H_DIMS || kd != K2_DIMS || hd != H_DIMS)
			return "kmode 2 なのに json の dims/k_dims/h_dims が 798/720/78 でない";
		std::vector<long long> ks;
		if (!json_ints_in_array(js, "k_sub", ks))
			return "kmode 2 なのに json に k_sub が無い";
		if (ks != std::vector<long long>{ 0, K2_DIR, K2_DIR, K2_DIMS })
			return "json の k_sub が [[方向, 0, 576], [集計, 576, 720]] でない";
	} else {
		return "json の kmode (" + std::to_string(kmode) + ") は 1 / 2 のどちらでもない";
	}
	if (out != THREAT_ACC_DIM)
		return "json の out (" + std::to_string(out) + ") が threat acc の幅 (" + std::to_string(THREAT_ACC_DIM) + ") と違う";
	if (off != 0)
		return "json の row_offset が 0 でない (threat acc と行が揃わない)";

	std::ifstream ifs(path, std::ios::binary);
	if (!ifs) return "開けない: " + path;
	std::vector<char> buf((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	const size_t need = (size_t) out * dims * sizeof(float);
	if (buf.size() != need)
		return "サイズが違う: " + std::to_string(buf.size()) + " byte (期待 " + std::to_string(need) + ")";
	std::vector<float> v((size_t) out * dims);
	std::memcpy(v.data(), buf.data(), need);   // x86 = LE

	static std::int16_t tmp[MAX_DIMS][THREAT_ACC_DIM];
	std::memset(tmp, 0, sizeof(tmp));
	double max_abs = 0.0;
	size_t k_nonzero = 0, h_nonzero = 0;
	for (int c = 0; c < (int) dims; ++c)
		for (int r = 0; r < out; ++r) {
			const float w = v[(size_t) c * out + r];
			// NaN / inf の検査はビット列で行う (-ffast-math では w != w が消されうる)。
			std::uint32_t bits; std::memcpy(&bits, &w, 4);
			if (((bits >> 23) & 0xFF) == 0xFF) return "NaN / inf が入っている (列 " + std::to_string(c) + ")";
			const double x = double(QA_KH) * double(w);
			const double q = (x >= 0.0) ? (double) (long long) (x + 0.5) : -(double) (long long) (-x + 0.5);  // Rust round = 0 から遠い方
			if (q > 32767.0 || q < -32768.0) return "量子化後に i16 を超える (列 " + std::to_string(c) + ")";
			if ((q < 0 ? -q : q) > max_abs) max_abs = (q < 0 ? -q : q);
			tmp[c][r] = (std::int16_t) q;
			// 部分の on/off は量子化後の値で決める (round で 0 になる重みは足しても何も変わらない)。
			if (tmp[c][r] != 0) ++(c < kd ? k_nonzero : h_nonzero);
		}
	std::memcpy(g_w, tmp, sizeof(g_w));
	g_kmode = (int) kmode;
	g_kdims = (int) kd;
	g_k_on  = k_nonzero != 0;
	g_h_on  = h_nonzero != 0;
	sync_cout << "info string KH: loaded " << path << " (kmode " << kmode << " / K 部 " << kd << " 列 非0 " << k_nonzero
	          << (g_k_on ? "" : " = 計算しない") << " / H 部 " << hd << " 列 非0 " << h_nonzero
	          << (g_h_on ? "" : " = 計算しない") << " / out " << out << ", QA " << QA_KH
	          << ", max|q| " << (long long) max_abs << ")" << sync_endl;
	return std::string();
}

// load_eval から呼ぶ。 path 空なら無効。 失敗したら **エンジンを止める** (黙って KH 無しで走らせない)。
static inline void load(const std::string& path) {
	g_loaded = false;
	if (path.empty()) return;
	const std::string err = load_impl(path);
	if (!err.empty()) {
		sync_cout << "info string Error! KHFile: " << err << sync_endl;
		std::fprintf(stderr, "Error! KHFile: %s\n", err.c_str());
		std::exit(EXIT_FAILURE);
	}
#if defined(P2_KH_FAST)
	fast::init();   // g_kdims が決まってから行番号の表を作る
	sync_cout << "info string KH: fast path (P2_KH_FAST)" << sync_endl;
#endif
	g_loaded = true;
}

// eval コマンド (dump) 用: 両視点の特徴 (学習器の index、 昇順。 K 部も H 部も重みに関係なく全部) と、
//   threat acc への加算量 (KHFile が読まれているときだけ) を出す。 KHFile 無しなら kmode 1 の並びで出す。
static inline void dump(const Position& pos, const std::int16_t* thr_s, const std::int16_t* thr_n) {
	const Color stm = pos.side_to_move();
	int fb[MAX_FEATS], fw[MAX_FEATS], nb, nw;
	features_both(pos, g_loaded ? g_kmode : 1, true, true, fb, nb, fw, nw);
	const int* f[2] = { (stm == BLACK) ? fb : fw, (stm == BLACK) ? fw : fb };
	const int  n[2] = { (stm == BLACK) ? nb : nw, (stm == BLACK) ? nw : nb };
	alignas(32) std::int16_t os[THREAT_ACC_DIM], on[THREAT_ACC_DIM];
	if (g_loaded) apply(pos, stm, thr_s, thr_n, os, on);
	for (int p = 0; p < 2; ++p) {
		int s[MAX_FEATS];
		for (int i = 0; i < n[p]; ++i) s[i] = f[p][i];
		std::sort(s, s + n[p]);
		std::string line = std::string("info string KH ") + (p == 0 ? "stm" : "nstm") + " feats=";
		for (int i = 0; i < n[p]; ++i) { if (i) line += ","; line += std::to_string(s[i]); }
		if (g_loaded) {
			const std::int16_t* o = p == 0 ? os : on;
			const std::int16_t* t = p == 0 ? thr_s : thr_n;
			line += " delta=";
			for (int k = 0; k < THREAT_ACC_DIM; ++k) {
				if (k) line += ",";
				line += std::to_string((int) (std::int16_t) (std::uint16_t) ((std::uint16_t) o[k] - (std::uint16_t) t[k]));
			}
		}
		sync_cout << line << sync_endl;
	}
#if defined(P2_KH_FAST)
	// 高速版と従来版の出力 (threat acc + Σ KH 行) を同じ局面で突き合わせる。 NG が 1 つでも出たら bit-exact でない。
	if (g_loaded)
		sync_cout << "info string KHFAST selfcheck " << (fast_selfcheck(pos, stm, thr_s, thr_n) ? "OK" : "NG") << sync_endl;
#endif
}

} // namespace kh

#endif // P2_KH
