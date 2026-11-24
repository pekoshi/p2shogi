// **threat の多段 lazy 差分 (LZ)** — 本番実装。 2026-11-06
//   設計 fable (runs/prereg/DESIGN-20261106-THREAT-LZ.md) / 実装 opus / 参照 source_fable_lz の probe。
//
// ■ 何を直すのか
//   本番の eval の **99.2%** が `ensure_threat_both` の **full 再計算** (add_threat_dual) で、
//   玉以外の**全駒**を attacker として走査していた (評価 1 回あたり攻め駒 30.7 枚 / 行 61.2 本)。
//   本当の差分は **1.78 枚/手** なので、 94% が無駄な走査だった。
//   -> 評価時に acc が valid な最寄りの祖先まで最大 DMAX 手遡り、 盤を巻き戻して 1 手ずつ Δ を作り、
//      途中の祖先にも acc を**物質化**する (次の兄弟が d=1 で済むので適用率が 43% -> 85% になる)。
//
// ■ 恒等式 (設計書 §2 / プローブで 0/482,594 不一致を確認済み)
//   Δ = −pairs_P(p@A) −pairs_P(c@B) +pairs_C(p@B)
//       + Σ_{X∈Att} [pairs_C(X)∩M − pairs_P(X)∩M]
//   Att = attackers_to(A, occ_min) ∪ attackers_to(B, occ_min) − {A,B},  occ_min = occ_P ∩ occ_C
//   M(飛び駒) = (attacks_P ⊕ attacks_C) ∪ {A,B},  M(歩み駒) = {A,B}
//   🎯飛び駒のレイ恒等式: 「レイ上の occupancy が変わった ⇔ occ_min で A/B に届く」なので between_bb 不要。
//
// ■ 守るべき不変条件 (設計書 §3.4)
//   1. valid[c] == true ⇒ acc[c] は「その局面 / 視点 c / 鏡像 = is_hm_mirror(その局面の玉) / 現在の群」のもの
//   2. base の採用は **群一致 ∧ base の鏡像 (玉を巻き戻して評価) == 現在の鏡像**
//   3. do_move / do_null_move が valid=false を落とす既存コードは据え置き
//   4. `ensure_progress` が走る node の集合を変えない (bucket_hint を受け取るだけ / 10-07 S0 の教訓)
//   5. 行の適用は full と同じ `apply_threat_rows` (順序非依存なので add/sub の順は自由)
//   6. 遡りの上限 DMAX、 対の上限 256/視点 (超えたら FULL) — 値を変えず費用だけ抑える安全弁
//   7. multiset(ADD) − multiset(SUB) == pairs(C) − pairs(P)。 attacker/target とも玉を含まない
//
// 🚨既定は off。 `-DP2_THREAT_LZ` で有効化する。 profile 0 専用。
#if defined(P2_THREAT_LZ)
static_assert(Threat::THREAT_PROFILE_ID == 0, "LZ assumes profile 0");

namespace lz {

// 2026-11-11 (N0 の結果): d>=3 の差分は full より高い (d=3 で build+apply 4,248 対 full 3,425 cycle)。
//   `-DP2_LZ_DMAX=<n>` で振れるようにした (既定 8 = 従来どおり)。 値を変えても評価値は不変 (full に落ちるだけ)。
#if !defined(P2_LZ_DMAX)
#define P2_LZ_DMAX 8
#endif
static constexpr int DMAX = P2_LZ_DMAX;
// 2026-11-16 (THREATINC 段 3 / A4): 対の上限 (1 視点あたり) を `-DP2_LZ_STEPCAP=<n>` で振れるようにした。
//   既定 256 = 従来どおり。 小さくすると Step (int x 4 x n) が縮み resolve のフレームが 4KB を切って
//   ___chkstk_ms が消える。 超えたら ovf -> full に落ちるだけなので評価値は不変。
#if !defined(P2_LZ_STEPCAP)
#define P2_LZ_STEPCAP 256
#endif

#if defined(P2_FABLE_PROBE)
#include <x86intrin.h>
// 2026-11-11 (N0 / LZSPLIT): LZ 経路の cycle 分解。 設計 = runs/prereg/PREREG-20261111-LZSPLIT.md。
//   🚨P2_FABLE_PROBE の中だけなので本番ビルドには 1 命令も入らない。
//   env `P2_LZ_PROBE_WARM=1` で、 適用する全行を計時区間の外で 1 度読む (値は変えない = 探索木は同じ)。
//   apply(cold) − apply(warm) = 行の待ち = 案 A (行 prefetch) の天井。
namespace probe {
static std::uint64_t n_call = 0, n_ok = 0, f_root = 0, f_dmax = 0, f_mir = 0, f_ovf = 0;
static std::uint64_t cyc_walk = 0, cyc_build = 0, cyc_apply = 0, cyc_fail = 0;
static std::uint64_t cyc_full = 0, n_full = 0, n_one = 0, n_early = 0;
static std::uint64_t rows = 0;
static std::uint64_t n_d[DMAX + 1] = {}, cb_d[DMAX + 1] = {}, ca_d[DMAX + 1] = {}, rows_d[DMAX + 1] = {};
static std::uint64_t far_h[5] = {};   // DMAX 失敗時の真の距離: 9-12 / 13-16 / 17-32 / 33-64 / 無し(root か 64 超)
// 2026-11-16 (THREATINC M0 / 設計 DESIGN-20261116-THREATINC-FABLE.md §1.4):
//   n_none_grp = 「64 手遡っても無い」のうち、 root が両視点 valid で群だけ違う回数 (群またぎの読みの検証)
//   n_none_sib = そのうち、 同じ親 (StateInfo の番地 + key + 群) で 2 回目以降だった回数 (案 B の天井)
//   n_nul_d1   = 差分成功のうち d=1 かつ null move の子だった回数
static std::uint64_t n_none_grp = 0, n_none_sib = 0, n_nul_d1 = 0;
static const void* sib_ptr[1024] = {};
static std::uint64_t sib_key[1024] = {};
static std::uint8_t sib_grp[1024] = {};
static std::uint64_t warm_sink = 0;
static bool warm() {
	static const bool v = [] { const char* e = std::getenv("P2_LZ_PROBE_WARM"); return e && *e && *e != '0'; }();
	return v;
}
// 2026-11-11 (build の内訳): env `P2_LZ_PROBE_EMIT2=1` で emit の index 計算をもう 1 回 (noinline) 行う
//   -> build の増分 = index 計算の費用 (+ 呼び出し 1 回分)。
//   env `P2_LZ_PROBE_BUILD2=1` で build_step を捨て用の Step にもう 1 回行う -> 増分 = build_step 全体
//   (列挙 + index + 分岐予測ミス)。 差 = 列挙 (bitboard と pop ループ) の費用。 どちらも値は変えない。
static std::uint64_t sink2 = 0;
static bool emit2() {
	static const bool v = [] { const char* e = std::getenv("P2_LZ_PROBE_EMIT2"); return e && *e && *e != '0'; }();
	return v;
}
static bool build2() {
	static const bool v = [] { const char* e = std::getenv("P2_LZ_PROBE_BUILD2"); return e && *e && *e != '0'; }();
	return v;
}
static void report() {
	std::fprintf(stderr, "LZPROBE n_call=%llu n_ok=%llu f_root=%llu f_dmax=%llu f_mir=%llu f_ovf=%llu "
		"cyc_walk=%llu cyc_build=%llu cyc_apply=%llu cyc_fail=%llu cyc_full=%llu n_full=%llu n_one=%llu n_early=%llu rows=%llu warm=%d\n",
		(unsigned long long)n_call, (unsigned long long)n_ok, (unsigned long long)f_root, (unsigned long long)f_dmax,
		(unsigned long long)f_mir, (unsigned long long)f_ovf, (unsigned long long)cyc_walk, (unsigned long long)cyc_build,
		(unsigned long long)cyc_apply, (unsigned long long)cyc_fail, (unsigned long long)cyc_full, (unsigned long long)n_full,
		(unsigned long long)n_one, (unsigned long long)n_early, (unsigned long long)rows, warm() ? 1 : 0);
	for (int d = 1; d <= DMAX; ++d)
		std::fprintf(stderr, "LZPROBE_D d=%d n=%llu cyc_build=%llu cyc_apply=%llu rows=%llu\n", d,
			(unsigned long long)n_d[d], (unsigned long long)cb_d[d], (unsigned long long)ca_d[d], (unsigned long long)rows_d[d]);
	std::fprintf(stderr, "LZPROBE_FAR d9_12=%llu d13_16=%llu d17_32=%llu d33_64=%llu none=%llu sink=%llu sink2=%llu\n",
		(unsigned long long)far_h[0], (unsigned long long)far_h[1], (unsigned long long)far_h[2],
		(unsigned long long)far_h[3], (unsigned long long)far_h[4], (unsigned long long)(warm_sink & 1),
		(unsigned long long)(sink2 & 1));
	std::fprintf(stderr, "LZPROBE_M0 none_grp=%llu none_sib=%llu nul_d1=%llu\n",
		(unsigned long long)n_none_grp, (unsigned long long)n_none_sib, (unsigned long long)n_nul_d1);
}
static struct Init { Init() { std::atexit(report); } } init_;
} // namespace probe
#endif
// Attack set of threat class ac (0..8) of colour c on sq. Non-sliders: empty-board table
// (== real attacks). Sliders: magic. Same sets as the production enumeration.
static inline Bitboard attacks_of(int ac, Color c, Square sq, const Bitboard& occ) {
	const bool slider = (ac == 1 || ac >= 5);
	return slider ? direct15_slider_effect(ac, c, sq, occ) : g_direct15_pseudo[c][ac][(int)sq];
}

// Class-combo bitboards used by att_to: 0 P, 1 N, 2 SILVER_HDK, 3 GOLDS_HDK, 4 BISHOP_HORSE, 5 ROOK_DRAGON, 6 LANCE
static inline std::uint8_t combo_of(PieceType pt) {
	switch (pt) {
	case PAWN:   return 1u << 0;
	case LANCE:  return 1u << 6;
	case KNIGHT: return 1u << 1;
	case SILVER: return 1u << 2;
	case GOLD: case PRO_PAWN: case PRO_LANCE: case PRO_KNIGHT: case PRO_SILVER: return 1u << 3;
	case BISHOP: return 1u << 4;
	case ROOK:   return 1u << 5;
	case HORSE:  return (1u << 2) | (1u << 3) | (1u << 4);
	case DRAGON: return (1u << 2) | (1u << 3) | (1u << 5);
	case KING:   return (1u << 2) | (1u << 3);
	default:     return 0;
	}
}

// Board view backed by the live Position (1-ply modes).
struct PosView {
	const Position& pos;
	Piece    piece_on(Square s) const { return pos.piece_on(s); }
	Bitboard occ() const { return pos.pieces(); }
	Bitboard bb(Color c, int combo) const {
		switch (combo) {
		case 0: return pos.pieces(c, PAWN);
		case 1: return pos.pieces(c, KNIGHT);
		case 2: return pos.pieces(SILVER_HDK) & pos.pieces(c);
		case 3: return pos.pieces(GOLDS_HDK) & pos.pieces(c);
		case 4: return pos.pieces(BISHOP_HORSE) & pos.pieces(c);
		case 5: return pos.pieces(ROOK_DRAGON) & pos.pieces(c);
		default: return pos.pieces(c, LANCE);
		}
	}
};

// Scratch board view for multi-ply reconstruction (mode 3).
struct ScratchView {
	Piece    bd[SQ_NB];
	Bitboard o;
	Bitboard cb[COLOR_NB][7];
	void init(const Position& pos) {
		for (int s = 0; s < SQ_NB; ++s) bd[s] = pos.piece_on((Square)s);
		o = pos.pieces();
		PosView pv{pos};
		for (int c = 0; c < COLOR_NB; ++c) for (int i = 0; i < 7; ++i) cb[c][i] = pv.bb((Color)c, i);
	}
	Piece    piece_on(Square s) const { return bd[(int)s]; }
	Bitboard occ() const { return o; }
	Bitboard bb(Color c, int combo) const { return cb[c][combo]; }
	void toggle(Piece pc, Square s) {
		const std::uint8_t m = combo_of(type_of(pc));
		const Color c = color_of(pc);
		for (int i = 0; i < 7; ++i) if (m & (1u << i)) cb[c][i] ^= Bitboard(s);
	}
	// undo move m (child -> parent): the piece now on B goes back to A as lmpt; cap returns to B.
	void undo(Move m, Piece cap, PieceType lmpt) {
		const Square B = m.to_sq();
		const Piece  p_after = bd[(int)B];
		toggle(p_after, B);
		bd[(int)B] = cap;
		if (cap == NO_PIECE) o ^= Bitboard(B); else toggle(cap, B);
		if (!m.is_drop()) {
			const Square A = m.from_sq();
			const Piece  p_before = make_piece(color_of(p_after), lmpt);
			bd[(int)A] = p_before;
			o |= Bitboard(A);
			toggle(p_before, A);
		}
	}
};

// Emitted index pairs (idx_b, idx_w) for SUB (parent only) and ADD (child only).
// 対の索引を溜める器。 ply ごとの区間境界を持つ (物質化の forward 積みに使う)。
//   ply i の寄与は sub[seg_s[i], seg_s[i+1]) / add[seg_a[i], seg_a[i+1])。
//   🚨build_step を呼ぶ**前**に mark(i) を打つこと。 最後に mark(d) で終端を打つ。
struct Step {
	int sub_b[P2_LZ_STEPCAP], sub_w[P2_LZ_STEPCAP], add_b[P2_LZ_STEPCAP], add_w[P2_LZ_STEPCAP];
	int ns = 0, na = 0;
	bool ovf = false;
	int seg_s[DMAX + 1] = {};
	int seg_a[DMAX + 1] = {};
	void mark(int i) { seg_s[i] = ns; seg_a[i] = na; }
#if defined(P2_LZ_ROWPF)
	// 2026-11-11 (案 A): 現在の群の threat 表の先頭 (バイト) と 1 行のバイト数。 emit が行を prefetch するのに使う。
	//   🚨本番 net (top3q-102) の threat 表は **i8** (1 行 64B = 1 ライン)。 i16 net なら 128B = 2 ライン。
	const char* pf = nullptr;
	std::size_t pf_row = 64;
#endif
};
// threat 行の先頭アドレス (i8 / i16 どちらの net でも)。 probe と案 A で使う。
static inline const char* lz_row_ptr(const Net& n, std::size_t tf) {
	return n.threat_is_i16 ? reinterpret_cast<const char*>(threat_row_i16(n, tf))
	                       : reinterpret_cast<const char*>(threat_row(n, tf));
}
#if defined(P2_LZ_ROWPF)
// 案 A: threat 行の prefetch。 値は変えない (bit-exact)。
//   P2_LZ_ROWPF=1: emit 時に T0 / =2: emit 時に T1 (L2 まで。 L1 を汚さない) / =3: build_chain 後にまとめて T0 (対照)
//   行が 128B (i16 net) なら 2 ライン目も出す。
static inline void lz_pf_row(const char* base, std::size_t row, int idx, int hint_t1) {
	const char* p = base + (std::size_t)idx * row;
	if (hint_t1) { _mm_prefetch(p, _MM_HINT_T1); if (row > 64) _mm_prefetch(p + 64, _MM_HINT_T1); }
	else         { _mm_prefetch(p, _MM_HINT_T0); if (row > 64) _mm_prefetch(p + 64, _MM_HINT_T0); }
}
#endif
#if defined(P2_FABLE_PROBE)
// emit と同じ index 計算 (probe の EMIT2 用 / noinline で CSE させない)。
static __attribute__((noinline)) int emit_idx_again(Piece apc, Square X, Piece tp, Square T,
                                                   const std::uint8_t* nrm_b, const std::uint8_t* nrm_w) {
	const int ac = Threat::tc_from_pt(type_of(apc));
	const int dc = Threat::tc_from_pt(type_of(tp));
	if (dc < 0) return 0;
	const Color acol = color_of(apc);
	const int   dsb  = (color_of(tp) == BLACK) ? 0 : 1;
	const int   patb = Threat::attack_pattern_id(ac, acol);
	const int   patw = Threat::attack_pattern_id(ac, ~acol);
	const int   fnb  = (int)nrm_b[(int)X];
	const int   fnw  = (int)nrm_w[(int)X];
	const int* const pbb = &g_threat_tables.pair_base[(acol == BLACK ? 0 : 162) + ac * 18];
	const int* const pbw = &g_threat_tables.pair_base[(acol == WHITE ? 0 : 162) + ac * 18];
	const int ib = pbb[dsb * 9 + dc]       + g_threat_tables.from_offset[patb][fnb] + (int)g_threat_tables.attack_order[patb][fnb][nrm_b[(int)T]];
	const int iw = pbw[(1 - dsb) * 9 + dc] + g_threat_tables.from_offset[patw][fnw] + (int)g_threat_tables.attack_order[patw][fnw][nrm_w[(int)T]];
	return ib + iw;
}
#endif
static inline void emit(Step& s, bool add, Piece apc, Square X, Piece tp, Square T,
                        const std::uint8_t* nrm_b, const std::uint8_t* nrm_w) {
#if defined(P2_FABLE_PROBE)
	if (probe::emit2()) probe::sink2 += (std::uint64_t)emit_idx_again(apc, X, tp, T, nrm_b, nrm_w);
#endif
	const int ac = Threat::tc_from_pt(type_of(apc));
	const int dc = Threat::tc_from_pt(type_of(tp));
	if (dc < 0) return;                       // king target: not a feature
	const Color acol = color_of(apc);
	const int   dsb  = (color_of(tp) == BLACK) ? 0 : 1;
	const int   patb = Threat::attack_pattern_id(ac, acol);
	const int   patw = Threat::attack_pattern_id(ac, ~acol);
	const int   fnb  = (int)nrm_b[(int)X];
	const int   fnw  = (int)nrm_w[(int)X];
	const int* const pbb = &g_threat_tables.pair_base[(acol == BLACK ? 0 : 162) + ac * 18];
	const int* const pbw = &g_threat_tables.pair_base[(acol == WHITE ? 0 : 162) + ac * 18];
	const int ib = pbb[dsb * 9 + dc]       + g_threat_tables.from_offset[patb][fnb] + (int)g_threat_tables.attack_order[patb][fnb][nrm_b[(int)T]];
	const int iw = pbw[(1 - dsb) * 9 + dc] + g_threat_tables.from_offset[patw][fnw] + (int)g_threat_tables.attack_order[patw][fnw][nrm_w[(int)T]];
#if defined(P2_LZ_ROWPF) && (P2_LZ_ROWPF == 1 || P2_LZ_ROWPF == 2)
	lz_pf_row(s.pf, s.pf_row, ib, P2_LZ_ROWPF == 2);
	lz_pf_row(s.pf, s.pf_row, iw, P2_LZ_ROWPF == 2);
#endif
	if (add) {
		if (s.na >= P2_LZ_STEPCAP) { s.ovf = true; return; }
		s.add_b[s.na] = ib; s.add_w[s.na] = iw; ++s.na;
	} else {
		if (s.ns >= P2_LZ_STEPCAP) { s.ovf = true; return; }
		s.sub_b[s.ns] = ib; s.sub_w[s.ns] = iw; ++s.ns;
	}
}

#if defined(P2_LZ_IDXV2)
// ===== 2026-11-11 IDXV2: 索引計算を攻め駒単位に hoist する =====
//   根拠 (実測 runs/prereg/VERDICT-LZSPLIT-RERUN-20261111.log): build 1,568 cycle/eval のうち
//   emit の index 計算が約 589。 emit は 1 対ごとに攻め駒 X 側 (ac / 色 / pattern / 正規化 from /
//   pair_base 行 / from_offset / attack_order 行) を引き直していた。 X は同じ build_step の中で
//   何度も現れる (emit_all / 飛び駒 2 ループ / 歩み駒 2 ループ) ので、 X 側は AttCtx に 1 回だけ作り、
//   受け駒 T ごとには「列 (dsb*9+dc)」「nrm[T]」「order」だけを足す。
//   P2_LZ_IDXV2=1: hoist のみ (order = attack_order の行ポインタ ordb[nrm_b[T]])
//   P2_LZ_IDXV2=2: hoist + order を popcount で出す (attack_order 92KB を引かない)
//   🚨値は従来の emit と 1 bit も違わないこと。 `-DP2_LZ_IDXV2_VERIFY` で全対を従来式と照合する。
#if P2_LZ_IDXV2 != 1 && P2_LZ_IDXV2 != 2
#error "P2_LZ_IDXV2 must be 1 (hoist) or 2 (hoist + popcount)"
#endif

// 受け駒 (Piece) -> pair_base の列。 kb = dsb*9+dc (先手視点) / kw = (1-dsb)*9+dc (後手視点)。
//   玉・空・未使用の駒番号は -1 (= 対にしない。 従来 emit の `dc < 0` と同じ)。
struct TgtKey { std::int8_t kb[32]; std::int8_t kw[32]; };
static_assert(PIECE_NB <= 32, "TgtKey の大きさ");
static const TgtKey g_tgt_key = [] {
	TgtKey t{};
	for (int pc = 0; pc < 32; ++pc) {
		const int dc = (pc == NO_PIECE || pc >= PIECE_NB) ? -1 : Threat::tc_from_pt(type_of((Piece)pc));
		if (dc < 0) { t.kb[pc] = t.kw[pc] = -1; continue; }
		const int dsb = (color_of((Piece)pc) == BLACK) ? 0 : 1;
		t.kb[pc] = (std::int8_t)(dsb * 9 + dc);
		t.kw[pc] = (std::int8_t)((1 - dsb) * 9 + dc);
	}
	return t;
}();

#if P2_LZ_IDXV2 == 2
// 正規化空間 (bullet raw = YO Square の int 0..80 と同じ番号) での空盤の利き升の集合。
//   🚨YO の Bitboard (p[0]/p[1] の割り付け) は使わない。 bit i = 正規化升 i の素の 128bit (lo = 0..63 / hi = 64..80)。
//   attack_order[pat][f][t] は build_pattern で「attacks_empty_board(tc, 色, f) を raw 昇順に並べたときの t の順位」
//   (p2nnue_threat.h build_pattern / attacks_empty_board の挿入ソート)。 ∴ t が集合に属するとき
//   attack_order[pat][f][t] == popcount(集合 & {升番号 < t})。 列挙する対の T は実際の利き升 ⊆ 空盤の利き升
//   (従来式もこれを前提にしていて、 外れれば 255 になる) なので常に成り立つ。 VERIFY で全対を照合する。
struct PatMask {
	std::uint64_t lo[Threat::NUM_ATTACK_PATTERNS][81];
	std::uint64_t hi[Threat::NUM_ATTACK_PATTERNS][81];
	std::uint64_t blo[81];   // {升番号 < t} の lo 側
	std::uint64_t bhi[81];   // 同 hi 側
};
static const PatMask g_pat_mask = [] {
	PatMask m{};
	std::uint8_t targets[36];
	auto put = [&](int tc, Color c, int pat) {
		for (int f = 0; f < 81; ++f) {
			const int cnt = Threat::attacks_empty_board(tc, c, f, targets);
			for (int i = 0; i < cnt; ++i) {
				const int t = targets[i];
				if (t < 64) m.lo[pat][f] |= 1ull << t; else m.hi[pat][f] |= 1ull << (t - 64);
			}
		}
	};
	// 組み方は Tables::build と同じ (先手 pattern = tc / 後手の方向性駒 = 9 + tc)
	for (int tc = 0; tc < Threat::NUM_THREAT_CLASSES; ++tc) {
		put(tc, BLACK, tc);
		if (Threat::is_directional(tc)) put(tc, WHITE, Threat::NUM_THREAT_CLASSES + tc);
	}
	for (int t = 0; t < 81; ++t) {
		m.blo[t] = (t >= 64) ? ~0ull : ((1ull << t) - 1);
		m.bhi[t] = (t >= 64) ? ((1ull << (t - 64)) - 1) : 0;
	}
	return m;
}();
#endif

// 攻め駒 X 側の前計算 (攻め駒 1 枚につき 1 回)。
struct AttCtx {
	const int* pbb;          // pair_base の行 (先手視点 / 攻め駒の色と ac で決まる 18 列)
	const int* pbw;          // 同 後手視点
	int        fob, fow;     // from_offset[pat][fn]
#if P2_LZ_IDXV2 == 1
	const std::uint8_t* ordb;   // &attack_order[patb][fnb][0] (81 バイト)
	const std::uint8_t* ordw;
#else
	std::uint64_t mlo_b, mhi_b, mlo_w, mhi_w;   // 正規化空間の空盤の利き升集合
#endif
#if defined(P2_FABLE_PROBE) || defined(P2_LZ_IDXV2_VERIFY)
	Piece  apc;              // probe (EMIT2) と検算が従来式を呼ぶのに使う
	Square X;
#endif
};
// 前提: ac = tc_from_pt(type_of(apc)) >= 0 (呼び元で玉を弾いてある)。
static inline void make_ctx(AttCtx& c, Piece apc, int ac, Square X,
                            const std::uint8_t* nrm_b, const std::uint8_t* nrm_w) {
	const Color acol = color_of(apc);
	const int   patb = Threat::attack_pattern_id(ac, acol);
	const int   patw = Threat::attack_pattern_id(ac, ~acol);
	const int   fnb  = (int)nrm_b[(int)X];
	const int   fnw  = (int)nrm_w[(int)X];
	c.pbb = &g_threat_tables.pair_base[(acol == BLACK ? 0 : 162) + ac * 18];
	c.pbw = &g_threat_tables.pair_base[(acol == WHITE ? 0 : 162) + ac * 18];
	c.fob = g_threat_tables.from_offset[patb][fnb];
	c.fow = g_threat_tables.from_offset[patw][fnw];
#if P2_LZ_IDXV2 == 1
	c.ordb = &g_threat_tables.attack_order[patb][fnb][0];
	c.ordw = &g_threat_tables.attack_order[patw][fnw][0];
#else
	c.mlo_b = g_pat_mask.lo[patb][fnb]; c.mhi_b = g_pat_mask.hi[patb][fnb];
	c.mlo_w = g_pat_mask.lo[patw][fnw]; c.mhi_w = g_pat_mask.hi[patw][fnw];
#endif
#if defined(P2_FABLE_PROBE) || defined(P2_LZ_IDXV2_VERIFY)
	c.apc = apc; c.X = X;
#endif
}

#if defined(P2_LZ_IDXV2_VERIFY)
// 検算: 照合した対の数と不一致の数を終了時に stderr へ。 不一致は 1 件目で stderr に出して abort する。
namespace idxv2_verify {
static std::uint64_t n_pair = 0, n_bad = 0;
static void report() {
	std::fprintf(stderr, "IDXV2_VERIFY mode=%d pairs=%llu mismatch=%llu\n", (int)P2_LZ_IDXV2,
		(unsigned long long)n_pair, (unsigned long long)n_bad);
}
static struct Init { Init() { std::atexit(report); } } init_;
// 従来の emit と同じ式 (noinline で新しい式と CSE させない)。
static __attribute__((noinline)) void legacy(Piece apc, Square X, Piece tp, Square T,
                                             const std::uint8_t* nrm_b, const std::uint8_t* nrm_w, int& ib, int& iw) {
	const int ac = Threat::tc_from_pt(type_of(apc));
	const int dc = Threat::tc_from_pt(type_of(tp));
	const Color acol = color_of(apc);
	const int   dsb  = (color_of(tp) == BLACK) ? 0 : 1;
	const int   patb = Threat::attack_pattern_id(ac, acol);
	const int   patw = Threat::attack_pattern_id(ac, ~acol);
	const int   fnb  = (int)nrm_b[(int)X];
	const int   fnw  = (int)nrm_w[(int)X];
	const int* const pbb = &g_threat_tables.pair_base[(acol == BLACK ? 0 : 162) + ac * 18];
	const int* const pbw = &g_threat_tables.pair_base[(acol == WHITE ? 0 : 162) + ac * 18];
	ib = pbb[dsb * 9 + dc]       + g_threat_tables.from_offset[patb][fnb] + (int)g_threat_tables.attack_order[patb][fnb][nrm_b[(int)T]];
	iw = pbw[(1 - dsb) * 9 + dc] + g_threat_tables.from_offset[patw][fnw] + (int)g_threat_tables.attack_order[patw][fnw][nrm_w[(int)T]];
}
} // namespace idxv2_verify
#endif

// 受け駒 T ぶんだけを足す emit。 書き込み先・ROWPF・ovf の扱いは従来の emit と同じ。
static inline void emit_v2(Step& s, bool add, const AttCtx& c, Piece tp, Square T,
                           const std::uint8_t* nrm_b, const std::uint8_t* nrm_w) {
#if defined(P2_FABLE_PROBE)
	if (probe::emit2()) probe::sink2 += (std::uint64_t)emit_idx_again(c.apc, c.X, tp, T, nrm_b, nrm_w);
#endif
	const int kb = g_tgt_key.kb[(int)tp];
#if defined(P2_LZ_IDXV2_VERIFY)
	// 対にする/しないの判定 (玉と空) も従来の `dc < 0` と一致すること。
	if ((kb < 0) != (Threat::tc_from_pt(type_of(tp)) < 0)) {
		++idxv2_verify::n_bad;
		std::fprintf(stderr, "IDXV2_VERIFY MISMATCH(skip) apc=%d X=%d tp=%d T=%d kb=%d\n",
			(int)c.apc, (int)c.X, (int)tp, (int)T, kb);
		idxv2_verify::report();
		std::abort();
	}
#endif
	if (kb < 0) return;                       // king target: not a feature (従来の dc < 0)
	const int kw = g_tgt_key.kw[(int)tp];
	const int tb = (int)nrm_b[(int)T];
	const int tw = (int)nrm_w[(int)T];
#if P2_LZ_IDXV2 == 1
	const int ob = (int)c.ordb[tb];
	const int ow = (int)c.ordw[tw];
#else
	const int ob = __builtin_popcountll(c.mlo_b & g_pat_mask.blo[tb]) + __builtin_popcountll(c.mhi_b & g_pat_mask.bhi[tb]);
	const int ow = __builtin_popcountll(c.mlo_w & g_pat_mask.blo[tw]) + __builtin_popcountll(c.mhi_w & g_pat_mask.bhi[tw]);
#endif
	const int ib = c.pbb[kb] + c.fob + ob;
	const int iw = c.pbw[kw] + c.fow + ow;
#if defined(P2_LZ_IDXV2_VERIFY)
	{
		int lb, lw;
		idxv2_verify::legacy(c.apc, c.X, tp, T, nrm_b, nrm_w, lb, lw);
		++idxv2_verify::n_pair;
		if (lb != ib || lw != iw) {
			++idxv2_verify::n_bad;
			std::fprintf(stderr, "IDXV2_VERIFY MISMATCH apc=%d X=%d tp=%d T=%d new=(%d,%d) old=(%d,%d)\n",
				(int)c.apc, (int)c.X, (int)tp, (int)T, ib, iw, lb, lw);
			idxv2_verify::report();
			std::abort();
		}
	}
#endif
#if defined(P2_LZ_ROWPF) && (P2_LZ_ROWPF == 1 || P2_LZ_ROWPF == 2)
	lz_pf_row(s.pf, s.pf_row, ib, P2_LZ_ROWPF == 2);
	lz_pf_row(s.pf, s.pf_row, iw, P2_LZ_ROWPF == 2);
#endif
	if (add) {
		if (s.na >= P2_LZ_STEPCAP) { s.ovf = true; return; }
		s.add_b[s.na] = ib; s.add_w[s.na] = iw; ++s.na;
	} else {
		if (s.ns >= P2_LZ_STEPCAP) { s.ovf = true; return; }
		s.sub_b[s.ns] = ib; s.sub_w[s.ns] = iw; ++s.ns;
	}
}
#endif // P2_LZ_IDXV2

// One delta step between parent and child (child = view v; parent differs on A and B only).
//   A/B      : from/to squares (A = -1 for a drop)
//   p_before : piece on A in the parent (NO_PIECE for a drop)
//   p_after  : piece on B in the child
//   cap      : piece on B in the parent (NO_PIECE if none)
// Invariant (bit-exact contract): multiset(ADD) - multiset(SUB) == pairs(child) - pairs(parent)
//   where pairs(.) is exactly what add_threat_dual enumerates (attacker non-king, target non-king).
template <typename V>
static void build_step(const V& v, Step& s,
                       int A, int B, Piece p_before, Piece p_after, Piece cap,
                       const std::uint8_t* nrm_b, const std::uint8_t* nrm_w) {
	const bool     drop    = (A < 0);
	const bool     capture = (cap != NO_PIECE);
	const Square   Bs      = (Square)B;
	const Square   As      = drop ? SQ_NB : (Square)A;
	const Bitboard bbB(Bs);
	const Bitboard occ_c   = v.occ();
	Bitboard       ab      = bbB;
	Bitboard       occ_p   = occ_c;                 // parent occupancy
	if (!drop)    { ab |= Bitboard(As); occ_p |= Bitboard(As); }
	if (!capture)   occ_p ^= bbB;                   // B empty in the parent
	const Bitboard occ_min = occ_c & ~ab;           // A and B both empty: rays pass through both

	auto piece_p = [&](Square sq) -> Piece {
		if (!drop && sq == As) return p_before;
		if (sq == Bs)          return cap;
		return v.piece_on(sq);
	};
	auto piece_c = [&](Square sq) -> Piece {
		if (sq == Bs) return p_after;
		return v.piece_on(sq);
	};

	// All pieces (both colours, child bitboards) attacking square t through occ_min.
	// Same identity as Position::attackers_to; kings are included here and dropped at tc<0.
	auto att_to = [&](Square t) -> Bitboard {
		return ((  (pawnEffect(WHITE, t)   & v.bb(BLACK, 0))
		         | (knightEffect(WHITE, t) & v.bb(BLACK, 1))
		         | (silverEffect(WHITE, t) & v.bb(BLACK, 2))
		         | (goldEffect(WHITE, t)   & v.bb(BLACK, 3))))
		     | ((  (pawnEffect(BLACK, t)   & v.bb(WHITE, 0))
		         | (knightEffect(BLACK, t) & v.bb(WHITE, 1))
		         | (silverEffect(BLACK, t) & v.bb(WHITE, 2))
		         | (goldEffect(BLACK, t)   & v.bb(WHITE, 3))))
		     | (bishopEffect(t, occ_min) & (v.bb(BLACK, 4) | v.bb(WHITE, 4)))
		     | (rookEffect(t, occ_min)   & ((v.bb(BLACK, 5) | v.bb(WHITE, 5))
		                                   | (v.bb(BLACK, 6) & lanceStepEffect(WHITE, t))
		                                   | (v.bb(WHITE, 6) & lanceStepEffect(BLACK, t))));
	};

	// 1 対の emit。 IDXV2 では攻め駒側の前計算 actx (各ループの直前で作る) を使う版に差し替える。
	//   IDXV2 未定義なら従来の emit 呼び出しそのもの (前処理後のコードは従来と同一)。
#if defined(P2_LZ_IDXV2)
#define P2_LZ_EMIT(add_, tp_, to_) emit_v2(s, (add_), actx, (tp_), (to_), nrm_b, nrm_w)
#else
#define P2_LZ_EMIT(add_, tp_, to_) emit(s, add_, apc, X, tp_, to_, nrm_b, nrm_w)   // 括弧を付けない = 前処理後が従来と同一
#endif
	// (a) the moved piece and the captured piece: all their pairs.
	auto emit_all = [&](bool add, Piece apc, Square X, const Bitboard& occ, auto&& board) {
		const int ac = Threat::tc_from_pt(type_of(apc));
		if (ac < 0) return;
		Bitboard t = attacks_of(ac, color_of(apc), X, occ) & occ;
#if defined(P2_LZ_IDXV2)
		AttCtx actx; make_ctx(actx, apc, ac, X, nrm_b, nrm_w);   // 攻め駒側を 1 回だけ
#endif
		while (t) { const Square to = t.pop(); P2_LZ_EMIT(add, board(to), to); }
	};
	if (!drop)   emit_all(false, p_before, As, occ_p, piece_p);
	if (capture) emit_all(false, cap,      Bs, occ_p, piece_p);
	emit_all(true, p_after, Bs, occ_c, piece_c);

	// (b)+(c) unmoved pieces whose pair set changes: those attacking A or B (through occ_min).
	Bitboard att = att_to(Bs);
	if (!drop) att |= att_to(As);
	att &= ~ab;
	while (att) {
		const Square X   = att.pop();
		const Piece  apc = v.piece_on(X);
		const int    ac  = Threat::tc_from_pt(type_of(apc));
		if (ac < 0) continue;                                   // king as attacker: not a feature
		const Color acol = color_of(apc);
#if defined(P2_LZ_IDXV2)
		AttCtx actx; make_ctx(actx, apc, ac, X, nrm_b, nrm_w);   // 攻め駒側を 1 回だけ
#endif
		if (ac == 1 || ac >= 5) {
			// slider: rays through A/B changed. Targets whose membership flipped, plus A/B themselves.
			const Bitboard ap   = attacks_of(ac, acol, X, occ_p) & occ_p;
			const Bitboard ac2  = attacks_of(ac, acol, X, occ_c) & occ_c;
			const Bitboard mask = (ap ^ ac2) | ab;
			Bitboard t = ap & mask;
			while (t) { const Square to = t.pop(); P2_LZ_EMIT(false, piece_p(to), to); }
			t = ac2 & mask;
			while (t) { const Square to = t.pop(); P2_LZ_EMIT(true,  piece_c(to), to); }
		} else {
			// step piece: only its pairs with targets on A/B change.
			const Bitboard rab = g_direct15_pseudo[acol][ac][(int)X] & ab;
			Bitboard t = rab & occ_p;
			while (t) { const Square to = t.pop(); P2_LZ_EMIT(false, piece_p(to), to); }
			t = rab & occ_c;
			while (t) { const Square to = t.pop(); P2_LZ_EMIT(true,  piece_c(to), to); }
		}
	}
#undef P2_LZ_EMIT
}

static inline void apply_step(std::int16_t* xb, std::int16_t* xw, const Step& s, std::size_t tto) {
	const Net& net = *g_net;
	apply_threat_rows(xb, net, s.add_b, s.na, false, tto);
	apply_threat_rows(xb, net, s.sub_b, s.ns, true,  tto);
	apply_threat_rows(xw, net, s.add_w, s.na, false, tto);
	apply_threat_rows(xw, net, s.sub_w, s.ns, true,  tto);
}

// ply i の Δ だけを適用する (物質化の forward 積み用)。 full と同じ apply_threat_rows を使う。
static inline void apply_seg(std::int16_t* xb, std::int16_t* xw, const Step& s, int i, std::size_t tto) {
	const Net& net = *g_net;
	const int as = s.seg_a[i], ae = s.seg_a[i + 1];
	const int ss = s.seg_s[i], se = s.seg_s[i + 1];
	if (ae > as) {
		apply_threat_rows(xb, net, s.add_b + as, ae - as, false, tto);
		apply_threat_rows(xw, net, s.add_w + as, ae - as, false, tto);
	}
	if (se > ss) {
		apply_threat_rows(xb, net, s.sub_b + ss, se - ss, true, tto);
		apply_threat_rows(xw, net, s.sub_w + ss, se - ss, true, tto);
	}
}

// 遡りの 1 手ぶんの記述。 template の build_chain で使うので namespace スコープに置く。
struct ChainStep { Move m; Piece cap; PieceType lmpt; bool nul; };

// 現局面側から base へ向かって Δ を積む。 V = PosView (d==1) / ScratchView (d>=2)。
//   🚨区間境界を build_step の**前**に打つ (mark(i))。 最後に呼び元が mark(d) を打つ。
template <class V>
static inline void build_chain(V& v, Step& tot, const ChainStep* steps, int d,
                               const std::uint8_t* nrm_b, const std::uint8_t* nrm_w) {
	for (int i = 0; i < d; ++i) {
		tot.mark(i);
		if (steps[i].nul) continue;
		const Move  m = steps[i].m;
		const int   B = (int)m.to_sq();
		const int   A = m.is_drop() ? -1 : (int)m.from_sq();
		const Piece p_after  = v.piece_on((Square)B);
		const Piece cap      = steps[i].cap;
		const Piece p_before = m.is_drop() ? NO_PIECE : make_piece(color_of(p_after), steps[i].lmpt);
#if defined(P2_FABLE_PROBE)
		if (probe::build2()) {   // 捨て用の Step にもう 1 回 (値は変えない)
			static thread_local Step dummy;
			dummy.ns = dummy.na = 0; dummy.ovf = false;
			build_step(v, dummy, A, B, p_before, p_after, cap, nrm_b, nrm_w);
			probe::sink2 += (std::uint64_t)(dummy.ns + dummy.na);
		}
#endif
		build_step(v, tot, A, B, p_before, p_after, cap, nrm_b, nrm_w);
		v.undo(m, cap, steps[i].lmpt);
	}
}

// 切り分け用: env `P2_LZ_NOMAT=1` なら物質化を行わない (適用率は落ちるが値は同じはず)。
static inline bool nomat() {
	static const bool v = [] {
		const char* e = std::getenv("P2_LZ_NOMAT");
		return e && *e && *e != '0';
	}();
	return v;
}

#if defined(P2_LZ_VERIFY)
// 2026-11-16 (THREATINC 段 2 の検算): resolve が true を返す直前に、 一時 buffer へ full (add_threat_dual) を作り
//   両視点 THREAT_ACC_DIM 列を要素比較する。 不一致は 1 件目で stderr に出して abort。 終了時に件数を出す。
//   🚨診断ビルド専用 (既定 off)。 add_threat_dual は pos を読むだけで st も探索も触らない。
namespace verify {
static std::uint64_t n_chk = 0, n_chk_d1 = 0, n_chk_nul = 0, n_bad = 0;
static void report() {
	std::fprintf(stderr, "LZ_VERIFY checked=%llu d1=%llu nul=%llu mismatch=%llu\n",
		(unsigned long long)n_chk, (unsigned long long)n_chk_d1, (unsigned long long)n_chk_nul, (unsigned long long)n_bad);
}
static struct Init { Init() { std::atexit(report); } } init_;
static void check(const Position& pos, const StateInfo* st, bool mir_b, bool mir_w, std::size_t tto, int d) {
	alignas(64) std::int16_t fb[THREAT_ACC_DIM], fw[THREAT_ACC_DIM];
	std::memset(fb, 0, sizeof fb);
	std::memset(fw, 0, sizeof fw);
	add_threat_dual(pos, mir_b, mir_w, fb, fw, tto);
	++n_chk;
	if (d == 1) ++n_chk_d1;
	if (d == 1 && st->dirtyPiece.dirty_num == 0) ++n_chk_nul;
	for (int k = 0; k < THREAT_ACC_DIM; ++k) {
		if (fb[k] != st->p2_threat_acc[BLACK][k] || fw[k] != st->p2_threat_acc[WHITE][k]) {
			++n_bad;
			std::fprintf(stderr, "LZ_VERIFY MISMATCH d=%d col=%d full=(%d,%d) lz=(%d,%d) sfen=%s\n", d, k,
				(int)fb[k], (int)fw[k], (int)st->p2_threat_acc[BLACK][k], (int)st->p2_threat_acc[WHITE][k],
				pos.sfen().c_str());
			report();
			std::abort();
		}
	}
}
} // namespace verify
#endif

// **本体**。 戻り値 true = 差分で解決した (acc / valid / group / source を書いた)。
//   false = 呼び元が従来の FULL 再計算に落とす。
//   2026-11-16 (THREATINC 段 3): resolve は ensure_threat_both (noinline) に inline され、 full 路の idx 配列と
//   Step が同じフレームに載って 6.7KB (___chkstk_ms あり / lzB_nopgo の逆アセンブルで確認)。 Step を縮めても
//   full 路の分が残るので、 `-DP2_LZ_RESOLVE_NOINLINE` で resolve を別フレームにできるようにした (値は不変)。
#if defined(P2_LZ_RESOLVE_NOINLINE)
__attribute__((noinline))
#endif
static bool resolve(const Position& pos, StateInfo* st, bool mir_b, bool mir_w,
                    std::size_t tto, std::uint8_t cur_group) {
	ChainStep steps[DMAX];
	bool mir_at_b[DMAX], mir_at_w[DMAX];   // mir_at_*[j] = node s_{j+1} の鏡像
	StateInfo* nodes[DMAX];                // nodes[i] = s_i (s_0 = st)
	StateInfo* s = st;
	int d = 0;
	Square kb = pos.square<KING>(BLACK), kw = pos.square<KING>(WHITE);
#if defined(P2_FABLE_PROBE)
	const std::uint64_t pt0 = __rdtsc();
	++probe::n_call;
#endif
	for (;;) {
#if defined(P2_FABLE_PROBE)
		if (s->previous == nullptr) { ++probe::f_root; probe::cyc_fail += __rdtsc() - pt0; return false; }
		if (d >= DMAX) {
			// 真の距離 (群一致の valid な祖先まで / 鏡像は見ない)。 上限 64。
			++probe::f_dmax;
			int k = d; const StateInfo* q = s; int hit = -1;
			while (q->previous != nullptr && k < 64) {
				++k; q = q->previous;
				if (q->p2_threat_valid[BLACK] && q->p2_threat_valid[WHITE]
				    && q->p2_threat_group[BLACK] == cur_group && q->p2_threat_group[WHITE] == cur_group) { hit = k; break; }
			}
			probe::far_h[hit < 0 ? 4 : hit <= 12 ? 0 : hit <= 16 ? 1 : hit <= 32 ? 2 : 3] += 1;
			if (hit < 0) {   // M0: root まで登って「root は valid で群だけ違う」かを見る
				while (q->previous != nullptr) q = q->previous;
				if (q->p2_threat_valid[BLACK] && q->p2_threat_valid[WHITE]
				    && (q->p2_threat_group[BLACK] != cur_group || q->p2_threat_group[WHITE] != cur_group)) {
					++probe::n_none_grp;
					const StateInfo* const par = st->previous;
					const int pi = (pos.game_ply() - 1) & 1023;
					const auto pkk = par->key();   // Key は 64 / 128 bit どちらでも先頭 8 バイトを使う
					std::uint64_t pk; std::memcpy(&pk, &pkk, sizeof pk);
					if (probe::sib_ptr[pi] == par && probe::sib_key[pi] == pk && probe::sib_grp[pi] == cur_group)
						++probe::n_none_sib;
					probe::sib_ptr[pi] = par; probe::sib_key[pi] = pk; probe::sib_grp[pi] = cur_group;
				}
			}
			probe::cyc_fail += __rdtsc() - pt0;
			return false;
		}
#endif
		if (s->previous == nullptr || d >= DMAX) return false;   // 不変条件 6
		nodes[d] = s;
		steps[d] = ChainStep{ s->lastMove, s->capturedPiece, s->lastMovedPieceType,
		                      s->dirtyPiece.dirty_num == 0 };
		// 玉を巻き戻す (鏡像の判定に使う)
		if (!steps[d].nul && steps[d].lmpt == KING && !steps[d].m.is_drop()) {
			const Square to = steps[d].m.to_sq();
			if (to == kb)      kb = steps[d].m.from_sq();
			else if (to == kw) kw = steps[d].m.from_sq();
		}
		mir_at_b[d] = is_hm_mirror(kb, BLACK);
		mir_at_w[d] = is_hm_mirror(kw, WHITE);
		++d;
		s = s->previous;
		if (s->p2_threat_valid[BLACK] && s->p2_threat_valid[WHITE]
		    && s->p2_threat_group[BLACK] == cur_group
		    && s->p2_threat_group[WHITE] == cur_group) break;
	}
	// 不変条件 2: base の鏡像が現在と違えば差分は使えない (索引の正規化が別物になる)
#if defined(P2_FABLE_PROBE)
	if (mir_at_b[d - 1] != mir_b || mir_at_w[d - 1] != mir_w) { ++probe::f_mir; probe::cyc_fail += __rdtsc() - pt0; return false; }
	const std::uint64_t pt1 = __rdtsc();
#endif
	if (mir_at_b[d - 1] != mir_b || mir_at_w[d - 1] != mir_w) return false;

	const std::uint8_t* const nrm_b = g_norm_sq[BLACK][mir_b ? 1 : 0];
	const std::uint8_t* const nrm_w = g_norm_sq[WHITE][mir_w ? 1 : 0];
	Step tot;
#if defined(P2_LZ_D1FAST)
	// 2026-11-16 (THREATINC 段 2 / A3): **d=1 専用路**。 d==1 は巻き戻す先が無いので ScratchView の
	//   初期化 (盤 81 升 + bitboard 14 本の写し) と最後の不要な undo を払わず、 生盤面 (PosView) から
	//   build_step を直接呼ぶ。 A/B/p_after/cap/p_before は build_chain と 1 文字も違えない式で作る
	//   (PosView と ScratchView::init 直後の内容は同一なので Δ の多重集合も同一 = bit-exact)。
	//   null move の子は build を呼ばず (na == ns == 0)、 下の memcpy と valid の書き込みだけが走る。
#if defined(P2_LZ_ROWPF)
	tot.pf = lz_row_ptr(*g_net, tto);
	tot.pf_row = g_net->threat_is_i16 ? 128 : 64;
#endif
	if (d == 1) {
		tot.mark(0);
		if (!steps[0].nul) {
			const PosView pv{pos};
			const Move  m = steps[0].m;
			const int   B = (int)m.to_sq();
			const int   A = m.is_drop() ? -1 : (int)m.from_sq();
			const Piece p_after  = pv.piece_on((Square)B);
			const Piece cap      = steps[0].cap;
			const Piece p_before = m.is_drop() ? NO_PIECE : make_piece(color_of(p_after), steps[0].lmpt);
			build_step(pv, tot, A, B, p_before, p_after, cap, nrm_b, nrm_w);
		}
	} else {
		ScratchView v; v.init(pos);
		build_chain(v, tot, steps, d, nrm_b, nrm_w);
	}
#else
	// 🚨v1 は **ScratchView 一本**。 PosView (生盤面) には undo が無く、 d==1 の高速路は
	//   `if constexpr` での分岐が必要になる。 盤の控えは 81 バイト + class bitboard 14 本なので
	//   列挙 1 回 (≈1.0 µs) に対して無視できる費用。 NPS が採用線に届かなかったときに見直す。
	//   -> 2026-11-16: `-DP2_LZ_D1FAST` で d=1 専用路を足した (上)。
	ScratchView v; v.init(pos);
#if defined(P2_LZ_ROWPF)
	tot.pf = lz_row_ptr(*g_net, tto);
	tot.pf_row = g_net->threat_is_i16 ? 128 : 64;
#endif
	build_chain(v, tot, steps, d, nrm_b, nrm_w);
#endif // P2_LZ_D1FAST
#if defined(P2_LZ_ROWPF) && P2_LZ_ROWPF == 3
	if (!tot.ovf) {   // 対照: リード時間ほぼ 0 の位置でまとめて prefetch
		for (int j = 0; j < tot.na; ++j) { lz_pf_row(tot.pf, tot.pf_row, tot.add_b[j], 0); lz_pf_row(tot.pf, tot.pf_row, tot.add_w[j], 0); }
		for (int j = 0; j < tot.ns; ++j) { lz_pf_row(tot.pf, tot.pf_row, tot.sub_b[j], 0); lz_pf_row(tot.pf, tot.pf_row, tot.sub_w[j], 0); }
	}
#endif
#if defined(P2_FABLE_PROBE)
	if (tot.ovf) { ++probe::f_ovf; probe::cyc_fail += __rdtsc() - pt0; return false; }
#endif
	if (tot.ovf) return false;            // 不変条件 6 (対が 256/視点 を超えた)
	tot.mark(d);                          // 終端の境界
#if defined(P2_FABLE_PROBE)
	const std::uint64_t pt2b = __rdtsc();   // build の終わり (warm の読みを build に含めない)
	if (probe::warm()) {   // 計時区間の外で全行を 1 度読む (値は変えない)
		const Net& wn = *g_net; std::uint64_t acc8 = 0;
		// 🚨本番 net は i8 (1 行 64B)。 i16 net なら 2 ライン目 (+64) も読む。
		const std::size_t rb = wn.threat_is_i16 ? 128 : 64;
		auto touch = [&](int idx) -> std::uint64_t {
			const char* p = lz_row_ptr(wn, tto + (std::size_t)idx);
			return (std::uint64_t)(unsigned char)p[0] + (rb > 64 ? (std::uint64_t)(unsigned char)p[64] : 0);
		};
		for (int j = 0; j < tot.na; ++j) acc8 += touch(tot.add_b[j]) + touch(tot.add_w[j]);
		for (int j = 0; j < tot.ns; ++j) acc8 += touch(tot.sub_b[j]) + touch(tot.sub_w[j]);
		probe::warm_sink += acc8;
		_mm_lfence();
	}
	const std::uint64_t pt2 = __rdtsc();
#endif

	std::int16_t* const ab = st->p2_threat_acc[BLACK];
	std::int16_t* const aw = st->p2_threat_acc[WHITE];
	std::memcpy(ab, s->p2_threat_acc[BLACK], sizeof(std::int16_t) * THREAT_ACC_DIM);
	std::memcpy(aw, s->p2_threat_acc[WHITE], sizeof(std::int16_t) * THREAT_ACC_DIM);
	// i = d-1 (最も古い手) から i = 0 へ適用していくと、 i を適用し終えた acc が node s_i のもの。
	for (int i = d - 1; i >= 0; --i) {
		apply_seg(ab, aw, tot, i, tto);
		if (i == 0) break;
		// **物質化**: node s_i の鏡像が現在と同じときだけ (不変条件 1/2)。
		//   これが適用率を 43% -> 85% に上げる本体。 次の兄弟は d=1 で済む。
		//   🚨2026-11-06: G2 で「手が違う」が出たので **env `P2_LZ_NOMAT=1` で物質化だけを切れる**
		//     ようにした (切り分け用)。 これで bit-exact になれば原因は物質化側、
		//     ならなければ Δ の適用側。 既定 (env 無し) は物質化あり。
		if (!nomat() && mir_at_b[i - 1] == mir_b && mir_at_w[i - 1] == mir_w) {
			StateInfo* const q = nodes[i];
			// 🚨**(A) 2026-11-06 の bit-exact 崩れの原因**(レビュー fable)。
			//   遡りの break は「valid ∧ 群一致」なので、 **valid だが群が違う** node は
			//   素通りして nodes[] に積まれる。 そこへ cur_group の acc を書くと、
			//   その node が自分の bucket で再評価されたとき (root の反復深化 / aspiration /
			//   null move の verification) **別の群の表の acc** を使ってしまい評価値が変わる。
			//   -> **valid な node には一切触らない**。 valid な acc はその node の群で正しい。
			//   適用率は落ちない (valid な node はそもそも base の候補側)。
			if (q->p2_threat_valid[BLACK] || q->p2_threat_valid[WHITE]) continue;
			std::memcpy(q->p2_threat_acc[BLACK], ab, sizeof(std::int16_t) * THREAT_ACC_DIM);
			std::memcpy(q->p2_threat_acc[WHITE], aw, sizeof(std::int16_t) * THREAT_ACC_DIM);
			q->p2_threat_valid[BLACK]  = q->p2_threat_valid[WHITE]  = true;
			q->p2_threat_group[BLACK]  = q->p2_threat_group[WHITE]  = cur_group;
			q->p2_threat_source[BLACK] = q->p2_threat_source[WHITE] = q;
		}
	}
	st->p2_threat_valid[BLACK]  = st->p2_threat_valid[WHITE]  = true;
	st->p2_threat_group[BLACK]  = st->p2_threat_group[WHITE]  = cur_group;
	st->p2_threat_source[BLACK] = st->p2_threat_source[WHITE] = st;
#if defined(P2_FABLE_PROBE)
	{
		const std::uint64_t pt3 = __rdtsc();
		const std::uint64_t r = 2u * (std::uint64_t)(tot.na + tot.ns);   // 視点 2 本ぶんの行適用数
		++probe::n_ok; probe::rows += r;
		probe::cyc_walk += pt1 - pt0; probe::cyc_build += pt2b - pt1; probe::cyc_apply += pt3 - pt2;
		probe::n_d[d] += 1; probe::cb_d[d] += pt2b - pt1; probe::ca_d[d] += pt3 - pt2; probe::rows_d[d] += r;
		if (d == 1 && steps[0].nul) ++probe::n_nul_d1;
	}
#endif
#if defined(P2_LZ_VERIFY)
	verify::check(pos, st, mir_b, mir_w, tto, d);
#endif
	return true;
}

} // namespace lz
#endif // P2_THREAT_LZ
