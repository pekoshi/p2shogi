// OPN (序盤帯専用の副 net) — エンジン側。 2026-11-14
//   `-DP2_OPN` のときだけ p2nnue.cpp から (namespace YaneuraOu::Eval::P2NNUE の中、 evaluate_internal の直前で) include される。
//   フラグが無ければ 1 命令も入らない。
//
// ■ 設計
//   C:/project/p2shogi/runs/prereg/DESIGN-20261114-OPNINT-FABLE.md (型 (C) の混合 / 5〜8 章)
//   C:/project/p2shogi/runs/prereg/DESIGN-20261114-OPN2048-FABLE.md (K16 / 6 章のつなぎ目)
//   ユーザー決定 (11-14): OPN は進行度 bucket 0-1 だけで使う (つなぎ目は bucket 1|2 の境 −1.0986 の内側)。
//     境の位置と幅は OPN_ProgHi / OPN_ProgWidth で変えられる。 λ_max < 1 の混合を本命として想定。
//
// ■ 評価値 (raw = 本体の raw_total、 style = 様式項の合計 (cp × eff_fv))
//   raw_opn_cal = ((int64) raw_opn × OPN_Scale[pb]) >> 10                (算術右シフト)
//   raw_main    = raw_total − style
//   raw_blend   = raw_main + (((int64)(raw_opn_cal − raw_main) × lam) >> 8)
//   cp          = (raw_blend + style) / eff_fv                            (C の整数除算)
//   lam = 0 なら OPN の経路に入らない (従来のコードそのまま)。 lam = 256 なら raw_blend = raw_opn_cal で、
//   探索中 (dump でない) は本体のアキュムレータ・threat・KH を一切呼ばずに返す。
//
// ■ λ (進行度 p = st->p2_prog_sum、 pb = progress_bucket)
//   pb >= 2                 : 0   (11-18: OPN は bucket 0〜1 だけ。 OPN_ProgHi の上限 −1099 は bucket 1|2 の境 −1.0986 の内側)
//   OPN_ProgWidth == 0      : p < P_hi ? 256 : 0                     (幅 0 の切り替え)
//   それ以外                : clamp((int)((P_hi − p) × (256 / W) + 0.5f), 0, 256)
//   lam = (lam × OPN_LambdaMax) >> 8
//   P_hi = OPN_ProgHi / 1000、 W = OPN_ProgWidth / 1000 (どちらも float)。
//   🚨設計書 5.1 は「幅 0 なら lam = 256」(bucket 0-2 全域) だが、 つなぎ目を bucket 1|2 に置く決定に合わせて
//     幅 0 でも P_hi で切る (P_hi = −511 なら設計書の式と同じ)。
//
// ■ net の形式 (学習器 shogi_layerstack の quantised.bin、 PSQT / threat / 追加 section なし)
//   u32 version (0x7AF32F20) / u32 nethash / u32 arch_len / arch / u32 ft_hash
//   LEB128 ブロック ft_bias [W] / LEB128 ブロック ft_w [rows × W] (feature-major)
//   head × 8 : u32 fc_hash / i32 b0[L1] / i8 w0[L1][pad32(W)] / i32 b1[L2] / i8 w1[L2][pad32(2(L1−1))]
//              / i32 b2 / i8 w2[pad32(L2)]
//   入力は HalfKA_hm (rows 73,305 = 45 区分 × 1629) か HalfKA_hmK16 (rows 26,064 = 16 区分 × 1629)。
//   K16 は arch の `HalfKA_hmK16(Friend)[26064->` と行数で見分ける (ft_hash は K45 と同じなので使わない)。
//   幅 W・L1・L2 は arch から読む (W ≤ P2_OPN_FT_MAX、 W % 32 == 0、 L1 % 8 == 0 かつ ≤ 32、 L2 % 8 == 0 かつ ≤ 64)。
//   head は進行度 bucket (0..7) で引く (学習は bucket 0-2 だけなので、 混ぜるのに使うのは 0..2)。
//
// ■ 順伝播 (runs/opnarch/opn_static2.py の forward_encoded と同じ整数演算)
//   acc = ft_bias + Σ ft_w[行] (i16 wrapping) / tr = (clamp(a[j],0,254) × clamp(a[j+W/2],0,254)) >> 9
//   l1 = b0 + w0·tr / v = l1[0..M) (M = L1−1) / l2in = [min((v·v)>>19,127), clamp(v>>6,0,127)]
//   l2 = clamp((b1 + w1·l2in) >> 6, 0, 127) / raw = b2 + w2·l2 + l1[M]
//
// ■ 守るべき不変条件 (レビューで見る点。 設計書 9 章)
//   1. p2_opn_computed[c] == true なら p2_opn_acc[c] はその局面・視点 c・その玉の位置と鏡像のもの
//      (do_move / do_null_move で false にする。 玉が動いた視点は祖先を使わない)。
//   2. OPN 無効時 (g_active == false かつ dump でない) は、 ensure_progress が走る node の集合も足す値も従来と同じ。
//   3. lam = 256 の経路は st->p2_acc / p2_threat_acc / p2_psqt を読まない。
//   4. maddubs の飽和条件 (入力 ≤ 126、 |重み| ≤ 128 で 2 項の和 ≤ 32,256 < 32,767) は本体と同じ。
//   5. StateInfo の追加欄 p2_opn_acc は [p2_acc, dirtyPiece) の非コピー領域の中。
#pragma once

#if defined(P2_OPN)

#if !defined(P2_OPN_CHAIN_MAX)
#define P2_OPN_CHAIN_MAX 8
#endif
// 2026-11-15 (Fable / OPT-20261115-OPN2048-FABLE.md): 幅 2048 向けの最適化。 -DP2_OPN_F2 で入る。 どれも値を変えない。
//   F2-1 fc0 の重み (幅 2048 × L1 16 で 32 KB = L1 キャッシュと同じ大きさ) を 2 KB 先まで先読みする。 重みは 64 byte 整列。
//   F2-2 差分更新の行を 4 line 先まで先読みする。
//   F2-3 FT を 4096 byte 整列で持つ (幅 2048 なら 1 行 = 1 ページ)。 実測 (4,096 評価ごとに 64 byte 整列の写しと交互に切り替えた
//        rdtsc、 F2 net、 置き換え、 OPN の全費用): −1.1 / −2.0 / −3.6 / −1.6%。 -DP2_OPN_F2_NOPAGE で 64 byte 整列に戻せる。
//   F2-4 K16 で、 玉が動いても行の区分 (16 区分) と鏡像が変わらない手は差分で処理する (Finny に落とさない)。
//   F2-5 priming (evaluate_with_no_return) では OPN の acc を作らず、 子が親を跨いで追いつくときに親の acc を作って残す。
//        実測 (4,096 評価ごとに方式を交互に切り替えた rdtsc、 F2 net、 OPN の全費用): 置き換え −4.7 / −4.3 / −3.7%、 混合 −2.3%。
//        -DP2_OPN_F2_EAGER_PRIME で従来の priming に戻せる。 🚨priming を単にやめるだけ (親を残さない) は +14.9% の悪化。
//   F2-6 評価する node の差分更新と transform を 1 回の列ループで行う (acc の読み直しを無くす)。
#if defined(P2_OPN_F2) && !defined(P2_OPN_ROWS64)
#define P2_OPN_ROWS64
#endif
#if !defined(P2_OPN_ROW_PF_LINES)
#define P2_OPN_ROW_PF_LINES 4      // F2-2: 行の先読み距離 (64 byte line 数)。 実測 3〜5 が最良、 0 / 8 / 16 は悪化
#endif
#if !defined(P2_OPN_FC0_PF_BYTES)
#define P2_OPN_FC0_PF_BYTES 2048   // F2-1: fc0 の重みの先読み距離 (byte)。 実測 128 → 2048 で単調に改善
#endif

namespace opn {

static constexpr int FT_MAX    = P2_OPN_FT_MAX;
static constexpr int L1_MAX    = 32;
static constexpr int L2_MAX    = 64;
static constexpr int NHEADS    = 8;
static constexpr int CHAIN_MAX = P2_OPN_CHAIN_MAX;
static constexpr int K45_ROWS  = 45 * PIECE_INPUTS;   // 73,305
static constexpr int K16_ROWS  = 16 * PIECE_INPUTS;   // 26,064
static constexpr std::uint32_t VERSION = 0x7AF32F20u;
static_assert(FT_MAX % 32 == 0 && FT_MAX >= 32, "P2_OPN_FT_MAX は 32 の倍数");
static_assert(FT_MAX == StateInfo::P2_OPN_FT_DIM, "StateInfo の OPN 幅と食い違う");
static_assert(CHAIN_MAX >= 1 && CHAIN_MAX <= P2_ACC_CHAIN_MAX, "P2_OPN_CHAIN_MAX は本体の chain[] (P2_ACC_CHAIN_MAX) 以下");

#if defined(P2_OPN_ROWS64)
// P3 (ANALYSIS-20261115 6 章): FT を 64 byte 整列で持つ (本体の WsVec と同じ AlignedAlloc。 large page は使わない)。
//   値も添字も変わらない (確保場所だけ)。
#if defined(P2_OPN_F2) && !defined(P2_OPN_F2_NOPAGE)
// F2-3: 4096 byte 整列。 幅 2048 の 1 行 (4096 byte) がちょうど 1 ページに載るので、 行の途中でページを跨がない
//   (ハードウェアの先読みはページ境界で止まる / TLB も 1 行 1 エントリ)。 値も添字も変わらない。 large page は使わない。
using FtVec = std::vector<std::int16_t, AlignedAlloc<std::int16_t, 4096>>;
#else
using FtVec = std::vector<std::int16_t, AlignedAlloc<std::int16_t, 64>>;
#endif
#else
using FtVec = std::vector<std::int16_t>;
#endif

static inline int pad32i(int n) { return (n + 31) / 32 * 32; }

// K16 (DESIGN-20261114-OPN2048-FABLE.md 2.3): kb = f*9 + r、 r >= 6 (持ち主から見て 7〜9 段) -> f*3 + (r − 6)、 それ以外 -> 15。
//   学習器 shogi_halfka.rs の king_bucket16 / opn_static2.py の KB16 と同じ式。
struct Kb16Table {
	std::uint8_t v[45];
	constexpr Kb16Table() : v() {
		for (int kb = 0; kb < 45; ++kb)
			v[kb] = (std::uint8_t) ((kb % 9) >= 6 ? (kb / 9) * 3 + (kb % 9 - 6) : 15);
	}
};
static constexpr Kb16Table KB16{};

struct Head {
	std::int32_t b0[L1_MAX] = {};
	std::int32_t b1[L2_MAX] = {};
	std::int32_t b2 = 0;
	alignas(32) std::int8_t w2[L2_MAX] = {};
	std::vector<std::int8_t> w0;     // [L1][W] (行優先、 pad32(W) == W)
	std::vector<std::int8_t> w1;     // [L2][L2IN_PAD]
#if defined(P2_OPN_F2)
	WsVec w0ws;                      // affine_par 用の scrambled [W/4][L1][4]。 64 byte 整列 (L1 = 16 なら 1 chunk = 1 line)
#else
	std::vector<std::int8_t> w0ws;   // affine_par 用の scrambled [W/4][L1][4]
#endif
	std::vector<std::int8_t> w1ws;   // [L2IN_PAD/4][L2][4]
#if defined(P2_FC1_PREC)
	std::vector<std::int16_t, AlignedAlloc<std::int16_t, 64>> w1ws16;   // P2_FC1_PREC: fc1 の i16 重み [L2IN_PAD/2][L2][2] (affine16_t)
	alignas(32) std::int16_t w2_16[L2_MAX] = {};                        // P2_FC1_PREC: fc2 の i16 重み (dot_i16)
#endif
#if defined(P2_FT_OUT16)
	std::vector<std::int16_t, AlignedAlloc<std::int16_t, 64>> w0ws16;   // P2_FT_OUT16: fc0 の i16 入力版 [W/2][L1][2]
	alignas(32) std::int32_t b0m[L1_MAX] = {};                          // m·b0
#endif
#if defined(P2_OPN_R2_KERNELS)
	bool g8_ok = false;              // R2-1: w0ws が affine_g8_safe を満たす (読み込み時に決める)
#endif
};

struct Net {
	int W = 0, half = 0, L1 = 0, L2 = 0, M = 0, L2IN_PAD = 0, rows = 0;
	bool k16 = false;
	std::uint32_t nethash = 0;
	std::string arch;
	FtVec ftb;   // [W]
	FtVec ftw;   // [rows][W]
	Head heads[NHEADS];
};

static std::unique_ptr<Net> g_onet;
static bool g_loaded = false;      // OPNFile を読み込み済み
static bool g_active = false;      // g_loaded && OPN_LambdaMax > 0 (探索で混ぜる)
static int  g_generation = 0;      // 読み込みのたびに ++ (Finny の無効化)
static std::string g_loaded_path;
static std::uint64_t g_loaded_size = 0;

// option (既定値では callback が呼ばれないので、 ここの初期値を add_options の既定と一致させること)
static int g_lam_max    = 256;     // OPN_LambdaMax 0..256
static int g_prog_hi    = -1299;   // 11-16 既定を対局時の値へ (旧 -1099)。 // OPN_ProgHi    (進行度 × 1000)。 −1099 = bucket 1|2 の境 (−1.0986) の内側
static int g_prog_width = 0;       // 11-16 既定を対局時の値へ (旧 200)。 // OPN_ProgWidth (× 1000)。 0 = 幅 0 の切り替え
static int g_scale[2]   = { 1196, 1204 };   // 11-16 既定を対局時の値へ (旧 1024 x3)。   // OPN_Scale0..1 (Q10)。 11-18: bucket 2 の Scale2 は削除
// OPN_Offset (11-18): OPN の評価に手番側の視点で足す定数。 単位は **cp (評価値の表示単位)**。 内部では cp × eff_fv (その局面の FV_SCALE)
//   を raw に足す (calibrate の後 = 本体の raw と同じ単位になった値に足す)。 λ = 256 の経路 (opn_eval_only) では表示値がちょうど +offset cp、
//   混ぜる経路 (λ < 256) では OPN 側の値が +offset cp ずれて λ/256 の重みで効く (どちらも同じ calibrate を通る)。 既定 0 = 従来と同じ値。
//   目的: 継ぎ目での OPN と本体の段差 (OPN が約 29cp 悲観的) を消す。
static int g_offset_cp  = 0;
static float g_phi = -1.299f;      // = g_prog_hi / 1000 (load で update_derived が上書き)
static float g_inv = 0.0f;         // = 256 / (g_prog_width / 1000)、 幅 0 は 0 (update_derived と同じ)

static inline void update_derived() {
	g_phi = (float) g_prog_hi / 1000.0f;
	g_inv = g_prog_width > 0 ? 256.0f / ((float) g_prog_width / 1000.0f) : 0.0f;
	g_active = g_loaded && g_lam_max > 0;
	// P2 (ANALYSIS-20261115 6 章): OPN 有効時は本体の追いつきを CHAIN_MAX 手までにして、 それより遠ければ Finny に落とす。
	//   探索中には呼ばれない (option の callback と load だけ)。 値は変わらない (p2nnue.cpp の g_acc_chain_limit の注記)。
	g_acc_chain_limit = g_active ? CHAIN_MAX : P2_ACC_CHAIN_MAX;
}

// 移行区間の λ (0..256)。 局面の関数 (進行度 p と bucket だけで決まる)。
static inline int lambda(int pb, float p) {
	if (pb >= 2) return 0;   // 11-18: OPN は bucket 0〜1 だけ (旧 pb >= 3)。 OPN_ProgHi <= −1099 なら p < phi の局面は bucket 0〜1 なので値は従来と同じ
	int lam;
	if (g_prog_width == 0) {
		lam = (p < g_phi) ? 256 : 0;
	} else {
		const float x = (g_phi - p) * g_inv + 0.5f;
		lam = x <= 0.0f ? 0 : (x >= 256.0f ? 256 : (int) x);
	}
	return (lam * g_lam_max) >> 8;
}

// raw_opn を本体の raw の単位に直し、 OPN_Offset (cp × eff_fv) を足す。 pb >= 2 は dump (表示) のときだけ来るので Scale1 を使う。
static inline std::int32_t calibrate(std::int32_t raw_opn, int pb, int eff_fv) {
	return (std::int32_t) (((std::int64_t) raw_opn * g_scale[pb < 1 ? 0 : 1]) >> 10) + g_offset_cp * eff_fv;
}

// raw_total (本体 + 様式項) と OPN を混ぜた raw_total を返す。 lam = 256 なら raw_opn_cal + style と一致する。
static inline std::int32_t blend(std::int32_t raw_total, std::int32_t style, std::int32_t opn_cal, int lam) {
	const std::int32_t raw_main = raw_total - style;
	const std::int64_t d = (std::int64_t) opn_cal - (std::int64_t) raw_main;
	return (std::int32_t) ((std::int64_t) raw_main + ((d * lam) >> 8)) + style;
}

// ---------------------------------------------------------------------------
// 計数 (probe ビルドだけ)。 G3 の「評価の種類別の回数 / 追いつきの距離と適用行数 / Finny に落ちた回数」。
//   P2_OPN_PROBE では差分更新のたびに全計算と突き合わせる (P4 の自己照合)。 🚨本番ビルドには入らない。
// ---------------------------------------------------------------------------
#if defined(P2_OPN_PROBE)
namespace probe {
static std::uint64_t ev_only = 0, ev_both = 0, ev_main = 0, ev_dump = 0;
static std::uint64_t ens = 0, ens_chain = 0, chain_len = 0, chain_rows = 0, ens_refresh = 0, refresh_rows = 0;
static std::uint64_t refresh_noprev = 0, refresh_king = 0, refresh_far = 0, refresh_overflow = 0;
static std::uint64_t prime_opn = 0, prime_skip_main = 0, sc_n = 0, sc_ng = 0, king_same_row = 0, lazy_parent = 0;
static std::uint64_t fc0_n = 0, fc0_nnz = 0, fc0_chunks = 0;   // P1: fc0 の入力の非零 4 byte chunk (合計 / 全 chunk 数)
static void report() {
	std::fprintf(stderr,
		"P2OPNPROBE ev_only=%llu ev_both=%llu ev_main=%llu ev_dump=%llu ens=%llu ens_chain=%llu chain_len=%llu chain_rows=%llu "
		"ens_refresh=%llu refresh_rows=%llu refresh_noprev=%llu refresh_king=%llu refresh_far=%llu refresh_overflow=%llu "
		"prime_opn=%llu prime_skip_main=%llu selfcheck_n=%llu selfcheck_ng=%llu\n",
		(unsigned long long) ev_only, (unsigned long long) ev_both, (unsigned long long) ev_main, (unsigned long long) ev_dump,
		(unsigned long long) ens, (unsigned long long) ens_chain, (unsigned long long) chain_len, (unsigned long long) chain_rows,
		(unsigned long long) ens_refresh, (unsigned long long) refresh_rows, (unsigned long long) refresh_noprev,
		(unsigned long long) refresh_king, (unsigned long long) refresh_far, (unsigned long long) refresh_overflow,
		(unsigned long long) prime_opn, (unsigned long long) prime_skip_main, (unsigned long long) sc_n, (unsigned long long) sc_ng);
	std::fprintf(stderr,
		"P2OPNPROBE2 king_same_row=%llu lazy_parent=%llu fc0_n=%llu fc0_nnz=%llu fc0_chunks=%llu main_ens=%llu main_chain_n=%llu main_chain_len=%llu main_refresh=%llu main_refresh_far=%llu acc_chain_limit=%d\n",
		(unsigned long long) king_same_row, (unsigned long long) lazy_parent, (unsigned long long) fc0_n, (unsigned long long) fc0_nnz, (unsigned long long) fc0_chunks,
		(unsigned long long) OpnMainProbe::ens, (unsigned long long) OpnMainProbe::chain_n, (unsigned long long) OpnMainProbe::chain_len,
		(unsigned long long) OpnMainProbe::refresh, (unsigned long long) OpnMainProbe::refresh_far, g_acc_chain_limit);
}
static struct Init { Init() { std::atexit(report); } } init_;
} // namespace probe
#define P2_OPN_COUNT(x) (++probe::x)
#define P2_OPN_ADD(x, v) (probe::x += (std::uint64_t) (v))
#else
#define P2_OPN_COUNT(x) ((void) 0)
#define P2_OPN_ADD(x, v) ((void) 0)
#endif

// ---------------------------------------------------------------------------
// 読み込み (設計書 8.2 の照合をすべて行う。 1 つでも外れたら失敗)
// ---------------------------------------------------------------------------
static inline bool parse_int(const std::string& s, std::size_t& p, int& out) {
	if (p >= s.size() || s[p] < '0' || s[p] > '9') return false;
	long v = 0;
	while (p < s.size() && s[p] >= '0' && s[p] <= '9') { v = v * 10 + (s[p++] - '0'); if (v > 100000000) return false; }
	out = (int) v;
	return true;
}

static std::string load_impl(const std::string& path, std::unique_ptr<Net>& out_net, std::uint64_t& out_size) {
	std::ifstream ifs(path, std::ios::binary);
	if (!ifs) return "開けない: " + path;
	std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	out_size = buf.size();
	Reader r(buf);
	auto net = std::make_unique<Net>();
	const std::uint32_t version = r.read_u32_le();
	net->nethash = r.read_u32_le();
	const std::uint32_t alen = r.read_u32_le();
	if (!r.ok() || alen == 0 || alen > 4096) return "header が読めない";
#if defined(P2_FC1_PREC)
	if (version != VERSION && version != P2_NET_VERSION_FTI32 && version != P2_NET_VERSION_FC1P) return "version が 0x7AF32F20 / 0x7AF32F21 / 0x7AF32F22 でない";
#else
	if (version != VERSION && version != P2_NET_VERSION_FTI32) return "version が 0x7AF32F20 / 0x7AF32F21 でない";
#endif
	net->arch = r.read_string(alen);
	if (!r.ok()) return "arch が読めない";
	// P2_FT_KS (11-18): 新形式の目印 "->I32K<k>:" (p2nnue.cpp の FT_KS の注記)。 このビルドの k と一致すれば取り除く。
	//   無ければ旧形式 (刻み 254) で、 FT を読んだ後で重み・bias を k 倍する。
	int ft_file_k = 1;
	{
		const std::size_t mp = net->arch.find("->I32K");
		if (mp != std::string::npos) {
			std::size_t q = mp + 6;
			int kk = 0;
			while (q < net->arch.size() && net->arch[q] >= '0' && net->arch[q] <= '9') kk = kk * 10 + (net->arch[q++] - '0');
			if (q >= net->arch.size() || net->arch[q] != ':' || kk != FT_K || FT_K == 1)
				return "FT k=" + std::to_string(kk) + " の net はこのビルド (k=" + std::to_string(FT_K) + ") で読めない";
			net->arch.erase(mp + 2, q + 1 - (mp + 2));
			ft_file_k = kk;
		}
	}
#if defined(P2_FC1_PREC)
	// P2_FC1_PREC (11-19): fc1 (外から 2 番目の AffineTransform) の token 名 "AffineTransformF1P128[" が新形式の目印 (版 0x7AF32F22。
	//   p2nnue.cpp の load_network と同じ)。 旧 exe は AffineTransform が 3 つに数えられず、 版でも止まる。 ここで既存の名前に戻す。
	bool fc1p_file = false;
	{
		const std::string f1p = "AffineTransformF1P128[";
		const std::size_t fp = net->arch.find(f1p);
		if (fp != std::string::npos) {
			int before = 0;
			for (std::size_t p = net->arch.find("AffineTransform["); p != std::string::npos && p < fp; p = net->arch.find("AffineTransform[", p + 1)) ++before;
			if (net->arch.find(f1p, fp + 1) != std::string::npos || before != 1 || version != P2_NET_VERSION_FC1P)
				return "FC1PREC の目印は 2 番目の AffineTransform (fc1) に 1 つだけ・版 0x7AF32F22 であること";
			net->arch.replace(fp, f1p.size(), "AffineTransform[");
			fc1p_file = true;
		} else if (version == P2_NET_VERSION_FC1P) {
			return "版 0x7AF32F22 なのに FC1PREC の目印が無い";
		}
	}
#endif
	const std::string& a = net->arch;
	// 4. 禁止 marker (opn_static2.py の BAD と同じ + LSMODE= / PIN= / DUAL=)
	static const char* const BAD[] = { "PSQT=", "Threat", "HandCountDense", "LSMODE=", "AUX", "MOB=", "L4=", "EGTMP", "DUAL=", "PIN=" };
	for (const char* b : BAD)
		if (a.find(b) != std::string::npos) return std::string("arch に禁止 marker '") + b + "' がある (OPN は PSQT / threat / 追加 section の無い 8 head の net だけ): " + a;
	// 2. 入力と幅
	int in_dim = 0, W = 0;
	{
		const std::string k16 = "Features=HalfKA_hmK16(Friend)[", k45 = "Features=HalfKA_hm(Friend)[";
		std::size_t p = a.find(k16);
		if (p != std::string::npos) { net->k16 = true; p += k16.size(); }
		else if ((p = a.find(k45)) != std::string::npos) { net->k16 = false; p += k45.size(); }
		else return "Features=HalfKA_hm(Friend)[ / HalfKA_hmK16(Friend)[ が無い: " + a;
		if (!parse_int(a, p, in_dim) || a.compare(p, 2, "->") != 0) return "入力次元が読めない: " + a;
		p += 2;
		if (!parse_int(a, p, W) || a.compare(p, 3, "x2]") != 0) return "幅が読めない: " + a;
	}
	const int want_rows = net->k16 ? K16_ROWS : K45_ROWS;
	if (in_dim != want_rows) return "入力次元 " + std::to_string(in_dim) + " が " + (net->k16 ? "K16" : "HalfKA_hm") + " の " + std::to_string(want_rows) + " と違う";
	if (W <= 0 || W % 32 != 0 || W > FT_MAX) return "幅 " + std::to_string(W) + " は非対応 (32 の倍数かつ P2_OPN_FT_MAX=" + std::to_string(FT_MAX) + " 以下)";
	// 3. 後段: AffineTransform[a<-b] がちょうど 3 つ = (L1 <- 2W)、 (L2 <- 2(L1−1))、 (1 <- L2)
	int L1 = -1, L2 = -1, n_aff = 0, aff_a[8], aff_b[8];
	{
		const std::string key = "AffineTransform[";
		for (std::size_t p = a.find(key); p != std::string::npos; p = a.find(key, p)) {
			p += key.size();
			int x, y;
			if (!parse_int(a, p, x) || a.compare(p, 2, "<-") != 0) return "AffineTransform が読めない: " + a;
			p += 2;
			if (!parse_int(a, p, y) || p >= a.size() || a[p] != ']') return "AffineTransform が読めない: " + a;
			if (n_aff >= 8) return "AffineTransform が多すぎる: " + a;
			aff_a[n_aff] = x; aff_b[n_aff] = y; ++n_aff;
		}
		if (n_aff != 3) return "AffineTransform が 3 つでない: " + a;
		for (int i = 0; i < 3; ++i) { if (aff_b[i] == 2 * W) L1 = aff_a[i]; if (aff_a[i] == 1) L2 = aff_b[i]; }
		bool mid = false;
		for (int i = 0; i < 3; ++i) if (aff_a[i] == L2 && aff_b[i] == 2 * (L1 - 1)) mid = true;
		if (L1 < 0 || L2 < 0 || !mid) return "後段が [L1<-2W] / [L2<-2(L1-1)] / [1<-L2] の形でない: " + a;
	}
	if (L1 % 8 != 0 || L1 < 8 || L1 > L1_MAX) return "L1=" + std::to_string(L1) + " は非対応 (8 の倍数かつ 32 以下)";
	if (L2 % 8 != 0 || L2 < 8 || L2 > L2_MAX) return "L2=" + std::to_string(L2) + " は非対応 (8 の倍数かつ 64 以下)";
	net->W = W; net->half = W / 2; net->L1 = L1; net->L2 = L2; net->M = L1 - 1;
	net->L2IN_PAD = pad32i(2 * net->M); net->rows = want_rows;
	if (net->L2IN_PAD > 64) return "2(L1-1) が 64 を超える";
	r.read_u32_le();   // ft_hash (K16 と K45 で同じなので照合しない)
	// 5. FT の 2 ブロックの要素数、 7. int16 に収まること
	auto read_leb_i16 = [&](FtVec& v, std::size_t count, const char* name) -> std::string {
		char magic[17];
		if (!r.read_bytes(magic, 17) || std::memcmp(magic, LEB128_MAGIC, 17) != 0) return std::string(name) + ": LEB128 の magic が無い";
		const std::uint32_t comp = r.read_u32_le();
		if (!r.ok() || comp == 0 || !r.has(comp)) return std::string(name) + ": 圧縮長が不正";
		const std::size_t end = r.pos() + comp;
		v.resize(count);
		std::size_t n = 0;
		while (r.pos() < end) {
			const std::int64_t x = r.read_signed_leb128();
			if (!r.ok()) return std::string(name) + ": LEB128 が壊れている";
			if (x < -32768 || x > 32767) return std::string(name) + ": int16 に収まらない値 " + std::to_string((long long) x);
			if (n >= count) return std::string(name) + ": 要素数が " + std::to_string(count) + " を超える";
			v[n++] = (std::int16_t) x;
		}
		if (r.pos() != end || n != count) return std::string(name) + ": 要素数 " + std::to_string(n) + " が " + std::to_string(count) + " と違う";
		return std::string();
	};
	std::string e = read_leb_i16(net->ftb, (std::size_t) W, "ft_bias");
	if (!e.empty()) return e;
	e = read_leb_i16(net->ftw, (std::size_t) want_rows * W, "ft_w");
	if (!e.empty()) return e;
	if (ft_file_k != FT_K) {
		// 旧形式 (刻み 254) の net を k 倍刻みにする (i16 に収まらなければ止める)。
		const int mul = FT_K / ft_file_k;
		for (auto& x : net->ftb) { const int y = (int) x * mul; if (y < -32768 || y > 32767) return "ft_bias の k 倍が i16 を超える"; x = (std::int16_t) y; }
		for (std::size_t i = 0; i < net->ftw.size(); ++i) {
			const int y = (int) net->ftw[i] * mul;
			if (y < -32768 || y > 32767) return "ft_w の k 倍が i16 を超える";
			net->ftw[i] = (std::int16_t) y;
		}
	}
	// 6. 残りバイト数 = head 1 本 × 8
	const std::size_t head_bytes = 4 + 4 * (std::size_t) L1 + (std::size_t) L1 * pad32i(W) + 4 * (std::size_t) L2
	                             + (std::size_t) L2 * net->L2IN_PAD + 4 + (std::size_t) pad32i(L2);
	if (r.remaining() != head_bytes * NHEADS)
		return "head の残り " + std::to_string(r.remaining()) + " byte が " + std::to_string(head_bytes) + " x 8 と合わない";
	auto build_ws = [](const std::vector<std::int8_t>& w, auto& ws, int in_pad, int nout) {
		ws.assign((std::size_t) (in_pad / 4) * nout * 4, 0);
		for (int o = 0; o < nout; ++o)
			for (int i = 0; i < in_pad; ++i)
				ws[(std::size_t) (i / 4) * nout * 4 + (std::size_t) o * 4 + (i % 4)] = w[(std::size_t) o * in_pad + i];
	};
	for (int h = 0; h < NHEADS; ++h) {
		Head& hd = net->heads[h];
		r.read_u32_le();   // fc_hash
		for (int o = 0; o < L1; ++o) hd.b0[o] = r.read_i32_le();
		hd.w0.resize((std::size_t) L1 * W);
		for (auto& x : hd.w0) x = (std::int8_t) r.read_u8();
		for (int o = 0; o < L2; ++o) hd.b1[o] = r.read_i32_le();
		hd.w1.resize((std::size_t) L2 * net->L2IN_PAD);
		for (auto& x : hd.w1) x = (std::int8_t) r.read_u8();
		hd.b2 = r.read_i32_le();
		for (int i = 0; i < pad32i(L2); ++i) { const std::int8_t x = (std::int8_t) r.read_u8(); if (i < L2_MAX) hd.w2[i] = x; }
		build_ws(hd.w0, hd.w0ws, W, L1);
#if defined(P2_OPN_R2_KERNELS)
		hd.g8_ok = (L1 == 16) && affine_g8_safe(hd.w0ws.data(), W, L1);
#endif
		build_ws(hd.w1, hd.w1ws, net->L2IN_PAD, L2);
#if defined(P2_FC1_PREC)
		// P2_FC1_PREC: 本体 (p2nnue.cpp の read_fc) と同じ。 新形式は fc1 重み 刻み 128 / b1 127·4·128 / b2 127·4·64。
		//   旧形式は fc1 重み x2・b1 x8・b2 x4 で同じ目盛りに載せる。 i32 に収まらなければ止める。
		{
			const std::int64_t bm1 = fc1p_file ? 1 : 8, bm2 = fc1p_file ? 1 : 4, wm = fc1p_file ? 1 : 2;
			for (int o = 0; o < L2; ++o) {
				const std::int64_t x = (std::int64_t) hd.b1[o] * bm1;
				if (x < INT32_MIN || x > INT32_MAX) return "b1 の x8 が i32 を超える";
				hd.b1[o] = (std::int32_t) x;
			}
			const std::int64_t x2 = (std::int64_t) hd.b2 * bm2;
			if (x2 < INT32_MIN || x2 > INT32_MAX) return "b2 の x4 が i32 を超える";
			hd.b2 = (std::int32_t) x2;
			const int P = net->L2IN_PAD;
			hd.w1ws16.assign((std::size_t) P * L2, 0);
			for (int o = 0; o < L2; ++o)
				for (int i = 0; i < P; ++i)
					hd.w1ws16[(std::size_t) (i / 2) * L2 * 2 + (std::size_t) o * 2 + (i % 2)] = (std::int16_t) (hd.w1[(std::size_t) o * P + i] * wm);
			for (int i = 0; i < L2_MAX; ++i) hd.w2_16[i] = hd.w2[i];
		}
#endif
#if defined(P2_FT_OUT16)
		hd.w0ws16.assign((std::size_t) W * L1, 0);
		for (int o = 0; o < L1; ++o)
			for (int i = 0; i < W; ++i)
				hd.w0ws16[(std::size_t) (i / 2) * L1 * 2 + (std::size_t) o * 2 + (i % 2)] = hd.w0[(std::size_t) o * W + i];
		for (int o = 0; o < L1_MAX; ++o) hd.b0m[o] = o < L1 ? hd.b0[o] * (1 << FT_MS) : 0;
#endif
	}
	if (!r.ok() || r.remaining() != 0) return "head の読み込みに失敗";
	out_net = std::move(net);
	return std::string();
}

// ---------------------------------------------------------------------------
// アキュムレータ (本体と同じ「祖先まで遡って差分をまとめる + Finny」。 遡りの上限は CHAIN_MAX 手)
// ---------------------------------------------------------------------------
struct CacheEntry {
	alignas(64) std::int16_t acc[FT_MAX];
	std::int32_t bp[PIECE_NUMBER_NB];
	bool valid = false;
};
struct Cache {
	CacheEntry e[2][45][2];   // [persp][king_bucket (45 区分)][mirror]。 K16 でも玉移動の判定は 45 区分のまま (設計 2.4)
	int generation = -1;
};
static thread_local Cache g_cache;

static inline const std::int16_t* row(const Net& n, int kb, int packed) {
	const int kr = n.k16 ? (int) KB16.v[kb] : kb;
	return n.ftw.data() + ((std::size_t) kr * PIECE_INPUTS + (std::size_t) packed) * (std::size_t) n.W;
}

// dst[k] = src[k] − Σ subs[s][k] + Σ adds[a][k]  (k < W、 i16 wrapping。 加減の順は本体と同じ)。 dst2 があれば同じ値を書く。
static inline void apply_rows(const std::int16_t* src, std::int16_t* dst, std::int16_t* dst2, int W,
                              const std::int16_t* const* subs, int ns, const std::int16_t* const* adds, int na) {
#if defined(USE_AVX2)
	// SAFETY: src / dst / dst2 / 各行は少なくとも W 要素 (W は 32 の倍数) を持つ (Net::ftw の行、 StateInfo::p2_opn_acc、
	//   CacheEntry::acc はどれも FT_MAX >= W 要素)。 loadu / storeu なので整列は要らない。
	//   不変条件: W % 16 == 0 (load で 32 の倍数を保証)。
	int k = 0;
#if defined(P2_OPN_ROWS64)
	// P3: 本体の ensure_perspective と同じ「64 列ごとに 4 レジスタ、 行は内側」。 列ごとに見れば引く順・足す順は
	//   下の 16 列ループと同じなので値は変わらない。 端数 (W % 64 == 32) は下の 16 列ループで処理する。
	for (; k + 64 <= W; k += 64) {
		__m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 0));
		__m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 16));
		__m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 32));
		__m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 48));
		for (int s = 0; s < ns; ++s) {
			const std::int16_t* q = subs[s] + k;
#if defined(P2_OPN_F2)
			// F2-2: この行の P2_OPN_ROW_PF_LINES line 先 (64 列 = 2 line ぶん) を先読みする。 prefetch は値を読まない
			//   (アーキテクチャ状態を変えない) ので結果は同じ。 行の外は先読みしない (k + PF < W)。
			if (k + P2_OPN_ROW_PF_LINES * 32 < W) {
				_mm_prefetch(reinterpret_cast<const char*>(q + P2_OPN_ROW_PF_LINES * 32), _MM_HINT_T0);
				_mm_prefetch(reinterpret_cast<const char*>(q + P2_OPN_ROW_PF_LINES * 32 + 32), _MM_HINT_T0);
			}
#endif
			v0 = _mm256_sub_epi16(v0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 0)));
			v1 = _mm256_sub_epi16(v1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 16)));
			v2 = _mm256_sub_epi16(v2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32)));
			v3 = _mm256_sub_epi16(v3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 48)));
		}
		for (int x = 0; x < na; ++x) {
			const std::int16_t* q = adds[x] + k;
#if defined(P2_OPN_F2)
			if (k + P2_OPN_ROW_PF_LINES * 32 < W) {
				_mm_prefetch(reinterpret_cast<const char*>(q + P2_OPN_ROW_PF_LINES * 32), _MM_HINT_T0);
				_mm_prefetch(reinterpret_cast<const char*>(q + P2_OPN_ROW_PF_LINES * 32 + 32), _MM_HINT_T0);
			}
#endif
			v0 = _mm256_add_epi16(v0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 0)));
			v1 = _mm256_add_epi16(v1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 16)));
			v2 = _mm256_add_epi16(v2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32)));
			v3 = _mm256_add_epi16(v3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 48)));
		}
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 0), v0);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 16), v1);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 32), v2);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 48), v3);
		if (dst2 != nullptr) {
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 0), v0);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 16), v1);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 32), v2);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 48), v3);
		}
	}
#endif
	for (; k < W; k += 16) {
		__m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k));
		for (int s = 0; s < ns; ++s) v = _mm256_sub_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(subs[s] + k)));
		for (int x = 0; x < na; ++x) v = _mm256_add_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(adds[x] + k)));
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k), v);
		if (dst2 != nullptr) _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k), v);
	}
#else
	for (int k = 0; k < W; ++k) {
		int v = src[k];
		for (int s = 0; s < ns; ++s) v -= subs[s][k];
		for (int x = 0; x < na; ++x) v += adds[x][k];
		dst[k] = (std::int16_t) v;
		if (dst2 != nullptr) dst2[k] = (std::int16_t) v;
	}
#endif
}

#if defined(P2_OPN_F2) && defined(USE_AVX2)
// F2-6: 評価する node の差分更新と transform を 1 回の列ループで行う。 apply_rows と同じ「64 列ごと・行ごとに 1 本の流れ」のまま、
//   後半 (k >= W/2) の列を作った時点で、 前半の同じ位置 (さっき dst に書いた値。 幅 2048 で 2 KB 前なので L1 に在る) と掛けて
//   transform の 1 視点ぶんを out に書く。 評価の直前に書いた acc (視点あたり 4 KB) を transform で読み直さずに済む。
//   実測 (評価ごとに交互に切り替えた rdtsc、 幅 2048): acc +269 / transform −733 = 評価 1 回 −430 cycle。
//   🚨行の流れを前半・後半の 2 本に分ける形 (列 j と j + W/2 を同時に処理) は acc が +1,750 cycle 遅くなり逆効果だった。
//   値: dst[k] は apply_rows と同じ (列ごとの引く順・足す順が同じ、 i16 wrapping)。 out[j] は transform と同じ式
//     (clamp(dst[j],0,254) << 7) と clamp(dst[j + W/2],0,254) の mulhi = 積 >> 9。 最大 (254 << 7) × 254 >> 16 = 126 なので
//     packus (符号付き i16 → u8 の飽和) は飽和しない。 packus は 128 bit lane ごとに詰めるので permute4x64 (0xD8) で列の順に戻す。
// SAFETY: src / dst / 各行は W 要素以上、 out は W/2 byte 以上。 loadu / storeu なので整列は要らない。
//   不変条件: W % 128 == 0 (呼び出し側で確かめる。 W/2 が 64 列の境に合う)。 dst と src は別の配列 (前半を dst から読み直すので。
//   差分更新では base と st は別の StateInfo)。 dst と out は重ならない。
//   R2-3 (2026-11-15): D2 == true なら dst と同じ値を dst2 にも書く (Finny の refresh 用。 このときは src == dst でもよい:
//     64 列ごとに src を読んでから dst に書き、 前半の読み直しは書き終えた列だけを読む)。
template <bool D2>
static inline void apply_rows_tr2_t(const std::int16_t* src, std::int16_t* dst, std::int16_t* dst2, int W, const std::int16_t* const* subs, int ns,
                                    const std::int16_t* const* adds, int na, TrT* out) {
	const int half = W / 2;
	const __m256i c254 = _mm256_set1_epi16(254);
	const __m256i zero = _mm256_setzero_si256();
	for (int k = 0; k < W; k += 64) {
		__m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 0));
		__m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 16));
		__m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 32));
		__m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 48));
		for (int s = 0; s < ns; ++s) {
			const std::int16_t* q = subs[s] + k;
			if (k + P2_OPN_ROW_PF_LINES * 32 < W) {   // F2-2 と同じ先読み
				_mm_prefetch(reinterpret_cast<const char*>(q + P2_OPN_ROW_PF_LINES * 32), _MM_HINT_T0);
				_mm_prefetch(reinterpret_cast<const char*>(q + P2_OPN_ROW_PF_LINES * 32 + 32), _MM_HINT_T0);
			}
			v0 = _mm256_sub_epi16(v0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 0)));
			v1 = _mm256_sub_epi16(v1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 16)));
			v2 = _mm256_sub_epi16(v2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32)));
			v3 = _mm256_sub_epi16(v3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 48)));
		}
		for (int x = 0; x < na; ++x) {
			const std::int16_t* q = adds[x] + k;
			if (k + P2_OPN_ROW_PF_LINES * 32 < W) {
				_mm_prefetch(reinterpret_cast<const char*>(q + P2_OPN_ROW_PF_LINES * 32), _MM_HINT_T0);
				_mm_prefetch(reinterpret_cast<const char*>(q + P2_OPN_ROW_PF_LINES * 32 + 32), _MM_HINT_T0);
			}
			v0 = _mm256_add_epi16(v0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 0)));
			v1 = _mm256_add_epi16(v1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 16)));
			v2 = _mm256_add_epi16(v2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32)));
			v3 = _mm256_add_epi16(v3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 48)));
		}
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 0), v0);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 16), v1);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 32), v2);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 48), v3);
		if (D2) {
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 0), v0);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 16), v1);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 32), v2);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 48), v3);
		}
#if defined(P2_FT_NEWTR)
		if (k >= half) {
			const int j = k - half;
			ft_store32(out + j, ft_prod16_i16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + j + 0)), v0),
			                    ft_prod16_i16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + j + 16)), v1));
			ft_store32(out + j + 32, ft_prod16_i16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + j + 32)), v2),
			                         ft_prod16_i16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + j + 48)), v3));
		}
		(void) c254; (void) zero;
#else
		if (k >= half) {
			const int j = k - half;
			const __m256i a0 = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + j + 0)), c254), zero), 7);
			const __m256i a1 = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + j + 16)), c254), zero), 7);
			const __m256i a2 = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + j + 32)), c254), zero), 7);
			const __m256i a3 = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + j + 48)), c254), zero), 7);
#if defined(P2_TR_STORE32)
			// P2_TR_STORE32: hi 側の max(0) を省く (本体 transform_pair の注記。 負なら積が 0 以下になり packus が 0 にする)。
			const __m256i p0 = _mm256_mulhi_epi16(a0, _mm256_min_epi16(v0, c254));
			const __m256i p1 = _mm256_mulhi_epi16(a1, _mm256_min_epi16(v1, c254));
			const __m256i p2 = _mm256_mulhi_epi16(a2, _mm256_min_epi16(v2, c254));
			const __m256i p3 = _mm256_mulhi_epi16(a3, _mm256_min_epi16(v3, c254));
#else
			const __m256i p0 = _mm256_mulhi_epi16(a0, _mm256_max_epi16(_mm256_min_epi16(v0, c254), zero));
			const __m256i p1 = _mm256_mulhi_epi16(a1, _mm256_max_epi16(_mm256_min_epi16(v1, c254), zero));
			const __m256i p2 = _mm256_mulhi_epi16(a2, _mm256_max_epi16(_mm256_min_epi16(v2, c254), zero));
			const __m256i p3 = _mm256_mulhi_epi16(a3, _mm256_max_epi16(_mm256_min_epi16(v3, c254), zero));
#endif
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j), _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p1), 0xD8));
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j + 32), _mm256_permute4x64_epi64(_mm256_packus_epi16(p2, p3), 0xD8));
		}
#endif
	}
}
static inline void apply_rows_tr2(const std::int16_t* src, std::int16_t* dst, int W, const std::int16_t* const* subs, int ns,
                                  const std::int16_t* const* adds, int na, TrT* out) {
#if defined(P2_OPN_R2_KERNELS)
	// R2-2: 1 手ぶんの差分は (引く 1・足す 1) = 駒を取らない手 / 打つ手 か、 (引く 2・足す 2) = 駒を取る手。
	//   同数で 1..4 本なら定数版 (p2nnue.cpp の r2_rows_fx) で作る。 値は同じ。
	if (R2K(K_TRFX, P2_OPN_R2_TRFX) && ns == na && r2_rows_n<false, true>(ns, src, dst, nullptr, W, subs, adds, out, P2_OPN_ROW_PF_LINES * 32)) return;
#endif
	apply_rows_tr2_t<false>(src, dst, nullptr, W, subs, ns, adds, na, out);
}
#endif

// Finny: (視点, 玉区分, 鏡像) ごとに最後に作った acc と駒の一覧を持ち、 今の局面との差だけ足し引きする。
//   R2-3: tr_out != nullptr なら transform の 1 視点ぶんも同じ列ループで書いて true を返す (-DP2_OPN_R2 のとき)。
//     false のとき tr_out は書かれていない (呼び出し側が acc から transform する)。
static bool refresh_perspective(const Position& pos, Color persp, TrT* tr_out = nullptr) {
	const Net& n = *g_onet;
	StateInfo* st = pos.state();
	const Square ksq = pos.square<KING>(persp);
	const int kb = king_bucket(ksq, persp);
	const bool mirror = is_hm_mirror(ksq, persp);
	Cache& cache = g_cache;
	if (cache.generation != g_generation) {
		for (int p = 0; p < 2; ++p)
			for (int k = 0; k < 45; ++k)
				for (int m = 0; m < 2; ++m) cache.e[p][k][m].valid = false;
		cache.generation = g_generation;
	}
	CacheEntry& en = cache.e[persp][kb][mirror ? 1 : 0];
	if (!en.valid) {
		for (int k = 0; k < n.W; ++k) en.acc[k] = n.ftb[k];
		for (int i = 0; i < PIECE_NUMBER_NB; ++i) en.bp[i] = (std::int32_t) BONA_PIECE_ZERO;
		en.valid = true;
	}
	const BonaPiece* list = (persp == BLACK) ? pos.eval_list()->piece_list_fb() : pos.eval_list()->piece_list_fw();
	const std::int16_t* subs[PIECE_NUMBER_NB]; int ns = 0;
	const std::int16_t* adds[PIECE_NUMBER_NB]; int na = 0;
	for (int i = 0; i < PIECE_NUMBER_NB; ++i) {
		const int ob = (int) en.bp[i], nb = (int) list[i];
		if (ob == nb) continue;
		if (ob != BONA_PIECE_ZERO) subs[ns++] = row(n, kb, pack_bonapiece(ob, mirror));
		if (nb != BONA_PIECE_ZERO) adds[na++] = row(n, kb, pack_bonapiece(nb, mirror));
		en.bp[i] = (std::int32_t) nb;
	}
	P2_OPN_COUNT(ens_refresh); P2_OPN_ADD(refresh_rows, ns + na);
#if defined(P2_OPN_R2_KERNELS) && defined(P2_OPN_F2)
	if (tr_out != nullptr && n.W % 128 == 0 && R2K(K_RFTR, P2_OPN_R2_RFTR)) {
		// R2-2: 引く行と足す行が同数 (駒が動いただけなら必ず同数。 空の entry を埋める初回だけ足す行のみ) なら定数版で 4 対ずつ。
		//   実測: 融合 (R2-3) −65〜−115 cycle / 評価、 定数版 −46 cycle / 評価。
		if (R2K(K_RFX, P2_OPN_R2_RFX) && ns == na && ns >= 1) r2_refresh_tr(ns, en.acc, st->p2_opn_acc[persp], n.W, subs, adds, tr_out, P2_OPN_ROW_PF_LINES * 32);
		else apply_rows_tr2_t<true>(en.acc, en.acc, st->p2_opn_acc[persp], n.W, subs, ns, adds, na, tr_out);
		return true;
	}
#endif
	(void) tr_out;
	apply_rows(en.acc, en.acc, st->p2_opn_acc[persp], n.W, subs, ns, adds, na);
	return false;
}

#if defined(P2_OPN_PROBE)
// P4 の自己照合: bias + Σ (今の局面の全駒の行) を作り直して st の acc と比べる。
static void selfcheck(const Position& pos, Color persp) {
	const Net& n = *g_onet;
	const Square ksq = pos.square<KING>(persp);
	const int kb = king_bucket(ksq, persp);
	const bool mirror = is_hm_mirror(ksq, persp);
	const BonaPiece* list = (persp == BLACK) ? pos.eval_list()->piece_list_fb() : pos.eval_list()->piece_list_fw();
	std::int16_t full[FT_MAX];
	for (int k = 0; k < n.W; ++k) full[k] = n.ftb[k];
	for (int i = 0; i < PIECE_NUMBER_NB; ++i) {
		if ((int) list[i] == BONA_PIECE_ZERO) continue;
		const std::int16_t* w = row(n, kb, pack_bonapiece((int) list[i], mirror));
		for (int k = 0; k < n.W; ++k) full[k] = (std::int16_t) (full[k] + w[k]);
	}
	++probe::sc_n;
	if (std::memcmp(full, pos.state()->p2_opn_acc[persp], sizeof(std::int16_t) * n.W) != 0) ++probe::sc_ng;
}
#endif

#if defined(P2_OPN_F2)
// F2-4: st に至る手で視点 persp の玉が動いたとき、 その前後で「FT の行を引く区分 (K16 の 16 区分) と鏡像」が同じなら true。
//   同じなら、 玉以外の駒の行は動く前と同じ行のままで、 変わるのは玉自身の行 (dirtyPiece に載っている) だけなので、
//   ほかの手と同じ差分で処理できる。 K45 (行の区分 = 45 区分) では玉が動けば必ず区分が変わるので呼ばない。
//   玉の BonaPiece は視点 persp から見た升で f_king + sq (自玉なので f_king 側。 evaluate_bona_piece.cpp の kpp_board_index)。
//   king_bucket / is_hm_mirror と同じ式 (file = sq / 9、 rank = sq % 9、 file >= 5 で鏡像)。
//   不変条件: 玉が動いたのに dirtyPiece に玉が無い / BonaPiece が f_king の範囲に無いときは false (= Finny に落とす)。
static inline bool king_move_keeps_row(const StateInfo* cur, Color persp) {
	const PieceNumber kingNo = (persp == BLACK) ? PIECE_NUMBER_BKING : PIECE_NUMBER_WKING;
	const Eval::DirtyPiece& dp = cur->dirtyPiece;
	for (int i = 0; i < dp.dirty_num; ++i) {
		if (dp.pieceNo[i] != kingNo) continue;
		const int ob = ((persp == BLACK) ? (int) dp.changed_piece[i].old_piece.fb : (int) dp.changed_piece[i].old_piece.fw) - (int) Eval::f_king;
		const int nb = ((persp == BLACK) ? (int) dp.changed_piece[i].new_piece.fb : (int) dp.changed_piece[i].new_piece.fw) - (int) Eval::f_king;
		if (ob < 0 || ob >= 81 || nb < 0 || nb >= 81) return false;
		const int of = ob / 9, nf = nb / 9;
		const bool om = of >= 5, nm = nf >= 5;
		const int okb = (om ? 8 - of : of) * 9 + ob % 9, nkb = (nm ? 8 - nf : nf) * 9 + nb % 9;
		return om == nm && KB16.v[okb] == KB16.v[nkb];
	}
	return false;
}
#endif

#if defined(P2_OPN_F2)
// F2-6: tr_out != nullptr なら、 差分で作れたときに transform の 1 視点ぶん (W/2 byte) も書いて true を返す。
//   false のとき tr_out は書かれていない (計算済みだった / Finny に落ちた / W % 128 != 0) ので、 呼び出し側が acc から transform する。
static bool ensure_perspective(const Position& pos, Color persp, TrT* tr_out = nullptr) {
	StateInfo* st = pos.state();
	if (st->p2_opn_computed[persp]) return false;
	bool tr_done = false;
#else
static void ensure_perspective(const Position& pos, Color persp) {
	StateInfo* st = pos.state();
	if (st->p2_opn_computed[persp]) return;
#endif
	P2_OPN_COUNT(ens);
	StateInfo* chain[CHAIN_MAX];
	int nch = 0;
	StateInfo* cur = st;
	StateInfo* base = nullptr;
	while (true) {
		if (cur->p2_opn_computed[persp]) { base = cur; break; }
		if (cur->previous == nullptr) { P2_OPN_COUNT(refresh_noprev); break; }
#if defined(P2_OPN_F2)
		if (friend_king_moved(cur, persp)) {
			if (!(g_onet->k16 && king_move_keeps_row(cur, persp))) { P2_OPN_COUNT(refresh_king); break; }
			P2_OPN_COUNT(king_same_row);
		}
#else
		if (friend_king_moved(cur, persp)) { P2_OPN_COUNT(refresh_king); break; }
#endif
		if (nch >= CHAIN_MAX) { P2_OPN_COUNT(refresh_far); break; }
		chain[nch++] = cur;
		cur = cur->previous;
	}
	const Net& n = *g_onet;
	bool done = false;
	if (base != nullptr) {
		const Square ksq = pos.square<KING>(persp);
		const int kb = king_bucket(ksq, persp);
		const bool mirror = is_hm_mirror(ksq, persp);
		constexpr int CAP = 4 * CHAIN_MAX + 8;   // 1 手で old / new 各 2 本まで (動いた駒 + 取られた駒)
		const std::int16_t* subs[CAP]; int ns = 0;
		const std::int16_t* adds[CAP]; int na = 0;
		bool overflow = false;
#if defined(P2_OPN_F2) && !defined(P2_OPN_F2_EAGER_PRIME)
		if (nch >= 2) {
			// F2-5: 親 (chain[1] = st->previous) の acc が無いまま 2 手以上を跨ぐときは、 先に親の acc を作って残す。
			//   priming をやめた代わり (p2nnue.cpp の evaluate_with_no_return)。 兄弟の node が同じ親から 1 手ぶんの差分で
			//   済むようになる。 使われない親 (priming の約 25%) の acc は作らずに済む。
			//   値: 親の acc = base + (chain[nch−1] .. chain[1] の差分)。 chain の中では行の区分と鏡像が変わらない
			//   (玉移動で打ち切るか、 F2-4 で同じと確かめてある) ので、 kb / mirror は st と同じものを使える。
			//   その値は親の局面の全駒の行の和 (i16 wrapping) と同じで、 priming が作っていた値と同じ。
			//   不変条件: 1 手の dirtyPiece は高々 2 駒 = sub 2 本・add 2 本なので、 (CHAIN_MAX − 1) 手で各 CAP 本に収まる。
#if defined(P2_ACC_PIECEWISE)
			// P2_ACC_PIECEWISE: 3 手以上 (親まで 2 手以上) を跨ぐときは駒単位に畳む (p2nnue.cpp の fold_pieces)。 値は同じ。
			if (nch >= 3) {
				int pob[PIECE_NUMBER_NB], pnb[PIECE_NUMBER_NB];
				const int np = fold_pieces(chain, nch - 1, 1, persp, pob, pnb);
				if (np < 0) overflow = true;
				for (int t = 0; t < np && !overflow; ++t) {
					if (pob[t] != BONA_PIECE_ZERO) { if (ns >= CAP) { overflow = true; break; } subs[ns++] = row(n, kb, pack_bonapiece(pob[t], mirror)); }
					if (pnb[t] != BONA_PIECE_ZERO) { if (na >= CAP) { overflow = true; break; } adds[na++] = row(n, kb, pack_bonapiece(pnb[t], mirror)); }
				}
			} else
#endif
			for (int i = nch - 1; i >= 1; --i) {
				const Eval::DirtyPiece& dp = chain[i]->dirtyPiece;
				for (int j = 0; j < dp.dirty_num; ++j) {
					const Eval::ChangedBonaPiece& cp = dp.changed_piece[j];
					const int ob = (persp == BLACK) ? (int) cp.old_piece.fb : (int) cp.old_piece.fw;
					const int nb = (persp == BLACK) ? (int) cp.new_piece.fb : (int) cp.new_piece.fw;
					if (ob != BONA_PIECE_ZERO) { if (ns >= CAP) { overflow = true; break; } subs[ns++] = row(n, kb, pack_bonapiece(ob, mirror)); }
					if (nb != BONA_PIECE_ZERO) { if (na >= CAP) { overflow = true; break; } adds[na++] = row(n, kb, pack_bonapiece(nb, mirror)); }
				}
				if (overflow) break;
			}
			if (!overflow) {
				{
					R2T_SCOPE(R_LAZY);
#if defined(P2_OPN_R2_KERNELS)
					// R2-2: 同数で 1..4 本なら定数版。 それ以外は汎用版 (値は同じ)。 実測 −43 cycle / 評価。
					if (!(R2K(K_LZX, P2_OPN_R2_LZX) && ns == na && n.W % 64 == 0
					      && r2_rows_n<false, false>(ns, base->p2_opn_acc[persp], chain[1]->p2_opn_acc[persp], nullptr, n.W, subs, adds, nullptr, P2_OPN_ROW_PF_LINES * 32)))
#endif
					apply_rows(base->p2_opn_acc[persp], chain[1]->p2_opn_acc[persp], nullptr, n.W, subs, ns, adds, na);
				}
				chain[1]->p2_opn_computed[persp] = true;
				P2_OPN_COUNT(lazy_parent);
				base = chain[1]; nch = 1;
			}
			ns = 0; na = 0; overflow = false;   // 溢れたら (到達しない) 従来どおり base から 1 回で作る
		}
#endif
#if defined(P2_ACC_PIECEWISE)
		if (nch >= 2) {
			int pob[PIECE_NUMBER_NB], pnb[PIECE_NUMBER_NB];
			const int np = fold_pieces(chain, nch - 1, 0, persp, pob, pnb);
			if (np < 0) overflow = true;
			for (int t = 0; t < np && !overflow; ++t) {
				if (pob[t] != BONA_PIECE_ZERO) { if (ns >= CAP) { overflow = true; break; } subs[ns++] = row(n, kb, pack_bonapiece(pob[t], mirror)); }
				if (pnb[t] != BONA_PIECE_ZERO) { if (na >= CAP) { overflow = true; break; } adds[na++] = row(n, kb, pack_bonapiece(pnb[t], mirror)); }
			}
#if defined(P2_ACC_PIECEWISE_VERIFY)
			++PieceVerify::n_opn_multi;
#endif
		} else
#endif
		for (int i = nch - 1; i >= 0 && !overflow; --i) {
			const Eval::DirtyPiece& dp = chain[i]->dirtyPiece;
			for (int j = 0; j < dp.dirty_num; ++j) {
				const Eval::ChangedBonaPiece& cp = dp.changed_piece[j];
				const int ob = (persp == BLACK) ? (int) cp.old_piece.fb : (int) cp.old_piece.fw;
				const int nb = (persp == BLACK) ? (int) cp.new_piece.fb : (int) cp.new_piece.fw;
				if (ob != BONA_PIECE_ZERO) { if (ns >= CAP) { overflow = true; break; } subs[ns++] = row(n, kb, pack_bonapiece(ob, mirror)); }
				if (nb != BONA_PIECE_ZERO) { if (na >= CAP) { overflow = true; break; } adds[na++] = row(n, kb, pack_bonapiece(nb, mirror)); }
			}
		}
		if (!overflow) {
#if defined(P2_OPN_F2) && defined(USE_AVX2)
			if (tr_out != nullptr && n.W % 128 == 0) {
				{ R2T_SCOPE(R_ACC1); apply_rows_tr2(base->p2_opn_acc[persp], st->p2_opn_acc[persp], n.W, subs, ns, adds, na, tr_out); }
				tr_done = true;
			} else
#endif
			apply_rows(base->p2_opn_acc[persp], st->p2_opn_acc[persp], nullptr, n.W, subs, ns, adds, na);
			P2_OPN_COUNT(ens_chain); P2_OPN_ADD(chain_len, nch); P2_OPN_ADD(chain_rows, ns + na);
			done = true;
		} else {
			P2_OPN_COUNT(refresh_overflow);
		}
	}
#if defined(P2_OPN_F2)
	if (!done) { R2T_SCOPE(R_REFR); if (refresh_perspective(pos, persp, tr_out)) tr_done = true; }
#else
	if (!done) refresh_perspective(pos, persp);
#endif
	st->p2_opn_computed[persp] = true;
#if defined(P2_OPN_PROBE)
	selfcheck(pos, persp);
#endif
#if defined(P2_ACC_PIECEWISE_VERIFY)
	// FTI32 (11-18): OPN の検算は元のソースでも不一致が出る (FTI32 と無関係) ので、 環境変数 FTI32_SKIP_OPNCHK で外せるようにする。
	static const bool skip_opnchk = std::getenv("FTI32_SKIP_OPNCHK") != nullptr;
	if (done && !skip_opnchk) {
		const Square vk = pos.square<KING>(persp);
		const int vkb = king_bucket(vk, persp);
		const bool vm = is_hm_mirror(vk, persp);
		const BonaPiece* vl = (persp == BLACK) ? pos.eval_list()->piece_list_fb() : pos.eval_list()->piece_list_fw();
		std::int16_t full[FT_MAX];
		for (int k = 0; k < n.W; ++k) full[k] = n.ftb[k];
		for (int i = 0; i < PIECE_NUMBER_NB; ++i) {
			if ((int) vl[i] == BONA_PIECE_ZERO) continue;
			const std::int16_t* w = row(n, vkb, pack_bonapiece((int) vl[i], vm));
			for (int k = 0; k < n.W; ++k) full[k] = (std::int16_t) (full[k] + w[k]);
		}
		++PieceVerify::n_opn;
		for (int k = 0; k < n.W; ++k)
			if (full[k] != st->p2_opn_acc[persp][k]) PieceVerify::fail("opn_acc", k, full[k], st->p2_opn_acc[persp][k], pos);
	}
#endif
#if defined(P2_OPN_F2)
	return tr_done;
#endif
}

static inline void ensure_both(const Position& pos) {
	ensure_perspective(pos, BLACK);
	ensure_perspective(pos, WHITE);
}

// ---------------------------------------------------------------------------
// 順伝播
// ---------------------------------------------------------------------------
// pmask: bit p が立っている視点だけ書く (F2-6 で apply_rows_tr2 が書いた視点を飛ばす)。 既定 3 = 両方。
static inline void transform(const Net& n, const std::int16_t* a_us, const std::int16_t* a_them, TrT* tr, int pmask = 3) {
	const int half = n.half;
	for (int p = 0; p < 2; ++p) {
		if (!(pmask & (1 << p))) continue;
		const std::int16_t* a = (p == 0) ? a_us : a_them;
		TrT* out = tr + p * half;
#if defined(P2_FT_NEWTR)
		// P2_FT_NEWTR: ft_prod16_i16 / ft_store32 (p2nnue.cpp) で書く。 SAFETY: a は W 要素、 out は half 要素、 half % 32 == 0
		//   (load で W % 64 == 0 を確かめる)。
#if defined(USE_AVX2)
		for (int j = 0; j < half; j += 32)
			ft_store32(out + j, ft_prod16_i16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j)), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j + half))),
			                    ft_prod16_i16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j + 16)), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j + 16 + half))));
#else
		for (int j = 0; j < half; ++j) out[j] = ft_pair_out(clampi(a[j], 0, FT_CLAMP), clampi(a[j + half], 0, FT_CLAMP));
#endif
		continue;
#endif
#if defined(USE_AVX2)
		// SAFETY: a は W 要素、 out は half 要素 (tr は FT_MAX 要素)。 half は 16 の倍数 (W が 32 の倍数)。
		//   (lo << 7) は最大 254 × 128 = 32,512 で int16 に収まる。 mulhi で (lo × hi) >> 9 (本体の transform_pair と同じ)。
		const __m256i c254 = _mm256_set1_epi16(254);
		const __m256i zero = _mm256_setzero_si256();
#if defined(P2_TR_STORE32)
		// P2_TR_STORE32: 本体の transform_pair と同じ形 (32 列ごとに packus → permute4x64 → 32 byte の store 1 回、 hi 側の max(0) を省く)。
		//   値は従来と同じ。 不変条件: half % 32 == 0 (load で W % 64 == 0 を確かめる。 下の端数は 16 列で処理する)。
		int j = 0;
		for (; j + 32 <= half; j += 32) {
			const __m256i l0 = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j)), c254), zero), 7);
			const __m256i l1 = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j + 16)), c254), zero), 7);
			const __m256i p0 = _mm256_mulhi_epi16(l0, _mm256_min_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j + half)), c254));
			const __m256i p1 = _mm256_mulhi_epi16(l1, _mm256_min_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j + 16 + half)), c254));
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j), _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p1), 0xD8));
		}
		for (; j < half; j += 16) {
#else
		for (int j = 0; j < half; j += 16) {
#endif
			__m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
			__m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j + half));
			lo = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(lo, c254), zero), 7);
			hi = _mm256_max_epi16(_mm256_min_epi16(hi, c254), zero);
			const __m256i prod = _mm256_mulhi_epi16(lo, hi);
			const __m256i packed = _mm256_packus_epi16(prod, prod);
			_mm_storel_epi64(reinterpret_cast<__m128i*>(out + j), _mm256_castsi256_si128(packed));
			_mm_storel_epi64(reinterpret_cast<__m128i*>(out + j + 8), _mm256_extracti128_si256(packed, 1));
		}
#else
		for (int j = 0; j < half; ++j) {
			const int x = clampi(a[j], 0, 254), y = clampi(a[j + half], 0, 254);
			out[j] = (std::uint8_t) ((x * y) >> 9);
		}
#endif
	}
}

#if defined(USE_AVX2)
// P1 (ANALYSIS-20261115 6 章): affine_par の nout を定数リテラルにして呼ぶ。 nout が実行時の値だと ng = nout / 8 が
//   定数にならず、 累算器 acc[8] がレジスタに載らずメモリ経由の依存鎖になっていた (chunk 1 個 約 10 cycle)。
//   [[clang::always_inline]] で case ごとに affine_par を展開させ、 定数を伝える。
//   bit-exact の根拠: 呼ぶ関数も引数も同じ (nout の値が同じ)。 疎版 (affine_par_sparse) から密版に替えた分は、
//   0 の chunk の寄与が 0 なので同じ整数の積和 (p2nnue.cpp の P2_SPARSE_FC0 の注記)。
//   不変条件: nout は 8 の倍数で 8..64 (load で L1 ≤ 32、 L2 ≤ 64 を検査済み)。 in_pad は 4 の倍数。
template <int NOUT>
static inline void affine_fixed(const std::uint8_t* in, const std::int8_t* ws, const std::int32_t* bias, int in_pad, std::int32_t* out) {
	[[clang::always_inline]] affine_par(in, ws, bias, in_pad, NOUT, out);
}
#if defined(P2_OPN_F2)
// F2-1: fc0 専用の密な積和。 affine_par と同じ式・同じ順 (chunk の昇順に各 lane へ足す) で、 重みを
//   P2_OPN_FC0_PF_BYTES 先まで先読みするところだけが違う。 幅 2048 × L1 16 の重みは 32 KB で L1 キャッシュ (32 KB) に
//   居続けられず、 評価のたびに L2 から 512 line を読む。 実測 (評価ごとに先読みの有無を交互に切り替えた rdtsc) で
//   fc0 + 後段が −16% (幅 2048) / −15〜17% (幅 1024)。
// SAFETY: in は in_pad byte 以上、 ws は (in_pad / 4) × NOUT × 4 byte、 bias / out は NOUT 要素。 loadu / storeu。
//   prefetch は配列の終端より先のアドレスも指すが、 prefetch は fault しない (読まない) ので安全。
//   不変条件: NOUT は 8 の倍数 (8..32)、 in_pad は 4 の倍数。 maddubs の飽和条件は affine_par と同じ (ヘッダ冒頭の 4)。
template <int NOUT>
static inline void affine_fc0_pf(const std::uint8_t* __restrict in, const std::int8_t* __restrict ws, const std::int32_t* __restrict bias,
                                 int in_pad, std::int32_t* __restrict out) {
	constexpr int NG = NOUT / 8;      // 8 出力ごとの lane 群
	constexpr int CB = NOUT * 4;      // 1 chunk (入力 4 byte) の重みの byte 数
	__m256i acc[NG];
	for (int g = 0; g < NG; ++g) acc[g] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + g * 8));
	const __m256i ones = _mm256_set1_epi16(1);
	const int nc = in_pad / 4;
	for (int ic = 0; ic < nc; ++ic) {
		std::int32_t a4;
		std::memcpy(&a4, in + ic * 4, 4);
		const __m256i a = _mm256_set1_epi32(a4);
		const std::int8_t* wb = ws + (std::size_t) ic * CB;
		for (int b = 0; b < CB; b += 64) _mm_prefetch(reinterpret_cast<const char*>(wb) + P2_OPN_FC0_PF_BYTES + b, _MM_HINT_T0);
		for (int g = 0; g < NG; ++g)
			acc[g] = _mm256_add_epi32(acc[g], _mm256_madd_epi16(_mm256_maddubs_epi16(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wb + g * 32))), ones));
	}
	for (int g = 0; g < NG; ++g) _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + g * 8), acc[g]);
}
static inline void affine_fc0(const std::uint8_t* in, const std::int8_t* ws, const std::int32_t* bias, int in_pad, int nout, std::int32_t* out) {
	switch (nout) {
	case 8:  affine_fc0_pf<8>(in, ws, bias, in_pad, out); return;
	case 16: affine_fc0_pf<16>(in, ws, bias, in_pad, out); return;
	case 24: affine_fc0_pf<24>(in, ws, bias, in_pad, out); return;
	case 32: affine_fc0_pf<32>(in, ws, bias, in_pad, out); return;
	default: affine_par(in, ws, bias, in_pad, nout, out); return;   // load の検査 (L1 ≤ 32) で到達しない
	}
}
#endif
static inline void affine_n(const std::uint8_t* in, const std::int8_t* ws, const std::int32_t* bias, int in_pad, int nout, std::int32_t* out) {
	switch (nout) {
	case 8:  affine_fixed<8>(in, ws, bias, in_pad, out); return;
	case 16: affine_fixed<16>(in, ws, bias, in_pad, out); return;
	case 24: affine_fixed<24>(in, ws, bias, in_pad, out); return;
	case 32: affine_fixed<32>(in, ws, bias, in_pad, out); return;
	case 40: affine_fixed<40>(in, ws, bias, in_pad, out); return;
	case 48: affine_fixed<48>(in, ws, bias, in_pad, out); return;
	case 56: affine_fixed<56>(in, ws, bias, in_pad, out); return;
	case 64: affine_fixed<64>(in, ws, bias, in_pad, out); return;
	default: affine_par(in, ws, bias, in_pad, nout, out); return;   // load の検査で到達しない
	}
}
#endif

static inline std::int32_t propagate(const Net& n, const Head& h, const TrT* tr) {
	R2T_SCOPE(R_OPNFWD);
	std::int32_t l1[L1_MAX];
#if defined(P2_FT_OUT16)
	// P2_FT_OUT16: fc0 を i16 入力で回す (p2nnue.cpp の affine16_pf_t。 l1 = (m·b0 + Σ x·w) >> log2 m)。 L1 は load で 8..32 の 8 の倍数。
	switch (n.L1) {
	case 8:  P2_FC0_16<8>(tr, h.w0ws16.data(), h.b0m, n.W, l1); break;
	case 16: P2_FC0_16<16>(tr, h.w0ws16.data(), h.b0m, n.W, l1); break;
	case 24: P2_FC0_16<24>(tr, h.w0ws16.data(), h.b0m, n.W, l1); break;
	default: P2_FC0_16<32>(tr, h.w0ws16.data(), h.b0m, n.W, l1); break;
	}
#elif defined(USE_AVX2)
#if defined(P2_OPN_PROBE)
	{
		std::uint16_t nnz[FT_MAX / 4 + 8];
		P2_OPN_COUNT(fc0_n); P2_OPN_ADD(fc0_nnz, find_nnz_chunks(tr, n.W, nnz)); P2_OPN_ADD(fc0_chunks, n.W / 4);
	}
#endif
#if defined(P2_OPN_R2_KERNELS) && defined(P2_OPN_F2)
	// R2-1: 重みが条件を満たす head (L1 = 16) は G8 で計算する (値は同じ。 p2nnue.cpp の affine_g8 の注記)。
	if (h.g8_ok && R2K(K_G8O, P2_OPN_R2_G8_OPN)) affine_g8<16>(tr, h.w0ws.data(), h.b0, n.W, l1);
	else affine_fc0(tr, h.w0ws.data(), h.b0, n.W, n.L1, l1);
#elif defined(P2_OPN_F2)
	affine_fc0(tr, h.w0ws.data(), h.b0, n.W, n.L1, l1);
#else
	affine_n(tr, h.w0ws.data(), h.b0, n.W, n.L1, l1);
#endif
#else
	for (int o = 0; o < n.L1; ++o) l1[o] = h.b0[o] + dot_u8i8(tr, &h.w0[(std::size_t) o * n.W], n.W);
#endif
#if defined(P2_FC1_PREC)
	// P2_FC1_PREC (11-19): 本体 (p2nnue.cpp の propagate_bucket) と同じ。 fc0 活性 0..508 (shift 19 -> 17 / 6 -> 4)、
	//   fc1 = i16 の madd (>> 7、 clamp 0..508)、 fc2 = i16 の内積、 (x + 2) >> 2 で 8128 目盛りに戻す。
	alignas(32) std::int16_t l2in[64] = {};
	for (int i = 0; i < n.M; ++i) {
		const std::int32_t v = l1[i];
		const std::int32_t sqr = (std::int32_t) (((std::int64_t) v * v) >> 17);
		l2in[i] = (std::int16_t) clampi(sqr, 0, FC1P_ACT_MAX);
		l2in[n.M + i] = (std::int16_t) clampi(v >> 4, 0, FC1P_ACT_MAX);
	}
	alignas(32) std::int16_t l2relu[L2_MAX] = {};
	alignas(32) std::int32_t l2out[L2_MAX];
	switch (n.L2) {   // load の検査で L2 は 8..64 の 8 の倍数
	case 8:  affine16_t<8>(l2in, h.w1ws16.data(), h.b1, n.L2IN_PAD, l2out); break;
	case 16: affine16_t<16>(l2in, h.w1ws16.data(), h.b1, n.L2IN_PAD, l2out); break;
	case 24: affine16_t<24>(l2in, h.w1ws16.data(), h.b1, n.L2IN_PAD, l2out); break;
	case 32: affine16_t<32>(l2in, h.w1ws16.data(), h.b1, n.L2IN_PAD, l2out); break;
	case 40: affine16_t<40>(l2in, h.w1ws16.data(), h.b1, n.L2IN_PAD, l2out); break;
	case 48: affine16_t<48>(l2in, h.w1ws16.data(), h.b1, n.L2IN_PAD, l2out); break;
	case 56: affine16_t<56>(l2in, h.w1ws16.data(), h.b1, n.L2IN_PAD, l2out); break;
	default: affine16_t<64>(l2in, h.w1ws16.data(), h.b1, n.L2IN_PAD, l2out); break;
	}
	for (int o = 0; o < n.L2; ++o) l2relu[o] = (std::int16_t) clampi(l2out[o] >> FC1P_FC1_SH, 0, FC1P_ACT_MAX);
	const std::int32_t out4 = h.b2 + dot_i16(l2relu, h.w2_16, n.L2);
	return ((out4 + 2) >> 2) + l1[n.M];
#else
	alignas(32) std::uint8_t l2in[64] = {};
	for (int i = 0; i < n.M; ++i) {
		const std::int32_t v = l1[i];
		const std::int32_t sqr = (std::int32_t) (((std::int64_t) v * v) >> 19);
		l2in[i] = (std::uint8_t) clampi(sqr, 0, 127);
		l2in[n.M + i] = (std::uint8_t) clampi(v >> 6, 0, 127);
	}
	alignas(32) std::uint8_t l2relu[L2_MAX] = {};
#if defined(USE_AVX2)
	std::int32_t l2out[L2_MAX];
	affine_n(l2in, h.w1ws.data(), h.b1, n.L2IN_PAD, n.L2, l2out);
	for (int o = 0; o < n.L2; ++o) l2relu[o] = (std::uint8_t) clampi(l2out[o] >> 6, 0, 127);
#else
	for (int o = 0; o < n.L2; ++o) {
		const std::int32_t s = h.b1[o] + dot_u8i8(l2in, &h.w1[(std::size_t) o * n.L2IN_PAD], n.L2IN_PAD);
		l2relu[o] = (std::uint8_t) clampi(s >> 6, 0, 127);
	}
#endif
	return h.b2 + dot_u8i8(l2relu, h.w2, n.L2) + l1[n.M];
#endif  // P2_FC1_PREC
}

// OPN の raw (手番側視点、 目盛り合わせ前)。 head = 進行度 bucket (0..7)。
static inline std::int32_t eval_raw(const Position& pos, int head) {
	R2T_SCOPE(R_OPN);
	const Net& n = *g_onet;
	const StateInfo* st = pos.state();
	const Color stm = pos.side_to_move();
#if defined(P2_OPN_F2)
	// F2-6: 差分で作る視点は transform まで同じループで済ませる。 残った視点 (計算済み / Finny) だけ acc から transform する。
	//   tr の並びは従来と同じ ([0, W/2) = 手番側、 [W/2, W) = 相手側)。
	alignas(64) TrT tr[FT_MAX];
	int pmask = 3;
	if (ensure_perspective(pos, stm, tr)) pmask &= ~1;
	if (ensure_perspective(pos, ~stm, tr + n.half)) pmask &= ~2;
	if (pmask != 0) { R2T_SCOPE(R_TRF); transform(n, st->p2_opn_acc[stm], st->p2_opn_acc[~stm], tr, pmask); }
#if defined(P2_TR_STORE32_VERIFY)
	// 検算: 融合核 (apply_rows_tr2 / r2_rows_fx / refresh) と transform が書いた tr を scalar の式と比べる。
	for (int p = 0; p < 2; ++p) {
		const std::int16_t* a = st->p2_opn_acc[p == 0 ? stm : ~stm];
		for (int j = 0; j < n.half; ++j) {
			const int ref = (int) ft_pair_out(clampi(a[j], 0, FT_CLAMP), clampi(a[j + n.half], 0, FT_CLAMP));
			if (ref != (int) tr[p * n.half + j]) {
				std::fprintf(stderr, "TR_STORE32_VERIFY MISMATCH opn p=%d j=%d ref=%d got=%d sfen=%s\n", p, j, ref, (int) tr[p * n.half + j], pos.sfen().c_str());
				std::abort();
			}
		}
	}
	++TrVerify::n_opn;
#endif
#else
	ensure_both(pos);
	alignas(32) TrT tr[FT_MAX];
	transform(n, st->p2_opn_acc[stm], st->p2_opn_acc[~stm], tr);
#endif
	return propagate(n, n.heads[head < 0 ? 0 : (head > NHEADS - 1 ? NHEADS - 1 : head)], tr);
}

// ---------------------------------------------------------------------------
// load_eval から呼ぶ。 path 空なら無効。 失敗したら **エンジンを止める** (黙って OPN 無しで走らせない)。
// ---------------------------------------------------------------------------
static inline void load(const std::string& path, bool progress_loaded, int lazy_threat_margin, int lazy_mode) {
	if (path.empty()) {
		g_loaded = false; g_onet.reset(); g_loaded_path.clear(); g_loaded_size = 0;
		update_derived();
		return;
	}
	auto die = [&](const std::string& why) {
		sync_cout << "info string Error! OPNFile: " << why << sync_endl;
		std::fprintf(stderr, "Error! OPNFile: %s\n", why.c_str());
		std::exit(EXIT_FAILURE);
	};
	// 併用できない設定 (設計書 6.4 / 8.3): どちらも本体の値で窓の判定をするので OPN だけの node と両立しない。
	if (!progress_loaded) die("LS_PROGRESS_COEFF (progress.bin) が読めていない。 OPN は進行度で混ぜるので必須");
	if (lazy_threat_margin > 0) die("P2LazyThreatMargin > 0 とは併用できない");
	if (lazy_mode == 2) die("P2LazyMode = 2 とは併用できない");
	// 同じファイル (path と大きさが同じ) なら読み直さない (isready のたびに数百 MB を復号しない)。
	std::uint64_t size_now = 0;
	{
		std::ifstream f(path, std::ios::binary | std::ios::ate);
		if (!f) die("開けない: " + path);
		size_now = (std::uint64_t) f.tellg();
	}
	if (!(g_loaded && g_onet && path == g_loaded_path && size_now == g_loaded_size)) {
		std::unique_ptr<Net> net;
		std::uint64_t sz = 0;
		const std::string err = load_impl(path, net, sz);
		if (!err.empty()) die(err);
		g_onet = std::move(net);
		g_loaded_path = path; g_loaded_size = sz;
		++g_generation;
	}
	g_loaded = true;
	update_derived();
	char hx[16];
	std::snprintf(hx, sizeof(hx), "%08x", (unsigned) g_onet->nethash);
	sync_cout << "info string P2OPN loaded: " << path << " ft=" << g_onet->W << " l1=" << g_onet->L1 << " l2=" << g_onet->L2
	          << " input=" << (g_onet->k16 ? "K16" : "K45") << " rows=" << g_onet->rows << " nethash=" << hx
	          << " lam_max=" << g_lam_max << " prog_hi=" << g_prog_hi << " width=" << g_prog_width
	          << " scale=" << g_scale[0] << "," << g_scale[1] << " offset_cp=" << g_offset_cp
	          << " active=" << (g_active ? 1 : 0) << " chain_max=" << CHAIN_MAX << " main_chain_limit=" << g_acc_chain_limit
	          << " ftw_mod64=" << (unsigned) (reinterpret_cast<std::uintptr_t>(g_onet->ftw.data()) % 64)
#if defined(P2_OPN_F2)
	          << " f2=1 ftw_mod4096=" << (unsigned) (reinterpret_cast<std::uintptr_t>(g_onet->ftw.data()) % 4096)
	          << " w0_mod64=" << (unsigned) (reinterpret_cast<std::uintptr_t>(g_onet->heads[0].w0ws.data()) % 64)
	          << " row_pf=" << P2_OPN_ROW_PF_LINES << " fc0_pf=" << P2_OPN_FC0_PF_BYTES
#if defined(P2_OPN_R2_KERNELS)
	          << " r2=1 g8=" << (g_onet->heads[0].g8_ok ? 1 : 0) << (g_onet->heads[1].g8_ok ? 1 : 0) << (g_onet->heads[2].g8_ok ? 1 : 0)
#endif
#endif
#if defined(P2_OPN_ROWS64)
	          << " rows64=1"
#endif
#if defined(P2_OPN_PROBE)
	          << " probe=1"
#endif
	          << sync_endl;
}

} // namespace opn

#endif // P2_OPN
