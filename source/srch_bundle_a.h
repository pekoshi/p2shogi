#ifndef SRCH_BUNDLE_A_H_INCLUDED
#define SRCH_BUNDLE_A_H_INCLUDED

// 束 A (2026-11-12): 探索部の bit-exact な高速化 (設計 fable / 実装 opus)。
//   -DP2_SRCH_BUNDLE_A のときだけ include される (未定義なら前処理後のソースは変更前と同一)。
//   -DP2_SRCH_BUNDLE_A_VERIFY を足すと、 各変更点で旧式と新式を両方計算して照合し、
//   不一致なら stderr に出して abort する。 照合回数は終了時に stderr へ BUNDLEA_VERIFY 行で出す。
//
//   M1 : NonPV の reduction() の idiv (delta==1 なので MULT / rootDelta を go の窓ごとに 1 回だけ割る)
//   L1 : MovePicker の quiet 採点の速い経路 (本番で常に偽の機能を外した noinline 関数)
//   L2 : SPSA パラメータによる除算を magic number の乗算に置き換える (このファイルの SDiv32)
//   L5 : node 入口の TLS 書き込みを、 値が読まれない設定のときは省く
//   L6 : search<NonPV> を depth<=0 で呼ぶ箇所を、 呼び出し側で qsearch に振り分ける

#if defined(P2_SRCH_BUNDLE_A)

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "types.h"
#include "spsa_params.h"

namespace YaneuraOu::BundleA {

// 正の除数 d (1 <= d <= 2^31-1) による符号付き 32bit 整数の除算 n / d (C++ と同じ 0 方向切り捨て) を
// 乗算 1 回とシフトで行う (libdivide / Granlund-Montgomery の round-up 法)。
//   l = ceil(log2 d), k = 31 + l, M = ceil(2^k / d) とおくと、 0 <= x <= 2^31 のすべての x で
//   floor(x * M / 2^k) == floor(x / d) が成り立つ。
//   証明: e = M*d - 2^k とおくと 0 <= e <= d-1 < 2^l。 x = q*d + r (0 <= r <= d-1) と書くと
//     x*M/2^k = q + (r + x*e/2^k) / d。 x*e < 2^31 * 2^l = 2^k より x*e/2^k < 1 なので
//     r + x*e/2^k < d となり、 床は q に一致する。
//   不変条件: d >= 1 (呼び出し側の option 範囲が下限 1 を保証する。 VERIFY で毎回照合)。
//     d > 2^(l-1) より M < 2^32 + 1 なので M <= 2^32、 x*M <= 2^63 で uint64 に収まる。
//   負の n は |n| を割って符号を戻す (= 0 方向切り捨て)。 |INT_MIN| = 2^31 も上の範囲に入る。
struct SDiv32 {
    std::uint64_t m  = std::uint64_t(1) << 31;  // d = 1 のときの値
    int           sh = 31;

    static constexpr SDiv32 make(int d) {
        int l = 0;
        while ((std::uint64_t(1) << l) < std::uint64_t(d))
            ++l;
        SDiv32 r;
        r.sh = 31 + l;
        r.m  = ((std::uint64_t(1) << r.sh) + std::uint64_t(d) - 1) / std::uint64_t(d);
        return r;
    }

    int div(int n) const {
        const std::int64_t  s = std::int64_t(n) >> 63;                     // n < 0 なら -1、 それ以外 0
        const std::uint64_t a = std::uint64_t((std::int64_t(n) ^ s) - s);  // |n| (<= 2^31)
        const std::int64_t  q = std::int64_t((a * m) >> sh);               // |n| / d (<= 2^31)
        return int((q ^ s) - s);
    }
};

// L2: 探索中に SPSA パラメータで割る箇所の magic number。
//   🚨パラメータは setoption でしか変わらず、 探索中は不変。 Worker ごとに go の開始時
//   (start_searching) に作り直すので、 go の前に setoption で変えた値は必ず反映される。
struct SearchDivs {
    SDiv32 futilityCorr;       // FUTILITY_CORRECTION_DIV        (futility_margin の |corr| 項)
    SDiv32 sfFutilityV2Corr;   // SF_FUTILITY_V2_CORR_DIV        (SF_FUTILITY_V2 の |corr| 項)
    SDiv32 nmpDepth;           // NMP_REDUCTION_DEPTH_DIV        (null move の R)
    SDiv32 conthistLmrDepth;   // CONTHIST_PRUNING_LMRDEPTH_DIV  (lmrDepth += history / X)
    SDiv32 lmrCorr;            // LMR_CORRECTION_DIV             (r -= |corr| / X)
    SDiv32 faillowStatScore;   // FAILLOW_BONUS_STATSCORE_DIV    (fail low の bonusScale)
    SDiv32 statBonusPrev;      // STATBONUS_PREV_DIV             (update_all_stats。 0 のときは割らない)

    void refresh() {
        futilityCorr     = SDiv32::make(SPSA::FUTILITY_CORRECTION_DIV);
        sfFutilityV2Corr = SDiv32::make(SPSA::SF_FUTILITY_V2_CORR_DIV);
        nmpDepth         = SDiv32::make(SPSA::NMP_REDUCTION_DEPTH_DIV);
        conthistLmrDepth = SDiv32::make(SPSA::CONTHIST_PRUNING_LMRDEPTH_DIV);
        lmrCorr          = SDiv32::make(SPSA::LMR_CORRECTION_DIV);
        faillowStatScore = SDiv32::make(SPSA::FAILLOW_BONUS_STATSCORE_DIV);
        // 0 は「項を使わない」の意味で、 使用箇所が 0 を見て割らないので、 ここでは 1 で埋めておく。
        statBonusPrev = SDiv32::make(SPSA::STATBONUS_PREV_DIV > 0 ? SPSA::STATBONUS_PREV_DIV : 1);
    }
};

// L2: MovePicker の GOOD_CAPTURE の `-value / GOOD_CAPTURE_SEE_DIV`。
//   MovePicker は Worker を持たないので global に置き、 setoption のコールバック
//   (p2nnue.cpp の SPSA_GOOD_CAPTURE_SEE_DIV) で作り直す。 初期値は既定値から作る
//   (GOOD_CAPTURE_SEE_DIV は定数初期化なので、 この動的初期化より必ず先に値が入っている)。
inline SDiv32 goodCaptureSeeDiv = SDiv32::make(SPSA::GOOD_CAPTURE_SEE_DIV);

// L2: LMR の allNode 項 `r * LMR_ALLNODE_COEFF / (256 * depth + 260)` の depth ごとの表。
//   depth の範囲: search 入口で 1..MAX_PLY-1 に揃えたあと、 hindsight (+1)・IIR (-1, IIR_MIN_DEPTH
//   の下限は 0)・SE (+1) で動くので 0..MAX_PLY+1。 表は 0..MAX_PLY+1 を持ち、 外は通常の除算に落とす。
inline constexpr int ALLNODE_DIV_SIZE = MAX_PLY + 2;
inline constexpr std::array<SDiv32, ALLNODE_DIV_SIZE> ALLNODE_DIV = [] {
    std::array<SDiv32, ALLNODE_DIV_SIZE> t{};
    for (int d = 0; d < ALLNODE_DIV_SIZE; ++d)
        t[d] = SDiv32::make(256 * d + 260);
    return t;
}();

// 照合点 (VERIFY ビルドで使う。 名前は終了時の BUNDLEA_VERIFY 行に出る)。
enum Site : int {
    V_M1_DELTA1,     // NonPV で delta == 1
    V_M1_REDUCTION,  // reduction の新旧
    V_L1_QUIET,      // quiet 採点の新旧 (1 手ごと)
    V_L2_FUTCORR,
    V_L2_SFFUTCORR,
    V_L2_NMP,
    V_L2_LMRDEPTH,
    V_L2_LMRCORR,
    V_L2_ALLNODE,
    V_L2_FAILLOW,
    V_L2_STATPREV,
    V_L2_GOODCAP,
    V_L5_FLAGS,      // 省いた TLS が読まれない設定のままか (node ごと)
    V_L6_DISPATCH,   // depth<=0 で qsearch に振り分けた回数
    V_SITE_NB
};

#if defined(P2_SRCH_BUNDLE_A_VERIFY)
inline const char* const SITE_NAME[V_SITE_NB] = {
  "M1_delta1",   "M1_reduction", "L1_quiet",   "L2_futcorr", "L2_sffutcorr",
  "L2_nmp",      "L2_lmrdepth",  "L2_lmrcorr", "L2_allnode", "L2_faillow",
  "L2_statprev", "L2_goodcap",   "L5_flags",   "L6_dispatch"};

inline std::atomic<std::uint64_t> g_checks[V_SITE_NB];

// fast (新式) と ref (旧式) を照合する。 不一致なら stderr に出して abort。
inline void check(Site site, long long fast, long long ref) {
    g_checks[site].fetch_add(1, std::memory_order_relaxed);
    if (fast != ref)
    {
        std::fprintf(stderr, "BUNDLEA_VERIFY MISMATCH site=%s fast=%lld ref=%lld\n", SITE_NAME[site],
                     fast, ref);
        std::fflush(stderr);
        std::abort();
    }
}

// 終了時 (exit) に照合回数を出す。 不一致があれば check() が abort するので、 ここに来たら 0。
struct Reporter {
    ~Reporter() {
        std::uint64_t total = 0;
        std::fprintf(stderr, "BUNDLEA_VERIFY");
        for (int i = 0; i < V_SITE_NB; ++i)
        {
            const std::uint64_t c = g_checks[i].load(std::memory_order_relaxed);
            total += c;
            std::fprintf(stderr, " %s=%llu", SITE_NAME[i], (unsigned long long) c);
        }
        std::fprintf(stderr, " total=%llu mismatch=0\n", (unsigned long long) total);
        std::fflush(stderr);
    }
};
inline Reporter g_reporter;
#endif  // P2_SRCH_BUNDLE_A_VERIFY

// n / d を magic で割る。 VERIFY ビルドでは `/` と照合する (d は照合にだけ使う)。
inline int vdiv(const SDiv32& s, int n, [[maybe_unused]] int d, [[maybe_unused]] Site site) {
    const int q = s.div(n);
#if defined(P2_SRCH_BUNDLE_A_VERIFY)
    check(site, q, n / d);
#endif
    return q;
}

// num / (256 * depth + 260)。 表の外の depth は通常の除算に落とす。
inline int allnode_div(int num, int depth) {
    return unsigned(depth) < unsigned(ALLNODE_DIV_SIZE)
           ? vdiv(ALLNODE_DIV[depth], num, 256 * depth + 260, V_L2_ALLNODE)
           : num / (256 * depth + 260);
}

}  // namespace YaneuraOu::BundleA

#endif  // P2_SRCH_BUNDLE_A
#endif  // SRCH_BUNDLE_A_H_INCLUDED
