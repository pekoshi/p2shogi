#ifndef SPSA_PARAMS_H_INCLUDED
#define SPSA_PARAMS_H_INCLUDED

#include "config.h"

#include <algorithm>
#include <array>

#if defined(YANEURAOU_ENGINE) || defined(EVAL_P2NNUE)

// -----------------------------------------------------------------------------
//  SPSA tuning parameters (rust rshogi-usi の SPSA_* オプションを C++ で真似たもの)
// -----------------------------------------------------------------------------
// ★各値の default は C++(Stockfish16系 やねうら王探索)の現行リテラル値。
//   setoption しなければ挙動は従来と完全一致(default = C++値)。
//   rust 側で調整した値を setoption すればその探索挙動を再現できる。
//   USIオプション登録は eval/p2nnue/p2nnue.cpp の add_options() で行う。
//   探索内での参照は engine/yaneuraou-engine/yaneuraou-search.cpp。
//
//   Phase 1: LMR / stat bonus / stat malus / futility / razoring / null move。
//   (history/correction 系や SF14/18 固有は Phase 2 以降)
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
//  Q8 固定小数スケール (NAGISA V3.1 互換)
// -----------------------------------------------------------------------------
// NAGISA V3.1 は static eval 補正と MovePicker のスコアリングの「小さい係数」を
// Q8 固定小数 (×256) で保持している (SearchStaticEvaluationQ8Scale / MovePickerScoreScale)。
// これにより係数を整数 1 刻みではなく 1/256 刻みで SPSA できる。
//
// P2_SCORE_SCALE はコンパイル時に -DP2_SCORE_SCALE=512 等で差し替え可能。
//   EXTRA_CPPFLAGS="-DP2_SCORE_SCALE=512"
// スケール自体は挙動を変えない (係数を 2 倍にすれば 2X/512 == X/256 で完全に同一)。
// 意味があるのは「SPSA を 1/512 刻みで回せる」という粒度だけ。
//
// ★2 の冪であること必須: 除算をシフトにコンパイルさせるため (MovePicker は最ホットパス)。
#if !defined(P2_SCORE_SCALE)
    #define P2_SCORE_SCALE 256
#endif

namespace YaneuraOu {
namespace SPSA {

/// Q8 固定小数の倍率。既定 256 (= NAGISA V3.1 と同じ)。
inline constexpr int SCORE_SCALE = P2_SCORE_SCALE;
/// Q16 (= Q8 係数 × Q8 値 を元のスケールへ戻すための除数)。
inline constexpr int SCORE_SCALE_SQ = SCORE_SCALE * SCORE_SCALE;

static_assert(SCORE_SCALE > 0 && (SCORE_SCALE & (SCORE_SCALE - 1)) == 0,
              "P2_SCORE_SCALE は 2 の冪でなければならない (除算をシフト化するため)");
static_assert(SCORE_SCALE <= 4096, "P2_SCORE_SCALE が大きすぎる (中間値が int で溢れる)");

// --- LMR (late move reduction) ---
// reductions[i] = int(LMR_TABLE_COEFF / 128.0 * std::log(i))
// ★2026-09-03 訂正: 旧コメントの「Nagisa has a different LMR-table formulation」は誤り。
//   NAGISA V3.1 のソース (yaneuraou-search.cpp:2801) を確認したところ式は完全に同一で、
//   値も TUNABLE_PARAM(YaneuraOuWorker_clear3_1, 2762) = ほぼ同値だった。
//   よって「同一ノードで NAGISA が 3ply 深く読む」原因は LMR テーブル係数ではない。
inline int LMR_TABLE_COEFF              = 2762; // NAGISA V3.1 は 2762 (同一式・ほぼ同値)

// --- reduction() の3定数 ---
// r = reductions[d]*reductions[mn] - delta*DELTA_MULT/rootDelta
//     + !improving * reductions[d]*reductions[mn] * NOIMPROVING/512 + BASE
// NAGISA V3.1 (yaneuraou-search.cpp:6005-6007) と**式は完全に同一**で、値のみ異なる:
//   NAGISA = 798 / 396 / 1076   (YaneuraOuWorker_reduction_1/2/3)
//   我々   = 757 / 218 / 1200   (rust rshogi alpha_beta.rs 由来)
// ★特に非improving係数が 218 vs 396 と約1.8倍違う = NAGISA は劣勢/停滞時に大幅に削る。
//   「同一 80,000 ノードで NAGISA depth 16 / 我々 13」の最有力候補。
//   既定は現行値のまま(=bit-exact)。 NAGISA 値は params preset で A/B してから採否を決める。
inline int LMR_REDUCTION_DELTA_MULT     = 798;  // NAGISA V3.1: 798
inline int LMR_REDUCTION_NOIMPROVING    = 396;  // NAGISA V3.1: 396
inline int LMR_REDUCTION_BASE           = 1076; // NAGISA V3.1: 1076

// Step18 (LMRスキップ時の完全探索) の深さ削減閾値。
// newDepth - (r>THRESHOLD1) - (r>THRESHOLD2 && newDepth>2)。
// C++旧値 4628/5772。rust(rshogi alpha_beta.rs:3200-3202)は 3212/4784=より低い閾値=1ply多く削減。
// 非PV第1手/深さ1という最高頻度経路でrustは2ply削るがC++旧値は1plyのみ→C++の木が太る主因。
// NAGISA V3.1 (Search_Full_depth_search_threshold_2b_1/2) は 2725/4752 = さらに低い閾値。
// 使用箇所の式は NAGISA と完全に同一 (newDepth - (r>TH1) - (r>TH2 && newDepth>2))。
inline int LMR_STEP18_THRESHOLD1        = 2725; // NAGISA V3.1: 2725
inline int LMR_STEP18_THRESHOLD2        = 4752; // NAGISA V3.1: 4752

// Nagisa Tune #171 direct-equation imports, exposed as named constants for future SPSA.
inline int LMR_RESEARCH_DEEPER_THRESHOLD = 50;
inline int LMR_RESEARCH_SHALLOWER_THRESHOLD = 10;
inline int CAPTURE_SEE_PRUNING_DEPTH_MULT = 143;
inline int CAPTURE_SEE_PRUNING_HISTORY_MULT = 40;
inline int FUTILITY_CORRECTION_DIV = 75964;
inline int GOOD_CAPTURE_SEE_DIV = 16;
inline int QUIET_PARTIAL_SORT_DEPTH_MULT = -2807;

// 限定 check extension: SEE安全な王手を+1延長し戦術refuteの水平線を延ばす。
// 将棋は王手が続くため爆発しやすい→ SEE>=0 + singular非stack + 経路ply上限で限定。
// CHECK_EXT=0で無効(baseline), 1で有効。PLY_LIMIT_MULT*rootDepth を超える経路では延長しない。
inline int CHECK_EXT                    = 0;
inline int CHECK_EXT_PLY_LIMIT_MULT     = 2;

// allNode(全子fail low予想の相手番ノード)での比例reduction上乗せ係数。
// r += r * COEFF / (256*depth+260)。rust は 0(無効)だが、固定ノードでは
// 無効化すると木が太り浅くなり refute をむしろ見逃す(48局実測: 勝率25.5%→14.6%, PV外率53%→100%)。
// → C++既定 273 を維持(深い探索の方が refute を捕まえる)。
inline int LMR_ALLNODE_COEFF            = 209; // Nagisa Tune #171: same all-node reduction equation
// 2026-10-16 訂正: 上のコメントの「273 を維持」は古い。
//   現行は **209** (Nagisa Tune #171)。 rshogi upstream は 272 / offset 285。
//   どちらも読めていない測定 (rust 0 は 4 局 / C++ 209 は 48 局) の産物。


// --- stat bonus : bonus = min(DEPTH_MULT*depth + OFFSET, MAX) + TT_BONUS*(bestMove==ttMove) + (ss-1)->statScore/PREV_DIV ---
// rust default(tune_params.rs)へ揃え(旧C++値: 128/-77/1529/353/32)
inline int STAT_BONUS_DEPTH_MULT        = 248; // Retained: history update structure differs from Nagisa
inline int STAT_BONUS_OFFSET            = -43;
inline int STAT_BONUS_MAX               = 1455;
inline int STAT_BONUS_TT_BONUS          = 688;
// H2 (SF 86f1df71 / rshogi upstream 94f178a): 非 PV ノードで
//   bonus += bonus * (quiets+captures) * NUM / (256*1024)
//   = 多く試した末の best move ほど信用する。
//   既定 **0 = 従来と bit-exact**。 SF/upstream 同式にするなら 1024。
inline int UPDATE_ALL_SEARCHED_COUNT_NUM = 200;  // 11-16 既定を対局時の値へ (旧 0)
// 2026-10-16 ユーザー指示「正規化」: H2 を**分岐数で割る**版。
//   SF の /256 は chess の分岐 ~35 手で SPSA された値。
//   将棋の中盤は合法手 **98.5-111.3 手** = 約 3 倍なので
//   固定の /256 だと効きが 3 倍になる。 そこで
//   bonus += bonus * count * NUM / (**合法手数** * 1024)
//   SF 相当の最大効果 (35/35 で 0.137) に合わせると NUM = **140**。
//   既定 **0 = bit-exact**。 有効時は update_all_stats で 1 回 movegen する。
inline int UPDATE_ALL_NORM_NUM           = 0;
// 2026-10-17: **qsearch の静かな王手生成**。 SF 703f17975b (2024-05-02) で stage 削除。
//   🎯将棋は持駒があるため終盤ほど王手が増える (chess は逆)。
//   既定 **0 = bit-exact**。 1 で qsearch の**入口の 1 層だけ**静かな王手も読む。
//   🚨王手ラッシュで爆発する恐れ -> **NPS の関門必須**
inline int QS_CHECKS                     = 0;
// 2026-10-17 **最終 info 行の汚染を直す**。 中断された iteration の
//   途中スコアを最終 info 行として吐くため、 棋譜の評価値が汚染する
//   (詰まされている局面が **+35281** = VALUE_SUPERIOR x 100/90 と記録された)。
//   🚨bestmove と投了判定は不変 = **対局結果は正しい**。 記録値だけが壊れる。
//   1 で **完了した iteration の値** (previousScore) を出す。 既定 0 = 従来。
inline int FINAL_PV_COMPLETED            = 1;  // 11-16 既定を対局時の値へ (旧 0)
// H3 (SF 86f1df71 / rshogi 94f178a): quiet malus を適用手ごとに幾何減衰させる係数。
//   既定 **977 = 従来(YO v9.40)と bit-exact**。 SF/upstream は 956。
inline int QUIET_MALUS_DECAY_NUM         = 977;
inline int STATBONUS_PREV_DIV           = 32;    // rust準拠: 0 で (ss-1)->statScore 項を無効化(旧C++=32)。 11-16 既定を対局時の値 32 へ (旧 0)

// --- stat malus : malus = min(DEPTH_MULT*depth + OFFSET, MAX) ---
// rust default へ揃え(旧C++値: 882/-204/2122)
inline int STAT_MALUS_DEPTH_MULT        = 669;
inline int STAT_MALUS_OFFSET            = -198;
inline int STAT_MALUS_MAX               = 2240;

// --- futility margin : mult = MARGIN_BASE - MARGIN_TT_BONUS*!ttHit ;
//     margin = mult*d - (IMPROVING_SCALE*improving + OPP_WORSENING_SCALE*opponentWorsening)*mult/1024 ---
inline int FUTILITY_MARGIN_BASE         = 96;  // Nagisa Tune #171: same futility equation
inline int FUTILITY_MARGIN_TT_BONUS     = 52;  // Nagisa Tune #171: same futility equation
inline int FUTILITY_IMPROVING_SCALE     = 2024; // Nagisa Tune #171: same futility equation
inline int FUTILITY_OPP_WORSENING_SCALE = 467;  // Nagisa Tune #171: same futility equation

// --- NALLOC (2026-09-28): 同じ総ノード数の中で、 揺れる局面に多く配る ---
//   V (P2Volm::value) の bin ごとの重み x1024。 既定は全部 1024 = 一様 = 現行と bit-exact。
//   重みは regret 検定 (runs/rescore/nalloc_regret.py) の a_i から決める。
//   🚨総和は「銀行」で保証する: 使いたい量は残高の範囲でしか上乗せできない (借金禁止)。
//   🚨残高は**要求量**で記帳する (実消費ではない)。 詰みで早期終了した手の余りを
//     回収すると NALLOC 側だけ約 6% ノードが増えて偽の利得が出る (実測 1 局の実消費は中央 93.8%)。
inline int NALLOC_W0                    = 1024;  // V=0
inline int NALLOC_W1                    = 1024;  // V=1-2
inline int NALLOC_W2                    = 1024;  // V=3-4
inline int NALLOC_W3                    = 1024;  // V=5-6
inline int NALLOC_W4                    = 1024;  // V>=7
inline int NALLOC_WMIN                  = 256;   // 下限 x1024 (小さすぎると停止判定の粒度を割る)
inline int NALLOC_PLACEBO               = 0;     // 1 = V ではなく局面ハッシュで割り当てる (プラセボ腕)

// --- razoring : eval < alpha - RAZORING_BASE - RAZORING_DEPTH2*depth*depth ---
inline int RAZORING_BASE                = 418; // Nagisa Tune #171: same razoring equation
inline int RAZORING_DEPTH2              = 354; // Nagisa Tune #171: same razoring equation
// --- RAZV (2026-09-28): 揺れる局面では razoring を止める ---
//   razoring は qsearch の値で打ち切るが、 **qsearch は王手も駒打ちも一切生成しない**
//   (movepick.cpp:150-159、 types.h:405 DEPTH_QS=0)。 一方 depth-1 の通常探索なら
//   `capture || givesCheck` の枝は futility の対象外なので王手だけは読む。
//   V1 (玉近傍の空升への敵利き) = **そこに駒を打って王手できる升の数**そのものなので、
//   V が高い局面で razoring は構造的に間違える。
//   実測 (本番 margin x2.0、 P2_PRUNE_VERIFY=4、 実戦 120 局面 x 1M):
//     見落とし率 V0 0.37% / V1-2 0.62% / V3-4 3.09% / V5-6 22.92% / V>=7 37.36%
//     節約/見落とし 589 / 429 / 112 / 19 / 15  (幅 3916%)
//   🚨x1.0 と比べると **x2.0 は低 V だけを直していた** (V0 1.28%→0.37% に対し V>=7 は 37.32%→37.36% で不変)
//     → margin では埋まらない = 詰み・王手筋。 止めるしかない。
//   0 = 無効 (現行と bit-exact)。 N>0 なら V >= N の局面で razoring を行わない。
inline int RAZV_VMIN                    = 3;  // 11-16 既定を対局時の値へ (旧 0)

// --- null move pruning ---
// 適用条件 : staticEval >= beta - NMP_MARGIN_DEPTH_MULT*depth - NMP_MARGIN_IMPROVING_BONUS*improving + NMP_MARGIN_OFFSET
inline int NMP_MARGIN_DEPTH_MULT        = 11;  // Nagisa Tune #171: same null-move margin term
inline int NMP_MARGIN_IMPROVING_BONUS   = 30;  // Nagisa Tune #171: same null-move margin term
inline int NMP_MARGIN_OFFSET            = 118;  // Nagisa Tune #171: same null-move margin term
// reduction : R = NMP_REDUCTION_BASE + depth / NMP_REDUCTION_DEPTH_DIV
inline int NMP_REDUCTION_BASE           = 8;   // Nagisa Tune #171: same null-move reduction term
inline int NMP_REDUCTION_DEPTH_DIV      = 2;   // Nagisa Tune #171: same null-move reduction term

// ===== Phase 2 : history / correction history =====
// --- history 初期化値(clear()) ---
// rust default へ揃え(旧C++値: 0/-678/-523/-1238/98)
inline int MAIN_HIST_INIT               = 68;     // mainHistory.fill
inline int CAPTURE_HIST_INIT            = -782;   // captureHistory.fill
inline int CONT_HIST_INIT               = -812;   // continuationHistory.fill
inline int PAWN_HIST_INIT               = -1264;  // pawnHistory.clear_range
inline int LOW_PLY_HIST_INIT            = 85;     // lowPlyHistory.fill

// --- correction history 合算重み ---
// rust default へ揃え(旧C++値: 12153/8620/12355/7982)
inline int CORR_PCV_WEIGHT              = 6540;
inline int CORR_MICV_WEIGHT             = 5808;
inline int CORR_NONPAWN_WEIGHT          = 8884;
inline int CORR_CNT_WEIGHT              = 13836;
// 将棋用: 持ち駒構成(先後の手駒)correction の合算重み。0=OFF(既定, bit-exact)。ON時は pawn相当(~9536)推奨。
inline int CORR_HAND_WEIGHT             = 0;

// --- correction history 更新係数 (bonus * X / 128) ---
// rust default へ揃え(旧C++値: 187/153/126/63)
inline int CORR_HIST_NONPAWN            = 139;    // nonPawnWeight
inline int CORR_HIST_MINOR              = 189;
inline int CORR_HIST_CONT_SS2           = 112;
inline int CORR_HIST_CONT_SS4           = 63;
inline int CORR_HIST_HAND               = 128;   // 将棋用: 持ち駒構成 correction 更新係数(128=pawn相当の等倍)
#if defined(P2_KHCORR)
// 2026-11-12 KHCORR (DESIGN-20261112-OPTIMISM 案①): 玉近傍 x 浮き駒 鍵 (144 通り) の correction。
//   CORR_KH_WEIGHT = 合算重み (0 = OFF / 既定 / bit-exact。 0 のときは鍵の計算も表の読み書きも走らない)。
//   CORR_HIST_KH   = 更新係数 (bonus * X / 128。 128 = pawn と等倍)。
inline int CORR_KH_WEIGHT               = 0;
inline int CORR_HIST_KH                 = 128;
#endif
#if defined(P2_KHGATE)
// 2026-11-12 KHGATE (取り合い・玉の危険局面の楽観の補い)。 全て 0 (既定) = OFF / bit-exact (鍵の計算も走らない)。
//   PESS_*      = 静的評価の固定の悲観補正 (cp / 1 単位あたり / 手番視点)。 hang・zaE は各 3 で頭打ち。
//                 static -= PESS_HANG*min(hang_us,3) + PESS_ZA*min(zaE_us,3)
//                 static += PESS_HANG_THEM*min(hang_them,3) + PESS_ZA_THEM*min(zaE_them,3)
//   SELEXT_*    = hang_us >= SELEXT_HANG_TH または zaE_us >= SELEXT_ZA_TH の node で
//                 LMR の r を SELEXT_LMR ply (x1024) 減らす / SELEXT_NOPRUNE=1 なら Step 8 futility と Step 9 NMP を止める。
//                 閾値だけでは何も起きない (SELEXT_LMR / SELEXT_NOPRUNE のどちらかが非 0 で有効)。
inline int PESS_HANG                    = 0;
inline int PESS_ZA                      = 0;
inline int PESS_HANG_THEM               = 0;
inline int PESS_ZA_THEM                 = 0;
inline int SELEXT_LMR                   = 0;
inline int SELEXT_NOPRUNE               = 0;
inline int SELEXT_HANG_TH               = 2;
inline int SELEXT_ZA_TH                 = 2;
#endif

// ===== Phase 3 : continuation history / pawn・low-ply multiplier =====
// update_continuation_histories: (ss-i)->contHist << bonus*WEIGHT_i/1024 + NEAR_PLY_OFFSET*(i<2)
inline int CONT_HISTORY_WEIGHT_1        = 1095;
inline int CONT_HISTORY_WEIGHT_2        = 687;
inline int CONT_HISTORY_WEIGHT_3        = 326;
inline int CONT_HISTORY_WEIGHT_4        = 975;
inline int CONT_HISTORY_WEIGHT_5        = 103;
inline int CONT_HISTORY_WEIGHT_6        = 1086;
inline int CONT_HISTORY_NEAR_PLY_OFFSET = 88;
// update_quiet_histories 内 multiplier (bonus * X / 1024)
inline int CONT_HISTORY_MULTIPLIER      = 820;   // update_continuation_histories 呼出時
inline int LOW_PLY_HISTORY_MULTIPLIER   = 751;   // lowPlyHistory <<
inline int PAWN_HISTORY_POS_MULTIPLIER  = 914;   // pawn_entry << (bonus>0)
inline int PAWN_HISTORY_NEG_MULTIPLIER  = 604;   // pawn_entry << (bonus<=0)

// ===== 方針指数(policy index) : quiet move ordering への薄いヒント =====
// eval には一切加算せず、MovePicker の quiet ordering の m.value にのみ小ボーナスを混ぜる
// (探索順序の誘導のみ = minimax 値は不変)。0 で完全に従来挙動(bit-exact)。
inline int POLICY_HINT_SCALE = 0;  // 0=OFF。>0 で root 局面の方針に沿った quiet 手を微加点。

// ===== 右玉誘導 (MIGI_MOVEORDER_SCALE) : quiet move ordering への薄いヒント (2026-10-20) =====
// ユーザー決定 2026-10-20「評価の下駄は 0 にして MovePicker 一本に / 先後とも押す」。
// 「飛車は右・玉はまだ右でない」局面でだけ、 玉を右側に入れる quiet 手に SCALE*weight/16 を足す。
// 🚨eval には一切加算しない (evaluate() は bit-exact) / 🚨**既に右玉なら 0** = 維持はしない
//   (既存の MigigyokuBonus は「維持」もしていたため、 崩すべきときに崩せなかった)。
// 既に右玉かの判定は**局面の関数**なので置換表 (TT) の意味を変えない。
// 0 で完全に従来挙動 (bit-exact)。 1024 で +16384 = 王手ボーナス相当。
inline int MIGI_MOVEORDER_SCALE = 0;
inline int MIGI_MOVEORDER_DEBUG = 0;  // 1 で root の QUIET_INIT に発火を info string で出す

// 駒打ち(drop) quiet 手専用の Drop Continuation History のスケール。0=OFF(既定, bit-exact)。
// >0 で「直前手駒種 × drop駒種 × drop升の敵玉相対位置」history を quiet ordering に薄く加算。
// eval には非関与。100 で概ね entry/4 相当の加算(movepick 側 * scale / 400)。
inline int DROP_CONT_HIST_SCALE = 0;

// History Confidence LMR: quiet 手の statScore 3成分(2*mainHistory, contHist[0], contHist[1])が
// 同符号=「高信頼」なときだけ reduction 調整を一段強める。0=OFF(既定, bit-exact)。
// 既存の線形項 r -= statScore*428/4096 に、符号一致時のみ追加で r -= statScore*SCALE/4096 する。
// 探索順序でなく削減量に作用。★採用確定(2026-07-12, SPRT nElo+26): 既定 150。
inline int HIST_CONFIDENCE_SCALE = 150;

// Uncertainty-linked LMR (第1パッチ): 評価急変度 nodeEvalVolatility がこの閾値以上の
// 高不確実性 quiet 手だけ、LMR reduction を厳密に1ply(=1024)弱める(離散・最大1ply)。
// 0=完全OFF(既定, bit-exact)。補正量は固定1ply、自由パラメータはこの閾値のみ(比例/CAP/多段なし)。
// 閾値は実測 volatility 分布(発火率~5%狙い)を根拠に決める。非採用(既定OFF)。
inline int UNCERT_LMR_THRESHOLD = 0;

// Threat Continuation History (Phase B): Null Move で採取した相手脅威手 threatMove の文脈で、
// beta cutoff した quiet response を学習し MovePicker の quiet ordering に薄く加点(history/8 固定)。
// 0=完全OFF、>0 で有効(値は ON/OFF ゲート、加算係数は /8 固定)。
// ★採用確定(2026-07-13): root-eval ゲート版で 480局 52.5%(逆転棋風・ポカ非増)。既定 ON=1。
inline int THREAT_HISTORY_SCALE = 0;  // 11-16 既定を対局時の値へ (旧 1)
// threatMove 採用に必要な null 局面 TT entry の最低 depth(Phase A 分布より leaf(depth<=0)を除く 1 を既定)。
inline int THREAT_MIN_DEPTH     = 1;
// 採用する ThreatClass の bitmask(bit i = ThreatClass i)。1=DropCheck,2=Check,3=Capture,4=Promotion,
// 5=KingZone,6=Other。既定 0x7E=全クラス(従来同等)。粗い Capture(bit3)を外すなら 0x76、鋭い脅威のみは 0x36。
inline int THREAT_CLASS_MASK    = 0x7E;
// Threat 加点/更新を「残り深さ(node depth)がこの値以上」のノードだけに限定する depth ゲート。
// 0=全ノード(従来同等)。浅ノードでの reorder が短時間で悪手を即選択してポカ源になるため、
// 深い(=深く検証される)ノードだけに効かせて意外な好手を残しつつポカを抑える。★採用: 9(教師データ主帯)。
inline int THREAT_MIN_APPLY_DEPTH = 9;
// Threat 加点/更新を「root(対局)の静的評価 threatRootEval がこの値未満」= 対局として不利なときだけに
// 限定する eval ゲート。負けている対局でのみ全ノードで threat が効き(=劣勢からの逆転狙い)、
// 優勢〜互角の対局では無効化(=無用なギャンブル/ポカを止め net を信頼)。
// 「+100までは net 信頼、-80で逆転狙い」の棋風。★採用: -80(sweep最良帯、負け局面でのみ発火)。
inline int THREAT_MAX_EVAL = -80;

// ---- 学習(更新)ゲート: 上記の適用(加点)ゲートから分離したもの ----
// 従来は学習ゲート == 適用ゲートだったため、「使い始める瞬間にテーブルが空」という問題があった
// (互角のうちは一切学習せず、劣勢に転じた瞬間に 空テーブル参照+学習開始 が同時に起きる)。
// 学習側だけを先に広げてテーブルを育てられるよう、専用パラメータに分離する。
// 既定値は現行の適用ゲートと同値なので、既定では従来と完全に同一挙動(bit-exact)。
inline int THREAT_LEARN_MIN_DEPTH = 9;
inline int THREAT_LEARN_MAX_EVAL  = -80;

inline int POLICY_DEBUG      = 0;  // 1 で root 局面の attack/castle/wait を info string 出力。

// ===== Phase 4 : NAGISA V3.1 (ENGINE_VERSION 9.60) の TUNABLE_PARAM 対応(分類B) =====
// NAGISA と式は完全に一致するがリテラル直書きだった 93 個をオプション化したもの。
// 既定値はすべて我々の現行リテラル値なので、setoption しなければ従来と bit-exact。
// 各行コメント末尾は NAGISA V3.1 の対応値(A/B 用プリセット runs/nagisa_v31_full.params)。
// 「Q8」と注記したものは NAGISA が SearchStaticEvaluationQ8Scale=256 等で分解能を上げた式のため、
// NAGISA の値をそのまま流用できない(我々のスケールへ換算した値をプリセット側に書いてある)。
inline int QSEARCH_SEE_PRUNING                   = -63;  // qsearch Step6 SEE枝刈り閾値 see_ge(move, X)。NAGISA -63
inline int QSEARCH_MOVE_COUNT_PRUNING            = 2;  // qsearch move count 枝刈り (moveCount > X で continue)。NAGISA 2
inline int CORR_HIST_BONUS_BESTMOVE              = 12;  // correction history bonus 係数 (bestMove あり)。NAGISA 12
inline int CORR_HIST_BONUS_NOBESTMOVE            = 17;  // correction history bonus 係数 (bestMove なし)。NAGISA 17
inline int CORR_HIST_BONUS_SCALE                 = 1263;  // update_correction_history(X * bonus / 1024)。NAGISA 1263
inline int FAILLOW_CAPTURE_BONUS                 = 1362;  // fail low 時 prior capture への captureHistory bonus。NAGISA 1362
inline int FAILLOW_BONUS_SCALE_BASE              = -198;  // fail low bonusScale 基準値。NAGISA -198
inline int FAILLOW_BONUS_STATSCORE_DIV           = 74;  // bonusScale -= (ss-1)->statScore / X。NAGISA 74
inline int FAILLOW_BONUS_DEPTH_MULT              = 72;  // bonusScale += min(X*depth, MAX)。NAGISA 72
inline int FAILLOW_BONUS_DEPTH_MAX               = 359;  // 同上の上限 MAX。NAGISA 359
inline int FAILLOW_BONUS_MOVECOUNT               = 186;  // bonusScale += X * ((ss-1)->moveCount > TH)。NAGISA 186
inline int FAILLOW_BONUS_MOVECOUNT_TH            = 7;  // 同上の閾値 TH。NAGISA 7
inline int FAILLOW_BONUS_EVAL                    = 133;  // bonusScale += X * (!inCheck && bestValue <= staticEval - MARGIN)。NAGISA 133
inline int FAILLOW_BONUS_EVAL_MARGIN             = 83;  // 同上の MARGIN。NAGISA 83
inline int FAILLOW_BONUS_PREVEVAL                = 153;  // 1手前 staticEval 条件の係数。NAGISA 153
inline int FAILLOW_BONUS_PREVEVAL_MARGIN         = 70;  // 同上の MARGIN。NAGISA 70
inline int FAILLOW_SCALED_DEPTH_MULT             = 135;  // scaledBonus = min(X*depth - OFFSET, MAX) * bonusScale。NAGISA 135
inline int FAILLOW_SCALED_OFFSET                 = 47;  // 同上の OFFSET(式では減算)。NAGISA 47
inline int FAILLOW_SCALED_MAX                    = 1338;  // 同上の MAX。NAGISA 1338
inline int FAILLOW_CONTHIST_MULT                 = 292;  // update_continuation_histories(ss-1, .., scaledBonus * X / 16384)。NAGISA 292
inline int FAILLOW_MAINHIST_MULT                 = 232;  // mainHistory << scaledBonus * X / 32768。NAGISA 232
inline int FAILLOW_PAWNHIST_MULT                 = 261;  // pawn_entry << scaledBonus * X / 8192。NAGISA 261
inline int IIR_MIN_DEPTH                         = 5;  // IIR 発動 depth 下限。NAGISA 5
inline int IIR_PRIOR_REDUCTION_MAX               = 3;  // IIR 発動条件 priorReduction <= X。NAGISA 3
inline int CONT_CORR_HIST_INIT                   = 7;  // continuationCorrectionHistory 初期値。NAGISA 7
inline int QSEARCH_FUTILITY_BASE_MARGIN          = 194;  // qsearch futilityBase = staticEval + X。NAGISA 97
inline int TTMOVE_HISTORY_BONUS                  = 738;  // ttMoveHistory 加点 (bestMove == ttMove)。NAGISA 738
inline int TTMOVE_HISTORY_MALUS                  = -645;  // ttMoveHistory 減点。NAGISA -645
inline int LMR_STEP18_NO_TTMOVE_ADD              = 1077;  // Step18 ttMove 無し時の r 加算。NAGISA 1077
inline int POST_LMR_CONTHIST_BONUS               = 596;  // LMR 後 continuation history 加点。NAGISA 596
inline int LMR_STATSCORE_MULT                    = 462;  // r -= statScore * X / 4096 (capture/quiet 共通)。NAGISA 462
inline int LMR_CAPTURE_STATSCORE_PIECEVALUE_MULT = 1096;  // capture 時 statScore の PieceValue 係数 (X * PieceValue / 128)。NAGISA 1096
inline int LMR_TTMOVE_REDUCTION                  = 2006;  // move == ttMove の reduction 減算。NAGISA 2006
inline int LMR_CUTOFFCNT_BASE                    = 229;  // cutoffCnt > 1 の基本加算。NAGISA 229
inline int LMR_CUTOFFCNT_GT2                     = 1453;  // cutoffCnt > 2 の追加加算。NAGISA 1453
inline int LMR_CUTOFFCNT_ALLNODE                 = 594;  // allNode 追加加算。NAGISA 594
inline int LMR_TTCAPTURE_ADD                     = 1625;  // ttCapture 時の reduction 加算。NAGISA 1625
inline int LMR_CUTNODE_ADD                       = 2127;  // cutNode の reduction 加算。NAGISA 2127
inline int LMR_CUTNODE_NO_TTMOVE_ADD             = 1260;  // cutNode かつ ttMove 無しの追加。NAGISA 1260
inline int LMR_MOVECOUNT_MULT                    = 13;  // r -= moveCount * X。NAGISA 13 (上限は既定 65 を含むよう 26→130 に拡張)
inline int LMR_CORRECTION_DIV                    = 29252;  // r -= abs(correctionValue) / X。NAGISA 29252
inline int LMR_BASE_OFFSET                       = 939;  // reduction 基準オフセット加算。NAGISA 939
inline int LMR_TTPV_BASE                         = 2445;  // ttPv 時 reduction 減算 (基本)。NAGISA 2445
inline int LMR_TTPV_PVNODE                       = 1360;  // ttPv かつ PvNode。NAGISA 1360
inline int LMR_TTPV_TTVALUE_GT_ALPHA             = 469;  // ttPv かつ ttValue > alpha。NAGISA 469 (上限は既定 905 を含むよう 882→1810 に拡張)
inline int LMR_TTPV_TTDEPTH                      = 843;  // ttPv かつ ttDepth >= depth。NAGISA 843
inline int LMR_TTPV_TTDEPTH_CUTNODE              = 1183;  // 同上 cutNode 追加。NAGISA 1183
inline int SE_TRIPLE_MARGIN_BASE                 = 71;  // SE tripleMargin 基準。NAGISA 71
inline int SE_TRIPLE_MARGIN_PVNODE               = 373;  // SE tripleMargin PvNode 項。NAGISA 373
inline int SE_TRIPLE_MARGIN_NOTTCAPTURE          = 0;  // SE tripleMargin !ttCapture 項(式では減算)。NAGISA は 0。
// 2026-09-08 撤回: 一度 218 -> 0 を「採用」したが、 その判断は成立していなかった。
//   対局は params128 (NAGISA 移植) を USI オプションで渡しており、 そこに
//   SPSA_SE_TRIPLE_MARGIN_NOTTCAPTURE=0 が最初から含まれていた。 つまりソース既定を
//   0 にしても**対局条件は何も変わっていない** (3 対局の args.txt が完全一致で確認)。
//   ext+3 が 11.7 倍という計測は params を渡さない素の go nodes = 既定 218 での話で、
//   実戦の対局には元から効いていなかった。 自己対局 A/B の +133 Elo も同じ疑いがある。
//   ★教訓: パラメータを変えたら、 まず **対局に実際に渡っている値** (args.txt / meta.json) を
//   確認する。 ソース既定と対局条件は別物。
inline int SE_TRIPLE_MARGIN_TTPV                 = 133;  // SE tripleMargin ttPv 項。NAGISA 133
inline int SE_TRIPLE_MARGIN_PLY                  = 49;  // SE tripleMargin ply > rootDepth 項(式では減算)。NAGISA 49
inline int SE_DOUBLE_MARGIN_BASE                 = 2;  // SE doubleMargin 基準。NAGISA 2
inline int SE_DOUBLE_MARGIN_PVNODE               = 205;  // SE doubleMargin PvNode 項。NAGISA 205
inline int SE_DOUBLE_MARGIN_NOTTCAPTURE          = 250;  // SE doubleMargin !ttCapture 項(式では減算)。NAGISA 250
inline int SE_DOUBLE_MARGIN_TTMOVEHIST           = 769;  // SE doubleMargin ttMoveHistory 項 (X * ttMoveHistory / 116517)。NAGISA 769
inline int SE_DOUBLE_MARGIN_PLY                  = 20;  // SE doubleMargin ply > rootDepth 項(式では減算)。NAGISA 20
inline int SE_CORRVAL_DIV                        = 278750;  // SE corrValAdj = abs(correctionValue) / X。NAGISA 278750
inline int SE_SINGULAR_BETA_BASE                 = 70;  // singularBeta 係数(基本)。NAGISA 70
inline int SE_SINGULAR_BETA_TTPV                 = 76;  // singularBeta ttPv && !PvNode 項。NAGISA 76
inline int SE_SINGULAR_BETA_DIV                  = 35;  // singularBeta 除数。NAGISA 35
inline int SEE_PRUNING_QUIET_MULT                = -45;  // quiet SEE 枝刈り see_ge(move, X * lmrDepth * lmrDepth)。NAGISA -45
inline int FUTILITY_PARENT_LMRDEPTH_MULT         = 58;  // parent futility の lmrDepth 係数。NAGISA 58 (上限は既定 120 に余裕を持たせ 240)
inline int FUTILITY_PARENT_ALPHA_BONUS           = 65;  // parent futility の staticEval > alpha 項。NAGISA 65
inline int CONTHIST_PRUNING_DEPTH_MULT           = -3971;  // cont history 枝刈り閾値 (history < X * depth)。NAGISA -3971
inline int CONTHIST_PRUNING_MAINHIST_MULT        = 38;  // history += X * mainHistory / 32。NAGISA 38 (上限は既定 71 に余裕を持たせ 144)
inline int CONTHIST_PRUNING_LMRDEPTH_DIV         = 3712;  // lmrDepth += history / X。NAGISA 3712
inline int CAPTURE_FUTILITY_BASE                 = 492;  // capture futility 基準。NAGISA 492
inline int CAPTURE_FUTILITY_LMRDEPTH_MULT        = 371;  // capture futility lmrDepth 係数。NAGISA 371
inline int CAPTURE_FUTILITY_CAPTHIST_MULT        = 132;  // capture futility captHist 係数 (X * captHist / 1024)。NAGISA 132
inline int LMR_TTPV_INCREASE                     = 299;  // ttPv ノードの reduction 加算 (r += X)。NAGISA 299
inline int SMALL_PROBCUT_MARGIN                  = 144;  // small ProbCut マージン。NAGISA 144
inline int PROBCUT_MARGIN                        = 139;  // ProbCut マージン。NAGISA 139
inline int PROBCUT_IMPROVING_MARGIN              = 80;  // ProbCut improving 項(式では減算)。NAGISA 80
inline int HINDSIGHT_EVAL_SUM_TH                 = 184;  // priorReduction >= 2 の depth-- 閾値。NAGISA 184
// --- static eval 差分による quiet 手オーダリング補正 (★Q8 単位 = 値 / SCORE_SCALE が実効値) ---
// 2026-09-03: NAGISA V3.1 と同一の Q8 定式化へ移行した。単位が 256 倍になっているので
//   旧スケールの値 (例 MAINHIST_MULT=10) をそのまま setoption すると 1/256 になる。
//   USI オプション名も誤用防止のため `_Q8` 付きへ改名済み (旧名は存在しない=setoption が弾かれる)。
// 式:
//   evalDiffQ8 = clamp(-(prevStaticEval + staticEval) * SCORE_SCALE, CLAMP_LO_Q8, CLAMP_HI_Q8)
//                + OFFSET_Q8
//   mainHistory << evalDiffQ8 * MAINHIST_MULT_Q8 / SCORE_SCALE_SQ
//   pawn_entry  << evalDiffQ8 * PAWNHIST_MULT_Q8 / SCORE_SCALE_SQ
// 既定値は旧スケール値 × SCORE_SCALE なので割り切れ、旧挙動と bit-exact。
inline int STATIC_EVAL_PAWNHIST_MULT_Q8          = 3818 * SCORE_SCALE / 256;   // 旧 12。NAGISA 3818 (実効 14.91)
inline int STATIC_EVAL_CLAMP_LO_Q8               = -39852 * SCORE_SCALE / 256; // 旧 -214。NAGISA -39852 (実効 -155.7)
inline int STATIC_EVAL_CLAMP_HI_Q8               = 17343 * SCORE_SCALE / 256;  // 旧 171。NAGISA 17343 (実効 67.7)
inline int STATIC_EVAL_OFFSET_Q8                 = 11351 * SCORE_SCALE / 256;   // 旧 60。NAGISA 11351 (実効 44.3)
inline int STATIC_EVAL_MAINHIST_MULT_Q8          = 1328 * SCORE_SCALE / 256;   // 旧 10。NAGISA 1328 (実効 5.19)
inline int TT_PREV_CONTHIST_MALUS                = -1116;  // TT hit 時 1手前 cont history ペナルティ。NAGISA -1116
inline int TT_QUIET_BONUS_DEPTH_MULT             = 28;  // TT hit quiet ttMove bonus depth 係数。NAGISA 28 (上限は既定 130 を含むよう 58→260 に拡張)
inline int TT_QUIET_BONUS_OFFSET                 = 90;  // 同上 オフセット(式では減算)。NAGISA 90
inline int TT_QUIET_BONUS_MAX                    = 1246;  // 同上 上限。NAGISA 1246
inline int ASPIRATION_DELTA_BASE                 = 28;  // aspiration window 初期幅。NAGISA 28
inline int ASPIRATION_MSS_DIV                    = 9172;  // meanSquaredScore 除数。NAGISA 9172
// p2shogi 2026-11-01: aspiration 窓の **thread 幅**。 delta += STEP * (threadIdx % 8)。
//   既定 1 = 従来と bit-exact。 🚨Stockfish は base 5 に対して +0..7 なので相対幅が
//   最大 2.4 倍だが、 我々は base 28 なので 28/29/30 = +3.6%/+7.1% で**多様性が実質ゼロ**。
//   base 28 で SF と同じ相対量にするには step 4-6。 Threads=1 では threadIdx=0 で不変。
//   根拠: SF 8ecfc3c が SMP STC 106,176 局 / LTC 133,076 局で通過 (効果は 1-2 Elo 級)。
inline int ASPIRATION_THREAD_STEP                = 1;  // thread 別 aspiration 幅。1 で従来
// p2shogi 2026-11-01: continuation history を thread 間で共有する (0 で従来 = bit-exact)。
//   Stockfish 2026-06-14 (PR #6905) の方式。 8T/16T/64T の SMP テストを通過。
//   🚨3 thread では利得が分解能 (nodes 8%) の下に沈むリスクが本物。 両指標の CI が
//     0 をまたいだら**延長せずに不採用**と決めてある。
//   🎯副作用: per-thread 53.7MB x 3 = 161MB -> 共有 54MB。 L3 8MB の機械では NPS が
//     上がる可能性 (SF の -4〜6% は 128 thread の false sharing)。
inline int SHARED_CONT_HIST                      = 0;  // 1 で thread 間共有
// p2shogi 2026-11-01: **Stockfish 61c6871 (2026-10-28) の移植**。 0 で従来 = bit-exact。
//   odd thread の null move reduction を 1 ply 減らす = 半分の thread が null move の
//   cutoff を**より慎重に検証する**。 片方が誤 cutoff を拾う。
//   SF の検証: SMP STC 279,256 局 (W71,709/L71,076) / LTC 328,158 局 (W85,355/L84,497)
//     = 約 +1.7 Elo。 **やねうら王には入っていない**。
//   🚨nodes-to-depth は**悪化して見えるのが正しい** (helper の仕事が増えるので)。
//     P-b (helper の仕事を減らす = idle 署名) とは符号が逆。 指標を間違えないこと。
//   🚨Threads=1 / thread 0 では threadIdx&1 == 0 なので**常に不変**。
inline int NMP_THREAD_PARITY                     = 0;  // 1 で odd thread の R を -1
// --- M1: 遅延 (ABDADA-lite / busy 表) 2026-11-02 ---
//   実測: 3 スレッドが各々「1 スレッドで読む木の 93-95%」を重複して読んでいた
//     (depth 15/18/20 で同一 / TTD 0.97-1.00x = 並列化がまったく効いていない)。
//     原因は多様性不足ではなく **分業の不在** (runs/prereg/DESIGN-20261102-SMP-DEFER.md)。
//   機構: 節点の手ループで、 2 手目以降の子局面が他 thread に読まれていれば
//     その手を後回しにして兄弟手へ進む。 後回しの手は **末尾で元の moveCount のまま必ず読む**
//     -> 単スレッドの探索意味論は不変。 正しさは busy 表に一切依存しない。
//   🚨0 = off で **bit-exact**。 残り深さがこの値以上の節点だけで行う。
//     大きくすると root 分割に退化する (旧「root 分割」案を内包)。
inline int SMP_DEFER_MIN_DEPTH                   = 0;  // 0 で従来 (bit-exact)
//   M2: helper の反復ずらし (SF9 の skipSize/skipPhase の復活)。 土台の作者自身が
//     search.cpp:2545 で「SF10 で廃止されて短い時間で弱くなった」と書いている。
inline int SMP_HELPER_SKIP                       = 0;  // 1 で helper の反復を飛ばす
//   M3: **根の YBWC** (2026-11-02)。 長兄 (1 手目) を master が読み終えてから、
//     兄弟手を空いた thread に配る。 共有 alpha を使うので投機の税を払わない。
//     🚨1 = on。 既定 0 で bit-exact (worker の rootSplit が nullptr になる)。
//     🚨天井は実測の集中率 0.84 から P=3 で **1.12 倍**。 根だけでは大きくは伸びない。
//       木の内部分割 (M4) の土台を兼ねる。 詳細は source/smp_root_ybwc.h
inline int ROOT_YBWC                             = 0;  // 1 で根の YBWC (既定 0 = bit-exact)
//   M4: **木の内部での仕事の委譲** (YBWC の実装形 / 2026-11-02)。
//     残り深さがこの値以上の節で、 2 手目以降の手を空いている thread に提示する。
//     helper は手順を再生してその手を master の窓・深さで読み、 結果は**置換表**に残る。
//     master は末尾 (M1 の後回しの 2 周目) で同じ手を読むが、 置換表に当たるので安い。
//     🎯**正しさが helper に依存しない** (master は必ず全手を自分で読む)。
//     🚨0 = off で bit-exact。 詳細は source/smp_deleg.h
inline int YBWC_MIN_DEPTH                        = 0;  // 0 で従来 (bit-exact)
//   🚨委譲する節の深さの上限は `DelegSlot::PATH_CAP` (source/smp_deleg.h) に 1 つだけ置く。
inline int MP_GOOD_QUIET_THRESHOLD               = -11688;  // MovePicker good quiet 判定閾値。NAGISA -11688
// --- MovePicker のスコアリング係数 (★_Q8 付きのみ Q8 単位 = 値 / SCORE_SCALE が実効値) ---
// 2026-09-03: NAGISA V3.1 (movepick.cpp の MovePickerScoreScale) と同一の Q8 定式化へ移行。
//   MP_QUIET_CHECK_SEE_TH / MP_QUIET_CHECK_BONUS は NAGISA でも Q8 化されていない
//   (式中で SCORE_SCALE を掛けている) ので単位は従来どおり。
inline int MP_LOW_PLY_HISTORY_MULT_Q8            = 1853 * SCORE_SCALE / 256;  // 旧 8。NAGISA 1853 (実効 7.24)
inline int MP_QUIET_MAINHIST_MULT_Q8             = 461 * SCORE_SCALE / 256;  // 旧 2。NAGISA 461 (実効 1.80)
inline int MP_QUIET_PAWNHIST_MULT_Q8             = 488 * SCORE_SCALE / 256;  // 旧 2。NAGISA 488 (実効 1.906)
inline int MP_QUIET_CHECK_SEE_TH                 = -61;  // MovePicker 王手ボーナスの SEE 閾値。NAGISA -61 (Q8 対象外)
inline int MP_QUIET_CHECK_BONUS                  = 16492;  // MovePicker 王手ボーナス。NAGISA 16492 (Q8 対象外)
inline int MP_CAPTURE_PIECEVALUE_MULT_Q8         = 1950 * SCORE_SCALE / 256;  // 旧 7。NAGISA 2215 (実効 8.65)

// ===== Phase 5 : Stockfish 2026-09 master の未移植探索手法の移植 (SF port, 2026-09-12) =====
// 調査元: memory/stockfish_unported_survey_20260912.md。 SF 原文 (行番号は 2026-09-12 の master):
//   #2  TT secondary aging  tt.cpp:122-134 (commit 94beadffb3 + be9df38f27 racy underflow guard)
//       TT penalize         tt.cpp:157-159 / search.cpp:913-921 (commit 319d61effd)
//   #4  mainHistory 減衰    search.cpp:327-329 (commit ced9f69834、現在は ×729/1024)
//   #5  CMHC                search.cpp:2073-2091 (commit 8b6d8def30、overflow 修正 47be34c)
//   #6  multicut corrhist   search.cpp:1285-1296 (commit 218c74ec4d)
//   #7a lmrDivisor[16]      search.cpp:55 / 1207-1219 (commit 95f3680080、定数 2995 を置換)
//   #7b ttMove else-if      search.cpp:1343-1346 (commit 2be0b2cdb4 → 924d29d3ca で単純化)
//   #7c alpha 項            search.cpp:1362-1363 (commit 5f7348f03f)
//   #7d futility 連続化     search.cpp:997-1008 (commits c66acdac99 / 7bd32a51f5 / 7da31a1f1d)
// ★すべて既定 OFF。OFF のときは探索が従来と bit-exact (ハッシュ構造・乱数・計算順序を変えない)。
// ★USI option 名は SF_* (登録は eval/p2nnue/p2nnue.cpp の add_options())。
//   設定値は setoption 時にこのグローバルへコピーされ、探索中は Options を引かない。
//
// ★評価値の目盛り (eval 単位の SF 定数をどう扱うか):
//   やねうら王 master (9.80git) は SF の eval 単位の定数を 1:1 で移植している
//   (futilityMult = 76 - 21*!ttHit、lmrDepth += history/3220、(2*beta+eval)/3 が SF 原文と同値)。
//   我々の 48/26/3712 は NAGISA V3.1 の将棋チューニング値であって単位換算ではない
//   (同じ futility 式の中でも 48/76=0.63 と 26/21=1.24 で方向が揃わない。SE の margin は
//    2/205/373 vs SF -2/204/279 でほぼ 1:1)。よって「単位」は名目上 1:1 とみなし、
//   eval 単位の定数は既定 SF_EVAL_SCALE=1024 (=SF 値そのまま) で使う。
//   実験用に SF_EVAL_SCALE で一括換算できる (候補: 647 = 48/76 = futility 係数比、
//   918 = 1/1.115 = 我々の cp が NAGISA の 1.115 倍であることの逆数)。
//   history 単位の除数 (#7a) だけは「SF が置換した定数 2995 ↔ 我々の 3712」の比で
//   既定スケール 1269 (=3712*1024/2995) とし、水準を保ったまま深さ依存の形だけ入れる。

// --- #2 TT secondary aging / TT penalize (tt.cpp から参照。エントリのレイアウトは不変) ---
inline bool SF_TT_AGING    = false;  // 保存時、非 EXACT かつ決着値かつ depth>=5 のエントリの depth を 1 減らす
inline bool SF_TT_PENALIZE = false;  // TT cutoff が bound 不整合だけで失敗したエントリの depth を 1 減らす

// --- #4 go ごとの mainHistory 減衰: 反復深化開始時に全要素 × SF_MAINHIST_DECAY_NUM / 1024 ---
inline bool SF_MAINHIST_DECAY     = false;
inline int  SF_MAINHIST_DECAY_NUM = 729;

// --- #5 CMHC (Counter-Move History Continuity) ---
// update_continuation_histories で「既に正の値を持つ継続履歴の数」に応じた乗数を掛ける。
// SF 現行: entry << bonus*weight*mult/65536 (weight は旧値の半分)。我々の weight は旧スケール
// (1024 基準) のままなので bonus*weight*mult/131072 が等価 (overflow 対策で int64 で計算)。
inline bool SF_CMHC = false;
inline constexpr int SF_CMHC_MULTIPLIERS[7] = {94, 103, 110, 106, 119, 126, 121};

// --- #6 multicut 時の correction history 更新 ---
// bonus = clamp((value - staticEval) * singularDepth * MULT / 1024, ±CORRECTION_HISTORY_LIMIT/4)
// (SF の通常更新は 12/18 /128 → 1061/1024、我々は 12/17 /128 → 1263/1024 で同構造 = 換算不要)
inline bool SF_MULTICUT_CORR      = false;
inline int  SF_MULTICUT_CORR_MULT = 177;

// --- #7a 深さ依存の LMR history 除数 ---
// SF: lmrDepth += history / lmrDivisor[min(depth,16)-1]。SF 原文の配列 (定数 2995 の置換)。
inline bool SF_LMR_DEPTH_DIV       = false;
inline int  SF_LMR_DEPTH_DIV_SCALE = 1269;  // 配列 × SCALE/1024。1269 = 3712/2995 (我々の水準に合わせる)。1024 で SF 生値。
inline constexpr std::array<int, 16> SF_LMR_DIVISOR_RAW = {3637, 2787, 2761, 2939, 3171, 3347, 3147, 2762,
                                                            2772, 3106, 3107, 3060, 3112, 2991, 3090, 3542};
constexpr std::array<int, 16> sf_lmr_divisor_scaled(int scale) {
    std::array<int, 16> a{};
    for (int i = 0; i < 16; ++i)
    {
        const int v = SF_LMR_DIVISOR_RAW[i] * scale / 1024;
        a[i]        = v < 1 ? 1 : v;
    }
    return a;
}
// 探索中に参照するスケール済み配列 (setoption 時に sf_rescale() で再計算。毎ノードの乗算を避ける)。
inline std::array<int, 16> SF_LMR_DIVISOR = sf_lmr_divisor_scaled(1269);

// --- #7b ttMove の reduction 減算を cutoffCnt>1 の else-if にする ---
inline bool SF_LMR_TTMOVE_ELSEIF = false;

// --- #7c LMR の alpha 項: !capture && !is_decisive(alpha) なら r += 3*clamp(alpha - eval, -64, 96) ---
// SF_EVAL_SCALE で換算: r += clamp(alpha-eval, LO, HI) * MULT_Q10 / 1024
//   LO = -64*S/1024, HI = 96*S/1024, MULT_Q10 = MULT*1024*1024/S (S=1024 なら 3*clamp と厳密に一致)
inline bool SF_LMR_ALPHA_TERM     = false;
inline int  SF_LMR_ALPHA_MULT     = 3;
inline int  SF_LMR_ALPHA_LO       = -64;    // sf_rescale() で再計算
inline int  SF_LMR_ALPHA_HI       = 96;     // 同上
inline int  SF_LMR_ALPHA_MULT_Q10 = 3 * 1024;  // 同上

// --- #7d futility pruning の連続化 ---
// depth < DEPTH_MAX、futilityMult = (min(45+4*depth, 85) - 20*!ttHit) * S/1024、
// margin = mult*depth - (2789*improving + 335*opponentWorsening)*mult/1024 + |corr|/CORR_DIV、
// return (661*beta + 363*eval)/1024。  (現行: depth<15、mult = 48-26*!ttHit、(2*beta+eval)/3)
inline bool SF_FUTILITY_V2           = false;
inline int  SF_FUTILITY_V2_DEPTH_MAX = 19;
inline int  SF_FUTILITY_V2_CORR_DIV  = 198435;  // = 198435*1024/S (sf_rescale() で再計算)

// --- eval 単位の一括換算係数 (1024 = SF 値そのまま) ---
inline int SF_EVAL_SCALE = 1024;

// SF_EVAL_SCALE / SF_LMR_ALPHA_MULT / SF_LMR_DEPTH_DIV_SCALE の setoption 時に呼ぶ (探索外)。
inline void sf_rescale() {
    const int s          = std::max(1, SF_EVAL_SCALE);
    SF_LMR_ALPHA_LO       = -64 * s / 1024;
    SF_LMR_ALPHA_HI       = 96 * s / 1024;
    SF_LMR_ALPHA_MULT_Q10 = SF_LMR_ALPHA_MULT * 1024 * 1024 / s;
    SF_FUTILITY_V2_CORR_DIV = std::max(1, int(198435LL * 1024 / s));
    SF_LMR_DIVISOR        = sf_lmr_divisor_scaled(std::max(1, SF_LMR_DEPTH_DIV_SCALE));
}


// --- 2026-09-14 時間管理: iterative deepening の止め時を決める係数 ---
//   use_time_management() のときだけ参照される。 go nodes の探索には一切影響しない。
//   既定値は Stockfish 由来のリテラルと同一 double になる整数表現。
inline int TM_FE_PREV_MULT  = 2115;   // fallingEval の 2.115  (/1000)
inline int TM_FE_ITER_MULT  = 987;    // fallingEval の 0.987  (/1000)
inline int TM_INSTAB_MULT   = 18956;  // bestMoveInstability の 1.8956 (/10000)
inline int TM_STABLE_CENTER = 1157;   // timeReduction の center +11.57 (/100)
inline int TM_EFFORT_MULT   = 666;    // highBestMoveEffort の 0.666 (/1000)
inline int TM_INCDEPTH_PCT  = 503;    // increaseDepth の 0.503 (/1000)
} // namespace SPSA
} // namespace YaneuraOu

#endif // YANEURAOU_ENGINE || EVAL_P2NNUE
#endif // SPSA_PARAMS_H_INCLUDED
