// 🏁🏁**2026-10-28 に棄却済みとして無効化** (ユーザー「非採用コードは全撤去」)。
//   `#if 0` で囲むのでバイナリには入らない (PGO のプロファイルも汚さない)。
//   削除ではなく無効化にした理由: 残す項 (K1 / OwnBoardB / TT salt) と**交互に並んでいて**、
//   行の削除は eval を壊すリスクが高い。 ビルドで未定義が出れば囲み漏れが分かる。
// p2shogi(rust) NNUE é§ç©ï½¾E¡é«¢E¢è¬¨E°ç¸ºE® C++ éï½»è®ã»(q_kda10_A.bin èE½¾è ¢ã»
//
// ç¹§E¢ç¹ï½¼ç¹§E­: HalfKA_hm(Friend) 1536 LayerStacks(8 bucket) + è®åEºé³³SQT + progress8kpabs
//         + dual_psqt / dual_cross / pairshape headç¸²ã»
// è¬¨E°èï½¤èå¢ï½§å¥ã» port/MATH_SPEC.md è¿ããEç¸²ãï½§E»è®æ¦ãE = C:\project\p2shogi\crates\rshogi-core\src\nnueç¸²ã»

#include "../../config.h"

#if defined(EVAL_P2NNUE)

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#if defined(USE_AVX2)
#include <immintrin.h>
#endif
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "../../bitboard.h"
#include "../../evaluate.h"
#include "../../memory.h"
#include "../../misc.h"
#include "../../position.h"
#include "../../usi.h"
#include "../../spsa_params.h"
#if defined(P2_SRCH_BUNDLE_A)
#include "../../srch_bundle_a.h"
#endif
#include "../../p2volm.h"	// V1 = P2Volm::value (案 B Regime LayerStack の鍵。 2026-10-07)
#include "p2nnue_reader.h"
#include "p2nnue_threat.h"
#include "p2nnue_threat_direct15.h"
#include "p2nnue_aux.h"
#include "p2nnue_auxcheap.h"
#include "p2nnue_endgame_tempo.h"
#include "p2nnue_mob.h"   // B6 可動域 (P2_MOB)
// 2026-11-05 (fable): profile 16 (hanging-split) だけが読む board_effect を、 LONG_EFFECT_LIBRARY の無い
//   ビルドでも compile できるようにする。 profile 16 は p2nnue_threat.h で LE 必須を #error しているので、
//   LE 無しビルドでこのマクロが 0 を返す経路は THREAT_HANG_SPLIT=false の定数畳み込みで消える。
#if defined(LONG_EFFECT_LIBRARY)
#define P2_HANG0(pos, c, sq) ((int)((pos).board_effect[(c)].e[(int)(sq)] == 0))
#else
#define P2_HANG0(pos, c, sq) (0)
#endif
// P2_ENDGAME_TEMPO ç¸ºE¯èEçEclang variant ç¸ºE§ç¸ºE®ç¸ºE¿è­çæè¹æ¶âE¹§ä¹Âã»

using namespace YaneuraOu;

namespace YaneuraOu::Eval::P2NNUE {

// ===== ç¹§E¢ç¹ï½¼ç¹§E­é«±æ¨ï½¾æï½­å¥ã»èï½ºè³å£¼E®å£½çE=====
static constexpr int HALFKA_DIM    = 73305;      // 45 * 1629(è¿ï½¹è ï½´é©¥åï½©Eºé«¢èÂã»Tç¹§Eµç¹§E¤ç¹§Eºç¸ºE«èæ´Eç¸ºå£»E¸Âè³ã»
static constexpr int PIECE_INPUTS  = 1629;       // E_KING
static constexpr int FE_HAND_END   = 90;
static constexpr int FE_OLD_END    = 1548;
static constexpr int E_KING_BP     = 1629;
// LayerStack èE½ºèãEbucket è¬¨E°ç¸²ãéè³ã»Legacy progress8 / P4H2 = é¨¾E²é¦æï½ºE¦4EE²æé¬§ã»)ç¸ºE¯ 8ç¸²ã»
// -DP2_LSMODE_P8H2 ç¹èÎç¹å³¨ã»ç¸ºE¿ P8H2(é¨¾E²é¦æï½ºE¦8EE²æé¬§ã»)= 16ç¸²ãEEèE½¦éè²ãE shogi_layerstack.rs ç¸ºE®
// NUM_BUCKETS(è­ï½¢è³ã»8 / feature lsmode-p8h2 ç¸ºE§ 16)ç¸ºE¨èE½¾éï½°ç¸²ESQT / Bucket é©æ¦ãE / auxcheap é²ã»
// net ç¸ºE® per-bucket ç¹§E»ç¹§E¯ç¹§E·ç¹ï½§ç¹ï½³è¬¨E°ç¹§è²éEè­ãâè±ï½ºç¹§âï½ç¸ºæºï½ç¸²E½et ç¹§åå¶ç¸ºã»âEE¦éåEã³ç¹ï½«ç¹å³¨âè ¢ã»âE³Âé¾E´ç¸ºè¼âç¹§ä¹ï¼E¸ºE¨ç¸²ã»
#if defined(P2_LSMODE_P8H2) || defined(P2_LSMODE_PROGREGIME)
static constexpr int NUM_BUCKETS   = 16;
#else
static constexpr int NUM_BUCKETS   = 8;
#endif
static constexpr int PROG_W        = 81 * FE_OLD_END; // 125388

// ç¹èEã£ç¹è¼ãéï½ºè«æçç¸ºE®è­Âèï½§èï½¤(èE½¾è ¢æãç¹ï½¼ç¹§E­ç¸ºE®è³ä¼åç¸²ã»024-8-64 ç¸ºE¯ L3=64)ç¸²ã»
// 🚨2026-11-08: 幅 1536 + 持駒 dense 14 列 では FC0_IN = pad32(1536+14) = **1568** になり
//   FT_MAX 1536 を超えて static_assert (205 行) で止まる。 1600 に上げる。
//   tr / full / ext_* の stack 配列が 1 本あたり +128 要素 (i16 なら +256 B) 増えるだけで、
//   幅 1024 のビルドでは使われない領域なので **NPS への影響は無い**
//   (バッファは大きくなるが触らない / ★ホットパスでのヒープ割り当て禁止は守っている)。
static constexpr int FT_MAX    = 1600;
// 2026-09-15 L1 24 (-DP2_L1_MAIN=24): L1_OUT 25 / L2_PAD 64 を収めるため上限を広げる。
// 旧 define P2_RESIDUAL_WIDE16 は P2_L1_MAIN=15 の別名として扱う (旧ビルドスクリプト互換)。
#if defined(P2_RESIDUAL_WIDE16) && !defined(P2_L1_MAIN)
#define P2_L1_MAIN 15
#endif
//   配列の上限だけなので WIDE16 ビルドの計算は変わらない。
static constexpr int L1OUT_MAX = 32;
static constexpr int L2PAD_MAX = 64;
static constexpr int L3_MAX    = 64;
// 2026-11-07: **L4 = 4 層目 (`fc1b`)**。 `-DP2_L4=<幅>` で有効。 **無指定は 0 = 現行と同一コード**
//   (constexpr 分岐なので無効ビルドは bit-exact。 検算 (b) で 49 局面の手と nodes を照合する)。
//
// 🎯幅 64 だけが**恒等初期化**できる: 前段の出力は既に `clampi(x >> 6, 0, 127)` の u8 で、
//   i8 の対角 **64** なら `clamp((64·x) >> 6, 0, 127) = x` が x = 0..127 の全てで成り立つ。
// 🚨`affine_par` は 8 出力単位でしか計算しないので幅は 8 の倍数。 配列上限は L4_MAX = 64。
#if defined(P2_L4)
static constexpr int L4 = P2_L4;
static_assert(L4 % 8 == 0 && L4 >= 8 && L4 <= 64, "P2_L4 は 8 の倍数で 8..64 (affine_par が 8 行単位)");
#else
static constexpr int L4 = 0;
#endif
static constexpr int L4_MAX = 64;

// ===== ç¹§E¢ç¹ï½¼ç¹§E­è°ºE¡èãEç¹§E³ç¹ï½³ç¹ä»£ãE¹ï½«è­ãå´è³ã»ç¹è¼Îè­Âé©ï½©è¹ã»=====
// ç¹§E¢ç¹ï½¼ç¹§E­è»ã»å´E ç¹èÎç¹ç»å¾ç¸ºE« P2_ARCH_1024 ç¹§ã»define ç¸ºå¶Eç¸ºE¨ 1024-8-64ç¸²ã»
//   èãE MSBuild ... /p:Configuration=Release-P2NNUE  (1536-16-32, è­ï½¢è³ã»
//       MSBuild ... è®åºã»ç¸ºE« P2_ARCH_1024 ç¹§å®ï½¿E½èï£°         (1024-8-64)
// load è­ãâEarchè­ãEE­æãEç¸ºä¹ï½è³æ»ï½¬E¡èãEEé±E­ç¸ºE¿ç¸²âï¼E¸ºèãEè³å£½çç¸ºE¨è³Âé¾E´ç¸ºå¶Eç¸ºåºE¤æE½¨E¼ç¸ºå¶EEè³å ºE¸Âé¾E´ç¸ºE¯ç¹§E¨ç¹ï½©ç¹ï½¼)ç¸²ã»
#if defined(P2_ARCH_768T)
// threat èï½¥èå¸·æ²¿ (2026-07-18): HalfKA_hm(73305) + Threat(216720) é¨¾E£é¨ä»ãEèå¸Âã»
// FT è·ã»768ç¸²Â´1/L2/L3 ç¸ºE¯ P2_ARCH_1024 ç¸ºE¨è·å¾¡E¸Â (8/14/64)ç¸²Eµhreat FT èï£°éå±â ç¹§ç¿«Âã»
static constexpr int FT_DIM = 768,  FT_HALF = 384, L1_OUT = 8,  MAIN_DIM = 7,  L2_IN = 14, L2_PAD = 32, L3 = 64;
#elif defined(P2_ARCH_1280)
// 2026-11-04: **幅 1280** (1024 と 1536 の間。 幅と NPS の関係が閾値か帯域かを分ける点)。
//   1024 -> 1280 は**広げる**側なので widen_ckpt.py の noise 0 で関数を厳密に保存できる
//   = 枝刈り net の交絡なしで測れる。 FT_HALF=640。
#if defined(P2_L1_MAIN)
static constexpr int FT_DIM = 1280, FT_HALF = 640, L1_OUT = P2_L1_MAIN + 1, MAIN_DIM = P2_L1_MAIN,
                     L2_IN = 2 * P2_L1_MAIN, L2_PAD = ((2 * P2_L1_MAIN + 31) / 32) * 32, L3 = 64;
#else
static constexpr int FT_DIM = 1280, FT_HALF = 640, L1_OUT = 16, MAIN_DIM = 15, L2_IN = 30, L2_PAD = 32, L3 = 64;
#endif
#elif defined(P2_ARCH_512)
// 2026-11-04: **幅 512** (幅と NPS の関係を確かめるための点。 768 は threat 抜きで ±0% だった)。
//   P2_ARCH_1024 / 768 / 1536 と同じ形 (L3=64) の幅違い。 FT_HALF=256 で lo=hi=32 は収まる。
#if defined(P2_L1_MAIN)
static constexpr int FT_DIM = 512, FT_HALF = 256, L1_OUT = P2_L1_MAIN + 1, MAIN_DIM = P2_L1_MAIN,
                     L2_IN = 2 * P2_L1_MAIN, L2_PAD = ((2 * P2_L1_MAIN + 31) / 32) * 32, L3 = 64;
#else
static constexpr int FT_DIM = 512, FT_HALF = 256, L1_OUT = 16, MAIN_DIM = 15, L2_IN = 30, L2_PAD = 32, L3 = 64;
#endif
#elif defined(P2_ARCH_768)
// 2026-11-04: **幅 768** (nodchip の最適幅 / NPS +11.3% = 約 +1.55pt を狙う腕)。
//   🚨P2_ARCH_768T は L1_OUT=8 / MAIN_DIM=7 / L2_IN=14 の**旧い形**で P2_L1_MAIN を見ない。
//     本番 (P2_ARCH_1024 + P2_L1_MAIN=15) と比較できないのでこちらを使う。
//   P2_ARCH_1024 / P2_ARCH_1536 と同じ形 (L3=64) の幅違い。
//   threat の着地は FT[0..THREAT_LO_DIM) と FT[FT_HALF..FT_HALF+THREAT_HI_DIM) で、
//   FT_HALF が 512 -> 384 に動くだけ (lo=hi=32 なので 384 に十分収まる)。
#if defined(P2_L1_MAIN)
static constexpr int FT_DIM = 768, FT_HALF = 384, L1_OUT = P2_L1_MAIN + 1, MAIN_DIM = P2_L1_MAIN,
                     L2_IN = 2 * P2_L1_MAIN, L2_PAD = ((2 * P2_L1_MAIN + 31) / 32) * 32, L3 = 64;
#else
static constexpr int FT_DIM = 768, FT_HALF = 384, L1_OUT = 16, MAIN_DIM = 15, L2_IN = 30, L2_PAD = 32, L3 = 64;
#endif
#elif defined(P2_ARCH_1024T)
  // Threat-E48: FT=1024, MixWide16 layers (main15 + skip1, L2 input32).
  static constexpr int FT_DIM = 1024, FT_HALF = 512, L1_OUT = 16, MAIN_DIM = 15, L2_IN = 32, L2_PAD = 32, L3 = 64;
#elif defined(P2_ARCH_1536)
// 2026-11-03: **幅 1536** (v4 と同じ重み予算で threat を持つ腕)。
//   v4 は SFNN_halfkahm2_**2304_15_64** / PSQT なし / threat なしで総重み 169.5M。
//   我々は 73,434 x W + 56,099,016 なので **W=1536 で 168.9M = v4 の 99.6%**。
//   🚨下の `#else` の 1536 分岐は **L3=32** の tanuki 互換なので使えない。 こちらは
//     P2_ARCH_1024 + P2_L1_MAIN と同じ形 (L3=64) の幅違い。
//   threat の着地は FT[0..THREAT_LO_DIM) と FT[FT_HALF..FT_HALF+THREAT_HI_DIM) で、
//   FT_HALF が 512 -> 768 に動くだけ (学習器 8551 の `half = ft_out/2` と同じ式)。
#if defined(P2_L1_MAIN)
static constexpr int FT_DIM = 1536, FT_HALF = 768, L1_OUT = P2_L1_MAIN + 1, MAIN_DIM = P2_L1_MAIN,
                     L2_IN = 2 * P2_L1_MAIN, L2_PAD = ((2 * P2_L1_MAIN + 31) / 32) * 32, L3 = 64;
#else
static constexpr int FT_DIM = 1536, FT_HALF = 768, L1_OUT = 16, MAIN_DIM = 15, L2_IN = 30, L2_PAD = 32, L3 = 64;
#endif
#elif defined(P2_ARCH_1024)
#if defined(P2_RESIDUAL_MIXWIDE16)
static constexpr int FT_DIM = 1024, FT_HALF = 512, L1_OUT = 16, MAIN_DIM = 15, L2_IN = 32, L2_PAD = 32, L3 = 64;
#elif defined(P2_L1_MAIN)
// 2026-09-15: 普通の SFNN 構成。 L1 = main P2_L1_MAIN 行 + skip 1 行 (出力へ直加算)、 L2 入力 = [main^2, main]。
//   例: -DP2_L1_MAIN=15 (従来の P2_RESIDUAL_WIDE16 と同一) / -DP2_L1_MAIN=24 (L1 24)。
//   ファイルの fc1 行は pad32(L2_IN) byte ずつ並ぶので L2_PAD もそれに合わせる (main 24 なら 48 → 64)。
static constexpr int FT_DIM = 1024, FT_HALF = 512, L1_OUT = P2_L1_MAIN + 1, MAIN_DIM = P2_L1_MAIN,
                     L2_IN = 2 * P2_L1_MAIN, L2_PAD = ((2 * P2_L1_MAIN + 31) / 32) * 32, L3 = 64;
#elif defined(P2_RESIDUAL_MIX16)
static constexpr int FT_DIM = 1024, FT_HALF = 512, L1_OUT = 8,  MAIN_DIM = 7,  L2_IN = 16, L2_PAD = 32, L3 = 64;
#else
static constexpr int FT_DIM = 1024, FT_HALF = 512, L1_OUT = 8,  MAIN_DIM = 7,  L2_IN = 14, L2_PAD = 32, L3 = 64;
#endif
#else
static constexpr int FT_DIM = 1536, FT_HALF = 768, L1_OUT = 16, MAIN_DIM = 15, L2_IN = 30, L2_PAD = 32, L3 = 32;
#endif
static_assert(L1_OUT <= L1OUT_MAX && L2_PAD <= L2PAD_MAX, "L1_OUT / L2_PAD exceed array bounds");
static_assert(L2_PAD == ((L2_IN + 31) / 32) * 32, "L2_PAD must equal the file row stride pad32(L2_IN)");
// fc0 の SIMD 用並べ替え表 (fc0_ws) に載せる行数。 affine_par (AVX2) は 8 出力単位でしか計算しないので 8 の倍数に切り捨て、
// 余りの行 (L1_OUT = 25 なら skip 行 1 本) は内積で別に計算する。
static constexpr int FC0_WS_ROWS = L1_OUT - L1_OUT % 8;

// 2026-11-07: **持駒 14 元 dense** (`-DP2_HAND_DENSE`)。 fc0 の入力に 14 列を concat する。
//   設計 runs/prereg/DESIGN-20261107-HANDFEAT-FABLE.md §3 候補 A。
//
// 🎯持駒の情報は既に HalfKA_hm の手駒 90 スロットから入っている。 足すのは**直通の経路**で、
//   (i) 自玉位置ごとに 45 分割されない / (ii) FT の clamp と pairwise の後ろ /
//   (iii) bucket 別の L1 と直接交互作用する、 3 点が違う。
// 🚨目盛りは学習器と厳密に揃える: u8 lane = `k * c`、 k = {歩 7, 香 31, 桂 31, 銀 31, 金 31, 角 63, 飛 63}。
//   最大 126 なので maddubs の飽和不変条件 (a <= 126 かつ |w| <= 127) を満たす。
// 🚨列の順は**学習器の HAND_PIECE_TYPES の順 = 歩 香 桂 銀 金 角 飛**。
//   YO の PieceType 順 (歩 香 桂 銀 **角 飛 金**) とは違う。
// 2026-11-07: **値を取る define** にした (`-DP2_HAND_DENSE=14` / `=76`)。
//   14 = スカラー 1 本 x 7 駒種 x 2 視点 (枚数に線形)
//   76 = 温度計符号化 (歩 18 / 香桂銀金 各 4 / 角飛 各 2) x 2 視点 (枚数に非線形)
//   🚨`-DP2_HAND_DENSE` (値なし) はコンパイルエラーになる。 必ず値を付ける。
#if defined(P2_HAND_DENSE)
static constexpr int HC_DIMS = P2_HAND_DENSE;
static_assert(HC_DIMS == 14 || HC_DIMS == 76, "P2_HAND_DENSE は 14 (スカラー) か 76 (温度計)");
// 学習器の pad32(FT_DIM + HC_DIMS) と**同じ式**で出す (決め打ちしない)。
static constexpr int HC_PAD = ((FT_DIM + HC_DIMS + 31) / 32) * 32 - FT_DIM;
#else
static constexpr int HC_DIMS = 0;
static constexpr int HC_PAD  = 0;
#endif
static constexpr int FC0_IN = FT_DIM + HC_PAD;
static_assert(FC0_IN <= FT_MAX, "FC0_IN が tr[FT_MAX] を超える");
static_assert(HC_PAD == 0 || FT_DIM + HC_PAD == ((FT_DIM + HC_DIMS + 31) / 32) * 32,
              "FC0_IN は学習器の pad32(FT_DIM + HC_DIMS) と一致しなければならない");

// 2026-09-09: threat accumulator の出力幅。 P2_THREAT_OUT でビルド時に絞れる。
//   full threat の accumulator 税 -18.5% は「36 行 x 幅」の i16 加減算に比例するので、
//   幅を 1024 -> 256 にすると行が 2 キャッシュラインに収まり L2 常駐化して 8 倍以上効く。
//   K = FT_DIM のとき従来と完全一致 (入れ子モデル)。
static constexpr int THREAT_ACC_DIM = StateInfo::P2_THREAT_ACC_DIM;
static_assert(THREAT_ACC_DIM > 0 && THREAT_ACC_DIM <= FT_DIM,
              "P2_THREAT_OUT は 1..FT_DIM の範囲であること");
// ��2026-09-10: threat �� FT �̑O���E�㔼��**�Ώ̂�**�z��B
//   threat acc �� K ���̂܂܁B ���̑O�� THREAT_LO_DIM �� FT[0..) �ɁA
//   �㔼 THREAT_HI_DIM �� FT[FT_HALF..) �ɉ��Z����B pairwise_mul ��
//   out[j] = clamp(acc[j]) * clamp(acc[j+FT_HALF]) �Ȃ̂ŁA �]�� (�擪 K ��������) ����
//   K < FT_HALF �̂Ƃ��ς�**�Б��ɂ��� threat �����Ȃ�**��Ώ̂ȍ\���������B
//   K == FT_DIM �̂Ƃ��� LO=HI=FT_HALF �őS�����ɏ��A �]���Ɗ��S��v�B
static constexpr int THREAT_LO_DIM = ((THREAT_ACC_DIM + 1) / 2) > FT_HALF ? FT_HALF : ((THREAT_ACC_DIM + 1) / 2);
static constexpr int THREAT_HI_DIM = THREAT_ACC_DIM - THREAT_LO_DIM;
// ---- qsearch 中の threat 影響度 (2026-09-11) ----
// 2026-09-11 の実測: 静的評価 (eval / MultiPV) では threat と baseline の差は数 cp なのに、
//   qsearch を通すと楽観度の差が **+20.61 cp (95%CI [+16.99, +24.15])** に増幅される。
//   qsearch は駒の取り合いを解決する探索 = 駒が互いに利いている局面の連続で、
//   **threat 特徴が最も強く反応する場所**。 ここで評価が余計に動くと qsearch の返り値がずれる。
// P2QSearchThreatScale (USI option、 0..256、 既定 256 = 等倍 = 従来と bit-exact) で
//   qsearch 中だけ threat accumulator の寄与を scale/256 倍する。 0 なら qsearch 中は threat 無効。
extern thread_local bool g_in_qsearch;                 // search 側が立てる
static int  g_qs_threat_scale = 256;      // 256 = 等倍
static_assert(THREAT_HI_DIM <= FT_HALF, "threat �㔼�� FT_HALF �𒴂��Ă���");

// ---- 案1 ブロック射影 (2026-09-30、 fable 設計) ----
//
// threat accumulator (K = THREAT_ACC_DIM = 64) は FT 1024 次元のうち **64 次元**
// (前半 32 + 後半 32) にしか着地していない。 pairwise_mul の積 512 個のうち
// threat が混ざるのは 32 個だけ = 表の情報が最終層まで届いていない。
//
// net の P2TPRJ section が持つ射影係数 p[j] (Q15、 学習器で ±1 に clip 済み) を使い
//     ft[j] += round(p[j] * thr[j % K] / 2^15)      (j = 0 .. FT_DIM)
// を **全次元**に足す。 threat 表も列挙手続きも変わらないので列挙側の NPS 税は据え置き。
//
// 🚨FT_HALF % K == 0 なので、 lo 側 (j..j+15) と hi 側 (j+half..j+half+15) は
//   **同じ** thr スライス b[j % K] を使う (j+half ≡ j (mod K))。 load は 1 回で済む。
#if defined(P2_THREAT_PROJ)
static constexpr bool P2_HAS_TPRJ = true;
alignas(64) static std::int16_t g_tprj_w[NUM_BUCKETS][FT_DIM] = {};
static bool g_tprj_loaded = false;
// hi 側 (FT の後半) が受ける threat チャンネルのずらし量。 net の section (version 2) から読む。
//   0 = 従来 (2 因子が同じチャンネル = t_c^2 しか作れない)。
//   32 = 512 個の積すべてが**異なる 2 チャンネルの積** t_c t_c' を持つ。
// 🚨16 の倍数であること (AVX2 の 16 lane 連続 load)。 load 時に検査する。
static int g_tprj_offset = 0;

// 16 次元ブロックごとのチャンネルずらし量 (section version 3)。 SIMD の 1 反復が 16 次元なので
// ブロック内で一定なら 16 本の連続 load を保てる。 値は 0..K-1 の任意 (複製バッファから読む)。
static int g_tprj_blk[FT_DIM / 16] = {};
static_assert(FT_HALF % THREAT_ACC_DIM == 0,
              "P2_THREAT_PROJ: FT_HALF must be a multiple of THREAT_ACC_DIM");
static_assert(THREAT_ACC_DIM % 16 == 0,
              "P2_THREAT_PROJ: THREAT_ACC_DIM must be a multiple of 16 (AVX2 lane)");
#else
static constexpr bool P2_HAS_TPRJ = false;
// 🚨`if constexpr` の捨てられる側も非テンプレート文脈では型検査されるので、
//   フラグ無しビルドでも実体が要る。 P2_HAS_TPRJ が false なので dead code になる。
alignas(64) static const std::int16_t g_tprj_w[NUM_BUCKETS][FT_DIM] = {};
static constexpr bool g_tprj_loaded = false;
static constexpr int g_tprj_offset = 0;
static constexpr int g_tprj_blk[FT_DIM / 16] = {};
#endif

// 🚨🚨2026-10-03: この TL1 の宣言は元々 `#if defined(P2_THREAT_PROJ)` の**中**にあったため、
//   射影なし (= 本番構成) でビルドすると P2_HAS_TL1 / g_tl1_loaded / g_tl1_w が
//   未宣言になってコンパイルエラーになっていた。
//   `if constexpr` の**捨てられる側も型検査される**ので、 ダミーは常に必要。
//   → 射影のガードの**外**に出した。 TL1 と射影は独立に切り替えられる。
// ---- (3) TL1: threat を L1 に直結する (FT の clamp を迂回)。 2026-10-01 ----
//
// FT の 48.4% が毎局面 clamp [0,254] されている (実測) ので、 threat が FT 経由で運ぶ情報は
// 約半分が捨てられている。 clamp を通さない経路を 1 本足す:
//   l1[o] += ( sum_c thr_stm[c]*w[bucket][o][c] + sum_c thr_ntm[c]*w[bucket][o][K+c] ) >> 5
// 目盛り: thr(i16) = 254*float、 L1 i32 = 8128*float → 8128/254 = 32 = 1024>>5
//   (w_q = round(1024*w)、 学習器側で ±2 に clip しているので |w_q| <= 2048)
#if defined(P2_THREAT_L1)
static constexpr bool P2_HAS_TL1 = true;
alignas(64) static std::int16_t g_tl1_w[NUM_BUCKETS][L1_OUT][2 * THREAT_ACC_DIM] = {};
static bool g_tl1_loaded = false;
// 2026-10-08: AVX2 の積和 (tl_dot_2k) を使ってよいか。 load 時に max|q| <= 2048 を確かめて立てる。
//   i32 レーン (16 項) があふれない上界の根拠: 16 * 32767 * 2048 = 1.07e9 < 2^31。
//   学習器は clip ±2 -> round(1024*w) なので |q| <= 2048 が保証されるが、 net を信用しない。
static bool g_tl1_simd_ok = false;
#else
static constexpr bool P2_HAS_TL1 = false;
// 🚨`if constexpr` の捨てられる側も非テンプレート文脈では型検査されるので実体が要る。
alignas(64) static const std::int16_t g_tl1_w[NUM_BUCKETS][L1_OUT][2 * THREAT_ACC_DIM] = {};
static constexpr bool g_tl1_loaded = false;
static constexpr bool g_tl1_simd_ok = false;
#endif

// ---- L1Q128 (2026-11-16): fc0 本体 MAIN_DIM 行の刻み。 DESIGN-20261116-L1Q128-FABLE.md ----
// 6 = 旧 net (本体の重み x64 / bias x8128) / 7 = arch に "AffineTransformL1Q128[" を持つ net (重み x128 / bias x16256)。
// skip 行 (l1[MAIN_DIM]) はどちらの net でも x64 / x8128 のまま。 load_network が成功して g_net を差し替える所で
// だけ書く (失敗した読み込みで前の net の値を壊さない)。 propagate_bucket が読む。
static int g_l1_main_shift = 6;

// ---- (4) TL2: threat を L2 入力に直結する (L1 = MAIN_DIM の首を迂回)。 2026-10-08 ----
//
// TL1 は L1 (L1_OUT 本) に足すので首を**通る**。 TL2 は sqr / crelu の**後ろ**、
// fc1 の入力 (L2_IN 本) に直接足すので L1 の幅という律速を迂回する。
//   l2in[j] += ( Σ_c thr_cat[c] * v[bucket][j][c] ) >> sh    (j = 0..L2_IN-1、 clamp の前)
// 目盛り: thr(i16) = 254·float、 L2 入力 (u8) = 127·float、 v_q = round(S·v)
//   127·Σ thr_f·v = (127/254)·Σ thr_i16·v = (1/2)·Σ thr_i16·v
//                 = (1/(2·S))·Σ thr_i16·v_q = (Σ thr_i16·v_q) >> log2(2S)   ✅
// 🚨S は P2TL2 section の version で決まる (2026-10-09)。
//   ver 1: S =   64 -> shift  7  (2026-10-08 の初版、 |v_q| <= 128)
//   ver 2: S = 1024 -> shift 11  (|v_q| <= 2048、 量子化誤差 34.28% -> 2.26%)
//   どちらも |v_q| <= 2048 の内側なので和が i32 レーンに收まり、
//   `tl_dot_2k` (`_mm256_madd_epi16`) をそのまま使える。
//   TL1 とは Q と着地が違うので export を使い回さない。
#if defined(P2_THREAT_L2)
static constexpr bool P2_HAS_TL2 = true;
alignas(64) static std::int16_t g_tl2_w[NUM_BUCKETS][L2_IN][2 * THREAT_ACC_DIM] = {};
static bool g_tl2_loaded = false;
// 🚨P2TL2 section の version = 量子化の目盛り S を決める (2026-10-09)。
//   ver 1 -> S =   64、 shift 7   (2026-10-08 の初版。 旧 net はこれで bit-exact)
//   ver 2 -> S = 1024、 shift 11  (量子化誤差 34.28% -> 2.26%)
// 導出: 127*Σ thr_f*v = (1/(2S))*Σ thr_i16*v_q なので shift = log2(2S)。
static std::uint32_t g_tl2_ver = 1;
// AVX2 の積和 (tl_dot_2k) を使ってよいか。 不変条件は TL1 と同一 (i32 レーン 16 項、
//   16 * 32767 * 2048 = 1.07e9 < 2^31)。 TL2 は ver1 で |q| <= 128、 ver2 で |q| <= 2048
//   のはずだが net を信用しない。
static bool g_tl2_simd_ok = false;
// ThreatL2Gain (USI option、 0..4096、 既定 256 = 等倍 = 従来と bit-exact) 。 2026-10-09。
//   t2[] の寄与を gain/256 倍する = net 側で q を定数倍したのと同値。
//   🚨既定 (256) では一切乗算をしない (分岐 1 つだけ、 j のループの外で読む)。
static int g_tl2_gain = 256;
#else
static constexpr bool P2_HAS_TL2 = false;
// 🚨`if constexpr` の捨てられる側も非テンプレート文脈では型検査されるので実体が要る。
alignas(64) static const std::int16_t g_tl2_w[NUM_BUCKETS][L2_IN][2 * THREAT_ACC_DIM] = {};
static constexpr bool g_tl2_loaded = false;
static constexpr bool g_tl2_simd_ok = false;
static constexpr std::uint32_t g_tl2_ver = 1;
static constexpr int g_tl2_gain = 256;
#endif


// ---- (5) TG: threat による **乗法ゲート** (sqr 側だけ)。 2026-10-10 ----
//   設計書 runs/prereg/DESIGN-20261009-THREAT-GATE.md の 2〜3 節。
//   TL2 (加法) の t2(thr) は L1 の値に依らない**主効果**。 TG は sqr (= l1^2 >> 19) に
//   threat 由来のゲート g(thr) を**掛ける**ので threat x L1 の**交差項**になり、
//   TL2 の関数族には含まれない (これが TG の売り。 fable レビュー 2026-10-09)。
//     pre_sq = (sqr * g) >> 8、  g = clamp(256 + tg[i], 0, 1024)   (256 = 等倍、 1024 = x4.0)
//   🚨lin 側 (v >> 6) は**一切触らない**。 lin のゲートは l1 のスケールと同じ形で、
//     パラメータも半分 (NUM_BUCKETS x MAIN_DIM x 2K = 8 x 15 x 128 = 15,360) で済む。
//
// 目盛り S_g = 1024 の導出 (🚨**学習器と同一の算術にすること**):
//   狙い    g/256 = 1 + Σ thr_f * u            (u = float のゲート重み)
//   engine  thr_i16 = 254 * thr_f、  u_q = round(1024 * u)
//           (Σ thr_i16 * u_q) >> 10 = 254 * Σ thr_f * u
//   → engine の g = 256 + 254 * Σ thr_f * u
//   🚨よって**学習器の順伝播は g = 1 + (254/256) * Σ thr_f * u** を使う約束。
//     256 で揃えると 0.79% ずれる (★engine_patch_verification_20260915:
//     eval が bit-exact でも学習と推論が食い違えば net は黙って壊れる)。
//   g の clamp [0,1024] も学習器の graph に同じ形で入れる (順伝播が違うと勾配がずれる)。
#if defined(P2_TG)
static constexpr bool P2_HAS_TG = true;
// i16[NUM_BUCKETS][MAIN_DIM][2K]。 sqr 側 MAIN_DIM 本だけなので TL2 (L2_IN 本) の半分。
alignas(64) static std::int16_t g_tg_w[NUM_BUCKETS][MAIN_DIM][2 * THREAT_ACC_DIM] = {};
static bool g_tg_loaded = false;
// AVX2 の積和 (tl_dot_2k) を使ってよいか。 不変条件は TL1 / TL2 と同一:
//   i32 レーンに 16 項、 |thr| <= 32767、 |q| <= 2048 で 16*32767*2048 = 1.07e9 < 2^31。
//   S_g = 1024 かつ学習器の clip |u| <= 2 なら |q| <= 2048 だが、 net を信用せず load 時に検査する。
static bool g_tg_simd_ok = false;
// ThreatGateScale (USI option、 0..1024、 既定 256 = 等倍)。 tg[] (= g の 256 からのずれ) を
//   scale/256 倍する。 🚨既定 256 では乗算を 1 回もしない = TG 無しと bit-exact。
static int g_tg_scale = 256;
#else
static constexpr bool P2_HAS_TG = false;
// 🚨`if constexpr` の捨てられる側も非テンプレート文脈では型検査されるので実体が要る。
alignas(64) static const std::int16_t g_tg_w[NUM_BUCKETS][MAIN_DIM][2 * THREAT_ACC_DIM] = {};
static constexpr bool g_tg_loaded = false;
static constexpr bool g_tg_simd_ok = false;
static constexpr int g_tg_scale = 256;
#endif


// threat èï½¥èãE216,720 è°ºE¡èãEç¹§ã»FT ç¸ºE«é¨¾E£é¨èâEç¹§ä¹Â°ç¸²EE_ARCH_768T ç¸ºE®ç¸ºE¨ç¸ºéªã»ç¸ºE¿è­çæç¸²ã»
// 🚨2026-11-03: arch を列挙する #if なので、 新しい幅を足すときは**ここにも足す**。
//   P2_ARCH_1536 を足したとき、 ここを漏らして DIRECT_THREAT_DIM 等が未定義になった。
#if defined(P2_ARCH_768T) || defined(P2_ARCH_1024T) || defined(P2_ARCH_1024)  || defined(P2_ARCH_1536) || defined(P2_ARCH_768) || defined(P2_ARCH_512) || defined(P2_ARCH_1280)
static constexpr bool P2_HAS_THREAT        = !Threat::THREAT_FACTORIZED && !Threat::THREAT_DIRECT;
static constexpr bool P2_HAS_THREAT_FACTOR =  Threat::THREAT_FACTORIZED;
static constexpr bool P2_HAS_THREAT_DIRECT =  Threat::THREAT_DIRECT;
static constexpr int  THREAT_DIM            = Threat::THREAT_DIM_PROFILE;
static constexpr int  DIRECT_THREAT_DIM     = Threat::THREAT_PROFILE_ID == Direct15::PROFILE_ID ? Direct15::DIM : THREAT_DIM;
#else
static constexpr bool P2_HAS_THREAT        = false;
static constexpr bool P2_HAS_THREAT_FACTOR = false;
static constexpr int  THREAT_DIM           = 0;
#endif
static constexpr int THREAT_FACTOR_DIM = Threat::THREAT_FACTOR_DIM;
#ifndef P2_THREAT_DIRECT_CAP_CP
#define P2_THREAT_DIRECT_CAP_CP 30
#endif
static_assert(P2_THREAT_DIRECT_CAP_CP >= 0 && P2_THREAT_DIRECT_CAP_CP <= 200,
              "P2_THREAT_DIRECT_CAP_CP must be in [0,200] (0 = no cap)");

// AUX èï½¥èãEKZ+HANG+STATKD, 600 è°ºE¡èãEpersp)ç¹§ã»FT ç¸ºE«é¨¾E£é¨èâEç¹§ä¹Â°ç¸²E£uild define -DP2_AUX ç¸ºE§è­çæç¸²ã»
// aux é©¥éªâ©ç¸ºE¯ i8 feature-major [600 * FT_DIM](~0.9MB, L2 è¶E¸é¬§ã»ç¸²ãEEè¬å¶ã»è·ã»eval ç¸ºE§ fresh éªè²E®ã»
// (éï½ºè½£E« ~35/perspç¹ï½»è»E©ç¸ºéªã»ç¹ãEã» 1 èãEç¸²ãæ´è³å¤ï½¾E©è­ãã»è æºæé¨¾å£¹EEç¹§E³ç¹ï½¼ç¹è½ã»è´èâ²è¾ï½¡èï½¹)ç¸²ã»
#if defined(P2_AUX)
static constexpr bool P2_HAS_AUX = true;
static constexpr int  AUX_DIM    = Aux::AUX_DIM_PER_PERSP; // 600
#else
static constexpr bool P2_HAS_AUX = false;
static constexpr int  AUX_DIM    = 0;
#endif

// ---- AUXCHEAP(KZ2x+HG6+OUT6, board_effect SIMD è¬ï½½èE½º, PSQT-only)----
// NPS ç¹§E²ç¹ï½¼ç¹éE¨åï½¸E¬èEçç¸²E£uild define -DP2_AUXCHEAP + -DLONG_EFFECT_LIBRARY ç¸ºE§è­çæç¸²ã»
// éï½ºè½£E« index ç«ãEPSQT 8 ç¹§E¹ç¹§E«ç¹ï½©ç¹ï½¼èï£°éãEFT é¦æåEéå±E ç¸ºEªç¸ºã»ç¸²ãã¾ç¸ºE¿ç¸ºE¯ 0 = eval è³æ¦ï½¤å³¨ã»è¬ï½¢éï½¢è­ï½¨è³æ¦ï½¤ã»
// (D=ON ç¸ºE¨ E=OFF ç¸ºE§ node è¬¨E°è³Âé¾E´ ç«ãEèï½¬è·E³ NPS)ç¸²ãEç¹§E­ç¹ï½£ç¹ï½³+PSQTèï£°éå±ã»è±ãEeval è ¢ã»âE³æº¯E¡å¾Âã»
#if defined(P2_AUXCHEAP)
static constexpr bool P2_HAS_AUXCHEAP = true;
// 0 é©¥éªâ© PSQT ç¹ãEã»ç¹æ¶ÎE[DIM_PER_PERSP * NUM_BUCKETS]ç¸²ãEEcompiler ç¸ºå¾Âæï½¸E¸ç¸ºE« 0ç¸²éªâéªE¼è­å¼±E ç¸ºE¦
// ç¹ï½«ç¹ï½¼ç¹å±Eè±¸å»âEGate1 ç¸ºE®é¨Eºè¬ï½¯ç¹§è´ï½ºåºèEç¸ºE®ç¹§å¸äºç¸ºèâç¹§âÂãolatile éï½±è­ï½¥ç¸ºE®è³æº¯E¡æ¢§å¾èï½¤ç¸ºE§è»æEè¹æ¶âE¹§ã»
// (è³E­éE½«ç¸ºE¯ 0 ç¸ºEªç¸ºE®ç¸ºE§ eval è³æ¦ï½¤å³¨Âââ¡ç¸ºã»optimizer ç¸ºE¯è³E­éE½«è³è´ã»ç¸ºE¨ç¸ºå±â»è¬E½±ç¸ºã»ç«ãEèï£°éå±Îç¹ï½¼ç¹é´æ¼é«¯E¤è´E»)ç¸²ã»
static volatile std::int32_t g_auxcheap_zero = 0;
static std::int32_t g_auxcheap_w[AuxCheap::DIM_PER_PERSP * NUM_BUCKETS];
static bool g_auxcheap_inited = false;
// STATKD-kd2 ç¸ºE® per-bucket side-weight ç¹ãEã»ç¹æ¶ÎEcount-linear SIMD-count è¿å»Âã»026-07-25)ç¸²ã»
//   g_statkd_wb[bucket][sd][idx]
//     = g_auxcheap_w[(STATKD_OFF + sd*SKD_PER_SIDE + idx)*NUM_BUCKETS + bucket]
//   sd = local side(0=us/side0, 1=them/side1)ç¸²EEx = (tc*SKD_NST+st)*SKD_NKB+kb (0..35, C1)ç¸²ã»
//   build_statkd() ç¸ºã»bytes_gtEEE ç¸ºE® SIMD ç¸ºE§èE½ºç¸ºåï½µE¶èE½¾æ¿¶E²ç¹§E«ç¹§E¦ç¹ï½³ç¹ãEraw.statkd[c][tc][st][kb]
//   (= color ç¸ºæâE flat 36 èãE C1 status4ç«ãE)ç¸ºE¨ç¸ºE®èã»E©éªãè³E¡éä¹ãç¸ºE® STATKD èEEE¸å¼±Eè å±Eç¸²ãE¼E¹§å¾ãE//   per-piece scan(occupied pop + board é¶E´é±E­ + LUT è¿ããE)ç¹§è²ï½»ã»E ç¸²ã»2 popcount + é¨æ¦æ±ç¸ºE«éï½®è¬ å¸Âã»
static std::int32_t g_statkd_wb[NUM_BUCKETS][2][AuxCheap::SKD_PER_SIDE];
static void build_statkd_wb() {
	using namespace AuxCheap;
	for (int b = 0; b < NUM_BUCKETS; ++b)
		for (int sd = 0; sd < 2; ++sd)
			for (int idx = 0; idx < SKD_PER_SIDE; ++idx) {
				const int local = STATKD_OFF + sd * SKD_PER_SIDE + idx;
				g_statkd_wb[b][sd][idx] =
					g_auxcheap_w[(std::size_t)local * NUM_BUCKETS + b];
			}
}
// net ç¸ºE® P2AUXC section ç¸ºä¹ï½è³æ»E¾ç¸ºE¿ç¹§åµÎç¹ï½¼ç¹ç»E¸å»â©ç¸ºã»load_network ç¸ºã»commit è­ãâéEä¹â»ç¹§ã»ç¸²ã»
// true ç¸ºE®é«¢èãE auxcheap_init() ç¸ºE®ç¹§E¼ç¹ï½­èä¹ï½ç¹§å®ï½¡å¾ï½ç¸ºEªç¸ºã»è³æ»E¾ç¸ºE¿ç¹§åE½¸é å¶ç¸ºéªE ç¸ºEªç¸ºã»ç¸²ã»
// marker è¾ï½¡ç¸ºã»net ç¸ºE¸ç¸ºE®èéªÎç¹ï½¼ç¹å³¨ãEfalse ç¸ºE«è¬ï½»ç¹§ç¿«ÂâµE¬E¡èãEeval ç¸ºE§è æºæç¸ºE©ç¸ºç¿«Eç¹§E¼ç¹ï½­èä¹ï½E// (net è¾ï½¡ç¸ºå±ã»ç¹ï½³ç¹â«Eµçï½·E¯ = volatile éï½±è­ï½¥ç¹§E¼ç¹ï½­ç¸ºE«ç¹§å»Eç¹ï½«ç¹ï½¼ç¹é´æ¼é«¯E¤è´E»ç¸²âãEè²ï½©èEEç¸²ã»
static bool g_auxcheap_loaded = false;
static inline void auxcheap_init() {
	if (g_auxcheap_inited) return;
	const std::int32_t z = g_auxcheap_zero;  // è³æº¯E¡æ¢§å¾ 0(volatile ç«ãEè³å£½çé¡E³ç¸ºE¿éï½¼ç¸ºE¿è³æ¦åºE
	for (int i = 0; i < AuxCheap::DIM_PER_PERSP * NUM_BUCKETS; ++i) g_auxcheap_w[i] = z;
	build_statkd_wb();  // ç¹§E¼ç¹ï½­é©¥éªâ©ç¸ºä¹ï½Eside-weight ç¹ãEã»ç¹æ¶Îç¹§åï½§çï½¯ã»è³ã»net ç¹ï½­ç¹ï½¼ç¹ç»å¾ç¸ºE¯ commit èï½´ç¸ºE§èè´E§çï½¯ã»ç¸²ã»
	g_auxcheap_inited = true;
}
// éªåï½¸E¬éï½¨ env(P2_NOAUXCHEAP=1 ç¸ºE§è¬ï½½èE½º+èï£°éç¤¼Eµçï½·E¯ç¹§åE½¸E¸ç¸ºæâEç¹§E¹ç¹§E­ç¹ãEã» = plain-768+LE é¶E¸è ãEç¸²ã»
static inline bool noauxcheap_env() {
	
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_NOAUXCHEAP") != nullptr;
	return v;
#else
	return false;   // ★NPS(A5): 関数内 static のガード変数ロードを消す
#endif
}
// è³E¡éä¹ãç¸ºE® auxcheap èEEE¸ã»active bucket ç¸ºE®ç¹§E¹ç¹§E«ç¹ï½©ç¹ï½¼ 2 èãEç¹§ã»board_effect ç¸ºä¹ï½Efresh éªè²E®å±Âã»
//   KZ2x/HG6/OUT6 = build_kzhgout + for_each_kzhgout(bucket ç¹§E¹ç¹§E«ç¹ï½©ç¹ï½¼è¹ã»ç¸²ã»
//   STATKD-kd2   = build_statkd(bytes_gtEEE SIMD + 72 popcount)ç¸ºE§é¨E¶èE½¾æ¿¶E²ç¹§E«ç¹§E¦ç¹ï½³ç¹å»Eè±ãEEC1)ç¸²ã»
//                  side-weight ç¹ãEã»ç¹æ¶ÎEg_statkd_wb ç¸ºE¨ç¸ºE®èã»E©éªãè³E¡éä¹ãèEEE¸å¼±Eè å±Eç¸²ã»
static void compute_auxcheap_psqt(const Position& pos, Color stm, Color nstm, int bucket,
                                  std::int32_t& out_stm, std::int32_t& out_nstm) {
	auxcheap_init();
	AuxCheap::Raw raw;
	AuxCheap::build_kzhgout(pos, raw);  // KZ/HG/OUT ç¸ºE®ç¸ºE¿(STATKD popcount ç¹§è²å±é©ï½¿)
	std::int32_t s_stm = 0, s_nstm = 0;
	AuxCheap::for_each_kzhgout(raw, stm, [&](int local) {
		s_stm += g_auxcheap_w[(std::size_t)local * NUM_BUCKETS + bucket];
	});
	AuxCheap::for_each_kzhgout(raw, nstm, [&](int local) {
		s_nstm += g_auxcheap_w[(std::size_t)local * NUM_BUCKETS + bucket];
	});
	// ---- STATKD-kd2: SIMD-count(build_statkd)+ èï½´è»E¥é©¥éªâ©èã»E©ã»----
	// build_statkd ç¸ºã»bytes_gtEEE + 72 popcount ç¸ºE§é¨E¶èE½¾æ¿¶E²ç¹§E«ç¹§E¦ç¹ï½³ç¹ãEraw.statkd[c][tc][st][kb]
	// (color ç¸ºæâE flat 36, C1)ç¹§åï½±ãE½ç¸²âãEè»E¥é©¥éªâ© g_statkd_wb[bucket][sd] ç¸ºE¨èã»E©éªE ç¸ºE¦è³E¡éä¹ãèEEE¸å¼±Eè å±Eç¸²ã»
	// bit-exact: bullet emit ç¸ºE¨è·å¾Â§ status(C1: att=a>=1 / def1=d>=1ç¸²âang/def2 ç¸ºE¯ drop)ç¹ï½»
	// è·å¾Â§ g_auxcheap_w ç¸ºä¹ï½èä¸»ã»ç¸²E´tm éä¹ãç¸ºE¯èï½´0=stmæ¿¶E²/èï½´1=nstmæ¿¶E²(dump for_each ç¸ºE¨è·å¾¡E¸Â relabel)ç¸²ã»
	AuxCheap::build_statkd(pos, raw);
	const std::int32_t* const W0 = g_statkd_wb[bucket][0];  // side0(us)
	const std::int32_t* const W1 = g_statkd_wb[bucket][1];  // side1(them)
	const std::int32_t* const cnt_stm  = &raw.statkd[(int)stm][0][0][0];
	const std::int32_t* const cnt_nstm = &raw.statkd[(int)nstm][0][0][0];
	for (int i = 0; i < AuxCheap::SKD_PER_SIDE; ++i) {
		s_stm  += cnt_stm[i]  * W0[i] + cnt_nstm[i] * W1[i];
		s_nstm += cnt_nstm[i] * W0[i] + cnt_stm[i]  * W1[i];
	}
	out_stm = s_stm;
	out_nstm = s_nstm;
}
#else
static constexpr bool P2_HAS_AUXCHEAP = false;
#endif

#if defined(P2_ENDGAME_TEMPO)
static constexpr bool P2_HAS_ENDGAME_TEMPO = true;
// runtime layout: [absolute color][class][bucket]ç¸²ã»
static std::int32_t g_endgame_tempo_w[2][EndgameTempo::CLASS_NB][EndgameTempo::BUCKET_NB]{};
static bool g_endgame_tempo_loaded = false;
static inline bool no_endgame_tempo_env() {
	// ★NPS(A5): 関数内 static のガード変数ロードを消す。 既定値に固定 = bit-exact
#if defined(P2_DIAG)
	static const bool value = std::getenv("P2_NOEGT") != nullptr;
	return value;
#else
	return false;
#endif
}

static inline std::int32_t compute_endgame_tempo(const Position& pos, Color stm, int bucket) {
	const Color nstm = ~stm;
	const int us_class = EndgameTempo::compute_class(pos, stm);
	const int them_class = EndgameTempo::compute_class(pos, nstm);
	return (g_endgame_tempo_w[(int)stm][us_class][bucket]
	      - g_endgame_tempo_w[(int)nstm][them_class][bucket]) * EndgameTempo::OUTPUT_SCALE;
}
#else
static constexpr bool P2_HAS_ENDGAME_TEMPO = false;
#endif

// ---------------------------------------------------------------------------
// B6 可動域 (mobility) — 反対称スカラー 1 本。 2026-09-29。
// ---------------------------------------------------------------------------
// 定義と根拠は p2nnue_mob.h。 EGT とまったく同じ形 (スカラー x bucket 重み -> psqt_val)。
// 🚨重みは net の P2MOB セクションから読む。 セクションが無ければ 0 のままで
//   **eval は base と bit-exact**。 S1 (NPS 税) はこの状態で測る。
#if defined(P2_MOB)
static constexpr bool P2_HAS_MOB = true;
static std::int32_t g_mob_w[NUM_BUCKETS]{};
// 学習器の MOB_INPUT_SCALE = 16 = 1 << 4 と対応する。
static constexpr int MOB_INPUT_SHIFT = 4;
#else
static constexpr bool P2_HAS_MOB = false;
#endif

// StateInfo èï½´ç¸ºE® accumulator é©æ¦ãEç¸ºE¨ç¹§E¢ç¹ï½¼ç¹§E­è°ºE¡èãEã»è³Âé¾E´ç¹§åE½¿æE½¨E¼(position.h)ç¸²ã»
static_assert(StateInfo::P2_FT_DIM == FT_DIM, "position.h P2_FT_DIM != p2nnue FT_DIM");
// StateInfo::p2_psqt[2][P2_NUM_BUCKETS] ç¸ºE® bucket è¬¨E°ç¸ºE¨ NUM_BUCKETS ç¸ºE®è³Âé¾E´ç¹§åE½¿æE½¨E¼ç¸²ã»
// è³å ºE¸Âé¾E´ç¸ºE°ç¸ºE¨ get_layer_stack_index ç¸ºçï½¿æâE 0..NUM_BUCKETS-1 ç¸ºE®è­ï½¸ç¸ºå´ï½¾E¼ç¸ºE¿ç¸ºã»StateInfo ç¹§ã»
// ç¸ºE¯ç¸ºE¿èE½ºç¸ºå±â»éï½´è¢ç¿«âE¹§ã»P8H2=16 ç¸ºE§éï½ºéæºï¼ ç¸ºæ»E¿£è°ºE° crash ç¸ºE®èå²å±é«¦E²è±E½¢)ç¸²ã»
static_assert(StateInfo::P2_NUM_BUCKETS == NUM_BUCKETS, "position.h P2_NUM_BUCKETS != p2nnue NUM_BUCKETS");

// ===== LayerStack é©ï½¸è¬æ§­Îç¹ï½¼ç¹ãEç¹§E³ç¹ï½³ç¹ä»£ãE¹ï½«è­ãE¢Eºè³ã»è³æº¯E¡æ¢§å¾è»ã»E²èãç¹ï½­)=====
// è­ï½¢è³ã»Legacy): progress8kpabs ç¸ºE® 0..7 bucket ç¹§åµâç¸ºE®ç¸ºE¾ç¸ºE¾ LayerStack index ç¸ºE«è´E¿ç¸ºã»Âã»
// P2_LSMODE_P4H2: LayerStack ç¹§åµÂç¢EE²é¦æï½ºE¦4è¹Eºè»ã»EEEè¬ãEE§å¸ã¼2è¹Eºè»ã»= 8ç¸²éªãé©ï½¸ç¸ºE¶ç¸²ã»
//   stack = progress_bucket4 * 2 + hand_bucket2   (0 <= stack < 8)
//   progress_bucket4 = progress_bucket8 / 2       (0..3)
//   hand_bucket2     = (weighted_hand_total >= P2_HAND_VALUE_THRESHOLD[progress_bucket4]) ? 1 : 0
// ç¬ãEE¡ã»(é¨¾E²é¦æï½ºE¦è»E¥é«¢E¾èï½¤): é¨¾E²é¦æï½ºE¦ç¸ºE¨è¬ãEE§å¸ã¼ç¸ºE¯é¶E¸é«¢E¢(è åEä¹¢=è¬ãEE§è²ï½°ã»é¨ãä¹¢=è¬ãEE§è²ï½¤ã»ç¸ºå¶Eç¸ºæºï½ç¸²ã»
//   èï½¨é¨¾E²é¦æï½ºE¦èï½±é¨¾å£¹ã»é«¢E¾èï½¤ç¸ºE°ç¸ºE¨ (è åEä¹¢ç¹ï½»èãE(é¨ãä¹¢ç¹ï½»èEE stack ç¸ºå¾âç¸ºE¼é¨Eºç¸ºE«ç¸ºEªç¹§ä¹ÂãE¨é¨¾E²é¦æï½ºE¦ç¹§ã»
//   é¾EªéE½«ç¸ºE®è³E­èï½®èï½¤èå©E¿ä»£ãè»ã»ç¡ç¸ºå¶Eé¨¾E²é¦æï½ºE¦è»E¥é«¢E¾èï½¤ç¹§åæ²»éï½¨(2026-07-18 è¬¨åï½¸E«è»ã»E¸ã»MEEº¶ovel/base ç¸ºE§éï½ºè³ã»ç¸²ã»
// ç¸ºèãEç¹ï½¢ç¹ï½¼ç¹å³¨ãèE¦éåEE ç¸ºã»net ç¸ºE¯ arch è­ãEE­æãEç¸ºE« "LSMODE=P4H2:<t0>-<t1>-<t2>-<t3>," marker ç¹§åæç¸ºE¤ç¸²ã»
// é«¢E¾èï½¤é©æ¦ãEç¸ºæªçEç¸ºEªç¹§ã»net ç¸ºE¯è»E¥ç¹éªã£ç¹åæ¡¶ç¸ºã»load è­ãâEmarker ç¸ºE¨ç¹èÎç¹çE®å£½çç¹§å ¤ã»è·å»E è³å ºE¸Âé¾E´ç¸ºE¯è«¡è²å)ç¸²ã»
#if defined(P2_LSMODE_P4H2)
static constexpr bool P2_LS_P4H2 = true;
#else
static constexpr bool P2_LS_P4H2 = false;
#endif

// P2_LSMODE_P8H2: LayerStack ç¹§åµÂç¢EE²é¦æï½ºE¦8è¹Eºè»ã»EEEè¬ãEE§å¸ã¼2è¹Eºè»ã»= 16ç¸²éªãé©ï½¸ç¸ºE¶ç¸²ã»
//   stack = progress_bucket8 * 2 + hand_bucket2   (0 <= stack < 16)
//   progress_bucket8 = Legacy ç¸ºE¨è·å¾¡E¸Âç¸ºE®é¨¾E²é¦æï½ºE¦8è¹Eºè»ã»  (0..7)
//   hand_bucket2     = P4H2 ç¸ºE¨è·å¾Â§ç¸²ãæEèï½¤ index ç¸ºE¯ progress_bucket4 = progress_bucket8/2 (0..3)
//                      ç¸ºE§ P2_HAND_VALUE_THRESHOLD[4] ç¹§åµâç¸ºE®ç¸ºE¾ç¸ºE¾èæ¦èéï½¨ç¸ºå¶Eç¸²ã»
// P4H2 ç¸ºE¨é¡E°ç¸ºEªç¹§ä¼EE²é¦æï½ºE¦ç¹§ã»/2 ç¸ºE§è¨E§éµE®ç¸ºå±âç¸ºã»âE¹§ã»NUM_BUCKETS=16ç¸²E¯et ç¸ºE¯ "LSMODE=P8H2:<t0>-..." marker ç¹§åæç¸ºE¤ç¸²ã»
#if defined(P2_LSMODE_P8H2)
static constexpr bool P2_LS_P8H2 = true;
#else
static constexpr bool P2_LS_P8H2 = false;
#endif

// P4H2 ç¸ºE¨ P8H2 ç¸ºE¯è¬åE»ã»NUM_BUCKETS ç¸ºå¾ãEç¸ºE¤ç¸ºä¹ï½Eç¸²ãEEè­ãE®å¤ï½¾E©ç¸ºE¯ç¹èÎç¹å³¨ãç¹ï½©ç¹ï½¼ç¸ºE«ç¸ºå¶Eç¸²ã»
static_assert(!(P2_LS_P4H2 && P2_LS_P8H2), "P2_LSMODE_P4H2 and P2_LSMODE_P8H2 are mutually exclusive");

// P2_LSMODE_PROGREGIME (案 B、 2026-10-07): LayerStack を 進行度8 x regime2 = 16 本に割る。
//   stack = progress_bucket8 * 2 + regime        (0 <= stack < 16。 P8H2 と同じ interleaved)
//   regime = (progress_bucket8 >= P2_REGIME_MIN_PB) && (P2Volm::value(pos) >= P2_REGIME_V1_TH)
//   V1 = 両玉 8 近傍の**空升**への敵の利き数の和 (p2volm.h)。 本番で SPSA_RAZV_VMIN=3 として稼働中。
//   🚨P2_REGIME_MIN_PB で**序盤を割らない**: 実測で ply 1-40 の regime 1 は 0.8% しか無く、
//     全 bucket を割ると前半の regime 1 側に勾配が来ない (★threat_group_starved_b01 と同じ飢餓)。
//   net は arch に "LSMODE=PROGREGIME:v1<T>-pb<P>," marker を持つ。
#if defined(P2_LSMODE_PROGREGIME)
static constexpr bool P2_LS_PROGREGIME = true;
#else
static constexpr bool P2_LS_PROGREGIME = false;
#endif
#if !defined(P2_REGIME_V1_TH)
#define P2_REGIME_V1_TH 3
#endif
#if !defined(P2_REGIME_MIN_PB)
#define P2_REGIME_MIN_PB 4
#endif
static constexpr int REGIME_V1_TH  = P2_REGIME_V1_TH;
static constexpr int REGIME_MIN_PB = P2_REGIME_MIN_PB;
static_assert(!(P2_LS_PROGREGIME && (P2_LS_P4H2 || P2_LS_P8H2)),
              "P2_LSMODE_PROGREGIME is mutually exclusive with P4H2 / P8H2");
static_assert(!P2_LS_PROGREGIME || NUM_BUCKETS == 16,
              "PROGREGIME needs NUM_BUCKETS == 16");

// è¬ãEE§å¸ã¼2è¹Eºè»ã»ã»é©¥éªâ©ç¸ºE¨é«¢E¾èï½¤(ç¬ãEE­E¦éè²ãE bullet ç¸ºE¨è³æãEè³Âé¾E´ç¸ºè¼âç¹§å¥é«ªè³Âç¸ºE®é«®ã»E´ã»ãEç¸²ã»
// index ç¸ºE¯ PieceType(PAWN=1..GOLD=7)ç¸²ãã¾ç¸ºE¿: è±E½©=1,é¬¥ã»2,è­¯ã»2,é©«Â=3,é©¥ã»4,éãE6,é¬ãE7ç¸²ã»
// total = weighted(BLACK) + weighted(WHITE)ç¸²Eµotal >= é«¢E¾èï½¤ ç¸ºEªç¹§ã»hand_bucket=1(èãEç¸²ã»
// é«¢E¾èï½¤ç¸ºE¯ progress_bucket4(0..3) ç¸ºæâE(è­¯ã»)ç¸²ãEEbullet HAND_VALUE_THRESHOLDS ç¸ºE¨è³æãEè³Âé¾E´è ¢ã»E°å»Âã»
static constexpr int P2_HAND_VALUE_THRESHOLD[4] = {6, 12, 20, 24};
static constexpr int P2_HAND_PIECE_WEIGHT[PIECE_HAND_NB] = {
	0,       // NO_PIECE_TYPE(è­ï½ªè´E¿éï½¨)
	1,       // PAWN   è±E½©
	2,       // LANCE  é¬¥ã»
	2,       // KNIGHT è­¯ã»
	3,       // SILVER é©«Â
	6,       // BISHOP éãE
	7,       // ROOK   é¬ãE
	4,       // GOLD   é©¥ã»
};

static inline int padded(int in) { return ((in + 31) / 32) * 32; }

// ===== èåE½¸Âç¹èEãç¹ãEã¨ç¸ºE® FC è»E¤éï½¤ =====
// ★NPS: fc0_ws を 64B に揃える。
//   実測で std::vector<int8_t>(16384) の data() は 48 mod 64 に落ちる。
//   affine_par のロードは ws + ic*64 + g*32 なので、 g=0 側の 256 本が
//   毎 evaluate でキャッシュラインを跨いでいた (split load)。
//   不変条件: バイト列も添字も変えない = bit-exact。 確保場所が変わるだけ。
template <class T, std::size_t A = 64>
struct AlignedAlloc {
	using value_type = T;
	AlignedAlloc() = default;
	template <class U> AlignedAlloc(const AlignedAlloc<U, A>&) noexcept {}
	template <class U> struct rebind { using other = AlignedAlloc<U, A>; };
	T*   allocate(std::size_t n) {
		return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(A)));
	}
	void deallocate(T* p, std::size_t n) noexcept {
		(void) n;
		::operator delete(p, std::align_val_t(A));
	}
	template <class U> bool operator==(const AlignedAlloc<U, A>&) const noexcept { return true; }
	template <class U> bool operator!=(const AlignedAlloc<U, A>&) const noexcept { return false; }
};
using WsVec = std::vector<std::int8_t, AlignedAlloc<std::int8_t, 64>>;

struct Bucket {
#if defined(P2_NET_ALIGN64)
	// **案 6 (設計 fable 2026-11-06)**: `Net` は make_unique (16 B 整列) の中に bias が
	//   inline で入るので 64 B line を跨ぎうる。 跨ぐと評価 1 回ごとに余分な line を読む
	//   (★fc0_weights_were_misaligned: 48 mod 64 に落ちていて毎 evaluate 1 line 余分だった)。
	//   🚨値は変わらないので **bit-exact** (padding が増えるだけ)。
	alignas(64) std::int32_t fc0_bias[L1OUT_MAX];
#else
	std::int32_t fc0_bias[L1OUT_MAX];
#endif
	std::vector<std::int8_t> fc0_w;    // row-major [L1_OUT * FT_DIM]
#if defined(P2_FT_OUT16)
	// P2_FT_OUT16: fc0 の i16 入力版の重み [FC0_IN / 2][FC0_WS_ROWS][2] と m 倍の bias (affine16_pf_t)。
	std::vector<std::int16_t, AlignedAlloc<std::int16_t, 64>> fc0_ws16;
	alignas(64) std::int32_t fc0_bias_m[L1OUT_MAX];
#endif
	WsVec fc0_ws;                         // scrambled [FT_DIM/4][L1_OUT][4](AVX2è³E¦è»ç¤¼çE
#if defined(P2_NET_ALIGN64)
	// **案 6 (設計 fable 2026-11-06)**: `Net` は make_unique (16 B 整列) の中に bias が
	//   inline で入るので 64 B line を跨ぎうる。 跨ぐと評価 1 回ごとに余分な line を読む
	//   (★fc0_weights_were_misaligned: 48 mod 64 に落ちていて毎 evaluate 1 line 余分だった)。
	//   🚨値は変わらないので **bit-exact** (padding が増えるだけ)。
	alignas(64) std::int32_t fc1_bias[L3_MAX];
#else
	std::int32_t fc1_bias[L3_MAX];
#endif
	std::vector<std::int8_t> fc1_w;    // row-major [L3 * L2_PAD]
	WsVec fc1_ws;                         // scrambled [8][32][4]
#if defined(P2_L4)
	// 2026-11-07: **L4 = 4 層目 (`fc1b`)**。 fc1 (L2) と fc2 (Output) の間に挟む。
	//   目盛りは fc1 と**同一規約** (bias 127·QB / 重み QB / `>> 6` して clamp)。
#if defined(P2_NET_ALIGN64)
	alignas(64) std::int32_t fc1b_bias[L4_MAX];
#else
	std::int32_t fc1b_bias[L4_MAX];
#endif
	std::vector<std::int8_t> fc1b_w;   // row-major [L4 * pad32(L3)]
	WsVec fc1b_ws;                        // scrambled [pad32(L3)/4][L4][4]
#endif
#if defined(P2_NET_ALIGN64)
	// **案 6 (設計 fable 2026-11-06)**: `Net` は make_unique (16 B 整列) の中に bias が
	//   inline で入るので 64 B line を跨ぎうる。 跨ぐと評価 1 回ごとに余分な line を読む
	//   (★fc0_weights_were_misaligned: 48 mod 64 に落ちていて毎 evaluate 1 line 余分だった)。
	//   🚨値は変わらないので **bit-exact** (padding が増えるだけ)。
	alignas(64) std::int32_t fc2_bias[1];
#else
	std::int32_t fc2_bias[1];
#endif
	std::vector<std::int8_t> fc2_w;    // [1 * L2_PAD]
#if defined(P2_FC1_PREC)
	// P2_FC1_PREC: fc1 の i16 重み [L2_PAD / 2][L3][2] (affine16_t) と fc2 の i16 重み [L3] (dot_i16)。 読み込み時に fc1_w / fc2_w から作る。
	std::vector<std::int16_t, AlignedAlloc<std::int16_t, 64>> fc1_ws16;
	alignas(32) std::int16_t fc2_w16[L3_MAX] = {};
#endif
};

// ===== large page ç¹èEã£ç¹è¼ã =====
// FTé©¥éªâ©(~150MB)é²å³¨âç¸ºE®ç¹ï½©ç¹ï½³ç¹Âç¹ï£°é¦å¾ãç¹§E¯ç¹§E»ç¹§E¹ç¸ºE¯ 4KBç¹å£¹ã»ç¹§E¸ç¸ºE°ç¸ºE¨ STLB ç¸ºE®ç¹§E«ç¹èEã»é½ã»å³E~6MB)ç¹§ã»
// èï½§ç¸ºéªE¥éãEâ´ç¸²â¬E¡çï½ªE­ç¸ºE¿ç¸ºE®ç¸ºæºãEç¸ºE«ç¹å£¹ã»ç¹§E¸ç¹§E¦ç¹§E©ç¹ï½¼ç¹§E¯ç¸ºæªå±éæºâEç¹§ä¹ÂET ç¸ºE¨è·å¾Â§
// aligned_large_pages_alloc(SeLockMemoryPrivilege ç¸ºæªâç¸ºä»£Eç¸ºE°é¨¾å£¼E¸E¸ç¹å£¹ã»ç¹§E¸ç¸ºE¸é¾Eªèè¼ãµç¹§E©ç¹ï½¼ç¹ï½«ç¹èEã£ç¹§E¯)
// ç¸ºE§éï½ºè«æ´E ç¸ºE¦ TLB ç¹æºãç¹§è²çè²å¸âEç¹§ä¹ÂE·ector è åé¤ç¸ºE®è­ÂèE¸ãç¹ï½³ç¹§E¿ç¹è¼ãç¹ï½¼ç¹§E¹ç¸ºE®ç¸ºE¿è¬ èï½¾å¸Âã»
template <typename T>
class LPBuffer {
public:
	LPBuffer() = default;
	~LPBuffer() { release(); }
	LPBuffer(const LPBuffer&) = delete;
	LPBuffer& operator=(const LPBuffer&) = delete;

	void resize(std::size_t n) {
		release();
		ptr_ = static_cast<T*>(aligned_large_pages_alloc(n * sizeof(T)));
		if (ptr_ == nullptr) {
			sync_cout << "info string Error! P2NNUE: memory allocation failed ("
			          << n * sizeof(T) << " bytes)" << sync_endl;
			std::exit(EXIT_FAILURE);
		}
		size_ = n;
	}
	void assign(const T* src, std::size_t n) {
		resize(n);
		std::memcpy(ptr_, src, n * sizeof(T));
	}
	T&       operator[](std::size_t i)       { return ptr_[i]; }
	const T& operator[](std::size_t i) const { return ptr_[i]; }
	T*       data()       { return ptr_; }
	const T* data() const { return ptr_; }
	std::size_t size() const { return size_; }
	bool empty() const { return size_ == 0; }
	void clear() { release(); }   // 2026-11-05 (fable): i8 変換後に i16 表を手放す

private:
	void release() {
		if (ptr_) { aligned_large_pages_free(ptr_); ptr_ = nullptr; size_ = 0; }
	}
	T*          ptr_  = nullptr;
	std::size_t size_ = 0;
};

// ===== ç¹éªã£ç¹å»Î¡ç¹ï½¼ç¹§E¯èï½¨è´ã»=====
struct Net {
	std::vector<std::int16_t> ft_bias;    // [FT_DIM]
	LPBuffer<std::int16_t>    ft_weight;  // [73305 * FT_DIM]
#if defined(P2_FT_I8)
	// 2026-11-05 (fable) P2_FT_I8: FT 行を i8 で持ち、 i8 に収まらない値 (本番 net で 0.03%) だけ
	//   行ごとの補正列 (k, w - clamp8(w)) で持つ。 acc の更新は i16 wrapping 加算なので
	//   「i8 部分を足してから補正を足す」= 元の i16 を足すのと同じ値 (bit-exact)。
	//   行が 2KB -> 1KB になり、 DRAM 流量と L3 の作業集合が半分になる (acc 更新は memory-bound)。
	LPBuffer<std::int8_t>      ft_w8;          // [73305 * FT_ROW]
	std::vector<std::uint32_t> ft_corr_start;  // [73305 + 1] 行 f の補正は [start[f], start[f+1])
	std::vector<std::uint16_t> ft_corr_k;      // 補正の列 index
	std::vector<std::int16_t>  ft_corr_d;      // 補正値 (w - clamp8(w))
#endif
	std::int32_t psqt_bias[NUM_BUCKETS] = {};
	LPBuffer<std::int32_t>    psqt_weight; // [73305 * 8]
	bool has_psqt = false;
	Bucket buckets[NUM_BUCKETS];
	// pairshape: disk [bucket8][persp2][pair_type12][shape16] i16 = 3072
	LPBuffer<std::int16_t> pairshape_w;
	// dual_psqt(P2DUAL2): disk [persp2][head2][bucket8][sq81][piece14] i16 = 36288
	LPBuffer<std::int16_t> dual_w;
	// dual_cross(P2DUALZ, head0ç¸ºE®ç¸ºE¿): disk [persp2][bucket8][sq81][own_p14][target_p14] i16 = 254016
	LPBuffer<std::int16_t> dualcross_w;
	bool has_dual_psqt = false, has_dual_cross = false, has_pairshape = false;
	// Dual3(P2DUAL3): shared bucket-major king-zone and hand-context weights.
	LPBuffer<std::int16_t> dual3_w; // [8 * 110] = 880
	bool has_dual3 = false;
		// CROSS27(P2CR27): perspective-major full-width central-three-ranks reach-state weights.
		LPBuffer<std::int16_t> cross27_w; // [2 * 8 * 27 * 4] = 1728
		bool has_cross27 = false;
	// threat èï½¥èãEP2_ARCH_768T): éãEi8ç¸²âeature-major [THREAT_DIM * FT_DIM]ç¸²ã»
	// HalfKA FT ç¸ºE¨è·å¾ãç¹§E±ç¹ï½¼ç¹ï½«(QA=127)ç¸²E¢cc ç¸ºE¸ç¸ºE¯ i8ç«æ®µ16 é¨E¦è¿E·è«¡E¡è ï½µç¸ºE§èï£°éå±Âã»
	LPBuffer<std::int8_t> threat_weight;
	// threat i16 è¿ãEarch marker "ThreatI16=1")ç¸²Eª8 ç¸ºE§ç¸ºE¯ Eï½±127 clamp ç¸ºE§è¿åæè­ãã»è­Âè ï½·
	// é¼ã»E¨â½E¿E¡è¿E·(max~500)ç¸ºç¢E½£E½è¥å¾âEç¹§ä¹âç¹§ã»i16 è«æï½­å¥âèE¾è ¢æÂãE½½E¿ç¸ºã»ã»ç¸ºE¯ threat_is_i16=true ç¸ºE®ç¸ºE¨ç¸ºéªã»ç¸ºE¿ç¸²ã»
	// acc ç¸ºE¸ç¸ºE¯ i16 é¶E´èï£°éãEacc_add_i16)ç¸²Eª8/i16 ç¸ºE¯ç¸ºE©ç¸ºE¡ç¹§å³¨Â°è³Âè­E½¹ç¸ºE®ç¸ºE¿éï½ºè«æ´EE¹§å¾ï½ç¸²ã»
	LPBuffer<std::int16_t> threat_weight_i16;
	bool threat_is_i16 = false;
	bool has_threat = false;
	// 2026-09-12: threat 表の bucket 群化 (arch marker "ThreatGroups=<G>:<b0>-<b1>-...,")。
	//   threat_weight / threat_weight_i16 に群 0..G-1 の表 (各 THREAT_DIM x THREAT_ACC_DIM) を連結して持ち、
	//   局面の LayerStack bucket が属する群 threat_bucket_group[bucket] の表を使う。
	//   群化しない net は threat_groups = 1 (表 1 枚、 従来と同じ)。
	int threat_groups = 1;
	std::uint8_t threat_bucket_group[NUM_BUCKETS] = {};
	// profile 13 compact factor code, feature-major [THREAT_DIM][16].
	LPBuffer<std::int8_t> threat_factor_code;
	std::int32_t threat_factor_out[NUM_BUCKETS][THREAT_FACTOR_DIM] = {};
  // profile 13 inference projection: [bucket][threat feature].
  // Stored as i32 only after load-time exact-range validation.
  LPBuffer<std::int32_t> threat_factor_proj;
	bool has_threat_factor = false;
  // P2THRD1: bucket-major direct raw scalar weights [8][THREAT_DIM].
  // profile15 では読み込み時に dc-minor へ並べ替え、 末尾に Direct15::LOAD_PAD の
  // ゼロ余白を確保する (ホットループの範囲判定分岐を消すため)。
  LPBuffer<std::int32_t> threat_direct_w;
  // profile15 推論用の i16 複製。 |w| <= 32767 を load 時に確認できた場合のみ有効。
  // 合計は i32 で累算するので値は i32 版と bit-exact、 帯域だけ半減する。
  LPBuffer<std::int16_t> threat_direct_w16;
  bool threat_direct_w16_ok = false;
  bool has_threat_direct = false;
  // Direct15 テーブル形状。 net 形式 (P2THRD2 header の dims / buckets) から load 時に確定し、
  // ホットパスでは分岐しない (旧 net では従来定数と同じ値になり評価値は bit-exact)。
  //   direct_bucket_stride : 1 bucket 分の要素数 (旧 net = DIM、 defended net (P5) = 2*DIM)
  //   direct_nstm_offset   : nstm 視点テーブルの先頭オフセット (旧 net = 0 = stm と共有、
  //                          stm/nstm 別テーブル net (P2) = NUM_BUCKETS * stride)
  //   direct_defended      : P5 net (被攻撃駒が守られているかで次元 x2、 P2THRD2 dims = 2*DIM)
  std::size_t direct_bucket_stride = 0;
  std::size_t direct_nstm_offset = 0;
  bool direct_defended = false;
  // net の arch 文字列に含まれる学習時 fv_scale (例 "fv_scale=14")。
  // learner は fv_scale = (QA*QB + scale/2)/scale の固定値を書き込む。
  // 0 = marker を持たない旧 net。
  int train_fv_scale = 0;
  // Direct15 head の raw 単位 cap。 学習時と同一目盛りにするため
  // P2_THREAT_DIRECT_CAP_CP * train_fv_scale で確定させる (実行時 FV_SCALE に非依存)。
  // 0 = 旧 net フォールバック (cap_cp * effective_fv という旧挙動)。
  std::int32_t direct_cap_raw = 0;
  // The direct branch may use an i32 accumulator only when the loaded
  // quantised weights are bounded below the exact board-combinatoric limit.
  bool threat_direct_i32_safe = false;	// aux èï½¥èãEP2_AUX): éãEi8ç¸²âeature-major [AUX_DIM * FT_DIM]ç¸²ã»alfKA FT ç¸ºE¨è·å¾ãç¹§E±ç¹ï½¼ç¹ï½«(QA=127)ç¸²ã»
	// acc ç¸ºE¸ç¸ºE¯ i8ç«æ®µ16 é¨E¦è¿E·è«¡E¡è ï½µç¸ºE§èï£°éãEthreat ç¸ºE¨è·å¾Â§ acc_add_threat_row ç¹§åï½µâ«çEç¸²ã»
	LPBuffer<std::int8_t> aux_weight;
	bool has_aux = false;
	bool loaded = false;
};

static std::unique_ptr<Net> g_net;
static LPBuffer<float>       g_progress; // [125388] (f64ç«æ­32)

// é¦å¾ãç¹§E¯ç¹§E»ç¹§Eµ: è¿ï½¹è ï½´é©¥ã»f ç¸ºE® FTé©¥éªâ©é¦å¾âE PSQTé¦å¾Âã»
// 2026-11-04: **FT 行の間隔を 4K ページに揃えるためのバディング**
//   l0w の行 i の先頭 offset = (2*FT_DIM*i) mod 4096。 行が 4K を跨ぐと
//   Intel の L2 ストリーマが境界で止まり TLB miss + prefetch 再始動が乗る。
//   跨ぎ率: 512 0%% / 768 25%% / **1024 0%%** / 1280 50%% / **1536 50%%**
//   -> 1536 の行を 2048 (4096 B) に広げると跨ぎが 0%% になる。
//   読む値は変わらず余りは 0 なので **評価は bit-exact**。
//   既定 (P2_FT_ROW_PAD 未定義) は FT_DIM = 従来と完全に同じ。
#if defined(P2_FT_ROW_PAD)
static constexpr std::size_t FT_ROW = (std::size_t)(P2_FT_ROW_PAD);
static_assert(P2_FT_ROW_PAD >= FT_DIM, "P2_FT_ROW_PAD must be >= FT_DIM");
#else
static constexpr std::size_t FT_ROW = (std::size_t)FT_DIM;
#endif
static inline const std::int16_t* ft_row(const Net& net, std::size_t f) {
	return &net.ft_weight[f * FT_ROW];
}
// ---------------------------------------------------------------------------
// P2_FT_I32 (2026-11-18 / runs/prereg/PREREG-20261118-FTI32.md): FT の重み・bias を k = 2^FT_KS 倍細かい刻み
//   (round(254·k·w) の i16) で持ち、 累積器 (StateInfo::p2_acc と Finny cache) を i32 にする。
//   transform は clamp [0, 254·k] → floor(lo·hi / (512·k²)) の u8 (0..126) = fc0 以降は従来のまま。
//   SIMD では mulhi(lo << (7 − 2·FT_KS), hi) で従来と同じ命令数 (lo << (7 − 2 FT_KS) <= 254·k·2^(7−2 FT_KS) <= 32512)。
//   threat 表 (i8・254 目盛り) と KH (i16・254 目盛り) は今の目盛りの別の累積器のまま、 transform で << FT_KS して足す。
//   旧形式の net (刻み 254) は読み込み時に重み・bias を k 倍する = 評価値は従来の exe と同じ (i16 の wrap が起きない限り)。
//   新形式の net は arch の FT 幅の前に "I32K<k>:" を置く ("[73305->I32K8:1024x2]")。 旧 exe は FT 幅が 0 と読まれて
//   "arch dims parse failed" で止まる (取り違えの関門)。 版番号は 0x7AF32F21。
// ---------------------------------------------------------------------------
// 11-18 改: define を 3 つに分けた。
//   -DP2_FT_KS=<log2 k>    FT の刻み k (1..3)。 0 / 未定義なら従来の 254 刻み。
//   -DP2_FT_OUT16=<log2 m> transform の出力を floor(lo·hi·m / (512 k²)) の i16 (0..126m) にし、 fc0 を i16 入力 (madd_epi16) にする。
//                          fc0 の出力は floor((m·bias + Σ x·w) / m) = (m·bias + Σ) >> log2 m (PREREG の (B))。
//   -DP2_FT_I32            累積器 (StateInfo::p2_acc / Finny cache) を i32 にする。 無ければ i16 のまま (wrapping 加算。
//                          値が正しいのは最終の和 (+ threat + KH) が i16 に入るときだけ = AccRange の計測で確かめた k に限る)。
//   既定の組 (S0 / 11-18 メイン指示) = KS 3 (k = 8) / OUT16 5 (m = 32)。
#if defined(P2_FT_KS)
static_assert(P2_FT_KS >= 0 && P2_FT_KS <= 3, "P2_FT_KS は log2(k) = 0..3");
static constexpr int FT_KS = P2_FT_KS;
#else
static constexpr int FT_KS = 0;
#endif
#if defined(P2_FT_OUT16)
static_assert(P2_FT_OUT16 >= 0 && P2_FT_OUT16 <= 7, "P2_FT_OUT16 は log2(m) = 0..7");
static constexpr int FT_MS = P2_FT_OUT16;
using TrT = std::int16_t;
#else
static constexpr int FT_MS = 0;
using TrT = std::uint8_t;
#endif
#if defined(P2_FT_I32)
using FtAcc = std::int32_t;
#else
using FtAcc = std::int16_t;
#endif
#if defined(P2_FT_I32) || defined(P2_FT_KS) || defined(P2_FT_OUT16)
#define P2_FT_NEWTR 1
#if defined(P2_FT_I8) || defined(P2_AUX) || defined(P2_THREAT_PROJ) || defined(P2_RESIDUAL_MIX16) || defined(P2_RESIDUAL_MIXWIDE16) \
    || defined(P2_HAND_DENSE) || defined(P2_SPARSE_FC0)
#error "P2_FT_I32 / P2_FT_KS / P2_FT_OUT16 は P2_FT_I8 / P2_AUX / P2_THREAT_PROJ / MIX16 / HAND_DENSE / SPARSE_FC0 と併用しない"
#endif
#endif
static constexpr int FT_K = 1 << FT_KS;
static constexpr int FT_CLAMP = 254 * FT_K;
// transform の右 shift: out = (lo·hi) >> FT_TRS (FT_TRS >= 2)。
static constexpr int FT_TRS = 9 + 2 * FT_KS - FT_MS;
// SIMD の mulhi(lo << FT_SA, hi << FT_SB) = (lo·hi·2^(FT_SA+FT_SB)) >> 16 = (lo·hi) >> FT_TRS。
//   lo, hi <= 254k なので lo << FT_SA <= 254k·2^(7−FT_KS) <= 32512 (FT_SB も同じ上限) = i16 に収まる。
static constexpr int FT_SAB = 16 - FT_TRS;
static constexpr int FT_SA = FT_SAB < 7 - FT_KS ? FT_SAB : 7 - FT_KS;
static constexpr int FT_SB = FT_SAB - FT_SA;
static_assert(FT_SAB >= 0 && FT_SB >= 0 && FT_SB <= 7 - FT_KS, "transform の shift の組が i16 に収まらない");
static_assert(126 * (1 << FT_MS) <= 32767, "OUT16 の出力が i16 に収まらない");
static constexpr std::uint32_t P2_NET_VERSION_FTI32 = 0x7AF32F21u;
// 累積器の値 a と threat 側の値 t (254 目盛りの i16) の和。 i16 累積器では従来どおり i16 wrapping、 i32 では t を k 倍して足す。
static inline int ft_add_thr(FtAcc a, int t) {
#if defined(P2_FT_I32)
	return (int) a + t * FT_K;
#else
	return (std::int16_t) (a + t * FT_K);
#endif
}
// scalar の transform の 1 列: x = clamp(lo), y = clamp(hi) から出力を作る (SIMD と同じ値)。
static inline TrT ft_pair_out(int x, int y) { return (TrT) ((x * y) >> FT_TRS); }
#if defined(P2_FT_I8)
static constexpr std::size_t FT_ROW_BYTES = FT_ROW * sizeof(std::int8_t);
static inline const std::int8_t* ft_row8(const Net& net, std::size_t f) {
	return &net.ft_w8[f * FT_ROW];
}
static inline const void* ft_row_bytes(const Net& net, std::size_t f) { return ft_row8(net, f); }
// 行 f の補正を acc に sign (+1/-1) で足す。 補正の無い行 (85%) は比較 1 回で抜ける。
static inline void ft_corr_apply(const Net& net, std::size_t f, std::int16_t* acc, int sign) {
	const std::uint32_t b = net.ft_corr_start[f], e = net.ft_corr_start[f + 1];
	for (std::uint32_t i = b; i < e; ++i)
		acc[net.ft_corr_k[i]] = (std::int16_t)(acc[net.ft_corr_k[i]] + sign * (int)net.ft_corr_d[i]);
}
// i16 表 -> i8 表 + 補正列。 変換後に i16 表 (150MB) は手放す。
static void build_ft_i8(Net& net) {
	const std::size_t rows = (std::size_t)HALFKA_DIM;
	net.ft_w8.resize(rows * FT_ROW);
	std::memset(net.ft_w8.data(), 0, rows * FT_ROW);
	net.ft_corr_start.assign(rows + 1, 0);
	net.ft_corr_k.clear(); net.ft_corr_d.clear();
	std::size_t rows_with_corr = 0;
	for (std::size_t f = 0; f < rows; ++f) {
		const std::int16_t* w = ft_row(net, f);
		std::int8_t* w8 = &net.ft_w8[f * FT_ROW];
		const std::size_t before = net.ft_corr_k.size();
		for (int k = 0; k < FT_DIM; ++k) {
			const int v = (int)w[k];
			const int c = v < -128 ? -128 : (v > 127 ? 127 : v);
			w8[k] = (std::int8_t)c;
			if (c != v) { net.ft_corr_k.push_back((std::uint16_t)k); net.ft_corr_d.push_back((std::int16_t)(v - c)); }
		}
		net.ft_corr_start[f + 1] = (std::uint32_t)net.ft_corr_k.size();
		if (net.ft_corr_k.size() != before) ++rows_with_corr;
	}
	net.ft_weight.clear();
	sync_cout << "info string P2NNUE FT i8 rows: " << rows << " rows, corrections " << net.ft_corr_k.size()
	          << " in " << rows_with_corr << " rows" << sync_endl;
}
#else
static constexpr std::size_t FT_ROW_BYTES = FT_ROW * sizeof(std::int16_t);
static inline const void* ft_row_bytes(const Net& net, std::size_t f) { return ft_row(net, f); }
#endif
static inline const std::int32_t* psqt_row(const Net& net, std::size_t f) {
	return &net.psqt_weight[f * (std::size_t)NUM_BUCKETS];
}
// threat è¿ï½¹è ï½´é©¥ã»tf(0..THREAT_DIM)ç¸ºE® FT é©¥éªâ©é¦ã»i8ç¸²ET_DIM èãEç¸²ã»
static inline const std::int8_t* threat_row(const Net& net, std::size_t tf) {
	return &net.threat_weight[tf * (std::size_t)THREAT_ACC_DIM];
}
// threat è¿ï½¹è ï½´é©¥ã»tf ç¸ºE® FT é©¥éªâ©é¦ã»i16ç¸²ET_DIM èãEç¸²Eµhreat_is_i16 net éï½¨ç¸²ã»
static inline const std::int16_t* threat_row_i16(const Net& net, std::size_t tf) {
	return &net.threat_weight_i16[tf * (std::size_t)THREAT_ACC_DIM];
}
static inline const std::int8_t* threat_factor_row(const Net& net, std::size_t tf) {
	return &net.threat_factor_code[tf * (std::size_t)THREAT_FACTOR_DIM];
}
// aux local index af(0..AUX_DIM)ç¸ºE® FT é©¥éªâ©é¦ã»i8ç¸²ET_DIM èãEç¸²ã»
static inline const std::int8_t* aux_row(const Net& net, std::size_t af) {
	return &net.aux_weight[af * (std::size_t)FT_DIM];
}
// threat index LUT(pair_base / from_offset / attack_order)ç¸²E­oad è­ãâEbuildç¸²ã»
static Threat::Tables g_threat_tables;
static Direct15::Tables g_direct15_tables;

// Direct15: クラス別の空盤面利き [color][class][from]。 25.9KB。
//  (a) 非飛び駒 (class 0,2,3,4,9) は空盤面利き == 実利きなので effects_from() を呼ばずに済む。
//  (b) 飛び駒は 実利き ⊆ 空盤面利き なので、 targets と交差しなければ利き計算ごと省略できる。
// どちらも集合として厳密に等価な置き換えなので評価値は bit-exact。
static Bitboard g_direct15_pseudo[COLOR_NB][Direct15::CLASS_NB][SQ_NB];
static bool g_direct15_pseudo_built = false;
// 飛び駒クラスの実利き。 effects_from() は別 TU の非 inline 関数かつ 32 分岐 switch
// なので、 クラス直の 5 分岐に置き換えて呼び出し段数を 1 つ減らす。 値は同一。
static inline Bitboard direct15_slider_effect(int ac, Color c, Square from, const Bitboard& occ) {
	switch (ac) {
	case 1:  return c == BLACK ? lanceEffect<BLACK>(from, occ) : lanceEffect<WHITE>(from, occ);
	case 5:  return bishopEffect(from, occ);
	case 6:  return rookEffect(from, occ);
	case 7:  return horseEffect(from, occ);
	default: return dragonEffect(from, occ);   // ac == 8
	}
}
static void build_direct15_pseudo() {
	if (g_direct15_pseudo_built) return;
	// クラス代表駒 (クラス 4 は金/成駒すべて goldEffect なので GOLD で代表できる)。
	static const PieceType rep[Direct15::CLASS_NB] = {
		PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK, HORSE, DRAGON, KING
	};
	for (int c = 0; c < COLOR_NB; ++c)
		for (int ac = 0; ac < Direct15::CLASS_NB; ++ac)
			for (int s = 0; s < SQ_NB; ++s)
				g_direct15_pseudo[c][ac][s] =
					effects_from(make_piece((Color)c, rep[ac]), (Square)s);
	g_direct15_pseudo_built = true;
}
// normalize_sq(persp+mirror ç¸ºE®è»ã»E²åE½®èï½¡ã»ç¸ºE® LUT è¹ã» [persp][mirror][sq] ç«ãEè±E½£éä¸å§ç¹æ§­ãç¸²ã»
// threat è»ç²å«Edelta ç¸ºE®è­Âèã»æç¸ºE§èå¤å±ç¸ºå¶Eç¸ºæºï½EL1 lookup ç¸ºE«éï½®è¬ ã»èï½¤ç¸ºE¯ normalize_sq ç¸ºE¨è·å¾¡E¸Â)ç¸²ã»
static std::uint8_t g_norm_sq[2][2][81];
static void build_norm_sq() {
	for (int p = 0; p < 2; ++p)
		for (int m = 0; m < 2; ++m)
			for (int s = 0; s < 81; ++s)
				g_norm_sq[p][m][s] = (std::uint8_t)Threat::normalize_sq(s, (Color)p, m != 0);
}
static bool                  g_progress_loaded = false;

// USI ç¹§Eªç¹å±ãç¹ï½§ç¹ï½³èï½¤
// 駒割の値付け補正 (2026-09-29)。
// 🚨**600 局で +0.08pt [−3.50, +3.67] の null** だったので既定では**コンパイルしない**。
//   値付けの差自体は実在する (歩 +1.97cp / 飛車 +21.6cp、 床は 0 を含む) が、
//   NAGISA に合わせても勝率は動かなかった。 再検証したくなったら -DP2_MATPRICE で復活する。
//   測定の記録: memory/material_pricing_gap_20260929.md
// 単位: raw の 1/1000 / 駒割 1 単位 (Eval::PieceValue の目盛り、 歩 = 90)。 実測値 612。
#if defined(P2_MATPRICE)
static int g_mat_price_adj = 0;
#endif
static int g_fv_scale  = 28;      // è³E­é¶E¤èï½¥é«¯ã»progress >= PROG_HI)ç¸ºE® FV_SCALE
static int g_fv_scale_opening = 26; /* 11-16 既定を対局時の値へ (旧 0) */ // è åEä¹¢(progress <= PROG_LO)ç¸ºE® FV_SCALE
// èæEåª FV_SCALE é¬æºä¿£ç¸ºE® progress é«¢E¾èï½¤ç¸²ãEEè´ã»= progress sum(p2_prog_sum)ç¸ºE® 1/100ç¸²ã»
//   progress sum ç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½« ç«ãE-2.0ç¸²ã»2.0 ç¸ºE® log-odds(bucket 0=è åEä¹¢/è´å¼±Eç¸²ã»=é¨ãä¹¢/é¬®å¥EEç¸²ã»
//   è­ï½¢è³ã» PROG_LO=-110(ç«ãE1.10, bucket1è EEéèå©E¿ã»è åEä¹¢)ç¸²å¼ROG_HI=0(bucket3è EEéEè³E­é¶E¤èï½¥è¿E£)ç¸²ã»
static int g_fv_prog_lo = -110;   // ç¸ºèï½è­ï½ªè²E(progress sum < -1.10)ç¸ºE¯è³æãEç¸ºE« opening èï½´
static int g_fv_prog_hi = 0;      // ç¸ºèï½éã»progress sum > 0.0)ç¸ºE¯è³æãEç¸ºE« FV_SCALE(è³E­é¶E¤)èï½´
// ç¬ãEç¹§E¢ç¹ï½³ç¹§E«ç¹ï½¼: é¨ãä¹¢ç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½«ç¸²ã»=è¾ï½¡èï½¹(è³E­é¶E¤èï½¤ç¸ºE«éï½½è ãEè æºæEç¹§E¢ç¹ï½³ç¹§E«ç¹ï½¼, bit-exact)ç¸²ã»0 ç¸ºE§è³E­é¶E¤ç«å ¤Eµãä¹¢ç¸ºE¸ revertç¸²ã»
static int g_fv_scale_endgame = 28;   /* 11-16 既定を対局時の値へ (旧 0) */ // é¨ãä¹¢(progress >= PROG_END ç¸ºE¾ç¸ºæºãE root eval >= ENDGAME_EVAL)ç¸ºE® FV_SCALE
static int g_fv_prog_end = 130;       // ç¸ºèï½éã»progress sum > 1.30)ç¸ºE¯é¨ãä¹¢èï½´ç¸ºE¸(midç«å¼¾nd é¬æºä¿£ç¸ºE®è³é¡ï½«E¯)
static int g_fv_endgame_eval = 32000; // root eval ç¸ºå¾ï¼E¸ºE®èï½¤èï½¥è³ç¿«âç¹§ã»èE½ªèï½¢=é¨ãä¹¢è¬E½±ç¸ºã»é¨ãä¹¢ç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½«ç¸²ãéè³ã»2000=è¾ï½¡èï½¹ç¸²ã»
// ç¬ãEE¡E¨éï½ºèEçç¹ï½ªç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½«: èæEåª FV ç¸ºE§ç¸ºE¯ phase ç¸ºE«ç¹§å»Eèã»ÎEcp ç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½«ç¸ºæï½¤å³¨Eç¹§ã»FV28è³E­é¶E¤=FV14è åEä¹¢ç¸ºE®èéã»)ç¸²ã»
//   èE½ºèå¸âEç¹§ã»score cp ç¸ºE®ç¸ºE¿èï½ºè²EEFV(g_fv_display)ç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½«ç¸ºE«è¬ å¸·E®å±E ç¸ºE¦é¨E±è³Âé¦E¨éï½ºç¸ºå¶Eç¸²ã»
//   0 = OFF(è æ¢§å©¿è åé¤)ç¸²ãç¾éï½¢/resign/mate é²å³¨ã»èã»Îç¹ï½­ç¹§E¸ç¹ãEãç¸ºE¯è³Âè»ã»E¤å³¨â´ç¸ºEªç¸ºã»Âã»
static int g_fv_display = 28;  // 11-16 既定を対局時の値へ (旧 0)
// ç¬ãEç¾éï½¢è³E­ç¸ºE® eff_fv ç¸ºE¯ç¸²å¾Îç¹ï½¼ç¹äº¥E±Âé«±E¢ç¸ºE® progressç¸²éªãEèæ§­â¡ç¸ºéï½¢Eºè³å£¹E ç¸²âµç¾éï½¢è­ï½¨ç¸ºE®èï½¨ç¹å¼±ã»ç¹å³¨ãèEºè³å£¹âE¹§ä¹Âã»
//   (ç¹ï½ªç¹ï½¼ç¹è¼ï¼E¸ºE¨ç¸ºE« eff_fv ç¸ºæï½¤å³¨Eç¹§ä¹âE eval ç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½«ç¸ºå¾ã®ç¹ï½¼ç¹è¾ä¿£ç¸ºE§è³å ºE¸Âé¾E´ç¸ºE«ç¸ºEªç¹§ã»alpha-beta ç¸ºæï½£ç¿«Eç¹§ä¹âç¹§ã»
//   go é«¢å¥E§åºå¾ç¸ºE« p2_search_begin() ç¸ºã»g_root_eff_fv ç¹§å®ï½¨E­è³å£¹E  g_root_fv_valid=true ç¸ºE«ç¸ºå¶Eç¸²ã»
//   è¬ï½¢éï½¢è³E­ç¸ºE® evaluate()(dump=false)ç¸ºE¯ç¸ºèãEèï½¤ç¹§åE½½E¿ç¸ºã»ÂãEEè¿¢E¬ eval ç¹§E³ç¹æ§­Î¦ç¹ãEdump=true)ç¸ºE¯è¿´E¾è»Âé«±E¢ç¸ºä¹ï½èå´ï½¨è²E®å±Âã»
static int g_root_eff_fv = 28;

// ---- capture futility margin を進行度で変える (2026-10-11 ユーザー提案 / 凍結前の宿題 1)
//   SPSA_CAPTURE_FUTILITY_BASE = 中盤・終盤の値、 SPSA_CAPFUT_OPENING = 序盤 (progress <= PROG_LO) の値。
//   補間は FV_SCALE / LMR と同じ progress_t() を共有する (FV_SCALE_PROG_LO/HI が区間)。
//   margin が大きい = futilityValue が alpha を上回りやすい = capture を刈らない = 広く読む。
//   既定 0 = 無効 (本体 SPSA_CAPTURE_FUTILITY_BASE に追従) で従来と bit-exact。
static int g_capfut_opening = 0;

// ---- 振り飛車の様式項 (2026-10-05)。 既定 0 = 完全に無効 (本番安全)
//   🎯選好 (振るかどうか) をエンジン側に持たせ、 net には嘘のないデータだけを入れる。
//     net のラベルに下駄を履かせる案は (a) 基データ 98% に 50:1 で希釈される
//     (b) 片手番だけに足すと minimax に ±G/2 の手番ノイズが入る
//     (c) net は手数を入力に持たない — の 3 点で壊れる。
//   🚨FV_SCALE 系が死んだのは「正の定数倍は手順序を変えない」から。
//     これは**形に依存する加算項**なので形の境界でだけ手順序を変える = 意図した効き方。
static int g_furi_bonus = 0;        // G [cp]。 0=無効
static int g_furi_fade_start = 30;  // この手数までは係数 1.0
static int g_furi_fade_end = 60;    // この手数以降は 0 (= 中盤以降は完全に正直)

// 振り飛車の様式項を cp で返す (**先手視点**)。
//   SAFETY: 読むのは盤面と手数だけ。 副作用なし。
//   不変条件: 先後を入れ替えると符号が反転する (反対称) ので minimax が壊れない。
//             相振り / 双方居飛車は ib == iw で 0 になる。
static inline int furibisha_bonus_black_view(const Position& pos) {
	if (g_furi_bonus == 0)
		return 0;
	const int ply = pos.game_ply();
	if (ply >= g_furi_fade_end)
		return 0;
	int num = 1, den = 1;
	if (ply > g_furi_fade_start) {
		den = g_furi_fade_end - g_furi_fade_start;
		if (den <= 0)
			return 0;
		num = g_furi_fade_end - ply;
	}
	// 振り飛車 = その色の飛車が内側の筋 (先手 5-8 筋 / 後手 2-5 筋) にいる
	int ib = 0, iw = 0;
	Bitboard rb = pos.pieces(BLACK, ROOK);
	while (rb) {
		const File f = file_of(rb.pop());
		if (f >= FILE_5 && f <= FILE_8) { ib = 1; break; }
	}
	Bitboard rw = pos.pieces(WHITE, ROOK);
	while (rw) {
		const File f = file_of(rw.pop());
		if (f >= FILE_2 && f <= FILE_5) { iw = 1; break; }
	}
	if (ib == iw)
		return 0;
	return (ib - iw) * g_furi_bonus * num / den;
}

// ---- 右玉の様式項 (2026-10-05)。 既定 0 = 完全に無効 (本番安全)
//   🎯右玉は**居飛車の中の作戦**なので、 戦型ごと変える振り飛車より税が小さいはず。
//     実戦上位は 27-29% 指すのに我々は 7% (★migigyoku_natural_choice)。
static int g_migi_bonus = 0;        // G [cp]。 0=無効
static int g_migi_fade_start = 30;  // この手数までは係数 1.0
static int g_migi_fade_end = 70;    // この手数以降は 0 (右玉は振り飛車より形になるのが遅い)
// 🚨🚨2026-10-20: MovePicker の右玉誘導の**序盤の下限**。 既定 0 = 下限なし。
//   fade は「FadeStart まで満額 / FadeEnd で 0」の三角形なので、 FadeStart を
//   60 に伸ばすと **ply 1〜60 が満額**になり、 序盤から玉を右へ押してしまう
//   (実測: 分岐 ply 中央 **4** / 右玉率が 4.0% -> 0.0% に下がった)。
//   -> この下限より前の ply では weight を 0 にする。 定跡の出口は 27 局の実測で
//      中央 **23** / 四分位 8-78 なので、 30 前後が「定跡を抜けた後」に当たる。
static int g_migi_mo_ply_min = 0;
static int g_migi_kakugawari = 1;   // 1 = 角換わり (双方が角を手駒に持つ) のときだけ効かせる
// ---- 先後で別の G を持てるようにする (2026-10-18 ユーザー指示「c」)
//   🚨🚨floodgate 51,776 局を**局単位・engine 別**で測ると、 右玉の値段は**先後で符号が割れる**:
//     先手の右玉  +2 〜 **+19.7 pt** (engine 別にほぼ全部プラス)
//     後手の右玉  -0.5 〜 **-38.7 pt** (engine 別にほぼ全部マイナス)
//   従来の `(ib - iw) * G` は**反対称**なので、 損な後手側も同じだけ押していた。
//   🚨ただしユーザー情報「角換わり後手右玉は今まさに定跡を掘っている人多い」= 上の数字は
//     定跡が未整備だった時期の棋譜を含むので、 後手を**切る**のではなく**別の値にできる**
//     形にする (★migigyoku_natural_choice の狙いは元々「角換わり後手右玉」)。
//   既定 **-9999 = 先手と同値** なので従来と **bit-exact**。
static int g_migi_bonus_white = -9999;
static int g_aiga_bonus_white = -9999;

/// 後手の右玉に乗せる G を返す。 番兵 -9999 のときは先手と同じ値 (= 従来の反対称)。
static inline int migi_gw(int sente, int white) {
	return (white == -9999) ? sente : white;
}

// ---- 相掛かり角交換右玉の下駄 (2026-10-13 ユーザー提案)。 既定 0 = 完全に無効。
//   floodgate で「先手相掛かり → 角交換 → 右玉」が出て、 上位陣もあまり指さない形なので
//   独創賞の軸として伸ばす ([[differentiation_strategy]])。
//   右玉の下駄 (MigigyokuBonus) との違いは **相掛かりの条件を足した**こと:
//     角交換 (双方が角を手駒) に加えて、 **2 筋と 8 筋の歩が盤上から消えている**こと。
//     = 双方が飛車先の歩を交換した痕跡 = 相掛かり。 案B (ユーザー決定 2026-10-13)。
//   🚨MigigyokuBonus と併用すると**二重に下駄を履く** (条件を両方満たす局面がある)。
//     切り分けて測るときは片方を 0 にすること。
static int g_aiga_bonus = 0;        // G [cp]。 0=無効
static int g_aiga_fade_start = 30;  // この手数までは係数 1.0
static int g_aiga_fade_end = 70;    // この手数以降は 0

// 相掛かり角交換右玉の下駄を cp で返す (**先手視点**)。
//   SAFETY: 読むのは盤面・手駒・手数だけ。 Position を変更しない。
//   不変条件: 先後を入れ替えると符号が反転する (対称)。
static inline int aigakari_migi_bonus_black_view(const Position& pos) {
	// 🚨先後で別の G を持つので「両方 0 のとき」だけ抜ける (片方だけ非 0 の設定を潰さない)
	if (g_aiga_bonus == 0 && migi_gw(g_aiga_bonus, g_aiga_bonus_white) == 0)
		return 0;
	const int ply = pos.game_ply();
	if (ply >= g_aiga_fade_end)
		return 0;
	int num = 1, den = 1;
	if (ply > g_aiga_fade_start) {
		den = g_aiga_fade_end - g_aiga_fade_start;
		if (den <= 0)
			return 0;
		num = g_aiga_fade_end - ply;
	}
	// ①角交換: 双方が角を手駒に持つ
	if (!(hand_count(pos.hand_of(BLACK), BISHOP) > 0
	   && hand_count(pos.hand_of(WHITE), BISHOP) > 0))
		return 0;
	// ②相掛かり: **双方が飛車先の歩を伸ばしている**こと。 案D (2026-10-13 ユーザー決定)。
	//   🚨最初は「2 筋と 8 筋に歩が 1 枚も無い」(歩交換済み) にしたが、 判定の棋譜 7,200 局で
	//     **21 局 (2.4%) しか無かった**。 右玉になる時点では歩交換は普通まだ済んでいない。
	//     伸ばしている形なら 367 局 (41%) あり、 角換わり (飛車先を伸ばさない) と区別できる。
	//   先手の 2 筋の歩が 2六以前 (RANK_6 以上手前) かつ 後手の 8 筋の歩が 8四以降。
	//   🚨rank は RANK_1 が先手の対面。 先手の歩が進む = rank が小さくなる。
	{
		bool adv_b = false, adv_w = false;
		Bitboard pb = pos.pieces(BLACK, PAWN);
		while (pb) {
			const Square sq = pb.pop();
			if (file_of(sq) == FILE_2 && rank_of(sq) <= RANK_6) { adv_b = true; break; }
		}
		Bitboard pw2 = pos.pieces(WHITE, PAWN);
		while (pw2) {
			const Square sq = pw2.pop();
			if (file_of(sq) == FILE_8 && rank_of(sq) >= RANK_4) { adv_w = true; break; }
		}
		if (!(adv_b && adv_w))
			return 0;
	}
	// ③片方だけが右玉 (玉と飛車が同じ側。 MigigyokuBonus と同じ閾値)
	int ib = 0, iw = 0;
	if (file_of(pos.square<KING>(BLACK)) <= FILE_4) {
		Bitboard rb = pos.pieces(BLACK, ROOK);
		while (rb) {
			if (file_of(rb.pop()) <= FILE_4) { ib = 1; break; }
		}
	}
	if (file_of(pos.square<KING>(WHITE)) >= FILE_6) {
		Bitboard rw = pos.pieces(WHITE, ROOK);
		while (rw) {
			if (file_of(rw.pop()) >= FILE_6) { iw = 1; break; }
		}
	}
	if (ib == iw)
		return 0;
	// 先手の右玉なら +g_aiga_bonus / 後手の右玉なら -gw。 gw == g_aiga_bonus なら従来と一致。
	{
		const int gw = migi_gw(g_aiga_bonus, g_aiga_bonus_white);
		return (ib ? g_aiga_bonus : -gw) * num / den;
	}
}

// 右玉の様式項を cp で返す (**先手視点**)。
//   SAFETY: 読むのは盤面・手駒・手数だけ。 副作用なし。
//   不変条件: 先後を入れ替えると符号が反転する (反対称)。 双方右玉なら 0。
// 前方宣言: 相掛かり角交換右玉の下駄と**排他**にするために参照する。
//   2026-10-13 ユーザー指示「条件が両方満たす場合はどちらかだけ / 160 を優先」。
static inline int aigakari_migi_bonus_black_view(const Position& pos);
static inline int migigyoku_bonus_black_view(const Position& pos) {
	// 🚨先後で別の G を持つので「両方 0 のとき」だけ抜ける
	if (g_migi_bonus == 0 && migi_gw(g_migi_bonus, g_migi_bonus_white) == 0)
		return 0;
	// 排他: 相掛かり側の下駄が発火する局面では、 こちらを 0 にする。
	//   (角交換 + 右玉 + 飛車先を双方伸ばした形 = 両方の条件を満たす)
	//   ユーザー指示 2026-10-13: **160 (相掛かり側) を優先**。 二重に乗らせない。
	if ((g_aiga_bonus != 0 || migi_gw(g_aiga_bonus, g_aiga_bonus_white) != 0)
	    && aigakari_migi_bonus_black_view(pos) != 0)
		return 0;
	const int ply = pos.game_ply();
	if (ply >= g_migi_fade_end)
		return 0;
	int num = 1, den = 1;
	if (ply > g_migi_fade_start) {
		den = g_migi_fade_end - g_migi_fade_start;
		if (den <= 0)
			return 0;
		num = g_migi_fade_end - ply;
	}
	// 角換わり限定 (既定): 双方が角を手駒に持っている
	if (g_migi_kakugawari
	    && !(hand_count(pos.hand_of(BLACK), BISHOP) > 0
	      && hand_count(pos.hand_of(WHITE), BISHOP) > 0))
		return 0;
	// 右玉 = 玉が自分の飛車と同じ側にいる居飛車。
	//   🚨「居飛車のまま」の条件が無いと振り飛車美濃 (先手玉 3八) まで拾ってしまう。
	int ib = 0, iw = 0;
	// 🚨閾値は既存の計測器 (runs/migigyoku/self_continue.py の `f >= 6`) と**揃えてある**。
	//   機構と器がずれると「右玉率が上がった」を自分の定義で自画自賛することになる。
	if (file_of(pos.square<KING>(BLACK)) <= FILE_4) {
		Bitboard rb = pos.pieces(BLACK, ROOK);
		while (rb) {
			if (file_of(rb.pop()) <= FILE_4) { ib = 1; break; }
		}
	}
	if (file_of(pos.square<KING>(WHITE)) >= FILE_6) {
		Bitboard rw = pos.pieces(WHITE, ROOK);
		while (rw) {
			if (file_of(rw.pop()) >= FILE_6) { iw = 1; break; }
		}
	}
	if (ib == iw)
		return 0;
	{
		const int gw = migi_gw(g_migi_bonus, g_migi_bonus_white);
		return (ib ? g_migi_bonus : -gw) * num / den;
	}
}

// ---- MovePicker 向けの公開判定 (2026-10-20)。 🚨**評価には使わない** (evaluate() は不変)。
//   ユーザー決定「右玉の誘導は MovePicker で行う / 先後とも押す」。
//   🚨閾値は上の migigyoku_bonus_black_view と**同じ**:
//     先手は玉と飛車が両方 FILE_4 以下 / 後手は両方 FILE_6 以上。
//     `pieces(c, ROOK)` は **飛車のみ** (龍は byTypeBB[ROOK] に入らない)。
//     片方を変えたらもう片方も変えること。
static inline bool migi_on_right(Color c, File f) {
	return (c == BLACK) ? (f <= FILE_4) : (f >= FILE_6);
}

/// 🚨🚨2026-10-20 **修正**: 「右玉が**完成した**か」だけ閾値を厳しくする。
///   旧実装は完成の判定も `migi_on_right` (先手 1-4 筋) を使っていたので、
///   **平手の先手玉 (5 筋) が 4 筋に動くだけで「完成」と判定され**、
///   そこでボーナスが消えて**本来の右玉 (玉 2-3 筋) に到達する前に止まった**。
///   実測: 右玉率が 基準 4.0% -> **0.0% (0/63 局)** に下がった (PO1024 / SCALE=1024)。
///   -> 完成の判定を **先手 2-3 筋 / 後手 7-8 筋** に厳しくする。
///   🚨「飛車が右か」は既存の下駄と**同じ閾値のまま** (FILE_4 / FILE_6)。
///      居飛車の飛車は先手 2 筋・後手 8 筋にいるので、 ここを厳しくする必要はない。
static inline bool migi_mo_done(Color c, File f) {
	return (c == BLACK) ? (f <= FILE_3) : (f >= FILE_7);
}

/// c 側の「玉が**右玉として完成**したか」「飛車 (龍を除く) のどれかが右側か」を 2 bit で返す。
/// bit0 = 玉が完成位置 (先手 2-3 筋 / 後手 7-8 筋) / bit1 = 飛車が右 (先手 1-4 / 後手 6-9)。
/// 🚨この関数は **MovePicker 専用**。 評価側の `migigyoku_bonus_black_view` は
///    従来どおり `migi_on_right` を両方に使う (右玉率の既存の測定と揃えるため)。
/// 副作用なし (盤面と手駒だけを読む)。
int migigyoku_side_bits(const Position& pos, Color c) {
	int bits = migi_mo_done(c, file_of(pos.square<KING>(c))) ? 1 : 0;
	Bitboard rb = pos.pieces(c, ROOK);
	while (rb) {
		if (migi_on_right(c, file_of(rb.pop()))) { bits |= 2; break; }
	}
	return bits;
}

/// c 側が「右玉に**組みかけ**」なら 0..256 の重み (fade 込み)、 それ以外は 0。
///   組みかけ = **飛車は右にいて、 玉はまだ右にいない** (bits == 2)。
///   🚨**既に右玉 (bits == 3) は 0** = 維持はしない。 崩すべきときに崩せるようにするため
///     (2026-10-20 ユーザー決定。 既存の下駄は維持もしていて、 それが問題だった)。
///   🚨玉だけ右 (bits == 1 = 振り飛車美濃) も 0 = 飛車を戻す手は押さない。
///   角換わり条件 (MigigyokuKakugawari) と fade (MigigyokuFadeStart/End) は
///   評価側の option を**共用**する。
int migigyoku_build_weight(const Position& pos, Color c) {
	const int ply = pos.game_ply();
	if (ply >= g_migi_fade_end)
		return 0;
	// 🚨序盤は押さない (既定 0 = 下限なし)。 fade の三角形では序盤が満額になるため。
	if (ply < g_migi_mo_ply_min)
		return 0;
	if (g_migi_kakugawari
	    && !(hand_count(pos.hand_of(BLACK), BISHOP) > 0
	      && hand_count(pos.hand_of(WHITE), BISHOP) > 0))
		return 0;
	if (migigyoku_side_bits(pos, c) != 2)
		return 0;
	if (ply <= g_migi_fade_start)
		return 256;
	const int den = g_migi_fade_end - g_migi_fade_start;   // > 0 は上の判定で保証される
	return 256 * (g_migi_fade_end - ply) / den;            // 1..255
}

// ---- 囲いの様式項 (2026-10-06)。 既定 0 = 完全に無効 (本番安全)
//   🎯floodgate 22,541 局の全数調査で見つけた「稀だが成績の良い配置」へ誘導する。
//     C|..S....G (玉が中央、 斜め右上に銀・斜め右下に金) は出現率 0.752% で
//     棋譜の前半 62.1% / 後半 60.4% と独立に再現した (2 位以下は 53-55%)。
//   🚨300 種類から選んでいるので上振れの余地は残る。 最後は 600 局で確かめる。
static int g_castle_bonus = 0;          // G [cp]。 0=無効
static int g_castle_shape = 8766;       // 位置帯*6561 + 3 進の配置。 既定 = C|..S....G
static int g_castle_fade_start = 20;    // 囲いを**作っている間**に効かせたいので早め
static int g_castle_fade_end = 90;

// 🚨tools/castle_survey.py の OFFS と**同じ順・同じ符号**。 ずれると別の形を誘導する。
//   自分から見た 左上 上 右上 / 左 右 / 左下 下 右下
static const int CASTLE_DF[8] = {+1, 0, -1, +1, -1, +1, 0, -1};  // 筋の差
static const int CASTLE_DR[8] = {-1, -1, -1, 0, 0, +1, +1, +1};  // 段の差

// その色が指定の囲いの形になっているか。
//   SAFETY: 読むのは盤面だけ。 盤外は必ず範囲検査してから piece_on を呼ぶ。
static inline bool castle_match(const Position& pos, Color c) {
	const int zone = g_castle_shape / 6561;
	int pat = g_castle_shape % 6561;
	const Square ksq = pos.square<KING>(c);
	const int kf = (int) file_of(ksq), kr = (int) rank_of(ksq);   // 0..8
	// 玉の位置帯 (自分から見た筋 1..9)。 後手は左右が反転する
	const int relf = (c == BLACK) ? (kf + 1) : (9 - kf);
	const int z = (relf <= 3) ? 0 : ((relf <= 6) ? 1 : 2);
	if (z != zone)
		return false;
	const Piece pg = make_piece(c, GOLD), ps = make_piece(c, SILVER);
	const int sgn = (c == BLACK) ? 1 : -1;
	for (int i = 0; i < 8; ++i) {
		const int want = pat % 3;
		pat /= 3;
		const int f = kf + sgn * CASTLE_DF[i];
		const int r = kr + sgn * CASTLE_DR[i];
		int got = 0;
		if (f >= 0 && f <= 8 && r >= 0 && r <= 8) {
			const Piece pc = pos.piece_on((File) f | (Rank) r);
			got = (pc == pg) ? 1 : ((pc == ps) ? 2 : 0);   // 🚨成銀・成金は数えない (調査と同じ)
		}
		if (got != want)
			return false;
	}
	return true;
}

// 囲いの様式項を cp で返す (**先手視点**)。 双方が同じ形なら 0 (反対称)。
#if 0   // 🏁MatCorr 定義 — MatCorrB=200 で対局 -41.9pt (2026-10-28 無効化)
// ---- 駒種ごとの値付け補正 (2026-10-21)。 **棋風のてこ**として使う。
//   ユーザー再定義: 「角を高くするとどうなるか」「香を活かす指し方にできるか」。
//   🚨値付けを「正しく直す」用途では期待値が無い (勝敗回帰は局クラスタ boot で 3/13 しか
//     残らなかった / 詳細 memory material_price_from_games_20261021)。 これは**棋風ノブ**。
//   単位は **cp/枚** (先手視点)。 既定 0 なら g_matcorr_any == false で 1 比較で抜ける。
static int  g_matcorr[PIECE_TYPE_NB] = {0};
static bool g_matcorr_any = false;
static void matcorr_refresh() {
	g_matcorr_any = false;
	for (int pt = PAWN; pt < PIECE_TYPE_NB; ++pt)
		if (g_matcorr[pt] != 0) { g_matcorr_any = true; return; }
}
/// 駒種ごとの補正の **先手視点**の和 [cp]。 盤上 13 種 + 持ち駒 7 種。
///   🚨持ち駒は**素の駒種**で数える (成駒は持ち駒にならない)。
///   🚨反対称 (先手 - 後手) なので minimax を壊さない。
#endif  // 🏁MatCorr 定義
// ---- 🎯**片側版: 自分の角 (+馬) が盤上にある枚数** (2026-10-22 / 設計 fable)。
//   `MatCorr*` と違い**反対称でない**。 fable の定理:
//     G*f_us = (G/2)(f_us - f_them) + (G/2)(f_us + f_them)
//   駒の**枚数**は保存量なので対称部が定数になり片側版 ≡ 両側版 (= B25) だが、
//   **盤上にあるか持ち駒か**は保存量でないので対称部が実体を持つ。
//   🎯対称な特徴は**勝率と相関 0** (s(swap)=s / p(swap)=1-p) なので、
//     この項は**定義上、 強さに寄与せず「どんな将棋にしたいか」だけを書く**。
//   🎯角狩りの動機が**厳密に 0** (非角で相手の角を取っても f_us は 1->1 で不変)。
//     動くのは「角交換が成立する (-G)」「持ち角を打つ (+G)」だけ。
//   🚨`g_own_color` は **root の手番** (`p2_search_begin` で毎 go セット)。 1 局の中では
//     我々の色は固定なので安全。 🚨🚨**本番 (USI_Ponder=true) に出す前に TT key の
//     salt が必須** — 連続 go / ponder で色が入れ替わると別の色の eval を再利用する。
//     判定器は ponder を使わず毎局 isready で TT を消すので判定では安全。
static int   g_ownboard_b    = 0;   // cp/枚
static int   g_ownboard_bias = 0;   // cp。 🎯実測 平均 0.494/0.504 なので **G/2** を推奨
static Color g_own_color     = BLACK;

// ---- 🎯**TT key の salt** (2026-10-22 / 設計 fable → 実装 opus → レビュー fable)。
//   root 色に依存する評価項 (`OwnBoardB` / `OwnKnightAdv`) を入れると、 1 プロセス内で
//   root 色が変わる go 列が TT を持ち越したときに**別の色で作った eval を再利用**する。
//   -> root 色ごとに cluster を分ける。
//
//   🚨🚨**「本番 (USI_Ponder=true) では必須」は誤りだった** (2026-10-22 fable レビューで訂正)。
//     YO の `go ponder` は「相手の予想手を指した後」が root なので**手番は常に我々**。
//     色が反転するのは `Stochastic_Ponder` (usi.cpp:802-806 / engine.cpp:100 既定 false)
//     のときだけで、 本番 toml には無い。 さらに csa_client は毎局 usinewgame + **isready**
//     を送り (rshogi-csa-client/src/engine.rs:332-334)、 isready は tt.clear() と
//     Worker::clear() を呼ぶ (yaneuraou-search.cpp:415-418) ので TT も history も消える。
//     -> 本番で salt が「分離」を行う場面は無く、 **無害な安全網**。
//     ✅**toml の規約: `Stochastic_Ponder` を ON にしないこと**。 ON にすると色が反転する。
//
//   🚨**既定 (片側版の option が全部 0) では salt = 0** なので XOR が恒等 = bit-exact。
//   🚨入れる場所は `TranspositionTable::first_entry()` の **1 箇所だけ**。
//     probe の呼び出し 6 箇所 + prefetch 3 箇所を手で XOR すると 1 箇所漏らすだけで壊れる。
//     first_entry は cluster index を決める関数なので、 そこを分ければ probe と
//     TTWriter (entry ポインタ経由で書く) が自動的に整合する。
//   🚨**分離は厳密ではなく確率的**: index(k) == index(k ^ SALT) となる局面が
//     約 1/clusterCount あり (本番 2048MB で 2^26)、 そこでは key16 が salt されていない
//     ので色をまたいで hit しうる。 実用上無視できる量。
//   🚨**TT の実効容量は減らない** (定数 XOR は全単射で mul_hi64 は全クラスタに写る)。
//     減るのは「両色のエントリが同時に居て取り合う」ときだけ = 持ち越し一般の話。
//   🚨**salt は汚染を閉じきらない**: correctionHistory / pawnHistory / mainHistory は
//     `isready` でしか消えない (Worker::clear) ので、 色が入れ替わる go 列では
//     反色で学習した補正が salt を素通りして次の go に効く。
//   🚨`p2_search_begin` は**全 Worker スレッド**が実行する (yaneuraou-search.cpp:1561) ので
//     **atomic** にする (x86 では relaxed load/store は plain mov でコスト 0)。
std::atomic<std::uint64_t> g_tt_salt{0};

#if 0   // 🏁OwnKnightAdv 定義 — 用量反応で棄却 (G=100 で 31.8%) (2026-10-28 無効化)
// ---- 🎯**片側版: 自分の桂が「1 回以上跳ねた」枚数** (2026-10-22)。
//   ユーザー指示「桂を責めに上手に使う棋風」。
//   🚨桂の**盤上枚数**は中盤セットで **96.8% が 2 枚** = 定数なので機構にならない。
//     位置なら揺れる: 跳ねた桂の枚数は {0: 41%, 1: 54%, 2: 4.5%} / **SD 0.568**
//     (角の盤上枚数の SD 0.502 より大きい)。
//   🚨初期位置は先手 2九/8九 = 相対段 **d=8**。 1 回跳ねて d=6、 2 回で d=4 (d は偶数のみ)。
//     「跳ねた」= **d <= 6**。 成桂は数えない (中盤セットでは 0 枚)。
//   🎯**相手の桂を一切数えない**ので狩りの動機はゼロ。 動くのは
//     「自分の桂を跳ねる (+G)」「跳ねた桂を取られる (-G)」だけ。
static int g_knightadv   = 0;   // cp/枚
static int g_knightadv_bias = 0;   // cp。 🎯実測 平均 0.631 なので **G*0.63** を推奨

#endif  // 🏁OwnKnightAdv 定義
// ---- 🎯片側項 K1 / K2 (2026-10-25 / fable 設計)。
//   K1 = 自玉 8 近傍の自分の**金銀** (成駒を含まない)。 決定感度 **5.7%** / SD 0.61 / 33 局で見える
//   K2 = 自陣 3 段 (相対段 d >= 6) の自分の**金銀**。 決定感度 **10.2%** / SD 0.91 / 73 局
//   🎯桂 (決定感度が低い一回性の決断) より**決定感度が 2〜4 倍**で、 反対称部は
//     「守備駒数の差」= net が既に重く値付けしている量なので G=10〜20 の嘘が相対的に小さい。
//   🎯**相手の駒を一切数えない**ので狩りの動機はゼロ。
static int g_kdef        = 0;   // cp/枚 (K1)
static int g_kdef_bias   = 0;   // cp。 実測の平均 1.06 枚なので **G** 前後を推奨
#if 0   // 🏁g_campgs (K2) — 同上 (2026-10-28 無効化)
static int g_campgs      = 0;   // cp/枚 (K2)
static int g_campgs_bias = 0;   // cp。 実測の平均 2.09 枚なので **G*2** 前後を推奨
#endif  // 🏁g_campgs (K2)

#if defined(P2_BUCKET_OFFSET)
// P2_BUCKET_OFFSET (2026-11-19 / PREREG-20261119-FC1PREC.md の追加 (2)): 本体の評価に LayerStack の bucket ごとに足す定数 [cp、 手番側の視点]。
//   USI option P2_BUCKET_OFFSET_B0..B7 (既定 0)。 OPN_Offset と同じく内部では cp × eff_fv (目盛りを合わせた後の raw) を本体の raw_total に足す
//   (OPN と混ぜる前。 lam = 256 で本体を呼ばない node には効かない)。 量子化で残った bucket ごとの片寄りを打ち消すためのもの。
//   全部 0 (既定) なら g_bucket_off_any = false で何も足さない = 従来と bit-exact。
// 先後別 (SIDEOFF / 2026-11-21): **root の手番 (= この探索で我々が先手か後手か)** で表を切り替える。
//   root が先手 (BLACK) のときは P2_BUCKET_OFFSET_SENTE_B0..B7 を、 後手 (WHITE) のときは既存の P2_BUCKET_OFFSET_B0..B7 を使う。
//   SENTE 表の各要素は既定 BUCKET_OFF_UNSET (= 未設定) で、 未設定の bucket は既存の表の値を使う (bucket ごとに部分上書き可)。
//   SENTE 表が全部未設定なら実際に足す表 g_bucket_off_act == g_bucket_off_cp = 従来と bit-exact。
//   ホットパス (evaluate_internal) は g_bucket_off_act / g_bucket_off_any を読むだけで、 root 色の分岐は持たない。
//   表の選択は select_bucket_off() で、 p2_search_begin (毎 go)・eval コマンド (dump)・option 変更時に行う。
//   🚨評価が (局面, root 色) の関数になる。 TT の salt は tt.cpp first_entry で 10-28 に外してある (#if 0) ので TT は色で分けない。
//     対局では root 色 (= 我々の色) は 1 局中変わらず (ponder の root も我々の手番 / Stochastic_Ponder は使わない)、
//     局の境目は csa_client・対局器とも isready (tt.clear + Worker::clear) を送るので、 別の色で作った eval は再利用されない。
//     🚨usinewgame は no-op なので、 局面集を 1 プロセスで連続に解く器では局面ごとに isready を送ること。
//     本体側に eval キャッシュ (EvalHash) は無い (grep 済み)。
static int g_bucket_off_cp[8] = {0, 0, 0, 0, 0, 0, 0, 0};       // 既存の表 (後手番 root で使う。 SENTE 未設定の bucket は先手番でも使う)
static constexpr int BUCKET_OFF_UNSET = -999;                     // SENTE 表の番兵 (= 未設定)。 option の最小値
static int g_bucket_off_sente[8] = {BUCKET_OFF_UNSET, BUCKET_OFF_UNSET, BUCKET_OFF_UNSET, BUCKET_OFF_UNSET,
                                    BUCKET_OFF_UNSET, BUCKET_OFF_UNSET, BUCKET_OFF_UNSET, BUCKET_OFF_UNSET};
static int g_bucket_off_act[8] = {0, 0, 0, 0, 0, 0, 0, 0};      // この探索で実際に足す表
static bool g_bucket_off_any = false;                             // g_bucket_off_act に 0 でない要素がある
static bool g_bucket_off_use_sente = true;                        // 最後に選んだのが SENTE 表か (option 変更時の再選択用。 初期 root 色 BLACK 相当)

// SIDEOFF2 (2026-11-21 / 案 A): 「我々が優勢なら後手でも SENTE 表を使う」切り替え。
//   USI option P2_SENTE_TABLE_SWITCH_CP [**表示 cp** = info score cp と同じ目盛り。 内部値に p2_display_rescale_cp(to_cp(v)) を掛けた値]、 既定 0 = 無効。
//   > 0 のとき p2_search_begin で、 同じ対局の直前の手 (= 直前に返した bestmove) の root の探索値 (我々視点の表示 cp) が
//   SWITCH_CP 以上なら、 我々が後手 (root = WHITE) でも SENTE 表を使う。 読みの途中では切り替えない (表は探索開始時に 1 回だけ決める)。
//   「同じ対局の直前の手」の判定: 記録した root 色が今の root 色と同じで、 今の game_ply - 記録した game_ply が 2 (通常・ponder hit)
//     か 0 (ponder miss: 外れた ponder 探索の root と同じ手数)。 isready (YaneuraOuEngine::clear) で記録を消す。 usinewgame はこのエンジンで no-op だが、
//     新しい対局は game_ply が直前の記録 + 2 にならないので従来どおりになる。 探索を飛ばした手 (定跡など) の score が未設定なら記録しない (= 無効)。
//   詰みの値は表示 cp でなく ±P2_PREV_MATE_CP として記録する (勝ち = 切り替え条件を満たす)。 ヒステリシスなし。
//   P2_SENTE_TABLE_SWITCH_LOG = true なら判定を info string P2SWITCH で出す (既定 false = 出さない。 全 Worker が出す)。
static int  g_sente_switch_cp  = 0;
static bool g_sente_switch_log = false;
static constexpr int P2_PREV_MATE_CP = 1000000;
static bool  g_prev_root_valid = false;
static Color g_prev_root_color = BLACK;
static int   g_prev_root_ply   = 0;
static int   g_prev_root_cp    = 0;   // 表示 cp (詰みは ±P2_PREV_MATE_CP)

/// SENTE 表を使うなら use_sente = true で、 bucket 補正の表を g_bucket_off_act に確定する。
//   p2_search_begin は全 Worker が呼ぶが、 同じ rootPos から同じ値を書くだけ (g_root_eff_fv と同じ扱い)。
//   途中状態を見せないよう、 any は最後に 1 回だけ書く。
static inline void select_bucket_off(bool use_sente) {
	g_bucket_off_use_sente = use_sente;
	bool any = false;
	for (int b = 0; b < 8; ++b) {
		const int s = g_bucket_off_sente[b];
		const int v = (use_sente && s != BUCKET_OFF_UNSET) ? s : g_bucket_off_cp[b];
		g_bucket_off_act[b] = v;
		if (v != 0) any = true;
	}
	g_bucket_off_any = any;
}
#endif

/// root 色が変わるたびに呼ぶ。 片側版の項が全部 0 なら salt は 0 のまま (bit-exact)。
static inline void update_tt_salt() {
	// 🚨**非対称な項が有効かどうか**で決める。 有効でないなら salt を立ててはいけない
	//   (立てると後手番の TT index が変わり、 既定の挙動と bit-exact でなくなる)。
	// 🏁2026-10-28: 棄却した項 (OwnKnightAdv / OwnCampGS) を条件から外した。
	//   どちらも常に 0 なので、 条件の値は変わらない = bit-exact。
	const bool asym = (g_ownboard_b != 0 || g_ownboard_bias != 0
	                   || g_kdef != 0 || g_kdef_bias != 0);
	// よく混ざった定数 (splitmix64 の increment)。 値そのものに意味はない。
	g_tt_salt.store((asym && g_own_color == WHITE) ? 0x9E3779B97F4A7C15ULL : 0ULL,
	                std::memory_order_relaxed);
}
/// 自分 (root 色) の桂で、 自陣からの進み d <= 6 のものを数える。
#if 0   // 🏁knightadv_count — OwnKnightAdv を無効化したため (2026-10-28)
static inline int knightadv_count(const Position& pos) {
	int n = 0;
	Bitboard bb = pos.pieces(g_own_color, KNIGHT);   // 🚨成桂 (PRO_KNIGHT) は含まない
	while (bb) {
		const Square sq = bb.pop();
		// 🚨先手は rank が小さいほど前。 後手は鏡。 相対段 d = 0..8
		const int r = (int) rank_of(sq);
		const int d = (g_own_color == BLACK) ? r : (8 - r);
		if (d <= 6)
			++n;
	}
	const int v = g_knightadv * n - g_knightadv_bias;
	return (g_own_color == BLACK) ? v : -v;
}
#endif  // 🏁knightadv_count

/// 🎯**先手視点**の値 [cp]。 root 色の盤上の角族の枚数だけを数える (相手側は読まない)。
static inline int ownboard_black_view(const Position& pos) {
	const int f_us = pos.pieces(g_own_color, BISHOP).pop_count()
	               + pos.pieces(g_own_color, HORSE).pop_count();
	const int v = g_ownboard_b * f_us - g_ownboard_bias;
	// 🚨root 色が後手なら「先手視点」にするため符号を反転する
	return (g_own_color == BLACK) ? v : -v;
}

/// 🎯**K1**: 先手視点の値 [cp]。 自玉 8 近傍の自分の金銀 (成駒を含まない) を数える。
static inline int kdef_black_view(const Position& pos) {
	const Square ksq = pos.square<KING>(g_own_color);
	// 🚨玉が無い局面 (詰将棋の入力など) では 0 にする。
	const int f_us = is_ok(ksq)
	               ? (kingEffect(ksq) & (pos.pieces(g_own_color, GOLD)
	                                     | pos.pieces(g_own_color, SILVER))).pop_count()
	               : 0;
	const int v = g_kdef * f_us - g_kdef_bias;
	// 🚨root 色が後手なら「先手視点」にするため符号を反転する
	return (g_own_color == BLACK) ? v : -v;
}

/// 🎯**K2**: 先手視点の値 [cp]。 自陣 3 段 (相対段 d >= 6) の自分の金銀を数える。
#if 0   // 🏁campgs_black_view (K2) — 棋風がこの局数では測れず未採用 (2026-10-28 無効化)
static inline int campgs_black_view(const Position& pos) {
	int n = 0;
	Bitboard bb = pos.pieces(g_own_color, GOLD) | pos.pieces(g_own_color, SILVER);
	while (bb) {
		const Square sq = bb.pop();
		// 🚨先手は rank が小さいほど前。 後手は鏡。 自陣 3 段 = d >= 6
		const int r = (int) rank_of(sq);
		const int d = (g_own_color == BLACK) ? r : (8 - r);
		if (d >= 6)
			++n;
	}
	const int v = g_campgs * n - g_campgs_bias;
	return (g_own_color == BLACK) ? v : -v;
}
#endif  // 🏁campgs_black_view (K2)

#if 0   // 🏁K★ (RootStyle) — MultiPV=1 で非 PV の score が -∞ = 発火 0 (2026-10-28 無効化)
// ---- 🎯K★ (RootStyleMargin / 2026-10-25)。 **root でだけ指す手を選び直す**ために外へ出す。
//   🚨`static` にしない (search から参照する)。 既定 0 なので search 側で 1 比較で抜ける。
int g_rootstyle_margin = 0;   // Value 単位。 0 = 無効 (bit-exact)
int g_rootstyle_kind   = 1;   // 1 = K1 (自玉 8 近傍の金銀) / 2 = K2 (自陣 3 段の金銀)

/// 🎯K★ 用: **root 色の側の様式量 (枚数)** を返す。 符号もスケールも付けない。
int rootstyle_count(const Position& pos) {
	if (g_rootstyle_kind == 2) {
		int n = 0;
		Bitboard bb = pos.pieces(g_own_color, GOLD) | pos.pieces(g_own_color, SILVER);
		while (bb) {
			const Square sq = bb.pop();
			const int r = (int) rank_of(sq);
			const int d = (g_own_color == BLACK) ? r : (8 - r);
			if (d >= 6)
				++n;
		}
		return n;
	}
	const Square ksq = pos.square<KING>(g_own_color);
	return is_ok(ksq)
	     ? (kingEffect(ksq) & (pos.pieces(g_own_color, GOLD)
	                           | pos.pieces(g_own_color, SILVER))).pop_count()
	     : 0;
}
#endif  // 🏁K★ (RootStyle)

#if 0   // 🏁matcorr_black_view — MatCorrB=200 で対局 -41.9pt (2026-10-28 無効化)
static inline int matcorr_black_view(const Position& pos) {
	int v = 0;
	for (int pt = PAWN; pt < PIECE_TYPE_NB; ++pt) {
		const int w = g_matcorr[pt];
		if (w == 0 || pt == KING)
			continue;
		const PieceType p = (PieceType) pt;
		v += w * (pos.pieces(BLACK, p).pop_count() - pos.pieces(WHITE, p).pop_count());
	}
	for (int pt = PAWN; pt < PIECE_HAND_NB; ++pt) {
		const int w = g_matcorr[pt];
		if (w == 0)
			continue;
		const PieceType p = (PieceType) pt;
		v += w * (hand_count(pos.hand_of(BLACK), p) - hand_count(pos.hand_of(WHITE), p));
	}
	return v;
}
#endif  // 🏁matcorr_black_view

static inline int castle_bonus_black_view(const Position& pos) {
	if (g_castle_bonus == 0)
		return 0;
	const int ply = pos.game_ply();
	if (ply >= g_castle_fade_end)
		return 0;
	int num = 1, den = 1;
	if (ply > g_castle_fade_start) {
		den = g_castle_fade_end - g_castle_fade_start;
		if (den <= 0)
			return 0;
		num = g_castle_fade_end - ply;
	}
	const int ib = castle_match(pos, BLACK) ? 1 : 0;
	const int iw = castle_match(pos, WHITE) ? 1 : 0;
	if (ib == iw)
		return 0;
	return (ib - iw) * g_castle_bonus * num / den;
}
static bool g_root_fv_valid = false;
// 2026-09-13: bucket 別 FV_SCALE (USI option FV_SCALE_BUCKETS、 カンマ区切り 8 個)。
//   held-out の校正で bucket ごとに目盛りが大きく違う (序盤 b0/b1 は我々の cp が教師の 5-8 倍) ため。
//   ★探索中に FV が変わると minimax の整合性が壊れるので、 **root の bucket で 1 回だけ選び、 探索中は固定**する
//     (既存の g_root_eff_fv と同じ扱い)。 0 のままの bucket は従来の 3 点補間値を使う = 部分上書きが可能。
//   全部 0 (既定) なら従来と bit-exact。
static int g_fv_scale_bucket[8] = {0, 0, 0, 0, 0, 0, 0, 0};
static bool g_fv_bucket_any = false;
// 2026-11-19 FVNODE: FV_SCALE_PER_NODE = true なら、 探索中の**各局面の** progress bucket (8 段) で FV を選ぶ
//   (表 = FV_SCALE_B0..B7。 0 の bucket はその bucket の中心の進行度で 3 点補間した値)。 false (既定) は従来と bit-exact。
//   表は p2_search_begin と eval コマンド (dump) の入口で作り直す (FV_SCALE 系の option を後から変えても追従する)。
static bool g_fv_per_node = false;
// P2_BUCKET_OFFSET の表は update_tt_salt の前 (SIDEOFF / 11-21 に移した) にある。
static int g_fv_node[8] = {28, 28, 28, 28, 28, 28, 28, 28};
// ç¬ãEMRè«ãçE(SPSA_LMR_TABLE_COEFF)ç¹§ã»FV ç¸ºE¨è·å¾¡E¸Âç¸ºE®é¬æºä¿£èï£°èEEt ç¸ºE§ç¹ï½«ç¹ï½¼ç¹äº¥å´è³å¤ÂE£èè¼ï¼E¸ºå¸ï½ç¸²ã»
//   è åEä¹¢ regime(FV_OPENING ç¹§åE½½E¿ç¸ºã»E½å²å¶Eç¸ºE§ç¸ºE¯ LMR ç¹§ãE½ºåä¹¢èï½¤ g_lmr_coeff_opening ç¹§åE½½E¿ç¸ºã»Âã»
//   è³E­é¶E¤èï½¤ç¸ºE¯è­ï½¢èEESPSA::LMR_TABLE_COEFFç¸²E±2_search_begin ç¸ºã»root ç¸ºE§1èæ§­â¡ç¸ºã»eff ç¹§å ¤E¢Eºè³å£¹E ç¸²ã»
//   start_searching ç¸ºã»reductions[] table ç¹§åµâç¸ºE®è«ãçEç¸ºE§èè´E§çï½¯å³¨âE¹§ã»è¬ï½¢éï½¢è³E­èï½ºè³ã»ç¸²ã»
static int g_lmr_coeff_opening = 2763; // è åEä¹¢ LMR è«ãçE(è­ï½¢è³ã»è¿´E¾é¦ã»LMR default ç¸ºE§ bit-exact)
static int g_root_eff_lmr = 2763;      // ç¹ï½«ç¹ï½¼ç¹å»ãéEºè³å£¹E ç¸ºã»LMR è«ãçE(è¬ï½¢éï½¢è³E­èï½ºè³ã»
// lazy head é§ç©ï½¾E¡ç¹ï½¢ç¹ï½¼ç¹ãE 0=full(è´E³èEE, 1=lazy-head(é«±é¸Vç¸ºE§éµâ«æEA), 2=margin lazy(B)ç¸²ã»
// Elo è±ç¢ï½¼ã»çç¸²ãéè³ã»1(lazy-head)ç¸²E®argin ç¸ºE¯ g_lazy_margin(cp)ç¸²ã»
static int g_lazy_mode = 1;
static int g_lazy_margin = 96;
// lazy-threat(éå°E¼E¼ç¹ï½»é«±ã»bit-exact): base(HalfKA+PSQT)ç¸ºE®ç¸ºE¿ç¸ºE®èï½® eval ç¸ºæ¢§ç¾éï½¢é¯ã»
// [alpha-M, beta+M] ç¸ºE®èãEfar node)ç¸ºEªç¹§ã»threat èEEE¸å¼±ã»éªè²E®å±ã»èï£°éå±Eè³E¸ç¸ºæâEéµâ«æç¸ºå¶Eç¸²ã»
// 0 = off(default, threat è³æãEé§ç©ï½¾E¡)ç¸²ã»0 = è­çæEèï½¤ç¸ºã»margin M, cp)ç¸²ESI: P2LazyThreatMarginç¸²ã»
static int g_lazy_threat_margin = 0;
// Experimental residual_v3 skip gate. Q8: 256=1.0, 128=0.5, 0=disabled.
static std::string g_bucket_mode = "progress8kpabs";
// 案 B の下準備 (2026-10-07)。 1 で root の V1 を info string に出す。 既定 0 = bit-exact。
int g_regime_dump = 0;

// ---------------------------------------------------------------------------
// è¿ï½¹è ï½´é©¥ä¸ãç¹ï½³ç¹ãEã£ç¹§E¯ç¹§E¹ (MATH_SPEC) : king_bucket / pack_bonapiece
// ---------------------------------------------------------------------------
static inline int king_bucket(Square ksq, Color persp) {
	Square sq = (persp == BLACK) ? ksq : Inv(ksq);
	int file = (int)sq / 9;
	int rank = (int)sq % 9;
	int file_m = (file >= 5) ? (8 - file) : file;
	return file_m * 9 + rank;
}
static inline bool is_hm_mirror(Square ksq, Color persp) {
	Square sq = (persp == BLACK) ? ksq : Inv(ksq);
	return ((int)sq / 9) >= 5;
}
static inline int pack_bonapiece(int pp, bool mirror) {
	if (mirror && pp >= FE_HAND_END) {
		int rel = pp - FE_HAND_END;
		int pidx = rel / 81;
		int s = rel % 81;
		int f = s / 9, r = s % 9;
		int s2 = (8 - f) * 9 + r;
		pp = FE_HAND_END + pidx * 81 + s2;
	}
	if (pp >= E_KING_BP) pp -= 81;
	return pp;
}

// ---------------------------------------------------------------------------
// FT / PSQT ç¹§E¢ç¹§E­ç¹ï½¥ç¹ï£°ç¹ï½¬ç¹ï½¼ç¹§E¿ (è¾E®è»ã»å³©è­E½°è­E½¹è ãE
//   acc[persp][k] = (int16)(ft_bias[k] + Eï½£_f ft_row(f)[k])     (i16 wrapping)
//   psqt[persp][b] = psqt_bias[b] + Eï½£_f psqt_row(f)[b]         (i32)
// StateInfo ç¸ºE«è«æEç¸ºå±Ââ¥o_move è¾E®è»ã»DirtyPiece)ç¸ºE§è­ï½´è­E½°ç¸²ããEéEEE§E»èå¢åEç¸ºE®ç¸ºE¿ç¹è¼ÎErefreshç¸²ã»
// ---------------------------------------------------------------------------

// ★2026-09-09: acc_add_i16 / acc_sub_i16 は HalfKA 側とも共用なので FT_DIM のまま。
//   threat 重みが i16 形式 (ThreatI16=1) の net では threat accumulator にもこれを使うため、
//   ★2026-09-10 解消: acc_add_threat_row_i16 (K 幅) を追加したので併用可能
//   (threat_acc_add / threat_acc_sub の i16 分岐)。 i8 net でのみ K を絞れる。
static inline void acc_add_i16(std::int16_t* acc, const std::int16_t* w);
static inline void acc_sub_i16(std::int16_t* acc, const std::int16_t* w);

// ===== accumulator cache(finnyç¹ãEã»ç¹æ¶ÎE=====
// é¾EªéEEE§E»èå¢åEç¸ºE®ç¹è¼ÎErefresh(èï½¨40é¬§æ´¥åTé¦çï½ªE­ç¸ºE¿ ç«ãE80-120KB ç¸ºE®ç¹ï½©ç¹ï½³ç¹Âç¹ï£°é±E­ç¸ºE¿)ç¹§åµÂã»
// è·å¾¡E¸Â (persp, king_bucket, mirror) ç¸ºE§èæ¦å±éªè²E®å±E ç¸ºã»accumulator ç¸ºE¨ç¸ºE®è¾E®è»ã»âéï½¨ç¸ºE«éï½®ç¸ºè´é¤ç¸ºå»Eç¸²ã»
// ç¹§E¨ç¹ï½³ç¹å»Îç¸ºE¯ç¸²æ¢§E­E£éï½ºç¸ºEªéï½¯é¨æ¦ÂE¤+ç¸ºæ´ã»è­ããç¸ºE® piece list ç¹§E¹ç¹ç¿«ã£ç¹å±ãç¹ï½§ç¹ãEã¨ç¸²éªEè¬âEâ ç¸²ã»
// è¾E®è»ã»âéï½¨ç¸ºE§è¶E¸ç¸ºE«è¿´E¾è»Âé«±E¢ç¸ºE®è±E½£éï½ºç¸ºEªéï½¯é¨éªâè»E°é©æâE¹§ä¹âç¹§â«Eµåæ£¡ç¸ºE¯ bit-exactç¸²ã»
// thread_local ç¸ºEªç¸ºE®ç¸ºE§ç¹§E¹ç¹ï½¬ç¹ãEã©é«¢éï½«E¶è·å»âç¸ºå±Âãï½©ç©ï½¾E¡ç¹è¼ãç¹§E¤ç¹ï½«èéªÎç¹ï½¼ç¹ç»å¾ç¸ºE¯è³ç´ï½»E£é¡Eªè¿E·ç¸ºE§è¾ï½¡èï½¹è¹æ¶Âã»
struct AccCacheEntry {
#if defined(P2_FT_I32)
	alignas(64) std::int32_t acc[FT_DIM];
#else
	alignas(32) std::int16_t acc[FT_DIM];
#endif
	std::int32_t psqt[NUM_BUCKETS];
	std::int32_t bp[PIECE_NUMBER_NB];  // ç¹§E¹ç¹ç¿«ã£ç¹å±ãç¹ï½§ç¹ãEã¨(BonaPiece èï½¤, BONA_PIECE_ZERO=é¨Eº)
	bool valid = false;
};
struct AccCache {
	AccCacheEntry e[2][45][2];  // [persp][king_bucket][mirror]
	int generation = -1;
};
static thread_local AccCache g_acc_cache;
static int g_net_generation = 0;  // é§ç©ï½¾E¡ç¹è¼ãç¹§E¤ç¹ï½«ç¹ï½­ç¹ï½¼ç¹ç»ã»èæºï¼E¸ºE¨ç¸ºE«++

// ç¹è¼Îéªè²E®ã»ç¹§E­ç¹ï½£ç¹ãEãç¹ï½¥é¨æªç½°): pos.state()->p2_acc[persp] / p2_psqt[persp] ç¸ºE¸è­ï½¸ç¸ºå´ï½¾E¼ç¹§Âç¸²ã»
// threat é©¥éªâ©é¦ã»i8)ç¹§ã»acc(i16)ç¸ºE¸é¨E¦è¿E·è«¡E¡è ï½µèï£°éå±Âã»
// ç¬ãE_restrict: int8_t ç¸ºE¯ char éï½»ç¸ºE§ strict aliasing è³ç¿«â ç¹§å³¨EE¹§å¥æ¢ç¸ºE¨ alias ç¸ºå±â§ç¹§ä¹âç¹§âÂã»
//   acc(è­ï½¸ç¸ºå´ï½¾E¼ç¸ºE¿èãEç¸ºE¨ weight é¦ã»net èã»ÂâµE±Eºç¸ºå±â»é©¥éªâç¹§å³¨âç¸ºã»ç¸ºE®é«±æ¨£ã¾éãEEè­å®ï½¤Eºç¸ºå±â»
//   ç¹§E³ç¹ï½³ç¹ä»£ãE¹ï½©ç¸ºE® hoisting/ç¹§E¹ç¹§E±ç¹§E¸ç¹ï½¥ç¹ï½¼ç¹ï½ªç¹ï½³ç¹§E°ç¹§å®ï½¨E±ç¸ºã»bit-exact è³æ¦ï½¤ã»ç¸²ã»
// ��2026-09-10: threat �d�݂� i16 �̏ꍇ�� K �������Z�B
//   �]���� acc_add_i16 (FT_DIM ���A HalfKA �Ƌ��p) ���Ă�ł�������
//   P2_THREAT_OUT < FT_DIM �ƕ��p�ł��Ȃ������B K == FT_DIM �Ȃ琶���R�[�h�͓���B
static inline void acc_add_threat_row_i16(std::int16_t* __restrict acc, const std::int16_t* __restrict w) {
#if defined(USE_AVX2)
	for (int k = 0; k < THREAT_ACC_DIM; k += 16) {
		__m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + k));
		__m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k));
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + k), _mm256_add_epi16(a, x));
	}
#else
	for (int k = 0; k < THREAT_ACC_DIM; ++k) acc[k] = (std::int16_t)(acc[k] + w[k]);
#endif
}
static inline void acc_sub_threat_row_i16(std::int16_t* __restrict acc, const std::int16_t* __restrict w) {
#if defined(USE_AVX2)
	for (int k = 0; k < THREAT_ACC_DIM; k += 16) {
		__m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + k));
		__m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k));
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + k), _mm256_sub_epi16(a, x));
	}
#else
	for (int k = 0; k < THREAT_ACC_DIM; ++k) acc[k] = (std::int16_t)(acc[k] - w[k]);
#endif
}

static inline void acc_add_threat_row(std::int16_t* __restrict acc, const std::int8_t* __restrict w) {
#if defined(USE_AVX2)
	for (int k = 0; k < THREAT_ACC_DIM; k += 16) {
		__m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + k));
		__m128i w8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + k)); // 16 x i8
		__m256i w16 = _mm256_cvtepi8_epi16(w8);                                // sign-extend ç«ãE16 x i16
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + k), _mm256_add_epi16(a, w16));
	}
#else
	for (int k = 0; k < THREAT_ACC_DIM; ++k) acc[k] = (std::int16_t)(acc[k] + (int)w[k]);
#endif
}

// threat é©¥éªâ©é¦ã»i8)ç¹§ã»acc(i16)ç¸ºE¸é¨E¦è¿E·è«¡E¡è ï½µè²å¸·E®ã»acc_add_threat_row ç¸ºE®èE½¾)ç¸²ã»
static inline void acc_sub_threat_row(std::int16_t* __restrict acc, const std::int8_t* __restrict w) {
#if defined(USE_AVX2)
	for (int k = 0; k < THREAT_ACC_DIM; k += 16) {
		__m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + k));
		__m128i w8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + k)); // 16 x i8
		__m256i w16 = _mm256_cvtepi8_epi16(w8);                                // sign-extend ç«ãE16 x i16
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + k), _mm256_sub_epi16(a, w16));
	}
#else
	for (int k = 0; k < THREAT_ACC_DIM; ++k) acc[k] = (std::int16_t)(acc[k] - (int)w[k]);
#endif
}

// threat index ç¸ºE®é©¥éªâ©é¦å¾ï½Eacc(i16)ç¸ºE¸èï£°éãEè²å¸·E®å±ÂE¯et ç¸ºE®ç¹è¼ãç¹ï½¼ç¹æ§­ã£ç¹å»ãè»ã»E²ã»
//   i8  net ç«ãEé¨E¦è¿E·è«¡E¡è ï½µèï£°è²å¸·E®ã»acc_add_threat_row / acc_sub_threat_row)ç¸²ã»
//   i16 net ç«ãEi16 é¶E´èï£°è²å¸·E®ã»acc_add_i16 / acc_sub_i16ç¸²EalfKA FT ç¸ºE¨è·å¾Â§)ç¸²ã»
static inline void threat_acc_add(std::int16_t* acc, const Net& net, std::size_t idx) {
	if (net.threat_is_i16) acc_add_threat_row_i16(acc, threat_row_i16(net, idx));
	else                   acc_add_threat_row(acc, threat_row(net, idx));
}
static inline void threat_acc_sub(std::int16_t* acc, const Net& net, std::size_t idx) {
	if (net.threat_is_i16) acc_sub_threat_row_i16(acc, threat_row_i16(net, idx));
	else                   acc_sub_threat_row(acc, threat_row(net, idx));
}

// ---- threat row-add ç¸ºE® NPS è­Âé©ï½©è¹ã»ç¹èEã£ç¹ãEâéï½¨+software prefetch)----
// threat weight table(~166MB)ç¸ºE¸ç¸ºE®ç¹ï½©ç¹ï½³ç¹Âç¹ï£°é¦çï½ªE­ç¸ºE¿ç¸ºå¾ãEç¹å»Îç¹éªã£ç¹§E¯(memory è ç©Âã»ç¸ºEªç¸ºæºï½ç¸²ã»
//   (1) index è»ç²å«è­ãâé¦å¾ãEèï½¨ç¹§E­ç¹ï½£ç¹ãEãç¹ï½¥ç¹ï½©ç¹§E¤ç¹ï½³ç¹§ã»prefetch ç¸ºå±â» DRAM ç¹ï½¬ç¹§E¤ç¹ãEÎ¦ç¹§E·ç¹§è²ãEè¬åã»éEEâé©¥å²ç³E//   (2) 4 é¦å¾âªç¸ºE¨ç¹§ââ» fused èï£°è²å¸·E®ã»acc ç¸ºE® load/store ç¹§ã»1/4 ç¸ºE«èé ï½¸ã»4 é¦å¾ãEé±E­ç¸ºE¿ç¹§åE½¸E¦é¦æªå±é¦ã»
// ç¸ºå¶Eç¸²Eª16 wrapping èï£°éå±ã»è¿E¯è¬ å¸ãEé¨ä»ç²é§ã»âç¸ºE®ç¸ºE§ç¸²ã»âéï½¨é¬EEEºä¸ãEç¹§E°ç¹ï½«ç¹ï½¼ç¹æÎ¦ç¹§E°ç¸ºE«èæ´Eç¸ºã»
// é¨¾åï½¬E¡ 1 é¦å¾âEç¸ºE¤ç¸ºE®é©ï½©éï½¨ç¸ºE¨ bit-exact ç¸ºE«è·å¾¡E¸Â(P2_THREAT_VERIFY ç¸ºE§è¾£E§è·ã»ç¸²ã»
// ç¬ãE8 net èEçç¸ºE®é¬®å¬Âæº½Eµçï½·E¯ç¸²Eª16 net(ThreatI16=1)ç¸ºE¯è æºæç¸ºE®é¨¾åï½¬E¡é©ï½©éï½¨ç¸ºE®ç¸ºE¾ç¸ºE¾(è¿´E¾é¦å¾ã¦ç¹§E¹ç¹ãEnet ç¸ºE§
//   è®æE½¨E¼ç¸ºE§ç¸ºéªâç¸ºã»Eµçï½·E¯ç¸ºE«è¬E¹ï½èE¥ç¹§å¾âEç¸ºã»ç¸²ã»

// i8 threat é¦ã»FT_DIM ç¹èEãE¹ãEç¸ºE®èï½¨ç¹§E­ç¹ï½£ç¹ãEãç¹ï½¥ç¹ï½©ç¹§E¤ç¹ï½³ç¹§ã»L1 ç¸ºE¸ prefetchç¸²ã»
static inline void prefetch_threat_row_i8(const Net& net, std::size_t idx) {
// profile11æ¿¶E²è»ã»å±¬è¿å»ã»è E¤è ä½Eè²ï½¬è³å£¹ãç¸ºE¯ç¸²âï½¹E³è®ã»é¦æªï½¨å¥EºE¦ç¸ºE®èE¸ã°ç¹ãEã¡ç¸ºE§èï½¨64Bç¹ï½©ç¹§E¤ç¹ï½³ç¹§ã»
// T0 prefetchç¸ºå¶EéE»éï½¨ç¸ºæäº«ç¸ºE£ç¸ºæºÂããEéï½ºè¬ãEE®å£½å¾ç¸ºE®ç¸ºE¿è­ï½§prefetchç¹§åE½½E¿ç¸ºå»Eç¹§å»â§è°¿ä¹âEç¸²ã»
#if defined(USE_AVX2) && defined(P2_THREAT_FORCE_PREFETCH)
	const char* p = reinterpret_cast<const char*>(threat_row(net, idx));
	for (int o = 0; o < FT_DIM; o += 64) _mm_prefetch(p + o, _MM_HINT_T0);
#else
	(void)net; (void)idx;
#endif
}

// 4 é¦ã»fused: acc Eï½±= w0+w1+w2+w3(i8ç«æ®µ16 é¨E¦è¿E·è«¡E¡è ï½µç¸²EE6 wrapping)ç¸²ã»
// w0..w3 ç¸ºE¯é±E­ç¸ºE¿è¿æ¶Eèãç(net èã»ç¸²â¥cc ç¸ºE¨é«±æ¨£ã¾éãEç«ãE__restrict è³çãEç¸²ã»
static inline void acc_addsub_threat_rows4(std::int16_t* __restrict acc,
    const std::int8_t* __restrict w0, const std::int8_t* __restrict w1,
    const std::int8_t* __restrict w2, const std::int8_t* __restrict w3, bool sub) {
#if defined(USE_AVX2)
	for (int k = 0; k < THREAT_ACC_DIM; k += 16) {
		__m256i s = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w0 + k)));
		s = _mm256_add_epi16(s, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w1 + k))));
		s = _mm256_add_epi16(s, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w2 + k))));
		s = _mm256_add_epi16(s, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w3 + k))));
		__m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + k));
		a = sub ? _mm256_sub_epi16(a, s) : _mm256_add_epi16(a, s);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + k), a);
	}
#else
	for (int k = 0; k < THREAT_ACC_DIM; ++k) {
		int s = (int)w0[k] + (int)w1[k] + (int)w2[k] + (int)w3[k];
		acc[k] = (std::int16_t)(sub ? acc[k] - s : acc[k] + s);
	}
#endif
}

// idx è»ã»n èä¹ãE i8 threat é¦å¾ï½Eacc ç¸ºE¸è³Âè«¡E¬èï£°éãEè²å¸·E®ã»4 é¦ã»fused+é¶E¯è¬¨E°ç¸ºE¯é¨¾åï½¬E¡)ç¸²ã»
// 2026-09-09 計測用: P2_THREAT_ENUMONLY=1 で「列挙はするが accumulator に加算しない」。
// full threat の税 -35.3% の内訳 (列挙 / accumulator 更新) を分離するため。
// 本番では未設定なので完全 no-op。
static inline bool threat_enumonly_env() {
	
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_ENUMONLY") != nullptr;
	return v;
#else
	return false;   // ★NPS(A5): 関数内 static のガード変数ロードを消す
#endif
}

// 2026-09-10 計測用: P2_THREAT_TOUCHONLY=1 で「列挙し、 各行の先頭 1 バイトだけ読み、 加算しない」。
//   nps_threat_split の A(threat OFF) / B(列挙のみ) / C(full) に T を加えると:
//     T - B = 純粋なミス待ち = **prefetch と再配置の天井そのもの**
//     C - T = カーネル計算 (i8->i16 展開と加算) = レジスタ常駐化などの対象
//   モデルを介さず天井が直接出るので、 効かない最適化に着手せずに済む。
static inline bool threat_touchonly_env() {
	
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_TOUCHONLY") != nullptr;
	return v;
#else
	return false;   // ★NPS(A5): 関数内 static のガード変数ロードを消す
#endif
}
#if defined(P2_PROBE_BUILD)
// 2026-09-16 計測専用 (本番ビルドには入らない): threat 列挙の「17.8pt」がどこで消えているかを切り分けるプローブ。
//   runs/threat_tax/enum_design.md §4。 どれも P2_THREAT_ENUMONLY と併用する前提 (加算しないので acc は 0 のまま、
//   探索木は P2_NOTHREAT と同一 = nodes 一致が実装の検算になる)。
//   P2_THREAT_ENUM_REPEAT=k : add_threat_dual を同じ acc に k 回呼ぶ (関数内の warm コスト)。 別バッファは作らない
//   P2_THREAT_ENUM_NOTABLES=1: 旧ループの表参照 (pair_base / from_offset / attack_order) を固定の小さな配列に置き換える
//   P2_THREAT_ENUM_ZERO=1   : add_threat_dual を呼ばない (列挙以外に漏れているコスト)
static inline int probe_enum_repeat() {
#if defined(P2_DIAG)
	static const int v = [] {
		const char* s = std::getenv("P2_THREAT_ENUM_REPEAT");
		int k = s ? std::atoi(s) : 1;
		return k < 1 ? 1 : (k > 8 ? 8 : k);
	}();
	return v;
#else
	return 1;   // ★NPS(A5)
#endif
}
static inline bool probe_enum_notables() {
	
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_ENUM_NOTABLES") != nullptr;
	return v;
#else
	return false;   // ★NPS(A5): 関数内 static のガード変数ロードを消す
#endif
}
static inline bool probe_enum_zero() {
	
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_ENUM_ZERO") != nullptr;
	return v;
#else
	return false;   // ★NPS(A5): 関数内 static のガード変数ロードを消す
#endif
}
// 2026-09-16 案B 段階 0 (planB_design.md §1.4 (a)): 区間ごとの REPEAT。 各回の前の n に戻すので別バッファは作らない。
//   P2_THREAT_ENUM_REP_PAWN=k   : 歩 V1 を k 回
//   P2_THREAT_ENUM_REP_STEP=k   : 旧ループを 桂銀金類 だけで k 回 (旧ループは 桂銀金類 → それ以外 の 2 パスになる)
//   P2_THREAT_ENUM_REP_SLIDER=k : 旧ループを 桂銀金類以外 (飛び駒) だけで k 回 (同上)
//   P2_THREAT_ENUM_SPLIT=1      : k=1 のまま 2 パス化だけ (順序効果の単独測定)
static inline int probe_env_k(const char* name) {
	const char* s = std::getenv(name);
	int k = s ? std::atoi(s) : 1;
	return k < 1 ? 1 : (k > 8 ? 8 : k);
}
static inline int probe_rep_pawn() {
#if defined(P2_DIAG)
	static const int v = probe_env_k("P2_THREAT_ENUM_REP_PAWN"); return v;
#else
	return 1;   // ★NPS(A5): 計測用。 既定に固定 = bit-exact
#endif
}
static inline int probe_rep_step() {
#if defined(P2_DIAG)
	static const int v = probe_env_k("P2_THREAT_ENUM_REP_STEP"); return v;
#else
	return 1;   // ★NPS(A5): 計測用。 既定に固定 = bit-exact
#endif
}
static inline int probe_rep_slider() {
#if defined(P2_DIAG)
	static const int v = probe_env_k("P2_THREAT_ENUM_REP_SLIDER"); return v;
#else
	return 1;   // ★NPS(A5): 計測用。 既定に固定 = bit-exact
#endif
}
static inline bool probe_enum_split() { 
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_ENUM_SPLIT") != nullptr; return v;
#else
	return false;   // ★NPS(A5): 関数内 static のガード変数ロードを消す
#endif
}
// 2026-10-07 **無条件カウンタ** (P2_PCOUNT_OUT=<file> で終了時に書き出す)。
//   既存のカウンタは全部「関数内 static の bool を見てから ++」なので、 その形に依らない
//   ゲート無しの計数を置いて「計時 14.4% 対 計数 6 回」のどちらが嘘かを決める。
//   🚨無条件 ++ なのでこの exe は計時に使わない。
static unsigned long long g_pc_ensure_both = 0, g_pc_early_ret = 0, g_pc_full_block = 0,
                          g_pc_dual = 0, g_pc_plan = 0, g_pc_feat = 0,
                          g_pc_eval = 0, g_pc_wt = 0, g_pc_skip = 0,
                          g_pc_grp_cmp = 0, g_pc_grp_same = 0, g_pc_bkt_same = 0,
// M2 (DESIGN-20261007-THREAT-DELTA.md §2.2): 差分が「使えたはず」の率。
//   m2_den  = full 経路のうち 親があり dirty_num != 0 の回数 (= 差分の母数)
//   pvalid  = うち 親の threat acc が valid (persp 別に数えるので最大 2*m2_den)
//   pv_grp  = うち 親の群 (threat 表 offset) が現局面と一致
//   pv_mir  = うち HM mirror も一致 (= 方式 (a) で実際に差分が使える件数)
//   both_ok = 両 persp で成立 (= add_threat_dual を丸ごと省ける回数)
                          g_pc_m2_den = 0, g_pc_pvalid = 0, g_pc_pv_grp = 0,
                          g_pc_pv_mir = 0, g_pc_both_ok = 0,
// funnel (コーディネータ指定の 4 段。 分母 = 2 * m2_calls (persp 別))。
//   a_any = valid な祖先が chain の上限内に居る (fb_valid_anc_dist の none 以外と同義)
//   a_mir = うち その祖先の HM mirror が現局面と一致 (resolve_threat_plan_impl と同じ式で玉を巻き戻す)
//   a_grp = うち 群 (threat 表 offset) も一致
//   a1_*  = 距離 1 (= 方式 (a) lazy で実際に使える形) に限ったもの
                          g_pc_m2_calls = 0, g_pc_a_any = 0, g_pc_a_mir = 0, g_pc_a_grp = 0,
                          g_pc_a1_any = 0, g_pc_a1_mir = 0, g_pc_a1_grp = 0,
                          g_pc_sink_calls = 0, g_pc_sink_tuples = 0;
// 距離のヒストグラム (1..9+)。 _m は mirror 一致のみ。
static unsigned long long g_pc_a_d[10] = {}, g_pc_a_dm[10] = {};
// 探索は深さ優先なので game_ply() を stack の添字に使えば [ply-1] は**現在の親**。
static int g_grp_at[1024] = {}, g_bkt_at[1024] = {};
static unsigned char g_grp_ok[1024] = {};
struct P2PcDump {
	~P2PcDump() {
		const char* e = std::getenv("P2_PCOUNT_OUT");
		if (!e || !*e) return;
		std::ofstream f(e, std::ios::app);
		if (!f) return;
		f << "ensure_both " << g_pc_ensure_both
		  << " early_ret " << g_pc_early_ret
		  << " full_block " << g_pc_full_block
		  << " dual " << g_pc_dual
		  << " plan " << g_pc_plan
		  << " feat " << g_pc_feat
		  << " eval " << g_pc_eval
		  << " want_threat " << g_pc_wt
		  << " skip_threat " << g_pc_skip
		  << " grp_cmp " << g_pc_grp_cmp
		  << " grp_same " << g_pc_grp_same
		  << " bkt_same " << g_pc_bkt_same
		  << " m2_den " << g_pc_m2_den
		  << " pvalid " << g_pc_pvalid
		  << " pv_grp " << g_pc_pv_grp
		  << " pv_mir " << g_pc_pv_mir
		  << " both_ok " << g_pc_both_ok
		  << " m2_calls " << g_pc_m2_calls
		  << " a_any " << g_pc_a_any
		  << " a_mir " << g_pc_a_mir
		  << " a_grp " << g_pc_a_grp
		  << " a1_any " << g_pc_a1_any
		  << " a1_mir " << g_pc_a1_mir
		  << " a1_grp " << g_pc_a1_grp
		  << " sink_calls " << g_pc_sink_calls
		  << " sink_tuples " << g_pc_sink_tuples;
		for (int i = 0; i < 10; ++i) f << " a_d" << (i + 1) << " " << g_pc_a_d[i];
		for (int i = 0; i < 10; ++i) f << " a_dm" << (i + 1) << " " << g_pc_a_dm[i];
		f << std::endl;
	}
};
static P2PcDump g_pc_dump;
// 2026-10-07 コスト側プローブ (DESIGN-20261007-THREAT-COST.md §2.3 / §3.3 / §4)。
//   P2_THREAT_ENUM_SLIDER_SETUPCONST=1 : 旧ループ (飛び駒のみ) の hoist 6 本を固定の行に置換する。
//       🚨NOTABLES と違い **表は本物のまま** (pair_base / attack_order の実体を読む)。 消えるのは
//       attack_pattern_id x2 / nrm x2 / from_offset x2 / attack_order 行引き x2 の **計算だけ**。
//       → ①「飛び駒の攻め手準備を表化」の天井そのもの (表化しても 1 load は残るので上界)。
//   P2_THREAT_ENUM_LE16SCAN=1          : 旧ループの**前**に targets を回して long_effect16 を読み
//       popcount を sink する走査を足す (列挙は不変)。 → ②転置が払う追加コストの天井。
//   P2_THREAT_ROWS_ADDSUB=1            : 各行を add した直後に sub する (acc 不変 = 探索木同一)。
//       → 1 行あたりの**加算**コストが直接出る (読む待ちは T レジームで既知)。 案 A の床の根拠。
static inline bool probe_enum_setupconst() {
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_ENUM_SLIDER_SETUPCONST") != nullptr; return v;
#else
	return false;
#endif
}
static inline bool probe_enum_le16scan() {
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_ENUM_LE16SCAN") != nullptr; return v;
#else
	return false;
#endif
}
// 2026-10-07 PREWARM: 旧ループが読む表の行を**先に prefetch** する。
//   ★cold が warm の 44 倍 = 初回 touch が列挙の時間の主。 その天井を測る。
//   prefetch は値を変えないので **構成的に bit-exact**。
// 2026-10-07 DT_COST: dirty-threat tuple の**構築費用だけ**を測る。
//   g_dt_build を立てて tuple は作るが、 plan.base を潰して評価は full のまま = **bit-exact**。
//   t(DT_COST) - t(base) = 構築費用。 これが列挙の 14.4% を超えたら差分化は入口で死ぬ。
static inline bool probe_dt_cost() {
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_DT_COST") != nullptr; return v;
#else
	return false;
#endif
}
static inline bool probe_enum_prewarm() {
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_ENUM_PREWARM") != nullptr; return v;
#else
	return false;
#endif
}
static inline bool probe_rows_addsub() {
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_ROWS_ADDSUB") != nullptr; return v;
#else
	return false;
#endif
}
// LE16SCAN の sink (volatile なので読みが実際に発行される)。
static volatile unsigned g_probe_le16_sink = 0;
// NOTABLES 用の固定表。 pbb[dsb*9+dc] (dsb<=1, dc<=8 → 18 要素) と ord[nrm[to]] (81 要素) を読んでも範囲内。
static const int g_probe_pair_base_zero[18] = {};
static const std::uint8_t g_probe_ord_zero[81] = {};
#else
static inline bool probe_rows_addsub() { return false; }
#endif
// =====================================================================
// 2026-10-07 S0/S1 プローブ (DESIGN-20261007-THREAT-DELTA.md §1.2 / §2.1)。
//   ★P2_PROBE_BUILD の外に置く (priming の判定が名前空間の外から呼ぶため)。
//   P2_DIAG が無いビルド (本番) では全部 false に畳まれて消える。
//
//   P2_THREAT_NOPRIME=1 : evaluate_with_no_return の priming を**常に切る** (P-L1)。
//       DT_COST と併用 → bit-exact になれば DT_COST の乖離の原因は priming (L1) で確定。
//   P2_THREAT_PRIME=1   : priming を**常に入れる** (P-L1b)。 DT_COST 無しで priming だけ強制する。
//       素と一致しなければ「priming = ensure_progress の経路変化」が単独で木を変える証拠。
//   P2_PROG_FULL=1      : ensure_progress の差分積み上げを使わず常に progress_full (P-L1c)。
//       これを両腕に立てて P-L1b を繰り返し、 一致すれば float の経路依存で確定。
//   P2_THREAT_DT_SINK=1 : full 経路で apply_threat_step を 1 回回すが StateInfo には**何も書かない**
//       (M1)。 push 先は volatile sink。 評価値・順序は不変 = 構成的に bit-exact。
//       t(base) - t(DT_SINK) = tuple 構築費用 (eval 1 回あたり)。
static inline bool probe_noprime() {
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_NOPRIME") != nullptr; return v;
#else
	return false;
#endif
}
static inline bool probe_prime() {
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_PRIME") != nullptr; return v;
#else
	return false;
#endif
}
static inline bool probe_prog_full() {
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_PROG_FULL") != nullptr; return v;
#else
	return false;
#endif
}
// M2 の funnel (祖先 walk は O(距離) なので既定の計時を汚さないよう env の裏に置く)。
static inline bool probe_m2() {
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_M2") != nullptr; return v;
#else
	return false;
#endif
}
static inline bool probe_dt_sink() {
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_DT_SINK") != nullptr; return v;
#else
	return false;
#endif
}
// =====================================================================
// 最適化で消されないための sink。 volatile なので読みは実際に発行される。
static volatile std::uint8_t g_threat_touch_sink = 0;
// i8 / i16 �̂ǂ���ł��g����o�b�`�K�p�B i16 �� 4 �s fused SIMD �������̂őf���ɉ񂷁B
static inline void apply_threat_rows(std::int16_t* acc, const Net& net,
                                     const int* idxs, int n, bool sub, std::size_t off = 0);
static inline void apply_threat_rows_i8(std::int16_t* acc, const Net& net,
                                        const int* idxs, int n, bool sub, std::size_t off = 0) {
	if (threat_enumonly_env()) return;   // 計測用: 加算をスキップ
	// 計測用: 行を読むが加算しない (T レジーム)。 ミス待ちだけを残す。
	//   T - B = prefetch/再配置の天井、 C - T = カーネル計算。
	if (threat_touchonly_env()) {
		std::uint8_t acc8 = 0;
		for (int i = 0; i < n; ++i) acc8 ^= (std::uint8_t)*threat_row(net, off + (std::size_t)idxs[i]);
		g_threat_touch_sink = acc8;
		return;
	}
	// 計測用 (2026-10-07): add した直後に sub する。 acc は不変 = 探索木同一 = bit-exact な時間差。
	//   t_addsub - t_normal = **1 行あたりの加算コスト** (2 回目は L1 hit なので読む待ちはほぼ出ない)。
	if (probe_rows_addsub()) {
		for (int j = 0; j < n; ++j) {
			acc_add_threat_row(acc, threat_row(net, off + (std::size_t)idxs[j]));
			acc_sub_threat_row(acc, threat_row(net, off + (std::size_t)idxs[j]));
		}
	}
	int i = 0;
	for (; i + 4 <= n; i += 4)
		acc_addsub_threat_rows4(acc,
			threat_row(net, off + (std::size_t)idxs[i]),     threat_row(net, off + (std::size_t)idxs[i + 1]),
			threat_row(net, off + (std::size_t)idxs[i + 2]), threat_row(net, off + (std::size_t)idxs[i + 3]), sub);
	for (; i < n; ++i) {
		if (sub) acc_sub_threat_row(acc, threat_row(net, off + (std::size_t)idxs[i]));
		else     acc_add_threat_row(acc, threat_row(net, off + (std::size_t)idxs[i]));
	}
}

static inline void apply_threat_rows(std::int16_t* acc, const Net& net,
                                     const int* idxs, int n, bool sub, std::size_t off) {
	if (!net.threat_is_i16) { apply_threat_rows_i8(acc, net, idxs, n, sub, off); return; }
	if (threat_enumonly_env()) return;   // �v���p: ���Z���X�L�b�v
	// 計測用: 行を読むが加算しない (T レジーム)。 ミス待ちだけを残す。
	//   T - B = prefetch/再配置の天井、 C - T = カーネル計算。
	if (threat_touchonly_env()) {
		std::uint8_t acc8 = 0;
		for (int i = 0; i < n; ++i) acc8 ^= (std::uint8_t)*threat_row(net, off + (std::size_t)idxs[i]);
		g_threat_touch_sink = acc8;
		return;
	}
	if (probe_rows_addsub()) {   // 2026-10-07: 上の i8 版と同じ趣旨
		for (int j = 0; j < n; ++j) {
			acc_add_threat_row_i16(acc, threat_row_i16(net, off + (std::size_t)idxs[j]));
			acc_sub_threat_row_i16(acc, threat_row_i16(net, off + (std::size_t)idxs[j]));
		}
	}
	for (int i = 0; i < n; ++i) {
		if (sub) acc_sub_threat_row_i16(acc, threat_row_i16(net, off + (std::size_t)idxs[i]));
		else     acc_add_threat_row_i16(acc, threat_row_i16(net, off + (std::size_t)idxs[i]));
	}
}

// full èå´ï½¨è²E®ç²å¾ç¸ºE® pair è¬¨E°è³ä¼å: attacker ç«ï½¤ 38 é¬§ã»EEEè¬¾E»è¬¦ã»ã»(èï£°è­å³¨ã»ç¹§E¹)ç«ï½¤ 8/é¬§ã»= 304ç¸²ã»
// è´åµE£è¼ï½è¬âEâE¸ºå¸â» 384(éãEâE­ãã»é¨¾æ¯ï½¸E­ flush ç¸ºå¶Eç¸ºE®ç¸ºE§è±E½£ç¸ºå±EE¸ºE«ç¸ºE¯è ï½±é«»E¿ç¸ºå±âç¸ºã»ç¸²ã»
static constexpr int THREAT_BATCH_CAP = 384;

// 2026-09-15 (A3 chkstk 除去): add_threat_dual (ensure_threat_both に inline) の idx 配列を小さくし、
//   ensure_threat_both のスタックフレームを 4KB 未満にして ___chkstk_ms (ページ probe) を eval 経路から消す。
//   旧: int idx_b[384+256], idx_w[384+256] = 5,120B → フレーム 6,040B (LTO 非 PGO 実測、 毎 eval で chkstk)。
//   新: int idx_b[320], idx_w[320] = 2,560B → フレーム ~3.5KB。
//   flush 単位 (BATCH_CAP) を 384→256 にしても加算は int16 wrapping (順序非依存) なので bit-exact。
//   K=64 では acc が 128B なので flush 粒度は性能に効かない。
//   不変条件 (配列外書き込みが起きない根拠):
//     (1) prelude (歩 V1 + s1_step_class 桂/銀/金類) は flush せず idx[n] に書く。 歩は 1 駒 1 slot、
//         s1_step_class は 1 駒あたり NMAX (桂2/銀5/金類6) slot を無条件に書いてから hit 数だけ n を進めるので、
//         書き込み index < Σ NMAX ≤ (盤上の玉以外の駒 ≤ 38) × 6 = 228 + (最後の駒の NMAX-1) < 234。
//     (2) prelude 後に n >= BATCH_CAP/2 (128) なら flush して n=0。 主ループは 1 slot ずつ書き n == BATCH_CAP で
//         flush するので、 主ループ開始時 n < 128 → 書き込み index < 256。
//     → 全書き込み index < max(234, 256) = 256 < IDX_CAP=320。 読み手 (apply_threat_rows) は [0, n) しか読まない。
//     配列への書き込みはスカラ (idx[n] = v) のみで SIMD 丸め書き込みは無い。
static constexpr int THREAT_DUAL_BATCH_CAP   = 256;
static constexpr int THREAT_DUAL_PRELUDE_MAX = 38 * 6 + 6;   // 234
static constexpr int THREAT_DUAL_IDX_CAP     = 320;
static_assert(THREAT_DUAL_BATCH_CAP <= THREAT_DUAL_IDX_CAP, "dual idx array must hold a full batch");
static_assert(THREAT_DUAL_PRELUDE_MAX <= THREAT_DUAL_IDX_CAP, "dual idx array must hold the prelude without flushing");

// ---- Direct15 の演算回数カウンタ (-DP2_DIRECT_STATS_BUILD + 実行時 P2_DIRECT_STATS=1) ----
// 目的: ns/node は同時実行中の対局などマシン負荷で簡単に汚れるが、 「1 eval あたり
// 何回 attacker を走査し、 何回 実利きを計算し、 何回 重みを引くか」 は負荷に依存しない。
// 差分化 (delta) が割に合うかは この回数比で判断できるので、 計時と切り離して数える。
// ホットパスは変更せず、 計数専用の関数を stats 有効時のみ呼ぶ。
#if defined(P2_DIRECT_STATS_BUILD)
struct Direct15Stats {
	long long calls = 0;        // threat_direct15_value 呼び出し回数 (= Direct15 を計算した eval 数)
	long long attackers = 0;    // attacker ループ反復数 (両 perspective 合計)
	long long pseudo_pass = 0;  // 空盤面利き ∩ targets が非空で残った attacker 数
	long long slider_calc = 0;  // occ 依存の実利き計算をした回数
	long long features = 0;     // 重み参照回数 (= 実際の attacker→target pair 数)
	long long cls_pieces = 0;   // 盤上駒数 (旧実装が毎 eval クラス分類していた枚数)
	long long pawn_hits = 0;    // 歩の一括処理で立った pair 数
	long long king_move = 0;    // 直前手が玉移動だった eval 数 (差分化で全再計算になる下限)
	long long bucket_known = 0, bucket_same = 0; // 親 bucket が判る eval / うち親と同一 bucket
	void print() const {
		sync_cout << "info string P2_DIRECT_STATS calls=" << calls
		          << " per_eval attackers=" << (calls ? (double)attackers / (double)calls : 0.0)
		          << " pseudo_pass=" << (calls ? (double)pseudo_pass / (double)calls : 0.0)
		          << " slider_calc=" << (calls ? (double)slider_calc / (double)calls : 0.0)
		          << " features=" << (calls ? (double)features / (double)calls : 0.0)
		          << " cls_pieces=" << (calls ? (double)cls_pieces / (double)calls : 0.0)
		          << " pawn_hits=" << (calls ? (double)pawn_hits / (double)calls : 0.0)
		          << " king_move_rate=" << (calls ? (double)king_move / (double)calls : 0.0)
		          << " bucket_same_rate=" << (bucket_known ? (double)bucket_same / (double)bucket_known : 0.0)
		          << " bucket_known=" << bucket_known << sync_endl;
	}
};
static Direct15Stats g_direct15_stats;
static const bool g_direct15_stats_on = []() {
	bool on = std::getenv("P2_DIRECT_STATS") != nullptr;
	if (on) std::atexit([]() { g_direct15_stats.print(); });
	return on;
}();
#endif

// ---- éªåï½¸E¬éï½¨é¨E±éªã»ç¹èÎç¹ç»å¾ -DP2_THREAT_STATS_BUILD + è³æº¯E¡æ¢§å¾ P2_THREAT_STATS=1)----
// chain delta(CAP>=2)ç¸ºE®éï½»éï½¨èE½¾èï½¹è­«æ·æè­E½­ç¸ºE®ç¸ºæºï½ç¸ºE®è³æ»ï½¸E¬: full/delta ç¸ºE®é¦æ¢§çç¸ºE¨
// fallback è­ãâè´å¢çEé©ï½¡ç¹§å¾ãE valid base ç¸ºE«è»ç¿«E¥ç¸ºã»chain é©ï½©éï½¨è¿E¯é­E½è«¤E§)ç¸²ã»
// è­ï½¢è³å£¹ã³ç¹ï½«ç¹å³¨ãç¸ºE¯è³æãEç¸ºE«ç¹§E³ç¹ï½³ç¹ä»£ãE¹ï½«ç¹§E¢ç¹§E¦ç¹ãEhot path ç¹§E³ç¹§E¹ç¹å»ãç¹ï½­)ç¸²ãEEè³Âç¹§E¹ç¹ï½¬ç¹ãEã© bench éªåï½¸E¬èEçç¸²ã»
#if defined(P2_THREAT_STATS_BUILD)
struct ThreatStats {
	long long full_calls = 0, delta_calls = 0;
	long long full_rows = 0, delta_rows = 0;

	long long zero_delta_calls = 0; // dirtyè¾E®è»ã»ã»è±E½£éä¸å§è æE¯ã»E¸å¼±â²è³æãEç¸ºE«ç¹§E¼ç¹ï½­ç¸ºE®èæ¨çE


	long long no_relevant_delta_calls = 0; // dirty tupleç¸ºç¶rofileè¿ï½¹è ï½´ç¸ºE¸è³Âè»ã»ã»ç¹§å³¨âç¸ºã»å±è¬¨E°
        // profile 12èï½£é¦E¨é¦æªï½µçï½·E¯ç¸ºE®fallbackéEEç½°ã»è²EµE±éªå»ã³ç¹ï½«ç¹è¾åè³å¤²E¼å³¨Âã»
        long long rep12_stable = 0, rep12_direct = 0, rep12_no_base = 0, rep12_nonlocal_chain = 0, rep12_changed = 0;
	long long fb_anc1 = 0, fb_anc2 = 0, fb_anc3 = 0, fb_anc4p = 0, fb_none = 0; // fallbackè­ãã»validé¾éã»éæå±¬
	long long lazy_far = 0, lazy_near = 0; // lazy-threat gate ç¸ºE® skip/é«±æ¹kip èæ¨çE
	// dirty-threat(do_move èï½´ç¸ºE®éæ»ãE)é¨E±éªã»
	long long dt_built = 0, dt_tuples = 0, dt_overflow = 0;
	int       dt_max = 0;
	// chain é©ï½©éï½¨è­ãã» chain é«ï½·ç¹åEãç¹å»ãç¹ï½©ç¹ï£°(index = ply è¬¨E°ç¸²ã» ç¸ºE¯ 9 èï½¥è³ã»
	long long chain_len[10] = {};
	void print() const {
		std::string ch;
		for (int i = 0; i < 10; ++i) { ch += std::to_string(chain_len[i]); if (i < 9) ch += "/"; }
		sync_cout << "info string P2_THREAT_STATS full_calls=" << full_calls
		          << " delta_calls=" << delta_calls
		          << " avg_full_rows=" << (full_calls ? (double)full_rows / (double)full_calls : 0.0)
		          << " avg_delta_rows=" << (delta_calls ? (double)delta_rows / (double)delta_calls : 0.0)
                          << " zero_delta_calls=" << zero_delta_calls
                          << " no_relevant_delta_calls=" << no_relevant_delta_calls
                          << " rep12_stable/direct/no_base/nonlocal/changed=" << rep12_stable << "/" << rep12_direct
                          << "/" << rep12_no_base << "/" << rep12_nonlocal_chain << "/" << rep12_changed
		          << " fb_valid_anc_dist(1flip/2/3/4+/none)=" << fb_anc1 << "/" << fb_anc2 << "/"
		          << fb_anc3 << "/" << fb_anc4p << "/" << fb_none
		          << " lazy_far=" << lazy_far << " lazy_near=" << lazy_near
		          << " dt_built=" << dt_built
		          << " dt_avg_tuples=" << (dt_built ? (double)dt_tuples / (double)dt_built : 0.0)
		          << " dt_max=" << dt_max
		          << " dt_overflow=" << dt_overflow
		          << " dt_overflow_rate=" << (dt_built ? (double)dt_overflow / (double)dt_built : 0.0)
		          << " chain_len(0..8,9+)=" << ch << sync_endl;
	}
};
static ThreatStats g_threat_stats;
// namespace ç¹§E¹ç¹§E³ç¹ï½¼ç¹å±ãEmain èéªâè»æEè¹ã»hot path ç¸ºE§ç¸ºE®ç¹§E¬ç¹ï½¼ç¹æEE»å¥â³ static è»æEè¹æ¶ã¡ç¹§E§ç¹ãEãç¹§è²å±é©ï½¿)ç¸²ã»
static const bool g_threat_stats_on = []() {
	bool on = std::getenv("P2_THREAT_STATS") != nullptr;
	if (on) std::atexit([]() { g_threat_stats.print(); });
	return on;
}();
static inline bool threat_stats_env() { return g_threat_stats_on; }
#define P2_TSTAT(expr) do { if (threat_stats_env()) { expr; } } while (0)
#else
#define P2_TSTAT(expr) do {} while (0)
#endif

// persp éä¹ãç¸ºE® threat è¿ï½¹è ï½´ç¹§è²ãEç¸ºE¦è»ç²å«ç¸ºå±Ââæ¨ active feature ç¸ºE® threat index ç¹§ã»f(idx) ç¸ºE¸è²ï½¡ç¸ºå¶Âã»
// bullet map_threat_features(Part2)ç¸ºE¨è·å¾¡E¸Âç¸²ãç«E81ç¹æ§­ãEnaive scan è¿å»Âã»
// è¿´E¾è¨E¨ç¸ºE¯ P2_THREAT_VERIFY ç¸ºE® ground truth èEçEé¬®å¬Âæº½æ²¿ = è³ä¹ãE bitboard è¿ãEç¸²ã»
// mirror = is_hm_mirror(king(persp), persp)ç¸²ã»
template <typename F>
static inline void for_each_threat_feature_naive(const Position& pos, Color persp, bool mirror, F&& f) {
	for (int from = 0; from < 81; ++from) {
		Piece pc = pos.piece_on((Square)from);
		if (pc == NO_PIECE) continue;
		int ac = Threat::tc_from_pt(type_of(pc));
		if (ac < 0) continue; // King / é«±æ§«E¯E¾éï½¡
		Color acol = color_of(pc);
		int file = from / 9, rank = from % 9;
		int from_n = Threat::normalize_sq(from, persp, mirror);
		Color oriented = (persp == BLACK) ? acol : ~acol;
		int as = (acol == persp) ? 0 : 1;

		// è¬¾E»è¬¦ã»ã» to(èï£°è­å³¨ã»ç¹§E¹)ç¸ºE§ threat pair ç¹§å ¤å±è½£E«ç¸²ãæ¼é¨Eºç¹ï½»é«±æ¨è¦ç¸ºE®ç¸ºE¿ç¸²ã»
		auto emit = [&](int to) {
			Piece tp = pos.piece_on((Square)to);
			if (tp == NO_PIECE) return;
			int dc = Threat::tc_from_pt(type_of(tp));
			if (dc < 0) return; // King
			// no-pawn-target profile: é²E«è¬¾E»è¬¦ã»ã»ç¸ºæ¢§E­E©(è­ï½ªè¬ãEç¸ºE® pair ç¸ºE¯ emit ç¸ºå±âç¸ºã»Âã»
			if (Threat::THREAT_EXCLUDE_PAWN_TARGET && dc == 0) return;
			int ds = (color_of(tp) == persp) ? 0 : 1;
			// profile 11ç¸ºEªç¸ºE©ç¸ºE§é«¯E¤èæ¶E ç¸ºæ®«airç¸ºE¯weight rowç¹§åæç¸ºæºâEç¸ºã»Âã»
			if (Threat::is_excluded_pair(as, ac, ds, dc)) return;
			int to_n = Threat::normalize_sq(to, persp, mirror);
			// profile 16 の ground truth。 board_effect (差分維持) ではなく attackers_to (magic bitboard)
			// から独立にひもを判定する。 これで「列挙」と「board_effect の差分維持」の両方を検算できる。
			const int h = Threat::THREAT_HANG_SPLIT
				? (int)(!pos.attackers_to(color_of(tp), (Square)to, pos.pieces())) : 0;
			const int idx = g_threat_tables.index(as, ac, oriented, ds, dc, from_n, to_n, h);
			if ((unsigned)idx >= (unsigned)Threat::THREAT_DIM_PROFILE) return;
			f(idx);
		};
		// stepper: ç¹æ§­ãç¸ºæé E­å³¨âç¹§ã»emitç¸²E´lider: è­Âè»æ´ã»èï£°è­å³¨ã»ç¹§E¹ç¸ºE¾ç¸ºE§é¨¾E²ç¸ºE¿ emit ç¸ºå±â»èæE­E¢ç¸²ã»
		auto step = [&](int sf, int sr) {
			if (sf >= 0 && sf < 9 && sr >= 0 && sr < 9) emit(sf * 9 + sr);
		};
		auto ray = [&](int df, int dr) {
			int rf = file + df, rr = rank + dr;
			while (rf >= 0 && rf < 9 && rr >= 0 && rr < 9) {
				if (pos.piece_on((Square)(rf * 9 + rr)) != NO_PIECE) { emit(rf * 9 + rr); break; }
				rf += df; rr += dr;
			}
		};
		const int fwd = (acol == BLACK) ? -1 : 1;
		switch (ac) {
		case 0: step(file, rank + fwd); break;                                       // Pawn
		case 1: ray(0, fwd); break;                                                  // Lance
		case 2: step(file - 1, rank + 2 * fwd); step(file + 1, rank + 2 * fwd); break; // Knight
		case 3: // Silver
			step(file - 1, rank + fwd); step(file, rank + fwd); step(file + 1, rank + fwd);
			step(file - 1, rank - fwd); step(file + 1, rank - fwd); break;
		case 4: // GoldLike
			step(file - 1, rank + fwd); step(file, rank + fwd); step(file + 1, rank + fwd);
			step(file - 1, rank); step(file + 1, rank); step(file, rank - fwd); break;
		case 5: ray(-1, -1); ray(-1, 1); ray(1, -1); ray(1, 1); break;               // Bishop
		case 6: ray(-1, 0); ray(1, 0); ray(0, -1); ray(0, 1); break;                 // Rook
		case 7: // Horse: è­æEslider + é¶E´è E¤step
			ray(-1, -1); ray(-1, 1); ray(1, -1); ray(1, 1);
			step(file - 1, rank); step(file + 1, rank); step(file, rank - 1); step(file, rank + 1); break;
		case 8: // Dragon: é¶E´è E¤slider + è­æEstep
			ray(-1, 0); ray(1, 0); ray(0, -1); ray(0, 1);
			step(file - 1, rank - 1); step(file - 1, rank + 1); step(file + 1, rank - 1); step(file + 1, rank + 1); break;
		default: break;
		}
	}
}

// persp éä¹ãç¸ºE® threat è¿ï½¹è ï½´ç¸ºE® bitboard é¬®å¬Âæº·ã»è¬ãEnaive è¿å»âEbit-exact è·å¾¡E¸Âç¸ºE® pair é«®ã»ç²Eç¸²ã»
//   attacker = é¶E¤è³ç¿«ã»é«±æ¨è¦é¬§åµÂEµarget = effects_from(pc,from,occ) ç«ï½© é«±æ¨è¦é¬§åµÂã»
//   effects_from ç¸ºE¯ YO è»E©ç¸ºå ºE»å¢ï½§ã»é¬¥ã»æ¿¶E²è»E¥èè´å©¿ç¸²ãlider ç¸ºE¯è­Âè»æ´ã»èï£°è­å³¨ã»ç¹§E¹ inclusiveç¸²ã»
//   è¬ä»ï½°åï½§ã»é©¥å¤§èç¸ºã»ç¸ºE§ç¸²E½aive è¿å»ã» step/ray éæ»ãEç¸ºE¨è·å¾¡E¸Âé«®ã»ç²ç¸ºE«ç¸ºEªç¹§ä¹Âã»
//   King é«¯E¤èãE attacker ç¸ºE¯ nonking ç¸ºE§ iterateç¸²ãarget ç¸ºE¯ targets_mask ç¸ºE§é«¯E¤èãE
//   (tc_from_pt<0 ç¹âEãç¹ãEãè³å´ï½¦ã»ç¸²E¦mit é¬EEã» naive ç¸ºE¨é¡E°ç¸ºEªç¹§ä¹â²ç¸²â¥cc ç¸ºE¸ç¸ºE® i16 wrapping
//   èï£°éå±ã»è¿E¯è¬ å¸âEç¸ºE®ç¸ºE§è·éE½¨å»ã» bit-exact(P2_THREAT_VERIFY ç¸ºE§è¾£E§è·ã»ç¸²ã»
// ---- threat **列挙** の演算回数カウンタ (-DP2_THREAT_ENUM_STATS_BUILD + 実行時 P2_THREAT_ENUM_STATS=1) ----
// ★既存の P2_THREAT_STATS (差分更新の統計) とは別物。 同じ名前にすると片方しか出ない。
// 目的: ns/node は同時実行の対局や学習で簡単に汚れるが、 「1 eval あたり何回 attacker を
// 走査し、 何回 effects_from を呼び、 何回 標的ループを回すか」 は負荷に依存しない。
// 列挙 -14〜16% の内訳が「攻め手側の固定コスト」か「標的側」かを、 計時と切り離して数える。
// ホットパスは #if で完全に消えるので通常ビルドの NPS は不変。
#if defined(P2_THREAT_ENUM_STATS_BUILD)
struct ThreatEnumStats {
	long long calls = 0;          // for_each_threat_feature 呼び出し回数
	long long attackers = 0;      // attacker ループ反復数 (= 盤上の玉以外の駒数)
	long long effects_calls = 0;  // effects_from 呼び出し回数
	long long atk_empty = 0;      // effects_from & targets が空だった attacker 数
	long long targets = 0;        // 標的ループ反復数
	long long excluded = 0;       // is_excluded_pair で捨てた pair 数
	long long emitted = 0;        // 実際に f(idx) を呼んだ数 (= accumulator に乗る pair 数)
	long long tgt_total = 0;      // 標的候補の升数 (= popcount(targets))
	long long tgt_zero_eff = 0;   // うち LONG_EFFECT の利き数が両色とも 0 の升 (転置なら magic 不要で飛ばせる)
	long long tgt_long_eff = 0;   // うち long_effect16 != 0 の升 (= 転置版が実際に逆引きする標的数)
	long long slider_hits = 0;    // 旧ループの hit のうち飛び駒由来 (= 転置版が出すべき pair 数)
	long long ps_nonslider = 0;   // 非飛び駒 attacker 数 (空盤面利き == 実利き -> magic 完全に不要)
	long long ps_slider = 0;      // 飛び駒 attacker 数
	long long ps_slider_skip = 0; //   うち 空盤面利き & targets が空 -> magic ごと省ける
	long long ps_slider_calc = 0; //   うち 実際に magic が要る
	long long ps_slider_waste = 0;
	long long pair_hist[324] = {};   // 2026-11-04: emitted pairs per BLACK-perspective pair id (as*162+ac*18+ds*9+dc)
	long long ac_scan[9] = {};       // 2026-11-04: old-loop attacker scans per attacker class
	long long ac_empty[9] = {};      // 2026-11-04: ... of which atk & targets was empty//   magic を呼んだが結果が空だった (= 取りこぼし)
	void print() const {
		const double c = calls ? (double)calls : 1.0;
		sync_cout << "info string P2_THREAT_ENUM_STATS calls=" << calls
		          << " per_eval attackers=" << attackers / c
		          << " effects_calls=" << effects_calls / c
		          << " atk_empty=" << atk_empty / c
		          << " targets=" << targets / c
		          << " excluded=" << excluded / c
		          << " emitted=" << emitted / c
		          << " atk_empty_rate=" << (attackers ? (double)atk_empty / (double)attackers : 0.0)
		          << " tgt_total=" << tgt_total / c
		          << " tgt_zero_eff=" << tgt_zero_eff / c
		          << " tgt_zero_rate=" << (tgt_total ? (double)tgt_zero_eff / (double)tgt_total : 0.0)
		          << " tgt_long_eff=" << tgt_long_eff / c
		          << " tgt_long_rate=" << (tgt_total ? (double)tgt_long_eff / (double)tgt_total : 0.0)
		          << " slider_hits=" << slider_hits / c
		          << " ps_nonslider=" << ps_nonslider / c
		          << " ps_slider=" << ps_slider / c
		          << " ps_slider_skip=" << ps_slider_skip / c
		          << " ps_slider_calc=" << ps_slider_calc / c
		          << " ps_slider_waste=" << ps_slider_waste / c
		          << sync_endl;
		for (int p = 0; p < 324; ++p) if (pair_hist[p])
			sync_cout << "info string P2_THREAT_PAIR_HIST as=" << (p / 162) << " ac=" << ((p / 18) % 9)
			          << " ds=" << ((p / 9) % 2) << " dc=" << (p % 9) << " n=" << pair_hist[p] << sync_endl;
		for (int a = 0; a < 9; ++a)
			sync_cout << "info string P2_THREAT_AC_SCAN ac=" << a << " scan=" << ac_scan[a] << " empty=" << ac_empty[a] << sync_endl;
	}
};
static ThreatEnumStats g_threat_enum_stats;
static inline bool threat_stats_on() {
	static const bool v = [] {
		const bool on = std::getenv("P2_THREAT_ENUM_STATS") != nullptr;
		if (on) std::atexit([]() { g_threat_enum_stats.print(); });
		return on;
	}();
	return v;
}
#define THRSTAT(field) do { if (threat_stats_on()) ++g_threat_enum_stats.field; } while (0)
// 2026-11-04: pair-class histogram (BLACK-perspective index -> pair id) and attacker-class scan counters.
static void thrstat_pair_impl(int ib);
#define THRSTAT_PAIR(ib) do { if (threat_stats_on()) thrstat_pair_impl(ib); } while (0)
#define THRSTAT_AC(ac, nonempty) do { if (threat_stats_on()) { ++g_threat_enum_stats.ac_scan[ac]; if (!(nonempty)) ++g_threat_enum_stats.ac_empty[ac]; } } while (0)
#else
#define THRSTAT(field) do { } while (0)
#define THRSTAT_PAIR(ib) do { } while (0)
#define THRSTAT_AC(ac, nonempty) do { } while (0)
#endif

template <typename F>
static inline void for_each_threat_feature(const Position& pos, Color persp, bool mirror, F&& f) {
#if defined(P2_PROBE_BUILD)
	++g_pc_feat;
#endif
	const Bitboard occ      = pos.pieces();
	const Bitboard nonking  = occ & ~pos.pieces(KING); // è³E¡éE¾åèã»
	// target èï½´ mask: no-pawn-target profile ç¸ºE§ç¸ºE¯è±E½©(è­ï½ªè¬èEÂâ½E¸E¡æ¿¶E²)ç¹§ãåèæ¶âE¹§ä¹Âã»
	// ç¬ãEos.pieces(PAWN) ç¸ºE¯è­ï½ªè¬èEã»è±E½©ç¸ºE®ç¸ºE¿(ç¸ºE¨é©¥ã»è¬å®£E¦åï½­å³¨ã»è»E¥ byTypeBB)ç¸ºEªç¸ºE®ç¸ºE§ GoldLike ç¸ºE¯è°¿ä¹ï½ç¸²ã»
	const Bitboard targets  = Threat::THREAT_EXCLUDE_PAWN_TARGET ? (nonking & ~pos.pieces(PAWN)) : nonking;
	const std::uint8_t* const nrm = g_norm_sq[persp][mirror ? 1 : 0]; // è±E½£éä¸å§ LUT(hoist)
	Bitboard attackers      = nonking;
	while (attackers) {
		THRSTAT(attackers);
		const Square from  = attackers.pop();
		const Piece  pc    = pos.piece_on(from);
		const int    ac    = Threat::tc_from_pt(type_of(pc)); // King é«¯E¤èåE¸ã»ç«ãEè¶E¸ç¸ºE« >= 0
		const Color  acol  = color_of(pc);
		const int    from_n   = (int)nrm[(int)from];
		const Color  oriented = (persp == BLACK) ? acol : ~acol;
		const int    as       = (acol == persp) ? 0 : 1;
		// attacker è³æ¦ï½¤å³¨ã» index è¬ä»ãEç¹§åµã¡ç¹ï½¼ç¹§E²ç¹ãEã¨ç¹ï½«ç¹ï½¼ç¹æï½¤æ¶âEhoist
		// (index = pair_base[as*162+ac*18 + ds*9+dc] + from_offset + attack_order ç¸ºE¨è·å¾¡E¸ÂéèE¡ã»ç¸²ã»
		const int    pattern  = Threat::attack_pattern_id(ac, oriented);
		const int* const pb   = Threat::THREAT_HANG_SPLIT
			? &g_threat_tables.pair_base2[0][as * 162 + ac * 18]
			: &g_threat_tables.pair_base[as * 162 + ac * 18];
		const int    fo       = g_threat_tables.from_offset[pattern][from_n];
		const std::uint8_t* const ord = g_threat_tables.attack_order[pattern][from_n];
		Bitboard atk = effects_from(pc, from, occ) & targets;
		THRSTAT(effects_calls);
		if (!atk) THRSTAT(atk_empty);
		while (atk) {
			THRSTAT(targets);
			const Square to   = atk.pop();
			const Piece  tp   = pos.piece_on(to);
			const int    dc   = Threat::tc_from_pt(type_of(tp));
			const int    ds   = (color_of(tp) == persp) ? 0 : 1;
			// profile 11ç¸ºEªç¸ºE©ç¸ºE§é«¯E¤èæ¶E ç¸ºæ®«airç¸ºE¯pair_base=-1ç¸ºE®ç¸ºæºï½Eï½ºé¦å¾ï¼ ç¸ºEªç¸ºã»Âã»
			if (Threat::is_excluded_pair(as, ac, ds, dc)) { THRSTAT(excluded); continue; }
			// h は被攻撃駒の**絶対色**の利き数で決まる (persp 相対の ds とは別物)。
			const int hofs = (Threat::THREAT_HANG_SPLIT
				? P2_HANG0(pos, color_of(tp), to) : 0) * Threat::NUM_PAIRS;
			const int idx = pb[hofs + ds * 9 + dc] + fo + (int)ord[nrm[(int)to]];
			// profileè»E¥tableé½ã»å³E¸ºE®èæ¶ã»é¨E¶èE½¾ç¸ºE«é¦æç¾è¾£E§ç¸ºå±âç¸ºã»Âã»
			if ((unsigned)idx >= (unsigned)Threat::THREAT_DIM_PROFILE) return;
			THRSTAT(emitted);
			f(idx);
		}
	}
}

// profile 12ç¸ºE®èï½£é¦E¨é¦ç¢âEè¬æ§­ÂE§eatureé¨Eºé«¢èãEprofile 11ç¸ºE¨è·å¾¡E¸Âç¸ºE«è«æ´â ç¸²ã»
// è·ã»E¦ä¹ãç¸ºE§ç¸²çï½¢E«è¬¾E»è¬¦ã»E§åE½¾E¡èï½¤ç«åå¤è¬¦ã»E§åE½¾E¡èï½¤ç«æ­eature indexç¸²éªã»èï½ºè³å¤ï£°ã»Eºä¸ï½¸è´ï½½ã»é¦å¾â¡ç¸ºä»£Eè°¿ä¹âEç¸²ã»
// èæEåªéï½ºè«æ´ã»ç¹§E½ç¹ï½¼ç¹å»ã»è´E¿ç¹§ä¸âEç¸²âµæèE§8éâ«E´E°ç¸ºE®é±å£¼E½E¢éï½®è¬ å¸ãèåµE£æï½é¯E­è¬âEâE¹§ä¹Âã»
struct ThreatRepresentativeRows {
        static constexpr int CAP = Threat::THREAT_REPRESENTATIVE_MAX_ROWS;
        int indices[CAP];
        std::uint8_t priorities[CAP];
        int count = 0;
        static inline bool better(std::uint8_t lhs_priority, int lhs_index,
                                  std::uint8_t rhs_priority, int rhs_index) {
                return lhs_priority > rhs_priority
                    || (lhs_priority == rhs_priority && lhs_index < rhs_index);
        }
        inline void offer(int index) {
                const std::uint8_t priority = g_threat_tables.rep_priority(index);
                if (count < CAP) {
                        indices[count] = index;
                        priorities[count] = priority;
                        ++count;
                        return;
                }
                int worst = 0;
                for (int i = 1; i < CAP; ++i)
                        if (better(priorities[worst], indices[worst], priorities[i], indices[i]))
                                worst = i;
                if (better(priority, index, priorities[worst], indices[worst])) {
                        indices[worst] = index;
                        priorities[worst] = priority;
                }
        }
        inline void apply_add(std::int16_t* acc, const Net& net) const {
                if (net.threat_is_i16) {
                        for (int i = 0; i < count; ++i)
                                threat_acc_add(acc, net, (std::size_t)indices[i]);
                        return;
                }
                for (int i = 0; i < count; ++i)
                        prefetch_threat_row_i8(net, (std::size_t)indices[i]);
                // count<=8ç¸ºEªç¸ºE®ç¸ºE§apply_threat_rows_i8èã»Îç¸ºE§è ¢ã»âEé¦æ²used SIMDç¹§åE½½E¿ç¸ºã»Âã»
                apply_threat_rows(acc, net, indices, count, false);
        }
};
// persp éä¹ãç¸ºE® threat èEEE¸å¼±EEdacc(HalfKA éï½ºè³å£¼E¾ã»ç¸ºE¸èï£°éå±Âã»
// i8 net ç¸ºE¯ç¸²æ¯ndex è¿æ±å¯E+prefetch)ç«ãE4 é¦ã»fused è³Âè«¡E¬èï£°éå±Âéªã»é¬®å¬Âæº½Eµçï½·E¯
// (i16 wrapping èï£°éå±ã»è¿E¯è¬ å¸¶ÂE§ç¸ºE«ç¹§å»Eé¨¾åï½¬E¡é©ï½©éï½¨ç¸ºE¨ bit-exact è·å¾¡E¸Â)ç¸²ã»
// off: 群化 net で使う表の先頭 index (群 * THREAT_DIM)。 群化しない net は 0。
static void add_threat_perspective(const Position& pos, Color persp, bool mirror, std::int16_t* dacc,
                                   std::size_t off = 0) {
        (void)off; // profile 12 (代表行) ビルドでは未使用 (群化 net は load で拒否する)
        if constexpr (Threat::THREAT_REPRESENTATIVE_ROWS) {
                ThreatRepresentativeRows representatives;
                for_each_threat_feature(pos, persp, mirror,
                        [&](int idx) { representatives.offer(idx); });
                StateInfo* const state = pos.state();
                state->p2_threat_rep_count[persp] = (std::uint8_t)representatives.count;
                for (int i = 0; i < representatives.count; ++i)
                        state->p2_threat_rep_idx[persp][i] = representatives.indices[i];
                P2_TSTAT(g_threat_stats.full_rows += representatives.count);
                representatives.apply_add(dacc, *g_net);
                return;
        }
	const Net& net = *g_net;
	int idxs[THREAT_BATCH_CAP];
	int n = 0;
	for_each_threat_feature(pos, persp, mirror, [&](int idx) {
		prefetch_threat_row_i8(net, (std::size_t)idx);
		idxs[n++] = idx;
		if (n == THREAT_BATCH_CAP) {
			P2_TSTAT(g_threat_stats.full_rows += n);
			apply_threat_rows(dacc, net, idxs, n, false, off); n = 0;
		}
	});
	P2_TSTAT(g_threat_stats.full_rows += n);
	apply_threat_rows(dacc, net, idxs, n, false, off);
}

// VERIFY éï½¨: naive scan è¿å»ãEthreat èEEE¸å¼±EèE°éãEbitboard è¿å»ã» ground truth)ç¸²ã»
static void add_threat_perspective_naive(const Position& pos, Color persp, bool mirror, std::int16_t* dacc,
                                         std::size_t off = 0) {
        (void)off;
        if constexpr (Threat::THREAT_REPRESENTATIVE_ROWS) {
                ThreatRepresentativeRows representatives;
                for_each_threat_feature_naive(pos, persp, mirror,
                        [&](int idx) { representatives.offer(idx); });
                representatives.apply_add(dacc, *g_net);
                return;
        }
	const Net& net = *g_net;
	for_each_threat_feature_naive(pos, persp, mirror,
		[&](int idx) { threat_acc_add(dacc, net, off + (std::size_t)idx); });
}

// ---------------------------------------------------------------------------
// threat èï½¥èå¸ãE incremental delta è­ï½´è­E½°(CAP=1)
// ---------------------------------------------------------------------------
// prev(=é¶E´èæ¦ï½±Âé«±E¢, valid base)ç¸ºE® p2_threat_acc[persp] ç¹§å®ï½µE·è½¤E¹ç¸ºE«ç¸²ãt->lastMove 1 è¬E¹ãEç¹§èãE
// threat pair è¾E®è»ã»Eé©ï½©éï½¨ç¸ºå±â»è¿´E¾è»Âé«±E¢ç¸ºE® p2_threat_acc[persp] ç¹§è²ï½¾å±Eç¸²ã»
//
// è¬EºE³ã» èçå§ç¸ºå¶EEthreat pair ç¸ºE¯ç¸²æé E­ãEé¬§å ¤E¨E®ç¸ºæï½¤å³¨Eç¸ºE£ç¸ºã»A(from)ç¹ï½»B(to) ç¸ºE«è¥ï½·èï£°ç¸ºå¶Eç¹§ãEEç¸²éªâé«¯èï½ç¹§å¾ï½ç¸²ã»
//   è ï½±é«»E¿ç¹§è²å¥³ç¸ºä»£EEattacker èãE=
//     (1) A/B è³ç¿«ã»é¬§ã»éï½»èåã» P@Aç¸²â«E§E»èåã» P@Bç¸²âåç¹§å³¨Eç¸ºæ»E½§ã»C@Bç¸²âµé ç¸ºE¡é¬§ã»
//     (2) A/B ç¹§è²èç¸ºåï½§ã»target-on-A/B)  çª¶E¦ A/B ç¸ºä¹ï½E8 è­E½¹è·ã»+ è­¯ã»ç¸ºE®é¨¾ã»scan
//     (3) A/B ç¹§å®ï½²E«ç¸ºã»slider(ray ç¸ºE®èï½¶é«ï½·/è»ã»E©E°) çª¶E¦ è·ã»scan ç¸ºE§è­Âè»æ´ã»èï£°è­è¾E§ã»
//   ç¹§å®ï½¦Eªocc/èE¡µcc è³E¡è­E½¹ç¸ºE§è¿æ±å¯Ededup)ç¸ºå±Ââæ¨ attacker ç¸ºE®èï½¨ pair ç¹§ã»
//     éï½ªocc ç¸ºE§ SUB / èE¡µcc ç¸ºE§ ADD ç¸ºå¶Eç¸²ã»
//   ç¸ºèï½ç¸ºE¯ detail Eï½§4 ç¸ºE® Group1-4(dedup/colinear/drop/King è·E«ç¹§Â)ç¸ºE¨è¬¨E°èE½¦é§ã»âéEæï½¾E¡ç¸ºE§ç¸ºãE½ç¸²ã»
//   Eï½§11 / audit(fable5)ç¸ºE®ç¸²çãEèãEscan ç¸ºE¸ç¸ºE®è³Âè­ï½¬è¹æ¶Âè´è³èï½¨ç¸ºE«è±ï½¿ç¸ºã»ÂE°ver-inclusion(è³æ»Eç¸ºE¯
//   pair ç¸ºæï½¤å³¨Eç¹§å³¨âç¸ºã»attacker ç¹§åï½·E·ç¸ºæï½Eç¸ºE¯ SUB/ADD ç¸ºæªå¶è°¿Eºç¸ºå±â»ç¹§E¼ç¹ï½­è¾E®è»ã»âç¸ºEªç¹§éE½®çãEç¸²ã»
//   King ç¸ºE¯ attacker/target ç¸ºE¨ç¹§ã»tc_from_pt<0 ç¸ºE§é¾Eªè¾ï½¶ç¸ºE«é«¯E¤èæ¶EE¹§å¾ï½E[BUG-7])ç¸²ã»
//
// ç¬ãEE¦Eªocc ç¸ºE®è»Âè¬Eèå²æ¨E[BUG-1] CAP=1): éï½ª occ/é¬§å ¤E¨E®ç¸ºE¯è¿´E¾è»Âé«±E¢(pos)ç¹§ã»A/B ç¸ºE«ç¸ºE¤ç¸ºã»â»ç¸ºE®ç¸ºE¿é¬æï½­E£ç¸ºå±â»è å±Eç¸²ã»
//   ([BUG-2] éï½ªç¸ºE® A ç¸ºE®é¬§å ¤E¨E®ç¸ºE¯è¬èEEèã»= st->lastMovedPieceTypeç¸²âï½­èãE B ç¸ºE®é¬§å ¤E¨E®ç¸ºE¯è¬èEEè ãE= pos.piece_on(B)ç¸²ã»
// ---- 1 è¬E¥ã»ç¸ºE® threat è¾E®è»ã»âéï½¨(accessor ç¹å¶ã»ç¹§E¹èï½±é¨¾å£¼E®æº¯E£ã» ----
// piece_parent / piece_child: ç¸ºæ´ã»è¬E¹ãEé¶E´èãEé¶E´è å¾ãEé¶E¤é«±E¢(èï½»è«¢ä¸ãEç¹§E¹ç¸ºE®é¬§ã»ç¸²ã»=-1 ç¸ºEªç¹§ç»é ç¸ºE¡ç¸²ã»
// chain delta(èå£½çè¾E®è»ã»ç¸ºE§ç¹§ãE½½E¿ç¸ºå»Eç¹§å»â§ç¸²â«ä¹¢é«±E¢è¿ããEç¹§ã»accessor ç¸ºE«è³Âéï½¬è¹æ¶Âã»
// push_pair(attacker Piece, attacker square, target Piece, target square, sign):
//   èçå§ pair ç¹§åµÂæªåEç¹§E¿ç¹å±Îç¸²éªãè²ï½¡ç¸ºå¶ÂãEEéä¹ãé«±æ¨ï½¾æï½­å¥ãç¸ºãE½ç¸ºèâE(persp / HM mirror ç¸ºE«èæE­å¥âE¹§ã»
//   è±E½£éä¸å§ç¹ï½»index è¹æ¶ã»è»E¼ç¸ºE³èE½ºç¸ºæãE = dt_to_index() ç¸ºçï½¡å¾â§)ç¸²ãE¼E¹§å¾âEç¹§å»EE1 è­ï½¬ç¸ºE®è¾E®è»ã»Îç¹§E¹ç¹å»ãE//   è³E¡éä¹ãç¹§å®ï½³ã»â´ç¹§ã»éä¹ãç¸ºæâEç¸ºE«éï½ªèEEãç¹§E­ç¹ï½£ç¹ï½³ç¹§ã»2 èæ§«å±ç¸ºåâé¬§ã»â²è±¸å»â´ç¹§ã»ç¸²ã»
template <typename FP, typename FC, typename PUSH>
static void apply_threat_step(int A, int B,
                              const Bitboard& occ_par, const Bitboard& occ_ch,
                              FP&& piece_parent, FC&& piece_child, PUSH&& push_pair) {
	// attacker apc@X ç¸ºE®è»E©ç¸ºæ¦ãE(èï£°è­å³¨ã»ç¹§E¹ç¸ºE®ç¸ºE¿)ç¹§ã»emit ç¸ºå¶Eç¸²ã»
	// atk ç¸ºE¯è»E¼ç¸ºE³èE½ºç¸ºæãEç¸ºE§ç¸²æ©ffects_from(apc, X, occ) & occç¸²éªâªç¸ºE§é¨æ§­â²ç¸ºE¦ç¸ºãE½ç¸²ã»
	// King target ç¸ºE¯ tc_from_pt<0 ç¸ºE§é«¯E¤èãEéï½ªocc ç¸ºE®éEEE½å²E½E®èå²æ¨ç¸ºçï½¦âï½Enonking mask ç¸ºE¯è´E¿ç¹§ä¸âEç¸ºã»ç¸²ã»
	auto emit_all = [&](Piece apc, int X, Bitboard atk, auto&& accessor, int sign) {
		while (atk) {
			const Square to = atk.pop();
			const Piece  tp = accessor((int)to);
			// King target é«¯E¤èãE+ profile é«¯E¤èãEno-pawn-target ç¸ºEªç¹§ç»æ´è¬èEã»è±E½© target ç¹§ãåèãEç¸²ã»
			// ç¬ãEUB/ADD è¿æ¢§å©¿ç¸ºE§è·å¾Â§è­ï½¡èï½¶ç¸ºE§è ï½¾ç¸ºä¸ãEç¸ºE§è¾E®è»ã»ã»è¬¨E´è·å»ã»è«æ´âE¹§å¾ï½ç¸²ã»
			if (Threat::target_excluded(type_of(tp))) continue;
			push_pair(apc, X, tp, (int)to, sign);
		}
	};

	// (1) è ï½±é«»E¿ attacker èãEã»è¿æ±å¯ç¸²E£itboard ç¸ºE® OR ç¸ºE§ dedup ç¸ºè¼ï½ç¹§ä¹ãEç¸ºE§ seen[81] ç¸ºE¯è³å´ï½¦âÂã»
	//   8 è­E½¹è·ä»£ã»ç¸²æ¢§æè»æ´ã»èï£°è­å³¨ã»ç¹§E¹ç¸²ã»  = (rookEffect | bishopEffect) & occ
	//     (rook/bishop ç¸ºE®è»E©ç¸ºéªã»è­Âè»æ´ã»èï£°è­å³¨ã»ç¹§E¹ inclusive ç¸ºEªç¸ºE®ç¸ºE§ & occ ç¸ºE§ç¸ºE¡ç¹§ã»â§ç¸ºE© terminal é«®ã»ç²E
	//   è­¯ãEEè»E©ç¸ºæ¦ãE(è³E¡æ¿¶E²)            = (knightEffect(BLACK,S) | knightEffect(WHITE,S)) & occ
	//     (S ç¹§è²èç¸ºä¹ï½¡ãEE S ç¸ºä¹ï½éä¹â»é¨¾ã»é«E¸ºã»2 è­E½¹è·ã»= è³E¡æ¿¶E²ç¸ºE®è­¯ãEç¸ºéªã»è¥ç¢å¯è·ã»
	//   over-inclusion ç¸ºE¯ SUB/ADD ç¸ºæªå¶è°¿Eºç¸ºå¶Eç¸ºE®ç¸ºE§è³çãE(è­ï½§ byte èåE½½ã»ray walk ç¸ºE¨è·å¾¡E¸Âé«®ã»ç²Eç¸²ã»
	auto collect = [](Square s, const Bitboard& occ) {
		return (rookEffect(s, occ) | bishopEffect(s, occ)
		        | knightEffect(BLACK, s) | knightEffect(WHITE, s)) & occ;
	};
	const Square Bs = (Square)B;
	Bitboard affected = collect(Bs, occ_par) | collect(Bs, occ_ch) | Bitboard(Bs);
	Bitboard ab       = Bitboard(Bs);            // é¬§å ¤E¨E®ç¸ºçï½¦EªèEEãèçå§ç¸ºå¶Eèã»= {A, B}
	if (A >= 0) {
		const Square As = (Square)A;
		affected |= collect(As, occ_par) | collect(As, occ_ch) | Bitboard(As);
		ab       |= Bitboard(As);
	}

	// (2) è·ã»attacker ç¸ºE®è¾E®è»ã»Eé©ï½©éï½¨ç¸²ã»
	while (affected) {
		const Square Xs = affected.pop();
		const int    X  = (int)Xs;
		if (X == A || X == B) {
			// é¬§å ¤E¨E®ç¸ºçï½¦EªèEEãé¡E°ç¸ºEªç¹§ã»éï½»èåã» P@A / éï½»èåã» P@B / è¿å¤ï½§ã»C@B)ç«ãEèï½¨ pair ç¹§ã»SUB/ADDç¸²ã»
			const Piece ppc = piece_parent(X);
			if (Threat::tc_from_pt(type_of(ppc)) >= 0)
				emit_all(ppc, X, effects_from(ppc, Xs, occ_par) & occ_par, piece_parent, -1);
			const Piece cpc = piece_child(X);
			if (Threat::tc_from_pt(type_of(cpc)) >= 0)
				emit_all(cpc, X, effects_from(cpc, Xs, occ_ch) & occ_ch, piece_child, +1);
		} else {
			// é¬§å ¤E¨E®è³æ¦ï½¤ã»Xç«é²A,B} ç¸ºEªç¸ºE®ç¸ºE§éï½ªèEEãè·å¾¡E¸Âé¬§ã»ç«ãEèçå§ç¸ºå±âEpair ç¸ºE®ç¸ºE¿ emitç¸²ã»
			//   mask = (éï½ªè»E©ç¸ºã»^ èE»èç¸ºã» | {A,B}
			//     - éï½ªèEE¸E¡è­E½¹ç¸ºE«è·E«ç¸ºE¾ç¹§å¾ÂâÂ°ç¸ºE¤ A/B èï½¥èæ¶ã»èãE= target é¬§åµEè³æ¦ï½¤ã»ç«ãEindex è·å¾¡E¸Âç¸ºE§
			//       SUB/ADD ç¸ºæï½®æãEé¶E¸è°¿Eºç¸ºå¶Eç¸ºE®ç¸ºE§é«¯E¤èæ¶E ç¸ºE¦ç¹§å»EErow-op èé ï½¸ã»ç¸²ã»
			//     - A/B ç¸ºE®èãEã»é¬§åµâ²è ¢ã»âEçå§ç¸ºå¶Eç¸ºE®ç¸ºE§è¶E¸ç¸ºE«èã»emitç¸²ã»
			//   ç¸ºèï½ç¸ºE¯è­ï½§ cray/cstep(è­E½¹è·ä»£EE¸ºE¨ç¸ºE® terminal è±ç¢ï½¼ã»ç¸ºE¨è·å¾¡E¸Âé«®ã»ç²ç¸²ã»
			const Piece apc = piece_child(X);
			const int   act = Threat::tc_from_pt(type_of(apc));
			if (act < 0) continue;                                // King / é«±æ§«E¯E¾éï½¡
			if (act != 1 && act < 5) {
				// èEE§ã»è±E½©/è­¯ã»é©«Â/é©¥é·¹E¡ã»: è»E©ç¸ºéªã» occ é«±æ¨ï½¾æï½­ã»reach = effects_from)ç¸²ã»
				//   ap = reach & occ_par, ac = reach & occ_ch ç¸ºEªç¸ºE®ç¸ºE§ ap^ac ç«ãEabç¸²ã»
				//   ç¹§å»â²ç¸ºE¦ mask ç«ï½© reach = reach & ab = rabç¸²ãE½¤çå§ç¸ºå±â§ç¹§ã»target ç¸ºE¯ A/B ç¸ºE®ç¸ºE¿ç¸²ã»
				const Bitboard rab = effects_from(apc, Xs) & ab;
				emit_all(apc, X, rab & occ_par, piece_parent, -1);
				emit_all(apc, X, rab & occ_ch,  piece_child,  +1);
			} else {
				// é¬å¸ãEé¬§ã»é¬¥ã»éãEé¬ãEé¬¥E¬/é®´ã»: ray terminal ç¸ºæèç¸ºä¸ãEç¸ºE§éï½ªèEEâç¹§ãï½¨è²E®å±âE¹§ä¹Âã»
				const Bitboard ap   = effects_from(apc, Xs, occ_par) & occ_par;
				const Bitboard ac   = effects_from(apc, Xs, occ_ch)  & occ_ch;
				const Bitboard mask = (ap ^ ac) | ab;
				emit_all(apc, X, ap & mask, piece_parent, -1);
				emit_all(apc, X, ac & mask, piece_child, +1);
			}
		}
	}
}

// ---------------------------------------------------------------------------
// dirty-threat éæ»ãE(Stockfish è­E½¹è ä¸Âã»026-07-21)
// ---------------------------------------------------------------------------
// do_move() ç¸ºE®è­ï½«èE½¾ç¸ºE§ç¸²å¾ï¼E¸ºE® 1 è¬E¹ãèçå§ç¸ºå±âEthreat pairç¸²éªEéæºã¡ç¹å±ÎEéä¹ãé«±æ¨ï½¾æï½­ã»ç¸ºE§
// StateInfo ç¸ºE«é¨éªã»ç¸²ãç«ECAP=1 è³æº¯E£ã»ensure_threat èã»ãéä¹ãç¸ºæâEç¸ºE«éï½ªèEEãç¹§E­ç¹ï½£ç¹ï½³)ç¸ºE¨ç¸ºE®é©è¼ï¼E
//   - ç¹§E¹ç¹§E­ç¹ï½£ç¹ï½³ç¸ºE¯ 1 è¬E¹âEç¸ºE¤ç¸ºã»1 èæ§­â¡ç¸ºã»è³E¡éä¹ãç¸ºE§èï½±è­ãEç¸²ã»
//   - éï½ª occupancy ç¸ºE®èå²æ¨ç¸ºã»ensure_threat èï½´ç¸ºE§è³å´ï½¦ã»-> chain é«ï½·ç¸ºE«è»E¶é«¯èâ²ç¸ºEªç¸ºã»Âã»
// éï½ª/èEä¹¢é«±E¢ç¸ºE¯è¿´E¾è»Âé«±E¢ pos ç¹§ã»A/B ç¸ºE«ç¸ºE¤ç¸ºã»â»ç¸ºE®ç¸ºE¿é¬æï½­E£ç¸ºå±âEE½é©¥ä¸ãç¹§E¯ç¹§E»ç¹§Eµç¸ºE§èå²æ¨ç¸ºå¶Eç¸²ã»
// (è­ï½§: èå£½çEchain(board overlay)è¿å»ã» 2026-07-19 ç¸ºE«è³æ»ï½¸E¬ -23ã»ã»36% ç¸ºE§è­½ã»åE¸²ãæ½è­E½¹è ä¸ãE
//  overlay ç¸ºE§ç¸ºE¯ç¸ºEªç¸ºã»do_move è­ãâéEºè³å£¹ã¡ç¹å±Îç¹§å ¤E©éªã»ç¸ºE®ç¸ºE§ç¸²â¦hain é©ï½©éï½¨è­ãâç¹§E¹ç¹§E­ç¹ï½£ç¹ï½³ç¸ºå¾¡E¸å´ï½¦âÂã»
static bool g_dt_build = false; // net load è­ãâéEºè³ã»threat è­çæEç¸ºä¹â½ full è ï½·è»E¶ç¸ºE§ç¸ºEªç¸ºã»

// p2_dt_state ç¸ºE®èï½¤ç¸²ã»
enum : std::uint8_t { DT_UNBUILT = 0, DT_OK = 1, DT_BROKEN = 2 };

// éæºã¡ç¹å±ÎEpack: bit0-6 attacker sq / bit7-13 target sq / bit14-18 attacker Piece
//                bit19-23 target Piece / bit24 add(1=+, 0=-)ç¸²Eiece < PIECE_NB(32)ç¸²ã¾quare 0..80ç¸²ã»
static inline std::uint32_t dt_pack(Piece apc, int X, Piece tp, int T, int sign) {
	return (std::uint32_t)X | ((std::uint32_t)T << 7)
	     | ((std::uint32_t)apc << 14) | ((std::uint32_t)tp << 19)
	     | ((sign > 0) ? (1u << 24) : 0u);
}

// éæºã¡ç¹å±ÎE-> threat indexç¸²ãEEpersp / HM mirror èæE­å¥ã»è±E½£éä¸å§ç¸ºE¯ç¸ºèï¼E¸ºE°ç¸ºä»£âé«®ã»E´ã»âE¹§ã»
// (è»ç²å«èE´ apply_threat_step ç¸ºE¯éä¹ãé«±æ¨ï½¾æï½­ã»ç¸²Eªndex() ç¸ºE¨è·å¾¡E¸ÂéèE¡èÂã»
static inline int dt_to_index(std::uint32_t d, Color persp, const std::uint8_t* nrm) {
	const int   X    = (int)(d & 0x7f);
	const int   T    = (int)((d >> 7) & 0x7f);
	const Piece apc  = (Piece)((d >> 14) & 0x1f);
	const Piece tp   = (Piece)((d >> 19) & 0x1f);
	const int   ac   = Threat::tc_from_pt(type_of(apc));
	const int   dc   = Threat::tc_from_pt(type_of(tp));
	const Color acol = color_of(apc);
	const int   as   = (acol == persp) ? 0 : 1;
	const int   ds   = (color_of(tp) == persp) ? 0 : 1;
	// profile 16 (hanging-split) は差分更新に非対応 (§4.5)。 ひもの有無は局面を見ないと決まらず、
	// この関数は Position を持たないので index を作れない。 g_dt_build を false にしてあるので
	// ここへは来ないはずだが、 来たら -1 で弾いて黙って壊れないようにする。
	if (Threat::THREAT_HANG_SPLIT) return -1;
	if (Threat::is_excluded_pair(as, ac, ds, dc)) return -1;
	const Color oriented = (persp == BLACK) ? acol : ~acol;
	const int   pattern  = Threat::attack_pattern_id(ac, oriented);
	const int   from_n   = (int)nrm[X];
	return g_threat_tables.pair_base[as * 162 + ac * 18 + ds * 9 + dc]
	     + g_threat_tables.from_offset[pattern][from_n]
	     + (int)g_threat_tables.attack_order[pattern][from_n][nrm[T]];
}

// ç¸ºèãE ply ç¸ºE®éæºã¡ç¹å±Îç¹§è²ï½®æ»Eç¸ºE«éæ»ãEç¸ºå¶Eç¸²ã»
// ç¬ãEçè¬ ã» pos ç¸ºã»st ç¸ºE®è»Âé«±E¢ç¸ºE§ç¸ºãE½ç¸ºèâE(éï½ªé¶E¤é«±E¢ç¹§ã»A/B é¬æï½­E£ç¸ºE§èå²æ¨ç¸ºå¶Eç¸ºæºï½Eç¸²ã»
//   eager ç¸ºEªç¹§ã»do_move è­ï½«èE½¾ç¸²ãazy ç¸ºEªç¹§çãEèãEensure_threat è­ãâè»E¼ç¸ºE°ç¹§å¾ï½ç¸²ãEEç¸ºE¡ç¹§å³¨Eèè´ç½²ç¹§åï½ºÂç¸ºæºâEç¸²ã»
static void build_dirty_threats_now(const Position& pos) {
	StateInfo* st = pos.state();

	const Move   m       = st->lastMove;
	const bool   drop    = m.is_drop();
	const int    B       = (int)m.to_sq();
	const int    A       = drop ? -1 : (int)m.from_sq();
	const Piece  pAfter  = pos.piece_on((Square)B);              // èE»ï½±Âé«±E¢ B ç¸ºE®é¬§ã»éï½»èãEè¬Eâ è å¾Ââµã»ç¹§é¡ï½¾E¼ç¸ºE¿)
	const Color  mover   = color_of(pAfter);
	const Piece  cap     = st->capturedPiece;                    // éï½ªè»Âé«±E¢ B ç¸ºE®é¬§ã»è¿æ¶Eç¹§å¾âé¬§ã»/ NO_PIECE
	const bool   capture = (cap != NO_PIECE);
	const Piece  pBefore = drop ? NO_PIECE
	                            : make_piece(mover, st->lastMovedPieceType); // éï½ªè»Âé«±E¢ A ç¸ºE®é¬§ã»è¬èEEèã»

	// éï½ª/èEEã»é¬§åµãE¹§E¯ç¹§E»ç¹§Eµ(A/B ç¸ºE®ç¸ºE¿è¿´E¾è»Âé«±E¢ç¸ºE¨é¡E°ç¸ºEªç¹§ã»ç¸²ã»
	auto piece_parent = [&](int sq) -> Piece {
		if (!drop && sq == A) return pBefore;
		if (sq == B)          return capture ? cap : NO_PIECE;
		return pos.piece_on((Square)sq);
	};
	auto piece_child = [&](int sq) -> Piece {
		if (!drop && sq == A) return NO_PIECE;
		if (sq == B)          return pAfter;
		return pos.piece_on((Square)sq);
	};

	// éï½ª/èEEã» occupancy(è¿´E¾è»Âé«±E¢ç¸ºE® occ ç¹§ã»A/B ç¸ºE® 2 èãEâ¡ç¸ºé¡ï½£æï½­E£)ç¸²ã»
	//   èEE= è¿´E¾è»Âé«±E¢ç¸²ãï½¦Eª = A ç¹§å ¤E«ä¹â»ç¸ºE¦(è¬Eâ ç¸ºE§ç¸ºEªç¸ºã»E°E´è·ã»ç¸²âåç¹§ç¿«ãç¸ºEªç¸ºã»âç¹§ã»B ç¹§å®èª ç¸ºE¨ç¸ºå¶Âã»
	const Bitboard occ_ch  = pos.pieces();
	Bitboard       occ_par = occ_ch;
	if (!drop)    occ_par |= Bitboard((Square)A);
	if (!capture) occ_par ^= Bitboard((Square)B);   // occ_ch ç¸ºE§ç¸ºE¯ B ç¸ºE¯è ¢ã»âEEä¹â²ç¸ºE¦ç¸ºã»EE
	Eval::P2DirtyThreat* const dst = st->p2_dirty_threats;
	int  n   = 0;
	bool ovf = false;
	apply_threat_step(A, B, occ_par, occ_ch, piece_parent, piece_child,
		[&](Piece apc, int X, Piece tp, int T, int sign) {
			if (n >= StateInfo::P2_DT_CAP) { ovf = true; return; }
			dst[n++].data = dt_pack(apc, X, tp, T, sign);
		});
	st->p2_dt_count = (std::uint16_t)n;
	st->p2_dt_state = ovf ? DT_BROKEN : DT_OK;
#if defined(P2_THREAT_STATS_BUILD)
	if (threat_stats_env()) {
		++g_threat_stats.dt_built;
		g_threat_stats.dt_tuples += n;
		if (ovf) ++g_threat_stats.dt_overflow;
		if (n > g_threat_stats.dt_max) g_threat_stats.dt_max = n;
	}
#endif
}

#if defined(P2_PROBE_BUILD)
// 2026-10-07 M1 (DESIGN-20261007-THREAT-DELTA.md §2.1): tuple 構築の費用**だけ**を測る。
//   build_dirty_threats_now と同じ入力 (親子の駒・occ) で apply_threat_step を 1 回回すが、
//   push 先は volatile な静的カウンタで **StateInfo には一切書かない**。
//   → p2_dirty_threats / p2_dt_count / p2_dt_state を汚さない = plan も acc も変わらない
//     = 評価値・探索順序は構成的に不変 (bit-exact)。
//   t(base) - t(DT_SINK) = eval 1 回あたりの構築費用。 これが 14.4 + 10.6 を食い潰すなら入口で死ぬ。
static volatile unsigned g_dt_sink = 0;
static void dt_build_cost_sink(const Position& pos) {
	const StateInfo* const st = pos.state();
	const Move   m       = st->lastMove;
	const bool   drop    = m.is_drop();
	const int    B       = (int)m.to_sq();
	const int    A       = drop ? -1 : (int)m.from_sq();
	const Piece  pAfter  = pos.piece_on((Square)B);
	const Color  mover   = color_of(pAfter);
	const Piece  cap     = st->capturedPiece;
	const bool   capture = (cap != NO_PIECE);
	const Piece  pBefore = drop ? NO_PIECE : make_piece(mover, st->lastMovedPieceType);
	auto piece_parent = [&](int sq) -> Piece {
		if (!drop && sq == A) return pBefore;
		if (sq == B)          return capture ? cap : NO_PIECE;
		return pos.piece_on((Square)sq);
	};
	auto piece_child = [&](int sq) -> Piece {
		if (!drop && sq == A) return NO_PIECE;
		if (sq == B)          return pAfter;
		return pos.piece_on((Square)sq);
	};
	const Bitboard occ_ch  = pos.pieces();
	Bitboard       occ_par = occ_ch;
	if (!drop)    occ_par |= Bitboard((Square)A);
	if (!capture) occ_par ^= Bitboard((Square)B);
	unsigned n = 0;
	apply_threat_step(A, B, occ_par, occ_ch, piece_parent, piece_child,
		[&](Piece, int, Piece, int, int) { ++n; });
	g_dt_sink += n;
	++g_pc_sink_calls; g_pc_sink_tuples += n;   // chain の cost 上限 (COST_MAX=128) が効くかの判断材料
}
#endif

// threat èEEE¸å¼±Eè¾ï½¡èï½¹è¹æ¶âE¹§çï½¨åï½¸E¬éï½¨ env(P2_NOTHREAT=1)ç¸²E¦valuate ç¸ºE¨ priming ç¸ºE§èï½±è­å³¨Âã»
static inline bool nothreat_env() {
	
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_NOTHREAT") != nullptr;
	return v;
#else
	return false;   // ★NPS(A5): 関数内 static のガード変数ロードを消す
#endif
}
// Direct15(profile15) head �����𖳌������� env(P2_NODIRECT=1)�B
// P2_NOTHREAT �� P2_HAS_THREAT �n�ɂ����������A profile15 �ł� P2_HAS_THREAT ��
// false �Ȃ̂� Direct15 ��؂�Ȃ��B head �� NPS �R�X�g�Ɗ��͊�^��
// ����o�C�i���E���� net �� A/B ���邽�߂Ɏg�� (P2_NOAUX / P2_NOEGT �Ɠ����`)�B
// USI option "P2NoDirect" �p�̃t���O�B �΋ǃn�[�l�X (tournament.exe) �͊��ϐ���
// �n���Ȃ����߁A Direct15 head �̗L���� A/B ����ɂ� USI option ���v��B
// env P2_NODIRECT �Ƃ� OR ����� (�v���X�N���v�g�� env ���g����������)�B
bool g_p2_no_direct_option = false;
// USI option "P2DirectCapRaw": Direct15 の raw 単位 cap を手動で上書きする (0 = auto)。
// 同一 exe・同一 net で cap だけ変えた A/B を再ビルド無しで回すための計測用スイッチ。
std::int32_t g_p2_direct_cap_raw_option = 0;
static inline bool nodirect_env() {
	// ★NPS(A5): 関数内 static のガード変数ロードを消す。 既定値に固定 = bit-exact
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_NODIRECT") != nullptr;
	return v || g_p2_no_direct_option;
#else
	return g_p2_no_direct_option;
#endif
}
// Direct15 �̃R�X�g�����؂蕪����f�f env (P2_DIRECT_DIAG)�B
//   1 = �d�ݔz��ւ̃����_���A�N�Z�X���O�� (index �v�Z�܂ł͍s��)
//   2 = index �v�Z���O�� (�����񋓂̂�)
// �P�v�@�\�ł͂Ȃ��œK���̓���������߂邽�߂̌v���p�B
static inline int direct_diag_env() {
	static const int v = [] { const char* s = std::getenv("P2_DIRECT_DIAG"); return s ? std::atoi(s) : 0; }();
	return v;
}
// Direct15 の i16 重みを強制的に無効化する env(P2_DIRECT_W32=1)。
// i16 化の寄与だけを単一バイナリで A/B するための計測用スイッチ (load 時のみ参照)。
static inline bool direct_force_w32_env() {
	
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_DIRECT_W32") != nullptr;
	return v;
#else
	return false;   // ★NPS(A5): 関数内 static のガード変数ロードを消す
#endif
}
// Direct15 �� i16 �d�݂�**�����I�ɗL����**���� env(P2_DIRECT_W16=1)�B ����� i32�B
static inline bool direct_force_w16_env() {
	
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_DIRECT_W16") != nullptr;
	return v;
#else
	return false;   // ★NPS(A5): 関数内 static のガード変数ロードを消す
#endif
}
// full èå´ï½¨è²E®æï½¼E·è»E¶ç¸ºE®éªåï½¸E¬éï½¨ env(P2_THREAT_FULL=1)ç¸²ãæèE¹è­ãã» priming ç¸ºå±âç¸ºã»
// (full éªEºè­E½­ç¹ï½¢ç¹ï½¼ç¹å³¨ãç¸ºE¯ priming ç¸ºã»ensure_threat è»E¼ç¸ºE³èE½ºç¸ºæï½¢ã»éæ³ï½¢å±ãç¹§E¹ç¹å»âç¸ºEªç¹§ä¹âç¹§ã»ç¸²ã»
static inline bool force_full_env() {
	
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_FULL") != nullptr;
	return v;
#else
	return false;   // ★NPS(A5): 関数内 static のガード変数ロードを消す
#endif
}
// dirty-threat ������**�L����**���� env (���� off)�B
//   2026-09-09 ���� (K=256�A ������ chain �o�O�C����A ���� exe/net �� env �ؑցE���ݎ��s):
//     threat �Ȃ� 434,368 / ���� 310,207 (-28.58%) / �S full �Čv�Z 325,811 (-24.99%)
//   �����̕��� 3.6 �|�C���g�x���B ���R�� ensure_threat_both �� fused dual scan
//   (�����_�� 1 �X�L�����ŏ������� add_threat_dual) ���u�����_�Ƃ� base �����v= �S full ��
//   �Ƃ��ɂ����g���Ȃ����߁B K �� 256 �ɍi���� full �Čv�Z�������Ȃ������ʁA
//   �����̌Œ�� (�e acc �� memcpy 512B + chain �T�� + dirty tuple ���� + 5.5 �s�K�p) ��
//   fused scan ���������B eager ���� (P2_DT_EAGER) �͍X�� -12.05% �Ř_�O�B
//   ���� fused dual ���������ɂ���ꂽ��ĕ]�����邽�߂ɃX�C�b�`�����c���B
static inline bool threat_delta_env() {
	
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_DELTA") != nullptr;
	return v;
#else
	return false;   // ★NPS(A5): 関数内 static のガード変数ロードを消す
#endif
}
// aux èEEE¸å¼±Eè¾ï½¡èï½¹è¹æ¶âE¹§çï½¨åï½¸E¬éï½¨ env(P2_NOAUX=1)ç¸²ã»
static inline bool noaux_env() {
	
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_NOAUX") != nullptr;
	return v;
#else
	return false;   // ★NPS(A5): 関数内 static のガード変数ロードを消す
#endif
}
// aux é¨çï½·E¯ç¹§è²ï½¼E·è»E¶é§ã»âè¥ï½°ç¹§å³¨âç¹§çï½¨åï½¸E¬éï½¨ env(P2_AUX_FORCE=1)ç¸²E¯et ç¸ºã»aux section ç¹§ã»
// è¬âEâE¸ºEªç¸ºã»E°E´è·å»ãç¹§ã»zero é©¥éªâ©ç¸ºE® aux ç¹ãEã»ç¹æ¶Îç¹§è²ç²è¬èEE  has_aux=true ç¸ºE«ç¸ºå¶EEeval èï½¤ç¸ºE¯
// è³æ¦ï½¤å³¨ÂâãEè¬ãEé¦æåEéå±ã»ç¹§E³ç¹§E¹ç¹å»ã»ç¸ºE¿éï½ºéãE= Gate1 NPS è²ï½¬è³å¤ç)ç¸²ã»
static inline bool aux_force_env() {
	
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_AUX_FORCE") != nullptr;
	return v;
#else
	return false;   // ★NPS(A5): 関数内 static のガード変数ロードを消す
#endif
}
// aux è»ç²å«ç¸ºE®ç¸ºE¿éªåï½¸E¬éï½¨ env(P2_AUX_ENUMONLY=1): éï½ºè½£E« index ç¸ºE®è»ç²å«ç¸ºE¯é¦å¾â§ç¸ºã»FT é¦æåEéå±EE// éµâ«æç¸ºå¶Eç¸²ãï½¡æåEéãEmemory bound)ç¸ºE¨è»ç²å«Eè»E©ç¸ºéªã»ç¹ãEã»éæ»ãE)ç¸ºE®ç¹§E³ç¹§E¹ç¹äº¥ã»ç¹§éãEç¸ºéç(DIAG)ç¸²ã»
static inline bool aux_enumonly_env() {
	
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_AUX_ENUMONLY") != nullptr;
	return v;
#else
	return false;   // ★NPS(A5): 関数内 static のガード変数ロードを消す
#endif
}

// ---- chain é©ï½©éï½¨ç¸ºE®ç¹ä»£Îç¹ï½¡ç¹ï½¼ç¹§E¿ ----
// P2_DT_CHAIN_MAX: é©ï½¡ç¹§åºæèE§ ply è¬¨E°(chain é©æ¦ãEç¹§Eµç¹§E¤ç¹§Eº)ç¸²ã»
// P2_DT_COST_MAX : chain è³ç¿«ã»ç¹§E¿ç¹å±Îé±ä¹çEç¸ºE®è³ä¼åç¸²ãï½¶ã»â´ç¸ºæºï½Efull ç¸ºE®è­E½¹ç¸ºæï½®å³¨Eç¸ºE®ç¸ºE§ full ç¸ºE¸ç¸²ã»
//   è­¬E¹è«¡E°: full 1 èãE= é¶E¤è³éãEé¬§ã»~38)ç¸ºE®è»E©ç¸ºæ¦ãEè¬ãE+ avg_full_rows é¦å¾ãEç¹ï½©ç¹ï½³ç¹Âç¹ï£°é¦çï½ªE­ç¸ºE¿ç¸²ã»
//   chain é©ï½©éï½¨ç¸ºE¯è»ç²å«ç¸ºå¾¡E¸å´ï½¦ã»ç¹§E¿ç¹å±Îç¸ºE¯ do_move è­ãâéEºè³å£½E¸å»â©)ç¸ºEªç¸ºE®ç¸ºE§ç¹§E³ç¹§E¹ç¹å»ã»ç¸ºE»ç¸ºE¼é¦çï½ªE­ç¸ºE¿ç¸ºE®ç¸ºE¿ç¸²ã»
//   ç¹§å»â²ç¸ºE¦ç¸²æ¶hain ç¸ºE®ç¹§E¿ç¹å±Îè¬¨E° <= full ç¸ºE®é¦æ¢§çç¸²éªâç¹§ã»chain ç¸ºæï½¿ã»âE³å³¨Eç¸²ãE½®æ»ï½¸E¬ avg_full_rows ç¸ºE¯
//   startpos 34 / è³E­é¶E¤è»Âé«±E¢ ~100 èæ¦ï½¾å¾âEç¸ºE®ç¸ºE§ç¸²â½E¿æï½®äº¥ã»ç¸ºE«èEEâç¸ºE¦ 128 ç¹§åE½¸ä¼åç¸ºE«ç¸ºå¶Eç¸²ã»
static constexpr int P2_DT_CHAIN_MAX = 24;
static constexpr int P2_DT_COST_MAX  = 128;

// chain è³ç¿«ã»èï½¨ ply ç¸ºE®éæºã¡ç¹å±Îç¹§å ¤æ¨è¨E¨ç¸ºE® persp/mirror ç¸ºE§ index è¹æ¶E ç¸ºE¦ acc ç¸ºE¸ add/sub ç¸ºå¶Eç¸²ã»
// i16 wrapping èï£°è²å¸·E®å±ã»è¿E¯è¬ å¸âEç¸ºE®ç¸ºE§ ply é¬EEã»add/sub é¬EEã»èï½»è«¢ã»é¨¾åï½¬E¡é©ï½©éï½¨ç¸ºE¨ bit-exact è·å¾¡E¸Â)ç¸²ã»
// i8 net ç¸ºE¯ç¸²æ¯ndex è¿æ±å¯E+prefetch)ç«ãE4 é¦ã»fused è³Âè«¡E¬é©ï½©éï½¨ç¸²éªã»é¬®å¬Âæº½Eµçï½·E¯ç¹§åE½½E¿ç¸ºã»Âã»
// 2026-10-07 (S2): off = 現局面の群の表の先頭 (群 * THREAT_DIM)。 base も同じ群なので表が合う。
//   🚨これが無いと群 != 0 の net で群 0 の表の行を足してしまう。
static void apply_dirty_chain(const StateInfo* const* chain, int nchain,
                              Color persp, bool mirror, std::int16_t* acc,
                              std::size_t off) {
	const Net& net = *g_net;
	const std::uint8_t* const nrm = g_norm_sq[persp][mirror ? 1 : 0]; // è±E½£éä¸å§ LUT(hoist)
	const bool i16net = net.threat_is_i16;
	constexpr int DCAP = 256;                 // > P2_DT_COST_MAX ç¸ºEªç¸ºE®ç¸ºE§è³æº¯E³Eª flush ç¸ºè¼ï½ç¸ºEªç¸ºã»
	// 2026-11-08: 分岐なし版 (P2_THREAT_BL) は 1 件先まで投機的に書くので余裕を持たせる。
	//   事前 flush で n_add + n <= DCAP を保証するため、 書き込みは必ず範囲内。
	int add_idx[DCAP + StateInfo::P2_DT_CAP]; int n_add = 0;
	int sub_idx[DCAP + StateInfo::P2_DT_CAP]; int n_sub = 0;
#if defined(P2_THREAT_BL)
	// 🎯2026-11-08: **分岐なしの振り分け** (`-DP2_THREAT_BL`)。
	//   PGO プロファイル (llvm-profdata) で `apply_dirty_chain` が **754 億ブロック**で
	//   最大のホットスポットと判明し、 threat の税 22.3% の本体は**分岐予測ミス**
	//   (cold 20.4 - warm 14.5 = 5.9pt) と測れている。 この関数の 1 件ごとの分岐は
	//     (a) idx < 0 の除外      (視点に写らない特徴。 予測不能)
	//     (b) add / sub の判定    (bit 24。 ほぼ乱数)
	//     (c) DCAP の溢れ判定     (ほぼ取られない)
	//   の 3 つ。 (c) は**ループの外に出し**、 (a)(b) は**両方の配列に投機的に書いて
	//   カウンタだけを進める**ことで消す。 計算結果は従来と完全に同じ (bit-exact)。
	for (int c = 0; c < nchain; ++c) {
		const StateInfo* const s = chain[c];
		const int n = (int)s->p2_dt_count;
		const Eval::P2DirtyThreat* const dt = s->p2_dirty_threats;
		// (c) を外へ: ここで余裕を確保しておけば 1 件ごとの判定が要らない
		if (!i16net && (n_add + n > DCAP || n_sub + n > DCAP)) {
			apply_threat_rows(acc, net, add_idx, n_add, false, off); n_add = 0;
			apply_threat_rows(acc, net, sub_idx, n_sub, true,  off); n_sub = 0;
		}
		for (int i = 0; i < n; ++i) {
			const std::uint32_t d = dt[i].data;
			const int idx = dt_to_index(d, persp, nrm);
			if (i16net) {
				if (idx < 0) continue;
				P2_TSTAT(++g_threat_stats.delta_rows);
				if (d & (1u << 24)) threat_acc_add(acc, net, off + (std::size_t)idx);
				else                threat_acc_sub(acc, net, off + (std::size_t)idx);
				continue;
			}
			// (a)(b) を算術に置き換える。 idx < 0 のときは safe=0 を書くが
			//   カウンタが進まないので次の件で上書きされる = 無害。
			const unsigned ok    = (unsigned)(idx >= 0);
			const unsigned isadd = (d >> 24) & 1u;
			const int      safe  = idx < 0 ? 0 : idx;
			P2_TSTAT(g_threat_stats.delta_rows += (int)ok);
			prefetch_threat_row_i8(net, off + (std::size_t)safe);
			add_idx[n_add] = safe; n_add += (int)(ok & isadd);
			sub_idx[n_sub] = safe; n_sub += (int)(ok & (isadd ^ 1u));
		}
	}
#else
	for (int c = 0; c < nchain; ++c) {
		const StateInfo* const s = chain[c];
		const int n = (int)s->p2_dt_count;
		const Eval::P2DirtyThreat* const dt = s->p2_dirty_threats;
		for (int i = 0; i < n; ++i) {
			const std::uint32_t d = dt[i].data;
			const int idx = dt_to_index(d, persp, nrm);
			if (idx < 0) continue;
			P2_TSTAT(++g_threat_stats.delta_rows);
			if (i16net) {
				if (d & (1u << 24)) threat_acc_add(acc, net, off + (std::size_t)idx);
				else                threat_acc_sub(acc, net, off + (std::size_t)idx);
				continue;
			}
			prefetch_threat_row_i8(net, off + (std::size_t)idx);
			if (d & (1u << 24)) {
				if (n_add == DCAP) { apply_threat_rows(acc, net, add_idx, n_add, false, off); n_add = 0; }
				add_idx[n_add++] = idx;
			} else {
				if (n_sub == DCAP) { apply_threat_rows(acc, net, sub_idx, n_sub, true, off); n_sub = 0; }
				sub_idx[n_sub++] = idx;
			}
		}
	}
#endif
	if (!i16net) {
		apply_threat_rows(acc, net, add_idx, n_add, false, off);
		apply_threat_rows(acc, net, sub_idx, n_sub, true, off);
	}
}

#if defined(P2_THREAT_STATS_BUILD)
// dirty chainç¹§åï½­E£éä¸å§ç¸ºå±âE­Âé¨ãE½¿ãçEç¸ºæãEindexç¸ºE§0ç¸ºEªç¹§å³¨Ââ¬E¦EªThreat accumulatorç¸ºE¯èEEâè³æãEè³Âé¾E´ç¸ºå¶Eç¸²ã»
// è³æ»Eç¸ºE®ç¹§E³ç¹æã»éµâ«æè­Âé©ï½©è¹æ¶ã»è¬ï½¡éå±Eè²ï½¬ç¹§çï½µE±éªäº¥E°ãçç¸²ãE´è³å£¼E®E¹é©¥ã»4EE¹¼hain24ç¸ºE§èæEåªéï½ºè«æ´âç¸ºå±Âã»
static bool dirty_chain_net_zero_for_stats(const StateInfo* const* chain, int nchain,
                                           Color persp, bool mirror) {
        constexpr int CAP = StateInfo::P2_DT_CAP * P2_DT_CHAIN_MAX;
        int idxs[CAP]; int counts[CAP]; int n = 0;
        const std::uint8_t* const nrm = g_norm_sq[persp][mirror ? 1 : 0];
        for (int c = 0; c < nchain; ++c) {
                const StateInfo* const s = chain[c];
                for (int i = 0; i < (int)s->p2_dt_count; ++i) {
                        const std::uint32_t d = s->p2_dirty_threats[i].data;
                        const int idx = dt_to_index(d, persp, nrm);
                        if (idx < 0) continue;
                        int k = 0; while (k < n && idxs[k] != idx) ++k;
                        if (k == n) { idxs[n] = idx; counts[n] = 0; ++n; }
                        counts[k] += (d & (1u << 24)) ? 1 : -1;
                }
        }
        for (int i = 0; i < n; ++i) if (counts[i] != 0) return false;
        return true;
}
#endif

#if defined(P2_THREAT_STATS_BUILD)
// è±E½£éä¸å§è å¾ãEThreat indexç¸ºå¾¡E¸Âç¸ºE¤ç¹§ãEEç¸ºã»âç¹§å³¨Ââ¬E¦Eªaccumulatorç¸ºE¨èE´³ccumulatorç¸ºE¯è³æãEè³Âé¾E´ç¸ºå¶Eç¸²ã»
// O(raw tupleè¬¨E°)ç¸ºE§è»E¤è³å£¹ãç¸ºéªÂâï½°ã»æç¸ºE®source cache fast pathç¸ºE®è­Âèï½§é©ï½©éï½¨éEEEè²ï½¬ç¹§ä¹Âã»
static bool dirty_chain_no_relevant_for_stats(const StateInfo* const* chain, int nchain,
                                              Color persp, bool mirror) {
        const std::uint8_t* const nrm = g_norm_sq[persp][mirror ? 1 : 0];
        for (int c = 0; c < nchain; ++c) {
                const StateInfo* const s = chain[c];
                for (int i = 0; i < (int)s->p2_dt_count; ++i)
                        if (dt_to_index(s->p2_dirty_threats[i].data, persp, nrm) >= 0)
                                return false;
        }
        return true;
}
#endif

// è³E¡éä¹ãç¸ºE® full èå´ï½¨è²E®å±EE1 èæ§­ã»è»ç²å«ç¸ºE§è·æ¢§å¾ç¸ºE«é¦å¾â§ç¸²ã»
//   ç¬ãEitboard è»ç²å«Eattacker èãEç«ãEeffects_from ç«ãEtarget èãEç¸ºE¯éä¹ãç¸ºE«èæE­å¥E ç¸ºEªç¸ºã»Âã»
//     éä¹ãèæE­å¥âç¸ºE®ç¸ºE¯ index è¹ã»as/ds/oriented/normalize_sq)ç¸ºE°ç¸ºä»£âç¸ºE®ç¸ºE§ç¸²ã»
//     1 èæ§­ã»ç¹§E¹ç¹§E­ç¹ï½£ç¹ï½³ç¸ºä¹ï½EBLACK/WHITE è³E¡è­E½¹ç¸ºE® index ç¹§åE½½æï½ç¹§ä¹Âã»
//     è æºæç¸ºE¯ for_each_threat_feature ç¹§å®ï½¦ä¹ãç¸ºæâEç¸ºE« 2 èæ§«å±ç¸ºå±â»ç¸ºã»âEç¹§E¹ç¹§E­ç¹ï½£ç¹ï½³éï½»éï½¨ 2 èãEç¸²ã»
//   i8 net èEçEi16 net ç¸ºE¯ per-perspective é¨çï½·E¯ç¸ºE®ç¸ºE¾ç¸ºE¾)ç¸²ã»

// profile 11èEçç¸ºE®è³E¡éä¹ãfullèå´ï½¨è²E®å±Âã»
// EnemyOnlyç¸ºE§ç¸ºE¯è³Âç¸ºE¤ç¸ºE®attack pairç¸ºæ²LACK/WHITEè³E¡è­E½¹ç¸ºE®è¿ï½¹è ï½´ç¸ºE«è»æ§­âE¹§ä¹ï¼E¸ºE¨ç¸ºE¯ç¸ºEªç¸ºã»Âã»
// è­ï½¢èE¥ã»èï½±è­å³¨ã°ç¹ãEã¡ç¸ºE¯è³Âç¸ºE¤ç¸ºE®nç¸ºE§ib/iwç¹§è²éEè­ãâé§E°ç¹§âï½ç¸ºæºï½ç¸²ã»åèæ´airç¸ºE®pair_base=-1ç¹§ã»
// è¿ãEã»ç¸ºE¸è²ï½¡ç¸ºå±â»ç¸ºå±âªç¸ºã»ÂãE¼E¸ºèãç¸ºE¯1èæ§­ã»board scanç¹§åE½¿æ´â ç¸ºE¤ç¸ºE¤ç¸²â¬E¦ä¹ãè»E¥ç¹èEã£ç¹ãEèç¹§è²ãEé«®E¢ç¸ºå±Âã»
// è·ã»ã°ç¹ãEã¡ç¸ºE¯è æºæç¸ºE¨è·å¾Â§apply_threat_rows_i8(4é¦æ²used SIMD)ç¸ºE§é©ï½©éï½¨ç¸ºå¶Eç¸²ã»
// profile11_color_split_full_kernel:
// EnemyOnlyç¸ºE¯BLACKéä¹ãç¸ºE§ç¸ºE¯WHITE attackerç«éLACK targetç¸²ä»HITEéä¹ãç¸ºE§ç¸ºE¯ç¸ºæ´ã»è¿æ¦ï½¯E¾ç¸ºE°ç¸ºä»£â²è­çæç¸²ã»
// ç¸ºå±âE¸ºå¾â²ç¸ºE¦attackeræ¿¶E²ç¸ºæâEç¸ºE«èE½¾éï½¡æ¿¶E²ç¸ºE¨èE½ºèå«ccumulatorç¹§è²å´è³å£¹E ç¸²âéEæ¿¶E²targetç¸ºE®è¥ï½°è­ï½»ç¹ï½»è³E¡éä¹ãç¸ºE®
// is_excluded_pairè»E¤è³å£¹ã»è³å´ï½¦ââEindexéªè²E®å±Eèç¿«Eç¸²ãé³è ï½´é«®ã»ç²ç¸ºE¨é¦æåEéå±ã»è æºæç¸ºE¨è³æãEè³Âé¾E´ç¸ºå¶Eç¸²ã»
// 4é¦æ²used SIMDç¹èEã£ç¹âEã»éä¹ãè»E¥ç¸ºE®ç¸ºE¾ç¸ºE¾é¯E­è¬âEâE¹§ä¹Âã»
static void add_threat_dual_profile11_fused(const Position& pos, bool mir_b, bool mir_w,
                                            std::int16_t* acc_b, std::int16_t* acc_w) {
        const Net& net = *g_net;
        const Bitboard occ     = pos.pieces();
        const Bitboard nonking = occ & ~pos.pieces(KING);
        const Bitboard black_pieces = nonking & pos.pieces(BLACK);
        const Bitboard white_pieces = nonking & pos.pieces(WHITE);
        const std::uint8_t* const nrm_b = g_norm_sq[BLACK][mir_b ? 1 : 0];
        const std::uint8_t* const nrm_w = g_norm_sq[WHITE][mir_w ? 1 : 0];
        int idx_b[THREAT_BATCH_CAP], idx_w[THREAT_BATCH_CAP];
        int nb = 0, nw = 0;

        // WHITE attacker ç«ãEBLACK target: BLACKéä¹ãç¸ºE°ç¸ºä»£âèã»E¸å¼±âE¹§ä¹Âã»
        Bitboard attackers = white_pieces;
        while (attackers) {
                const Square from = attackers.pop();
                const Piece  pc   = pos.piece_on(from);
                const int    ac   = Threat::tc_from_pt(type_of(pc));
                constexpr int ASB = 1; // WHITE attacker in BLACK-normalized view
                const int pat = Threat::attack_pattern_id(ac, WHITE);
                const int fn  = (int)nrm_b[(int)from];
                const int fo  = g_threat_tables.from_offset[pat][fn];
                const std::uint8_t* const ord = g_threat_tables.attack_order[pat][fn];
                const int* const pb = Threat::THREAT_HANG_SPLIT
                        ? &g_threat_tables.pair_base2[0][ASB * 162 + ac * 18]
                        : &g_threat_tables.pair_base[ASB * 162 + ac * 18];
                Bitboard atk = effects_from(pc, from, occ) & black_pieces;
                while (atk) {
                        const Square to = atk.pop();
                        const Piece tp  = pos.piece_on(to);
                        const int dc = Threat::tc_from_pt(type_of(tp));
                        constexpr int DSB = 0; // BLACK target
                        if (!Threat::is_excluded_pair(ASB, ac, DSB, dc)) {
                                // 標的は black_pieces なので被攻撃駒は BLACK。
                                const int hofs = (Threat::THREAT_HANG_SPLIT
                                        ? P2_HANG0(pos, BLACK, to) : 0) * Threat::NUM_PAIRS;
                                const int idx = pb[hofs + DSB * 9 + dc] + fo + (int)ord[nrm_b[(int)to]];
                                if ((unsigned)idx < (unsigned)Threat::THREAT_DIM_PROFILE) {
                                        prefetch_threat_row_i8(net, (std::size_t)idx);
                                        idx_b[nb++] = idx;
                                        if (nb == THREAT_BATCH_CAP) {
                                                P2_TSTAT(g_threat_stats.full_rows += nb);
                                                apply_threat_rows(acc_b, net, idx_b, nb, false);
                                                nb = 0;
                                        }
                                }
                        }
                }
        }

        // BLACK attacker ç«ãEWHITE target: WHITEéä¹ãç¸ºE°ç¸ºä»£âèã»E¸å¼±âE¹§ä¹Âã»
        attackers = black_pieces;
        while (attackers) {
                const Square from = attackers.pop();
                const Piece  pc   = pos.piece_on(from);
                const int    ac   = Threat::tc_from_pt(type_of(pc));
                constexpr int ASW = 1; // BLACK attacker in WHITE-normalized view
                const int pat = Threat::attack_pattern_id(ac, WHITE);
                const int fn  = (int)nrm_w[(int)from];
                const int fo  = g_threat_tables.from_offset[pat][fn];
                const std::uint8_t* const ord = g_threat_tables.attack_order[pat][fn];
                const int* const pb = Threat::THREAT_HANG_SPLIT
                        ? &g_threat_tables.pair_base2[0][ASW * 162 + ac * 18]
                        : &g_threat_tables.pair_base[ASW * 162 + ac * 18];
                Bitboard atk = effects_from(pc, from, occ) & white_pieces;
                while (atk) {
                        const Square to = atk.pop();
                        const Piece tp  = pos.piece_on(to);
                        const int dc = Threat::tc_from_pt(type_of(tp));
                        constexpr int DSW = 0; // WHITE target in WHITE-normalized view
                        if (!Threat::is_excluded_pair(ASW, ac, DSW, dc)) {
                                // 標的は white_pieces なので被攻撃駒は WHITE。
                                const int hofs = (Threat::THREAT_HANG_SPLIT
                                        ? P2_HANG0(pos, WHITE, to) : 0) * Threat::NUM_PAIRS;
                                const int idx = pb[hofs + DSW * 9 + dc] + fo + (int)ord[nrm_w[(int)to]];
                                if ((unsigned)idx < (unsigned)Threat::THREAT_DIM_PROFILE) {
                                        prefetch_threat_row_i8(net, (std::size_t)idx);
                                        idx_w[nw++] = idx;
                                        if (nw == THREAT_BATCH_CAP) {
                                                P2_TSTAT(g_threat_stats.full_rows += nw);
                                                apply_threat_rows(acc_w, net, idx_w, nw, false);
                                                nw = 0;
                                        }
                                }
                        }
                }
        }
        P2_TSTAT(g_threat_stats.full_rows += nb + nw);
        apply_threat_rows(acc_b, net, idx_b, nb, false);
        apply_threat_rows(acc_w, net, idx_w, nw, false);
}

// ---- S1' (2026-09-11): step-piece pairs by fixed-length candidate lists + predication ----
//   P2_S1A level: 0 = old path only / 1 = pawns / 2 = +knights / 3 = +silvers / 4 (default) = +golds.
//   Each level moves one piece class out of the nested attacker/target loop into a loop whose
//   inner body is a compile-time-unrolled candidate list (no data-dependent trip count, no
//   `!atk` branch). The multiset of emitted pairs is identical to the old path; i16 wrapping
//   adds are order-independent, so the result is bit-exact.
static int s1a_level_env() {
	static const int v = [] {
		const char* e = std::getenv("P2_S1A");
		// 2026-09-11: rdtsc (interleaved, min-of-5) says L1 = -18.5% cycles/call,
		// L3 -15.6%, L2 -12.3%, L4 -6.6%. Pawns are the best case (1 candidate,
		// ord always 0); silver/gold waste work on missed candidates.
		// 2026-11-05: P2_S1A_DEFAULT=k makes level k the build default (same tree for any k = lossless;
		//   measured level 3 = +3.3pt [+1.7, +5.8] CPU, runs/threat_tax/partial_20261104/F.log). Unset = 1 as before.
#if defined(P2_S1A_DEFAULT)
		if (!e || !*e) return (int)P2_S1A_DEFAULT;
#else
		if (!e || !*e) return 1;
#endif
		const int k = std::atoi(e);
		return k < 0 ? 0 : (k > 4 ? 4 : k);
	}();
	return v;
}

// Tables for knight / silver / gold-like (threat class 2/3/4 -> k = cls-2, NMAX 2/5/6).
//   to[acol][k][from][i]                  : raw candidate target square (SQ_NB = none; piece_on(SQ_NB) == NO_PIECE)
//   base[persp][mirror][acol][k][from][i] : from_offset + attack_order in that perspective frame
// Both are derived from g_direct15_pseudo (empty-board attacks) and g_threat_tables, so the
// candidate list is exactly the old path's `pseudo & targets` set for these classes.
struct S1StepTables {
	std::uint8_t  to[COLOR_NB][3][SQ_NB][6];
	std::uint16_t base[2][2][COLOR_NB][3][SQ_NB][6];
	std::int8_t   tc_of_piece[PIECE_NB];
	bool built = false;
};
static S1StepTables g_s1step;
static constexpr int S1_NMAX[3] = { 2, 5, 6 };
static void build_s1step() {
	if (g_s1step.built) return;
	build_direct15_pseudo();
	for (int pc = 0; pc < PIECE_NB; ++pc)
		g_s1step.tc_of_piece[pc] = (std::int8_t)Threat::tc_from_pt(type_of((Piece)pc));
	int bad = 0;
	for (int c = 0; c < COLOR_NB; ++c)
		for (int k = 0; k < 3; ++k) {
			const int cls = k + 2;
			for (int from = 0; from < SQ_NB; ++from) {
				Bitboard bb = g_direct15_pseudo[c][cls][from];
				int cnt = 0;
				while (bb) {
					const Square t = bb.pop();
					if (cnt < S1_NMAX[k]) g_s1step.to[c][k][from][cnt] = (std::uint8_t)t;
					++cnt;
				}
				if (cnt > S1_NMAX[k]) ++bad;
				for (int i = cnt; i < 6; ++i) g_s1step.to[c][k][from][i] = (std::uint8_t)SQ_NB;
				for (int persp = 0; persp < 2; ++persp)
					for (int mir = 0; mir < 2; ++mir) {
						const std::uint8_t* const nrm = g_norm_sq[persp][mir];
						const Color oriented = (persp == BLACK) ? (Color)c : ~(Color)c;
						const int pat = Threat::attack_pattern_id(cls, oriented);
						const int fn  = (int)nrm[from];
						const int fo  = g_threat_tables.from_offset[pat][fn];
						for (int i = 0; i < 6; ++i) {
							const int t = g_s1step.to[c][k][from][i];
							int v = 0;
							if (t != SQ_NB) {
								const int ord = g_threat_tables.attack_order[pat][fn][nrm[t]];
								if (ord == 255) ++bad;
								v = fo + ord;
							}
							g_s1step.base[persp][mir][c][k][from][i] = (std::uint16_t)v;
						}
					}
			}
		}
	if (bad) sync_cout << "info string P2NNUE: S1 step table inconsistent (" << bad << ")" << sync_endl;
	g_s1step.built = true;
}

// One step class for one attacker colour. Inner loop is NMAX (constant) -> fully unrolled,
// no data-dependent branch inside; hit is folded into the index-array advance.
template <int K, int NMAX>
static inline void s1_step_class(const Position& pos, Color acol, Bitboard bb,
                                 bool mir_b, bool mir_w, int* idx_b, int* idx_w, int& n) {
	if (!bb) return;
	constexpr int cls = K + 2;
	const int* const pbb = Threat::THREAT_HANG_SPLIT
		? &g_threat_tables.pair_base2[0][(acol == BLACK ? 0 : 162) + cls * 18]
		: &g_threat_tables.pair_base[(acol == BLACK ? 0 : 162) + cls * 18];
	const int* const pbw = Threat::THREAT_HANG_SPLIT
		? &g_threat_tables.pair_base2[0][(acol == WHITE ? 0 : 162) + cls * 18]
		: &g_threat_tables.pair_base[(acol == WHITE ? 0 : 162) + cls * 18];
	const std::uint8_t  (* const tto)[6] = g_s1step.to[acol][K];
	const std::uint16_t (* const tbb)[6] = g_s1step.base[BLACK][mir_b ? 1 : 0][acol][K];
	const std::uint16_t (* const tbw)[6] = g_s1step.base[WHITE][mir_w ? 1 : 0][acol][K];
	const std::int8_t* const tcp = g_s1step.tc_of_piece;
	int m = n;
	while (bb) {
		const Square from = bb.pop();
		const std::uint8_t*  const to = tto[from];
		const std::uint16_t* const tb = tbb[from];
		const std::uint16_t* const tw = tbw[from];
		for (int i = 0; i < NMAX; ++i) {
			const Piece tp  = pos.piece_on((Square)to[i]);       // SQ_NB -> NO_PIECE
			const int   dc  = (int)tcp[tp];                       // -1: none / king
			const bool  hit = (dc >= 0) && !(Threat::THREAT_EXCLUDE_PAWN_TARGET && dc == 0);
			const int   dcc = hit ? dc : 0;
			const int   dsb = (color_of(tp) == BLACK) ? 0 : 1;
			const int hofs = (Threat::THREAT_HANG_SPLIT
				? P2_HANG0(pos, (Color)dsb, to[i]) : 0) * Threat::NUM_PAIRS;
			idx_b[m] = pbb[hofs + dsb * 9 + dcc]       + (int)tb[i];
			idx_w[m] = pbw[hofs + (1 - dsb) * 9 + dcc] + (int)tw[i];
			m += (int)hit;
		}
	}
	n = m;
}

// ---- 案B (2026-09-16): 桂/銀/金類の方向主導列挙 (runs/threat_tax/planB_design.md) ----
//   駒ごとに利きを引く代わりに、 (手番色, クラス, 方向) ごとに「その方向へ利かせる駒の集合」を 1 回シフトし、
//   当たった升 (hit) だけを回す。 非飛び駒は 空盤面利き == 実利き なので、 方向ごとの
//   shift(src & SRC_OK) & targets の和は旧ループの pseudo & targets と同じ集合、 (from, 方向) が違えば to も違うので多重度も一致。
//   index は同じ from_offset + attack_order を表 (dir_base) に焼いたもの。 i16 wrapping 加算は順序非依存 → bit-exact。
//   ★Bitboard は p[0] = sq 0..62、 p[1] = sq 63..80 (継ぎ目は bit 63)。 Bitboard::operator<< は 64bit ごとの別シフトで
//     継ぎ目を跨ぐ bit が消える (S1a v2-4 の失敗原因) ので使わず、 shift128 で通しシフトする。
//   P2_S1B=<mask> (bit0 桂 / bit1 銀 / bit2 金類)、 既定は P2_S1B_DEFAULT (未定義なら 0 = 旧経路)。
#if !defined(P2_S1B_DEFAULT)
#define P2_S1B_DEFAULT 0
#endif
#if defined(P2_PROBE_BUILD)
// 計測 exe だけ env で切り替える。
static int s1b_mask_env() {
	static const int v = [] {
		const char* e = std::getenv("P2_S1B");
		if (!e || !*e) return (int)P2_S1B_DEFAULT;
		const int k = std::atoi(e);
		return k < 0 ? 0 : (k > 7 ? 7 : k);
	}();
	return v;
}
#else
// 本番はコンパイル時定数 (fable レビュー 09-16: env 読みの guard load と分岐をホットパスから消す)。
static constexpr int s1b_mask_env() { return (int)P2_S1B_DEFAULT; }
#endif
struct DirDF { int df, dr; };
static constexpr int S1B_NDIR[3] = { 2, 5, 6 };   // 桂 / 銀 / 金類
// [acol][cls-2][k] = (筋の差, 段の差)。 先手の前方 = 段 -1。 後手は段の差を反転。 起動時に空盤面利きと全数照合する。
static constexpr DirDF S1B_DIRS[COLOR_NB][3][6] = {
	{ { {-1,-2}, {+1,-2}, {0,0}, {0,0}, {0,0}, {0,0} },
	  { {-1,-1}, { 0,-1}, {+1,-1}, {-1,+1}, {+1,+1}, {0,0} },
	  { {-1,-1}, { 0,-1}, {+1,-1}, {-1, 0}, {+1, 0}, { 0,+1} } },
	{ { {-1,+2}, {+1,+2}, {0,0}, {0,0}, {0,0}, {0,0} },
	  { {-1,+1}, { 0,+1}, {+1,+1}, {-1,-1}, {+1,-1}, {0,0} },
	  { {-1,+1}, { 0,+1}, {+1,+1}, {-1, 0}, {+1, 0}, { 0,-1} } },
};
struct U128 { std::uint64_t p0, p1; };
// 盤内に着地する発地の集合 (シフト前に掛けるので筋の折り返しは起きない)。
static constexpr U128 s1b_src_ok(int df, int dr) {
	U128 r{0, 0};
	for (int f = 0; f < 9; ++f)
		for (int k = 0; k < 9; ++k) {
			if (f + df < 0 || f + df > 8 || k + dr < 0 || k + dr > 8) continue;
			const int sq = f * 9 + k;
			if (sq < 63) r.p0 |= 1ULL << sq; else r.p1 |= 1ULL << (sq - 63);
		}
	return r;
}
// 128bit (論理 bit 0..80、 継ぎ目 63) の通しシフト。 1 <= |d| <= 11。 出力の p0 bit63 は落とす。
static inline U128 shift128(std::uint64_t p0, std::uint64_t p1, int d) {
	constexpr std::uint64_t M63 = 0x7FFFFFFFFFFFFFFFULL;
	if (d > 0) return U128{ (p0 << d) & M63, (p1 << d) | (p0 >> (63 - d)) };
	return U128{ (p0 >> -d) | ((p1 << (63 + d)) & M63), p1 >> -d };
}
struct DirTables {
	std::uint16_t dir_base[2][2][COLOR_NB][3][6][SQ_NB];   // [persp][mir][acol][cls-2][k][from] = from_offset + attack_order。 23,328B
	bool built = false;
	bool ok = false;
};
static DirTables g_dirtab;
static void build_dir_tables() {
	if (g_dirtab.built) return;
	build_direct15_pseudo();
	int bad = 0;
	for (int c = 0; c < COLOR_NB; ++c)
		for (int k = 0; k < 3; ++k) {
			const int cls = k + 2;
			for (int from = 0; from < SQ_NB; ++from) {
				U128 uni{0, 0};
				int cnt = 0;
				for (int di = 0; di < 6; ++di) {
					if (di >= S1B_NDIR[k]) {
						for (int persp = 0; persp < 2; ++persp)
							for (int mir = 0; mir < 2; ++mir) g_dirtab.dir_base[persp][mir][c][k][di][from] = 0;
						continue;
					}
					const DirDF dd = S1B_DIRS[c][k][di];
					const int d = 9 * dd.df + dd.dr;
					const U128 ok = s1b_src_ok(dd.df, dd.dr);
					const bool in = from < 63 ? ((ok.p0 >> from) & 1) : ((ok.p1 >> (from - 63)) & 1);
					// (i) シフト自体: 1 bit を通して from+d の 1 bit になるか (発地外なら空)
					const std::uint64_t b0 = from < 63 ? (1ULL << from) : 0, b1 = from < 63 ? 0 : (1ULL << (from - 63));
					const U128 s = shift128(b0 & ok.p0, b1 & ok.p1, d);
					if (in) {
						const int t = from + d;
						const std::uint64_t e0 = t < 63 ? (1ULL << t) : 0, e1 = t < 63 ? 0 : (1ULL << (t - 63));
						if (s.p0 != e0 || s.p1 != e1) ++bad;
						uni.p0 |= e0; uni.p1 |= e1; ++cnt;
					} else if (s.p0 | s.p1) ++bad;
					for (int persp = 0; persp < 2; ++persp)
						for (int mir = 0; mir < 2; ++mir) {
							int v = 0;
							if (in) {
								const std::uint8_t* const nrm = g_norm_sq[persp][mir];
								const Color oriented = (persp == BLACK) ? (Color)c : ~(Color)c;
								const int pat = Threat::attack_pattern_id(cls, oriented);
								const int fn  = (int)nrm[from];
								const int ord = g_threat_tables.attack_order[pat][fn][nrm[from + d]];
								if (ord == 255) ++bad;
								v = g_threat_tables.from_offset[pat][fn] + ord;
								if (v < 0 || v > 65535) ++bad;
							}
							g_dirtab.dir_base[persp][mir][c][k][di][from] = (std::uint16_t)v;
						}
				}
				// (ii) 方向の和集合 == 空盤面利き、 かつ方向ごとに着地升が異なる (個数一致)
				const Bitboard ps = g_direct15_pseudo[c][cls][from];
				if (uni.p0 != ps.extract64<0>() || uni.p1 != ps.extract64<1>() || cnt != ps.pop_count()) ++bad;
			}
		}
	if (bad) sync_cout << "info string P2NNUE: S1B dir tables inconsistent (" << bad << ")" << sync_endl;
	g_dirtab.ok = (bad == 0);
	g_dirtab.built = true;
}
// 1 方向分。 C/CLS/K はコンパイル時定数なので d と発地マスクは即値になる。
template <int C, int CLS, int K>
[[gnu::always_inline]] static inline void s1b_dir(const Position& pos, std::uint64_t s0, std::uint64_t s1, std::uint64_t t0, std::uint64_t t1,
                           const int* pbb, const int* pbw, const std::uint16_t* tb, const std::uint16_t* tw,
                           int* idx_b, int* idx_w, int& m) {
	constexpr DirDF dd = S1B_DIRS[C][CLS - 2][K];
	constexpr int   d  = 9 * dd.df + dd.dr;
	constexpr U128  ok = s1b_src_ok(dd.df, dd.dr);
	const U128 h = shift128(s0 & ok.p0, s1 & ok.p1, d);
	const std::int8_t* const tcp = g_s1step.tc_of_piece;
	auto emit = [&](int to) {
		const int   from = to - d;
		const Piece tp   = pos.piece_on((Square)to);
		const int   dc   = (int)tcp[tp];                        // targets ⊆ nonking なので >= 0
		const int   dsb  = (color_of(tp) == BLACK) ? 0 : 1;
		// profile 16: 被攻撃駒にひもが無い (持ち主側の利きが 0 本) なら h=1 の行を引く。
		const int   hg   = Threat::THREAT_HANG_SPLIT
			? P2_HANG0(pos, (Color)dsb, to) : 0;
		const int   hofs = hg * Threat::NUM_PAIRS;
		idx_b[m] = pbb[hofs + dsb * 9 + dc]       + (int)tb[from];
		idx_w[m] = pbw[hofs + (1 - dsb) * 9 + dc] + (int)tw[from];
		++m;
	};
	for (std::uint64_t x = h.p0 & t0; x; x &= x - 1) emit((int)LSB64(x));
	for (std::uint64_t x = h.p1 & t1; x; x &= x - 1) emit((int)LSB64(x) + 63);
}
// 1 (手番色, クラス) 分。 書き込みは hit ごとに 1 slot (上限 6 × 駒数)。
template <int C, int CLS>
[[gnu::always_inline]] static inline void s1b_class(const Position& pos, const Bitboard& src, const Bitboard& targets,
                             bool mir_b, bool mir_w, int* idx_b, int* idx_w, int& n) {
	if (!src) return;
	constexpr int k = CLS - 2;
	const int* const pbb = Threat::THREAT_HANG_SPLIT
		? &g_threat_tables.pair_base2[0][(C == BLACK ? 0 : 162) + CLS * 18]
		: &g_threat_tables.pair_base[(C == BLACK ? 0 : 162) + CLS * 18];
	const int* const pbw = Threat::THREAT_HANG_SPLIT
		? &g_threat_tables.pair_base2[0][(C == WHITE ? 0 : 162) + CLS * 18]
		: &g_threat_tables.pair_base[(C == WHITE ? 0 : 162) + CLS * 18];
	const std::uint16_t (* const tb)[SQ_NB] = g_dirtab.dir_base[BLACK][mir_b ? 1 : 0][C][k];
	const std::uint16_t (* const tw)[SQ_NB] = g_dirtab.dir_base[WHITE][mir_w ? 1 : 0][C][k];
	const std::uint64_t s0 = src.extract64<0>(), s1 = src.extract64<1>();
	const std::uint64_t t0 = targets.extract64<0>(), t1 = targets.extract64<1>();
	int m = n;
	s1b_dir<C, CLS, 0>(pos, s0, s1, t0, t1, pbb, pbw, tb[0], tw[0], idx_b, idx_w, m);
	s1b_dir<C, CLS, 1>(pos, s0, s1, t0, t1, pbb, pbw, tb[1], tw[1], idx_b, idx_w, m);
	if constexpr (S1B_NDIR[CLS - 2] > 2) {
		s1b_dir<C, CLS, 2>(pos, s0, s1, t0, t1, pbb, pbw, tb[2], tw[2], idx_b, idx_w, m);
		s1b_dir<C, CLS, 3>(pos, s0, s1, t0, t1, pbb, pbw, tb[3], tw[3], idx_b, idx_w, m);
		s1b_dir<C, CLS, 4>(pos, s0, s1, t0, t1, pbb, pbw, tb[4], tw[4], idx_b, idx_w, m);
	}
	if constexpr (S1B_NDIR[CLS - 2] > 5) {
		s1b_dir<C, CLS, 5>(pos, s0, s1, t0, t1, pbb, pbw, tb[5], tw[5], idx_b, idx_w, m);
	}
	n = m;
}

// ---- rdtsc probe (2026-09-11) ----
//   NPS resolution in this box is only +-1.5%, which cannot tell whether an
//   enumeration optimisation worked. Set P2_RDTSC_OUT=<path> to append
//   calls / total cycles / pairs for add_threat_dual at process exit.
static bool p2rd_on() {
	// ★NPS(A5): 関数内 static のガード変数ロードを消す。 既定値に固定 = bit-exact
#if defined(P2_DIAG)
	static const bool v = [] { const char* e = std::getenv("P2_RDTSC_OUT"); return e && *e; }();
	return v;
#else
	return false;
#endif
}
static unsigned long long g_rd_total = 0, g_rd_calls = 0, g_rd_pairs = 0;
// 2026-10-07: REPEAT の **反復ごと**に cycles を分ける (§4 の cold 4.7pt の切り分け)。
//   1 回目 - 2 回目 = cold (表の L1D ミス + 分岐の初回ミス)。 g_rd_iter は REPEAT ループが設定する。
static int g_rd_iter = 0;
static unsigned long long g_rd_total_i[9] = {}, g_rd_calls_i[9] = {};
struct P2RdDump {
	~P2RdDump() {
		const char* e = std::getenv("P2_RDTSC_OUT");
		if (!e || !*e || g_rd_calls == 0) return;
		std::ofstream f(e, std::ios::app);
		if (!f) return;
		f << "calls " << g_rd_calls
		  << " total_cycles " << g_rd_total
		  << " cycles_per_call " << (double)g_rd_total / (double)g_rd_calls
		  << " pairs " << g_rd_pairs
		  << " pairs_per_call " << (double)g_rd_pairs / (double)g_rd_calls
		  << std::endl;
		for (int i = 0; i < 9; ++i) {
			if (!g_rd_calls_i[i]) continue;
			f << "  iter " << (i + 1) << " calls " << g_rd_calls_i[i]
			  << " cycles_per_call " << (double)g_rd_total_i[i] / (double)g_rd_calls_i[i]
			  << std::endl;
		}
	}
};
static P2RdDump g_rd_dump;
struct P2RdGuard {
	unsigned long long t0; const int* np; bool on;
	explicit P2RdGuard(const int* n) : t0(0), np(n), on(p2rd_on()) { if (on) t0 = __rdtsc(); }
	~P2RdGuard() { if (on) { const unsigned long long d = __rdtsc() - t0;
		g_rd_total += d; ++g_rd_calls; g_rd_pairs += (unsigned long long)(*np);
		const int k = (g_rd_iter >= 0 && g_rd_iter < 9) ? g_rd_iter : 8;
		g_rd_total_i[k] += d; ++g_rd_calls_i[k]; } }
};

#if defined(P2_THREAT_ENUM_STATS_BUILD)
// 2026-11-04: pair id of a BLACK-perspective threat index = last non-excluded pair whose base <= ib
//   (pair_base is cumulative in pair-id order, see Threat::Tables::build). Stats build only.
static void thrstat_pair_impl(int ib) {
	int best = -1;
	for (int p = 0; p < Threat::NUM_PAIRS; ++p) {
		const int bb = g_threat_tables.pair_base[p];
		if (bb != Threat::EXCLUDED_PAIR_BASE && bb <= ib) best = p;
	}
	if (best >= 0) ++g_threat_enum_stats.pair_hist[best];
}
#endif
static void add_threat_dual(const Position& pos, bool mir_b, bool mir_w,
                            std::int16_t* acc_b, std::int16_t* acc_w, std::size_t off = 0) {
        (void)off; // profile 11 / 12 ビルドでは未使用 (群化 net は load で拒否する)
	const Net& net = *g_net;
                if constexpr (Threat::THREAT_REPRESENTATIVE_ROWS) {
                // profile 12ç¸ºE¯éä¹ãç¸ºæâEç¸ºE«è¿¢E¬é¶ä¹ï¼ ç¸ºE¦è³è´ï½½ã»é¦å¾ï½é©ï½¸ç¸ºE¶ç¸²ãE¨éä¹ãç¸ºE®é¦æåEéå±ã»
                // add_threat_perspectiveèã»ã»4é¦æ²used SIMDç¸ºE§è³æº¯E¡å¾ï¼E¹§å¾ï½ç¸²ã»
                add_threat_perspective(pos, BLACK, mir_b, acc_b);
                add_threat_perspective(pos, WHITE, mir_w, acc_w);
                return;
        }
if constexpr (Threat::THREAT_PROFILE_ID == 11) {
                add_threat_dual_profile11_fused(pos, mir_b, mir_w, acc_b, acc_w);
                return;
        }
	const Bitboard occ     = pos.pieces();
	const Bitboard nonking = occ & ~pos.pieces(KING);   // è³E¡éE¾åèã»
	// target èï½´ mask(no-pawn-target profile ç¸ºE§ç¸ºE¯è­ï½ªè¬èEã»è±E½©ç¹§ãåèãEç¸²ã»
// 2026-11-05: profile 17 (enemy-only-all) uses the one-perspective enumeration path written as cost probe 2.
//   Only as==1 && ds==0 pairs exist in the table, and that is exactly what path 2 emits (attacker colour != perspective).
#if defined(P2_THREAT_PROFILE) && P2_THREAT_PROFILE == 17 && !defined(P2_THREAT_PARTIAL)
#define P2_THREAT_PARTIAL 2
#endif
#if defined(P2_THREAT_PROFILE) && P2_THREAT_PROFILE == 17 && P2_THREAT_PARTIAL != 2
#error profile 17 requires the one-perspective enumeration (P2_THREAT_PARTIAL == 2)
#endif
#if defined(P2_THREAT_PARTIAL)
	// 2026-11-04 partial-threat COST PROBE (engine only, full net layout kept; values differ from the trained net).
	//   1 = cross-side pairs only (attacker colour != target colour), both perspectives
	//   2 = cross-side pairs, enemy-attacker perspective only (one row per pair)
	//   3 = no pawn targets
#if P2_S1B_DEFAULT != 0
#error P2_THREAT_PARTIAL probe assumes the old loop (P2_S1B_DEFAULT=0)
#endif
#endif
#if defined(P2_THREAT_PARTIAL) && P2_THREAT_PARTIAL == 3
	const Bitboard targets = nonking & ~pos.pieces(PAWN);
#else
	const Bitboard targets = Threat::THREAT_EXCLUDE_PAWN_TARGET ? (nonking & ~pos.pieces(PAWN)) : nonking;
#endif
#if defined(P2_THREAT_PARTIAL) && (P2_THREAT_PARTIAL == 1 || P2_THREAT_PARTIAL == 2)
	const Bitboard targets_of[COLOR_NB] = { targets & pos.pieces(WHITE), targets & pos.pieces(BLACK) };
#elif defined(P2_THREAT_PARTIAL) && P2_THREAT_PARTIAL == 5
	//   5 = FULL pair set, each pair emitted into one perspective only (routed by attacker colour)
	const Bitboard targets_of[COLOR_NB] = { targets, targets };
#elif defined(P2_THREAT_PARTIAL) && P2_THREAT_PARTIAL == 4
	//   4 = all cross-side pairs + same-side pairs whose target is S/G/B/R/H/D (no same-side P/L/N protection)
	const Bitboard cheap = pos.pieces(PAWN) | pos.pieces(LANCE) | pos.pieces(KNIGHT);
	const Bitboard targets_of[COLOR_NB] = { targets & ~(pos.pieces(BLACK) & cheap), targets & ~(pos.pieces(WHITE) & cheap) };
#endif
#if defined(P2_THREAT_PARTIAL) && (P2_THREAT_PARTIAL == 2 || P2_THREAT_PARTIAL == 5)
	int nb = 0, nw = 0;
#endif
	const std::uint8_t* const nrm_b = g_norm_sq[BLACK][mir_b ? 1 : 0];
	const std::uint8_t* const nrm_w = g_norm_sq[WHITE][mir_w ? 1 : 0];
	// 2026-09-15 (A3): 配列長 320 / flush 256。 根拠は THREAT_DUAL_IDX_CAP の定義コメント (書き込み index < 256)。
	int idx_b[THREAT_DUAL_IDX_CAP], idx_w[THREAT_DUAL_IDX_CAP];   // prelude (< 234 slot) は flush しない
	int n = 0;
	P2RdGuard _rdg(&n);
#if defined(P2_PROBE_BUILD)
	++g_pc_dual;
#endif
	THRSTAT(calls);
	// ---- V1 (2026-09-11): pawn pairs, predicated (no inner loop, no `!atk` branch) ----
	//   A pawn has exactly one candidate target square (from-1 for BLACK, from+1 for WHITE),
	//   so we read that square, form hit = (non-king piece there), write both indices
	//   unconditionally and advance n by hit. Pawns are ~40% of attacker iterations and the
	//   main source of the 47.9% empty-attack branch. If NPS moves, the enumeration cost is
	//   dominated by variable-trip-loop mispredicts (the premise of D6 / S1').
	//   Multiset of pairs is unchanged: a pawn pattern has one target per from, so
	//   attack_order is always 0 and idx = pair_base + from_offset. i16 wrapping add
	//   is order-independent -> bit-exact. P2_S1A=0 restores the old path (A/B).
	const int  s1_lvl = s1a_level_env();
	const bool v1_on  = s1_lvl >= 1;
#if defined(P2_PROBE_BUILD)
	// 計測専用 REP_PAWN: V1 ブロックを k 回。 最後の回以外は n を戻す (書き込み index は常に < 18+n0 < 256)。
	for (int _rp = probe_rep_pawn(); _rp > 0; --_rp) { const int _n0p = n;
#endif
	if (v1_on) {
		for (int ci = 0; ci < 2; ++ci) {
			const Color acol = (Color)ci;
			Bitboard pawns = pos.pieces(acol, PAWN);
			if (!pawns) continue;
			const int delta = (acol == BLACK) ? -1 : +1;   // BLACK pawn attacks rank-1 = sq-1
			const int patb = Threat::attack_pattern_id(0, acol);
			const int patw = Threat::attack_pattern_id(0, ~acol);
			// profile 16 (hanging-split): pair_base2[0] を起点にし、 h=1 は +NUM_PAIRS の位置を引く。
			// それ以外の profile では h が常に 0 なので添字は現行と同一に畳まれる。
			const int* const pbb = Threat::THREAT_HANG_SPLIT
				? &g_threat_tables.pair_base2[0][(acol == BLACK ? 0 : 162) + 0 * 18]
				: &g_threat_tables.pair_base[(acol == BLACK ? 0 : 162) + 0 * 18];
			const int* const pbw = Threat::THREAT_HANG_SPLIT
				? &g_threat_tables.pair_base2[0][(acol == WHITE ? 0 : 162) + 0 * 18]
				: &g_threat_tables.pair_base[(acol == WHITE ? 0 : 162) + 0 * 18];
			const int* const fob = g_threat_tables.from_offset[patb];
			const int* const fow = g_threat_tables.from_offset[patw];
			while (pawns) {
				const Square from = pawns.pop();
				const Square to   = (Square)((int)from + delta);   // unpromoted pawn is never on the last rank
				const Piece  tp   = pos.piece_on(to);
				const int    dc   = Threat::tc_from_pt(type_of(tp));   // NO_PIECE / KING -> -1
#if defined(P2_THREAT_PARTIAL) && (P2_THREAT_PARTIAL == 1 || P2_THREAT_PARTIAL == 2)
				const bool   hit  = (dc >= 0) && (color_of(tp) != acol);
#elif defined(P2_THREAT_PARTIAL) && P2_THREAT_PARTIAL == 3
				const bool   hit  = (dc > 0);
#elif defined(P2_THREAT_PARTIAL) && P2_THREAT_PARTIAL == 4
				const bool   hit  = (dc >= 0) && (color_of(tp) != acol || dc >= 3);
#else
				const bool   hit  = (dc >= 0) && !(Threat::THREAT_EXCLUDE_PAWN_TARGET && dc == 0);
#endif
				const int    dcc  = hit ? dc : 0;
				const int    dsb  = (color_of(tp) == BLACK) ? 0 : 1;
				// h = 1 は「被攻撃駒にひもが無い」= その升に持ち主側の利きが 0 本。
				// hit=false のとき to は空升だが、 読んでも無害 (index は n += hit で捨てられる)。
				const int    h    = Threat::THREAT_HANG_SPLIT
					? P2_HANG0(pos, (Color)dsb, to) : 0;
				const int    hofs = h * Threat::NUM_PAIRS;
				const int    ib   = pbb[hofs + dsb * 9 + dcc]       + fob[nrm_b[(int)from]];
				const int    iw   = pbw[hofs + (1 - dsb) * 9 + dcc] + fow[nrm_w[(int)from]];
#if defined(P2_THREAT_PARTIAL) && (P2_THREAT_PARTIAL == 2 || P2_THREAT_PARTIAL == 5)
				if (acol == WHITE) { idx_b[nb] = ib; nb += (int)hit; } else { idx_w[nw] = iw; nw += (int)hit; }
#else
				idx_b[n] = ib; idx_w[n] = iw;
				n += (int)hit;
#endif
				THRSTAT(attackers); if (hit) { THRSTAT(targets); THRSTAT_PAIR(ib); }
			}
		}
	}
#if defined(P2_PROBE_BUILD)
	if (_rp > 1) n = _n0p; }
#endif
#if defined(P2_THREAT_ENUM_STATS_BUILD)
	// 転置 (標的ごとに attackers_to) が割に合うかの事前計測。
	//   LONG_EFFECT の board_effect は do_move で差分維持されている利き数。
	//   両色とも 0 の升には attacker が居ないので、 転置形なら magic 参照ゼロで飛ばせる。
	if (threat_stats_on()) {
		Bitboard tg = targets;
		while (tg) {
			const Square t = tg.pop();
			++g_threat_enum_stats.tgt_total;
			if (!pos.effected_to(BLACK, t) && !pos.effected_to(WHITE, t))
				++g_threat_enum_stats.tgt_zero_eff;
			// 転置版 (§3.2) は long_effect16 の方向 bit からレイの最初の駒を逆引きする。
			// le16 == 0 の升は飛び駒に射られていないので逆引き自体が不要 = 回す標的はこの数。
			if (pos.long_effect.long_effect16(t) != 0)
				++g_threat_enum_stats.tgt_long_eff;
		}
	}
#endif
	// 2026-09-16 案B: 桂/銀/金類を方向主導で。 有効なクラスは S1' より優先 (二重に出さない)。
	const int s1b = s1b_mask_env();
	if (s1b & 1) {
		s1b_class<BLACK, 2>(pos, pos.pieces(BLACK, KNIGHT), targets, mir_b, mir_w, idx_b, idx_w, n);
		s1b_class<WHITE, 2>(pos, pos.pieces(WHITE, KNIGHT), targets, mir_b, mir_w, idx_b, idx_w, n);
	}
	if (s1b & 2) {
		s1b_class<BLACK, 3>(pos, pos.pieces(BLACK, SILVER), targets, mir_b, mir_w, idx_b, idx_w, n);
		s1b_class<WHITE, 3>(pos, pos.pieces(WHITE, SILVER), targets, mir_b, mir_w, idx_b, idx_w, n);
	}
	if (s1b & 4) {
		s1b_class<BLACK, 4>(pos, pos.pieces(BLACK, GOLDS), targets, mir_b, mir_w, idx_b, idx_w, n);
		s1b_class<WHITE, 4>(pos, pos.pieces(WHITE, GOLDS), targets, mir_b, mir_w, idx_b, idx_w, n);
	}
	// S1' levels 2..4: knights / silvers / gold-likes through fixed-length candidate lists.
	if (s1b == 0 && s1_lvl >= 2) {
		s1_step_class<0, 2>(pos, BLACK, pos.pieces(BLACK, KNIGHT), mir_b, mir_w, idx_b, idx_w, n);
		s1_step_class<0, 2>(pos, WHITE, pos.pieces(WHITE, KNIGHT), mir_b, mir_w, idx_b, idx_w, n);
	}
	if (s1b == 0 && s1_lvl >= 3) {
		s1_step_class<1, 5>(pos, BLACK, pos.pieces(BLACK, SILVER), mir_b, mir_w, idx_b, idx_w, n);
		s1_step_class<1, 5>(pos, WHITE, pos.pieces(WHITE, SILVER), mir_b, mir_w, idx_b, idx_w, n);
	}
	if (s1b == 0 && s1_lvl >= 4) {
		s1_step_class<2, 6>(pos, BLACK, pos.pieces(BLACK, GOLDS), mir_b, mir_w, idx_b, idx_w, n);
		s1_step_class<2, 6>(pos, WHITE, pos.pieces(WHITE, GOLDS), mir_b, mir_w, idx_b, idx_w, n);
	}
#if defined(P2_THREAT_PARTIAL) && (P2_THREAT_PARTIAL == 2 || P2_THREAT_PARTIAL == 5)
	if (nb >= THREAT_DUAL_BATCH_CAP / 2) { apply_threat_rows(acc_b, net, idx_b, nb, false, off); nb = 0; }
	if (nw >= THREAT_DUAL_BATCH_CAP / 2) { apply_threat_rows(acc_w, net, idx_w, nw, false, off); nw = 0; }
#else
	if (n >= THREAT_DUAL_BATCH_CAP / 2) {   // rare (prelude >= 128 pairs); keeps the main loop's == CAP flush valid
		P2_TSTAT(g_threat_stats.full_rows += 2 * n);
		apply_threat_rows(acc_b, net, idx_b, n, false, off);
		apply_threat_rows(acc_w, net, idx_w, n, false, off);
		n = 0;
	}
#endif
	// Old nested loop handles whatever classes were not taken above (always the sliders).
	Bitboard excl(0, 0);
	if (s1_lvl >= 1) excl |= pos.pieces(PAWN);
	if (s1b != 0) {
		if (s1b & 1) excl |= pos.pieces(KNIGHT);
		if (s1b & 2) excl |= pos.pieces(SILVER);
		if (s1b & 4) excl |= pos.pieces(GOLDS);
	} else {
		if (s1_lvl >= 2) excl |= pos.pieces(KNIGHT);
		if (s1_lvl >= 3) excl |= pos.pieces(SILVER);
		if (s1_lvl >= 4) excl |= pos.pieces(GOLDS);
	}
	Bitboard attackers = nonking & ~excl;
#if defined(P2_THREAT_SCAN_ORDER) && !defined(P2_PROBE_BUILD)
	// 2026-11-05 lossless scan order: visit attackers class by class instead of in square order.
	//   The pair multiset is unchanged (i16 wrapping adds are order independent) => bit-exact.
	//   Probe build measured the 2-pass split (step pieces, then sliders) at +3.2pt [+0.8, +4.6] CPU, same tree
	//   (runs/threat_tax/partial_20261104/C.log): uniform class per pass = predictable is_slider / inner trip counts.
#if P2_THREAT_SCAN_ORDER == 1
	const Bitboard _stepset = attackers & (pos.pieces(KNIGHT) | pos.pieces(SILVER) | pos.pieces(GOLDS));
	const Bitboard _pass[2] = { _stepset, attackers & ~_stepset };
	for (int _ps = 0; _ps < 2; ++_ps) { Bitboard attackers = _pass[_ps];
#else
	const Bitboard _pass[8] = { attackers & pos.pieces(KNIGHT), attackers & pos.pieces(SILVER), attackers & pos.pieces(GOLDS),
	                            attackers & pos.pieces(LANCE),  attackers & pos.pieces(BISHOP), attackers & pos.pieces(ROOK),
	                            attackers & pos.pieces(HORSE),  attackers & pos.pieces(DRAGON) };
	for (int _ps = 0; _ps < 8; ++_ps) { Bitboard attackers = _pass[_ps];
#endif
#endif
#if defined(P2_PROBE_BUILD)
	// 2026-10-07 LE16SCAN: 転置版 (§3.2) が払う「標的を回して le16 を読む」追加コストの天井。
	//   列挙そのものは一切変えない (出力 pair は不変 = bit-exact な時間差)。
	if (probe_enum_le16scan()) {
		unsigned sink = 0;
		Bitboard _tg = targets;
		while (_tg) sink += (unsigned)__builtin_popcount(pos.long_effect.long_effect16(_tg.pop()));
		g_probe_le16_sink = sink;
	}
	// 2026-10-07 PREWARM: attacker をもう一周して、 読むことになる表の行を先に prefetch する。
	//   後ろの attacker ほど待ち時間が隠れる。 **prefetch のみ = bit-exact**。
	if (probe_enum_prewarm()) {
		Bitboard _pw = attackers;
		while (_pw) {
			const Square _f = _pw.pop();
			const Piece  _p = pos.piece_on(_f);
			const int    _a = Threat::tc_from_pt(type_of(_p));
			const Color  _c = color_of(_p);
			__builtin_prefetch(&g_direct15_pseudo[_c][_a][_f]);
			const int _pb = Threat::attack_pattern_id(_a, _c);
			const int _pw2 = Threat::attack_pattern_id(_a, ~_c);
			const int _fnb = (int)nrm_b[(int)_f], _fnw = (int)nrm_w[(int)_f];
			__builtin_prefetch(&g_threat_tables.from_offset[_pb][_fnb]);
			__builtin_prefetch(&g_threat_tables.from_offset[_pw2][_fnw]);
			__builtin_prefetch(g_threat_tables.attack_order[_pb][_fnb]);
			__builtin_prefetch(g_threat_tables.attack_order[_pw2][_fnw]);
		}
	}
#endif
#if defined(P2_PROBE_BUILD)
	// 計測専用 REP_STEP / REP_SLIDER / SPLIT: 旧ループを 桂銀金類 と それ以外 の 2 パスに分け、 各パスを k 回。
	//   最後の回以外は n を回す前に戻す。 途中で flush (n==CAP → 0) が起きても戻し先 _n0 < 256 なので書き込みは範囲内。
	//   ENUMONLY 併用なので二重 flush は acc に影響しない。 既定 (全 k=1, SPLIT 無し) は 1 パス = 本番と同じ順序。
	const int  _kst = probe_rep_step(), _ksl = probe_rep_slider();
	const bool _split = probe_enum_split() || _kst > 1 || _ksl > 1;
	const Bitboard _stepset = attackers & (pos.pieces(KNIGHT) | pos.pieces(SILVER) | pos.pieces(GOLDS));
	const Bitboard _pass_set[2] = { _split ? _stepset : attackers, attackers & ~_stepset };
	const int      _pass_rep[2] = { _split ? _kst : 1, _ksl };
	for (int _ps = 0; _ps < (_split ? 2 : 1); ++_ps)
	for (int _r = _pass_rep[_ps]; _r > 0; --_r) { const int _n0 = n; Bitboard attackers = _pass_set[_ps];
#endif
	while (attackers) {
		THRSTAT(attackers);
		const Square from = attackers.pop();
		const Piece  pc   = pos.piece_on(from);
		const int    ac   = Threat::tc_from_pt(type_of(pc));  // King é«¯E¤èåE¸ã»ç«ãEè¶E¸ç¸ºE« >= 0
		const Color  acol = color_of(pc);
		// ��2026-09-09: ������ 1 ���������Ă��Ȃ� attacker (�����ő�) ��
		//   index �p�� hoist 8 ���ۑ��Ȃ̂ŁA atk ���ɍ���đ��� continue ����B
		//   �o��y�A�̑��d�W���͕s�� = bit-exact�B
		// 2026-09-10: 空盤面利きで magic bitboard を省く。 実測 (P2_THREAT_ENUM_STATS) で
		//   attacker の 79.4% は非飛び駒 (歩桂銀金玉) = 空盤面利き == 実利き なので magic が不要。
		//   飛び駒は 実利き ⊆ 空盤面利き なので、 空盤面利き & targets が空なら magic ごと省ける。
		//   どちらも集合として厳密に等価なので **bit-exact**。
		//   magic 呼び出し 31.29 -> 6.35 回/eval (-79.7%)。 Direct15 で -53〜-83% の実績がある手法。
		const bool is_slider = (ac == 1 || (ac >= 5 && ac <= 8));   // 香/角/飛/馬/竜
#if defined(P2_THREAT_PARTIAL) && (P2_THREAT_PARTIAL == 1 || P2_THREAT_PARTIAL == 2 || P2_THREAT_PARTIAL == 4 || P2_THREAT_PARTIAL == 5)
		Bitboard atk = g_direct15_pseudo[acol][ac][from] & targets_of[acol];
		if (atk && is_slider) atk = direct15_slider_effect(ac, acol, from, occ) & targets_of[acol];
#else
		Bitboard atk = g_direct15_pseudo[acol][ac][from] & targets;
		if (atk && is_slider) atk = direct15_slider_effect(ac, acol, from, occ) & targets;
#endif
		THRSTAT(effects_calls); if (!atk) THRSTAT(atk_empty); THRSTAT_AC(ac, atk);
		if (!atk) continue;
		// attacker è³æ¦ï½¤å³¨ã» index è¬ä»ãEç¹§å®ï½¦ä¹ãç¸ºæâEç¸ºE« hoist(index() ç¸ºE¨è·å¾¡E¸ÂéèE¡ã»ç¸²ã»
		const int    patb = Threat::attack_pattern_id(ac, acol);
		const int    patw = Threat::attack_pattern_id(ac, ~acol);
		const int    fnb  = (int)nrm_b[(int)from];
		const int    fnw  = (int)nrm_w[(int)from];
#if defined(P2_PROBE_BUILD)
		// 計測専用: NOTABLES では表の読み先を L1 に載る固定の小配列にする (ENUMONLY 併用なので idx の値は使われない)。
		const bool _pnt = probe_enum_notables();
		// 2026-10-07 SETUPCONST: 表は**本物のまま**、 行だけ固定にする。 pat/fn の計算と
		//   from_offset / attack_order の行引きが死んで消える = ①表化の天井 (上界)。
		//   🚨NOTABLES と混ぜない (NOTABLES が勝つ順で書く)。
		const bool _psc = !_pnt && probe_enum_setupconst();
		const int* const pbb = _pnt ? g_probe_pair_base_zero
			: &g_threat_tables.pair_base[(_psc ? 0 : ((acol == BLACK ? 0 : 162) + ac * 18))];
		const int* const pbw = _pnt ? g_probe_pair_base_zero
			: &g_threat_tables.pair_base[(_psc ? 0 : ((acol == WHITE ? 0 : 162) + ac * 18))];
		const int    fob = (_pnt || _psc) ? 0 : g_threat_tables.from_offset[patb][fnb];
		const int    fow = (_pnt || _psc) ? 0 : g_threat_tables.from_offset[patw][fnw];
		const std::uint8_t* const ordb = _pnt ? g_probe_ord_zero
			: (_psc ? g_threat_tables.attack_order[0][0] : g_threat_tables.attack_order[patb][fnb]);
		const std::uint8_t* const ordw = _pnt ? g_probe_ord_zero
			: (_psc ? g_threat_tables.attack_order[0][0] : g_threat_tables.attack_order[patw][fnw]);
#else
		const int* const pbb = &g_threat_tables.pair_base[(acol == BLACK ? 0 : 162) + ac * 18];
		const int* const pbw = &g_threat_tables.pair_base[(acol == WHITE ? 0 : 162) + ac * 18];
		const int    fob = g_threat_tables.from_offset[patb][fnb];
		const int    fow = g_threat_tables.from_offset[patw][fnw];
		const std::uint8_t* const ordb = g_threat_tables.attack_order[patb][fnb];
		const std::uint8_t* const ordw = g_threat_tables.attack_order[patw][fnw];
#endif
		while (atk) {
			const Square to  = atk.pop();
			const Piece  tp  = pos.piece_on(to);
			const int    dc  = Threat::tc_from_pt(type_of(tp));
			const int    dsb = (color_of(tp) == BLACK) ? 0 : 1;   // WHITE éä¹ãç¸ºE¯ 1 - dsb
#if defined(P2_THREAT_PARTIAL) && (P2_THREAT_PARTIAL == 2 || P2_THREAT_PARTIAL == 5)
			if (acol == WHITE) {
				const int ib = pbb[dsb * 9 + dc] + fob + (int)ordb[nrm_b[(int)to]];
				idx_b[nb++] = ib;
				if (nb == THREAT_DUAL_BATCH_CAP) { apply_threat_rows(acc_b, net, idx_b, nb, false, off); nb = 0; }
			} else {
				const int iw = pbw[(1 - dsb) * 9 + dc] + fow + (int)ordw[nrm_w[(int)to]];
				idx_w[nw++] = iw;
				if (nw == THREAT_DUAL_BATCH_CAP) { apply_threat_rows(acc_w, net, idx_w, nw, false, off); nw = 0; }
			}
#else
			const int    ib  = pbb[dsb * 9 + dc]       + fob + (int)ordb[nrm_b[(int)to]];
			const int    iw  = pbw[(1 - dsb) * 9 + dc] + fow + (int)ordw[nrm_w[(int)to]];
			prefetch_threat_row_i8(net, (std::size_t)ib);
			prefetch_threat_row_i8(net, (std::size_t)iw);
			THRSTAT(targets); THRSTAT_PAIR(ib);
			THRSTAT(slider_hits);   // 旧ループに残るのは飛び駒だけ (歩 V1 / 桂銀金 planB で抜けている)
			idx_b[n] = ib; idx_w[n] = iw; ++n;
			if (n == THREAT_DUAL_BATCH_CAP) {
				P2_TSTAT(g_threat_stats.full_rows += 2 * n);
				apply_threat_rows(acc_b, net, idx_b, n, false, off);
				apply_threat_rows(acc_w, net, idx_w, n, false, off);
				n = 0;
			}
#endif
		}
	}
#if defined(P2_PROBE_BUILD)
	if (_r > 1) n = _n0; }
#endif
#if defined(P2_THREAT_SCAN_ORDER) && !defined(P2_PROBE_BUILD)
	}
#endif
#if defined(P2_THREAT_PARTIAL) && (P2_THREAT_PARTIAL == 2 || P2_THREAT_PARTIAL == 5)
	apply_threat_rows(acc_b, net, idx_b, nb, false, off);
	apply_threat_rows(acc_w, net, idx_w, nw, false, off);
#else
	P2_TSTAT(g_threat_stats.full_rows += 2 * n);
	apply_threat_rows(acc_b, net, idx_b, n, false, off);
	apply_threat_rows(acc_w, net, idx_w, n, false, off);
#endif
}

// ---- chain éï½£è±ï½ºé¨åæ£¡ ----
// base == nullptr ç¸ºEªç¹§ã»full èå´ï½¨è²E®å±ÂãEç¸ºã»ãç¸ºEªç¸ºä»£Eç¸ºE° base->p2_threat_acc ç¸ºä¹ï½Echain è¾E®è»ã»Eé©ï½©éï½¨ç¸²ã»
// Profile 13: profile-11 EnemyOnly enumeration, but each feature reads a compact
// 16-byte factor row and contributes directly to a scalar output branch.
static void add_threat_factor_dual_profile13(const Position& pos, bool mir_b, bool mir_w,
                                             std::int32_t* acc_b, std::int32_t* acc_w) {
	const Net& net = *g_net;
	std::memset(acc_b, 0, sizeof(std::int32_t) * THREAT_FACTOR_DIM);
	std::memset(acc_w, 0, sizeof(std::int32_t) * THREAT_FACTOR_DIM);
	const Bitboard occ = pos.pieces();
	const Bitboard nonking = occ & ~pos.pieces(KING);
	const Bitboard black_pieces = nonking & pos.pieces(BLACK);
	const Bitboard white_pieces = nonking & pos.pieces(WHITE);
	const std::uint8_t* const nrm_b = g_norm_sq[BLACK][mir_b ? 1 : 0];
	const std::uint8_t* const nrm_w = g_norm_sq[WHITE][mir_w ? 1 : 0];
	auto add_row = [&](std::int32_t* acc, int idx) {
		const std::int8_t* const row = threat_factor_row(net, (std::size_t)idx);
#if defined(USE_AVX2)
		static_assert(THREAT_FACTOR_DIM == 16, "profile 13 AVX2 path assumes factor dim 16");
		const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row));
		const __m256i lo = _mm256_cvtepi8_epi32(bytes);
		const __m256i hi = _mm256_cvtepi8_epi32(_mm_srli_si128(bytes, 8));
		const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc));
		const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 8));
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(acc), _mm256_add_epi32(a0, lo));
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 8), _mm256_add_epi32(a1, hi));
#else
		for (int d = 0; d < THREAT_FACTOR_DIM; ++d) acc[d] += (std::int32_t)row[d];
#endif
	};
	auto collect = [&](Bitboard attackers, Bitboard targets, const std::uint8_t* nrm, std::int32_t* acc) {
		while (attackers) {
			const Square from = attackers.pop();
			const Piece pc = pos.piece_on(from);
			const int ac = Threat::tc_from_pt(type_of(pc));
			constexpr int AS = 1, DS = 0;
			const int pat = Threat::attack_pattern_id(ac, WHITE);
			const int fn = (int)nrm[(int)from];
			const int fo = g_threat_tables.from_offset[pat][fn];
			const std::uint8_t* const ord = g_threat_tables.attack_order[pat][fn];
			const int* const pb = &g_threat_tables.pair_base[AS * 162 + ac * 18];
			Bitboard atk = effects_from(pc, from, occ) & targets;
			while (atk) {
				const Square to = atk.pop();
				const int dc = Threat::tc_from_pt(type_of(pos.piece_on(to)));
				if (!Threat::is_excluded_pair(AS, ac, DS, dc)) {
					const int idx = pb[DS * 9 + dc] + fo + (int)ord[nrm[(int)to]];
					if ((unsigned)idx < (unsigned)THREAT_DIM) add_row(acc, idx);
				}
			}
		}
	};
	collect(white_pieces, black_pieces, nrm_b, acc_b);
	collect(black_pieces, white_pieces, nrm_w, acc_w);
}

static std::int64_t threat_factor_projected_perspective_profile13(const Position& pos, Color persp,
                                                                   bool mirror, int bucket) {
	const std::int32_t* const proj = g_net->threat_factor_proj.data()
		+ (std::size_t)bucket * THREAT_DIM;
	const Bitboard occ = pos.pieces();
	const Bitboard nonking = occ & ~pos.pieces(KING);
	const Color enemy = (persp == BLACK) ? WHITE : BLACK;
	Bitboard attackers = nonking & pos.pieces(enemy);
	const Bitboard targets = nonking & pos.pieces(persp);
	const std::uint8_t* const nrm = g_norm_sq[persp][mirror ? 1 : 0];
	constexpr int AS = 1, DS = 0;
	std::int64_t sum = 0;
	while (attackers) {
		const Square from = attackers.pop();
		const Piece pc = pos.piece_on(from);
		const int ac = Threat::tc_from_pt(type_of(pc));
		const int pat = Threat::attack_pattern_id(ac, WHITE);
		const int fn = (int)nrm[(int)from];
		const int fo = g_threat_tables.from_offset[pat][fn];
		const std::uint8_t* const ord = g_threat_tables.attack_order[pat][fn];
		const int* const pb = &g_threat_tables.pair_base[AS * 162 + ac * 18];
		Bitboard tos = effects_from(pc, from, occ) & targets;
		while (tos) {
			const Square to = tos.pop();
			const int dc = Threat::tc_from_pt(type_of(pos.piece_on(to)));
			if (!Threat::is_excluded_pair(AS, ac, DS, dc)) {
				const int idx = pb[DS * 9 + dc] + fo + (int)ord[nrm[(int)to]];
				if ((unsigned)idx < (unsigned)THREAT_DIM) sum += proj[idx];
			}
		}
	}
	return sum;
}

static std::int64_t threat_factor_projected_perspective_profile13_shared(const Position& pos, Color persp,
                                                                          bool mirror, int bucket,
                                                                          const Bitboard occ, Bitboard attackers,
                                                                          const Bitboard targets) {
        const std::int32_t* const proj = g_net->threat_factor_proj.data()
                + (std::size_t)bucket * THREAT_DIM;
        const std::uint8_t* const nrm = g_norm_sq[persp][mirror ? 1 : 0];
	constexpr int AS = 1, DS = 0;
	std::int64_t sum = 0;
	while (attackers) {
		const Square from = attackers.pop();
		const Piece pc = pos.piece_on(from);
		const int ac = Threat::tc_from_pt(type_of(pc));
		const int pat = Threat::attack_pattern_id(ac, WHITE);
		const int fn = (int)nrm[(int)from];
		const int fo = g_threat_tables.from_offset[pat][fn];
		const std::uint8_t* const ord = g_threat_tables.attack_order[pat][fn];
		const int* const pb = &g_threat_tables.pair_base[AS * 162 + ac * 18];
		Bitboard tos = effects_from(pc, from, occ) & targets;
		while (tos) {
			const Square to = tos.pop();
			const int dc = Threat::tc_from_pt(type_of(pos.piece_on(to)));
			if (!Threat::is_excluded_pair(AS, ac, DS, dc)) {
				const int idx = pb[DS * 9 + dc] + fo + (int)ord[nrm[(int)to]];
				if ((unsigned)idx < (unsigned)THREAT_DIM) sum += proj[idx];
			}
		}
	}
	return sum;
}

static std::int32_t threat_factor_value(const Position& pos, Color stm, Color nstm, int bucket) {
 // Both perspectives use identical occupancy and non-king piece sets. Reuse
 // them and only exchange attacker/target bitboards between the two scans.
 const Bitboard occ = pos.pieces();
 const Bitboard nonking = occ & ~pos.pieces(KING);
 const Bitboard stm_pieces = nonking & pos.pieces(stm);
 const Bitboard nstm_pieces = nonking & pos.pieces(nstm);
 const bool stm_mirror = is_hm_mirror(pos.square<KING>(stm), stm);
 const bool nstm_mirror = is_hm_mirror(pos.square<KING>(nstm), nstm);
 const std::int64_t sum = threat_factor_projected_perspective_profile13_shared(
         pos, stm, stm_mirror, bucket, occ, nstm_pieces, stm_pieces)
         - threat_factor_projected_perspective_profile13_shared(
                 pos, nstm, nstm_mirror, bucket, occ, stm_pieces, nstm_pieces);
 return (std::int32_t)(sum >= 0 ? (sum + 63) / 127 : -((-sum + 63) / 127));
}

// Profile 14 is the rank-1 form of the former shared factor projection. It has
// a direct raw-domain weight for each [bucket][Threat feature], thereby removing
// factor rotation/scale freedom while retaining all eight progress buckets.
template<typename SumT>
static SumT threat_direct_perspective_profile14_shared(
        const Position& pos, Color persp, bool mirror, int bucket,
        const Bitboard occ, Bitboard attackers, const Bitboard targets) {
    const std::int32_t* const weights = g_net->threat_direct_w.data()
            + (std::size_t)bucket * THREAT_DIM;
    const std::uint8_t* const nrm = g_norm_sq[persp][mirror ? 1 : 0];
    constexpr int AS = 1, DS = 0;
    // Every target square is revisited for several attacking pieces. Classify
    // the current perspective's targets once, outside the attacker hot loop.
    std::uint8_t profile14_target_class[SQ_NB];
    Bitboard profile14_targets_left = targets;
    while (profile14_targets_left) {
        const Square sq = profile14_targets_left.pop();
        profile14_target_class[(int)sq] = (std::uint8_t)Threat::tc_from_pt(type_of(pos.piece_on(sq)));
    }
    SumT sum = 0;
    while (attackers) {
        const Square from = attackers.pop();
        const Piece pc = pos.piece_on(from);
        const int ac = Threat::tc_from_pt(type_of(pc));
        const int pat = Threat::attack_pattern_id(ac, WHITE);
        const int fn = (int)nrm[(int)from];
        const int fo = g_threat_tables.from_offset[pat][fn];
        const std::uint8_t* const ord = g_threat_tables.attack_order[pat][fn];
        const int* const pb = &g_threat_tables.pair_base[AS * 162 + ac * 18];
        Bitboard tos = effects_from(pc, from, occ) & targets;
        while (tos) {
            const Square to = tos.pop();
            const int dc = (int)profile14_target_class[(int)to];
            // Scope 1 keeps same-class rows only for minor classes (0..4).
            // Scope 2/3 retain all profile14 rows; scope 3 additionally scans kings as GoldLike.
            if (P2_THREAT_DIRECT_SCOPE == 1 && ac == dc && ac > 4) continue;
            if (!Threat::is_excluded_pair(AS, ac, DS, dc)) {
                const int idx = pb[DS * 9 + dc] + fo + (int)ord[nrm[(int)to]];
                if ((unsigned)idx < (unsigned)THREAT_DIM) sum += weights[idx];
            }
        }
    }
    return sum;
}

static std::int32_t threat_direct_value(const Position& pos, Color stm, Color nstm, int bucket) {
    const Bitboard occ = pos.pieces();
    // Scope 3 scans king as GoldLike for NPS measurements only.
    const Bitboard direct_pieces = P2_THREAT_DIRECT_SCOPE == 3
        ? occ
        : (occ & ~pos.pieces(KING));
    const Bitboard stm_pieces = direct_pieces & pos.pieces(stm);
    const Bitboard nstm_pieces = direct_pieces & pos.pieces(nstm);
    const bool stm_mirror = is_hm_mirror(pos.square<KING>(stm), stm);
    const bool nstm_mirror = is_hm_mirror(pos.square<KING>(nstm), nstm);
    if (g_net->threat_direct_i32_safe) {
        const std::int32_t sum = threat_direct_perspective_profile14_shared<std::int32_t>(
                pos, stm, stm_mirror, bucket, occ, nstm_pieces, stm_pieces)
                - threat_direct_perspective_profile14_shared<std::int32_t>(
                        pos, nstm, nstm_mirror, bucket, occ, stm_pieces, nstm_pieces);
        return sum;
    }
    const std::int64_t sum = threat_direct_perspective_profile14_shared<std::int64_t>(
            pos, stm, stm_mirror, bucket, occ, nstm_pieces, stm_pieces)
            - threat_direct_perspective_profile14_shared<std::int64_t>(
                    pos, nstm, nstm_mirror, bucket, occ, stm_pieces, nstm_pieces);
    return (std::int32_t)sum;
}

// Profile 15 uses a standalone 10-class LUT. King is class 9 and is never
// remapped to GoldLike. The normalized coordinate convention matches Profile 14:
// after perspective normalization, every enemy attacker is treated as White.
//
// 高速化 (2026-09-02, いずれも bit-exact):
//  ① 空盤面利き table による事前フィルタ。 実利き ⊆ 空盤面利き なので
//     空盤面利き ∩ targets が空なら Qugiy/magic の利き計算ごと省略できる。
//     さらに非飛び駒は空盤面利き == 実利きなので effects_from() 自体を呼ばない。
//  ② index 計算の巻き上げ。 pattern / from_offset / attack_order 行 / クラス基点は
//     attacker ごとに不変なので内側ループから追い出す (profile14 側は既にこの形)。
//  ③ 重みを dc-minor に並べ替え + i16 化。 同一 attacker の複数 target が
//     近接アドレスになり、 帯域も半減する。
//  ④ 診断分岐 (P2_DIRECT_DIAG) をホットループから撤去し専用の低速関数へ分離。
//  ⑤ クラス別 Bitboard 走査 (2026-09-02)。 P2_DIRECT_STATS 実測で 1 eval あたり
//     attacker ループは 27-39 回 / 実利き計算 4.9-7.7 回 に対し、 重み参照は 2.6-11.9 回
//     しかない。 つまりコストは列挙側に偏っているので、 列挙の回数そのものを削る:
//     (a) 盤上全駒のクラス表 cls[SQ_NB] を作る 1 パス (27-39 反復) を廃止し、
//         target のクラスは pair が立った数回だけ piece_on から引く。
//         attacker のクラスは クラス別 Bitboard で回すので走査時点で既知。
//     (b) attacker クラスが群ごとに定数になるので pattern / class_base / 空盤面利き行 /
//         飛び駒判定を内側ループから巻き上げる (integral_constant で定数畳み込み)。
//     (c) 盤上駒の約半数を占める歩は 1 枚ずつ回さず Bitboard ごとシフトして一括で
//         targets と交差させる。 歩の利きは 1 マスなので、 当たった target から
//         シフトを戻せば attacker が一意に定まる。
//     走査する pair 集合は同一なので評価値は bit-exact。
template<typename SumT, typename WT>
static SumT threat_direct_perspective_profile15(
        const Position& pos, Color persp, bool mirror, Color attacker_color,
        const WT* const weights, const Bitboard occ, const Bitboard targets) {
    const std::uint8_t* const nrm = g_norm_sq[persp][mirror ? 1 : 0];
    const Direct15::Tables& tbl = g_direct15_tables;
    const Bitboard (* const pseudo)[SQ_NB] = g_direct15_pseudo[attacker_color];
    SumT sum = 0;

    // attacker (from) が確定した後の index 基点算出 + target ループ。
    // target のクラスはここでだけ引く (実測 2.6-11.9 回/eval)。
    auto scan_from = [&](const int ac, const int pat, const Square from, Bitboard tos) {
        const int from_n = (int)nrm[(int)from];
        const std::uint8_t* const ord = tbl.attack_order[pat][from_n];
        const WT* const wrow = weights + tbl.class_base[ac]
                + (std::size_t)tbl.from_offset[pat][from_n] * Direct15::CLASS_NB;
        do {
            const Square to = tos.pop();
            const int dc = Direct15::class_from_piece_type(type_of(pos.piece_on(to)));
            sum += (SumT)wrow[(int)ord[nrm[(int)to]] * Direct15::CLASS_NB + dc];
        } while (tos);
    };

    // --- 歩 (class 0) は Bitboard ごと一括処理。 ---
    {
        constexpr int AC = 0;
        const int pat = Direct15::attack_pattern_id(AC, WHITE);
        // pawnBbEffect のシフトは筋の境界をまたいで桁あふれするので、 利きが盤外に
        // なる段を先に落としてから使う (成らずに最終段にいる歩は非合法なので通常は空。
        // 不正な SFEN を渡されても従来と同じ「利きなし」に落ちるようマスクしておく)。
        const Bitboard pawns = pos.pieces(attacker_color, PAWN)
                & ~(attacker_color == BLACK ? rank_bb(RANK_1) : rank_bb(RANK_9));
        Bitboard hits = pawnBbEffect(attacker_color, pawns) & targets;
        while (hits) {
            const Square to   = hits.pop();
            const Square from = to + (attacker_color == BLACK ? SQ_D : SQ_U);
            scan_from(AC, pat, from, Bitboard(to));
        }
    }

    // --- 歩以外はクラス別 Bitboard で走査。 ---
    // 非飛び駒 (桂/銀/金相当/玉) は 空盤面利き == 実利き なので effects_from を呼ばない。
    // 飛び駒 (香/角/飛/馬/龍) は 実利き ⊆ 空盤面利き なので、 交差が無ければ利き計算ごと省く。
    auto scan_class = [&](auto ac_tag, Bitboard bb) {
        constexpr int AC = decltype(ac_tag)::value;
        constexpr bool SLIDER = ((Direct15::SLIDER_CLASS_MASK >> AC) & 1u) != 0;
        if (!bb) return;
        const int pat = Direct15::attack_pattern_id(AC, WHITE);
        const Bitboard* const prow = pseudo[AC];
        do {
            const Square from = bb.pop();
            Bitboard tos = prow[(int)from] & targets;
            if (!tos) continue;
            if constexpr (SLIDER) {
                tos = direct15_slider_effect(AC, attacker_color, from, occ) & targets;
                if (!tos) continue;
            }
            scan_from(AC, pat, from, tos);
        } while (bb);
    };
    scan_class(std::integral_constant<int, 1>{}, pos.pieces(attacker_color, LANCE));
    scan_class(std::integral_constant<int, 2>{}, pos.pieces(attacker_color, KNIGHT));
    scan_class(std::integral_constant<int, 3>{}, pos.pieces(attacker_color, SILVER));
    scan_class(std::integral_constant<int, 4>{}, pos.pieces(attacker_color, GOLDS));
    scan_class(std::integral_constant<int, 5>{}, pos.pieces(attacker_color, BISHOP));
    scan_class(std::integral_constant<int, 6>{}, pos.pieces(attacker_color, ROOK));
    scan_class(std::integral_constant<int, 7>{}, pos.pieces(attacker_color, HORSE));
    scan_class(std::integral_constant<int, 8>{}, pos.pieces(attacker_color, DRAGON));
    scan_class(std::integral_constant<int, 9>{}, pos.pieces(attacker_color, KING));
    return sum;
}

// P2_DIRECT_DIAG 用の計測専用パス。 通常経路から分岐を追い出すため別関数にしてある。
//   1 = 重み参照を外す (index 計算までは実行)
//   2 = index 計算を外す (利き列挙のみ)
// (通常経路と同じクラス別走査の形にしてある。 そうしないと内訳が現行ホットループの
//  ものにならない。)
template<typename SumT>
static __attribute__((noinline)) SumT threat_direct_perspective_profile15_diag(
        const Position& pos, Color persp, bool mirror, Color attacker_color, int diag,
        const Bitboard occ, const Bitboard targets) {
    const std::uint8_t* const nrm = g_norm_sq[persp][mirror ? 1 : 0];
    const Direct15::Tables& tbl = g_direct15_tables;
    const Bitboard (* const pseudo)[SQ_NB] = g_direct15_pseudo[attacker_color];
    SumT sum = 0;

    auto scan_from = [&](const int ac, const int pat, const Square from, Bitboard tos) {
        if (diag == 2) { do { tos.pop(); sum += 1; } while (tos); return; }
        const int from_n = (int)nrm[(int)from];
        const std::uint8_t* const ord = tbl.attack_order[pat][from_n];
        const int wbase = tbl.class_base[ac]
                + tbl.from_offset[pat][from_n] * Direct15::CLASS_NB;
        do {
            const Square to = tos.pop();
            const int dc = Direct15::class_from_piece_type(type_of(pos.piece_on(to)));
            sum += (wbase + (int)ord[nrm[(int)to]] * Direct15::CLASS_NB + dc) & 1;
        } while (tos);
    };

    {
        constexpr int AC = 0;
        const int pat = Direct15::attack_pattern_id(AC, WHITE);
        const Bitboard pawns = pos.pieces(attacker_color, PAWN)
                & ~(attacker_color == BLACK ? rank_bb(RANK_1) : rank_bb(RANK_9));
        Bitboard hits = pawnBbEffect(attacker_color, pawns) & targets;
        while (hits) {
            const Square to   = hits.pop();
            const Square from = to + (attacker_color == BLACK ? SQ_D : SQ_U);
            scan_from(AC, pat, from, Bitboard(to));
        }
    }
    auto scan_class = [&](auto ac_tag, Bitboard bb) {
        constexpr int AC = decltype(ac_tag)::value;
        constexpr bool SLIDER = ((Direct15::SLIDER_CLASS_MASK >> AC) & 1u) != 0;
        if (!bb) return;
        const int pat = Direct15::attack_pattern_id(AC, WHITE);
        const Bitboard* const prow = pseudo[AC];
        do {
            const Square from = bb.pop();
            Bitboard tos = prow[(int)from] & targets;
            if (!tos) continue;
            if constexpr (SLIDER) {
                tos = direct15_slider_effect(AC, attacker_color, from, occ) & targets;
                if (!tos) continue;
            }
            scan_from(AC, pat, from, tos);
        } while (bb);
    };
    scan_class(std::integral_constant<int, 1>{}, pos.pieces(attacker_color, LANCE));
    scan_class(std::integral_constant<int, 2>{}, pos.pieces(attacker_color, KNIGHT));
    scan_class(std::integral_constant<int, 3>{}, pos.pieces(attacker_color, SILVER));
    scan_class(std::integral_constant<int, 4>{}, pos.pieces(attacker_color, GOLDS));
    scan_class(std::integral_constant<int, 5>{}, pos.pieces(attacker_color, BISHOP));
    scan_class(std::integral_constant<int, 6>{}, pos.pieces(attacker_color, ROOK));
    scan_class(std::integral_constant<int, 7>{}, pos.pieces(attacker_color, HORSE));
    scan_class(std::integral_constant<int, 8>{}, pos.pieces(attacker_color, DRAGON));
    scan_class(std::integral_constant<int, 9>{}, pos.pieces(attacker_color, KING));
    return sum;
}

#if defined(P2_DIRECT_STATS_BUILD)
// 計数専用。 ホットループと同じ順序で走査して回数だけ数える (値は捨てる)。
// 通常経路には一切分岐を足さないため、 stats ビルドでのみ呼ばれる別関数にしてある。
// クラス別走査後の形で数える。 歩は一括処理なので attacker ループには現れない
// (pawn_batch_hits として別に数える)。
static __attribute__((noinline)) void direct15_count_perspective(
        const Position& pos, Color attacker_color,
        const Bitboard occ, const Bitboard targets) {
    const Bitboard (* const pseudo)[SQ_NB] = g_direct15_pseudo[attacker_color];
    {
        const Bitboard pawns = pos.pieces(attacker_color, PAWN)
                & ~(attacker_color == BLACK ? rank_bb(RANK_1) : rank_bb(RANK_9));
        Bitboard hits = pawnBbEffect(attacker_color, pawns) & targets;
        while (hits) { hits.pop(); ++g_direct15_stats.features; ++g_direct15_stats.pawn_hits; }
    }
    auto count_class = [&](int ac, Bitboard bb) {
        while (bb) {
            const Square from = bb.pop();
            ++g_direct15_stats.attackers;
            Bitboard tos = pseudo[ac][(int)from] & targets;
            if (!tos) continue;
            ++g_direct15_stats.pseudo_pass;
            if ((Direct15::SLIDER_CLASS_MASK >> ac) & 1u) {
                ++g_direct15_stats.slider_calc;
                tos = direct15_slider_effect(ac, attacker_color, from, occ) & targets;
                if (!tos) continue;
            }
            do { tos.pop(); ++g_direct15_stats.features; } while (tos);
        }
    };
    count_class(1, pos.pieces(attacker_color, LANCE));
    count_class(2, pos.pieces(attacker_color, KNIGHT));
    count_class(3, pos.pieces(attacker_color, SILVER));
    count_class(4, pos.pieces(attacker_color, GOLDS));
    count_class(5, pos.pieces(attacker_color, BISHOP));
    count_class(6, pos.pieces(attacker_color, ROOK));
    count_class(7, pos.pieces(attacker_color, HORSE));
    count_class(8, pos.pieces(attacker_color, DRAGON));
    count_class(9, pos.pieces(attacker_color, KING));
}

static __attribute__((noinline)) void direct15_count_stats(
        const Position& pos, Color stm, Color nstm, int bucket, const Bitboard occ,
        const Bitboard stm_pieces, const Bitboard nstm_pieces) {
    ++g_direct15_stats.calls;
    g_direct15_stats.cls_pieces += occ.pop_count();
    direct15_count_perspective(pos, nstm, occ, stm_pieces);
    direct15_count_perspective(pos, stm,  occ, nstm_pieces);
    const StateInfo* const st = pos.state();
    if (st->previous != nullptr && st->dirtyPiece.dirty_num != 0
        && !st->lastMove.is_drop() && st->lastMovedPieceType == KING)
        ++g_direct15_stats.king_move;
    // 親の bucket は p2_prog_sum が計算済みのときだけ判る (progress_bucket と同じ閾値)。
    if (st->previous != nullptr && st->previous->p2_prog_computed) {
        static const float TH[7] = {
            -1.9459101f, -1.0986123f, -0.5108256f, 0.0f, 0.5108256f, 1.0986123f, 1.9459101f
        };
        const float s = st->previous->p2_prog_sum;
        int pb = 0;
        for (int i = 0; i < 7; ++i) if (s >= TH[i]) pb++;
        ++g_direct15_stats.bucket_known;
        if (pb == bucket) ++g_direct15_stats.bucket_same;
    }
}
#endif

// P5: 被攻撃駒が守られているかで Direct15 の次元を x2 にした net (P2THRD2 dims = 2*DIM) 用の経路。
// 旧 net はこの関数を通らない (旧経路 threat_direct_perspective_profile15 の命令列は不変)。
//
//   defended(to) = 「to にいる persp 側の駒に、 persp 側の (他の) 駒の実利きが 1 つ以上ある」
//
// 実利きは Direct15 の pair 列挙と同じもの (占有を考慮、 歩・玉を含む、 空盤面利き表 +
// 飛び駒は Qugiy/magic) の和集合で定義するので、 学習側
// (bullet_lib shogi_halfka_hm_threat_direct15.rs の defended map) と定義が一致する。
// 重みの索引は 守備なし → +0、 守備あり → +DIM (ファイル上 [守備なし DIM | 守備あり DIM])。
//
// 実装は 2 段:
//  1) 両色の全駒について実利きを 1 回ずつ求め、 (a) 色ごとの利き和集合 att_map[c] と
//     (b) 敵駒に当たっている attacker の (ac, pat, from, tos) を小バッファに記録する。
//     従来経路は「空盤面利き ∩ 敵駒 が空なら飛び駒の利き計算を省く」が、 P5 では味方駒への
//     利きも要るので事前フィルタを「∩ 全占有」に緩める (盤上駒に届かない飛び駒だけ省く)。
//     att_map[c] は c の駒のある升でだけ参照されるので、 これで十分。
//  2) 記録した pair を走査し、 target を att_map[persp] で 守備なし / あり の 2 つの Bitboard に
//     分けてそれぞれ固定オフセットで重みを引く (target ごとの分岐・test() 呼び出しなし)。
template<typename SumT, typename WT>
static SumT threat_direct15_value_defended(
        const Position& pos, Color stm, Color nstm, bool stm_mirror, bool nstm_mirror,
        const WT* const w_stm, const WT* const w_nstm,
        const Bitboard occ, const Bitboard stm_pieces, const Bitboard nstm_pieces) {
    struct Hit { Bitboard tos; std::uint8_t ac, pat, from; };
    // 1 色の盤上駒は最大 40 枚。 歩は当たった target ごとに 1 record だが 1 枚の歩の利きは
    // 1 升なので record 数 <= 歩の枚数。 よって色ごとの record 数 <= 40。
    Hit att_hits[COLOR_NB][40];
    int  att_n[COLOR_NB] = {0, 0};
    Bitboard att_map[COLOR_NB];

    // 色 c の全駒の実利きを集め、 targets (敵駒) への当たりを記録する。
    auto collect = [&](const Color c, const Bitboard targets) {
        Hit* const out = att_hits[c];
        int n = 0;
        Bitboard a(ZERO);
        const Bitboard (* const pseudo)[SQ_NB] = g_direct15_pseudo[c];
        {   // 歩: Bitboard ごと一括。 最終段の歩は利きが盤外なので落とす (旧経路と同じ)。
            const Bitboard pawns = pos.pieces(c, PAWN)
                    & ~(c == BLACK ? rank_bb(RANK_1) : rank_bb(RANK_9));
            const Bitboard pe = pawnBbEffect(c, pawns);
            a |= pe;
            Bitboard hits = pe & targets;
            const std::uint8_t pat0 = (std::uint8_t)Direct15::attack_pattern_id(0, WHITE);
            while (hits) {
                const Square to   = hits.pop();
                const Square from = to + (c == BLACK ? SQ_D : SQ_U);
                out[n++] = Hit{Bitboard(to), (std::uint8_t)0, pat0, (std::uint8_t)from};
            }
        }
        auto scan_class = [&](auto ac_tag, Bitboard bb) {
            constexpr int AC = decltype(ac_tag)::value;
            constexpr bool SLIDER = ((Direct15::SLIDER_CLASS_MASK >> AC) & 1u) != 0;
            if (!bb) return;
            const std::uint8_t pat = (std::uint8_t)Direct15::attack_pattern_id(AC, WHITE);
            const Bitboard* const prow = pseudo[AC];
            do {
                const Square from = bb.pop();
                Bitboard e = prow[(int)from];
                if constexpr (SLIDER) {
                    // 実利き ⊆ 空盤面利き。 盤上のどの駒にも届かない飛び駒は利き計算ごと省く。
                    if (!(e & occ)) continue;
                    e = direct15_slider_effect(AC, c, from, occ);
                }
                a |= e;
                const Bitboard tos = e & targets;
                if (tos) out[n++] = Hit{tos, (std::uint8_t)AC, pat, (std::uint8_t)from};
            } while (bb);
        };
        scan_class(std::integral_constant<int, 1>{}, pos.pieces(c, LANCE));
        scan_class(std::integral_constant<int, 2>{}, pos.pieces(c, KNIGHT));
        scan_class(std::integral_constant<int, 3>{}, pos.pieces(c, SILVER));
        scan_class(std::integral_constant<int, 4>{}, pos.pieces(c, GOLDS));
        scan_class(std::integral_constant<int, 5>{}, pos.pieces(c, BISHOP));
        scan_class(std::integral_constant<int, 6>{}, pos.pieces(c, ROOK));
        scan_class(std::integral_constant<int, 7>{}, pos.pieces(c, HORSE));
        scan_class(std::integral_constant<int, 8>{}, pos.pieces(c, DRAGON));
        scan_class(std::integral_constant<int, 9>{}, pos.pieces(c, KING));
        att_map[c] = a;
        att_n[c]   = n;
    };
    collect(stm,  nstm_pieces);   // stm の駒の利き、 nstm の駒への当たり (= nstm 視点の pair)
    collect(nstm, stm_pieces);    // nstm の駒の利き、 stm の駒への当たり (= stm 視点の pair)

    const Direct15::Tables& tbl = g_direct15_tables;
    // persp 視点の和: attacker = ~persp の記録を走査し、 target の被守備を att_map[persp] で判定。
    auto accumulate = [&](const Color persp, const bool mirror, const WT* const w, const Color attacker) -> SumT {
        const std::uint8_t* const nrm = g_norm_sq[persp][mirror ? 1 : 0];
        const Bitboard def_map = att_map[persp];
        const Hit* const hs = att_hits[attacker];
        const int n = att_n[attacker];
        SumT sum = 0;
        for (int i = 0; i < n; ++i) {
            const Hit& h = hs[i];
            const int from_n = (int)nrm[h.from];
            const std::uint8_t* const ord = tbl.attack_order[h.pat][from_n];
            const WT* const wrow = w + tbl.class_base[h.ac]
                    + (std::size_t)tbl.from_offset[h.pat][from_n] * Direct15::CLASS_NB;
            auto scan = [&](Bitboard tos, const WT* const wr) {
                while (tos) {
                    const Square to = tos.pop();
                    const int dc = Direct15::class_from_piece_type(type_of(pos.piece_on(to)));
                    sum += (SumT)wr[(int)ord[nrm[(int)to]] * Direct15::CLASS_NB + dc];
                }
            };
            scan(def_map.andnot(h.tos), wrow);                 // 守備なし → +0
            scan(h.tos & def_map,       wrow + Direct15::DIM); // 守備あり → +DIM
        }
        return sum;
    };
    return accumulate(stm,  stm_mirror,  w_stm,  nstm)
         - accumulate(nstm, nstm_mirror, w_nstm, stm);
}

static std::int32_t threat_direct15_value(const Position& pos, Color stm, Color nstm, int bucket) {
    const Bitboard occ = pos.pieces();
    const Bitboard stm_pieces = occ & pos.pieces(stm);
    const Bitboard nstm_pieces = occ & pos.pieces(nstm);
    const bool stm_mirror = is_hm_mirror(pos.square<KING>(stm), stm);
    const bool nstm_mirror = is_hm_mirror(pos.square<KING>(nstm), nstm);
    // 旧実装はここで盤上全駒のクラス表 cls[SQ_NB] を作っていたが、 実測 (P2_DIRECT_STATS)
    // では 1 eval あたり 盤上駒 27-39 枚に対し 重み参照は 2.6-11.9 回しかなく、
    // クラス表の大半が使われずに捨てられていた。 attacker のクラスはクラス別 Bitboard 走査で
    // 既知、 target のクラスは pair が立ったときだけ引けばよいのでこの 1 パスごと廃止した。

#if defined(P2_DIRECT_STATS_BUILD)
    if (g_direct15_stats_on)
        direct15_count_stats(pos, stm, nstm, bucket, occ, stm_pieces, nstm_pieces);
#endif

    const int diag = direct_diag_env();
    if (diag != 0) {
        return threat_direct_perspective_profile15_diag<std::int32_t>(
                       pos, stm, stm_mirror, nstm, diag, occ, stm_pieces)
             - threat_direct_perspective_profile15_diag<std::int32_t>(
                       pos, nstm, nstm_mirror, stm, diag, occ, nstm_pieces);
    }

    // テーブル位置は load 時に確定した形状値から計算する (旧 net: stride=DIM, nstm_offset=0 で
    // 従来と同じアドレス = 評価値 bit-exact)。 P2 net は nstm 視点だけ別テーブルを引く。
    const std::size_t off   = (std::size_t)bucket * g_net->direct_bucket_stride;
    const std::size_t off_n = off + g_net->direct_nstm_offset;
    if (g_net->direct_defended) {
        // P5 net (被守備ビット付き)。 旧 net はここを通らないので旧経路の命令列は不変。
        if (g_net->threat_direct_w16_ok) {
            const std::int16_t* const w = g_net->threat_direct_w16.data();
            return threat_direct15_value_defended<std::int32_t>(
                    pos, stm, nstm, stm_mirror, nstm_mirror, w + off, w + off_n, occ, stm_pieces, nstm_pieces);
        }
        const std::int32_t* const w = g_net->threat_direct_w.data();
        if (g_net->threat_direct_i32_safe)
            return threat_direct15_value_defended<std::int32_t>(
                    pos, stm, nstm, stm_mirror, nstm_mirror, w + off, w + off_n, occ, stm_pieces, nstm_pieces);
        return (std::int32_t)threat_direct15_value_defended<std::int64_t>(
                pos, stm, nstm, stm_mirror, nstm_mirror, w + off, w + off_n, occ, stm_pieces, nstm_pieces);
    }
    // i16 重みは load 時に |w| <= 32767 を確認できた場合のみ有効 (P2_DIRECT_W32=1 で無効化)。
    if (g_net->threat_direct_w16_ok) {
        const std::int16_t* const w = g_net->threat_direct_w16.data();
        return threat_direct_perspective_profile15<std::int32_t>(pos, stm, stm_mirror, nstm, w + off, occ, stm_pieces)
             - threat_direct_perspective_profile15<std::int32_t>(pos, nstm, nstm_mirror, stm, w + off_n, occ, nstm_pieces);
    }
    const std::int32_t* const w = g_net->threat_direct_w.data();
    if (g_net->threat_direct_i32_safe) {
        return threat_direct_perspective_profile15<std::int32_t>(pos, stm, stm_mirror, nstm, w + off, occ, stm_pieces)
             - threat_direct_perspective_profile15<std::int32_t>(pos, nstm, nstm_mirror, stm, w + off_n, occ, nstm_pieces);
    }
    const std::int64_t sum = threat_direct_perspective_profile15<std::int64_t>(pos, stm, stm_mirror, nstm, w + off, occ, stm_pieces)
                           - threat_direct_perspective_profile15<std::int64_t>(pos, nstm, nstm_mirror, stm, w + off_n, occ, nstm_pieces);
    return (std::int32_t)sum;
}

// Direct15 の raw 単位 cap を返す。
// 優先順位: USI option P2DirectCapRaw > net の fv_scale から確定した cap > 旧挙動。
// 旧挙動 (cap_cp * effective_fv) は学習時 cap と一致しないため、 fv_scale marker を
// 持たない旧 net でのみ使うフォールバック。
static inline std::int64_t profile15_direct_cap_raw(int effective_fv) {
	if (g_p2_direct_cap_raw_option != 0) return (std::int64_t)g_p2_direct_cap_raw_option;
	if (g_net->direct_cap_raw != 0)      return (std::int64_t)g_net->direct_cap_raw;
	return (std::int64_t)P2_THREAT_DIRECT_CAP_CP * std::max(1, effective_fv);
}

static std::int32_t cap_profile15_threat_direct(std::int32_t raw, int effective_fv) {
	// P2_THREAT_DIRECT_CAP_CP=0 は「上限なし」。 学習側 --direct15-cap-cp=0 と対。
	if (P2_THREAT_DIRECT_CAP_CP == 0) return raw;
	const std::int64_t cap = profile15_direct_cap_raw(effective_fv);
	if (raw > cap) return (std::int32_t)cap;
	if (raw < -cap) return (std::int32_t)-cap;
	return raw;
}
struct ThreatPlan {
	const StateInfo* base;
	const StateInfo* chain[P2_DT_CHAIN_MAX];
	int              nchain;
	bool             mirror;
	// 2026-10-07 (S2 §3.1): この acc を確定する表の群 (= 現局面の群)。 base 採用条件と
	//   行の offset (群 * THREAT_DIM) の両方に使う。 群化しない net は 0。
	std::uint8_t     group;
};

// éªåï½¸E¬éï½¨ env(hot path ç¸ºä¹ï½Egetenv ç¹§å®ï½¿E½ç¸ºã»ã»ç¸ºå¶âE¹§ã»æªè¬¨E° static ç¸ºE«é¡E³ç¹§Â)ç¸²ã»
static inline bool threat_force_full_env() {
	// ★NPS(A5): 関数内 static のガード変数ロードを消す。 既定値に固定 = bit-exact
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_FULL") != nullptr;   // è¶E¸ç¸ºE« full recompute
	return v;
#else
	return false;
#endif
}
static inline bool threat_verify_env() {
	// ★NPS(A5): 関数内 static のガード変数ロードを消す。 既定値に固定 = bit-exact
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_VERIFY") != nullptr; // è¾E®è»ã»âEfull ç¹§å ¤ã»è·ã»
	return v;
#else
	return false;
#endif
}
static inline bool threat_naive_env() {
  
#if defined(P2_DIAG)
	static const bool v = std::getenv("P2_THREAT_NAIVE") != nullptr;
  return v;
#else
	return false;   // ★NPS(A5): 関数内 static のガード変数ロードを消す
#endif
}

// persp éä¹ãç¸ºE«ç¸ºE¤ç¸ºã»â»ç¸²è¨Elid ç¸ºEªé¾éã»(base)+ é©ï½©éï½¨ç¸ºå¶âç¸ºã»dirty-threat chainç¸²éªEè±ï½ºç¹§âï½ç¸²ã»
// è¿´E¾ ply ç¸ºE®éæºã¡ç¹å±Îç¸ºæ¢§æ´éæ»ãEç¸ºEªç¹§ã»éï½ªç¸ºã»valid ç¸ºä¹â½ mirror è³Âé¾E´ç¸ºE®ç¸ºE¨ç¸ºéªâ¡ç¸ºã»ç¸ºèï¼E¸ºE§éæ»ãEç¸ºå¶Eç¸²ã»
// 2026-10-07 DT_COST 用の wrapper: 本体を呼んでから plan.base を潰す (出口が複数あるため)。
static void resolve_threat_plan_impl(const Position& pos, Color persp, ThreatPlan& plan,
                                     std::uint8_t group);
static inline void resolve_threat_plan(const Position& pos, Color persp, ThreatPlan& plan,
                                       std::uint8_t group) {
	resolve_threat_plan_impl(pos, persp, plan, group);
#if defined(P2_PROBE_BUILD)
	// 計測専用: tuple は構築させるが、 基点は使わせない → 評価は full のまま = bit-exact
	if (probe_dt_cost()) { plan.base = nullptr; plan.nchain = 0; }
#endif
}
static void resolve_threat_plan_impl(const Position& pos, Color persp, ThreatPlan& plan,
                                     std::uint8_t group) {
	StateInfo* const st  = pos.state();
	const Square     ksq = pos.square<KING>(persp);
	plan.base   = nullptr;
	plan.nchain = 0;
	plan.mirror = is_hm_mirror(ksq, persp);
	plan.group  = group;
	if (threat_force_full_env()) return;

	const bool mirror = plan.mirror;
	int cost = 0;

	// lazy éæ»ãE: ç¸ºèãE ply(= è¿´E¾è»Âé«±E¢)ç¸ºE®éæºã¡ç¹å±Îç¸ºå¾âªç¸ºE°è¾ï½¡ç¸ºä»£Eç¸ºE°èè´ï½½æï½ç¸²ã»
	// pos ç¸ºã»st ç¸ºE®è»Âé«±E¢ç¸ºE§ç¸ºãE½èç¿«â¡ç¸ºé¡ï½¦Eªé¶E¤é«±E¢ç¹§ã»A/B é¬æï½­E£ç¸ºE§èå²æ¨ç¸ºE§ç¸ºéªEç¸ºE®ç¸ºE§ç¸²â«å½è¬èEãç¸ºéªEç¸ºE®ç¸ºE¯
	// ç¸²æªæ¨è¨E¨ç¸ºE® plyç¸²éªã»ç¸ºE¿ç¸²ãï½¥éã»ç¸ºE®è­ï½ªéæ»ãE ply ç¸ºE¯ chain ç¹§å¸Âå£¹âç¸ºEªç¸ºã»= ç¸ºæ´ã»è¬E¥çç¸ºE§è¬Eâ è»ã»EEç¸²ã»
	// ç¬ãEå½è¬åEµåæ£¡ç¸ºE¯ StateInfo ç¸ºE«è°¿ä¹ï½ç¸ºE®ç¸ºE§ç¸²âï½ç¸ºã»E¸Âè­E½¹ç¸ºE®éä¹ãç¸ºE¯ç¹§E¹ç¹§E­ç¹ï½£ç¹ï½³è¾ï½¡ç¸ºå±ãèæ¦èéï½¨ç¸ºE§ç¸ºéªEE	//   (è­ï½§è³æº¯E£ã»ã»ç¸²çï½¦ä¹ãç¸ºæâEç¸ºE«éï½ªèEEãç¹§E­ç¹ï½£ç¹ï½³ 2 èæ§­Âéªâ²ç¸ºèï¼E¸ºE§ 1 èæ§­âç¸ºEªç¹§ã»ç¸²ã»
	// ç¬ãEã»éï½ªç¸ºã»valid ç¸ºä¹â½ mirror è³Âé¾E´ç¸ºE®ç¸ºE¨ç¸ºéªÂå¾â¡ç¸ºä»£Âå²å½è¬èEâE¹§ä¹ÂãEç¸ºã»ãç¸ºEªç¸ºã»E°E´è·å»ã»é¨ä»ï½±Â
	//   full èå´ï½¨è²E®å±âç¸ºEªç¹§ç¿«Ââ«å½è¬èEE ç¸ºæºã¡ç¹å±Îç¸ºå¾¡E½E¿ç¹§ä¸ï½ç¸ºå¤ï½´ç²éç¸ºE«ç¸ºEªç¹§ã»è³æ»ï½¸E¬: è¾ï½¡è­ï½¡èï½¶éæ»ãEç¸ºE°ç¸ºE¨
	//   ç¹§E¹ç¹§E­ç¹ï½£ç¹ï½³è¬¨E° 492k > è­ï½§è³æº¯E£ã»ã» delta 444k ç¸ºE§é¨¾ã»âé©ãEE¥ç¸ºEªç¸ºE£ç¸ºã»ç¸²ã»
	// ��eager �r���h�ł� do_move ���Ő����ς݂Ȃ̂ł����͑f�ʂ� (�c���Ă��Ă��Q�͂Ȃ�)�B
	if (g_dt_build && st->p2_dt_state == DT_UNBUILT && st->previous != nullptr
	    && st->dirtyPiece.dirty_num != 0 && st->previous->p2_threat_valid[persp]) {
		bool parent_ok = true;
		const Move m = st->lastMove;
		if (!m.is_drop() && m.to_sq() == ksq && st->lastMovedPieceType == KING)
			parent_ok = (is_hm_mirror(m.from_sq(), persp) == mirror);
		if (parent_ok) build_dirty_threats_now(pos);
	}

	const StateInfo* s = st;
	Square kcur = ksq;                 // s è­ããç¸ºE§ç¸ºE® persp ç¸ºE®éEEE½å²E½E®(é©ï½¡ç¹§ç¿«âç¸ºå¾ï½éE½éï½¡)
	for (;;) {
		if (s->p2_threat_valid[persp]) {
			// ç¬ãEirror flip ç¹âEãç¹ãEãE é¾éã»ç¸ºE® HM mirror ç¸ºæªæ¨è¨E¨ç¸ºE¨è·å¾Â§ç¸ºE¨ç¸ºéªâ¡ç¸ºã»chain è¿E¯ç¸²ã»
			//   è¾E®è»ã»ã¡ç¹å±Îç¸ºE¯éï½ªèEEã»é«®ã»ç²è¾E®ç¸ºEªç¸ºE®ç¸ºE§ telescoping ç¸ºå±Ââç²éªã»= feature(è¿´E¾) - feature(é¾éã»)ç¸²ã»
			//   index è¹æ¶ã»ç¸²æªæ¨è¨E¨ç¸ºE®ç¸²èersp/mirror ç¸ºE§é¦å¾â§ç¸ºæºï½ç¸²â«E¥éã» acc ç¹§ãEEç¸ºã»mirror ç¸ºE§
			//   è´æï½ç¹§å¾â»ç¸ºã»âç¸ºä»£Eç¸ºE°è¬¨E´è·å»E ç¸ºEªç¸ºã»é¡E°ç¸ºEªç¹§ä¹âEç¹§ã»full ç¸ºE¸ç¹è¼ãç¹ï½¼ç¹ï½«ç¹èEã£ç¹§E¯)ç¸²ã»
			// 2026-10-07 (S2 §3.1): mirror に加えて**群一致**を要求する。 群が違うと base の acc が
			//   別の表で作られているので telescoping が成立しない (0xFF = 未設定も必ず弾かれる)。
			if (is_hm_mirror(kcur, persp) == mirror && s->p2_threat_group[persp] == group)
				plan.base = s;
			break;
		}
		// 🚨2026-10-09: 差分更新が無効 (g_dt_build == false / 群化 net) でも、 **null move の子だけは
		//   p2_dt_state == DT_OK** なのでここで break せず親まで進み、 plan.base が立つ
		//   (do_null_move が g_dt_build を見ずに DT_OK を書く。 position.cpp の該当コメント参照)。
		//   dt_count == 0 なので行数 0 の memcpy になり、 **値は full 再計算と bit-exact**。
		//   実測 (rdtsc probe): ガード据え置きでも plan が eval の 1.1% に立ち、 全部 base != nullptr で行 0。
		//   P2_THREAT_VERIFY=1 (毎 eval で naive full と全要素比較) で MISMATCH 0 件を確認済み。
		// 🚨threat 特徴を**手番依存** (または pliesFromNull / 千日手状態依存) にすると、
		//   resolve_threat_plan は手番を一切見ないのでこの memcpy が**黙って**壊れる。
		if (s->previous == nullptr || s->p2_dt_state != DT_OK || plan.nchain >= P2_DT_CHAIN_MAX) break;
		cost += (int)s->p2_dt_count;
		if (cost > P2_DT_COST_MAX) break;   // chain �̃R�X�g������𒴂����� full �Čv�Z�ɗ��Ƃ�
		plan.chain[plan.nchain++] = s;
		// éEEE½å²E½E®ç¸ºE®é©ï½¡ç¹§ã» ç¸ºèãE ply ç¸ºã»persp ç¸ºE®éE³¨ã»éï½»èè¼âEç¹§å³¨Ââ¬E¦Eªç¸ºE§ç¸ºE¯ from ç¸ºE«ç¸ºã»âE¸²ã»
		// (null move(dirty_num==0)ç¸ºE¯ lastMove ç¸ºã»stale ç¸ºEªç¸ºE®ç¸ºE§é«¯E¤èæ¶Âãèç¸ºE¯è¿æ¶Eç¹§å¾âEç¸ºã»ã»ç¸ºE§
		//  kcur ç¸ºæï½¤å³¨Eç¹§ä¹ãEç¸ºE¯éE½ã»éE½«ç¸ºE®éï½»èè¼ãEç¸ºE¨ç¸ºéªâ¡ç¸ºä»£Âã»
		if (s->dirtyPiece.dirty_num != 0) {
			const Move m = s->lastMove;
			if (!m.is_drop() && m.to_sq() == kcur && s->lastMovedPieceType == KING)
				kcur = m.from_sq();
		}
		s = s->previous;
	}
}

// P2_THREAT_VERIFY: naive scan è¿ãEground truth)ç¸ºE¨è¾£E§è·å»E ç¸²ââEç¹§å¾âç¹§è¥æµ£è±E½£ç¸ºå±â»è£E±è»ç¿«âE¹§ä¹Âã»
static inline int get_layer_stack_index(const Position& pos);
// 2026-09-12: threat 表の bucket 群化。 局面の LayerStack bucket が属する群の表の先頭 index (群 * THREAT_DIM)。
//   群化しない net (threat_groups == 1) は常に 0 を返し bucket も計算しない (= 従来と同じ動作)。
//   群化 net では threat 差分更新を load 時に無効化しているので、 acc は常にこの局面の群の表で full 再計算される。
// 2026-09-15 (A2): bucket を既に持っている呼び出し元 (evaluate_internal) はこちらに渡す。
//   従来は eval 1 回につき get_layer_stack_index (progress_bucket = ensure_progress + 7 比較、
//   P4H2/P8H2 では加えて weighted_hand_total 2 色 x 7 駒種) を evaluate_internal と
//   ensure_threat_both → add_threat_dual → threat_table_offset で二重に計算していた。 値は同一局面の純関数なので bit-exact。
// 2026-10-07 (S2): bucket -> 群 id (0..threat_groups-1)。 群化しない net は 0。
//   StateInfo::p2_threat_group / ThreatPlan::group に入れて base の群一致条件に使う。
static inline std::uint8_t threat_group_b(int bucket) {
	const Net& net = *g_net;
	if (net.threat_groups <= 1) return 0;
	return net.threat_bucket_group[bucket];
}
static inline std::size_t threat_table_offset_b(int bucket) {
	const Net& net = *g_net;
	if (net.threat_groups <= 1) return 0;
	return (std::size_t)net.threat_bucket_group[bucket] * (std::size_t)THREAT_DIM;
}
static inline std::size_t threat_table_offset(const Position& pos) {
	if (g_net->threat_groups <= 1) return 0;
	return threat_table_offset_b(get_layer_stack_index(pos));
}

static void verify_threat_acc(const Position& pos, Color persp, const ThreatPlan& plan) {
	StateInfo* const st = pos.state();

	// accç¸ºE¯è®æE½¨E¼èï½±è¬¨ç²å¾ç¸ºE«ç¸ºèãEStateInfoç¸ºE¸éï½¯è±E½£ç¹§åå¶ç¸ºä»£Eè³æ»ï½½èÂâ¦ompare_accç¸ºE¯èï½±è­çã»ç¹§è²æ§ç¹§Âé§ç©ï½¾E¡èE½¾éï½¡ç¸²ã»

	std::int16_t* const acc = st->p2_threat_acc[persp];

	const StateInfo* const src = st->p2_threat_source[persp];

	const std::int16_t* const compare_acc = (src != nullptr ? src : st)->p2_threat_acc[persp];

	alignas(32) std::int16_t full[FT_MAX];
	std::memset(full, 0, sizeof(std::int16_t) * THREAT_ACC_DIM);
	add_threat_perspective_naive(pos, persp, plan.mirror, full,
	                             (std::size_t)plan.group * (std::size_t)THREAT_DIM);
	int first = -1;
	// ��2026-09-09 �C��: threat accumulator �� THREAT_ACC_DIM ����������
	//   (p2_threat_acc[2][P2_THREAT_ACC_DIM])�B FT_DIM �܂Ŕ�r�����
	//   K<FT_DIM �̃r���h�Ŕz��O (�ׂ̎��_) �Ɩ������� full[] ��˂����킹�A
	//   ���� first_idx=K �̋U MISMATCH ���o�Ă����B
	for (int k = 0; k < THREAT_ACC_DIM; ++k)
		if (compare_acc[k] != full[k]) { first = k; break; }
	if (first < 0) return;
	const Move m = st->lastMove;
	sync_cout << "info string P2_THREAT_VERIFY MISMATCH persp=" << (int)persp
	          << " path=" << (plan.base == nullptr ? "full" : "chain")
	          << " chain_len=" << plan.nchain
	          << " from=" << (m.is_drop() ? -1 : (int)m.from_sq())
	          << " to=" << (int)m.to_sq()
	          << " drop=" << (int)m.is_drop()
	          << " promote=" << (int)m.is_promote()
	          << " capture=" << (int)(st->capturedPiece != NO_PIECE)
	          << " first_idx=" << first
	          << " delta=" << (int)acc[first] << " full=" << (int)full[first]
	          << sync_endl;
	std::memcpy(acc, full, sizeof(std::int16_t) * THREAT_ACC_DIM); // è¬ï½¢éï½¢éï½´é¯E»èæ¨£â©ç¸ºE®ç¸ºæºï½EE¯è±E½£

	st->p2_threat_source[persp] = st;
}

// P2_THREAT_STATS: full fallback è­ãâè´å¢çEé©ï½¡ç¹§å¾ãE valid base ç¸ºE«è»ç¿«E¥ç¸ºä¹ï½éªå¬é¹¸ç¸ºå¶Eç¸²ã»
static inline void tstat_fallback_distance(const StateInfo* st, Color persp) {
#if defined(P2_THREAT_STATS_BUILD)
	if (!threat_stats_env()) return;
	++g_threat_stats.full_calls;
	int d = 0; const StateInfo* c = st;
	while (c->previous != nullptr && d < 8) { c = c->previous; ++d; if (c->p2_threat_valid[persp]) break; }
	if (c != nullptr && c->p2_threat_valid[persp]) {
		if      (d <= 1) ++g_threat_stats.fb_anc1;
		else if (d == 2) ++g_threat_stats.fb_anc2;
		else if (d == 3) ++g_threat_stats.fb_anc3;
		else             ++g_threat_stats.fb_anc4p;
	} else ++g_threat_stats.fb_none;
#else
	(void)st; (void)persp;
#endif
}

// validç¸ºEªThreat accumulatorç¸ºE®è³æ»ï½½èï½è å±Eç¸²E´ourceç¸ºE¯é¾Eªè»ã»ã»éE½«ç¸ºE¾ç¸ºæºãEè¬ï½¢éï½¢ç¹§E¹ç¹§E¿ç¹ãEãè³ç¿«ã»é¾éã»ç¹§åæ¬ ç¸ºå¶Âã»
// sourceç¸ºç³»ullç¸ºEªç¸ºE®ç¸ºE¯full materializeèéªâ¡ç¸ºä»£ãç¸²ââç¸ºE®è£E´è·å»ã»é¾Eªè»ã»ã»éE½«ç¸ºE®é¬E¼æ²ºç¹§å®ï½¿æâEç¸²ã»
static inline const std::int16_t* threat_acc_ptr(const StateInfo* st, Color persp) {
        const StateInfo* const src = st->p2_threat_source[persp];
        return (src != nullptr ? src : st)->p2_threat_acc[persp];
}

// profile 11ç¸ºE§dirty tupleç¸ºå¾ï¼E¸ºE®éä¹ãç¹ï½»mirrorç¸ºE®è¿ï½¹è ï½´indexç¹§åE½¸Âç¸ºE¤ç¹§ãåEè¬èEE ç¸ºEªç¸ºã»Â°ç¹§è²æEè³å£¹âE¹§ä¹Âã»
// è±E½£ç¸ºEªç¹§çï½­èãEThreatè¿ï½¹è ï½´é«®ã»ç²ç¸ºE¯éï½ªç¸ºE¨è³æãEè³Âé¾E´ç¸ºå±Ââ¥ccumulatorè­ï½¬è´èï½ç¹§E³ç¹æã»ç¸ºå¶Eè ¢ã»E¦ââ²ç¸ºEªç¸ºã»Âã»
static bool dirty_chain_no_relevant_cached(const StateInfo* const* chain, int nchain,
                                           Color persp, bool mirror) {
        const std::uint8_t* const nrm = g_norm_sq[persp][mirror ? 1 : 0];
        for (int c = 0; c < nchain; ++c) {
                const StateInfo* const s = chain[c];
                for (int d = 0; d < (int)s->p2_dt_count; ++d)
                        if (dt_to_index(s->p2_dirty_threats[d].data, persp, nrm) >= 0)
                                return false;
        }
        return true;
}

// profile 12ç¸ºE®é¬®å¬Âæº·E®çãEé¨çï½·E¯ç¸²E¥irty tupleç¸ºçï½¦Eªç¸ºE®èï½£é¦E¨8é¦å¾ï½èä¼åç¸ºå¸âEç¸²ã»
// è­E½°éå°±DDç¹§ãæè³å¶E½å ºE»E£é¦E¨é¦å¾ï½è³éå±ç¹§å³¨âç¸ºã»âç¹§å³¨Ââ½E»E£é¦E¨é«®ã»ç²ç¹§ãEç¸ºE®èï£°éç¤¼Eµåæ£¡ç¹§ãï½¦Eªç¸ºE¨è³æãEè³Âé¾E´ç¸ºå¶Eç¸²ã»
// ç¸ºæ´ã»è£E´è·å»ã»è¿ï½¹è ï½´é¦å¾ãESUB/ADDç¹§åE½¸Âè»ã»E¡å¾ï½ç¸ºå£¹Ââ¬E¦Eªaccç¹§åµãç¹æã»ç¸ºå±â»é¨ãE½ºã»ãç¸ºéªEç¸²ã»
static bool rep8_dirty_preserves_representatives(const StateInfo* child, Color persp, bool mirror) {
        const StateInfo* const parent = child->previous;
        if (parent == nullptr || child->p2_dt_state != DT_OK) return false;
        const int nrep = (int)parent->p2_threat_rep_count[persp];
        if (nrep < 0 || nrep > StateInfo::P2_THREAT_REP_CAP) return false;
        const std::uint8_t* const nrm = g_norm_sq[persp][mirror ? 1 : 0];
        int worst = -1;
        if (nrep == StateInfo::P2_THREAT_REP_CAP) {
                worst = 0;
                for (int i = 1; i < nrep; ++i) {
                        const int a = parent->p2_threat_rep_idx[persp][worst];
                        const int b = parent->p2_threat_rep_idx[persp][i];
                        if (ThreatRepresentativeRows::better(g_threat_tables.rep_priority(a), a,
                                                             g_threat_tables.rep_priority(b), b))
                                worst = i;
                }
        }
        for (int i = 0; i < (int)child->p2_dt_count; ++i) {
                const std::uint32_t data = child->p2_dirty_threats[i].data;
                const int idx = dt_to_index(data, persp, nrm);
                if (idx < 0) continue;
                bool is_representative = false;
                for (int r = 0; r < nrep; ++r)
                        if (parent->p2_threat_rep_idx[persp][r] == idx) { is_representative = true; break; }
                const bool is_add = (data & (1u << 24)) != 0;
                if (!is_add && is_representative) return false;
                if (is_add) {
                        // èï½£é¦E¨é«®ã»ç²ç¸ºã»è­ï½ªè²Eç¸ºEªç¹§ç»ç éå°E¡å¾ãEè ¢ã»âEE¥ç¹§ä¹Âã»é¦å¾âEç¹§ç»æè³å¶E½éªEéã»â´ç¹§åºå¾ç¸ºE°ç¸ºå¤§E¤çå§ç¸²ã»
                        if (nrep < StateInfo::P2_THREAT_REP_CAP) return false;
                        const int floor_idx = parent->p2_threat_rep_idx[persp][worst];
                        if (ThreatRepresentativeRows::better(g_threat_tables.rep_priority(idx), idx,
                                                             g_threat_tables.rep_priority(floor_idx), floor_idx))
                                return false;
                }
        }
        return true;
}
// profile 12ç¸ºE§éï½ªç¸ºE®èï½£é¦E¨é«®ã»ç²ç¸ºã»è­ï½ªè²Eç¸ºEªç¹§å³¨Âââç¹§å¾ãEéï½ªè»Âé«±E¢ç¸ºE®èï½¨activeé¦å¾âç¸ºE®ç¹§ãEEç¸²ã»
// ç¸ºèãE1è¬E¹ãEdirty tupleç¹§è²å®èã»âé©ï½©éï½¨ç¸ºå±â»ç¹§ã»é¦å¾ï½éã»â´ç¸ºEªç¸ºã»E°E´è·å»â¡ç¸ºä»£Ââï½­å®£å¯è·å»EèE¨activeé¦ã»
// ç¸ºE¨ç¸ºEªç¹§ä¹ÂãE¼ ç¸ºæºâ²ç¸ºE£ç¸ºE¦fullè»ç²å«ç¹ï½»èï½£é¦E¨é©ï½¸è¬æ§­âç¸ºå±ãéEªaccç¸ºä¹ï½è±E½£éï½ºç¸ºE«è­ï½´è­E½°ç¸ºE§ç¸ºéªEç¸²ã»
// ADD/SUBç¸ºE®è¿æ±å¯è å¾ãEè­ï½¢èE¥ã»4é¦æ²used SIMDç¹èEã£ç¹âEãé©ï½©éï½¨ç¸ºå¶Eç¸²ã»
static bool rep8_apply_underfilled_delta(StateInfo* child, Color persp, bool mirror,
                                         std::int16_t* acc) {
        const StateInfo* const parent = child->previous;
        if (parent == nullptr || child->p2_dt_state != DT_OK) return false;
        const int parent_n = (int)parent->p2_threat_rep_count[persp];
        if (parent_n < 0 || parent_n >= StateInfo::P2_THREAT_REP_CAP) return false;
        const std::uint8_t* const nrm = g_norm_sq[persp][mirror ? 1 : 0];
        constexpr int CAP = StateInfo::P2_THREAT_REP_CAP;
        int simulated[CAP];
        int n_simulated = parent_n;
        for (int i = 0; i < parent_n; ++i) simulated[i] = parent->p2_threat_rep_idx[persp][i];
        auto find_index = [](const int* xs, int n, int idx) {
                for (int i = 0; i < n; ++i) if (xs[i] == idx) return i;
                return -1;
        };
        // é¨E¬1ç¹ä»£ãE childé«®ã»ç²ç¸ºé´APèï½¥è³ä¹ãç¸ºãE½ç¸ºèâEç¸ºE°ç¸ºä»£EéEºé±éªâE¹§ä¹ÂãE½¤E±è¬¨ç²å¾ç¸ºE«è¿¥E¶è«·ä¹ï½èç»å³©ç¸ºå±âç¸ºã»Âã»
        for (int i = 0; i < (int)child->p2_dt_count; ++i) {
                const std::uint32_t data = child->p2_dirty_threats[i].data;
                const int idx = dt_to_index(data, persp, nrm);
                if (idx < 0) continue;
                const int found = find_index(simulated, n_simulated, idx);
                if (data & (1u << 24)) {
                        if (found >= 0) continue;
                        if (n_simulated == CAP) return false;
                        simulated[n_simulated++] = idx;
                } else if (found >= 0) {
                        simulated[found] = simulated[--n_simulated];
                }
        }
        // é¨E¬2ç¹ä»£ãE éï½ªaccç¸ºä¹ï½èçå§é¦å¾â¡ç¸ºä»£EèE°è²å¸·E®å±âE¹§ä¹ÂEª8ç¸ºE¯4é¦æ²used SIMDç¹§å ¤E¶E­è¬âEâE¹§ä¹Âã»
        std::memcpy(acc, parent->p2_threat_acc[persp], sizeof(std::int16_t) * THREAT_ACC_DIM);
        int current[CAP];
        int n_current = parent_n;
        for (int i = 0; i < parent_n; ++i) current[i] = parent->p2_threat_rep_idx[persp][i];
        int add_idx[StateInfo::P2_DT_CAP], sub_idx[StateInfo::P2_DT_CAP];
        int n_add = 0, n_sub = 0;
        const Net& net = *g_net;
        const bool i16net = net.threat_is_i16;
        for (int i = 0; i < (int)child->p2_dt_count; ++i) {
                const std::uint32_t data = child->p2_dirty_threats[i].data;
                const int idx = dt_to_index(data, persp, nrm);
                if (idx < 0) continue;
                const int found = find_index(current, n_current, idx);
                if (data & (1u << 24)) {
                        if (found >= 0) continue;
                        current[n_current++] = idx;
                        P2_TSTAT(++g_threat_stats.delta_rows);
                        if (i16net) threat_acc_add(acc, net, (std::size_t)idx);
                        else { prefetch_threat_row_i8(net, (std::size_t)idx); add_idx[n_add++] = idx; }
                } else if (found >= 0) {
                        current[found] = current[--n_current];
                        P2_TSTAT(++g_threat_stats.delta_rows);
                        if (i16net) threat_acc_sub(acc, net, (std::size_t)idx);
                        else { prefetch_threat_row_i8(net, (std::size_t)idx); sub_idx[n_sub++] = idx; }
                }
        }
        if (!i16net) {
                apply_threat_rows(acc, net, add_idx, n_add, false);
                apply_threat_rows(acc, net, sub_idx, n_sub, true);
        }
        child->p2_threat_rep_count[persp] = (std::uint8_t)n_current;
        for (int i = 0; i < n_current; ++i) child->p2_threat_rep_idx[persp][i] = current[i];
        return true;
}

// plan ç¸ºE«è èâ²ç¸ºE¦ p2_threat_acc[persp] ç¹§è²æ²ç¹§âï½Eèå©E¦ä¹ã)ç¸²ã»
static void apply_threat_plan(const Position& pos, Color persp, const ThreatPlan& plan) {
#if defined(P2_PROBE_BUILD)
	++g_pc_plan;
#endif
	StateInfo* const st  = pos.state();
	std::int16_t* const acc = st->p2_threat_acc[persp];
	        if constexpr (Threat::THREAT_REPRESENTATIVE_ROWS) {
                bool stable_representatives = false;
                bool direct_underfilled_delta = false;
                if (plan.base == nullptr) {
                        P2_TSTAT(++g_threat_stats.rep12_no_base);
                } else if (plan.nchain != 1 || plan.chain[0] != st) {
                        P2_TSTAT(++g_threat_stats.rep12_nonlocal_chain);
                } else if (rep8_apply_underfilled_delta(st, persp, plan.mirror, acc)) {
                        direct_underfilled_delta = true;
                        P2_TSTAT(++g_threat_stats.rep12_direct);
                        P2_TSTAT(++g_threat_stats.delta_calls);
                } else if (rep8_dirty_preserves_representatives(st, persp, plan.mirror)) {
                        stable_representatives = true;
                        P2_TSTAT(++g_threat_stats.rep12_stable);
                        P2_TSTAT(++g_threat_stats.delta_calls);
                        std::memcpy(acc, threat_acc_ptr(plan.base, persp), sizeof(std::int16_t) * THREAT_ACC_DIM);
                        st->p2_threat_rep_count[persp] = plan.base->p2_threat_rep_count[persp];
                        std::memcpy(st->p2_threat_rep_idx[persp], plan.base->p2_threat_rep_idx[persp],
                                    sizeof(st->p2_threat_rep_idx[persp]));
                } else {
                        P2_TSTAT(++g_threat_stats.rep12_changed);
                }
                if (!stable_representatives && !direct_underfilled_delta) {
                        tstat_fallback_distance(st, persp);
                        std::memset(acc, 0, sizeof(std::int16_t) * THREAT_ACC_DIM);
                        add_threat_perspective(pos, persp, plan.mirror, acc,
                                               (std::size_t)plan.group * (std::size_t)THREAT_DIM);
                }
        } else if (plan.base != nullptr) {
#if defined(P2_THREAT_STATS_BUILD)
                if (threat_stats_env()) {
                        if (dirty_chain_net_zero_for_stats(plan.chain, plan.nchain, persp, plan.mirror))
                                ++g_threat_stats.zero_delta_calls;
                        if (dirty_chain_no_relevant_for_stats(plan.chain, plan.nchain, persp, plan.mirror))
                                ++g_threat_stats.no_relevant_delta_calls;
                }
#endif
                P2_TSTAT(++g_threat_stats.delta_calls);
                P2_TSTAT(++g_threat_stats.chain_len[plan.nchain < 9 ? plan.nchain : 9]);
                if constexpr (Threat::THREAT_PROFILE_ID == 11) {
                        if (dirty_chain_no_relevant_cached(plan.chain, plan.nchain, persp, plan.mirror)) {
                                st->p2_threat_source[persp] = plan.base->p2_threat_source[persp];
                        } else {
                                std::memcpy(acc, threat_acc_ptr(plan.base, persp), sizeof(std::int16_t) * THREAT_ACC_DIM);
                                apply_dirty_chain(plan.chain, plan.nchain, persp, plan.mirror, acc,
                                                  (std::size_t)plan.group * (std::size_t)THREAT_DIM);
                                st->p2_threat_source[persp] = st;
                        }
                } else {
                        std::memcpy(acc, threat_acc_ptr(plan.base, persp), sizeof(std::int16_t) * THREAT_ACC_DIM);
                        apply_dirty_chain(plan.chain, plan.nchain, persp, plan.mirror, acc,
                                          (std::size_t)plan.group * (std::size_t)THREAT_DIM);
                        st->p2_threat_source[persp] = st;
                }
        } else {
                tstat_fallback_distance(st, persp);
                std::memset(acc, 0, sizeof(std::int16_t) * THREAT_ACC_DIM);
                // 2026-10-07 (S2): 群は plan から引く (**get_layer_stack_index を呼ばない** = §1.3)。
                const std::size_t ptto = (std::size_t)plan.group * (std::size_t)THREAT_DIM;
                if (threat_naive_env()) add_threat_perspective_naive(pos, persp, plan.mirror, acc, ptto);
                else                    add_threat_perspective(pos, persp, plan.mirror, acc, ptto);
        }
if (st->p2_threat_source[persp] == nullptr) st->p2_threat_source[persp] = st;

if (threat_verify_env()) verify_threat_acc(pos, persp, plan);

// 2026-10-07 (S2): この acc を作った表の群を記録する (子の base 採用条件に使われる)。
st->p2_threat_group[persp] = plan.group;
st->p2_threat_valid[persp] = true;
}

// persp éä¹ãç¸ºE® p2_threat_acc[persp] ç¹§å¸â¦èï½¶éªè²E®å±ãéEºè³æºâEèä¹ï½ç¹§ã»dirty-threat chain è¾E®è»ã»or full)ç¸²ã»

#if defined(P2_ADDR_DUMP)
// **① アドレス dump (設計 fable 2026-11-06 / env `P2_ADDR_DUMP=1`)**。 案 6 (整列の棚卸し) の前提。
//   `Net` は `make_unique` (16 B 整列) の中に `fc0_bias[15]` (60 B) / `fc1_bias` / `fc2_bias` /
//   `fc2_w` が inline で入っているので、 64 B line を跨いでいる可能性がある。 跨いでいれば
//   評価 1 回ごとに余分な line を読む (★fc0_weights_were_misaligned の前例: 48 mod 64 に落ちていた)。
//   跨ぎが見つかったら `alignas(64)` を付ける (無料)。
//   🚨診断専用。 本番 DEFINES には入れない (出力するだけなので値は変わらないが、 分岐が増える)。
namespace P2NNUE {
static void addr_dump_once() {
	static bool done = false;
	if (done) return;
	done = true;
	const char* e = std::getenv("P2_ADDR_DUMP");
	if (!e || !*e) return;
	const Net& net = *g_net;
	auto r  = [](const void* p) { return (unsigned)((std::uintptr_t)p % 64); };
	auto pg = [](const void* p) { return (unsigned)((std::uintptr_t)p % 4096); };
	sync_cout << "info string ADDR ft_bias %64=" << r(net.ft_bias.data())
	          << " ft_weight %4096=" << pg(net.ft_weight.data())
	          << " threat_weight %4096=" << pg(net.threat_weight.data())
	          << " psqt_weight %4096=" << pg(net.psqt_weight.data()) << sync_endl;
	for (int b = 0; b < NUM_BUCKETS; ++b)
		sync_cout << "info string ADDR bucket " << b
		          << " fc0_bias %64=" << r(net.buckets[b].fc0_bias)
		          << " fc0_ws %64=" << r(net.buckets[b].fc0_ws.data())
		          << " fc1_bias %64=" << r(net.buckets[b].fc1_bias)
		          << " fc1_ws %64=" << r(net.buckets[b].fc1_ws.data())
		          << " fc2_bias %64=" << r(net.buckets[b].fc2_bias)
		          << " fc2_w %64=" << r(net.buckets[b].fc2_w.data()) << sync_endl;
	sync_cout << "info string ADDR g_norm_sq %64=" << r(g_norm_sq)
	          << " acc_cache %64=" << r(g_acc_cache.e[0][0][0].acc)
	          << " sizeof(AccCacheEntry)=" << sizeof(AccCacheEntry)
	          << " g_net %64=" << r(g_net.get()) << " sizeof(Net)=" << sizeof(Net)
	          << " sizeof(StateInfo)=" << sizeof(StateInfo) << sync_endl;
}
} // namespace P2NNUE
#endif

#include "p2nnue_lz.h"
#if defined(P2_KH)
#include "p2nnue_kh.h"   // KH の H 部 (2026-11-12)
#endif

static void ensure_threat(const Position& pos, Color persp, int bucket_hint = -1) {
	StateInfo* st = pos.state();
	if (st->p2_threat_valid[persp]) return;
	ThreatPlan plan;
	// 2026-10-07 (S2): 群化しない net では bucket を**計算しない** (従来と同じ動作)。
	const std::uint8_t grp = (g_net->threat_groups <= 1) ? (std::uint8_t)0
		: threat_group_b(bucket_hint >= 0 ? bucket_hint : get_layer_stack_index(pos));
	resolve_threat_plan(pos, persp, plan, grp);
	apply_threat_plan(pos, persp, plan);
}

// è³E¡éä¹ãç¹§åµâªç¸ºE¨ç¹§ââ»éï½ºè³å£¹âE¹§ä¹ÂãE½¸E¡è­E½¹ full ç¸ºE«ç¸ºEªç¹§ä¹âEç¸ºéªâ¡ç¸ºå¤§ã»è¬å¶EE1 èæ§­âèE±è­å³¨âE¹§ã»
// (evaluate / priming ç¸ºE¯ç¸ºE©ç¸ºE¡ç¹§å³¨Eè¶E¸ç¸ºE«è³E¡éä¹ãç¹§è²ï½¿ã»E¦ââEç¸ºå¶Eç¸ºE®ç¸ºE§ç¸²âï¼E¹§å¾â²è­ï½¢è³å£¹ã»èï½¥è¿E£)ç¸²ã»
// 2026-09-15 (A2): bucket_hint >= 0 なら群化 net の表 offset をその bucket から引く (get_layer_stack_index の再計算を省く)。
//   -1 (既定) なら従来どおり局面から計算する。 offset は両 persp 共通 (局面の bucket で決まる)。
// 2026-09-15 (A3): noinline。 evaluate_internal (フレーム 2,200B) に inline されると合算で 4KB を超えて
//   ___chkstk_ms が evaluate_internal 側に復活する (PGO は inline 判断が変わるので明示する)。
//   本関数のフレームは idx 配列 2,560B + ThreatPlan x2 (416B) + spill で ~3.5KB < 4KB。
static __attribute__((noinline)) void ensure_threat_both(const Position& pos, int bucket_hint = -1) {
	StateInfo* const st = pos.state();
#if defined(P2_ADDR_DUMP)
	P2NNUE::addr_dump_once();
#endif
#if defined(P2_PROBE_BUILD)
	++g_pc_ensure_both;
#endif
#if defined(P2_THREAT_LZ)
	// **(B) 保険 (2026-11-06 / レビュー fable)**: valid 判定に**群の一致**を要求する。
	//   従来経路では valid な node は必ず自分の群で作られるのでこの guard は一度も発火しない
	//   (= bit-exact)。 LZ の物質化が万一別の群で上書きしても、 ここで full 再計算に落ちて守る。
	//   🚨新たに get_layer_stack_index を呼ばない (bucket_hint は evaluate_internal から常に >= 0。
	//     -1 で来るのは priming だけで本番では死んでいる = 不変条件 4)。
	{
		const std::uint8_t g0 = (g_net->threat_groups <= 1) ? (std::uint8_t)0
			: threat_group_b(bucket_hint >= 0 ? bucket_hint : get_layer_stack_index(pos));
		if (st->p2_threat_valid[BLACK] && st->p2_threat_group[BLACK] != g0)
			st->p2_threat_valid[BLACK] = false;
		if (st->p2_threat_valid[WHITE] && st->p2_threat_group[WHITE] != g0)
			st->p2_threat_valid[WHITE] = false;
	}
#endif
	const bool need_b = !st->p2_threat_valid[BLACK];
	const bool need_w = !st->p2_threat_valid[WHITE];
	if (!need_b && !need_w) {
#if defined(P2_PROBE_BUILD)
		++g_pc_early_ret;
#endif
#if defined(P2_THREAT_LZ) && defined(P2_FABLE_PROBE)
		++lz::probe::n_early;   // 2026-11-11 (N0 / LZSPLIT)
#endif
		return;
	}
#if defined(P2_THREAT_LZ) && defined(P2_FABLE_PROBE)
	if (!need_b || !need_w) ++lz::probe::n_one;   // 2026-11-11 (N0 / LZSPLIT)
#endif
	if (!need_b || !need_w) { ensure_threat(pos, need_b ? BLACK : WHITE, bucket_hint); return; }

	// 2026-10-07 (S2 §3.1): 現局面の群は evaluate_internal が既に持っている bucket から引く。
	//   🚨**新たに get_layer_stack_index を呼ばない**。 S0 で確定したとおり、 ensure_progress が
	//     走る node の集合を変えると p2_prog_sum の float 加算経路が変わり bit-exact が壊れる。
	//   群化しない net では bucket を計算しない (従来と同じ動作)。
	const std::uint8_t cur_group = (g_net->threat_groups <= 1) ? (std::uint8_t)0
		: threat_group_b(bucket_hint >= 0 ? bucket_hint : get_layer_stack_index(pos));

#if defined(P2_THREAT_LZ)
	// **多段 lazy 差分 (LZ)**: 本番 eval の 99.2% が full 再計算 (玉以外の全駒を走査) だったのを、
	//   acc が valid な最寄りの祖先からの差分で済ませる。
	//   設計 fable: runs/prereg/DESIGN-20261106-THREAT-LZ.md / 実装 opus 2026-11-06。
	//   🚨同じ値を別経路で作るだけなので **bit-exact** (関門 G1/G2 で検証する)。
	//   失敗 (祖先が無い / 鏡像が不一致 / 対が上限超え) のときは下の従来経路にそのまま落ちる。
	{
		const bool mir_b = is_hm_mirror(pos.square<KING>(BLACK), BLACK);
		const bool mir_w = is_hm_mirror(pos.square<KING>(WHITE), WHITE);
		const std::size_t tto_lz = (std::size_t)cur_group * (std::size_t)THREAT_DIM;
		if (lz::resolve(pos, st, mir_b, mir_w, tto_lz, cur_group))
			return;
	}
#endif
#if defined(P2_THREAT_LZ) && defined(P2_FABLE_PROBE)
	// 2026-11-11 (N0 / LZSPLIT): LZ が落ちたあとの full 再計算の cycle。 関数末尾までを RAII で測る。
	struct LzFullTimer { std::uint64_t t0 = __rdtsc(); ~LzFullTimer() { lz::probe::cyc_full += __rdtsc() - t0; ++lz::probe::n_full; } } lz_full_timer_;
#endif

	ThreatPlan pb, pw;
	resolve_threat_plan(pos, BLACK, pb, cur_group);
	resolve_threat_plan(pos, WHITE, pw, cur_group);

	if (pb.base == nullptr && pw.base == nullptr
	    && !threat_naive_env()) {
		// ç¬ãEE¸E¡éä¹ã full ç«ãEç¹§E¹ç¹§E­ç¹ï½£ç¹ï½³ç¹§ã»1 èæ§­ãèE±è­ãEindex è¹æ¶â¡ç¸ºé¡ï½¦ä¹ãè»E¥)ç¸²ã»
#if defined(P2_PROBE_BUILD)
		++g_pc_full_block;
#endif
		std::int16_t* const ab = st->p2_threat_acc[BLACK];
		std::int16_t* const aw = st->p2_threat_acc[WHITE];
		tstat_fallback_distance(st, BLACK);
		tstat_fallback_distance(st, WHITE);
		std::memset(ab, 0, sizeof(std::int16_t) * THREAT_ACC_DIM);
		std::memset(aw, 0, sizeof(std::int16_t) * THREAT_ACC_DIM);
		// 2026-10-07 (S2): 上で引いた群から直に作る (従来の 2 経路と同値 = bit-exact)。
		const std::size_t tto = (std::size_t)cur_group * (std::size_t)THREAT_DIM;
#if defined(P2_PROBE_BUILD)
		// 2026-10-07: 親子で群 / bucket が変わる率。 「群が同じときだけ差分」のカバー率そのもの。
		{
			const int _g = (int)tto;
			const int _bk = (bucket_hint >= 0) ? bucket_hint : get_layer_stack_index(pos);
			const int _ply = pos.game_ply();
			if (_ply > 0 && _ply < 1024) {
				if (g_grp_ok[_ply - 1]) {
					++g_pc_grp_cmp;
					if (g_grp_at[_ply - 1] == _g)  ++g_pc_grp_same;
					if (g_bkt_at[_ply - 1] == _bk) ++g_pc_bkt_same;
				}
				// ---- M2 (§2.2): 方式 (a) (lazy, priming 無し) で差分が使えたはずの件数 ----
				//   chain は親までの 1 本しか伸びない (tuple は現局面しか作れない) ので、
				//   条件は 親 valid ∧ 群一致 ∧ HM mirror 一致 の 3 つだけ。
				//   mirror は resolve_threat_plan_impl の parent_ok と同じ式で親の玉位置を復元する。
				//   🚨親の群は「直前に ply-1 で full 経路を通った局面」の記録 (DFS なので親)。
				//     親が full 経路を通っていない場合は g_grp_ok が古い = grp_cmp と同じ既知の制約。
				const StateInfo* const _pv = st->previous;
				if (_pv != nullptr && st->dirtyPiece.dirty_num != 0) {
					++g_pc_m2_den;
					const bool _grp = (g_grp_ok[_ply - 1] != 0 && g_grp_at[_ply - 1] == _g);
					const Move _m = st->lastMove;
					int _nok = 0;
					for (Color _c = BLACK; _c <= WHITE; ++_c) {
						if (!_pv->p2_threat_valid[_c]) continue;
						++g_pc_pvalid;
						if (!_grp) continue;
						++g_pc_pv_grp;
						const Square _k = pos.square<KING>(_c);
						Square _kp = _k;
						if (!_m.is_drop() && _m.to_sq() == _k && st->lastMovedPieceType == KING)
							_kp = _m.from_sq();
						if (is_hm_mirror(_kp, _c) == is_hm_mirror(_k, _c)) { ++g_pc_pv_mir; ++_nok; }
					}
					if (_nok == 2) ++g_pc_both_ok;
				}
				// ---- funnel (祖先版。 方式 (b) eager の上限も出す) ----
				//   resolve_threat_plan_impl の chain walk を**そのまま**再現する (玉の巻き戻し込み)。
				//   ただし p2_dt_state は見ない (素では全部 DT_UNBUILT なので、 tuple があったらの上限)。
				//   🚨祖先の群は g_grp_at[ply-d] (DFS なので現在の経路の祖先) を代理に使う。
				if (probe_m2()) {
					++g_pc_m2_calls;
					for (Color _c = BLACK; _c <= WHITE; ++_c) {
						const Square _k0 = pos.square<KING>(_c);
						const bool _mir0 = is_hm_mirror(_k0, _c);
						const StateInfo* _s = st;
						Square _kc = _k0;
						int _d = 0;
						for (;;) {
							if (_s->p2_threat_valid[_c]) break;
							if (_s->previous == nullptr || _d >= P2_DT_CHAIN_MAX) { _s = nullptr; break; }
							++_d;
							if (_s->dirtyPiece.dirty_num != 0) {
								const Move _mm = _s->lastMove;
								if (!_mm.is_drop() && _mm.to_sq() == _kc && _s->lastMovedPieceType == KING)
									_kc = _mm.from_sq();
							}
							_s = _s->previous;
						}
						if (_s == nullptr || _d == 0) continue;
						++g_pc_a_any;
						if (_d == 1) ++g_pc_a1_any;
						const int _h = (_d < 10) ? (_d - 1) : 9;
						++g_pc_a_d[_h];
						if (is_hm_mirror(_kc, _c) != _mir0) continue;
						++g_pc_a_mir; ++g_pc_a_dm[_h];
						if (_d == 1) ++g_pc_a1_mir;
						const int _ai = _ply - _d;
						const bool _ag = (_ai >= 0 && g_grp_ok[_ai] != 0 && g_grp_at[_ai] == _g);
						if (!_ag) continue;
						++g_pc_a_grp;
						if (_d == 1) ++g_pc_a1_grp;
					}
				}
				g_grp_at[_ply] = _g; g_bkt_at[_ply] = _bk; g_grp_ok[_ply] = 1;
			}
		}
#endif
#if defined(P2_PROBE_BUILD)
		// 2026-10-07 M1 (§2.1): tuple 構築費用だけを測る。 st には書かないので bit-exact。
		//   full 経路 1 回あたり 1 回だけ呼ぶ (= 差分化したときに払う構築の回数と同じ正規化)。
		if (probe_dt_sink() && st->previous != nullptr && st->dirtyPiece.dirty_num != 0)
			dt_build_cost_sink(pos);
		// 計測専用: ZERO は列挙を呼ばない (acc は上の memset で 0)、 REPEAT は同じ acc に k 回 (ENUMONLY 併用で値不変)。
		if (!probe_enum_zero()) {
			const int _k = probe_enum_repeat();
			for (int _rep = _k; _rep > 0; --_rep) {
				g_rd_iter = _k - _rep;   // 0 始まり。 P2RdGuard が反復ごとに cycles を分ける
				add_threat_dual(pos, pb.mirror, pw.mirror, ab, aw, tto);
			}
			g_rd_iter = 0;
		}
#else
		add_threat_dual(pos, pb.mirror, pw.mirror, ab, aw, tto);
#endif
		st->p2_threat_source[BLACK] = st;

		st->p2_threat_source[WHITE] = st;

		// 2026-10-07 (S2): full で作った表の群を記録する (子の base 採用条件に使われる)。
		st->p2_threat_group[BLACK] = cur_group;
		st->p2_threat_group[WHITE] = cur_group;
		st->p2_threat_valid[BLACK] = true;

		st->p2_threat_valid[WHITE] = true;
		if (threat_verify_env()) {
			verify_threat_acc(pos, BLACK, pb);
			verify_threat_acc(pos, WHITE, pw);
		}
		return;
	}
	apply_threat_plan(pos, BLACK, pb);
	apply_threat_plan(pos, WHITE, pw);
}

// ---------------------------------------------------------------------------
// AUX(KZ+HANG+STATKD)èEEE¸å¼±ã»è·åãE(fresh éªè²E®å±ã»é«±ã»incremental)
// ---------------------------------------------------------------------------
// è³E¡éä¹ãç¸ºE® aux èEEE¸å¼±EEfresh éªè²E®å±E  out_stm / out_nstm(è·ã»FT_DIM ç¸ºE® i16)ç¸ºE«è­ï½¸ç¸ºä¸Âã»
// seed_* ç¸ºç¢æ¼ null ç¸ºEªç¹§å³¨âç¹§å¾ãè»æEè¹ã»threat acc + aux ç¸ºE®è·åãE)ç¸²E½ull ç¸ºEªç¹§ã»0 è»æEè¹æ¶Âã»
// aux ç¸ºE¯éï½ºè½£E« ~35/perspç¹ï½»è»E©ç¸ºéªã»ç¹ãEã» 1 èæ§­ãè³å³¨E¥ç¸²EEcremental è¹æ¶âç¸ºå£½E¯ã»eval éªè²E®å±âE¹§ä¹Âã»
static void compute_aux_both(const Position& pos, Color stm, Color nstm,
                             std::int16_t* __restrict out_stm, std::int16_t* __restrict out_nstm,
                             const std::int16_t* seed_stm, const std::int16_t* seed_nstm) {
	const Net& net = *g_net;
	const Aux::AttackMaps maps(pos); // è»E©ç¸ºéªã»ç¹ãEã»ç¸ºE¯ board èæE­ã»éä¹ãé«±æ¨ï½¾æï½­ã»ç«ãE1 èæ§­â¡ç¸ºéåEè¬ãE
	const bool enum_only = aux_enumonly_env();

	// ��2026-09-09: out_* �� transform_pair_sum �� FT_DIM ���œǂށB seed (threat acc) ��
	//   �擪 THREAT_ACC_DIM ���������̂ŁA �c��� 0 ���߂��� (K == FT_DIM �Ȃ� no-op)�B
	if (seed_stm) std::memcpy(out_stm, seed_stm, sizeof(std::int16_t) * THREAT_ACC_DIM);
	else          std::memset(out_stm, 0, sizeof(std::int16_t) * THREAT_ACC_DIM);
	if constexpr (THREAT_ACC_DIM < FT_DIM)
		std::memset(out_stm + THREAT_ACC_DIM, 0, sizeof(std::int16_t) * (FT_DIM - THREAT_ACC_DIM));
	Aux::for_each_aux_feature(pos, maps, stm, [&](int local) {
		if (!enum_only) acc_add_threat_row(out_stm, aux_row(net, (std::size_t)local));
	});

	if (seed_nstm) std::memcpy(out_nstm, seed_nstm, sizeof(std::int16_t) * THREAT_ACC_DIM);
	else           std::memset(out_nstm, 0, sizeof(std::int16_t) * THREAT_ACC_DIM);
	if constexpr (THREAT_ACC_DIM < FT_DIM)
		std::memset(out_nstm + THREAT_ACC_DIM, 0, sizeof(std::int16_t) * (FT_DIM - THREAT_ACC_DIM));
	Aux::for_each_aux_feature(pos, maps, nstm, [&](int local) {
		if (!enum_only) acc_add_threat_row(out_nstm, aux_row(net, (std::size_t)local));
	});
}

// bit-exact è®æE½¨E¼éï½¨: è¿´E¾è»Âé«±E¢ç¸ºE® aux éï½ºè½£E« local index ç¹§ã»STM/NSTM è»E¥ç¸ºE«è­ãEE°ã»E¡E¨éï½º
// (bullet example shogi_aux_canonical ç¸ºE®èE½ºèå¸âEè¾£E§è·ã»ç¸²ã»
void dump_aux_indices(const Position& pos) {
	Color stm  = pos.side_to_move();
	Color nstm = ~stm;
	Aux::AttackMaps maps(pos);
	auto dump = [&](Color persp, const char* tag) {
		std::vector<int> idxs;
		Aux::for_each_aux_feature(pos, maps, persp, [&](int local) { idxs.push_back(local); });
		std::sort(idxs.begin(), idxs.end());
		std::string s;
		for (std::size_t i = 0; i < idxs.size(); ++i) {
			s += std::to_string(idxs[i]);
			if (i + 1 < idxs.size()) s += ",";
		}
		sync_cout << "info string p2aux " << tag << " count=" << idxs.size()
		          << " indices=" << s << sync_endl;
	};
	dump(stm,  "STM");
	dump(nstm, "NSTM");
}

// bit-exact è®æE½¨E¼éï½¨: è¿´E¾è»Âé«±E¢ç¸ºE® auxcheap éï½ºè½£E« index(STM/NSTM è»E¥è­ãEE°ã»ç¸ºE¨ STATKD-kd2 ç¸ºE®
// é¨E¶èE½¾æ¿¶E²ç¹§E«ç¹§E¦ç¹ï½³ç¹ãEcnt[c][tc][st][kb] ç¹§å®ï½¡E¨éï½ºç¸²E´cratchpad/statkd_bbverify.py bitboardè¿ãE
// (kd2 merge)ç¸ºE¨è¾£E§è·å»âE¹§ä¹ÂEE_AUXCHEAP è¾ï½¡èï½¹è­ãã» notice ç¸ºE®ç¸ºE¿ç¸²ã»
void dump_auxcheap_indices(const Position& pos) {
#if defined(P2_AUXCHEAP)
	Color stm  = pos.side_to_move();
	Color nstm = ~stm;
	AuxCheap::Raw raw;
	AuxCheap::build_raw(pos, raw);
	auto dump = [&](Color persp, const char* tag) {
		std::vector<int> idxs;
		AuxCheap::for_each(raw, persp, [&](int local) { idxs.push_back(local); });
		std::sort(idxs.begin(), idxs.end());
		std::string s;
		for (std::size_t i = 0; i < idxs.size(); ++i) {
			s += std::to_string(idxs[i]);
			if (i + 1 < idxs.size()) s += ",";
		}
		sync_cout << "info string p2auxcheap " << tag << " count=" << idxs.size()
		          << " indices=" << s << sync_endl;
	};
	dump(stm,  "STM");
	dump(nstm, "NSTM");
	// STATKD-kd2 C1 é¨E¶èE½¾æ¿¶E²ç¹§E«ç¹§E¦ç¹ï½³ç¹ãEé¬EE c=0..1, tc=0..8, st'=0..1{att,def1}, kb=0..1 = 72 èï½¤)ç¸²ã»
	std::string sc;
	for (int c = 0; c < 2; ++c)
		for (int tc = 0; tc < AuxCheap::SKD_NTC; ++tc)
			for (int st = 0; st < AuxCheap::SKD_NST; ++st)
				for (int kb = 0; kb < AuxCheap::SKD_NKB; ++kb) {
					sc += std::to_string(raw.statkd[c][tc][st][kb]);
					if (!(c == 1 && tc == AuxCheap::SKD_NTC - 1 && st == AuxCheap::SKD_NST - 1
					      && kb == AuxCheap::SKD_NKB - 1))
						sc += ",";
				}
	sync_cout << "info string p2auxcheap STATKD_kd2_cnt order=c,tc,st,kb n=72 vals=" << sc << sync_endl;
#else
	(void)pos;
	sync_cout << "info string p2auxcheap DISABLED (build without -DP2_AUXCHEAP)" << sync_endl;
#endif
}

void dump_endgame_tempo(const Position& pos) {
#if defined(P2_ENDGAME_TEMPO)
	sync_cout << "info string p2egt black=" << EndgameTempo::compute_class(pos, BLACK)
	          << " white=" << EndgameTempo::compute_class(pos, WHITE) << sync_endl;
#else
	(void)pos;
	sync_cout << "info string p2egt DISABLED" << sync_endl;
#endif
}

// bit-exact è®æE½¨E¼éï½¨: è¿´E¾è»Âé«±E¢ STM éä¹ãç¸ºE® threat index ç¹§åãEé¬EEE¡E¨éï½º(bullet canonical vector è±ç¢ï½¼ã»ç¸²ã»
// 2026-10-01: threat accumulator (K 本) をそのまま吐く。 **幅が足りているか**の測定用。
//   出力: "thracc <persp> <v0> <v1> ... <v(K-1)>"  (persp 0=黒 1=白)
//   🚨accumulator は evaluate 内で作られるので、 まず evaluate を呼んで state を作る。
// FT accumulator が clamp [0,254] に当たっている本数を数える。 2026-10-01。
//   射影 (p_j * t_c) を足すほど外へ押し出されるので、 「p が育ちすぎると害があるか」を
//   機構として測れる。 transform_pair_sum の scalar 経路と**同じ式**で計算する。
// FT の飽和を **次元ごと**に出す。 2026-10-01。
//   平均本数だけでは「死んでいる次元がある」のか「全次元が疎なだけ」なのか区別できない。
//   1 局面 1 行、 1024 文字 (. 範囲内 / u 下限超 / o 上限超) で出して外で集計する。
void dump_ft_saturation_dim(const Position& pos) {
	if (g_net == nullptr) { sync_cout << "info string ftsatd: net が無い" << sync_endl; return; }
	(void) evaluate(pos);
	// 射影は bucket 別の表を使うので、 ここでも bucket を求める。
	const int bucket = get_layer_stack_index(pos);
	const StateInfo* st = pos.state();
	const int half = FT_HALF;
	for (int persp = 0; persp < 2; ++persp) {
		const FtAcc* a = st->p2_acc[(Color) persp];
		const std::int16_t* b = threat_acc_ptr(st, (Color) persp);
		if (a == nullptr || b == nullptr) continue;
		std::string lo(half, '.'), hi(half, '.');
		for (int j = 0; j < half; ++j) {
			int av = ft_add_thr(a[j], j < THREAT_LO_DIM ? b[j] : 0);
			int bv = ft_add_thr(a[j + half], j < THREAT_HI_DIM ? b[THREAT_LO_DIM + j] : 0);
			if constexpr (P2_HAS_TPRJ) {
				if (g_tprj_loaded) {
					const int tvl = b[(j + g_tprj_blk[j / 16]) % THREAT_ACC_DIM];
					const int tvh = b[(j + g_tprj_blk[(j + half) / 16]) % THREAT_ACC_DIM];
					av = (std::int16_t)(av + (std::int16_t)(((int) g_tprj_w[bucket][j] * tvl + 16384) >> 15));
					bv = (std::int16_t)(bv + (std::int16_t)(((int) g_tprj_w[bucket][j + half] * tvh + 16384) >> 15));
				}
			}
			lo[(std::size_t) j] = (av < 0) ? 'u' : (av > FT_CLAMP ? 'o' : '.');
			hi[(std::size_t) j] = (bv < 0) ? 'u' : (bv > FT_CLAMP ? 'o' : '.');
		}
		sync_cout << "ftsatd " << persp << " " << lo << hi << sync_endl;
	}
}

void dump_ft_saturation(const Position& pos) {
	if (g_net == nullptr) { sync_cout << "info string ftsat: net が無い" << sync_endl; return; }
	(void) evaluate(pos);
	// 射影は bucket 別の表を使うので、 ここでも bucket を求める。
	const int bucket = get_layer_stack_index(pos);
	const StateInfo* st = pos.state();
	const int half = FT_HALF;
	for (int persp = 0; persp < 2; ++persp) {
		const FtAcc* a = st->p2_acc[(Color) persp];
		const std::int16_t* b = threat_acc_ptr(st, (Color) persp);
		if (a == nullptr || b == nullptr) continue;
		int under = 0, over = 0;
		for (int j = 0; j < half; ++j) {
			int av = ft_add_thr(a[j], j < THREAT_LO_DIM ? b[j] : 0);
			int bv = ft_add_thr(a[j + half], j < THREAT_HI_DIM ? b[THREAT_LO_DIM + j] : 0);
			if constexpr (P2_HAS_TPRJ) {
				if (g_tprj_loaded) {
					const int tvl = b[(j + g_tprj_blk[j / 16]) % THREAT_ACC_DIM];
					const int tvh = b[(j + g_tprj_blk[(j + half) / 16]) % THREAT_ACC_DIM];
					av = (std::int16_t)(av + (std::int16_t)(((int) g_tprj_w[bucket][j] * tvl + 16384) >> 15));
					bv = (std::int16_t)(bv + (std::int16_t)(((int) g_tprj_w[bucket][j + half] * tvh + 16384) >> 15));
				}
			}
			if (av < 0) ++under; else if (av > FT_CLAMP) ++over;
			if (bv < 0) ++under; else if (bv > FT_CLAMP) ++over;
		}
		sync_cout << "ftsat " << persp << " under " << under << " over " << over
		          << " total " << (half * 2) << sync_endl;
	}
}

void dump_threat_acc(const Position& pos) {
	if (g_net == nullptr || !g_net->has_threat) {
		sync_cout << "info string thracc: threat net ではない" << sync_endl;
		return;
	}
	// accumulator を作らせる (返り値は使わない)
	(void) evaluate(pos);
	const StateInfo* st = pos.state();
	for (int persp = 0; persp < 2; ++persp) {
		const std::int16_t* a = threat_acc_ptr(st, (Color) persp);
		if (a == nullptr) continue;
		std::string line = "thracc " + std::to_string(persp);
		for (int i = 0; i < THREAT_ACC_DIM; ++i) { line += " "; line += std::to_string((int) a[i]); }
		sync_cout << line << sync_endl;
	}
}

void dump_threat_indices(const Position& pos) {
	g_threat_tables.build();
	build_norm_sq();
	Color stm = pos.side_to_move();
	bool mirror = is_hm_mirror(pos.square<KING>(stm), stm);
	std::vector<int> idxs;
	for_each_threat_feature(pos, stm, mirror, [&](int idx) { idxs.push_back(idx); });
	std::sort(idxs.begin(), idxs.end());
	std::string s;
	for (std::size_t i = 0; i < idxs.size(); ++i) {
		s += std::to_string(idxs[i]);
		if (i + 1 < idxs.size()) s += ",";
	}
	sync_cout << "info string p2threat stm=" << (int)stm << " count=" << idxs.size()
	          << " indices=" << s << sync_endl;
}

#if defined(P2_FABLE_PROBE)
namespace FableProbe { extern std::uint64_t rows_chain, n_chain, rows_refresh, n_refresh; }
#endif
// 2026-11-05 (fable) P2_FT_PREFETCH_ALL: acc 更新は「行の数 x 初回アクセスの遅延」で律速されている
//   (probe: 評価 1 回に 15-29 行、 i8 化で byte を半減しても速くならなかった = 帯域でなく遅延)。
//   k ループは行を 128B ずつ横断するので、 HW の L2 streamer が各行で立ち上がるまでの遅延が行数ぶん乗る。
//   ループの前に全行・全ラインを T1 (L2) に prefetch して、 まとめて memory-level parallelism を取る。
//   bit-exact の根拠: prefetch は値を変えない。 守るべき不変条件: 行の範囲 (FT_ROW_BYTES) を越えない。
#if defined(P2_FT_PREFETCH_ALL)
static inline void prefetch_rows_all(const void* const* rows, int n) {
	constexpr int LINES = (int)((FT_ROW_BYTES + 63) / 64);
	for (int i = 0; i < n; ++i) {
		const char* p = static_cast<const char*>(rows[i]);
		for (int l = 0; l < LINES; ++l) _mm_prefetch(p + 64 * l, _MM_HINT_T1);
	}
}
#endif
#if defined(P2_OPN) && defined(P2_OPN_R2)
// ---------------------------------------------------------------------------
// R2 (2026-11-15 Fable 第 2 ラウンド / runs/prereg/OPT2-20261115-OPN2048-FABLE.md)。 -DP2_OPN_R2 で入る。 どれも値を変えない。
//   個別に戻すときは下の P2_OPN_R2_* を -D...=0 で渡す。 -DP2_OPN_R2T は計測ビルド (本番に入れない)。
//   前提: -DP2_OPN_F2 と USE_AVX2 (無ければ従来の経路のまま)。
// ---------------------------------------------------------------------------
#if !defined(P2_OPN_R2_G8_OPN)
#define P2_OPN_R2_G8_OPN 1         // R2-1: OPN の fc0 を G8 (chunk 8 個ぶんを i16 で足す) にする
#endif
#if !defined(P2_OPN_R2_TRFX)
#define P2_OPN_R2_TRFX 1           // R2-2: OPN の差分更新 + transform を定数版で回す
#endif
#if !defined(P2_OPN_R2_LZX)
#define P2_OPN_R2_LZX 1            // R2-2: OPN の親の遅延作成 (F2-5) を定数版で回す
#endif
#if !defined(P2_OPN_R2_RFTR)
#define P2_OPN_R2_RFTR 1           // R2-3: OPN の Finny refresh と transform を 1 回の列ループで行う
#endif
#if !defined(P2_OPN_R2_RFX)
#define P2_OPN_R2_RFX 1            // R2-2: OPN の Finny refresh を定数版 (4 対ずつ) で回す。 P2_OPN_R2_RFTR が前提
#endif
#if !defined(P2_OPN_R2_MFX)
#define P2_OPN_R2_MFX 1            // R2-2: 本体の差分更新を定数版で回す
#endif
#if !defined(P2_OPN_R2_MREF)
#define P2_OPN_R2_MREF 1           // R2-4: 本体の Finny refresh を 64 列ごと 4 レジスタにする
#endif
#if defined(P2_OPN_R2T)
// 計測ビルド: 上のスイッチを実行時の値にし、 OPNR2_AB=<番号>:<A>:<B> で OPNR2_BLK (既定 4096) 評価ごとに A / B を交互に切り替える。
//   OPNR2_SET=<番号>=<値>,... で固定値。 番号は下の enum の順 (0 = G8O ... 6 = MREF)。
//   区間ごとの rdtsc を状態 (A = 0 / B = 1) ごとに積み、 終了時に stderr へ出す (器 runs/opnint/fable2048r2/run1.py + r2fmt.py)。
//   1 回 150,000 cycle 超 (割り込み) は捨てる。 🚨Threads 1 専用 (計数は thread_local でない)。 本番ビルドには入らない。
namespace R2T {
enum { K_G8O, K_TRFX, K_LZX, K_RFTR, K_RFX, K_MFX, K_MREF, NK };
static int kv[NK] = { P2_OPN_R2_G8_OPN, P2_OPN_R2_TRFX, P2_OPN_R2_LZX, P2_OPN_R2_RFTR, P2_OPN_R2_RFX, P2_OPN_R2_MFX, P2_OPN_R2_MREF };
enum { R_EVAL, R_PRIME, R_OPN, R_OPNFWD, R_ACC1, R_LAZY, R_REFR, R_TRF, R_MACC, R_MTHR, R_MPROP, NR };
static std::uint64_t cyc[2][NR], cnt[2][NR];
static int state = 0, ab_k = -1, ab_v[2] = { 0, 0 }, blk = 4096;
static std::uint64_t tick_n = 0;
static inline void tick() {
	if (ab_k >= 0 && (++tick_n % (std::uint64_t) blk) == 0) { state ^= 1; kv[ab_k] = ab_v[state]; }
}
struct Scope {
	int r; std::uint64_t t0;
	explicit Scope(int r_) : r(r_), t0(__rdtsc()) {}
	~Scope() { const std::uint64_t d = __rdtsc() - t0; if (d < 150000) { cyc[state][r] += d; ++cnt[state][r]; } }
};
static void report() {
	for (int s = 0; s < 2; ++s) {
		std::fprintf(stderr, "OPNR2 s=%d ab=%d:%d", s, ab_k, ab_k >= 0 ? ab_v[s] : 0);
		for (int r = 0; r < NR; ++r) std::fprintf(stderr, " r%d=%llu/%llu", r, (unsigned long long) cnt[s][r], (unsigned long long) cyc[s][r]);
		std::fprintf(stderr, "\n");
	}
}
static struct Init {
	Init() {
		if (const char* e = std::getenv("OPNR2_SET")) {   // "k=v,k=v" (k は番号)
			while (*e) {
				const int k = std::atoi(e); const char* q = std::strchr(e, '=');
				if (q == nullptr) break;
				if (k >= 0 && k < NK) kv[k] = std::atoi(q + 1);
				const char* c = std::strchr(q, ','); if (c == nullptr) break; e = c + 1;
			}
		}
		if (const char* e = std::getenv("OPNR2_AB")) {    // "k:A:B"
			const int k = std::atoi(e); const char* q = std::strchr(e, ':');
			if (q != nullptr && k >= 0 && k < NK) {
				const char* q2 = std::strchr(q + 1, ':');
				if (q2 != nullptr) { ab_k = k; ab_v[0] = std::atoi(q + 1); ab_v[1] = std::atoi(q2 + 1); kv[k] = ab_v[0]; }
			}
		}
		if (const char* e = std::getenv("OPNR2_BLK")) { const int v = std::atoi(e); if (v > 0) blk = v; }
		std::atexit(report);
	}
} init_;
} // namespace R2T
#define R2K(name, def) (P2NNUE::R2T::kv[P2NNUE::R2T::name])
#define R2T_SCOPE(r) P2NNUE::R2T::Scope r2t_scope_##r(P2NNUE::R2T::r)
#define R2T_TICK() P2NNUE::R2T::tick()
#else
#define R2K(name, def) (def)
#define R2T_SCOPE(r) ((void) 0)
#define R2T_TICK() ((void) 0)
#endif
#if defined(P2_TR_STORE32_VERIFY)
#if !defined(P2_TR_STORE32)
#error "P2_TR_STORE32_VERIFY は P2_TR_STORE32 と一緒に使う"
#endif
// P2_TR_STORE32 の検算の計数 (診断ビルド専用)。 main = 本体の transform、 opn = OPN の transform (融合核を含む)。
namespace TrVerify {
static std::uint64_t n_main = 0, n_opn = 0;
static void report() { std::fprintf(stderr, "TR_STORE32_VERIFY main=%llu opn=%llu mismatch=0\n", (unsigned long long) n_main, (unsigned long long) n_opn); }
static struct Init { Init() { std::atexit(report); } } init_;
}
#endif
#if defined(P2_ACC_PIECEWISE_VERIFY) && !defined(P2_ACC_PIECEWISE)
#error "P2_ACC_PIECEWISE_VERIFY は P2_ACC_PIECEWISE と一緒に使う"
#endif
#if defined(USE_AVX2) && !defined(P2_FT_I8)
// R2-1: fc0 の密な積和を「入力 32 byte (chunk 8 個) ごと」に回す。
//   従来 (affine_par / affine_fc0_pf) は chunk 1 個ごとに maddubs (u8 × i8 の 2 項和 → i16) → madd (i16 2 つ → i32) → i32 の加算で、
//   1 chunk あたり乗算系の µop が 4 個 (maddubs 2 + madd 2、 どれも port 0/1) と、 入力の broadcast のロード 1 個、 重みの先読み 1 個だった。
//   ここでは chunk 8 個ぶんの maddubs の結果を i16 のまま足し、 madd と i32 の加算は 8 chunk に 1 回だけにする
//   (乗算系 µop は 1 chunk あたり 2.25 個)。 入力は 32 byte を 1 回読み、 vpermd で各 chunk (4 byte) を全 lane に広げる。
//   重みの先読みはしない (この形では先読み無しが最良だった。 2 KB 先 +130 cycle、 1 KB 先 +127 cycle)。
//   実測 (同じプロセスで 4,096 評価ごとに交互、 rdtsc、 幅 2048 × L1 16): fc0 + 後段 −290〜−390 cycle / OPN の評価 1 回。
//   値: maddubs の lane (出力 o、 chunk 内の前半 / 後半) ごとに、 8 chunk ぶんの部分和が i16 に収まるなら i16 の加算は溢れず、
//     整数の和として従来と同じ (加算の順が違うだけ)。 収まることは読み込み時に重みから確かめる (affine_g8_safe)。
//     収まらない重みの head では使わない (従来の経路)。
// SAFETY: in は in_pad byte 以上、 ws は (in_pad / 4) × NOUT × 4 byte、 bias / out は NOUT 要素。 loadu / storeu。
//   不変条件: NOUT は 8 の倍数 (8..32)、 in_pad は 32 の倍数、 入力は 127 以下 (従来の maddubs の飽和条件と同じ前提)、
//     affine_g8_safe(ws, in_pad, NOUT) == true。
template <int NOUT>
static inline void affine_g8(const std::uint8_t* __restrict in, const std::int8_t* __restrict ws, const std::int32_t* __restrict bias,
                             int in_pad, std::int32_t* __restrict out) {
	constexpr int NG = NOUT / 8;      // 8 出力ごとの lane 群
	constexpr int CB = NOUT * 4;      // 1 chunk (入力 4 byte) の重みの byte 数
	__m256i acc[NG];
	for (int g = 0; g < NG; ++g) acc[g] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + g * 8));
	const __m256i ones = _mm256_set1_epi16(1);
	const int nb = in_pad / 32;
	for (int ib = 0; ib < nb; ++ib) {
		const __m256i in8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + ib * 32));
		const std::int8_t* wb = ws + (std::size_t) ib * 8 * CB;
		__m256i s[NG], t[NG];
		for (int g = 0; g < NG; ++g) { s[g] = _mm256_setzero_si256(); t[g] = _mm256_setzero_si256(); }
		for (int c = 0; c < 8; c += 2) {
			const __m256i a = _mm256_permutevar8x32_epi32(in8, _mm256_set1_epi32(c));
			const __m256i b = _mm256_permutevar8x32_epi32(in8, _mm256_set1_epi32(c + 1));
			for (int g = 0; g < NG; ++g) {
				s[g] = _mm256_add_epi16(s[g], _mm256_maddubs_epi16(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wb + c * CB + g * 32))));
				t[g] = _mm256_add_epi16(t[g], _mm256_maddubs_epi16(b, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wb + (c + 1) * CB + g * 32))));
			}
		}
		for (int g = 0; g < NG; ++g) acc[g] = _mm256_add_epi32(acc[g], _mm256_madd_epi16(_mm256_add_epi16(s[g], t[g]), ones));
	}
	for (int g = 0; g < NG; ++g) _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + g * 8), acc[g]);
}
// affine_g8 を使ってよい重みか。 ws = scrambled [in_pad / 4][nout][4]。 lane = (出力 o、 前半 / 後半 h) ごとに、 chunk 8 個の
//   正の重みの和 P と負の重みの絶対値の和 N を取り、 127 × P <= 32767 かつ 127 × N <= 32768 なら、 入力が 0..127 のどんな値でも
//   部分和は [−127 N, 127 P] の中 = i16 に収まる。 読み込み時に 1 回だけ呼ぶ (探索中には呼ばない)。
static inline bool affine_g8_safe(const std::int8_t* ws, int in_pad, int nout) {
	if (in_pad <= 0 || in_pad % 32 != 0 || nout <= 0 || nout % 8 != 0) return false;
	const int nb = in_pad / 32;
	for (int ib = 0; ib < nb; ++ib)
		for (int o = 0; o < nout; ++o)
			for (int h = 0; h < 2; ++h) {
				int P = 0, N = 0;
				for (int c = 0; c < 8; ++c)
					for (int j = 0; j < 2; ++j) {
						const int w = ws[((std::size_t) (ib * 8 + c) * nout + o) * 4 + h * 2 + j];
						if (w > 0) P += w; else N -= w;
					}
				if (127 * P > 32767 || 127 * N > 32768) return false;
			}
	return true;
}
// R2-2: 差分更新の「引く行 N 本・足す行 N 本」を定数にした核。 OPN と本体の両方で使う。
//   汎用版 (opn::apply_rows / apply_rows_tr2、 本体の ensure_perspective の A4) は 64 列ごとに行ポインタを配列から読み直し、
//   行ごとの内側ループ (ポインタのロード・カウンタ・分岐) を回す。 1 手ぶんの差分は (1,1) = 駒を取らない手・打つ手 か
//   (2,2) = 駒を取る手 なので、 行ポインタをループの外でレジスタに置き、 64 列ブロックの中を分岐なしで処理する。
//   実測 (同じプロセスで 4,096 評価ごとに交互、 rdtsc): OPN の差分更新 + transform −220〜−250 cycle / 評価、
//     本体の差分更新 −190〜−290 cycle / 評価 (priming の分を含む)。
//   D2: dst と同じ値を dst2 にも書く (Finny の refresh)。 TR: 後半の列を作った時点で transform の 1 視点ぶんを out に書く (F2-6 と同じ)。
//   pfd > 0 なら各行の pfd 要素先 (2 line ぶん) を先読みする (OPN は F2-2 と同じ距離を渡す。 本体は 0)。
//   値: 列ごとに「src から subs を引き、 adds を足す」(i16 wrapping)。 加減の順は compiler が入れ替えうるが、 i16 wrapping の加減は
//     交換則・結合則が成り立つので値は同じ。 transform の式は opn::apply_rows_tr2 と同じ。
// SAFETY: src / dst / dst2 / 各行は W 要素以上、 out は W/2 byte 以上。 loadu / storeu なので整列は要らない。
//   不変条件: W % 64 == 0 (TR なら W % 128 == 0)。 N は 1..4。 src == dst でもよい (64 列ごとに src を読んでから dst に書き、
//     TR の前半の読み直しは書き終えた列だけを読む)。 dst と out は重ならない。
#if defined(P2_FT_NEWTR)
// P2_FT_NEWTR (11-18): i16 の累積器の 16 列 lo / hi から transform の積を作る (FT_KS / FT_MS の注記)。 OPN の融合核と transform が使う。
//   lo, hi とも clamp [0, 254k] の後に shift するので i16 に収まる (FT_SA / FT_SB の上限)。 FT_SB == 0 のときは hi の max(0) を省く
//   (hi < 0 なら積は 0 以下で、 ft_store32 が u8 は packus、 i16 は max(0) で 0 にする)。
static inline __m256i ft_prod16_i16(__m256i lo, __m256i hi) {
	const __m256i ccl = _mm256_set1_epi16((short) FT_CLAMP);
	const __m256i zero = _mm256_setzero_si256();
	lo = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(lo, ccl), zero), FT_SA);
	if constexpr (FT_SB > 0) hi = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(hi, ccl), zero), FT_SB);
	else hi = _mm256_min_epi16(hi, ccl);
	return _mm256_mulhi_epi16(lo, hi);
}
// 列の順に並んだ 16 列 × 2 (p0 = 列 0..15、 p1 = 16..31) を out に書く。 SAFETY: out は 32 要素以上 (storeu)。
static inline void ft_store32(TrT* out, __m256i p0, __m256i p1) {
#if defined(P2_FT_OUT16)
	if constexpr (FT_SB == 0) { p0 = _mm256_max_epi16(p0, _mm256_setzero_si256()); p1 = _mm256_max_epi16(p1, _mm256_setzero_si256()); }
	_mm256_storeu_si256(reinterpret_cast<__m256i*>(out), p0);
	_mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 16), p1);
#else
	_mm256_storeu_si256(reinterpret_cast<__m256i*>(out), _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p1), 0xD8));
#endif
}
#endif
template <int N, bool D2, bool TR>
static inline void r2_rows_fx(const std::int16_t* src, std::int16_t* dst, std::int16_t* dst2, int W, const std::int16_t* const* subs,
                              const std::int16_t* const* adds, TrT* out, int pfd) {
	const int half = W / 2;
	const __m256i c254 = _mm256_set1_epi16(254);
	const __m256i zero = _mm256_setzero_si256();
	const std::int16_t* s[N]; const std::int16_t* a[N];
	for (int r = 0; r < N; ++r) { s[r] = subs[r]; a[r] = adds[r]; }
	for (int k = 0; k < W; k += 64) {
		if (pfd > 0 && k + pfd < W)
			for (int r = 0; r < N; ++r) {
				_mm_prefetch(reinterpret_cast<const char*>(s[r] + k + pfd), _MM_HINT_T0); _mm_prefetch(reinterpret_cast<const char*>(s[r] + k + pfd + 32), _MM_HINT_T0);
				_mm_prefetch(reinterpret_cast<const char*>(a[r] + k + pfd), _MM_HINT_T0); _mm_prefetch(reinterpret_cast<const char*>(a[r] + k + pfd + 32), _MM_HINT_T0);
			}
		__m256i v[4];
		for (int i = 0; i < 4; ++i) {
			__m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 16 * i));
			for (int r = 0; r < N; ++r) x = _mm256_sub_epi16(x, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s[r] + k + 16 * i)));
			for (int r = 0; r < N; ++r) x = _mm256_add_epi16(x, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a[r] + k + 16 * i)));
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 16 * i), x);
			if (D2) _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 16 * i), x);
			v[i] = x;
		}
#if defined(P2_FT_NEWTR)
		if (TR && k >= half) {
			const int j = k - half;
			__m256i p[4];
			for (int i = 0; i < 4; ++i)
				p[i] = ft_prod16_i16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + j + 16 * i)), v[i]);
			ft_store32(out + j, p[0], p[1]);
			ft_store32(out + j + 32, p[2], p[3]);
		}
		(void) c254; (void) zero;
#else
		if (TR && k >= half) {
			const int j = k - half;
			__m256i p[4];
			for (int i = 0; i < 4; ++i) {
				const __m256i f = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + j + 16 * i)), c254), zero), 7);
#if defined(P2_TR_STORE32)
				// P2_TR_STORE32: hi 側の max(0) を省く (transform_pair の注記。 負なら積が 0 以下になり packus が 0 にする)。
				p[i] = _mm256_mulhi_epi16(f, _mm256_min_epi16(v[i], c254));
#else
				p[i] = _mm256_mulhi_epi16(f, _mm256_max_epi16(_mm256_min_epi16(v[i], c254), zero));
#endif
			}
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j), _mm256_permute4x64_epi64(_mm256_packus_epi16(p[0], p[1]), 0xD8));
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j + 32), _mm256_permute4x64_epi64(_mm256_packus_epi16(p[2], p[3]), 0xD8));
		}
#endif
	}
}
// n 対 (引く n 本・足す n 本) を定数版で適用する。 n が 1..4 でなければ何もせず false (呼び出し側が汎用版を使う)。
template <bool D2, bool TR>
static inline bool r2_rows_n(int n, const std::int16_t* src, std::int16_t* dst, std::int16_t* dst2, int W, const std::int16_t* const* subs,
                             const std::int16_t* const* adds, TrT* out, int pfd) {
	switch (n) {
	case 1: r2_rows_fx<1, D2, TR>(src, dst, dst2, W, subs, adds, out, pfd); return true;
	case 2: r2_rows_fx<2, D2, TR>(src, dst, dst2, W, subs, adds, out, pfd); return true;
	case 3: r2_rows_fx<3, D2, TR>(src, dst, dst2, W, subs, adds, out, pfd); return true;
	case 4: r2_rows_fx<4, D2, TR>(src, dst, dst2, W, subs, adds, out, pfd); return true;
	default: return false;
	}
}
// Finny の refresh 用: acc (in place) と dst2 に「acc − Σsubs + Σadds」を書き、 out に transform の 1 視点ぶんを書く。
//   n = 引く行数 = 足す行数。 n > 4 のときは 4 対ずつ acc を in place で更新し、 最後の 1..4 対で dst2 と out も書く。
//   値: i16 wrapping の加減は交換則・結合則が成り立つので、 対ごとに (引く・足す) を交互に適用しても、 全部引いてから全部足すのと同じ値。
//   不変条件: n >= 1。 acc と dst2 は別の配列。 W % 128 == 0。
static inline void r2_refresh_tr(int n, std::int16_t* acc, std::int16_t* dst2, int W, const std::int16_t* const* subs,
                                 const std::int16_t* const* adds, TrT* out, int pfd) {
	while (n > 4) {
		r2_rows_fx<4, false, false>(acc, acc, nullptr, W, subs, adds, nullptr, pfd);
		subs += 4; adds += 4; n -= 4;
	}
	r2_rows_n<true, true>(n, acc, acc, dst2, W, subs, adds, out, pfd);
}
#define P2_OPN_R2_KERNELS 1
#endif
#else
#define R2T_SCOPE(r) ((void) 0)
#define R2T_TICK() ((void) 0)
#endif

#if defined(P2_ACC_RANGE)
// 診断ビルド専用 (2026-11-18): transform に入る累積器の値の範囲を数える。 [0] = HalfKA の累積器 (bias + HalfKA)、
//   [1] = 最終の和 (+ threat + KH を k 倍したもの)。 終了時に stderr へ。 i16 で保てる k を決めるための計測。
namespace AccRange {
static long long mn[2] = {0, 0}, mx[2] = {0, 0};
static unsigned long long n_tr = 0, over[2][4] = {};   // over[i][s]: |v| > 32767 >> (3 - s) (= k = 2^s で i16 を超える) の列数
static void report() {
	std::fprintf(stderr, "ACC_RANGE k=%d transforms=%llu halfka[min=%lld max=%lld] final[min=%lld max=%lld]\n", FT_K, n_tr, mn[0], mx[0], mn[1], mx[1]);
	for (int i = 0; i < 2; ++i)
		std::fprintf(stderr, "ACC_RANGE %s over_i16_if_k=1,2,4,8: %llu %llu %llu %llu (of %llu cols)\n", i ? "final" : "halfka",
		             over[i][0], over[i][1], over[i][2], over[i][3], n_tr * (unsigned long long) FT_DIM);
}
static struct Init { Init() { std::atexit(report); } } init_;
static inline void add(int i, long long v) {
	if (v < mn[i]) mn[i] = v;
	if (v > mx[i]) mx[i] = v;
	// v は k = FT_K の目盛り。 k' = 2^s の目盛りに直すと v * k' / FT_K。
	const long long a = v < 0 ? -v : v;
	for (int s = 0; s < 4; ++s) if (a * (1 << s) > 32767LL * FT_K) ++over[i][s];
}
}
#endif
#if defined(P2_FT_I32)
// P2_FT_I32: dst = src − Σ subs + Σ adds (i32)。 dst2 != nullptr なら同じ値を dst2 にも書く (Finny の refresh)。
//   値: 列ごとの整数の和 (i32 は溢れない: |bias| + 40 行 × |w| <= 958 + 40 × 3691 < 2^31)。 加減の順は値に効かない。
// SAFETY: src / dst / dst2 は FT_DIM 要素 (i32)、 各行は FT_DIM 要素 (i16)。 loadu / storeu なので整列は要らない。
//   不変条件: FT_DIM % 32 == 0。 src == dst でもよい (32 列ごとに読んでから書く)。 dst2 は src / dst と重ならない。
static_assert(FT_DIM % 32 == 0, "P2_FT_I32 の核は 32 列単位");
#if !defined(P2_FT32_PF_LINES)
#define P2_FT32_PF_LINES 0   // S2 (11-18): 行の先読み距離 (64 byte line 数)。 0 = 先読みしない。 A/B で決める
#endif
// S2 (11-18) P2_FT32_FUSE16: 行を i16 のまま G 本ずつ足し引きしてから i32 に広げて足す (広げる命令を行ごとでなく G 行ごとにする)。
//   値: G 本の i16 の部分和は |Σ| <= G·max|w| <= 32767 (G = g_ft32_group、 読み込み時に max|w| から決める) なので溢れず、
//   整数の和として従来 (行ごとに広げる) と同じ。 不変条件: g_ft32_group >= 1。
static int g_ft32_group = 1;
// S2 (11-18) P2_FT32_NOINLINE: i cache の影響を見るための切り替え (ft32_apply を呼び出し側に展開しない)。 値は同じ。
#if defined(P2_FT32_NOINLINE)
__attribute__((noinline))
#endif
static inline void ft32_apply(const std::int32_t* src, std::int32_t* dst, std::int32_t* dst2,
                              const std::int16_t* const* subs, int ns, const std::int16_t* const* adds, int na) {
#if defined(USE_AVX2) && defined(P2_FT32_FUSE16)
	// SAFETY: src / dst / dst2 は FT_DIM 要素 (i32)、 各行は FT_DIM 要素 (i16)。 loadu / storeu。 prefetch は fault しない。
	//   不変条件: FT_DIM % 32 == 0。 src == dst でもよい。 dst2 は src / dst と重ならない。
	const int G = g_ft32_group;
	for (int k = 0; k < FT_DIM; k += 32) {
		__m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 0));
		__m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 8));
		__m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 16));
		__m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 24));
		__m256i d0 = _mm256_setzero_si256(), d1 = _mm256_setzero_si256();
		int c = 0;
		auto flush = [&]() {
			v0 = _mm256_add_epi32(v0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(d0)));
			v1 = _mm256_add_epi32(v1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(d0, 1)));
			v2 = _mm256_add_epi32(v2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(d1)));
			v3 = _mm256_add_epi32(v3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(d1, 1)));
			d0 = _mm256_setzero_si256(); d1 = _mm256_setzero_si256(); c = 0;
		};
		for (int s = 0; s < ns; ++s) {
			const std::int16_t* q = subs[s] + k;
			if (P2_FT32_PF_LINES > 0 && k + P2_FT32_PF_LINES * 32 < FT_DIM) _mm_prefetch(reinterpret_cast<const char*>(q + P2_FT32_PF_LINES * 32), _MM_HINT_T0);
			d0 = _mm256_sub_epi16(d0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 0)));
			d1 = _mm256_sub_epi16(d1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 16)));
			if (++c == G) flush();
		}
		for (int a = 0; a < na; ++a) {
			const std::int16_t* q = adds[a] + k;
			if (P2_FT32_PF_LINES > 0 && k + P2_FT32_PF_LINES * 32 < FT_DIM) _mm_prefetch(reinterpret_cast<const char*>(q + P2_FT32_PF_LINES * 32), _MM_HINT_T0);
			d0 = _mm256_add_epi16(d0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 0)));
			d1 = _mm256_add_epi16(d1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 16)));
			if (++c == G) flush();
		}
		if (c > 0) flush();
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 0), v0);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 8), v1);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 16), v2);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 24), v3);
		if (dst2 != nullptr) {
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 0), v0);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 8), v1);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 16), v2);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 24), v3);
		}
	}
#elif defined(USE_AVX2)
	for (int k = 0; k < FT_DIM; k += 32) {
		__m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 0));
		__m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 8));
		__m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 16));
		__m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k + 24));
		for (int s = 0; s < ns; ++s) {
			const std::int16_t* q = subs[s] + k;
			if (P2_FT32_PF_LINES > 0 && k + P2_FT32_PF_LINES * 32 < FT_DIM) _mm_prefetch(reinterpret_cast<const char*>(q + P2_FT32_PF_LINES * 32), _MM_HINT_T0);
			v0 = _mm256_sub_epi32(v0, _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 0))));
			v1 = _mm256_sub_epi32(v1, _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 8))));
			v2 = _mm256_sub_epi32(v2, _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 16))));
			v3 = _mm256_sub_epi32(v3, _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 24))));
		}
		for (int a = 0; a < na; ++a) {
			const std::int16_t* q = adds[a] + k;
			if (P2_FT32_PF_LINES > 0 && k + P2_FT32_PF_LINES * 32 < FT_DIM) _mm_prefetch(reinterpret_cast<const char*>(q + P2_FT32_PF_LINES * 32), _MM_HINT_T0);
			v0 = _mm256_add_epi32(v0, _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 0))));
			v1 = _mm256_add_epi32(v1, _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 8))));
			v2 = _mm256_add_epi32(v2, _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 16))));
			v3 = _mm256_add_epi32(v3, _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 24))));
		}
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 0), v0);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 8), v1);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 16), v2);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k + 24), v3);
		if (dst2 != nullptr) {
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 0), v0);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 8), v1);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 16), v2);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst2 + k + 24), v3);
		}
	}
#else
	for (int k = 0; k < FT_DIM; ++k) {
		std::int32_t v = src[k];
		for (int s = 0; s < ns; ++s) v -= subs[s][k];
		for (int a = 0; a < na; ++a) v += adds[a][k];
		dst[k] = v;
		if (dst2 != nullptr) dst2[k] = v;
	}
#endif
}
#endif

static void refresh_perspective(const Position& pos, Color persp) {
	const Net& net = *g_net;
	StateInfo* st = pos.state();
	Square ksq = pos.square<KING>(persp);
	int kb = king_bucket(ksq, persp);
	bool mirror = is_hm_mirror(ksq, persp);

	AccCache& cache = g_acc_cache;
	if (cache.generation != g_net_generation) {
		for (int p = 0; p < 2; ++p)
			for (int k = 0; k < 45; ++k)
				for (int m = 0; m < 2; ++m)
					cache.e[p][k][m].valid = false;
		cache.generation = g_net_generation;
	}
	AccCacheEntry& en = cache.e[persp][kb][mirror ? 1 : 0];
#if defined(P2_DISABLE_FINNY)
        // è±ç¢ï½¼ã»ã»ç¹ï½³ç¹â«çE è±ä¸»å±Esnapshot ç¹§å ¤âèE¹è¹æ¶E ç¸²âï½¾æºæç¸ºE®èï½¨ refresh ç¹§è²ï½¼E·è»E¶ç¸ºå¶Eç¸²ã»
        en.valid = false;
#endif
        if (!en.valid) {
		// é¨Eºé¶E¤é«±E¢(bias ç¸ºE®ç¸ºE¿)ç¸ºE¨ç¸ºå±â»è»æEè¹ã»ç«ãEè³ä¹ãEè¾E®è»ã»Îç¹ï½¼ç¹å±â²èï½¨é¬§åµEEadd ç¸ºå¶Eç¸²ã»
		for (int k = 0; k < FT_DIM; ++k) en.acc[k] = net.ft_bias[k];
		for (int b = 0; b < NUM_BUCKETS; ++b) en.psqt[b] = net.psqt_bias[b];
		for (int i = 0; i < PIECE_NUMBER_NB; ++i) en.bp[i] = (std::int32_t)BONA_PIECE_ZERO;
		en.valid = true;
	}

	// ç¹§E¹ç¹ç¿«ã£ç¹å±ãç¹ï½§ç¹ãEã¨ç¸ºE¨ç¸ºE®è¾E®è»ã»åº¶é«®ã»ç¹§E¹ç¹ï½­ç¹ãEã¨èåE½½ã»ç¸²ã»
	const BonaPiece* list = (persp == BLACK) ? pos.eval_list()->piece_list_fb()
	                                         : pos.eval_list()->piece_list_fw();
#if defined(P2_FT_I8)
	const std::int8_t* subs[PIECE_NUMBER_NB]; int ns = 0; std::uint32_t subf[PIECE_NUMBER_NB];
	const std::int8_t* adds[PIECE_NUMBER_NB]; int na = 0; std::uint32_t addf[PIECE_NUMBER_NB];
#else
	const std::int16_t* subs[PIECE_NUMBER_NB]; int ns = 0;
	const std::int16_t* adds[PIECE_NUMBER_NB]; int na = 0;
#endif
	std::int32_t psqt_d[NUM_BUCKETS] = {};
	for (int i = 0; i < PIECE_NUMBER_NB; ++i) {
		int ob = (int)en.bp[i];
		int nb = (int)list[i];
		if (ob == nb) continue;
		if (ob != BONA_PIECE_ZERO) {
			int f = kb * PIECE_INPUTS + pack_bonapiece(ob, mirror);
#if defined(P2_FT_I8)
			subf[ns] = (std::uint32_t)f; subs[ns++] = ft_row8(net, f);
#else
			subs[ns++] = ft_row(net, f);
#endif
			if (net.has_psqt) { const std::int32_t* pw = psqt_row(net, f); for (int b = 0; b < NUM_BUCKETS; ++b) psqt_d[b] -= pw[b]; }
		}
		if (nb != BONA_PIECE_ZERO) {
			int f = kb * PIECE_INPUTS + pack_bonapiece(nb, mirror);
#if defined(P2_FT_I8)
			addf[na] = (std::uint32_t)f; adds[na++] = ft_row8(net, f);
#else
			adds[na++] = ft_row(net, f);
#endif
			if (net.has_psqt) { const std::int32_t* pw = psqt_row(net, f); for (int b = 0; b < NUM_BUCKETS; ++b) psqt_d[b] += pw[b]; }
		}
		en.bp[i] = (std::int32_t)nb;
	}

#if defined(P2_FABLE_PROBE)
	FableProbe::rows_refresh += (std::uint64_t)(ns + na); ++FableProbe::n_refresh;
#endif
#if defined(P2_FT_I32)
	// P2_FT_I32: en.acc (in place) と dacc に en.acc − Σsubs + Σadds を書く。
	if (ns == 0 && na == 0) std::memcpy(st->p2_acc[persp], en.acc, sizeof(std::int32_t) * FT_DIM);
	else ft32_apply(en.acc, en.acc, st->p2_acc[persp], subs, ns, adds, na);
#else
	std::int16_t* dacc = st->p2_acc[persp];
	if (ns == 0 && na == 0) {
		// ��2026-09-09 �C��: ������ **FT** accumulator (FT_DIM ��) �̃R�s�[�B
		//   THREAT_ACC_DIM ���g���Ă������� P2_THREAT_OUT<FT_DIM �̃r���h��
		//   �擪 K �������������ꂸ�A �c�� (FT_DIM-K) ���O�ǖʂ̒l�̂܂܎c���Ă����B
		//   K == FT_DIM �̃r���h�ł͓��� (= �]���� bit-exact)�B
		std::memcpy(dacc, en.acc, sizeof(std::int16_t) * FT_DIM);
	} else {
#if defined(P2_FT_PREFETCH_ALL)
		prefetch_rows_all(reinterpret_cast<const void* const*>(subs), ns);
		prefetch_rows_all(reinterpret_cast<const void* const*>(adds), na);
#endif
#if defined(USE_AVX2)
#if defined(P2_OPN_R2_KERNELS)
		int k = 0;
		if (R2K(K_MREF, P2_OPN_R2_MREF)) {
			// R2-4: 差分更新 (ensure_perspective の A4) と同じ「64 列ごとに 4 レジスタ、 行は内側」。 従来は 16 列ごとに全行を回していて、
			//   行ポインタの読み直しとループ制御が 16 列ごとに乗っていた。 列ごとに見れば subs を順に引き adds を順に足す順は同じ
			//   (i16 wrapping) なので値は変わらない。 実測 −60〜−95 cycle / 評価。
			// SAFETY: en.acc は 64 byte 整列 (従来も _mm256_load_si256 で読んでいる)、 dacc と各行は FT_DIM 要素以上。
			//   不変条件: 端数 (FT_DIM % 64) は下の 16 列ループで処理する。
			for (; k + 64 <= FT_DIM; k += 64) {
				__m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(en.acc + k + 0));
				__m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(en.acc + k + 16));
				__m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(en.acc + k + 32));
				__m256i v3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(en.acc + k + 48));
				for (int s = 0; s < ns; ++s) {
					const std::int16_t* q_ = subs[s] + k;
					v0 = _mm256_sub_epi16(v0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 0)));
					v1 = _mm256_sub_epi16(v1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 16)));
					v2 = _mm256_sub_epi16(v2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 32)));
					v3 = _mm256_sub_epi16(v3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 48)));
				}
				for (int a = 0; a < na; ++a) {
					const std::int16_t* q_ = adds[a] + k;
					v0 = _mm256_add_epi16(v0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 0)));
					v1 = _mm256_add_epi16(v1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 16)));
					v2 = _mm256_add_epi16(v2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 32)));
					v3 = _mm256_add_epi16(v3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 48)));
				}
				_mm256_store_si256(reinterpret_cast<__m256i*>(en.acc + k + 0), v0);
				_mm256_store_si256(reinterpret_cast<__m256i*>(en.acc + k + 16), v1);
				_mm256_store_si256(reinterpret_cast<__m256i*>(en.acc + k + 32), v2);
				_mm256_store_si256(reinterpret_cast<__m256i*>(en.acc + k + 48), v3);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(dacc + k + 0), v0);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(dacc + k + 16), v1);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(dacc + k + 32), v2);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(dacc + k + 48), v3);
			}
		}
		for (; k < FT_DIM; k += 16) {
#else
		for (int k = 0; k < FT_DIM; k += 16) {
#endif
			__m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(en.acc + k));
#if defined(P2_FT_I8)
			for (int s = 0; s < ns; ++s) v = _mm256_sub_epi16(v, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(subs[s] + k))));
			for (int a = 0; a < na; ++a) v = _mm256_add_epi16(v, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(adds[a] + k))));
#else
			for (int s = 0; s < ns; ++s) v = _mm256_sub_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(subs[s] + k)));
			for (int a = 0; a < na; ++a) v = _mm256_add_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(adds[a] + k)));
#endif
			_mm256_store_si256(reinterpret_cast<__m256i*>(en.acc + k), v);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dacc + k), v);
		}
#if defined(P2_FT_I8)
		// 補正 (i8 に収まらなかった値) を en.acc と dacc の両方に足す。 wrapping 加算なので順序不問。
		for (int s = 0; s < ns; ++s) { ft_corr_apply(net, subf[s], en.acc, -1); ft_corr_apply(net, subf[s], dacc, -1); }
		for (int a = 0; a < na; ++a) { ft_corr_apply(net, addf[a], en.acc, +1); ft_corr_apply(net, addf[a], dacc, +1); }
#endif
#else
		for (int k = 0; k < FT_DIM; ++k) {
			int v = en.acc[k];
			for (int s = 0; s < ns; ++s) v -= subs[s][k];
			for (int a = 0; a < na; ++a) v += adds[a][k];
			en.acc[k] = (std::int16_t)v;
			dacc[k]   = (std::int16_t)v;
		}
#if defined(P2_FT_I8)
		for (int s = 0; s < ns; ++s) { ft_corr_apply(net, subf[s], en.acc, -1); ft_corr_apply(net, subf[s], dacc, -1); }
		for (int a = 0; a < na; ++a) { ft_corr_apply(net, addf[a], en.acc, +1); ft_corr_apply(net, addf[a], dacc, +1); }
#endif
#endif
	}
#endif  // P2_FT_I32
	for (int b = 0; b < NUM_BUCKETS; ++b) {
		en.psqt[b] += psqt_d[b];
		st->p2_psqt[persp][b] = en.psqt[b];
	}
}

// acc[0..1536] += w / -= w (i16 wrapping)ç¸²ã»VX2ç¸²E¢cc ç¸ºE¨ w(net é©¥éªâ©é¦ã»ç¸ºE¯é«±æ¨£ã¾éãEÂã»
static inline void acc_add_i16(std::int16_t* __restrict acc, const std::int16_t* __restrict w) {
	const int n = FT_DIM;
#if defined(USE_AVX2)
	for (int k = 0; k < n; k += 16) {
		__m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + k));
		__m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k));
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + k), _mm256_add_epi16(a, b));
	}
#else
	for (int k = 0; k < n; ++k) acc[k] = (std::int16_t)(acc[k] + w[k]);
#endif
}
static inline void acc_sub_i16(std::int16_t* __restrict acc, const std::int16_t* __restrict w) {
	const int n = FT_DIM;
#if defined(USE_AVX2)
	for (int k = 0; k < n; k += 16) {
		__m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + k));
		__m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k));
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + k), _mm256_sub_epi16(a, b));
	}
#else
	for (int k = 0; k < n; ++k) acc[k] = (std::int16_t)(acc[k] - w[k]);
#endif
}

// 1 è¬E¥ã»ç¸ºE®è¾E®è»ã»DirtyPiece)ç¹§ã»acc/psqt ç¸ºE«é©ï½©éï½¨(é¾EªéE¾æ¼éï½»èæEE¦ä¹ãç¸ºE®ç¸ºE¿ç¸²E¬b/mirror ç¸ºE¯è³æ¦ï½¤ã»ç¸²ã»
#if !defined(P2_FT_I8)   // 2026-11-05 (fable): 未使用 (i16 行専用)
static inline void apply_delta(std::int16_t* acc, std::int32_t* psqt,
                               const Eval::DirtyPiece& dp, Color persp, int kb, bool mirror) {
	const Net& net = *g_net;
	for (int i = 0; i < dp.dirty_num; ++i) {
		const Eval::ChangedBonaPiece& cp = dp.changed_piece[i];
		int old_bp = (persp == BLACK) ? (int)cp.old_piece.fb : (int)cp.old_piece.fw;
		int new_bp = (persp == BLACK) ? (int)cp.new_piece.fb : (int)cp.new_piece.fw;
		if (old_bp != BONA_PIECE_ZERO) {
			int feat = kb * PIECE_INPUTS + pack_bonapiece(old_bp, mirror);
			acc_sub_i16(acc, ft_row(net, feat));
			if (net.has_psqt) {
				const std::int32_t* pw = psqt_row(net, feat);
				for (int b = 0; b < NUM_BUCKETS; ++b) psqt[b] -= pw[b];
			}
		}
		if (new_bp != BONA_PIECE_ZERO) {
			int feat = kb * PIECE_INPUTS + pack_bonapiece(new_bp, mirror);
			acc_add_i16(acc, ft_row(net, feat));
			if (net.has_psqt) {
				const std::int32_t* pw = psqt_row(net, feat);
				for (int b = 0; b < NUM_BUCKETS; ++b) psqt[b] += pw[b];
			}
		}
	}
}
#endif

// ç¸ºèãEè¿¥E¶è«·ä¹ãE DirtyPiece ç¸ºE« persp ç¸ºE®é¾EªéEEE§E»èè¼â²è·E«ç¸ºE¾ç¹§å¾ï½ç¸ºã»è·E«ç¹§Âç¸ºEªç¹§ã»kb èçå§=ç¹è¼ÎErefresh)ç¸²ã»
static inline bool friend_king_moved(const StateInfo* st, Color persp) {
	PieceNumber kingNo = (persp == BLACK) ? PIECE_NUMBER_BKING : PIECE_NUMBER_WKING;
	const Eval::DirtyPiece& dp = st->dirtyPiece;
	for (int i = 0; i < dp.dirty_num; ++i)
		if (dp.pieceNo[i] == kingNo) return true;
	return false;
}

// è¿´E¾è»Âé«±E¢ç¸ºE® persp accumulator ç¹§å ¤E¢Eºè³æºâEéªè²E®ç²E¸å»â©ç¸ºE«ç¸ºå¶EEé©ãEE»E¶è¾E®è»ã»å³©è­E½°)ç¸²ã»
// 2026-09-15 (A3 chkstk 除去): chain[300] (2,400B) でフレームが 4,712B になり、 毎 eval 2 回 (両 persp)
//   ___chkstk_ms が走っていた。 chain を 128 に縮めてフレームを ~3.3KB にする。
//   結果は不変 (bit-exact): 上限に達した場合は base=nullptr → refresh_perspective (全再計算) に落ちるが、
//   差分更新と全再計算は int16 wrapping 加算で同一値になる (既存の前提)。 また chain が 128 を超えて
//   かつ add/sub が CAP=96 に収まる (= overflow せず差分で済む) ことは実際には起きない:
//   null move 以外の 1 手は add と sub を各 1 つ以上生み、 null move は連続しないので、 chain 長 n の
//   うち非 null move は ≥ n/2 > 96 → n > 192 では必ず overflow → refresh。 128 < n ≤ 192 で差分が
//   成立するには chain 内に 32 個以上の null move が要り、 探索経路上あり得ない。
//   noinline: evaluate_internal に 2 回 inline されて合算 4KB 超になるのを防ぐ (PGO 対策)。
static constexpr int P2_ACC_CHAIN_MAX = 128;
#if defined(P2_OPN)
// 2026-11-15 (ANALYSIS-20261115-OPN-NPS-FABLE P2): OPN 有効時は本体の追いつきの上限を P2_OPN_CHAIN_MAX (8) 手にする。
//   置き換えの帯では OPN だけの node で本体の acc を作らないので、 本体が要る node では root 近くまで遡って
//   数十手ぶんの行を足し引きしていた (1 回 約 20,000 cycle)。 上限を超えたら Finny (refresh_perspective) に落とす。
//   bit-exact の根拠: 上の 128 の注記と同じ (差分と全計算は i16 wrapping の和で同じ値、 PSQT も refresh が作る)。
//   不変条件: 値は 1..P2_ACC_CHAIN_MAX (chain[] の大きさ)。 opn::update_derived() だけが書く。 OPN 無効時は 128。
static int g_acc_chain_limit = P2_ACC_CHAIN_MAX;
#define P2_ACC_CHAIN_LIMIT g_acc_chain_limit
#if defined(P2_OPN_PROBE)
// 本体の追いつきの計数 (probe ビルドだけ)。 p2nnue_opn.h の probe::report が出す。
namespace OpnMainProbe { static std::uint64_t ens = 0, chain_n = 0, chain_len = 0, refresh = 0, refresh_far = 0; }
#endif
#else
#define P2_ACC_CHAIN_LIMIT P2_ACC_CHAIN_MAX
#endif
#if defined(P2_FABLE_PROBE)
// 2026-11-11 (PRIMEW / Fable 5.2 探索レビュー 案 A): priming の無駄率 w = 1 − used / upd。
//   upd = priming で HalfKA acc を実際に計算した回数 (視点ごと) / used = その acc が後で子の base に採用された回数。
//   🚨probe ビルドだけ。 値は変えない。
namespace PrimeProbe {
static thread_local bool in_prime = false;
static std::uint64_t upd = 0, used = 0, upd_refresh = 0, ev_upd = 0;
static void report() {
	std::fprintf(stderr, "PRIMEPROBE upd=%llu used=%llu upd_refresh=%llu ev_upd=%llu\n",
		(unsigned long long)upd, (unsigned long long)used, (unsigned long long)upd_refresh, (unsigned long long)ev_upd);
}
static struct Init { Init() { std::atexit(report); } } init_;
}
#endif
#if defined(P2_ACC_PIECEWISE)
#if defined(P2_FT_I8)
#error "P2_ACC_PIECEWISE は i16 行 (P2_FT_I8 なし) 専用"
#endif
// P2_ACC_PIECEWISE (2026-11-17): 多段の追いつき (計算済みの祖先から何手分も差分で作る) を駒単位にまとめる。
//   本家 nnue_feature_transformer.h の update_from_computed_ancestor と同じ考え方: 駒番号ごとに「祖先での BonaPiece (最初の old)」と
//   「今の BonaPiece (最後の new)」だけを残す。 途中で 2 回以上動いた駒も、 引く 1 本・足す 1 本で済む。 元の位置に戻った駒は 0 本。
//   値: 同じ駒番号の連続する変化は old(k+1) == new(k) (EvalList の駒番号ごとの BonaPiece を書き換えた記録) なので、
//     Σ (new − old) は new(最後) − old(最初) に畳める (i16 wrapping の加減は可換で、 引くことは足すことの逆元)。
//     行の多重集合としての和が同じなので acc / PSQT とも従来と同じ値 (bit-exact)。 chain の中では自玉が動かない
//     (行の区分と鏡像が同じ) ことは呼び出し側が保証する。 検算は -DP2_ACC_PIECEWISE_VERIFY (全駒からの作り直しと要素比較)。
//   chain[i_hi] が最も古い手、 chain[i_lo] が最も新しい手。 pob / pnb は PIECE_NUMBER_NB 要素以上。
//   戻り値 = 変化の残った駒の数 (pob[t] != pnb[t])。 駒番号が範囲外なら −1 (呼び出し側は Finny / 従来の経路に落とす)。
static inline int fold_pieces(StateInfo* const* chain, int i_hi, int i_lo, Color persp, int* pob, int* pnb) {
	std::int8_t slot[PIECE_NUMBER_NB];
	std::memset(slot, -1, sizeof slot);
	int np = 0;
	for (int i = i_hi; i >= i_lo; --i) {
		const Eval::DirtyPiece& dp = chain[i]->dirtyPiece;
		for (int j = 0; j < dp.dirty_num; ++j) {
			const int pn = (int) dp.pieceNo[j];
			if (pn < 0 || pn >= (int) PIECE_NUMBER_NB) return -1;
			const Eval::ChangedBonaPiece& cp = dp.changed_piece[j];
			const int ob = (persp == BLACK) ? (int) cp.old_piece.fb : (int) cp.old_piece.fw;
			const int nb = (persp == BLACK) ? (int) cp.new_piece.fb : (int) cp.new_piece.fw;
			if (slot[pn] < 0) { slot[pn] = (std::int8_t) np; pob[np] = ob; pnb[np] = nb; ++np; }
			else pnb[slot[pn]] = nb;
		}
	}
	int m = 0;
	for (int t = 0; t < np; ++t)
		if (pob[t] != pnb[t]) { pob[m] = pob[t]; pnb[m] = pnb[t]; ++m; }
	return m;
}
#if defined(P2_ACC_PIECEWISE_VERIFY)
// 検算 (診断ビルド専用): 差分で作った本体の acc / PSQT を、 bias + 今の局面の全駒の行 (i16 wrapping) と要素比較する。
//   不一致は 1 件目で stderr に出して abort。 終了時に件数と行数 (畳む前 / 畳んだ後) を出す。
namespace PieceVerify {
static std::uint64_t n_chk = 0, n_multi = 0, rows_before = 0, rows_after = 0, n_bad = 0, n_opn = 0, n_opn_multi = 0;
static void report() {
	std::fprintf(stderr, "PIECEWISE_VERIFY checked=%llu multi=%llu rows_before=%llu rows_after=%llu opn_checked=%llu opn_multi=%llu mismatch=%llu\n",
		(unsigned long long) n_chk, (unsigned long long) n_multi, (unsigned long long) rows_before, (unsigned long long) rows_after,
		(unsigned long long) n_opn, (unsigned long long) n_opn_multi, (unsigned long long) n_bad);
}
static struct Init { Init() { std::atexit(report); } } init_;
static void fail(const char* what, int col, int full, int got, const Position& pos) {
	++n_bad;
	std::fprintf(stderr, "PIECEWISE_VERIFY MISMATCH %s col=%d full=%d got=%d sfen=%s\n", what, col, full, got, pos.sfen().c_str());
	report();
	std::abort();
}
} // namespace PieceVerify
static void piecewise_check_main(const Position& pos, Color persp) {
	const Net& net = *g_net;
	const StateInfo* st = pos.state();
	const Square ksq = pos.square<KING>(persp);
	const int kb = king_bucket(ksq, persp);
	const bool mirror = is_hm_mirror(ksq, persp);
	const BonaPiece* list = (persp == BLACK) ? pos.eval_list()->piece_list_fb() : pos.eval_list()->piece_list_fw();
	alignas(64) FtAcc full[FT_DIM];
	std::int32_t ps[NUM_BUCKETS];
	for (int k = 0; k < FT_DIM; ++k) full[k] = net.ft_bias[k];
	for (int b = 0; b < NUM_BUCKETS; ++b) ps[b] = net.psqt_bias[b];
	for (int i = 0; i < PIECE_NUMBER_NB; ++i) {
		if ((int) list[i] == BONA_PIECE_ZERO) continue;
		const int f = kb * PIECE_INPUTS + pack_bonapiece((int) list[i], mirror);
		const std::int16_t* w = ft_row(net, f);
		for (int k = 0; k < FT_DIM; ++k) full[k] = (FtAcc) (full[k] + w[k]);
		if (net.has_psqt) { const std::int32_t* pw = psqt_row(net, f); for (int b = 0; b < NUM_BUCKETS; ++b) ps[b] += pw[b]; }
	}
	++PieceVerify::n_chk;
	for (int k = 0; k < FT_DIM; ++k)
		if (full[k] != st->p2_acc[persp][k]) PieceVerify::fail("main_acc", k, full[k], st->p2_acc[persp][k], pos);
	for (int b = 0; b < NUM_BUCKETS; ++b)
		if (ps[b] != st->p2_psqt[persp][b]) PieceVerify::fail("main_psqt", b, ps[b], st->p2_psqt[persp][b], pos);
}
#endif
#endif
static __attribute__((noinline)) void ensure_perspective(const Position& pos, Color persp) {
	StateInfo* st = pos.state();
	if (st->p2_computed[persp]) return;
#if defined(P2_OPN) && defined(P2_OPN_PROBE)
	++OpnMainProbe::ens;
#endif

	// éªè²E®ç²E¸å»â©é¾éã»ç¸ºE¾ç¸ºE§é©ï½¡ç¹§ã»é¾EªéEEE§E»èè¼ï½éE¨ç¸ºå¾âEç¸ºã»E¯ã»å³Eç¸²ã»
	StateInfo* chain[P2_ACC_CHAIN_MAX];
	int n = 0;
	StateInfo* cur = st;
	StateInfo* base = nullptr;
	while (true) {
		if (cur->p2_computed[persp]) {
			base = cur;
#if defined(P2_FABLE_PROBE)
			if (cur->p2_prime_probe[persp] == 1) { cur->p2_prime_probe[persp] = 2; ++PrimeProbe::used; }
#endif
			break;
		}
		if (cur->previous == nullptr || friend_king_moved(cur, persp)) break;
		if (n >= P2_ACC_CHAIN_LIMIT) {
#if defined(P2_OPN) && defined(P2_OPN_PROBE)
			++OpnMainProbe::refresh_far;
#endif
			break;
		}
		chain[n++] = cur;
		cur = cur->previous;
	}

	Square ksq = pos.square<KING>(persp);
	int kb = king_bucket(ksq, persp);
	bool mirror = is_hm_mirror(ksq, persp);

	const Net& net = *g_net;
	// chain èï½¨è´èãE add/sub è¿ï½¹è ï½´é©¥ä¸ï½è¿æ±å¯Ekb/mirror ç¸ºE¯ chain èã»ãè³æ¦ï½¤ã»è¿´E¾è¨E¨èï½¤)ç¸²ã»
	// 1ç¹ä»£ãç¸ºE§ base + Eï½£add - Eï½£sub ç¹§å¸âEéï½¨ç¸ºå±â»ç¹ï½¡ç¹ï½¢ç¹ï½ªè Âè ï½©ç¹§è²çè²å¸âEç¹§ä¹Âã»
	constexpr int CAP = 96;
#if defined(P2_FT_I8)
	const std::int8_t* adds[CAP]; int na = 0; std::uint32_t addf[CAP];
	const std::int8_t* subs[CAP]; int ns = 0; std::uint32_t subf[CAP];
#else
	const std::int16_t* adds[CAP]; int na = 0;
	const std::int16_t* subs[CAP]; int ns = 0;
#endif
	std::int32_t psqt_d[NUM_BUCKETS] = {};
	bool overflow = false;
	if (base != nullptr && n < P2_ACC_CHAIN_LIMIT) {
#if defined(P2_ACC_PIECEWISE)
		// P2_ACC_PIECEWISE: 2 手以上を跨ぐときは駒単位に畳んでから行を集める (fold_pieces の注記)。 1 手なら従来と同じ。
		if (n >= 2) {
			int pob[PIECE_NUMBER_NB], pnb[PIECE_NUMBER_NB];
			const int np = fold_pieces(chain, n - 1, 0, persp, pob, pnb);
			if (np < 0) overflow = true;
			for (int t = 0; t < np && !overflow; ++t) {
				if (pob[t] != BONA_PIECE_ZERO) {
					const int f = kb * PIECE_INPUTS + pack_bonapiece(pob[t], mirror);
					if (ns >= CAP) { overflow = true; break; }
					subs[ns++] = ft_row(net, f);
					if (net.has_psqt) { const std::int32_t* pw = psqt_row(net, f); for (int b = 0; b < NUM_BUCKETS; ++b) psqt_d[b] -= pw[b]; }
				}
				if (pnb[t] != BONA_PIECE_ZERO) {
					const int f = kb * PIECE_INPUTS + pack_bonapiece(pnb[t], mirror);
					if (na >= CAP) { overflow = true; break; }
					adds[na++] = ft_row(net, f);
					if (net.has_psqt) { const std::int32_t* pw = psqt_row(net, f); for (int b = 0; b < NUM_BUCKETS; ++b) psqt_d[b] += pw[b]; }
				}
			}
#if defined(P2_ACC_PIECEWISE_VERIFY)
			++PieceVerify::n_multi; PieceVerify::rows_after += (std::uint64_t) (ns + na);
			for (int i = n - 1; i >= 0; --i) {
				const Eval::DirtyPiece& dp = chain[i]->dirtyPiece;
				for (int j = 0; j < dp.dirty_num; ++j) {
					const Eval::ChangedBonaPiece& cp = dp.changed_piece[j];
					PieceVerify::rows_before += (((persp == BLACK) ? (int) cp.old_piece.fb : (int) cp.old_piece.fw) != BONA_PIECE_ZERO);
					PieceVerify::rows_before += (((persp == BLACK) ? (int) cp.new_piece.fb : (int) cp.new_piece.fw) != BONA_PIECE_ZERO);
				}
			}
#endif
		} else
#endif
		for (int i = n - 1; i >= 0 && !overflow; --i) {
			const Eval::DirtyPiece& dp = chain[i]->dirtyPiece;
			for (int j = 0; j < dp.dirty_num; ++j) {
				const Eval::ChangedBonaPiece& cp = dp.changed_piece[j];
				int ob = (persp == BLACK) ? (int)cp.old_piece.fb : (int)cp.old_piece.fw;
				int nb = (persp == BLACK) ? (int)cp.new_piece.fb : (int)cp.new_piece.fw;
				if (ob != BONA_PIECE_ZERO) {
					int f = kb * PIECE_INPUTS + pack_bonapiece(ob, mirror);
					if (ns >= CAP) { overflow = true; break; }
#if defined(P2_FT_I8)
					subf[ns] = (std::uint32_t)f; subs[ns++] = ft_row8(net, f);
#else
					subs[ns++] = ft_row(net, f);
#endif
					if (net.has_psqt) { const std::int32_t* pw = psqt_row(net, f); for (int b = 0; b < NUM_BUCKETS; ++b) psqt_d[b] -= pw[b]; }
				}
				if (nb != BONA_PIECE_ZERO) {
					int f = kb * PIECE_INPUTS + pack_bonapiece(nb, mirror);
					if (na >= CAP) { overflow = true; break; }
#if defined(P2_FT_I8)
					addf[na] = (std::uint32_t)f; adds[na++] = ft_row8(net, f);
#else
					adds[na++] = ft_row(net, f);
#endif
					if (net.has_psqt) { const std::int32_t* pw = psqt_row(net, f); for (int b = 0; b < NUM_BUCKETS; ++b) psqt_d[b] += pw[b]; }
				}
			}
		}
	}

	if (base != nullptr && n < P2_ACC_CHAIN_LIMIT && !overflow) {
#if defined(P2_OPN) && defined(P2_OPN_PROBE)
		++OpnMainProbe::chain_n; OpnMainProbe::chain_len += (std::uint64_t) n;
#endif
#if defined(P2_FT_I32)
#if defined(P2_FT_PREFETCH_ALL)
		prefetch_rows_all(reinterpret_cast<const void* const*>(subs), ns);
		prefetch_rows_all(reinterpret_cast<const void* const*>(adds), na);
#endif
		ft32_apply(base->p2_acc[persp], st->p2_acc[persp], nullptr, subs, ns, adds, na);
#else
		const std::int16_t* bacc = base->p2_acc[persp];
		std::int16_t* dacc = st->p2_acc[persp];
		const int nft = FT_DIM;
#if defined(P2_FT_PREFETCH_ALL)
		prefetch_rows_all(reinterpret_cast<const void* const*>(subs), ns);
		prefetch_rows_all(reinterpret_cast<const void* const*>(adds), na);
#endif
#if defined(USE_AVX2)
		// ★NPS(A4): 4 レジスタ (64 要素) 単位に展開してループ制御を 1/4 に。
		//   nft=1024 なので外側 64 反復ぶんの制御が、 短い内側ループのたびに乗っていた。
		//   bit-exact: 要素ごとに見れば subs を順に引き adds を順に足す順序は変わらない。
		int k = 0;
#if defined(P2_OPN_R2_KERNELS)
		// R2-2: 引く行と足す行が同数で 1..4 本なら定数版 (r2_rows_fx) で作る (値は同じ)。 作ったら k = nft にして下の汎用ループを飛ばす。
		if (R2K(K_MFX, P2_OPN_R2_MFX) && FT_DIM % 64 == 0 && ns == na
		    && r2_rows_n<false, false>(ns, bacc, dacc, nullptr, FT_DIM, subs, adds, nullptr, 0)) k = nft;
#endif
		for (; k + 64 <= nft; k += 64) {
			__m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bacc + k + 0));
			__m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bacc + k + 16));
			__m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bacc + k + 32));
			__m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bacc + k + 48));
#if defined(P2_FT_I8)
			// i8 行: 16 バイトを vpmovsxbw で i16 に符号拡張して足す (値は i16 行と同一 = bit-exact)。
			for (int s = 0; s < ns; ++s) {
				const std::int8_t* q_ = subs[s] + k;
				v0 = _mm256_sub_epi16(v0, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q_ + 0))));
				v1 = _mm256_sub_epi16(v1, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q_ + 16))));
				v2 = _mm256_sub_epi16(v2, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q_ + 32))));
				v3 = _mm256_sub_epi16(v3, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q_ + 48))));
			}
			for (int a = 0; a < na; ++a) {
				const std::int8_t* q_ = adds[a] + k;
				v0 = _mm256_add_epi16(v0, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q_ + 0))));
				v1 = _mm256_add_epi16(v1, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q_ + 16))));
				v2 = _mm256_add_epi16(v2, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q_ + 32))));
				v3 = _mm256_add_epi16(v3, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q_ + 48))));
			}
#else
			for (int s = 0; s < ns; ++s) {
				const std::int16_t* q_ = subs[s] + k;
				v0 = _mm256_sub_epi16(v0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 0)));
				v1 = _mm256_sub_epi16(v1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 16)));
				v2 = _mm256_sub_epi16(v2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 32)));
				v3 = _mm256_sub_epi16(v3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 48)));
			}
			for (int a = 0; a < na; ++a) {
				const std::int16_t* q_ = adds[a] + k;
				v0 = _mm256_add_epi16(v0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 0)));
				v1 = _mm256_add_epi16(v1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 16)));
				v2 = _mm256_add_epi16(v2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 32)));
				v3 = _mm256_add_epi16(v3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_ + 48)));
			}
#endif
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dacc + k + 0), v0);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dacc + k + 16), v1);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dacc + k + 32), v2);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dacc + k + 48), v3);
		}
		for (; k < nft; k += 16) {          // 余り
			__m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bacc + k));
#if defined(P2_FT_I8)
			for (int s = 0; s < ns; ++s) v = _mm256_sub_epi16(v, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(subs[s] + k))));
			for (int a = 0; a < na; ++a) v = _mm256_add_epi16(v, _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(adds[a] + k))));
#else
			for (int s = 0; s < ns; ++s) v = _mm256_sub_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(subs[s] + k)));
			for (int a = 0; a < na; ++a) v = _mm256_add_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(adds[a] + k)));
#endif
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dacc + k), v);
		}
#else
		for (int k = 0; k < nft; ++k) {
			int v = bacc[k];
			for (int s = 0; s < ns; ++s) v -= subs[s][k];
			for (int a = 0; a < na; ++a) v += adds[a][k];
			dacc[k] = (std::int16_t)v;
		}
#endif
#endif  // P2_FT_I32
#if defined(P2_FT_I8)
		// 補正 (i8 に収まらなかった値)。 dacc は base + Σ(i8 行) まで出来ているので、 ここで足せば元の i16 行の和と同じ。
		for (int s = 0; s < ns; ++s) ft_corr_apply(net, subf[s], dacc, -1);
		for (int a = 0; a < na; ++a) ft_corr_apply(net, addf[a], dacc, +1);
#endif
#if defined(P2_FABLE_PROBE)
		FableProbe::rows_chain += (std::uint64_t)(ns + na); ++FableProbe::n_chain;
#endif
		for (int b = 0; b < NUM_BUCKETS; ++b) st->p2_psqt[persp][b] = base->p2_psqt[persp][b] + psqt_d[b];
#if defined(P2_ACC_PIECEWISE_VERIFY)
		piecewise_check_main(pos, persp);
#endif
	} else {
#if defined(P2_OPN) && defined(P2_OPN_PROBE)
		++OpnMainProbe::refresh;
#endif
		refresh_perspective(pos, persp);
#if defined(P2_ACC_PIECEWISE_VERIFY) && defined(P2_FT_I32)
		piecewise_check_main(pos, persp);   // P2_FT_I32: Finny の refresh の結果も全駒からの作り直しと比べる
#endif
#if defined(P2_FABLE_PROBE)
		if (PrimeProbe::in_prime) ++PrimeProbe::upd_refresh;
#endif
	}
	st->p2_computed[persp] = true;
#if defined(P2_FABLE_PROBE)
	st->p2_prime_probe[persp] = PrimeProbe::in_prime ? 1 : 0;
	if (PrimeProbe::in_prime) ++PrimeProbe::upd; else ++PrimeProbe::ev_upd;
#endif
}

static inline void ensure_accumulator(const Position& pos) {
	ensure_perspective(pos, BLACK);
	ensure_perspective(pos, WHITE);
}

// ---------------------------------------------------------------------------
// progress8kpabs : sum ç«ãEbucket
// ---------------------------------------------------------------------------
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// progress sum èï½¨é¬§åµãç¹§E­ç¹ï½£ç¹ï½³(éE¾åèã»ç¸²ã»
static float progress_full(const Position& pos) {
	int sq_bk = (int)pos.square<KING>(BLACK);
	int sq_wk = (int)Inv(pos.square<KING>(WHITE));
	const float* wb = &g_progress[(std::size_t)sq_bk * FE_OLD_END];
	const float* ww = &g_progress[(std::size_t)sq_wk * FE_OLD_END];
	float sum = 0.0f;
	const BonaPiece* fb = pos.eval_list()->piece_list_fb();
	const BonaPiece* fw = pos.eval_list()->piece_list_fw();
	for (int i = 0; i < PIECE_NUMBER_NB; ++i) {
		int b = (int)fb[i], w = (int)fw[i];
		if (b != BONA_PIECE_ZERO && b < FE_OLD_END) sum += wb[b];
		if (w != BONA_PIECE_ZERO && w < FE_OLD_END) sum += ww[w];
	}
	return sum;
}

// dirtyPiece ç¸ºE« ç¸ºã»âE¹§å¾Â°ç¸ºE®éEEE§E»èè¼â²è·E«ç¸ºE¾ç¹§å¾ï½ç¸ºã»è·E«ç¹§Âç¸ºEªç¹§ã»progress weights é¦å¾â²èçå§=ç¹è¼Îèå´ï½¨è²E®ã»ç¸²ã»
static inline bool any_king_moved(const StateInfo* st) {
	const Eval::DirtyPiece& dp = st->dirtyPiece;
	for (int i = 0; i < dp.dirty_num; ++i)
		if (dp.pieceNo[i] == PIECE_NUMBER_BKING || dp.pieceNo[i] == PIECE_NUMBER_WKING) return true;
	return false;
}

// progress sum ç¹§å¸â¦èï½¶è¾E®è»ã»å³©è­E½°ç¸ºE§éï½ºè³æºâEéªè²E®å±Âã»
static void ensure_progress(const Position& pos) {
	StateInfo* st = pos.state();
	if (st->p2_prog_computed) return;
	// 2026-10-07 P-L1c (§1.2): 差分の積み上げを使わず常に全計算する。
	//   p2_prog_sum は「最寄りの computed な祖先から float で差分を積む」ので、 積む経路が
	//   変わると丸めが変わり bucket 境界の局面で bucket が 1 つずれる。 これを殺して
	//   P-L1b を繰り返し、 一致すれば「float の経路依存」で原因が確定する。
	//   🚨これ自体は本番と bit-exact ではない (経路非依存にする = 値が変わる) ので計測専用。
	if (probe_prog_full()) {
		st->p2_prog_sum = progress_full(pos);
		st->p2_prog_computed = true;
		return;
	}
	StateInfo* chain[300];
	int n = 0;
	StateInfo* cur = st;
	StateInfo* base = nullptr;
	while (true) {
		if (cur->p2_prog_computed) { base = cur; break; }
		if (cur->previous == nullptr || any_king_moved(cur) || n >= 300) break;
		chain[n++] = cur;
		cur = cur->previous;
	}
	if (base != nullptr && n < 300) {
		int sq_bk = (int)pos.square<KING>(BLACK);
		int sq_wk = (int)Inv(pos.square<KING>(WHITE));
		const float* wb = &g_progress[(std::size_t)sq_bk * FE_OLD_END];
		const float* ww = &g_progress[(std::size_t)sq_wk * FE_OLD_END];
		float sum = base->p2_prog_sum;
		for (int i = n - 1; i >= 0; --i) {
			const Eval::DirtyPiece& dp = chain[i]->dirtyPiece;
			for (int j = 0; j < dp.dirty_num; ++j) {
				const Eval::ChangedBonaPiece& cp = dp.changed_piece[j];
				int ob = (int)cp.old_piece.fb, ow = (int)cp.old_piece.fw;
				int nb = (int)cp.new_piece.fb, nw = (int)cp.new_piece.fw;
				if (ob != BONA_PIECE_ZERO && ob < FE_OLD_END) sum -= wb[ob];
				if (ow != BONA_PIECE_ZERO && ow < FE_OLD_END) sum -= ww[ow];
				if (nb != BONA_PIECE_ZERO && nb < FE_OLD_END) sum += wb[nb];
				if (nw != BONA_PIECE_ZERO && nw < FE_OLD_END) sum += ww[nw];
			}
		}
		st->p2_prog_sum = sum;
	} else {
		st->p2_prog_sum = progress_full(pos);
	}
	st->p2_prog_computed = true;
}

static int progress_bucket(const Position& pos) {
	if (!g_progress_loaded) return 0;
	ensure_progress(pos);
	float sum = pos.state()->p2_prog_sum;
	static const float TH[7] = {
		-1.9459101f, -1.0986123f, -0.5108256f, 0.0f, 0.5108256f, 1.0986123f, 1.9459101f
	};
	int bucket = 0;
	for (int i = 0; i < 7; ++i) if (sum >= TH[i]) bucket++;
	return bucket;
}

// è³E¡é ã»ç²éªå»ã»é©¥éªâ©èå¥â³è¬ãEE§å¸ã¼(P2_HAND_PIECE_WEIGHT ç¸ºE§é©¥éªâ©èå¥E )ç¸²ã»
static inline int weighted_hand_total(const Position& pos) {
	int total = 0;
	for (Color c = BLACK; c <= WHITE; ++c) {
		Hand h = pos.hand_of(c);
		for (PieceType pt = PAWN; pt < PIECE_HAND_NB; ++pt)
			total += hand_count(h, pt) * P2_HAND_PIECE_WEIGHT[pt];
	}
	return total;
}

// è¬ãEE§å¸ã¼2è¹Eºè»ã»ÂãÂE²é¦æï½ºE¦è»E¥é«¢E¾èï½¤èï½¥è³ã»1(èãEç¸²âµæ´è²E=0(èEEç¸²ãEEèE½¦éè²ãEç¸ºE¨è±ç¢ï½¼ã»E¼ç®E®æï½­ã»>=)ç¹§åE½¸Âé¾E´ç¸ºè¼âç¹§ä¹Âã»
#ifdef P2_PRUNE_STATS
// 🚨🚨2026-10-21 **計測ビルド限定**にした。
//   候補 A (bucket 跳ぎで刃りを緩める) が閉じたので本番では誰も呼ばない。
//   呼び出し元 (yaneuraou-search.cpp の P2PruneStat) も同じ #ifdef の中にある。
/// 🎯**計測用の公開ラッパ** (2026-10-20)。 search から bucket を読むために足した。
///   ユーザー指示「候補 A の関門やって」= 親子で bucket が違うノードの誤刈り率を測る。
///   🚨**StateInfo に計算済みのときだけ読む**。 未計算なら **-1 (不明)** を返す。
///     `ensure_progress()` を新しい node で呼ぶと差分の累積経路が変わって
///     bit-exact が崩れる (★threat_delta_off_for_groups の副産物で一度踏んだ)。
///   🚨王手ノードは evaluate を呼ばないので未計算になる -> 呼び出し側で非跨ぎ扱いにする。
int progress_bucket_if_computed(const Position& pos) {
	const StateInfo* st = pos.state();
	if (st == nullptr || !st->p2_prog_computed)
		return -1;
	// progress_bucket() と同じ閾値で切る (閾値の定義はそこに 1 箇所だけある)
	return progress_bucket(pos);
}

/// 🎯**計測用: progress が最寄りの閾値から δ 以内か** (2026-10-20)。
///   「bucket を**越えた**」ことが効いているのか、 単に「境界**付近**の局面」なのかを
///   切り分ける最後の対照。 δ は環境変数 `P2_PROG_NEAR_DELTA` (既定 0.15)。
///   🚨`ensure_progress()` は呼ばず **StateInfo の値だけ**を読む (bit-exact を崩さない)。
///   閾値は `progress_bucket()` と同一の 7 本。
int progress_near_threshold_if_computed(const Position& pos) {
	const StateInfo* st = pos.state();
	if (st == nullptr || !st->p2_prog_computed)
		return -1;
	static const float TH[7] = {
		-1.9459101f, -1.0986123f, -0.5108256f, 0.0f, 0.5108256f, 1.0986123f, 1.9459101f
	};
	static const float D = [] {
		const char* e = std::getenv("P2_PROG_NEAR_DELTA");
		return e ? float(std::atof(e)) : 0.15f;
	}();
	const float s = st->p2_prog_sum;
	for (int i = 0; i < 7; ++i)
		if (s - TH[i] <= D && TH[i] - s <= D)
			return 1;
	return 0;
}

#endif  // P2_PRUNE_STATS

static inline int hand_bucket2(const Position& pos, int progress_bucket4) {
	return weighted_hand_total(pos) >= P2_HAND_VALUE_THRESHOLD[progress_bucket4] ? 1 : 0;
}

// LayerStack index ç¹§å®ï½¿æâEèï½±é¨¾å¤âEè¬æ¨£æªè¬¨E°(ç¹§E³ç¹ï½³ç¹ä»£ãE¹ï½«è­ãâç¹ï½¢ç¹ï½¼ç¹èEE¢Eºè³ã»è³æº¯E¡æ¢§å¾è»ã»E²èâEç¸ºã»ç¸²ã»
// P2_LS_REGIME_MEASURE: NPS tax measurement only (2026-09-26).
//   Computes the B7 regime predicate: ctl_c = sum over empty squares in c's enemy camp of
//   sign(effect_c - effect_opp), then regime = (ctl_black >= 3 || ctl_white >= 3).
//   The result is stored to a volatile sink so the optimiser cannot remove it, and the
//   LayerStack index returned below is UNCHANGED -> the search tree stays bit-identical,
//   so an A/B on the same nodes measures the predicate's cost alone.
//   Design: runs/prereg/DESIGN-20260926-REGIME-IMPL.md section 2.
#if defined(P2_LS_REGIME_MEASURE)
volatile int g_p2_regime_sink = 0;
static inline int p2_regime7(const Position& pos) {
	const uint8_t* eb = pos.board_effect[BLACK].e;
	const uint8_t* ew = pos.board_effect[WHITE].e;
	int ctl_b = 0, ctl_w = 0;
	for (int f = 0; f < 9; ++f) {
		const int base = f * 9;
		for (int r = 0; r < 3; ++r) {		// black's enemy camp = rank 1-3
			const int sq = base + r;
			if (pos.piece_on((Square)sq) != NO_PIECE) continue;
			const int a = eb[sq], dd = ew[sq];
			ctl_b += (a > dd) - (a < dd);
		}
		for (int r = 6; r < 9; ++r) {		// white's enemy camp = rank 7-9
			const int sq = base + r;
			if (pos.piece_on((Square)sq) != NO_PIECE) continue;
			const int a = ew[sq], dd = eb[sq];
			ctl_w += (a > dd) - (a < dd);
		}
	}
	return (ctl_b >= 3 || ctl_w >= 3) ? 1 : 0;
}
#endif

static inline int get_layer_stack_index(const Position& pos) {
#if defined(P2_LS_REGIME_MEASURE)
	g_p2_regime_sink = p2_regime7(pos);
#endif
	if (P2_LS_PROGREGIME) {
		// 案 B: 進行度8 x regime2。 序盤 (pb8 < REGIME_MIN_PB) は regime を立てない。
		//   🚨&& の短絡評価で、 序盤では P2Volm::value を**呼ばない** (NPS 税を払わない)。
		const int pb8 = progress_bucket(pos);                 // 0..7
#if defined(P2_REGIME_DIAG)
		// 診断用: pb8 が 0..7 の外に出ていないかを一度だけ報告する。
		//   crash の原因が「pb8 が想定外 → bucket が 0..15 の外 → OOB」かを確かめる。
		if (pb8 < 0 || pb8 > 7) {
			static bool once = false;
			if (!once) { once = true;
				sync_cout << "info string P2REGIME_BADPB pb8=" << pb8
				          << " prog=" << pos.state()->p2_prog_sum << sync_endl; }
		}
#endif
		const int r   = (pb8 >= REGIME_MIN_PB
		                 && YaneuraOu::P2Volm::value(pos) >= REGIME_V1_TH) ? 1 : 0;
		return pb8 * 2 + r;                                   // 0..15
	}
	if (P2_LS_P8H2) {
		int pb8 = progress_bucket(pos);                       // 0..7
		int pb4 = pb8 / 2;                                    // 0..3(é«¢E¾èï½¤ indexç¸²E£4H2 ç¸ºE¨èï½±é¨¾ã»
		return pb8 * 2 + hand_bucket2(pos, pb4);              // 0..15
	}
	if (P2_LS_P4H2) {
		int progress_bucket4 = progress_bucket(pos) / 2;              // 0..7 ç«ãE0..3
		return progress_bucket4 * 2 + hand_bucket2(pos, progress_bucket4); // 0..7
	}
	return progress_bucket(pos); // Legacy: progress8 ç¹§åµâç¸ºE®ç¸ºE¾ç¸ºE¾
}

// dual_psqt(P2DUAL2) / dual_cross(P2DUALZ) / pairshape(P2PAIR) éï½¨ç¸ºE® bucket indexç¸²ã»
// ç¸ºèï½ç¹§å³¨ã»ç¹§E»ç¹§E¯ç¹§E·ç¹ï½§ç¹ï½³ç¸ºE¯ NUM_DUAL_PSQT_BUCKETS = PAIRSHAPE_NUM_BUCKETS = 8 èï½ºè³ã»NUM_BUCKETS é«±æ¨ï½¾æï½­ã»
// ç¸ºE§ç¸²âï½­E¦éè²ãEç¸ºE¯ compute_dual_bucket() = ShogiProgressKPAbs::bucket() ç¸ºE§ 0..7 ç¹§è²ç¡ç¹§éE½½èâ»ç¹§ä¹Âã»
// ShogiProgressKPAbs ç¸ºE¯ P8H2 èãEShogiProgress8Hand2KPAbs)ç¸ºE¨ç¸ºE¯è»E¥èä¹ãç¸²E£8H2 èE½¦éåEãç¸ºE¯
// layerstack_p4h2 ç¹§E°ç¹ï½­ç¹ï½¼ç¹èEÎç¸ºã»false ç¸ºE®ç¸ºæºï½Eraw progress8(0..7)ç¹§å®ï½¿æâEç¸²ã»
// ç¬ãEet_layer_stack_index ç¸ºE¯ P8H2 ç¸ºE§ 0..15 ç¹§å®ï½¿æâEç¸ºæºï½ç¸²ââç¹§å¾ï½E8bucket é©æ¦ãEç¸ºE«è´E¿ç¸ºã»âE//   bucket>=8(é¨¾E²é¦æï½ºE¦>=4=è³E­é¨ãä¹¢)ç¸ºE§ OOB ç«ãEé«¢æºE¬E° segfault ç¸ºE«ç¸ºEªç¹§ä¹ÂãE¼E¸ºèãEè¶E¸ç¸ºE« 0..7 ç¹§å®ï½¿æâEç¸²ã»
static inline int get_dual_pair_bucket(const Position& pos) {
	if (P2_LS_P4H2) {
		int pb4 = progress_bucket(pos) / 2;               // 0..3
		return pb4 * 2 + hand_bucket2(pos, pb4);          // 0..7(P4H2: compute_dual_bucket ç¸ºE¨è³Âé¾E´)
	}
	// Legacy(progress8)ç¸ºç¿«Eç¸ºE³ P8H2: ShogiProgressKPAbs ç¸ºE¯ p4h2=false ç¸ºE§ raw progress8 ç¹§å®ï½¿æâEç¸²ã»
	return progress_bucket(pos);                          // 0..7
}

// USI "p2stack [sfen]" éï½¨: LayerStack é©ï½¸è¬æ§­ã»èã»E¨E³ç¹§å®ï½¡E¨éï½º(èE½¦éè²ãE test vector ç¸ºE¨è±ç¢ï½¼ã»ç¸²ã»
void dump_layer_stack(const Position& pos) {
	int pb8 = progress_bucket(pos);
	int pb4 = pb8 / 2;
	float prog_sum = g_progress_loaded ? (ensure_progress(pos), pos.state()->p2_prog_sum) : 0.0f;
	int wb = 0, ww = 0;
	Hand hb = pos.hand_of(BLACK), hw = pos.hand_of(WHITE);
	for (PieceType pt = PAWN; pt < PIECE_HAND_NB; ++pt) {
		wb += hand_count(hb, pt) * P2_HAND_PIECE_WEIGHT[pt];
		ww += hand_count(hw, pt) * P2_HAND_PIECE_WEIGHT[pt];
	}
	int total = wb + ww;
	int thr = P2_HAND_VALUE_THRESHOLD[pb4];
	int hbk = total >= thr ? 1 : 0;
	int stack = get_layer_stack_index(pos);
	const int rg_v1 = P2_LS_PROGREGIME ? YaneuraOu::P2Volm::value(pos) : -1;
	sync_cout << "info string p2stack mode="
	          << (P2_LS_PROGREGIME ? "PROGREGIME"
	              : (P2_LS_P8H2 ? "P8H2" : (P2_LS_P4H2 ? "P4H2" : "LEGACY")))
	          << " v1=" << rg_v1
	          << " prog_sum=" << prog_sum
	          << " progress_bucket8=" << pb8 << " progress_bucket4=" << pb4
	          << " threshold[pb4]=" << thr
	          << " hand[B: P=" << hand_count(hb, PAWN) << " L=" << hand_count(hb, LANCE)
	          << " N=" << hand_count(hb, KNIGHT) << " S=" << hand_count(hb, SILVER)
	          << " B=" << hand_count(hb, BISHOP) << " R=" << hand_count(hb, ROOK)
	          << " G=" << hand_count(hb, GOLD) << "]"
	          << " hand[W: P=" << hand_count(hw, PAWN) << " L=" << hand_count(hw, LANCE)
	          << " N=" << hand_count(hw, KNIGHT) << " S=" << hand_count(hw, SILVER)
	          << " B=" << hand_count(hw, BISHOP) << " R=" << hand_count(hw, ROOK)
	          << " G=" << hand_count(hw, GOLD) << "]"
	          << " black_weighted=" << wb << " white_weighted=" << ww
	          << " total=" << total
	          << " hand_bucket=" << hbk << " stack=" << stack << sync_endl;
}

// u8(èï½¥èãEEE8(é©¥éªâ©)ç¸ºE®èã»E©éªÂã»VX2 maddubs+madd(è­ï½¬é½ã»å³E¸ºE§ç¸ºE¯é¬ï½½è¥å¾âç¸ºã»scalar ç¸ºE¨è³Âé¾E´)ç¸²ã»
// n ç¸ºE¯ 32 ç¸ºE®èè´çèè´ç½²(é¶E¯è¬¨E°ç¸ºE¯ scalar)ç¸²ã»
static inline std::int32_t dot_u8i8(const std::uint8_t* __restrict a, const std::int8_t* __restrict w, int n) {
#if defined(USE_AVX2)
	__m256i acc = _mm256_setzero_si256();
	const __m256i ones = _mm256_set1_epi16(1);
	int i = 0;
	for (; i + 32 <= n; i += 32) {
		__m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
		__m256i vw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
		__m256i prod = _mm256_maddubs_epi16(va, vw);
		acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, ones));
	}
	__m128i lo = _mm256_castsi256_si128(acc);
	__m128i hi = _mm256_extracti128_si256(acc, 1);
	__m128i s = _mm_add_epi32(lo, hi);
	s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
	s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
	std::int32_t sum = _mm_cvtsi128_si32(s);
	for (; i < n; ++i) sum += (std::int32_t)w[i] * (std::int32_t)a[i];
	return sum;
#else
	std::int32_t sum = 0;
	for (int i = 0; i < n; ++i) sum += (std::int32_t)w[i] * (std::int32_t)a[i];
	return sum;
#endif
}

// scrambled affine: èE½ºèãEOUT(8ç¸ºE®èè´çE ç¹§åE½¸E¦è»èå¡é¨éªÂE¸s ç¹ï½¬ç¹§E¤ç¹§E¢ç¹§E¦ç¹ãE[i_chunk][OUT][4]ç¸²ã»
// out[o] = bias[o] + Eï½£_i in[i]*w[o][i]ç¸²Eªn ç¸ºE¯ u8ç¸²ãEç¸ºE¯ i8(è­ï½¬é½ã»å³E¸ºE§é¬ï½½è¥å¾âç¸ºå£½E­E£éï½º)ç¸²ã»
#if defined(P2_FC0_PF) && defined(USE_AVX2)
#if !defined(P2_FC0_PF_BYTES)
#define P2_FC0_PF_BYTES 2048
#endif
// P2_FC0_PF: affine_par (下) と同じ式・同じ順の密な積和で、 重みを PFB byte 先まで先読みするところだけが違う
//   (OPN の affine_fc0_pf と同じ。 PFB を引数にした写し)。
// SAFETY: in は in_pad byte 以上、 ws は (in_pad / 4) × NOUT × 4 byte、 bias / out は NOUT 要素。 loadu / storeu。
//   prefetch は配列の終端より先も指しうるが、 prefetch は fault しない (値を読まない) ので安全。
//   不変条件: NOUT は 8 の倍数 (8..64)、 in_pad は 4 の倍数。 maddubs の飽和条件は affine_par と同じ。
template <int NOUT, int PFB>
static inline void affine_pf_t(const std::uint8_t* __restrict in, const std::int8_t* __restrict ws, const std::int32_t* __restrict bias,
                               int in_pad, std::int32_t* __restrict out) {
	static_assert(NOUT % 8 == 0 && NOUT >= 8 && NOUT <= 64, "affine_pf_t の NOUT は 8 の倍数で 8..64");
	constexpr int NG = NOUT / 8;
	constexpr int CB = NOUT * 4;
	__m256i acc[NG];
	for (int g = 0; g < NG; ++g) acc[g] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + g * 8));
	const __m256i ones = _mm256_set1_epi16(1);
	const int nc = in_pad / 4;
	for (int ic = 0; ic < nc; ++ic) {
		std::int32_t a4;
		std::memcpy(&a4, in + ic * 4, 4);
		const __m256i a = _mm256_set1_epi32(a4);
		const std::int8_t* wb = ws + (std::size_t) ic * CB;
		for (int b = 0; b < CB; b += 64) _mm_prefetch(reinterpret_cast<const char*>(wb) + PFB + b, _MM_HINT_T0);
		for (int g = 0; g < NG; ++g)
			acc[g] = _mm256_add_epi32(acc[g], _mm256_madd_epi16(_mm256_maddubs_epi16(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wb + g * 32))), ones));
	}
	for (int g = 0; g < NG; ++g) _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + g * 8), acc[g]);
}
#endif
#if defined(P2_FT_OUT16)
#if !defined(P2_FC0_PF16_BYTES)
#define P2_FC0_PF16_BYTES 2048
#endif
// P2_FT_OUT16: fc0 の i16 入力版。 ws16 = [in_pad / 2][NOUT][2] の i16 (読み込み時に fc0_w から作る)、 bias_m = m·bias。
//   入力 2 列を i32 に詰めて broadcast し、 madd_epi16 で 8 出力ずつ i32 に積和する。 最後に >> FT_MS (算術 shift = floor)。
//   値: |x| <= 126·m <= 4032 (m <= 32)、 |w| <= 128 なので madd の 2 項和 <= 1.04e6、 1024 列の総和 <= 5.3e8 < 2^31 (溢れない)。
//   m = 128 のときだけ総和が 2^31 に近づく (16128·128·1024 = 2.1e9) ので FT_MS <= 6 を要求する。
// SAFETY: in は in_pad 要素 (i16)、 ws16 は in_pad × NOUT 要素、 bias / out は NOUT 要素。 loadu / storeu。 prefetch は fault しない。
//   不変条件: NOUT は 8 の倍数、 in_pad は 2 の倍数。
static_assert(FT_MS <= 6, "fc0 の i32 積和が溢れうる");
template <int NOUT>
static inline void affine16_pf_t(const std::int16_t* __restrict in, const std::int16_t* __restrict ws16, const std::int32_t* __restrict bias_m,
                                 int in_pad, std::int32_t* __restrict out) {
#if defined(USE_AVX2)
	static_assert(NOUT % 8 == 0 && NOUT >= 8 && NOUT <= 64, "affine16_pf_t の NOUT は 8 の倍数で 8..64");
	constexpr int NG = NOUT / 8;
	constexpr int CE = NOUT * 2;   // 入力 2 列ぶんの重みの要素数 (i16)
	__m256i acc[NG];
	for (int g = 0; g < NG; ++g) acc[g] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias_m + g * 8));
	const int np = in_pad / 2;
	for (int ip = 0; ip < np; ++ip) {
		std::int32_t a2;
		std::memcpy(&a2, in + ip * 2, 4);
		const __m256i a = _mm256_set1_epi32(a2);
		const std::int16_t* wb = ws16 + (std::size_t) ip * CE;
		for (int b = 0; b < CE * 2; b += 64) _mm_prefetch(reinterpret_cast<const char*>(wb) + P2_FC0_PF16_BYTES + b, _MM_HINT_T0);
		for (int g = 0; g < NG; ++g)
			acc[g] = _mm256_add_epi32(acc[g], _mm256_madd_epi16(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wb + g * 16))));
	}
	for (int g = 0; g < NG; ++g) _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + g * 8), _mm256_srai_epi32(acc[g], FT_MS));
#else
	for (int o = 0; o < NOUT; ++o) {
		std::int64_t s = bias_m[o];
		for (int i = 0; i < in_pad; ++i) s += (std::int64_t) in[i] * ws16[(std::size_t) (i / 2) * NOUT * 2 + o * 2 + (i % 2)];
		out[o] = (std::int32_t) (s >> FT_MS);
	}
#endif
}
#endif
#if defined(P2_FC1_PREC)
#if defined(P2_RESIDUAL_MIX16) || defined(P2_RESIDUAL_MIXWIDE16) || defined(P2_L4) || defined(P2_THREAT_L1) || defined(P2_THREAT_L2) || defined(P2_TG)
#error "P2_FC1_PREC は MIX16 / L4 / TL1 / TL2 / TG と併用しない (それらは u8 の l2in / l2relu と 127 目盛りを仮定している)"
#endif
// ---------------------------------------------------------------------------
// P2_FC1_PREC (2026-11-19 / runs/prereg/PREREG-20261119-FC1PREC.md の案 (5−)): fc0 の出力から後ろの丸めを細かくする。
//   fc0 活性 (SqrClippedReLU / ClippedReLU) を 2 ビット細かく (0..508 の i16)、 fc1 は i16 入力の madd (重み 刻み 128)、
//   fc1 活性も 2 ビット細かく (0..508 の i16)、 fc2 は i16 の内積 (重み 刻み 64 のまま)。
//   目盛り: fc0 活性 127·4 / fc1 の和 127·4·128 (bias も同じ目盛り) / >> 7 で 127·4 / fc2 の和 127·4·64 = 4·8128 → (x + 2) >> 2 で 8128。
// ---------------------------------------------------------------------------
static constexpr int FC1P_ACT_MAX = 127 << 2;   // 508 (fc0 活性・fc1 活性の上限)
static constexpr int FC1P_FC1_SH  = 7;          // fc1 の和 (127·4·128) -> 127·4
static constexpr std::uint32_t P2_NET_VERSION_FC1P = 0x7AF32F22u;
#if defined(P2_FC1PREC_STATS)
// (c) 探索中の実測: clamp 508 に当たった数と、 madd に入る値が i16 の外だった数 (構造上 0)。 終了時に stderr へ。
struct Fc1pStats {
	std::uint64_t n_eval = 0, act0_hi = 0, act0_n = 0, act1_hi = 0, act1_n = 0, out16 = 0;
	std::int64_t max_fc1 = 0, max_fc2 = 0;   // |fc1 の和|・|fc2 の和| の最大 (i32 の余裕の確認)
	~Fc1pStats() {
		std::fprintf(stderr, "FC1PREC_STATS n_eval=%llu act0_hi=%llu/%llu act1_hi=%llu/%llu out_of_i16=%llu max|fc1|=%lld max|fc2|=%lld\n",
		             (unsigned long long) n_eval, (unsigned long long) act0_hi, (unsigned long long) act0_n,
		             (unsigned long long) act1_hi, (unsigned long long) act1_n, (unsigned long long) out16,
		             (long long) max_fc1, (long long) max_fc2);
	}
};
static Fc1pStats g_fc1p_stats;   // 計測ビルドは Threads=1 で使う (計数は非 atomic)
#define FC1P_STAT(x) do { x; } while (0)
#else
#define FC1P_STAT(x) do { } while (0)
#endif
// fc1: out[o] = bias[o] + Σ in[i]·w[o][i] (i16 x i16 -> i32、 shift なし)。 ws16 = [in_pad / 2][NOUT][2]。
//   affine16_pf_t (fc0) の shift なし・先読みなし版 (重みは 1 bucket 4 KB で L1 に収まる)。
//   値: 0 <= in <= 508、 |w| <= 254 (旧 net の x2 を含む) なので madd の 2 項和 <= 258,064、 32 列の和 <= 4.2e6 (+ bias) で i32 は溢れない。
//   madd_epi16 は i16 x i16 の積を i32 で 2 つずつ足すだけで飽和しない (飽和するのは maddubs) ので、 スカラの整数内積と同じ値になる。
// SAFETY: in は in_pad 要素 (i16)、 ws16 は in_pad × NOUT 要素、 bias / out は NOUT 要素。 どれも loadu / storeu (整列を仮定しない)。
//   不変条件: NOUT は 8 の倍数 (8..64)、 in_pad は 2 の倍数。
template <int NOUT>
static inline void affine16_t(const std::int16_t* __restrict in, const std::int16_t* __restrict ws16, const std::int32_t* __restrict bias,
                              int in_pad, std::int32_t* __restrict out) {
	static_assert(NOUT % 8 == 0 && NOUT >= 8 && NOUT <= 64, "affine16_t の NOUT は 8 の倍数で 8..64");
#if defined(USE_AVX2)
	constexpr int NG = NOUT / 8;
	constexpr int CE = NOUT * 2;
	__m256i acc[NG];
	for (int g = 0; g < NG; ++g) acc[g] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + g * 8));
	const int np = in_pad / 2;
	for (int ip = 0; ip < np; ++ip) {
		std::int32_t a2;
		std::memcpy(&a2, in + ip * 2, 4);
		const __m256i a = _mm256_set1_epi32(a2);
		const std::int16_t* wb = ws16 + (std::size_t) ip * CE;
		for (int g = 0; g < NG; ++g)
			acc[g] = _mm256_add_epi32(acc[g], _mm256_madd_epi16(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wb + g * 16))));
	}
	for (int g = 0; g < NG; ++g) _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + g * 8), acc[g]);
#else
	for (int o = 0; o < NOUT; ++o) {
		std::int32_t s = bias[o];
		for (int i = 0; i < in_pad; ++i) s += (std::int32_t) in[i] * ws16[(std::size_t) (i / 2) * NOUT * 2 + o * 2 + (i % 2)];
		out[o] = s;
	}
#endif
}
// fc2: Σ a[i]·w[i] (i16 x i16 -> i32)。 n は 16 の倍数 (端数はスカラで足す)。
//   値: 0 <= a <= 508、 |w| <= 127 なので 64 要素の和 <= 4.2e6 (i32 は溢れない)。 madd は飽和しない (上の affine16_t の注記)。
// SAFETY: a / w は n 要素 (i16)。 loadu (整列を仮定しない)。
static inline std::int32_t dot_i16(const std::int16_t* __restrict a, const std::int16_t* __restrict w, int n) {
#if defined(USE_AVX2)
	__m256i acc = _mm256_setzero_si256();
	int i = 0;
	for (; i + 16 <= n; i += 16)
		acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)),
		                                              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i))));
	__m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
	s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
	s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
	std::int32_t sum = _mm_cvtsi128_si32(s);
	for (; i < n; ++i) sum += (std::int32_t) a[i] * (std::int32_t) w[i];
	return sum;
#else
	std::int32_t sum = 0;
	for (int i = 0; i < n; ++i) sum += (std::int32_t) a[i] * (std::int32_t) w[i];
	return sum;
#endif
}
#endif  // P2_FC1_PREC
#if defined(P2_FC1_REF)
// P2_FC1_REF (11-19、 計測専用 / PREREG-20261119-FC1PREC.md 段 3 (d)): 本体の後段 (fc0 から後ろ) を f64 で計算した基準値を eval の dump に出す。
//   入力はエンジンの transform の出力 tr (HalfKA + threat + KH の FT 和の pairwise 積、 1/2^FT_MS 刻み) = FT 側の丸めは基準にも入る。
//   重みは環境変数 P2_FC1REF の側ファイル (fc1prec_write.py --ref が weights.bin の f32 から書く。 fc0 の重みも f32)。
//   書式: "P2FC1REF" / i32 NB, L1, IN, L3, L2IN / bucket ごとに f64 の B0[L1] W0[L1][IN] B1[L3] W1[L3][L2IN] B2 W2[L3]。
//   出力 "info string P2FC1REF bucket=.. main_raw=.. ref=.." (どちらも 8128 目盛り、 psqt を含まない main_raw と同じ量)。
namespace fc1ref {
static std::vector<double> g_w;
static int g_nb = 0, g_l1 = 0, g_in = 0, g_l3 = 0, g_l2in = 0;
static bool g_tried = false;
static bool load() {
	if (g_tried) return !g_w.empty();
	g_tried = true;
	const char* p = std::getenv("P2_FC1REF");
	if (p == nullptr) return false;
	std::ifstream f(p, std::ios::binary);
	char mg[8] = {};
	std::int32_t h[5] = {};
	if (!f.read(mg, 8) || std::memcmp(mg, "P2FC1REF", 8) != 0 || !f.read(reinterpret_cast<char*>(h), sizeof(h))) return false;
	g_nb = h[0]; g_l1 = h[1]; g_in = h[2]; g_l3 = h[3]; g_l2in = h[4];
	if (g_nb != NUM_BUCKETS || g_l1 != L1_OUT || g_in != FC0_IN || g_l3 != L3 || g_l2in != L2_IN) return false;
	const std::size_t per = (std::size_t) g_l1 * (1 + g_in) + (std::size_t) g_l3 * (1 + g_l2in) + 1 + g_l3;
	g_w.resize(per * g_nb);
	if (!f.read(reinterpret_cast<char*>(g_w.data()), (std::streamsize) (g_w.size() * sizeof(double)))) { g_w.clear(); return false; }
	return true;
}
template <class T>
static void print(int b, const T* tr, std::int32_t main_raw) {
	if (!load()) { sync_cout << "info string P2FC1REF unavailable" << sync_endl; return; }
	const std::size_t per = (std::size_t) g_l1 * (1 + g_in) + (std::size_t) g_l3 * (1 + g_l2in) + 1 + g_l3;
	const double* B0 = g_w.data() + per * b;
	const double* W0 = B0 + g_l1;
	const double* B1 = W0 + (std::size_t) g_l1 * g_in;
	const double* W1 = B1 + g_l3;
	const double* B2 = W1 + (std::size_t) g_l3 * g_l2in;
	const double* W2 = B2 + 1;
	const double tsc = 1.0 / (double) (1 << FT_MS);
	double l1[L1OUT_MAX], l2[L2PAD_MAX] = {};
	for (int o = 0; o < g_l1; ++o) {
		double s = B0[o];
		for (int i = 0; i < g_in; ++i) s += W0[(std::size_t) o * g_in + i] * ((double) tr[i] * tsc);
		l1[o] = s;
	}
	const int M = g_l1 - 1, sh = g_l1_main_shift;
	for (int i = 0; i < M; ++i) {
		const double v = l1[i];
		l2[i] = std::min(std::max(v * v / (double) (1LL << (2 * sh + 7)), 0.0), 127.0);
		l2[M + i] = std::min(std::max(v / (double) (1LL << sh), 0.0), 127.0);
	}
	double out = B2[0] + l1[M];
	for (int o = 0; o < g_l3; ++o) {
		double s = B1[o];
		for (int j = 0; j < g_l2in; ++j) s += W1[(std::size_t) o * g_l2in + j] * l2[j];
		out += W2[o] * std::min(std::max(s, 0.0), 127.0);
	}
	char buf[160];
	std::snprintf(buf, sizeof(buf), "info string P2FC1REF bucket=%d main_raw=%d ref=%.6f", b, (int) main_raw, out);
	sync_cout << buf << sync_endl;
}
}  // namespace fc1ref
#endif
static inline void affine_par(const std::uint8_t* __restrict in, const std::int8_t* __restrict ws,

                              const std::int32_t* __restrict bias, int in_pad, int nout, std::int32_t* __restrict out) {
#if defined(USE_AVX2)
	const int ng = nout / 8;           // èE½ºèå¸ãç¹ï½«ç¹ï½¼ç¹ç²çEè·ã»èE½ºèãE
	const int nc = in_pad / 4;         // 4èï½¥èå¸ã¡ç¹ï½£ç¹ï½³ç¹§E¯è¬¨E°
	__m256i acc[8];
	for (int g = 0; g < ng; ++g)
		acc[g] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + g * 8));
	const __m256i ones = _mm256_set1_epi16(1);
	for (int ic = 0; ic < nc; ++ic) {
		__m256i a = _mm256_set1_epi32(*reinterpret_cast<const std::int32_t*>(in + ic * 4));
		const std::int8_t* wb = ws + (std::size_t)ic * nout * 4;
		for (int g = 0; g < ng; ++g) {
			__m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wb + g * 32));
			acc[g] = _mm256_add_epi32(acc[g], _mm256_madd_epi16(_mm256_maddubs_epi16(a, w), ones));
		}
	}
	for (int g = 0; g < ng; ++g)
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(out + g * 8), acc[g]);
#else
	(void)ws; (void)in; (void)in_pad;
	for (int o = 0; o < nout; ++o) out[o] = bias[o];
#endif
}

// ---------------------------------------------------------------------------
// 2026-11-05 (fable) P2_SPARSE_FC0: fc0 の入力 (pairwise 積の u8) は大半が 0 なので、
//   非零の 4 バイト chunk だけを積和する (Stockfish の sparse input と同じ機構)。
//   bit-exact の根拠:
//     ①飛ばす chunk は入力 4 バイトがすべて 0 なので、 密な版での寄与 maddubs(0, w) = 0 は
//       レーンに 0 を足していただけ。 ②maddubs は a <= 126 (pairwise 出力の上限) かつ |w| <= 127
//       なので 2 項の和 <= 32,004 < 32,767 で飽和しない。 madd(ones) 後は i32 で、 256 chunk の
//       総和 <= 16.4M なので wrap もしない。 → 密な版の各レーンは**厳密な整数内積**であり、
//       整数の加算は順序・項の省略 (0 項) に依らず同じ値になる。
//   守るべき不変条件: in の 4 バイト chunk が 0 のときだけ飛ばす (非零 chunk を 1 つでも
//     落とすと値が変わる)。 nnz[] の容量は in_pad/4 以上 (全 chunk 非零でも溢れない)。
// ---------------------------------------------------------------------------
#if defined(USE_AVX2)
struct NnzLut {
	alignas(64) std::uint16_t idx[256][8];
	NnzLut() {
		for (int m = 0; m < 256; ++m) {
			int k = 0;
			for (int b = 0; b < 8; ++b) if (m & (1 << b)) idx[m][k++] = (std::uint16_t)b;
			for (; k < 8; ++k) idx[m][k] = 0;
		}
	}
};
static const NnzLut g_nnz_lut;

// in[0..n_bytes) を 4 バイト chunk で走査し、 非零 chunk の番号を out に昇順で積んで個数を返す。
// n_bytes は 32 の倍数。 out は n_bytes/4 + 8 要素以上 (末尾の 8 要素は書き捨て用の余白)。
static inline int find_nnz_chunks(const std::uint8_t* __restrict in, int n_bytes, std::uint16_t* __restrict out) {
	const __m256i zero = _mm256_setzero_si256();
	const __m128i inc  = _mm_set1_epi16(8);
	__m128i base = _mm_setzero_si128();
	int count = 0;
	for (int i = 0; i < n_bytes; i += 32) {
		const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
		// 1 bit = その 4 バイト chunk が非零
		const unsigned mask = (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, zero))) ^ 0xFFu;
		const __m128i idx = _mm_load_si128(reinterpret_cast<const __m128i*>(g_nnz_lut.idx[mask]));
		_mm_storeu_si128(reinterpret_cast<__m128i*>(out + count), _mm_add_epi16(idx, base));
		count += __builtin_popcount(mask);
		base = _mm_add_epi16(base, inc);
	}
	return count;
}

#if defined(P2_FT_OUT16) && defined(P2_FT_SPARSE16)
// S2 (11-18) P2_FT_SPARSE16: fc0 の i16 入力版を、 非零の入力 2 列 (4 byte = find_nnz_chunks の 1 chunk) だけ回す。
//   値: 飛ばすのは 2 列とも 0 の組だけで、 その寄与 madd(0, w) = 0 なので、 affine16_pf_t と同じ整数の和 (溢れの上限も同じ)。
// SAFETY: in は in_pad 要素 (i16 = in_pad * 2 byte、 32 の倍数)、 ws16 は in_pad × NOUT 要素、 bias_m / out は NOUT 要素。
//   nnz は in_pad / 2 + 8 要素 (find_nnz_chunks の末尾の書き捨て 8 要素を含む)。 不変条件: in_pad <= 4096。
template <int NOUT>
static inline void affine16_sparse_t(const std::int16_t* __restrict in, const std::int16_t* __restrict ws16, const std::int32_t* __restrict bias_m,
                                     int in_pad, std::int32_t* __restrict out) {
	constexpr int NG = NOUT / 8;
	alignas(32) std::uint16_t nnz[4096 / 2 + 8];
	const int cnt = find_nnz_chunks(reinterpret_cast<const std::uint8_t*>(in), in_pad * 2, nnz);
	__m256i acc[NG];
	for (int g = 0; g < NG; ++g) acc[g] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias_m + g * 8));
	for (int k = 0; k < cnt; ++k) {
		const int ip = nnz[k];
		std::int32_t a2;
		std::memcpy(&a2, in + ip * 2, 4);
		const __m256i a = _mm256_set1_epi32(a2);
		const std::int16_t* wb = ws16 + (std::size_t) ip * NOUT * 2;
		for (int g = 0; g < NG; ++g)
			acc[g] = _mm256_add_epi32(acc[g], _mm256_madd_epi16(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wb + g * 16))));
	}
	for (int g = 0; g < NG; ++g) _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + g * 8), _mm256_srai_epi32(acc[g], FT_MS));
}
#define P2_FC0_16 affine16_sparse_t
#elif defined(P2_FT_OUT16)
#define P2_FC0_16 affine16_pf_t
#endif
// affine_par と同じ scrambled 重み [ic][nout][4] を、 nnz に並んだ chunk だけ回す版。
static inline void affine_par_sparse(const std::uint8_t* __restrict in, const std::int8_t* __restrict ws,
                                     const std::int32_t* __restrict bias, const std::uint16_t* __restrict nnz, int nnz_cnt,
                                     int nout, std::int32_t* __restrict out) {
	const int ng = nout / 8;
	__m256i acc[8];
	for (int g = 0; g < ng; ++g)
		acc[g] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + g * 8));
	const __m256i ones = _mm256_set1_epi16(1);
	for (int k = 0; k < nnz_cnt; ++k) {
		const int ic = nnz[k];
		std::int32_t a4; std::memcpy(&a4, in + ic * 4, 4);
		const __m256i a = _mm256_set1_epi32(a4);
		const std::int8_t* wb = ws + (std::size_t)ic * nout * 4;
		for (int g = 0; g < ng; ++g) {
			const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wb + g * 32));
			acc[g] = _mm256_add_epi32(acc[g], _mm256_madd_epi16(_mm256_maddubs_epi16(a, w), ones));
		}
	}
	for (int g = 0; g < ng; ++g)
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(out + g * 8), acc[g]);
}
#endif

// ---------------------------------------------------------------------------
// LayerStacks propagate (1 bucket)
// ---------------------------------------------------------------------------
// 2026-10-08: TL1 / TL2 の積和カーネル (i16 x i16 -> i64)。 **スカラ版と bit-exact**。
//   整数加算は結合則・交換則を満たすので、 **桁あふれしない限り**順序を変えても和は同一。
//   桁あふれの上界 (i32 レーンあたり 16 項):
//     |thr| <= 32767 (i16) かつ |w_q| <= 2048 (学習器の clip ±2 x 1024。 load 時に検査)
//     → 16 * 32767 * 2048 = 1,073,725,440 < 2^31 = 2,147,483,648  ✅ レーンは絶対にあふれない
//   レーンの水平和は 8 * 1.07e9 = 8.6e9 で i32 を超えるので **i64 で**足す。
//   🚨n2k が 16 の倍数でない構成 (P2_THREAT_OUT が 8 の倍数でない等) では端数をスカラで処理する。
//   🚨TL2 (threat -> L2 入力、 3,840 MAC/eval) でもそのまま使えるように関数にしてある。
static inline std::int64_t tl_dot_2k(const std::int16_t* __restrict thr_cat,
                                     const std::int16_t* __restrict w, int n2k) {
	std::int64_t s = 0;
	int c = 0;
#if defined(USE_AVX2)
	if (n2k >= 16) {
		__m256i acc = _mm256_setzero_si256();
		for (; c + 16 <= n2k; c += 16) {
			const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(thr_cat + c));
			const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c));
			// madd_epi16: 隣り合う 2 項を i32 に足す。 1 命令で 16 MAC。
			acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a, v));
		}
		alignas(32) std::int32_t lane[8];
		_mm256_store_si256(reinterpret_cast<__m256i*>(lane), acc);
		for (int i = 0; i < 8; ++i) s += (std::int64_t) lane[i];   // ★水平和は i64
	}
#endif
	for (; c < n2k; ++c) s += (std::int64_t) thr_cat[c] * (std::int64_t) w[c];
	return s;
}

// 🚨tl1_bucket / tl1_thr_* は **threat acc が有効な経路からだけ**渡す (fable 指摘 C-2)。
//   既定の nullptr では TL1 を足さない。 want_aux や threat 無しの経路で足すと
//   古い/未定義の accumulator を読むことになる。
#if defined(P2_FABLE_PROBE)
// 2026-11-10 (PROPSPLIT): T_prop (2,942 cycle = eval の 17%) を
//   **fc0** (2,048 u8 x 16 出力) と **残り** (SqrClippedReLU + fc1 32->64 + fc2 64->1) に割る。
//   🚨namespace FableProbe は下 (6727 行) なのでここでは使えない。 素の static で持つ。
static std::uint64_t fp2_cyc_fc0 = 0, fp2_cyc_rest = 0, fp2_n_fc0 = 0;
#endif
static std::int32_t propagate_bucket(const Bucket& bk, const TrT* transformed,
                                     int tl1_bucket = -1,
                                     const std::int16_t* tl1_thr_stm = nullptr,
                                     const std::int16_t* tl1_thr_ntm = nullptr,
                                     std::int32_t* l1_dbg_out = nullptr,
                                     std::int32_t* l2pre_dbg_out = nullptr,
                                     std::int32_t* l2out_dbg_out = nullptr
#if defined(P2_TG)
                                     // (5) TG の検査用 (tgchk)。 3 * MAIN_DIM 要素:
                                     //   [0,M) = g (clamp 後) / [M,2M) = sqr (ゲート前) /
                                     //   [2M,3M) = pre_sq (clamp 前)。 本番は nullptr。
                                     , std::int32_t* tg_dbg_out = nullptr
#endif
                                     ) {
	// fc0: FT -> L1_OUT
	std::int32_t l1[L1OUT_MAX];
#if defined(P2_FABLE_PROBE)
	const std::uint64_t fp_p0 = __rdtsc();
#endif
#if defined(P2_FT_OUT16)
	// P2_FT_OUT16: fc0 を i16 入力で回す (affine16_pf_t)。 出力 = (m·bias + Σ x·w) >> log2 m = floor((m·bias + Σ x·w) / m)。
	static_assert(FC0_WS_ROWS == L1_OUT, "P2_FT_OUT16 の fc0 は 8 の倍数の行だけ (余りの行の内積は u8 専用)");
	R2T_SCOPE(R_MPROP);
	P2_FC0_16<FC0_WS_ROWS>(transformed, bk.fc0_ws16.data(), bk.fc0_bias_m, FC0_IN, l1);
#elif defined(USE_AVX2)
#if defined(P2_SPARSE_FC0)
	std::uint16_t nnz[FC0_IN / 4 + 8];
	const int nnz_cnt = find_nnz_chunks(transformed, FC0_IN, nnz);
	affine_par_sparse(transformed, bk.fc0_ws.data(), bk.fc0_bias, nnz, nnz_cnt, FC0_WS_ROWS, l1);
#else
	R2T_SCOPE(R_MPROP);   // 関数の終わりまで (fc0 + 後段)
#if defined(P2_FC0_PF)
	// P2_FC0_PF (2026-11-17): OPN の F2-1 (affine_fc0_pf) と同じく fc0 の重みを P2_FC0_PF_BYTES 先まで先読みする。
	//   本体の fc0 の重みは bucket ごとに 16 KB (幅 1024 × 16 行) × 16 bucket = 256 KB で、 評価ごとに bucket が替わると L1 に居ない。
	//   式と足す順は affine_par と同じ (chunk の昇順に各 lane へ足す) なので値は同じ。 FC0_WS_ROWS は定数なので核は定数版になる。
	affine_pf_t<FC0_WS_ROWS, P2_FC0_PF_BYTES>(transformed, bk.fc0_ws.data(), bk.fc0_bias, FC0_IN, l1);
#else
	affine_par(transformed, bk.fc0_ws.data(), bk.fc0_bias, FC0_IN, FC0_WS_ROWS, l1);
#endif
#endif
	// 2026-09-15 L1 24: affine_par は 8 の倍数の行 (FC0_WS_ROWS) しか計算しないので、 余りの行 (L1_OUT 25 なら skip) を
	//   内積で足す。 どちらも飽和の無い整数の和なので、 まとめて計算した場合と同じ値になる。
	for (int o = FC0_WS_ROWS; o < L1_OUT; ++o)
		l1[o] = bk.fc0_bias[o] + dot_u8i8(transformed, &bk.fc0_w[(std::size_t)o * FC0_IN], FC0_IN);
#else
	for (int o = 0; o < L1_OUT; ++o)
		l1[o] = bk.fc0_bias[o] + dot_u8i8(transformed, &bk.fc0_w[(std::size_t)o * FC0_IN], FC0_IN);
#endif
	// (3) TL1: threat を clamp を通さず L1 に足す。 2026-10-01。
#if defined(P2_FABLE_PROBE)
	const std::uint64_t fp_p1 = __rdtsc();
	fp2_cyc_fc0 += fp_p1 - fp_p0;
	fp2_n_fc0 += 1;
#endif
	if constexpr (P2_HAS_TL1) {
		if (g_tl1_loaded && tl1_bucket >= 0 && tl1_thr_stm != nullptr && tl1_thr_ntm != nullptr) {
			// FT 経路と同じ qsearch 倍率をかけて揃える。
			const int qs = (g_in_qsearch ? g_qs_threat_scale : 256);
			// 2026-10-08: thr_stm / thr_ntm を 1 本の連続配列にしてから SIMD で回す。
			//   w は [0..K) が stm、 [K..2K) が ntm なので並びがそのまま対応する
			//   (= スカラ版の 2 本のループと同じ項。 bit-exact)。
			//   256 B の memcpy 1 回 (o 16 本で共有) なので 2,048 MAC に対して無視できる。
			alignas(32) std::int16_t thr_cat[2 * THREAT_ACC_DIM];
			std::memcpy(thr_cat, tl1_thr_stm, sizeof(std::int16_t) * THREAT_ACC_DIM);
			std::memcpy(thr_cat + THREAT_ACC_DIM, tl1_thr_ntm,
			            sizeof(std::int16_t) * THREAT_ACC_DIM);
			for (int o = 0; o < L1_OUT; ++o) {
				const std::int16_t* w = g_tl1_w[tl1_bucket][o];
				std::int64_t acc;
				if (g_tl1_simd_ok) {
					acc = tl_dot_2k(thr_cat, w, 2 * THREAT_ACC_DIM);
				} else {
					// 🚨max|q| > 2048 の net (学習器の clip の外) ではレーンがあふれ得るのでスカラに落とす。
					acc = 0;
					for (int c = 0; c < 2 * THREAT_ACC_DIM; ++c)
						acc += (std::int64_t) thr_cat[c] * (std::int64_t) w[c];
				}
				if (qs != 256) acc = (acc * qs) >> 8;
				l1[o] += (std::int32_t)(acc >> 5);
			}
		}
	}
	// Golden Forward 検査用: l1[] をそのまま外へ出す (本番は nullptr なので何もしない)。
	if (l1_dbg_out != nullptr)
		for (int o = 0; o < L1_OUT; ++o) l1_dbg_out[o] = l1[o];
	std::int32_t l1_skip = l1[MAIN_DIM];
	// (4) TL2: threat を L2 入力に足す項 t2[] を **l2in を作る前に**計算する。 2026-10-08。
	//   TL2 を無効にしたビルドでは要素 1 本の配列になり、 以下の if constexpr から参照されない
	//   (= 現行と 1 命令も変わらない)。 零初期化しているので TL2 net が無いときは t2 = 0。
	constexpr int T2_DIM = P2_HAS_TL2 ? L2_IN : 1;
	std::int32_t t2[T2_DIM] = {};
	if constexpr (P2_HAS_TL2) {
		if (g_tl2_loaded && tl1_bucket >= 0 && tl1_thr_stm != nullptr && tl1_thr_ntm != nullptr) {
			// FT 経路 / TL1 と同じ qsearch 倍率をかけて揃える。
			const int qs = (g_in_qsearch ? g_qs_threat_scale : 256);
			// v の並び ([0..K) = stm、 [K..2K) = ntm) に合わせて 1 本の連続配列にする。
			alignas(32) std::int16_t thr_cat[2 * THREAT_ACC_DIM];
			std::memcpy(thr_cat, tl1_thr_stm, sizeof(std::int16_t) * THREAT_ACC_DIM);
			std::memcpy(thr_cat + THREAT_ACC_DIM, tl1_thr_ntm,
			            sizeof(std::int16_t) * THREAT_ACC_DIM);
			// 🚨目盛り S は net の section version で決まる (ver1: S=64 -> 7、 ver2: S=1024 -> 11)。
			//   j のループの外に出して 1 回のロードだけにする (NPS を落とさない)。
			const int sh = (g_tl2_ver == 2) ? 11 : 7;
			// ThreatL2Gain: t2[] の寄与を gain/256 倍する (2026-10-09)。
			//   🚨ループの外で 1 回だけ読む。 既定 256 なら乗算しない = 従来と bit-exact。
			//   overflow: |acc| <= 128 * 32767 * 2048 = 8.59e9、 acc * 4096 = 3.52e13 < 2^63。
			const int gain = g_tl2_gain;
			for (int j = 0; j < L2_IN; ++j) {
				const std::int16_t* v = g_tl2_w[tl1_bucket][j];
				std::int64_t acc;
				if (g_tl2_simd_ok) {
					acc = tl_dot_2k(thr_cat, v, 2 * THREAT_ACC_DIM);
				} else {
					// 🚨max|q| > 2048 の net では i32 レーンがあふれ得るのでスカラに落とす (値は同一)。
					acc = 0;
					for (int c = 0; c < 2 * THREAT_ACC_DIM; ++c)
						acc += (std::int64_t) thr_cat[c] * (std::int64_t) v[c];
				}
				if (qs != 256) acc = (acc * qs) >> 8;
				if (gain != 256) acc = (acc * gain) >> 8;
				t2[j] = (std::int32_t)(acc >> sh);
			}
		}
	}
	// (5) TG: sqr 側に掛けるゲート項 tg[] を **l2in を作る前に**計算する。 2026-10-10。
	//   TG を無効にしたビルドでは要素 1 本の配列になり、 下の if constexpr から参照されない
	//   (= -DP2_TG の無いビルドは 1 命令も変わらない)。 零初期化しているので TG net が
	//   無いときは tg = 0 → g = 256 (等倍) → 従来と bit-exact。
	constexpr int TG_DIM = P2_HAS_TG ? MAIN_DIM : 1;
	std::int32_t tg[TG_DIM] = {};
	if constexpr (P2_HAS_TG) {
		if (g_tg_loaded && tl1_bucket >= 0 && tl1_thr_stm != nullptr && tl1_thr_ntm != nullptr) {
			// FT 経路 / TL1 / TL2 と同じ qsearch 倍率をかけて揃える。
			const int qs = (g_in_qsearch ? g_qs_threat_scale : 256);
			// u の並び ([0..K) = stm、 [K..2K) = ntm) に合わせて 1 本の連続配列にする。
			alignas(32) std::int16_t thr_cat[2 * THREAT_ACC_DIM];
			std::memcpy(thr_cat, tl1_thr_stm, sizeof(std::int16_t) * THREAT_ACC_DIM);
			std::memcpy(thr_cat + THREAT_ACC_DIM, tl1_thr_ntm,
			            sizeof(std::int16_t) * THREAT_ACC_DIM);
			// ThreatGateScale。 🚨i のループの外で 1 回だけ読む (NPS を落とさない)。
			//   overflow: |acc| <= 128 * 32767 * 2048 = 8.59e9、 acc * 1024 = 8.8e12 < 2^63。
			const int gs = g_tg_scale;
			for (int i = 0; i < MAIN_DIM; ++i) {
				const std::int16_t* u = g_tg_w[tl1_bucket][i];
				std::int64_t acc;
				if (g_tg_simd_ok) {
					acc = tl_dot_2k(thr_cat, u, 2 * THREAT_ACC_DIM);
				} else {
					// 🚨max|q| > 2048 の net では i32 レーンがあふれ得るのでスカラに落とす (値は同一)。
					acc = 0;
					for (int c = 0; c < 2 * THREAT_ACC_DIM; ++c)
						acc += (std::int64_t) thr_cat[c] * (std::int64_t) u[c];
				}
				if (qs != 256) acc = (acc * qs) >> 8;
				if (gs != 256) acc = (acc * gs) >> 8;
				// 🚨目盛り S_g = 1024 固定なので shift は 10。 根拠は g_tg_w の宣言部:
				//   (Σ thr_i16 * u_q) >> 10 = 254 * Σ thr_f * u、 g = 256 + これ。
				tg[i] = (std::int32_t)(acc >> 10);
			}
		}
	}
	// L2 èï½¥èãE(30, padding 0)
#if defined(P2_FC1_PREC)
	// P2_FC1_PREC: fc0 活性を 2 ビット細かく (0..508、 127·4 目盛りの i16)。 shift を 2 減らし clamp の上限を 4 倍にするだけで式の形は同じ。
	alignas(32) std::int16_t l2in[L2PAD_MAX] = {};
	using L2T = std::int16_t;
	constexpr int L2IN_MAX = FC1P_ACT_MAX, L2IN_XB = 2;
#else
	std::uint8_t l2in[L2PAD_MAX] = {};
	using L2T = std::uint8_t;
	constexpr int L2IN_MAX = 127, L2IN_XB = 0;
#endif
	// L1Q128 (2026-11-16): 本体 MAIN_DIM 行の目盛りは net ごとに違う (s = 6: x64 / s = 7: x128)。
	//   lin = v >> s、 sqr = (v*v) >> (2s + 7)。 L2 入力 (u8) の目盛りはどちらも同じ:
	//   s = 6: 8128^2 / 2^19 = 126.008、 8128 / 2^6 = 127 / s = 7: 16256^2 / 2^21 = 126.008、 16256 / 2^7 = 127。
	//   旧 net (s = 6) では >> 19 / >> 6 で式が従来と同一。 skip 行 (l1[MAIN_DIM]) は常に x64 (8128 目盛り) のまま
	//   なので、 MIX16 の skip_sqr >> 19 と最後の out + l1_skip は触らない。
	//   不変条件: v は i32 なので v*v は i64 に収まり、 sqr >= 0。 本体の理論上限 |v| <= 1024*126*128 + bias ~ 1.65e7
	//   で (v*v) >> 21 ~ 1.3e8 は i32 に収まる。 ループの外で 1 回だけ読む。
	const int l1s = g_l1_main_shift;
	const int l1s_sq = 2 * l1s + 7 - L2IN_XB;
	const int l1s_li = l1s - L2IN_XB;   // P2_FC1_PREC: 7 -> 5 (L1Q128) / 6 -> 4 (旧 net)。 なしなら l1s のまま
	for (int i = 0; i < MAIN_DIM; ++i) {
		std::int32_t v = l1[i];
		std::int32_t sqr = (std::int32_t)(((std::int64_t)v * v) >> l1s_sq);
		// (4) TL2: clamp の **前**に足す (これが「clamp を迂回しない代わりに L1 の幅を迂回する」形)。
		std::int32_t pre_sq = sqr, pre_li = v >> l1s_li;
		// (5) TG: **sqr 側だけ**に乗法ゲートをかける (lin 側は触らない)。 2026-10-10。
		//   g = clamp(256 + tg[i], 0, 1024) = x0.0 〜 x4.0。
		//   不変条件: sqr = (v*v) >> 19 なので常に sqr >= 0。 よって tg = 0 のとき
		//   (sqr * 256) >> 8 == sqr で、 **TG net が無ければ既存と bit-exact**。
		//   i64 で計算する理由: sqr * 1024 は sqr が大きいと i32 を越え得る。
		if constexpr (P2_HAS_TG) {
			const std::int32_t g = clampi(256 + tg[i], 0, 1024);
			pre_sq = (std::int32_t)(((std::int64_t) sqr * g) >> 8);
			// 🚨tg_dbg_out は -DP2_TG のときだけ存在する引数。 `if constexpr` の
			//   捨てられる側も非テンプレート文脈では型検査されるので、 参照は
			//   **前処理で**囲わないと非 TG ビルドが未宣言エラーで落ちる。
#if defined(P2_TG)
			if (tg_dbg_out != nullptr) {
				tg_dbg_out[i]            = g;
				tg_dbg_out[MAIN_DIM + i] = sqr;
			}
#endif
		}
		if constexpr (P2_HAS_TL2) { pre_sq += t2[i]; pre_li += t2[MAIN_DIM + i]; }
#if defined(P2_TG)
		// TG の検査用: TL2 を足した後・clamp の前の pre_sq (tgchk pre)。
		if (tg_dbg_out != nullptr) tg_dbg_out[2 * MAIN_DIM + i] = pre_sq;
#endif
		if (l2pre_dbg_out != nullptr) {
			l2pre_dbg_out[i] = pre_sq;
			l2pre_dbg_out[MAIN_DIM + i] = pre_li;
		}
#if defined(P2_FC1PREC_STATS)
		g_fc1p_stats.act0_hi += (pre_sq > L2IN_MAX) + (pre_li > L2IN_MAX); g_fc1p_stats.act0_n += 2;
#endif
		l2in[i] = (L2T)clampi(pre_sq, 0, L2IN_MAX);
		l2in[MAIN_DIM + i] = (L2T)clampi(pre_li, 0, L2IN_MAX);
	}
        #if defined(P2_RESIDUAL_MIX16) || defined(P2_RESIDUAL_MIXWIDE16)
        // R-Mix: add raw skip amplitude/sign to the deep path while retaining final raw skip.
        // 🚨この構成では L2_IN = 2*MAIN_DIM + 2 なので TL2 もこの 2 本に足す
        //   (学習器の l2_in の並びと必ず同じ式にすること)。 本番 DEFINES には入っていない。
        std::int32_t skip_sqr = (std::int32_t)(((std::int64_t)l1_skip * l1_skip) >> 19);
        std::int32_t skip_lin = l1_skip >> 6;
        if constexpr (P2_HAS_TL2) { skip_sqr += t2[2 * MAIN_DIM]; skip_lin += t2[2 * MAIN_DIM + 1]; }
        if (l2pre_dbg_out != nullptr) {
                l2pre_dbg_out[2 * MAIN_DIM]     = skip_sqr;
                l2pre_dbg_out[2 * MAIN_DIM + 1] = skip_lin;
        }
        l2in[2 * MAIN_DIM]     = (std::uint8_t)clampi(skip_sqr, 0, 127);
        l2in[2 * MAIN_DIM + 1] = (std::uint8_t)clampi(skip_lin, 0, 127);
        #endif
   // fc1: L2_IN(pad L2_PAD) -> L3
#if defined(P2_FC1_PREC)
	// P2_FC1_PREC: fc1 = i16 入力の madd (重み 刻み 128、 和の目盛り 127·4·128) -> >> 7 で 127·4 -> clamp 0..508 (i16)。
	//   fc2 = i16 の内積 (重み 刻み 64、 和の目盛り 127·4·64 = 4·8128) -> (x + 2) >> 2 で 8128 目盛りに戻す (四捨五入)。
	alignas(32) std::int16_t l2relu16[L3_MAX] = {};
	{
		alignas(32) std::int32_t l2out[L3_MAX];
		affine16_t<L3>(l2in, bk.fc1_ws16.data(), bk.fc1_bias, L2_PAD, l2out);
		if (l2out_dbg_out != nullptr)
			for (int o = 0; o < L3; ++o) l2out_dbg_out[o] = l2out[o];
		for (int o = 0; o < L3; ++o) {
			FC1P_STAT((g_fc1p_stats.act1_hi += ((l2out[o] >> FC1P_FC1_SH) > FC1P_ACT_MAX), ++g_fc1p_stats.act1_n,
			           g_fc1p_stats.max_fc1 = std::max<std::int64_t>(g_fc1p_stats.max_fc1, l2out[o] < 0 ? -(std::int64_t) l2out[o] : l2out[o])));
			l2relu16[o] = (std::int16_t) clampi(l2out[o] >> FC1P_FC1_SH, 0, FC1P_ACT_MAX);
		}
	}
#if defined(P2_FC1PREC_STATS)
	++g_fc1p_stats.n_eval;
	for (int i = 0; i < L2_PAD; ++i) g_fc1p_stats.out16 += (l2in[i] < 0 || l2in[i] > FC1P_ACT_MAX);   // madd に入る値 (0..508 の外 = i16 の範囲の誤り)
	for (int o = 0; o < L3; ++o) g_fc1p_stats.out16 += (l2relu16[o] < 0 || l2relu16[o] > FC1P_ACT_MAX);
#endif
	const std::int32_t out4 = bk.fc2_bias[0] + dot_i16(l2relu16, bk.fc2_w16, L3);
	FC1P_STAT((g_fc1p_stats.max_fc2 = std::max<std::int64_t>(g_fc1p_stats.max_fc2, out4 < 0 ? -(std::int64_t) out4 : out4)));
#if defined(P2_FABLE_PROBE)
	fp2_cyc_rest += __rdtsc() - fp_p1;
#endif
	return ((out4 + 2) >> 2) + l1_skip;
#else
	std::uint8_t l2relu[L3_MAX];
#if defined(USE_AVX2)
	std::int32_t l2out[L3_MAX];
	affine_par(l2in, bk.fc1_ws.data(), bk.fc1_bias, L2_PAD, L3, l2out);
	// 注入点の候補 (B) の観測用: `>> 6` と clamp の **前**の fc1 出力 (i32)。 本番は nullptr。
	if (l2out_dbg_out != nullptr)
		for (int o = 0; o < L3; ++o) l2out_dbg_out[o] = l2out[o];
	for (int o = 0; o < L3; ++o) l2relu[o] = (std::uint8_t)clampi(l2out[o] >> 6, 0, 127);
#else
	for (int o = 0; o < L3; ++o) {
		std::int32_t s = bk.fc1_bias[o] + dot_u8i8(l2in, &bk.fc1_w[(std::size_t)o * L2_PAD], L2_PAD);
		if (l2out_dbg_out != nullptr) l2out_dbg_out[o] = s;
		l2relu[o] = (std::uint8_t)clampi(s >> 6, 0, 127);
	}
#endif
#if defined(P2_L4)
	// 2026-11-07: **L4 = 4 層目 (`fc1b`)**。 fc1 と同じ形で 1 段挟む。
	//   l2relu は既に 127 目盛りの u8 なので、 fc1b の bias/重みも fc1 と同一規約にすれば
	//   i8 の対角 64 が**恒等**になる (`(64·x) >> 6 = x`)。
	static constexpr int L3_PAD = ((L3 + 31) / 32) * 32;   // = 64
	std::uint8_t l4relu[L4_MAX];
#if defined(USE_AVX2)
	std::int32_t l4out[L4_MAX];
	affine_par(l2relu, bk.fc1b_ws.data(), bk.fc1b_bias, L3_PAD, L4, l4out);
	for (int o = 0; o < L4; ++o) l4relu[o] = (std::uint8_t)clampi(l4out[o] >> 6, 0, 127);
#else
	for (int o = 0; o < L4; ++o) {
		std::int32_t s = bk.fc1b_bias[o] + dot_u8i8(l2relu, &bk.fc1b_w[(std::size_t)o * L3_PAD], L3_PAD);
		l4relu[o] = (std::uint8_t)clampi(s >> 6, 0, 127);
	}
#endif
	// fc2: L4 -> 1
	std::int32_t out = bk.fc2_bias[0] + dot_u8i8(l4relu, &bk.fc2_w[0], L4);
#else
	// fc2: 32 -> 1
	std::int32_t out = bk.fc2_bias[0] + dot_u8i8(l2relu, &bk.fc2_w[0], L3);
#endif
#if defined(P2_FABLE_PROBE)
	fp2_cyc_rest += __rdtsc() - fp_p1;
#endif
        return out + l1_skip;
#endif  // P2_FC1_PREC
}
// ---------------------------------------------------------------------------
// pairshape : 12 pair_type EEE16 shape EEE8 bucket EEE2 persp
// ---------------------------------------------------------------------------
static const PieceType PAIR_A[12] = { GOLD, GOLD, SILVER, SILVER, ROOK, BISHOP,
                                      PRO_PAWN, HORSE, DRAGON, PAWN, SILVER, ROOK };
static const PieceType PAIR_B[12] = { SILVER, GOLD, SILVER, KNIGHT, PAWN, SILVER,
                                      GOLD, GOLD, GOLD, KNIGHT, BISHOP, GOLD };

// shape è»ã»E¡ã»(rust classify_pair_shape ç¸ºE¨è·å¾¡E¸Â)ç¸²ãç¶¾ç¹§éEE¤ 0..15ç¸²ã»
static int classify_pair_shape(Color persp, Square a, Square b) {
	int fa = (int)file_of(a), fb = (int)file_of(b);
	int ra_raw = (int)rank_of(a), rb_raw = (int)rank_of(b);
	int ra, rb;
	if (persp == BLACK) { ra = ra_raw; rb = rb_raw; }
	else { ra = 8 - ra_raw; rb = 8 - rb_raw; }
	int df_s = fa - fb, dr_s = ra - rb;
	int df = df_s < 0 ? -df_s : df_s;
	int dr = dr_s < 0 ? -dr_s : dr_s;
	if (df == 0) return 0;  // SameFile
	if (dr == 0) return 1;  // SameRank
	if ((df == 1 && dr == 2) || (df == 2 && dr == 1)) return 2; // KnightShape
	if (df == 1 && dr == 1) return 3;  // DiagonalAdjacent
	if (df == 1) return 4;  // AdjacentFile
	if (dr == 1) return 5;  // AdjacentRank
	int cheby = df > dr ? df : dr;
	if (cheby == 1) return 6;  // Distance1
	if (cheby == 2) return 7;  // Distance2
	if (dr_s < 0 && dr > df) return 9;   // Ahead
	if (dr_s > 0 && dr > df) return 10;  // Behind
	bool fa_l = fa <= 2, fa_r = fa >= 6, fa_c = (fa >= 3 && fa <= 5);
	bool fb_l = fb <= 2, fb_r = fb >= 6, fb_c = (fb >= 3 && fb <= 5);
	if (fa_l && fb_l) return 11;  // SameFlankLeft
	if (fa_r && fb_r) return 12;  // SameFlankRight
	if ((fa_l && fb_r) || (fa_r && fb_l)) return 13;  // CrossFlank
	if (fa_c && fb_c) return 14;  // Centered
	if (cheby >= 3) return 8;  // Distance3Plus
	return 15;  // Other
}

// shape ç¸ºE¯ (éä¹ã, 2ç¹æ§­ãE ç¸ºE®ç¸ºE¿ç¸ºE§è±ï½ºç¸ºE¾ç¹§ä¹ãEç¸ºE§è å¥çéªè²E®å±ã¦ç¹ï½¼ç¹æ¶Îè¹ã»è»ã»E²ã»5èéã»1è¿ããE)ç¸²ã»
static std::uint8_t g_shape_tbl[2][81][81];
static void init_shape_table() {
	for (int p = 0; p < 2; ++p)
		for (int a = 0; a < 81; ++a)
			for (int b = 0; b < 81; ++b)
				g_shape_tbl[p][a][b] = (std::uint8_t)classify_pair_shape((Color)p, (Square)a, (Square)b);
}

// è¬ãEE®ã»perspectiveç¹ï½»bucket ç¸ºE® pairshape é©¥éªâ©è·éE½¨å»Âã»
static std::int32_t pairshape_accum(const Position& pos, Color p, int bucket) {
	const Net& net = *g_net;
	if (net.pairshape_w.empty()) return 0;
	std::int32_t sum = 0;
	const std::uint8_t (*shp)[81] = g_shape_tbl[(int)p];
	for (int pt = 0; pt < 12; ++pt) {
		PieceType A = PAIR_A[pt], B = PAIR_B[pt];
		Bitboard a_bb = pos.pieces(p, A);
		Bitboard b_bb = pos.pieces(p, B);
		if (!a_bb || !b_bb) continue;
		bool same = (A == B);
		// disk index = bucket*(2*192) + persp*192 + pt*16 + shape
		const std::int16_t* wbase = &net.pairshape_w[(std::size_t)bucket * (2 * 192)
		                              + (std::size_t)(int)p * 192 + (std::size_t)pt * 16];
		for (Bitboard ai = a_bb; ai; ) {
			Square asq = ai.pop();
			for (Bitboard bi = b_bb; bi; ) {
				Square bsq = bi.pop();
				if (same && !((int)asq < (int)bsq)) continue;
				sum += (std::int32_t)wbase[shp[(int)asq][(int)bsq]];
			}
		}
	}
	return sum;
}


// ---------------------------------------------------------------------------
// dual_psqt(King-Attacker) / dual_cross(threat) èï½±é¨¾ã» é¬§åµã»è»E©ç¸ºã»
// ---------------------------------------------------------------------------
static Bitboard piece_attacks(Piece pc, Square sq, const Bitboard& occ) {
	Color c = color_of(pc);
	switch (type_of(pc)) {
	case PAWN:   return pawnEffect(c, sq);
	case LANCE:  return lanceEffect(c, sq, occ);
	case KNIGHT: return knightEffect(c, sq);
	case SILVER: return silverEffect(c, sq);
	case GOLD: case PRO_PAWN: case PRO_LANCE: case PRO_KNIGHT: case PRO_SILVER:
	             return goldEffect(c, sq);
	case BISHOP: return bishopEffect(sq, occ);
	case ROOK:   return rookEffect(sq, occ);
	case HORSE:  return horseEffect(sq, occ);
	case DRAGON: return dragonEffect(sq, occ);
	case KING:   return kingEffect(sq);
	default:     return Bitboard(ZERO);
	}
}
static inline int oriented_sq(Square sq, Color persp) {
	return (persp == BLACK) ? (int)sq : 80 - (int)sq;
}

// dual_psqt(King-Attacker) ç¸ºE¨ dual_cross(threat) ç¹§ã»1 ç¹ä»£ãç¸ºE§éªè²E®ã»
// (piece_attacks ç¹§å¸ï½§åµEE¸ºE¨ç¸ºE«1èæ§­â¡ç¸ºé¡ï½¨è²E®å±E ç¸ºE¦è³E¡ head ç¸ºE«èå¸·EµE¦)ç¸²ã»
// Dual3: compact king-zone action plus hand pressure.  It contains no
// target-piece relation and therefore does not duplicate Direct Threat rows.
static inline int dual3_class(PieceType pt) {
    switch (pt) {
    case PAWN: return 0; case LANCE: return 1; case KNIGHT: return 2; case SILVER: return 3;
    case GOLD: case PRO_PAWN: case PRO_LANCE: case PRO_KNIGHT: case PRO_SILVER: return 4;
    case BISHOP: return 5; case ROOK: return 6; case HORSE: return 7; case DRAGON: return 8;
    default: return -1;
    }
}
static inline int dual3_distance_band(Square from, Square king) {
    const int d = dist(from, king);
    return d <= 1 ? 0 : (d <= 3 ? 1 : 2);
}
static inline int dual3_cover_band(int n) { return n <= 0 ? 0 : (n == 1 ? 1 : (n == 2 ? 2 : 3)); }
static inline int dual3_hand_category(const Position& pos, Color c) {
    const Hand h = pos.hand_of(c);
    if (hand_count(h, ROOK) || hand_count(h, BISHOP)) return 3;
    if (hand_count(h, GOLD) || hand_count(h, SILVER)) return 2;
    if (hand_count(h, PAWN) || hand_count(h, LANCE) || hand_count(h, KNIGHT)) return 1;
    return 0;
}
// P2_DUAL3_CAP_CP=0 Í dual3³øÌÎÆrhÆµÄg¢Aùèl25ÍÅIcpóÔÌãÀB
#ifndef P2_DUAL3_CAP_CP
#define P2_DUAL3_CAP_CP 25
#endif
static inline std::int32_t dual3_limited(std::int64_t z, int effective_fv) {
    constexpr std::int64_t DUAL3_CAP_CP = P2_DUAL3_CAP_CP;
    if (DUAL3_CAP_CP == 0) return 0;
    const std::int64_t CAP = DUAL3_CAP_CP * std::max(1, effective_fv);
    const std::int64_t az = z >= 0 ? z : -z;
    const std::int64_t v = (az * CAP + (CAP + az) / 2) / (CAP + az);
    return (std::int32_t)(z >= 0 ? v : -v);
}
static std::int32_t dual3_value(const Position& pos, int bucket, int effective_fv) {
    if (P2_DUAL3_CAP_CP == 0) return 0; // cap0ÎÆÅÍDual3Á¥Ê¸àÈª·éB
    if (!g_net->has_dual3) return 0;
    constexpr int ZONE_BASE = 864;
    const std::int16_t* const w = g_net->dual3_w.data();
    const Bitboard occ = pos.pieces();
    std::int64_t total = 0;
    for (Color persp : {BLACK, WHITE}) {
        const Color enemy = ~persp;
        const Square enemy_king = pos.square<KING>(enemy);
        const Bitboard enemy_zone = kingEffect(enemy_king) | Bitboard(enemy_king);
        const int attack_hand = dual3_hand_category(pos, persp);
        int attackers = 0;
        std::int64_t local = 0;
        for (Bitboard pieces = pos.pieces(persp) & ~pos.pieces(KING); pieces; ) {
            const Square from = pieces.pop();
            const Piece pc = pos.piece_on(from);
            const int cls = dual3_class(type_of(pc));
            if (cls < 0) continue;
            const Bitboard attacks = effects_from(pc, from, occ);
            if (!(attacks & enemy_zone)) continue;
            ++attackers;
            const int distance = dual3_distance_band(from, enemy_king);
            const int idx = ((bucket * 9 + cls) * 3 + distance) * 4 + attack_hand;
            local += w[idx];
        }
        if (attackers > 0) local += w[ZONE_BASE + bucket * 2 + (attackers >= 2 ? 1 : 0)];
        total += persp == BLACK ? local : -local;
    }
    return dual3_limited(total, effective_fv);
}

// CROSS27: 全9筋×4〜6段の利き状態だけを扱う独立scalar補正。
// state = 0:none / 1:own only / 2:both / 3:enemy only。
#ifndef P2_CROSS27_CAP_CP
#define P2_CROSS27_CAP_CP 16
#endif
static inline std::int32_t cross27_limited(std::int64_t z, int effective_fv) {
	constexpr std::int64_t CROSS27_CAP_CP = P2_CROSS27_CAP_CP;
	if (CROSS27_CAP_CP == 0) return 0;
	const std::int64_t cap = CROSS27_CAP_CP * std::max(1, effective_fv);
    const std::int64_t az = z >= 0 ? z : -z;
    const std::int64_t v = (az * cap + (cap + az) / 2) / (cap + az);
    return static_cast<std::int32_t>(z >= 0 ? v : -v);
}
static inline int cross27_phase_gate(const Position& pos) {
    int non_pawn_non_king = 0;
    for (Bitboard pieces = pos.pieces() & ~pos.pieces(PAWN) & ~pos.pieces(KING); pieces; ) {
        pieces.pop();
        ++non_pawn_non_king;
    }
    return non_pawn_non_king >= 12 ? 2 : (non_pawn_non_king >= 8 ? 1 : 0);
}
static std::int32_t cross27_value(const Position& pos, int bucket, int effective_fv) {
	if (P2_CROSS27_CAP_CP == 0) return 0; // cap0では利き走査も省略する。
	if (!g_net->has_cross27) return 0;
	const int gate2 = cross27_phase_gate(pos);
	if (gate2 == 0) return 0;
	static constexpr int CENTRAL_BAND[27] = {
		3, 4, 5, 12, 13, 14, 21, 22, 23,
		30, 31, 32, 39, 40, 41, 48, 49, 50,
		57, 58, 59, 66, 67, 68, 75, 76, 77
	};
	const Bitboard occ = pos.pieces();
	const std::int16_t* const w = g_net->cross27_w.data();
	auto sum_for = [&](Color perspective) {
		bool own[27] = {};
		bool enemy[27] = {};
        for (Bitboard pieces = pos.pieces(); pieces; ) {
            const Square from = pieces.pop();
            const Piece pc = pos.piece_on(from);
            const Bitboard attacks = effects_from(pc, from, occ);
			for (int band = 0; band < 27; ++band) {
				if (!(attacks & Bitboard(static_cast<Square>(CENTRAL_BAND[band])))) continue;
				if (color_of(pc) == perspective) own[band] = true;
				else enemy[band] = true;
            }
        }
        std::int64_t sum = 0;
        const int p = perspective == BLACK ? 0 : 1;
		for (int band = 0; band < 27; ++band) {
			const int state = own[band] ? (enemy[band] ? 2 : 1) : (enemy[band] ? 3 : 0);
			const int idx = (((p * 8 + bucket) * 27 + band) * 4) + state;
            sum += w[idx];
        }
        return sum;
    };
	const std::int64_t raw = sum_for(BLACK) - sum_for(WHITE);
	return cross27_limited(gate2 == 2 ? raw : raw / 2, effective_fv);
}
static void cross27_debug_dump(const Position& pos, int bucket, int effective_fv) {
	const int gate2 = cross27_phase_gate(pos);
	const std::int32_t value = cross27_value(pos, bucket, effective_fv);
	sync_cout << "info string P2NNUE CROSS27 bucket=" << bucket
              << " phase_gate2=" << gate2 << " limited=" << value << sync_endl;
}


// PÆeval_v¾¯Dual3ÌÁ¥­ÎÆdÝqbgð\¦·éBTõÌzbgpX©çÍÄÎÈ¢B
static void dual3_debug_dump(const Position& pos, int bucket, int effective_fv) {
    if (!g_net->has_dual3) return;
    constexpr int ZONE_BASE = 864;
    const std::int16_t* const w = g_net->dual3_w.data();
    const Bitboard occ = pos.pieces();
    std::int64_t total = 0;
    int active[2] = {0, 0}, nonzero[2] = {0, 0};
    int active_group[4] = {0, 0, 0, 0}, nonzero_group[4] = {0, 0, 0, 0};
    std::int64_t local_sum[2] = {0, 0};
    for (Color persp : {BLACK, WHITE}) {
        const int ci = (persp == BLACK) ? 0 : 1;
        const Color enemy = ~persp;
        const Square enemy_king = pos.square<KING>(enemy);
        const Bitboard enemy_zone = kingEffect(enemy_king) | Bitboard(enemy_king);
        const int attack_hand = dual3_hand_category(pos, persp);
        int attackers = 0;
        std::int64_t local = 0;
        auto add = [&](int idx, int group) {
            ++active[ci]; ++active_group[group];
            const int q = w[idx];
            if (q != 0) { ++nonzero[ci]; ++nonzero_group[group]; }
            local += q;
        };
        for (Bitboard pieces = pos.pieces(persp) & ~pos.pieces(KING); pieces; ) {
            const Square from = pieces.pop();
            const Piece pc = pos.piece_on(from);
            const int cls = dual3_class(type_of(pc));
            if (cls < 0) continue;
            const Bitboard attacks = effects_from(pc, from, occ);
            if (!(attacks & enemy_zone)) continue;
            ++attackers;
            const int idx = ((bucket * 9 + cls) * 3 + dual3_distance_band(from, enemy_king)) * 4 + attack_hand;
            add(idx, 0);
        }
        if (attackers > 0) add(ZONE_BASE + bucket * 2 + (attackers >= 2 ? 1 : 0), 1);
        local_sum[ci] = local;
        total += (persp == BLACK) ? local : -local;
    }
    sync_cout << "info string P2NNUE Dual3Debug bucket=" << bucket
              << " raw=" << total << " limited=" << dual3_limited(total, effective_fv)
              << " active_b=" << active[0] << " active_w=" << active[1]
              << " nz_b=" << nonzero[0] << " nz_w=" << nonzero[1]
              << " groups_active=" << active_group[0] << "," << active_group[1] << ",0,0"
              << " groups_nz=" << nonzero_group[0] << "," << nonzero_group[1] << ",0,0"
              << " local_b=" << local_sum[0] << " local_w=" << local_sum[1]
              << sync_endl;
}


static void dual_eval(const Position& pos, int bucket,
                      std::int32_t& dual_psqt_out, std::int32_t& dual_cross_out) {
	const Net& net = *g_net;
	dual_psqt_out = 0; dual_cross_out = 0;
	const bool has_dp = !net.dual_w.empty();
	const bool has_dc = !net.dualcross_w.empty();
	if (!has_dp && !has_dc) return;
	const std::int16_t* DW = has_dp ? net.dual_w.data() : nullptr;
	const std::int16_t* CW = has_dc ? net.dualcross_w.data() : nullptr;
	Bitboard occ = pos.pieces();
	std::int32_t attack_sum = 0, defend_sum = 0, cross_total = 0;
	for (Color persp : {BLACK, WHITE}) {
		Color enemy = ~persp;
		Square ok = pos.square<KING>(persp), ek = pos.square<KING>(enemy);
		Bitboard own_kz = kingEffect(ok) | Bitboard(ok);
		Bitboard enemy_kz = kingEffect(ek) | Bitboard(ek);
		Bitboard enemy_bb = pos.pieces(enemy);
		int dpbase = (int)persp * 18144;   // dual_psqt: 2*8*81*14
		int dcbase = (int)persp * 127008;  // dual_cross: 8*81*14*14
		for (Bitboard b = pos.pieces(persp); b; ) {
			Square sq = b.pop();
			Piece pc = pos.piece_on(sq);
			int pidx = (int)type_of(pc) - 1;
			int sqi = oriented_sq(sq, persp);
			Bitboard atk = piece_attacks(pc, sq, occ);
			if (has_dp) {
				if (atk & enemy_kz) attack_sum += DW[dpbase + bucket * 1134 + sqi * 14 + pidx];
				if (atk & own_kz)   defend_sum += DW[dpbase + 9072 + bucket * 1134 + sqi * 14 + pidx];
			}
			if (has_dc) {
				int rowbase = dcbase + bucket * 15876 + sqi * 196 + pidx * 14;
				for (Bitboard t = atk & enemy_bb; t; ) {
					Square d = t.pop();
					cross_total += CW[rowbase + ((int)type_of(pos.piece_on(d)) - 1)];
				}
			}
		}
	}
	dual_psqt_out = attack_sum - defend_sum; // DUAL_OUTPUT_SCALE=1
	dual_cross_out = cross_total;            // DUAL_CROSS_OUTPUT_SCALE=1
}


// ---------------------------------------------------------------------------
// èæEåª FV_SCALE / LMR ç¸ºE®èï½±è­è½E£æºä¿£èï£°èEEtç¸²E±rogress sum ç¹§ã»FV_SCALE_PROG_LO/HI ç¸ºE§è±E½£éä¸å§ç¸²ã»
//   FV ç¸ºE¨ LMR ç¸ºE¯ç¸ºèãEç¸²æéEè³Âç¸ºE® tç¸²éªãè³æãEç¹ï½­ç¹ãEãç¹§E¹ç¹ãEã£ç¹å±âè»ã»å´ç¹§ä¸ï½EFV ç¸ºæï½ºåä¹¢ regime ç¸ºE«
//   èï½¥ç¹§åºå¾ LMR ç¹§ãEEè­ãâè åEä¹¢ regime ç¸ºE«èï½¥ç¹§ã»ç¸²Eµ ç¸ºE¯ p2_search_begin ç¸ºE§1èæ§­â¡ç¸ºé¡ï½¨è²E®å±E èï½±è­å³¨Âã»
// ---------------------------------------------------------------------------
static inline float progress_t(float prog_sum) {
	const float lo = (float)g_fv_prog_lo * 0.01f;
	const float hi = (float)g_fv_prog_hi * 0.01f;
	if (hi <= lo) return (prog_sum >= hi) ? 1.0f : 0.0f; // é¨¾Âè¹ã»é«¢E¾èï½¤é¨¾ã»E»E¢/è·æÂE¤): hi ç¸ºE§ç¹§E¹ç¹ãEã£ç¹ãE
	float t = (prog_sum - lo) / (hi - lo);
	return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
}
// aç«é ç¹§ã»t ç¸ºE§é±å£¼E½E¢é¬æºä¿£ç¸ºå±â»è³E¸ç¹§ã»a,b >= 0 èè´ç½²)ç¸²ã»
static inline int lerp_round(int a, int b, float t) {
	float v = (float)a + t * ((float)b - (float)a);
	return (int)(v + 0.5f);
}

// aç«é ç¹§ã»x ç¸ºã»x0ç«é1 ç¸ºE¸èè¼ï¿¥ç¸ºE¨ç¸ºå²E·å£¼E½E¢é¬æºä¿£(é½ã»å³Eæ¶ã»ç¹§E¯ç¹ï½©ç¹ï½³ç¹å±Ââ¥,b>=0èè´ç½²)ç¸²ã»
static inline int fv_seg(int a, int b, float x, float x0, float x1) {
	if (x1 <= x0) return (x >= x1) ? b : a;
	float t = (x - x0) / (x1 - x0);
	t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
	return lerp_round(a, b, t);
}
// èæEåª FV_SCALE(3ç¹§E¢ç¹ï½³ç¹§E«ç¹ï½¼): è åEä¹¢(opening)ç«åE¸E­é¶E¤(g_fv_scale)ç«å ¤Eµãä¹¢(g_fv_scale_endgame)ç¸²ã»
//   progress sum ç¸ºE§ openingç«ç¥id([LO,HI])ç«å¼¾nd([HI,PROG_END]) ç¹§å®ï½£æºä¿£ç¸²ã»
//   ç¸ºè¼ï½ç¸ºE« root eval >= g_fv_endgame_eval(èE½ªèï½¢)ç¸ºEªç¹§è¾ÂE²é¦æï½ºE¦ç¸ºE«é«¢E¢ç¹§ä¸ï½ç¸ºå¤ï½µãä¹¢ç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½«(=revert)ç¸²ã»
//   è æ¢§å©¿è åé¤: open==mid==end ç¸ºEªç¹§çï½®å£½çç¸²E¦nd==mid ç¸ºEªç¹§çï½¾æºæç¸ºE®2ç¹§E¢ç¹ï½³ç¹§E«ç¹ï½¼(openingç«ç¥id)ç¸ºE¨è³Âé¾E´ç¸²ã»
//   èå¡å³¡ eval ç¹§E³ç¹æ§­Î¦ç¹ãEdump)ç¸ºE®ç¹è¼ãç¹ï½¼ç¹ï½«ç¹èEã£ç¹§E¯é¦E¨éï½ºéï½¨ç¸²ãç¾éï½¢è³E­ç¸ºE¯ g_root_eff_fv ç¹§åE½½E¿ç¸ºã»Âã»
static inline int effective_fv_scale(float prog_sum) {
	const int open = (g_fv_scale_opening > 0) ? g_fv_scale_opening : g_fv_scale;  // 0 = mid に追従 (FV_SCALE だけ渡して補間が暴発する事故を防ぐ)
	const int mid  = g_fv_scale;
	const int end  = (g_fv_scale_endgame > 0) ? g_fv_scale_endgame : mid;  // 0=é¨ãä¹¢revertè¾ï½¡èï½¹(=mid)
	if (!g_progress_loaded) return mid;
	if (open == mid && mid == end) return mid;   // è æ¢§å©¿è åé¤(è³å£½çEè æºæç¸ºE® bit-exact é¨çï½·E¯)
	const float lo = (float)g_fv_prog_lo  * 0.01f;
	const float hi = (float)g_fv_prog_hi  * 0.01f;
	const float ed = (float)g_fv_prog_end * 0.01f;
	int eff = (prog_sum <= hi) ? fv_seg(open, mid, prog_sum, lo, hi)
	                           : fv_seg(mid, end, prog_sum, hi, ed);
	if (eff < 1) eff = 1;       // 0 é«¯E¤éå±ãç¹ï½¼ç¹ãEé¬æºä¿£é¨çï½·E¯ç¸ºE®ç¸ºE¿)
	return eff;
}

// FVNODE: 表を作る。 0 の bucket は bucket の中心 logit((b+0.5)/8) の進行度で effective_fv_scale。
static void rebuild_fv_node() {
	static const float C[8] = { -2.7080502f, -1.4663371f, -0.7949299f, -0.2513144f,
	                            0.2513144f,  0.7949299f,  1.4663371f,  2.7080502f };
	for (int k = 0; k < 8; ++k)
		g_fv_node[k] = (g_fv_scale_bucket[k] > 0) ? g_fv_scale_bucket[k] : effective_fv_scale(C[k]);
}
// FVNODE: LayerStack の index から progress bucket (0..7) を取り出す。 legacy なら同じ値、 P8H2 / PROGREGIME は >> 1。
//   P4H2 は進行度 4 段しか持たないので progress_bucket を呼ぶ (ensure_progress は get_layer_stack_index で済んでいる)。
static inline int fv_pb8(const Position& pos, int ls_bucket) {
	if (P2_LS_P8H2 || P2_LS_PROGREGIME) return ls_bucket >> 1;
	if (P2_LS_P4H2) return progress_bucket(pos);
	return ls_bucket;
}

// ---------------------------------------------------------------------------
// evaluate è­ï½¬è´ã»
// ---------------------------------------------------------------------------

#if defined(P2_FABLE_PROBE)
// 2026-11-05 fable: 冗長 evaluate の計数と evaluate 内の区間 cycle (rdtsc)。 本番ビルドには入らない。
#include <x86intrin.h>
#include <chrono>
namespace FableProbe {
static std::uint64_t n_eval = 0, slot_hit = 0, tab22_hit = 0, tab17_hit = 0, tab14_hit = 0;
static std::uint64_t cyc_eval = 0, cyc_acc = 0, cyc_fwd = 0, cyc_heads = 0;
// 2026-11-10 (G0): ensure_accumulator の区間 (fp_t1 - fp_t0) の**分布**。
//   平均だけでは「T_acc が小さい eval の割合」が判らず、 列挙を別スレッドへ逃がす案の
//   窓 (= 隠せる時間) を過大に見積もる。 帯は <1K / 1-2K / 2-4K / 4-8K / >=8K。
//   🚨P2_FABLE_PROBE の中だけなので本番ビルドには 1 命令も入らない。
static std::uint64_t acc_h[5] = {0, 0, 0, 0, 0};
// 2026-11-10: cyc_fwd を **列挙** と **transform + propagate** に割るための計数。
static std::uint64_t cyc_enum = 0, n_enum = 0;
// 2026-11-10 (TRSPLIT): cyc_fwd の残り (T_tr+prop 2,029 cycle = node 時間の 10.9%) を
//   **transform** (i16 acc -> u8 tr の clamp/pack) と **propagate** (fc0/fc1/fc2) に割る。
//   設計 = runs/prereg/DESIGN-20261110-TRSPLIT.md。 本番経路 (threat あり / aux なし) だけ数える。
static std::uint64_t cyc_tr = 0, n_tr = 0, cyc_prop = 0, n_prop = 0;
static inline void acc_hist(std::uint64_t d) {
	acc_h[d < 1000 ? 0 : d < 2000 ? 1 : d < 4000 ? 2 : d < 8000 ? 3 : 4] += 1;
}
static std::uint64_t zero_chunks = 0, zero_bytes = 0, chunks = 0;
std::uint64_t rows_chain = 0, n_chain = 0, rows_refresh = 0, n_refresh = 0;
static std::uint64_t last_key[1024];
static std::uint64_t *tab22, *tab17, *tab14;
static std::uint64_t tsc0; static std::chrono::steady_clock::time_point wall0;
static void report() {
	double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
	std::fprintf(stderr, "FABLEPROBE eval=%llu slot_hit=%llu tab22=%llu tab17=%llu tab14=%llu cyc_eval=%llu cyc_acc=%llu cyc_fwd=%llu cyc_heads=%llu zc=%llu zb=%llu ch=%llu rows_chain=%llu n_chain=%llu rows_refresh=%llu n_refresh=%llu tsc_hz=%.0f\n",
		(unsigned long long)n_eval, (unsigned long long)slot_hit, (unsigned long long)tab22_hit, (unsigned long long)tab17_hit, (unsigned long long)tab14_hit,
		(unsigned long long)cyc_eval, (unsigned long long)cyc_acc, (unsigned long long)cyc_fwd, (unsigned long long)cyc_heads, (unsigned long long)zero_chunks, (unsigned long long)zero_bytes, (unsigned long long)chunks, (unsigned long long)rows_chain, (unsigned long long)n_chain, (unsigned long long)rows_refresh, (unsigned long long)n_refresh,
		(double)(__rdtsc() - tsc0) / (wall > 0 ? wall : 1.0));
	// 2026-11-10 (G0): T_acc の分布。 帯は <1K / 1-2K / 2-4K / 4-8K / >=8K。
	std::fprintf(stderr, "FABLEPROBE_FWDSPLIT cyc_enum=%llu n_enum=%llu\n",
		(unsigned long long)cyc_enum, (unsigned long long)n_enum);
	std::fprintf(stderr, "FABLEPROBE_TRSPLIT cyc_tr=%llu n_tr=%llu cyc_prop=%llu n_prop=%llu\n",
		(unsigned long long)cyc_tr, (unsigned long long)n_tr,
		(unsigned long long)cyc_prop, (unsigned long long)n_prop);
	std::fprintf(stderr, "FABLEPROBE_PROPSPLIT cyc_fc0=%llu cyc_rest=%llu n_fc0=%llu\n",
		(unsigned long long)fp2_cyc_fc0, (unsigned long long)fp2_cyc_rest,
		(unsigned long long)fp2_n_fc0);
	std::fprintf(stderr, "FABLEPROBE_ACCHIST lt1k=%llu 1k2k=%llu 2k4k=%llu 4k8k=%llu ge8k=%llu\n",
		(unsigned long long)acc_h[0], (unsigned long long)acc_h[1], (unsigned long long)acc_h[2],
		(unsigned long long)acc_h[3], (unsigned long long)acc_h[4]);
}
static struct Init { Init() { tab22 = new std::uint64_t[1u << 22](); tab17 = new std::uint64_t[1u << 17](); tab14 = new std::uint64_t[1u << 14]();
	tsc0 = __rdtsc(); wall0 = std::chrono::steady_clock::now(); std::atexit(report); } } init_;
static inline void probe_key(std::uint64_t key, int ply) {
	++n_eval;
	if (last_key[ply & 1023] == key) ++slot_hit; last_key[ply & 1023] = key;
	std::uint64_t h = key * 0x9E3779B97F4A7C15ull;
	std::uint64_t& a = tab22[h >> (64 - 22)]; if (a == key) ++tab22_hit; else a = key;
	std::uint64_t& b = tab17[h >> (64 - 17)]; if (b == key) ++tab17_hit; else b = key;
	std::uint64_t& c = tab14[h >> (64 - 14)]; if (c == key) ++tab14_hit; else c = key;
}
}
#endif
// 様式項の合計 (cp × eff_fv、 手番基準)。 2026-11-14 (OPN): evaluate_internal から括り出した (中身は従来のまま)。
//   🚨様式項は先手視点なので後手番では反転する (P2_MATPRICE と同じ作法)。 既定 (全部 0) では各 1 比較で抜ける。
static inline __attribute__((always_inline)) std::int32_t style_raw_sum(const Position& pos, Color stm, int eff_fv) {
	std::int32_t s = 0;
	// ---- 振り飛車の様式項 (2026-10-05)。 g_furi_bonus==0 なら 1 比較で抜ける
	//   🚨psqt_val は `base_cp = (main_raw + psqt_val) / eff_fv` の目盛りなので、
	//     cp で指定した G を足すには **G * eff_fv** にする。
	//   🚨psqt_val は手番基準。 様式項は先手視点なので後手番では反転する
	//     (P2_MATPRICE と同じ作法)。
	if (g_furi_bonus != 0) {
		const int fb_black = furibisha_bonus_black_view(pos);
		if (fb_black != 0) {
			const int fb_stm = (stm == BLACK) ? fb_black : -fb_black;
			s += (std::int32_t) ((std::int64_t) fb_stm * eff_fv);
		}
	}
	// ---- 右玉の様式項 (2026-10-05)。 作法は振り飛車と同じ (手番基準・eff_fv 倍)
	if (g_migi_bonus != 0 || migi_gw(g_migi_bonus, g_migi_bonus_white) != 0) {
		const int mb_black = migigyoku_bonus_black_view(pos);
		if (mb_black != 0) {
			const int mb_stm = (stm == BLACK) ? mb_black : -mb_black;
			s += (std::int32_t) ((std::int64_t) mb_stm * eff_fv);
		}
	}
	// ---- 相掛かり角交換右玉の下駄 (2026-10-13)。 作法は右玉と同じ。
	if (g_aiga_bonus != 0 || migi_gw(g_aiga_bonus, g_aiga_bonus_white) != 0) {
		const int ab_black = aigakari_migi_bonus_black_view(pos);
		if (ab_black != 0) {
			const int ab_stm = (stm == BLACK) ? ab_black : -ab_black;
			s += (std::int32_t) ((std::int64_t) ab_stm * eff_fv);
		}
	}
	// ---- 囲いの様式項 (2026-10-06)。 作法は振り飛車・右玉と同じ
	if (g_castle_bonus != 0) {
		const int cb_black = castle_bonus_black_view(pos);
		if (cb_black != 0) {
			const int cb_stm = (stm == BLACK) ? cb_black : -cb_black;
			s += (std::int32_t) ((std::int64_t) cb_stm * eff_fv);
		}
	}
#if 0   // 🏁MatCorr 適用 (2026-10-28 棄却済み)
	// ---- 駒種ごとの値付け補正 (2026-10-21)。 作法は上の様式項と同じ (手番基準・eff_fv 倍)。
	//   🚨既定 (13 本すべて 0) では g_matcorr_any == false なので**ここで 1 比較で抜ける**
	//     = bit-exact / NPS 税 0。
	if (g_matcorr_any) {
		const int mc_black = matcorr_black_view(pos);
		if (mc_black != 0) {
			const int mc_stm = (stm == BLACK) ? mc_black : -mc_black;
			s += (std::int32_t) ((std::int64_t) mc_stm * eff_fv);
		}
	}
#endif
	// ---- 🎯片側版 OwnBoardB (2026-10-22)。 作法は上と同じ (手番基準・eff_fv 倍)。
	//   🚨既定 (0 / 0) では**ここで 1 比較で抜ける** = bit-exact / NPS 税 0。
	if (g_ownboard_b != 0 || g_ownboard_bias != 0) {
		const int ob_black = ownboard_black_view(pos);
		if (ob_black != 0) {
			const int ob_stm = (stm == BLACK) ? ob_black : -ob_black;
			s += (std::int32_t) ((std::int64_t) ob_stm * eff_fv);
		}
	}
#if 0   // 🏁OwnKnightAdv 適用 (2026-10-28 棄却済み)
	// ---- 🎯片側版 OwnKnightAdv (2026-10-22)。 作法は上と同じ。
	//   🚨既定 (0 / 0) では**ここで 1 比較で抜ける** = bit-exact / NPS 税 0。
	if (g_knightadv != 0 || g_knightadv_bias != 0) {
		const int ka_black = knightadv_count(pos);
		if (ka_black != 0) {
			const int ka_stm = (stm == BLACK) ? ka_black : -ka_black;
			s += (std::int32_t) ((std::int64_t) ka_stm * eff_fv);
		}
	}
#endif
	// ---- 🎯片側項 K1 (自玉 8 近傍の金銀 / 2026-10-25)。 作法は上と同じ。
	//   🚨既定 (0 / 0) では**ここで 1 比較で抜ける** = bit-exact / NPS 税 0。
	if (g_kdef != 0 || g_kdef_bias != 0) {
		const int kd_black = kdef_black_view(pos);
		if (kd_black != 0) {
			const int kd_stm = (stm == BLACK) ? kd_black : -kd_black;
			s += (std::int32_t) ((std::int64_t) kd_stm * eff_fv);
		}
	}
#if 0   // 🏁K2 適用 (2026-10-28 棄却済み)
	// ---- 🎯片側項 K2 (自陣 3 段の金銀 / 2026-10-25)。 作法は上と同じ。
	if (g_campgs != 0 || g_campgs_bias != 0) {
		const int cg_black = campgs_black_view(pos);
		if (cg_black != 0) {
			const int cg_stm = (stm == BLACK) ? cg_black : -cg_black;
			s += (std::int32_t) ((std::int64_t) cg_stm * eff_fv);
		}
	}
#endif
	return s;
}

#if defined(P2_OPN)
#include "p2nnue_opn.h"   // OPN (序盤帯専用の副 net) 2026-11-14
// lam = 256 の node: 本体のアキュムレータ・threat・KH を呼ばずに OPN だけで返す (p2nnue_opn.h の不変条件 3)。
//   値は evaluate_internal の混合式で lam = 256 としたもの ((raw_opn_cal + style) / eff_fv) と同じ。
static __attribute__((noinline)) int opn_eval_only(const Position& pos, int pb) {
	const Color stm = pos.side_to_move();
	const float prog_sum = g_progress_loaded ? pos.state()->p2_prog_sum : 0.0f;
	const int eff_fv = g_fv_per_node ? g_fv_node[pb]
	                 : (g_root_fv_valid ? g_root_eff_fv : effective_fv_scale(prog_sum));
	const std::int32_t cal = opn::calibrate(opn::eval_raw(pos, pb), pb, eff_fv);
	const std::int32_t style = style_raw_sum(pos, stm, eff_fv);
#if defined(P2_OPN_PROBE)
	++opn::probe::ev_only;
#endif
	return (eff_fv > 0) ? (cal + style) / eff_fv : 0;
}
#endif
static int evaluate_internal(const Position& pos, bool dump) {
#if defined(P2_PROBE_BUILD)
	++g_pc_eval;
#endif
	if (!g_net || !g_net->loaded) return 0;
	Color stm = pos.side_to_move();
	Color nstm = ~stm;
#if defined(P2_OPN)
	// OPN (2026-11-14): 無効時 (g_active == false かつ dump でない) は 1 分岐だけ。 進行度は従来もこの後の
	//   get_layer_stack_index で計算するので、 ensure_progress が走る node の集合は変わらない。
	int opn_lam = 0, opn_pb = 0;
	if (opn::g_active || (dump && opn::g_loaded)) {
		opn_pb  = progress_bucket(pos);
		opn_lam = opn::lambda(opn_pb, pos.state()->p2_prog_sum);
		if (opn_lam == 256 && !dump) return opn_eval_only(pos, opn_pb);
	}
#endif

#if defined(P2_FABLE_PROBE)
	const std::uint64_t fp_t0 = __rdtsc();
	if (!dump) FableProbe::probe_key((std::uint64_t)pos.key(), pos.game_ply());
#endif
	{ R2T_SCOPE(R_MACC); ensure_accumulator(pos); }
#if defined(P2_FABLE_PROBE)
	const std::uint64_t fp_t1 = __rdtsc();
#endif
	StateInfo* st = pos.state();
	// threat è¿ãE st->p2_acc ç¸ºE¯ HalfKA éç®E²ã»incremental)ç¸ºE®ç¸ºE¾ç¸ºE¾é¯E­è¬âEE ç¸²ã»
	// forward è­ãâEtransform_pair_sum ç¸ºE§ threat acc ç¸ºE¨ on-the-fly è·åãEç¸ºå¶EEè¾E®è»ã»E±å£½æ²èæ¨£â©)ç¸²ã»
	const bool want_threat = P2_HAS_THREAT && g_net->has_threat && !nothreat_env();
	const bool want_threat_factor = P2_HAS_THREAT_FACTOR && g_net->has_threat_factor && !nothreat_env();
	const bool want_aux    = P2_HAS_AUX && g_net->has_aux && !noaux_env();

	const std::int32_t* psqt_us  = st->p2_psqt[stm];
	const std::int32_t* psqt_them = st->p2_psqt[nstm];

	int bucket = get_layer_stack_index(pos);
	// èæEåª FV_SCALE:
	//   è¬ï½¢éï½¢è³E­(dump=false ç¸ºä¹â½ root èï½¤éï½ºè³å£½E¸å»â©)= ç¹ï½«ç¹ï½¼ç¹äº¥E±Âé«±E¢ç¸ºE§èï½ºè³å£¹E ç¸ºã»g_root_eff_fv ç¹§åE½½E¿éï½¨ç¸²ã»
	//     ç«ãE1èæ§­ã»è¬ï½¢éï½¢èã»ã»èï½¨ç¹å¼±ã»ç¹çéè³Âç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½«ç¸ºE«ç¸ºEªç¹§ã»alpha-beta ç¸ºE® minimax è±ç¢ï½¼ã»â²è¢ç¿«Eç¸ºEªç¸ºã»Âã»
	//   èå¡å³¡ eval ç¹§E³ç¹æ§­Î¦ç¹ãEdump=true)ç¹§ã»root è­ï½ªéï½ºè³å£½å¾ = è¿´E¾è»Âé«±E¢ progress ç¸ºä¹ï½é¬æºä¿£(é¦E¨éï½º/ç¹è¼ãç¹ï½¼ç¹ï½«ç¹èEã£ç¹§E¯)ç¸²ã»
	float prog_sum = g_progress_loaded ? st->p2_prog_sum : 0.0f;
	if (dump && g_fv_per_node) rebuild_fv_node();
#if defined(P2_BUCKET_OFFSET)
	// SIDEOFF (11-21): eval コマンド (dump) はその局面の手番を root とみなして表を選ぶ (次の go で p2_search_begin が選び直す)。
	if (dump) select_bucket_off(pos.side_to_move() == BLACK);
#endif
	int eff_fv = g_fv_per_node ? g_fv_node[fv_pb8(pos, bucket)]
	           : ((!dump && g_root_fv_valid) ? g_root_eff_fv : effective_fv_scale(prog_sum));
	std::int32_t psqt_val = g_net->has_psqt ? (psqt_us[bucket] - psqt_them[bucket]) / 2 : 0;

#if defined(P2_AUXCHEAP)
	// AUXCHEAP: board_effect ç¹§E¹ç¹§E­ç¹ï½£ç¹ï½³ + PSQT-only èï£°éãEè±ãEeval è ¢ã»âE³æº¯E¡ã»= short-circuit éâµE­E¢)ç¸²ã»
	// é©¥éªâ© 0 ç¸ºE®ç¸ºæºï½Epsqt_val ç¸ºE¸ç¸ºE®èEEE¸å¼±ã» 0(eval è³æ¦ï½¤ã»ç¸²EE_NOAUXCHEAP=1 ç¸ºE§é¨çï½·E¯ç¹§E¹ç¹§E­ç¹ãEã»ç¸²ã»
	if (P2_HAS_AUXCHEAP && !noauxcheap_env()) {
		std::int32_t ac_stm = 0, ac_nstm = 0;
		compute_auxcheap_psqt(pos, stm, nstm, bucket, ac_stm, ac_nstm);
		psqt_val += (ac_stm - ac_nstm) / 2;
	}
#endif

#if defined(P2_ENDGAME_TEMPO)
	// EGT ç¸ºE¯è·ã»ç¡ 1 lookup ç¸ºE® scalar headç¸²ãE½­E¦éãEgraph ç¸ºE¨è·å¾Â§ stm - nstm ç¸ºE§è·åãEç¸ºå¶Eç¸²ã»
	if (P2_HAS_ENDGAME_TEMPO && g_endgame_tempo_loaded && !no_endgame_tempo_env())
		psqt_val += compute_endgame_tempo(pos, stm, bucket);
#endif

#if defined(P2_MATPRICE) && defined(USE_PIECE_VALUE)
	// 🚨既定では**この分岐ごと消える** (null だったため)。
	// materialValue は**先手視点**、 eval は**手番側視点**なので符号を合わせる。
	if (g_mat_price_adj) {
		const int mat_stm = (stm == BLACK) ? (int) st->materialValue : -(int) st->materialValue;
		psqt_val -= (std::int32_t) ((std::int64_t) mat_stm * g_mat_price_adj / 1000);
	}
#endif

#if defined(P2_MOB)
	// B6 可動域: 反対称スカラー 1 本を bucket 別の重みで psqt_val に足す。
	// 🚨g_mob_w は section が無ければ全 0 なので、 その場合 eval は base と bit-exact。
	//   計算自体は走るので、 この状態の A/B が **純粋な NPS 税**になる。
	if (P2_HAS_MOB && g_mob_w[bucket]) {
		// 学習器の入力は mob_diff / 16 なので、 ここで >> 4 して戻す。
		// 🚨算術右シフトは −∞ 方向に丸まる。 B6 は反対称な量なので符号で分けて対称に丸める。
		const std::int32_t v = Eval::P2NNUE::Mob::mob_diff(pos) * g_mob_w[bucket];
		psqt_val += (v >= 0) ? (v >> MOB_INPUT_SHIFT) : -((-v) >> MOB_INPUT_SHIFT);
	}
#endif

	// ---- 様式項 (振り飛車 / 右玉 / 相掛かり / 囲い / OwnBoardB / K1)。 2026-11-14 (OPN) に style_raw_sum() へ括り出した。
	//   足す値と順は従来と同じ (整数の和なので 1 回で足しても同じ値 = bit-exact)。 OPN の混合では「net の外の下駄」として分けて使う。
	const std::int32_t style_raw = style_raw_sum(pos, stm, eff_fv);
	psqt_val += style_raw;

	// transform (pairwise, ç«ç½®8[1536]): tr[off+j] = (clamp(a[j],0,127)*clamp(a[j+768],0,127))>>7
	alignas(32) TrT tr[FT_MAX];
#if defined(P2_HAND_DENSE)
	// 2026-11-07 (E2): **持駒 14 元 dense** を fc0 の入力の末尾に置く。
	//   [FT_DIM   .. FT_DIM+7 )  = **手番側**の 歩 香 桂 銀 金 角 飛
	//   [FT_DIM+7 .. FT_DIM+14)  = **相手側**の 同じ順
	//   [FT_DIM+14 .. FT_DIM+32) = 0 (padding。 学習器の pad32 と揃える)
	//
	// 🚨🚨**列の順は学習器の `HAND_PIECE_TYPES` = 歩 香 桂 銀 金 角 飛**。
	//   YO の `PieceType` は 歩 香 桂 銀 **角 飛 金** (types.h:542) で**違う**ので下の表で並べ替える。
	//   0 列の net では `0 x 何でも 0` なのでこの誤りは検算 (a) で**検出できない**。
	//   非零の検査 net (`hand_widen_ckpt.py --probe`) で検算 (c') を通すこと。
	//
	// 🚨目盛りは学習器の `k*c/127` と対で `u8 = k*c`。 最大 126 なので
	//   maddubs の飽和不変条件 (a <= 126 かつ |w| <= 127) を満たす。
	//
	// 🎯transform の**前**に埋めてよい (FT の accumulator に依存しない量)。 ここに置けば
	//   transform_pair_sum / transform_pair / ext_* のどの経路でも同じ値が入る。
	{
		static constexpr PieceType HC_PT[7] = { PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK };
		static constexpr int       HC_K [7] = { 7,    31,    31,     31,     31,   63,     63   };
		const Hand h_us   = pos.hand_of(stm);
		const Hand h_them = pos.hand_of(~stm);
		// 🚨先に **HC_PAD 全域を 0 に**してから埋める。 こうしないと HC_DIMS=76 (温度計) の
		//   ビルドで 14..76 が未初期化のごみになる (埋め込みは未実装なので 0 のままが正しい)。
		for (int i = 0; i < HC_PAD; ++i)
			tr[FT_DIM + i] = 0;
		for (int i = 0; i < 7; ++i) {
			tr[FT_DIM + i]     = (std::uint8_t) (HC_K[i] * hand_count(h_us,   HC_PT[i]));
			tr[FT_DIM + 7 + i] = (std::uint8_t) (HC_K[i] * hand_count(h_them, HC_PT[i]));
		}
	}
#endif
	const int half = FT_HALF;
#if defined(P2_TR_STORE32_VERIFY) && !defined(P2_FT_NEWTR)
	// P2_TR_STORE32 の検算 (診断ビルド専用): SIMD の transform が書いた tr を、 従来の scalar の式 (下の #else と同じ) と要素比較する。
	//   b == nullptr なら threat を足さない (transform_pair)。 不一致は 1 件目で abort。
	auto tr_verify = [&](const std::int16_t* a_us, const std::int16_t* b_us, const std::int16_t* a_them, const std::int16_t* b_them) {
		const int qs_scale = (g_in_qsearch ? g_qs_threat_scale : 256);
		auto thr_s = [&](int w) { return qs_scale == 256 ? w : (int)(((long)w * qs_scale) >> 8); };
		const bool use_proj = (b_us != nullptr) && P2_HAS_TPRJ && g_tprj_loaded;
		for (int p = 0; p < 2; ++p) {
			const std::int16_t* a = (p == 0) ? a_us : a_them;
			const std::int16_t* b = (p == 0) ? b_us : b_them;
			for (int j = 0; j < half; ++j) {
				std::int16_t av = a[j], bv = a[j + half];
				if (b != nullptr) {
					av = (std::int16_t) (av + (j < THREAT_LO_DIM ? (std::int16_t) thr_s(b[j]) : (std::int16_t) 0));
					bv = (std::int16_t) (bv + (j < THREAT_HI_DIM ? (std::int16_t) thr_s(b[THREAT_LO_DIM + j]) : (std::int16_t) 0));
				}
				if constexpr (P2_HAS_TPRJ) {
					if (use_proj) {
						const int tvl = thr_s(b[(j + g_tprj_blk[j / 16]) % THREAT_ACC_DIM]);
						const int tvh = thr_s(b[(j + g_tprj_blk[(j + half) / 16]) % THREAT_ACC_DIM]);
						av = (std::int16_t)(av + (std::int16_t)(((int) g_tprj_w[bucket][j] * tvl + 16384) >> 15));
						bv = (std::int16_t)(bv + (std::int16_t)(((int) g_tprj_w[bucket][j + half] * tvh + 16384) >> 15));
					}
				}
				const int x = clampi(av, 0, 254), y = clampi(bv, 0, 254);
				const int ref = (x * y) >> 9;
				if (ref != (int) tr[p * half + j]) {
					std::fprintf(stderr, "TR_STORE32_VERIFY MISMATCH main p=%d j=%d ref=%d got=%d sfen=%s\n", p, j, ref, (int) tr[p * half + j], pos.sfen().c_str());
					std::abort();
				}
			}
		}
		++TrVerify::n_main;
	};
#endif
	// 2èï½¥èå¸æ±è¿ãE èï½¥èãE= a[k]+b[k](i16 wrapping)ç¸²E£ase acc + threat acc ç¸ºE® merge ç¹§ã»
	// transform ç¸ºE«éæ¦ç²ç¸ºå±Ââ½E¸E­é«¢èã°ç¹ãEãµç¹§E¡(us_t/them_t 3KB)ç¸ºE¸ç¸ºE® storeç«æ± eload ç¹§è²çè²å¸âEç¹§ä¹Âã»
	// è²ç®E®æãEç¸ºE¯ç¸²è©£rapping add ç«ãEclamp ç«ãEmul ç«ãE>>7 ç«ãEpackç¸²éªãè»ã»å±¬è¿å»âè³æãEè·å¾¡E¸Â = bit-exactç¸²ã»
#if defined(P2_FT_NEWTR)
	// P2_FT_NEWTR の transform (FT_KS / FT_MS の注記)。 a = 累積器 (FtAcc、 2 * half 列)、 b = threat 側 (254 目盛りの i16、 THREAT_ACC_DIM 列) か nullptr。
	//   列 j: lo = clamp(a[j] + k·b[j], 0, 254k)、 hi = clamp(a[j + half] + k·b[LO + j], 0, 254k)、 out[j] = (lo·hi) >> FT_TRS
	//   (u8 か i16)。 qsearch の threat 倍率 (thr_scaled) は従来と同じ i16 の式で b に掛けてから k 倍する。
	auto transform_x = [&](const FtAcc* a, const std::int16_t* b, TrT* out) {
		const int  qs_scale = (g_in_qsearch ? g_qs_threat_scale : 256);
#if defined(P2_ACC_RANGE)
		++AccRange::n_tr;
		for (int j = 0; j < half; ++j) {
			for (int h = 0; h < 2; ++h) {
				const long long v = a[j + h * half];
				AccRange::add(0, v);
				const int tj = h == 0 ? (j < THREAT_LO_DIM ? j : -1) : (j < THREAT_HI_DIM ? THREAT_LO_DIM + j : -1);
				AccRange::add(1, v + ((b != nullptr && tj >= 0) ? (long long) b[tj] * FT_K : 0));
			}
		}
#endif
#if defined(USE_AVX2)
		// SAFETY: a は 2 * half 要素、 b は THREAT_ACC_DIM 要素、 out は half 要素以上。 loadu / storeu。
		//   不変条件: half % 32 == 0、 THREAT_LO_DIM / THREAT_HI_DIM は 16 の倍数 (従来の transform と同じ 16 列ごとの範囲判定)。
		//   i32 累積器: packs_epi32 は i16 へ飽和するが、 直後の clamp [0, 254k] (254k <= 2032) の外側は飽和しても同じ値になる。
		//     packs は 128 bit lane ごとに並べるので 16 列の並びは [0-3, 8-11 | 4-7, 12-15] になる (lo と hi で同じ並び)。
		//   i16 累積器: a + (b << KS) は i16 wrapping (従来の transform と同じ。 最終の和が i16 に入る前提)。 並びは列の順。
		//   lo / hi とも max(0) と min(254k) を通してから shift するので、 shift の後も i16 に収まり符号も正 (FT_SA / FT_SB の注記)。
		static_assert(FT_HALF % 32 == 0 && THREAT_LO_DIM % 16 == 0 && THREAT_HI_DIM % 16 == 0, "transform の列の単位");
		const __m256i ccl  = _mm256_set1_epi16((short) FT_CLAMP);
		const __m256i zero = _mm256_setzero_si256();
		const __m256i qs_v = _mm256_set1_epi16((short) qs_scale);
		auto thr16 = [&](const std::int16_t* p) {
			__m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
			if (qs_scale == 256) return w;
			__m256i lo16 = _mm256_mullo_epi16(w, qs_v);
			__m256i hi16 = _mm256_mulhi_epi16(w, qs_v);
			return _mm256_or_si256(_mm256_srli_epi16(lo16, 8), _mm256_slli_epi16(hi16, 8));
		};
		// 16 列 (a + c) を i16 に詰めて返す。 t != nullptr なら threat を k 倍して足す。
		auto side16 = [&](int c, const std::int16_t* t) {
#if defined(P2_FT_I32)
			__m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + c));
			__m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + c + 8));
			if (t != nullptr) {
				const __m256i tv = thr16(t);
				x0 = _mm256_add_epi32(x0, _mm256_slli_epi32(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(tv)), FT_KS));
				x1 = _mm256_add_epi32(x1, _mm256_slli_epi32(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(tv, 1)), FT_KS));
			}
			return _mm256_packs_epi32(x0, x1);
#else
			__m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + c));
			if (t != nullptr) x = _mm256_add_epi16(x, _mm256_slli_epi16(thr16(t), FT_KS));
			return x;
#endif
		};
		auto prod16 = [&](int j) {
			__m256i lo = side16(j, (b != nullptr && j < THREAT_LO_DIM) ? b + j : nullptr);
			__m256i hi = side16(j + half, (b != nullptr && j < THREAT_HI_DIM) ? b + THREAT_LO_DIM + j : nullptr);
			lo = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(lo, ccl), zero), FT_SA);
			if constexpr (FT_SB > 0) hi = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(hi, ccl), zero), FT_SB);
			else hi = _mm256_min_epi16(hi, ccl);   // hi < 0 なら積は 0 以下 (u8 は packus が 0 にする / i16 は下で max(0))
			return _mm256_mulhi_epi16(lo, hi);
		};
#if defined(P2_FT_OUT16)
		for (int j = 0; j < half; j += 16) {
			__m256i p = prod16(j);
			if constexpr (FT_SB == 0) p = _mm256_max_epi16(p, zero);
#if defined(P2_FT_I32)
			p = _mm256_permute4x64_epi64(p, 0xD8);   // [0-3, 8-11, 4-7, 12-15] → 列の順
#endif
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j), p);
		}
#else
		for (int j = 0; j < half; j += 32) {
			const __m256i p0 = prod16(j), p1 = prod16(j + 16);
#if defined(P2_FT_I32)
			// packus も lane ごと: 32 列の 4 byte 群は [0,8,16,24 | 4,12,20,28] → permutevar8x32 (0,4,1,5,2,6,3,7) で列の順。
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j),
			                    _mm256_permutevar8x32_epi32(_mm256_packus_epi16(p0, p1), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
#else
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j), _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p1), 0xD8));
#endif
		}
#endif
#else
		auto thr_s = [&](int w) { return qs_scale == 256 ? w : (int)(((long)w * qs_scale) >> 8); };
		for (int j = 0; j < half; ++j) {
			const int av = ft_add_thr(a[j], (b != nullptr && j < THREAT_LO_DIM) ? (std::int16_t) thr_s(b[j]) : 0);
			const int bv = ft_add_thr(a[j + half], (b != nullptr && j < THREAT_HI_DIM) ? (std::int16_t) thr_s(b[THREAT_LO_DIM + j]) : 0);
			out[j] = ft_pair_out(clampi(av, 0, FT_CLAMP), clampi(bv, 0, FT_CLAMP));
		}
#endif
#if defined(P2_TR_STORE32_VERIFY)
		// 検算 (診断ビルド専用): scalar の式と要素比較する。
		for (int j = 0; j < half; ++j) {
			auto ts = [&](int w) { return qs_scale == 256 ? w : (int)(((long)w * qs_scale) >> 8); };
			const int av = ft_add_thr(a[j], (b != nullptr && j < THREAT_LO_DIM) ? (std::int16_t) ts(b[j]) : 0);
			const int bv = ft_add_thr(a[j + half], (b != nullptr && j < THREAT_HI_DIM) ? (std::int16_t) ts(b[THREAT_LO_DIM + j]) : 0);
			const int ref = ft_pair_out(clampi(av, 0, FT_CLAMP), clampi(bv, 0, FT_CLAMP));
			if (ref != (int) out[j]) {
				std::fprintf(stderr, "TR_STORE32_VERIFY MISMATCH ftnew j=%d ref=%d got=%d sfen=%s\n", j, ref, (int) out[j], pos.sfen().c_str());
				std::abort();
			}
		}
		++TrVerify::n_main;
#endif
	};
	auto transform_pair_sum = [&](const FtAcc* a_us, const std::int16_t* b_us,
	                              const FtAcc* a_them, const std::int16_t* b_them) {
		transform_x(a_us, b_us, tr);
		transform_x(a_them, b_them, tr + half);
	};
	auto transform_pair = [&](const FtAcc* a_us, const FtAcc* a_them) {
		transform_x(a_us, nullptr, tr);
		transform_x(a_them, nullptr, tr + half);
	};
#else
	auto transform_pair_sum = [&](const std::int16_t* a_us, const std::int16_t* b_us,
	                              const std::int16_t* a_them, const std::int16_t* b_them) {
		for (int p = 0; p < 2; ++p) {
			const std::int16_t* a = (p == 0) ? a_us : a_them;
			const std::int16_t* b = (p == 0) ? b_us : b_them;
			std::uint8_t* out = tr + p * half;
#if defined(USE_AVX2)
			// FT accumulator の clamp を 0..254 にし、 pairwise を (lo*hi)>>9 で行う。
			// mullo だと 254*254=64516 が int16 を溢れるため、 NAGISA V3.1 / sfnn-1536.h と同じく
			// mulhi(lo << 7, hi) = (lo*128*hi) >> 16 = (lo*hi) >> 9 で上位側に逃がす。
			// lo<<7 は最大 254*128 = 32512 で int16 に収まる。 出力レンジは 126 で従来と同一。
			const __m256i c254 = _mm256_set1_epi16(254);
			const __m256i zero = _mm256_setzero_si256();
			// 2026-09-09: threat acc は先頭 THREAT_ACC_DIM 次元しか持たない。
			//   j (または j+half) がその範囲外なら threat の加算を省く。
			//   THREAT_ACC_DIM == FT_DIM のときは全域で足すので従来と完全一致。
			// 2026-09-11: qsearch 中だけ threat の寄与を scale/256 倍する。
			//   scale==256 (既定) のときは乗算経路に入らないので **従来と bit-exact**。
			const int  qs_scale = (g_in_qsearch ? g_qs_threat_scale : 256);
			const bool qs_full  = (qs_scale == 256);
			const __m256i qs_v  = _mm256_set1_epi16((short)qs_scale);
			auto thr_scaled = [&](const std::int16_t* p) {
				__m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
				if (qs_full) return w;
				// (w * scale) >> 8。 mulhi(w<<8, scale) だと桁が溢れるので lo/hi に分けて 16bit 内で処理。
				__m256i lo16 = _mm256_mullo_epi16(w, qs_v);
				__m256i hi16 = _mm256_mulhi_epi16(w, qs_v);
				// 32bit 値 (hi16:lo16) を 8 bit 右シフト = lo16>>8 | hi16<<8
				return _mm256_or_si256(_mm256_srli_epi16(lo16, 8), _mm256_slli_epi16(hi16, 8));
			};
			// 案1 ブロック射影 (2026-09-30)。 net に P2TPRJ section があるときだけ有効。
			//   🚨ループ不変なので分岐予測は必ず当たる。 -DP2_THREAT_PROJ 無しなら
			//     `if constexpr` ごと消えるので **既存ビルドは bit-exact かつ 1 命令も増えない**。
			const bool use_proj = P2_HAS_TPRJ && g_tprj_loaded;
			// 🚨射影用: qsearch 倍率をかけた accumulator を **2 周ぶん複製**しておく。
			//   こうするとどのずらし量でも 16 本の連続 load が回り込まない。
			alignas(32) std::int16_t tdup[2 * THREAT_ACC_DIM];
			if constexpr (P2_HAS_TPRJ) {
				if (use_proj) {
					for (int c = 0; c < THREAT_ACC_DIM; c += 16) {
						const __m256i v = thr_scaled(b + c);
						_mm256_store_si256(reinterpret_cast<__m256i*>(tdup + c), v);
						_mm256_store_si256(reinterpret_cast<__m256i*>(tdup + c + THREAT_ACC_DIM), v);
					}
				}
			}
#if defined(P2_TR_STORE32)
			// P2_TR_STORE32 (2026-11-17): 16 列ごとに 8 byte の store を 2 回 (packus の lane を extract) していたのを、
			//   32 列ごとに packus(p0, p1) → permute4x64(0xD8) で列の順に戻して 32 byte の store 1 回にする
			//   (OPN の apply_rows_tr2_t / 本家 nnue_feature_transformer.h と同じ形。 本家は permute の代わりに FT の重みを並べ替える)。
			//   hi 側の max(0) は省く: hi < 0 のとき lo << 7 (>= 0) との mulhi は 0 以下で、 packus (i16 → u8 の飽和) が 0 にする
			//   = clamp(hi, 0, 254) で 0 にしたのと同じ値。 lo 側の max(0) は残す (両方負だと積が正になるので)。
			//   値は従来と同じ (bit-exact)。 検算は -DP2_TR_STORE32_VERIFY。
			// SAFETY: a は 2 * half 要素、 b は THREAT_ACC_DIM 要素、 out (tr + p * half) は half byte 以上。 loadu / storeu。
			//   不変条件: half % 32 == 0 (static_assert)。 j + 16 の threat の範囲判定は 16 列ごとに従来と同じ式で行う。
			static_assert(FT_HALF % 32 == 0, "P2_TR_STORE32 は FT_HALF が 32 の倍数");
			auto prod16 = [&](int j) -> __m256i {
				__m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
				if (j < THREAT_LO_DIM)
					lo = _mm256_add_epi16(lo, thr_scaled(b + j));
				__m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j + half));
				if (j < THREAT_HI_DIM)
					hi = _mm256_add_epi16(hi, thr_scaled(b + THREAT_LO_DIM + j));
				if constexpr (P2_HAS_TPRJ) {
					if (use_proj) {
						const int cb = j % THREAT_ACC_DIM;
						const __m256i tvl = _mm256_loadu_si256(
							reinterpret_cast<const __m256i*>(tdup + cb + g_tprj_blk[j / 16]));
						const __m256i tvh = _mm256_loadu_si256(
							reinterpret_cast<const __m256i*>(tdup + cb + g_tprj_blk[(j + half) / 16]));
						lo = _mm256_add_epi16(lo, _mm256_mulhrs_epi16(
							_mm256_load_si256(reinterpret_cast<const __m256i*>(g_tprj_w[bucket] + j)), tvl));
						hi = _mm256_add_epi16(hi, _mm256_mulhrs_epi16(
							_mm256_load_si256(reinterpret_cast<const __m256i*>(g_tprj_w[bucket] + j + half)), tvh));
					}
				}
				lo = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(lo, c254), zero), 7);
				hi = _mm256_min_epi16(hi, c254);
				return _mm256_mulhi_epi16(lo, hi);
			};
			for (int j = 0; j < half; j += 32) {
				const __m256i p0 = prod16(j), p1 = prod16(j + 16);
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j), _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p1), 0xD8));
			}
#else
			for (int j = 0; j < half; j += 16) {
				__m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
				if (j < THREAT_LO_DIM)
					lo = _mm256_add_epi16(lo, thr_scaled(b + j));
				__m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j + half));
				if (j < THREAT_HI_DIM)
					hi = _mm256_add_epi16(hi, thr_scaled(b + THREAT_LO_DIM + j));
				if constexpr (P2_HAS_TPRJ) {
					if (use_proj) {
						// j + half == j (mod K) なので lo/hi は同じ thr スライスを使う。
						// 🚨tdup は accumulator を 2 周ぶん複製したもの。 どのずらし量でも
						//   16 本の連続 load が回り込まずに取れる。
						const int cb = j % THREAT_ACC_DIM;
						const __m256i tvl = _mm256_loadu_si256(
							reinterpret_cast<const __m256i*>(tdup + cb + g_tprj_blk[j / 16]));
						const __m256i tvh = _mm256_loadu_si256(
							reinterpret_cast<const __m256i*>(tdup + cb + g_tprj_blk[(j + half) / 16]));
						lo = _mm256_add_epi16(lo, _mm256_mulhrs_epi16(
							_mm256_load_si256(reinterpret_cast<const __m256i*>(g_tprj_w[bucket] + j)), tvl));
						hi = _mm256_add_epi16(hi, _mm256_mulhrs_epi16(
							_mm256_load_si256(reinterpret_cast<const __m256i*>(g_tprj_w[bucket] + j + half)), tvh));
					}
				}
				lo = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(lo, c254), zero), 7);
				hi = _mm256_max_epi16(_mm256_min_epi16(hi, c254), zero);
				__m256i prod = _mm256_mulhi_epi16(lo, hi);
				__m256i packed = _mm256_packus_epi16(prod, prod);
				_mm_storel_epi64(reinterpret_cast<__m128i*>(out + j), _mm256_castsi256_si128(packed));
				_mm_storel_epi64(reinterpret_cast<__m128i*>(out + j + 8), _mm256_extracti128_si256(packed, 1));
			}
#endif
#else
			const int qs_scale = (g_in_qsearch ? g_qs_threat_scale : 256);
			auto thr_s = [&](int w) { return qs_scale == 256 ? w : (int)(((long)w * qs_scale) >> 8); };
			// 案1 ブロック射影 (2026-09-30)。 AVX2 の _mm256_mulhrs_epi16 と同じ丸め
			//   ((a*b + 2^14) >> 15) にして bit-exact を保つ。
			const bool use_proj = P2_HAS_TPRJ && g_tprj_loaded;
			for (int j = 0; j < half; ++j) {
				std::int16_t av = a[j] + (j < THREAT_LO_DIM ? (std::int16_t)thr_s(b[j]) : (std::int16_t)0);
				std::int16_t bv = a[j + half] + (j < THREAT_HI_DIM ? (std::int16_t)thr_s(b[THREAT_LO_DIM + j]) : (std::int16_t)0);
				if constexpr (P2_HAS_TPRJ) {
					if (use_proj) {
						const int tvl = thr_s(b[(j + g_tprj_blk[j / 16]) % THREAT_ACC_DIM]);
						const int tvh = thr_s(b[(j + g_tprj_blk[(j + half) / 16]) % THREAT_ACC_DIM]);
						av = (std::int16_t)(av + (std::int16_t)(((int) g_tprj_w[bucket][j] * tvl + 16384) >> 15));
						bv = (std::int16_t)(bv + (std::int16_t)(((int) g_tprj_w[bucket][j + half] * tvh + 16384) >> 15));
					}
				}
				int x = clampi(av, 0, 254);
				int y = clampi(bv, 0, 254);
				out[j] = (std::uint8_t)((x * y) >> 9);
			}
#endif
		}
#if defined(P2_TR_STORE32_VERIFY)
		tr_verify(a_us, b_us, a_them, b_them);
#endif
	};
	auto transform_pair = [&](const std::int16_t* a_us, const std::int16_t* a_them) {
		for (int p = 0; p < 2; ++p) {
			const std::int16_t* a = (p == 0) ? a_us : a_them;
			std::uint8_t* out = tr + p * half;
#if defined(USE_AVX2)
			// FT accumulator の clamp を 0..254 にし、 pairwise を (lo*hi)>>9 で行う。
			// mullo だと 254*254=64516 が int16 を溢れるため、 NAGISA V3.1 / sfnn-1536.h と同じく
			// mulhi(lo << 7, hi) = (lo*128*hi) >> 16 = (lo*hi) >> 9 で上位側に逃がす。
			// lo<<7 は最大 254*128 = 32512 で int16 に収まる。 出力レンジは 126 で従来と同一。
			const __m256i c254 = _mm256_set1_epi16(254);
			const __m256i zero = _mm256_setzero_si256();
#if defined(P2_TR_STORE32)
			// P2_TR_STORE32: transform_pair_sum と同じ形 (32 列ごとに 32 byte の store 1 回、 hi 側の max(0) を省く)。 値は従来と同じ。
			// SAFETY: a は 2 * half 要素、 out は half byte 以上。 不変条件: half % 32 == 0 (transform_pair_sum の static_assert)。
			for (int j = 0; j < half; j += 32) {
				const __m256i l0 = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j)), c254), zero), 7);
				const __m256i l1 = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j + 16)), c254), zero), 7);
				const __m256i p0 = _mm256_mulhi_epi16(l0, _mm256_min_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j + half)), c254));
				const __m256i p1 = _mm256_mulhi_epi16(l1, _mm256_min_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j + 16 + half)), c254));
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j), _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p1), 0xD8));
			}
#else
			for (int j = 0; j < half; j += 16) {
				__m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
				__m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j + half));
				lo = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(lo, c254), zero), 7);
				hi = _mm256_max_epi16(_mm256_min_epi16(hi, c254), zero);
				__m256i prod = _mm256_mulhi_epi16(lo, hi);
				__m256i packed = _mm256_packus_epi16(prod, prod); // [b0..7,b0..7,b8..15,b8..15]
				_mm_storel_epi64(reinterpret_cast<__m128i*>(out + j), _mm256_castsi256_si128(packed));
				_mm_storel_epi64(reinterpret_cast<__m128i*>(out + j + 8), _mm256_extracti128_si256(packed, 1));
			}
#endif
#else
			for (int j = 0; j < half; ++j) {
				int x = clampi(a[j], 0, 254), y = clampi(a[j + half], 0, 254);
				out[j] = (std::uint8_t)((x * y) >> 9);
			}
#endif
		}
#if defined(P2_TR_STORE32_VERIFY)
		tr_verify(a_us, nullptr, a_them, nullptr);
#endif
	};
#endif  // P2_FT_NEWTR

#if defined(P2_FABLE_PROBE)
	const std::uint64_t fp_t2 = __rdtsc();
#endif
	std::int32_t main_raw;
if (want_threat) {
		// lazy-threat gate(éå°E¼E¼ç¹ï½»é«±ã»bit-exactç¸²ãSI: P2LazyThreatMargin > 0 ç¸ºE§è­çæE:
		//   base(HalfKA acc + PSQT)ç¸ºE®ç¸ºE¿ç¸ºE§èï½® eval ç¸ºå±Ââµç¾éï½¢é¯ã»[alpha-M, beta+M] ç¸ºE®èãEfar node)
		//   ç¸ºEªç¹§ã»threat èEEE¸ã»ensure_threat + merge + èã»forward)ç¹§åE½¸E¸ç¸ºæâEéµâ«æç¸ºå¶Eç¸²ã»
		//   é¯çãE(near node)ç¸ºE¯è æºæé¨¾å£¹EEthreat éï½¼ç¸ºE¿ç¸ºE§èã»forward(base forward ç¸ºå¾¡E½åãEç¸ºE« 1 èæ§­Â°ç¸ºä¹ï½Eç¸²ã»
		//   ç¬ãEux è´Eµéï½¨è­ãEwant_aux)ç¸ºE¯ lazy gate ç¹§åE½½E¿ç¹§ä¸âEç¸ºã»aux ç¸ºã»far node ç¸ºE§ç¹§ãE½¿ã»E¦ââEç¸ºæºï½Eç¸²ã»
#if defined(P2_PROBE_BUILD)
		++g_pc_wt;
#endif
		bool skip_threat = false;
		if (g_lazy_threat_margin > 0 && !dump && !want_aux) {
			transform_pair(st->p2_acc[stm], st->p2_acc[nstm]);
			main_raw = propagate_bucket(g_net->buckets[bucket], tr);
			int base_cp = (eff_fv > 0) ? (int)((main_raw + psqt_val) / eff_fv) : 0;
			skip_threat = (base_cp - g_lazy_threat_margin >= p2_win_b)
			           || (base_cp + g_lazy_threat_margin <= p2_win_a);
			P2_TSTAT(skip_threat ? ++g_threat_stats.lazy_far : ++g_threat_stats.lazy_near);
		}
#if defined(P2_PROBE_BUILD)
		if (skip_threat) ++g_pc_skip;
#endif
		if (!skip_threat) {
			// threat èEEE¸å¼±EEincremental accumulator(p2_threat_acc)ç¸ºE§éï½¨è«¢ä¸ï¼ ç¸²ã»
			// forward èï½¥èãE= p2_acc[persp] + p2_threat_acc[persp] ç¹§ã»transform ç¸ºE«éæ¦ç²ç¸ºå±â»è·åãE
			// (è³E­é«¢èã°ç¹ãEãµç¹§E¡ç¸ºE¸ç¸ºE® merge storeç«æ± eload ç¹§è²çè²å¸ÂE£it-exact)ç¸²ã»
			{ R2T_SCOPE(R_MTHR); ensure_threat_both(pos, bucket); }   // 2026-09-15 (A2): bucket は上で計算済み
#if defined(P2_FABLE_PROBE)
			// 2026-11-10: cyc_fwd (6,334 cycle/eval = eval の 63%) を
			//   **列挙** と **transform + propagate** に割る。 後者は 2,134 cycle/eval =
			//   node 時間の約 10% と見積もられるが、 一度も測られていない。
			//   🚨P2_FABLE_PROBE の中だけなので本番ビルドには 1 命令も入らない。
			FableProbe::cyc_enum += __rdtsc() - fp_t2;
			FableProbe::n_enum += 1;
#endif
			if (want_aux) {
				// extra = threat acc + aux(fresh)ç¸²ãE½¸E¡é ã»Eè·åãEç¸ºå±âEextra ç¹§ã»transform ç¸ºE«è±¬ââEç¸²ã»
				alignas(32) std::int16_t ext_stm[FT_MAX];
				alignas(32) std::int16_t ext_nstm[FT_MAX];
				compute_aux_both(pos, stm, nstm, ext_stm, ext_nstm,
				                 st->p2_threat_acc[stm], st->p2_threat_acc[nstm]);
				transform_pair_sum(st->p2_acc[stm], ext_stm, st->p2_acc[nstm], ext_nstm);
			} else {
#if defined(P2_FABLE_PROBE)
				const std::uint64_t fp_ta = __rdtsc();
#endif
#if defined(P2_KH)
				// KH (2026-11-12): threat acc + Σ KH 行を一時 buffer に作る (st->p2_threat_acc は書かない = LZ の差分を壊さない)。
				//   TL1 (propagate_bucket へ渡す threat acc) には足さない (学習器の TL1 は thr_stm だけを見る)。
				if (kh::g_loaded) {
					alignas(32) std::int16_t kh_s[THREAT_ACC_DIM];
					alignas(32) std::int16_t kh_n[THREAT_ACC_DIM];
#if defined(P2_KH_PROBE)
					const std::uint64_t kp_a = __rdtsc();
#endif
#if defined(P2_KH_FAST)
					// 2026-11-13 (P2_KH_FAST): 浮き駒が無ければ buffer を作らず threat acc をそのまま渡す (値は従来と同じ)。
					const std::int16_t* kh_ps = threat_acc_ptr(st, stm);
					const std::int16_t* kh_pn = threat_acc_ptr(st, nstm);
					if (kh::apply_fast(pos, stm, kh_ps, kh_pn, kh_s, kh_n)) { kh_ps = kh_s; kh_pn = kh_n; }
#if defined(P2_KH_PROBE)
					else ++kh::g_probe.n_zero;
					kh::g_probe.cyc_all += __rdtsc() - kp_a; ++kh::g_probe.n;
#endif
					transform_pair_sum(st->p2_acc[stm], kh_ps, st->p2_acc[nstm], kh_pn);
#else
					kh::apply(pos, stm, threat_acc_ptr(st, stm), threat_acc_ptr(st, nstm), kh_s, kh_n);
#if defined(P2_KH_PROBE)
					kh::g_probe.cyc_all += __rdtsc() - kp_a; ++kh::g_probe.n;
#endif
					transform_pair_sum(st->p2_acc[stm], kh_s, st->p2_acc[nstm], kh_n);
#endif
				} else
#endif
				transform_pair_sum(st->p2_acc[stm],  threat_acc_ptr(st, stm),

				                   st->p2_acc[nstm], threat_acc_ptr(st, nstm));
#if defined(P2_FABLE_PROBE)
				FableProbe::cyc_tr += __rdtsc() - fp_ta;
				FableProbe::n_tr += 1;
#endif
			}
			// 2026-10-07 (W0-b) TL1 の配線: **ここだけ** threat acc が valid なので TL1 を渡せる。
			//   🚨これが無いと g_tl1_w は tl1chk (dump) 以外から読まれず、 TL1 net を
			//     「TL1 項抜き」で評価してしまう (学習器の forward には TL1 項がある =
			//     学習と推論が食い違う)。 2026-10-07 に陽性対照で発見 (RESULTS/PLAN 参照)。
			//   🚨stm / nstm の順で渡す。 学習器は thr_stm.concat(thr_ntm) = [0..K) が手番側。
			//   🚨bucket は LayerStack の bucket (0..NUM_BUCKETS-1)。 threat の群 (4) ではない。
			//   lazy の base 試算 (上) と threat 無し / aux だけの経路 (下) は acc が無効なので渡さない。
#if defined(P2_FABLE_PROBE)
			const std::uint64_t fp_tc = __rdtsc();
#endif
			main_raw = propagate_bucket(g_net->buckets[bucket], tr, bucket,
			                            threat_acc_ptr(st, stm), threat_acc_ptr(st, nstm));
#if defined(P2_FABLE_PROBE)
			FableProbe::cyc_prop += __rdtsc() - fp_tc;
			FableProbe::n_prop += 1;
#endif
		}
	} else if (want_aux) {
		// threat è¾ï½¡èï½¹ç¹ï½»aux è­çæE base(HalfKA acc)+ aux(fresh)ç¹§ã»transform ç¸ºE«éæ¦ç²ç¸²ã»
		alignas(32) std::int16_t ext_stm[FT_MAX];
		alignas(32) std::int16_t ext_nstm[FT_MAX];
		compute_aux_both(pos, stm, nstm, ext_stm, ext_nstm, nullptr, nullptr);
		transform_pair_sum(st->p2_acc[stm], ext_stm, st->p2_acc[nstm], ext_nstm);
		main_raw = propagate_bucket(g_net->buckets[bucket], tr);
	} else {
		transform_pair(st->p2_acc[stm], st->p2_acc[nstm]);
		main_raw = propagate_bucket(g_net->buckets[bucket], tr);
	}
#if defined(P2_FABLE_PROBE)
	const std::uint64_t fp_t3 = __rdtsc();
	for (int ic = 0; ic < FT_DIM / 4; ++ic) { std::uint32_t v; std::memcpy(&v, tr + 4 * ic, 4); FableProbe::zero_chunks += (v == 0); FableProbe::zero_bytes += (tr[4*ic]==0) + (tr[4*ic+1]==0) + (tr[4*ic+2]==0) + (tr[4*ic+3]==0); }
	FableProbe::chunks += FT_DIM / 4;
#endif
	std::int32_t pairshape_val = 0, dual_psqt_val = 0, dual_cross_val = 0;
	  std::int32_t dual3_val = 0, cross27_val = 0;
	// ★NPS(A5): evaluate 1 回ごとにガード変数を 3 回読んでいた。 既定に固定 = bit-exact
#if defined(P2_DIAG)
	static const bool baseonly = std::getenv("P2_BASEONLY") != nullptr;
	static const bool nodual = std::getenv("P2_NODUAL") != nullptr;
	static const bool nopair = std::getenv("P2_NOPAIR") != nullptr;
#else
	constexpr bool baseonly = false, nodual = false, nopair = false;
#endif
	// head éªè²E®ã» è­ï½¢è³å£¹ã»è­ï½§ç¹è¼Îèå´ï½¨è²E®ã»C++ ç¸ºE§ç¸ºE¯è¾E®è»ã»Eç¹§ä¼Eæºï¼Eç¸²EE_DIFF=1 ç¸ºE§è¾E®è»ã»æ²¿(è³æãEè³Âé¾E´ç¹ï½»è®æE½¨E¼è²å»â¡ç¸ºå¾¡E½æ±Âã»ç¸²ã»
	// lazy head é§ç©ï½¾E¡ç¹ï½¢ç¹ï½¼ç¹ãEUSI: P2LazyMode)ç¸²E¥ump(evalç¹§E³ç¹æ§­Î¦ç¹ãEè­ãã»è¶E¸ç¸ºE« full(è´E³èEEç¸²ã»
	bool do_heads = !baseonly;
	if (do_heads && !dump) {
		if (g_lazy_mode == 1) {                 // (A) lazy-head (p2_lazy_head �͌���ǂ������ true �ɂ���Ȃ��̂Ŏ����㖳��)
			if (p2_lazy_head) do_heads = false;
		} else if (g_lazy_mode == 2) {          // (B) margin lazy: é¯çï½¤æ¶âç¹§è¥æé¡E¥
			int base_cp = (eff_fv > 0) ? (main_raw + psqt_val) / eff_fv : 0;
			if (base_cp - g_lazy_margin >= p2_win_b || base_cp + g_lazy_margin <= p2_win_a) do_heads = false;
		}
	}
	if (do_heads) {
		// dual/pairshape ç¸ºE¯ 8bucket èï½ºè³ã»NUM_DUAL_PSQT_BUCKETS=PAIRSHAPE_NUM_BUCKETS=8)ç¸²ã»
		// LayerStack ç¸ºE® 0..15(P8H2)ç¸ºE§ç¸ºE¯ç¸ºEªç¸ºä¸ï½­E¦éè²ãE compute_dual_bucket é¶E¸è èã» 0..7 ç¸ºE§è è¼ï¿¥ç¸²ã»
		// 2026-09-15 (A2): pairshape / dual 系 head が無い net では bucket を計算しない (値は使われない)。
		const bool need_dp = g_net->has_pairshape || g_net->has_dual_psqt || g_net->has_dual_cross;
		const int dp_bucket = need_dp ? get_dual_pair_bucket(pos) : 0;
		if (g_net->has_pairshape && !nopair)
			pairshape_val = (pairshape_accum(pos, stm, dp_bucket) - pairshape_accum(pos, nstm, dp_bucket)) * 8; // PAIRSHAPE_OUTPUT_SCALE
		if (!nodual)
			dual_eval(pos, dp_bucket, dual_psqt_val, dual_cross_val);
	}

  // 2026-09-15 (A2): threat 系 custom head (factor / direct / dual3 / cross27) が 1 つも無い net では
  //   bucket を計算しない (従来は無条件に get_dual_pair_bucket を呼んでいた。 値はどの head にも渡らない)。
  const bool need_tfb = g_net->has_threat_factor || g_net->has_threat_direct || g_net->has_dual3 || g_net->has_cross27;
  const int threat_factor_bucket = need_tfb ? get_dual_pair_bucket(pos) : 0; // Threat custom heads store exactly 8 buckets.
  const std::int32_t threat_factor_val = want_threat_factor ? threat_factor_value(pos, stm, nstm, threat_factor_bucket) : 0;
  const std::int32_t threat_direct_raw = (g_net->has_threat_direct && !nodirect_env())
      ? (Threat::THREAT_PROFILE_ID == Direct15::PROFILE_ID
          ? threat_direct15_value(pos, stm, nstm, threat_factor_bucket)
          : threat_direct_value(pos, stm, nstm, threat_factor_bucket))
      : 0;
  const std::int32_t threat_direct_val = Threat::THREAT_PROFILE_ID == Direct15::PROFILE_ID
      ? cap_profile15_threat_direct(threat_direct_raw, eff_fv)
      : threat_direct_raw;
  dual3_val = g_net->has_dual3 ? dual3_value(pos, threat_factor_bucket, eff_fv) : 0;
	  cross27_val = g_net->has_cross27 ? cross27_value(pos, threat_factor_bucket, eff_fv) : 0;
  if (dump && g_net->has_dual3) dual3_debug_dump(pos, threat_factor_bucket, eff_fv);
	  if (dump && g_net->has_cross27) cross27_debug_dump(pos, threat_factor_bucket, eff_fv);
  std::int32_t raw_total = main_raw + psqt_val + dual_psqt_val + dual_cross_val
	                         + pairshape_val + threat_factor_val + threat_direct_val + dual3_val + cross27_val;
#if defined(P2_BUCKET_OFFSET)
	// SIDEOFF (11-21): 表は root 色で選んだ g_bucket_off_act (select_bucket_off)。 ここでは root 色を見ない。
	if (g_bucket_off_any && bucket >= 0 && bucket < 8) raw_total += g_bucket_off_act[bucket] * eff_fv;
#endif
#if defined(P2_OPN)
	// OPN の混合 (p2nnue_opn.h 冒頭の式)。 lam = 0 かつ dump でなければ何もしない (= 従来と同じ値)。
	const std::int32_t opn_main_total = raw_total;
	std::int32_t opn_raw = 0, opn_cal = 0;
	if (opn_lam > 0 || (dump && opn::g_loaded)) {
		opn_raw = opn::eval_raw(pos, opn_pb);
		opn_cal = opn::calibrate(opn_raw, opn_pb, eff_fv);
		if (opn_lam > 0) raw_total = opn::blend(raw_total, style_raw, opn_cal, opn_lam);
	}
#if defined(P2_OPN_PROBE)
	if (dump) ++opn::probe::ev_dump; else if (opn_lam > 0) ++opn::probe::ev_both; else if (opn::g_active) ++opn::probe::ev_main;
#endif
#endif
	int total_cp = (eff_fv > 0) ? raw_total / eff_fv : 0;
#if defined(P2_FABLE_PROBE)
	{ const std::uint64_t fp_t4 = __rdtsc(); FableProbe::cyc_acc += fp_t1 - fp_t0; FableProbe::cyc_fwd += fp_t3 - fp_t2; FableProbe::cyc_heads += fp_t4 - fp_t3; FableProbe::cyc_eval += fp_t4 - fp_t0;
	  FableProbe::acc_hist(fp_t1 - fp_t0); }
#endif


	if (dump) {
		sync_cout << "info string P2NNUE bucket=" << bucket
		          << " main_raw=" << main_raw << " psqt=" << psqt_val
		          << " dual_psqt=" << dual_psqt_val << " dual_cross=" << dual_cross_val
		          << " pairshape=" << pairshape_val
                    << " threat_factor=" << threat_factor_val
                    << " threat_direct_raw=" << threat_direct_raw
                     << " threat_direct=" << threat_direct_val
                     << " threat_direct_cap_cp=" << (Threat::THREAT_PROFILE_ID == Direct15::PROFILE_ID ? P2_THREAT_DIRECT_CAP_CP : 0)
                     << " threat_direct_cap_raw=" << ((Threat::THREAT_PROFILE_ID == Direct15::PROFILE_ID && P2_THREAT_DIRECT_CAP_CP != 0) ? profile15_direct_cap_raw(eff_fv) : (std::int64_t)0)
                    << " dual3=" << dual3_val
	                    << " cross27=" << cross27_val
                    << " raw_total=" << raw_total << " cp=" << total_cp
		          << " fv_scale=" << g_fv_scale << " fv_open=" << g_fv_scale_opening
		          << " prog_sum=" << prog_sum << " eff_fv=" << eff_fv << sync_endl;
#if defined(P2_FC1_REF)
		fc1ref::print(bucket, tr, main_raw);
#endif
#if defined(P2_KH)
		if (want_threat) kh::dump(pos, threat_acc_ptr(st, stm), threat_acc_ptr(st, nstm));
#endif
#if defined(P2_OPN)
		if (opn::g_loaded) {
			// P1-P3 の照合用。 main_total = 本体の raw_total (混ぜる前、 様式項込み) / blend_total = 混ぜた後 (上の行の raw_total と同じ)。
			char pbuf[48], cbuf[64];
			std::snprintf(pbuf, sizeof(pbuf), "%.9g", (double) prog_sum);
			// λ の定数 (option から float で作った値)。 python の P3 はこの値で式を再現する。
			std::snprintf(cbuf, sizeof(cbuf), " phi=%.9g inv=%.9g", (double) opn::g_phi, (double) opn::g_inv);
			sync_cout << "info string P2OPN pb=" << opn_pb << " prog_sum=" << pbuf << " lam=" << opn_lam
			          << " opn_raw=" << opn_raw << " opn_cal=" << opn_cal << " main_total=" << opn_main_total
			          << " style=" << style_raw << " blend_total=" << raw_total << " cp=" << total_cp
			          << " eff_fv=" << eff_fv << cbuf << sync_endl;
		}
#endif
	}
	return total_cp;
}

// ---------------------------------------------------------------------------
// é±E­ç¸ºE¿éï½¼ç¸ºE¿
// ---------------------------------------------------------------------------
// Golden Forward 検査: TL1 の算術をエンジン実装ごと照合できる形で吐く。 2026-10-01。
//   出力 (1 局面):
//     tl1chk bucket <b>
//     tl1chk thr 0 <K 個>        (stm 視点の生 i16)
//     tl1chk thr 1 <K 個>        (ntm 視点)
//     tl1chk l1 with <L1_OUT 個>
//     tl1chk l1 without <L1_OUT 個>
//   外で  with - without == (Σ thr*w) >> 5  を厳密に照合する。
void dump_tl1_check(const Position& pos) {
	if (g_net == nullptr || !g_net->has_threat) {
		sync_cout << "info string tl1chk: threat net ではない" << sync_endl;
		return;
	}
	(void) evaluate(pos);   // accumulator を作らせる
	const StateInfo* st = pos.state();
	const Color stm = pos.side_to_move();
	const Color nstm = ~stm;
	const int bucket = get_layer_stack_index(pos);
	const std::int16_t* ts = threat_acc_ptr(st, stm);
	const std::int16_t* tn = threat_acc_ptr(st, nstm);
	if (ts == nullptr || tn == nullptr) {
		sync_cout << "info string tl1chk: threat acc が無い" << sync_endl;
		return;
	}
	// transform は evaluate と同じ経路を使う必要があるが、 ここでは l1 の差だけを見たいので
	// 同じ transformed を 2 回使う。 tr は evaluate 内のバッファなので作り直す。
	// 🚨2026-11-07 (E6): 持駒 dense 有効時は `propagate_bucket` が **FC0_IN = 1056 B** 読むので、
	//   `2 * FT_HALF` (= 1024 B) のままだと**バッファを 32 B 読み越える**し、
	//   持駒の列が未初期化のごみになって検査経路だけ本番と違う値を出す。
	alignas(64) static thread_local TrT trbuf[FC0_IN];
	{
		const int half = FT_HALF;
#if defined(P2_HAND_DENSE)
		// 本番経路 (E2) と**同じ表・同じ順・同じ目盛り**で埋める。 2 箇所に書くのは
		// 本来避けたいが、 検査経路は本番の `tr` を使わない作りなので複製せざるを得ない。
		// 🚨順は学習器の 歩 香 桂 銀 金 角 飛 (YO の PieceType 順ではない)。
		{
			static constexpr PieceType HC_PT[7] = { PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK };
			static constexpr int       HC_K [7] = { 7,    31,    31,     31,     31,   63,     63   };
			const Hand h_us   = pos.hand_of(stm);
			const Hand h_them = pos.hand_of(nstm);
			for (int i = 0; i < HC_PAD; ++i)
				trbuf[FT_DIM + i] = 0;
			for (int i = 0; i < 7; ++i) {
				trbuf[FT_DIM + i]     = (std::uint8_t) (HC_K[i] * hand_count(h_us,   HC_PT[i]));
				trbuf[FT_DIM + 7 + i] = (std::uint8_t) (HC_K[i] * hand_count(h_them, HC_PT[i]));
			}
		}
#endif
		const int qs = (g_in_qsearch ? g_qs_threat_scale : 256);
		for (int p = 0; p < 2; ++p) {
			const FtAcc* a = st->p2_acc[p == 0 ? stm : nstm];
			const std::int16_t* b = (p == 0) ? ts : tn;
			TrT* out = trbuf + p * half;
			for (int j = 0; j < half; ++j) {
				auto sc = [&](int v) { return qs == 256 ? v : (int)(((long) v * qs) >> 8); };
				int av = ft_add_thr(a[j], j < THREAT_LO_DIM ? (std::int16_t) sc(b[j]) : 0);
				int bv = ft_add_thr(a[j + half], j < THREAT_HI_DIM ? (std::int16_t) sc(b[THREAT_LO_DIM + j]) : 0);
				if constexpr (P2_HAS_TPRJ) {
					if (g_tprj_loaded) {
						const int tvl = sc(b[(j + g_tprj_blk[j / 16]) % THREAT_ACC_DIM]);
						const int tvh = sc(b[(j + g_tprj_blk[(j + half) / 16]) % THREAT_ACC_DIM]);
						av = (std::int16_t)(av + (std::int16_t)(((int) g_tprj_w[bucket][j] * tvl + 16384) >> 15));
						bv = (std::int16_t)(bv + (std::int16_t)(((int) g_tprj_w[bucket][j + half] * tvh + 16384) >> 15));
					}
				}
				out[j] = ft_pair_out(clampi(av, 0, FT_CLAMP), clampi(bv, 0, FT_CLAMP));
			}
		}
	}
	std::int32_t l1_with[L1OUT_MAX] = {}, l1_without[L1OUT_MAX] = {};
	// (4) TL2 の Golden Forward 用: clamp の **前**の L2 入力 (i32) を取り出す。
	//   clamp 後の u8 では飽和した要素の差が消えて照合できないため、 pre-clamp を出す。
	std::int32_t l2pre_on[L2PAD_MAX] = {}, l2pre_off[L2PAD_MAX] = {};
	// 注入点の候補 (B) = fc1 出力の `>>6` / clamp の前。 飽和率の比較のために両方出す。
	std::int32_t l2out_on[L3_MAX] = {}, l2out_off[L3_MAX] = {};
#if defined(P2_TG)
	// (5) TG の検査 (tgchk): [0,M) = g (clamp 後) / [M,2M) = sqr (ゲート前) /
	//     [2M,3M) = pre_sq (clamp 前)。
	std::int32_t tg_dbg[3 * MAIN_DIM] = {};
#endif
	(void) propagate_bucket(g_net->buckets[bucket], trbuf, bucket, ts, tn, l1_with, l2pre_on, l2out_on
#if defined(P2_TG)
	                        , tg_dbg
#endif
	                        );
	(void) propagate_bucket(g_net->buckets[bucket], trbuf, -1, nullptr, nullptr, l1_without,
	                        l2pre_off, l2out_off);
	{
		std::string l = "tl1chk bucket " + std::to_string(bucket);
		sync_cout << l << sync_endl;
	}
	for (int p = 0; p < 2; ++p) {
		const std::int16_t* t = (p == 0) ? ts : tn;
		std::string l = "tl1chk thr " + std::to_string(p);
		for (int c = 0; c < THREAT_ACC_DIM; ++c) { l += " "; l += std::to_string((int) t[c]); }
		sync_cout << l << sync_endl;
	}
	{
		std::string l = "tl1chk l1 with";
		for (int o = 0; o < L1_OUT; ++o) { l += " "; l += std::to_string(l1_with[o]); }
		sync_cout << l << sync_endl;
	}
	{
		std::string l = "tl1chk l1 without";
		for (int o = 0; o < L1_OUT; ++o) { l += " "; l += std::to_string(l1_without[o]); }
		sync_cout << l << sync_endl;
	}
	// 🚨ラベルは "on" / "off" にする。 "with" / "without" は**前方一致してしまう**ので
	//   外の器が with を without で上書きして「差分が全部 0」という偽の結論を出す
	//   (2026-10-07 に TL1 の Golden Forward で実際に踏んだ)。
	for (int p = 0; p < 2; ++p) {
		const std::int32_t* src = (p == 0) ? l2pre_on : l2pre_off;
		std::string l = (p == 0) ? "tl1chk l2pre on" : "tl1chk l2pre off";
		for (int j = 0; j < L2_IN; ++j) { l += " "; l += std::to_string(src[j]); }
		sync_cout << l << sync_endl;
	}
	// 注入点の候補 (B): fc1 出力 L3 本の `>>6` / clamp の前の i32。
	for (int p = 0; p < 2; ++p) {
		const std::int32_t* src = (p == 0) ? l2out_on : l2out_off;
		std::string l = (p == 0) ? "tl1chk l2out on" : "tl1chk l2out off";
		for (int o = 0; o < L3; ++o) { l += " "; l += std::to_string(src[o]); }
		sync_cout << l << sync_endl;
	}
#if defined(P2_TG)
	// (5) TG の Golden Forward 検査。
	//   🚨usi.cpp / evaluate.h を触らない制約のため、 独立した `tgchk` トークンではなく
	//     **`tl1chk <sfen>` の出力に追記**する形にしてある (行の接頭辞は "tgchk ")。
	//   TG section を持たない net では tg = 0 なので g は全部 256 (= 等倍) になる。
	{
		std::string l = "tgchk bucket " + std::to_string(bucket);
		sync_cout << l << sync_endl;
	}
	{
		std::string l = "tgchk g";
		for (int i = 0; i < MAIN_DIM; ++i) { l += " "; l += std::to_string(tg_dbg[i]); }
		sync_cout << l << sync_endl;
	}
	{
		std::string l = "tgchk sqr";
		for (int i = 0; i < MAIN_DIM; ++i) { l += " "; l += std::to_string(tg_dbg[MAIN_DIM + i]); }
		sync_cout << l << sync_endl;
	}
	{
		std::string l = "tgchk pre";
		for (int i = 0; i < MAIN_DIM; ++i) { l += " "; l += std::to_string(tg_dbg[2 * MAIN_DIM + i]); }
		sync_cout << l << sync_endl;
	}
#endif
}

#if defined(P2_FC1_PREC)
// fc1p_file: net が P2_FC1_PREC 形式 (版 0x7AF32F22 + "AffineTransformF1P128[") か。 偽なら旧形式を読み込み時に同じ式へ載せる (下の注記)。
static bool read_fc(Reader& r, Bucket& bk, bool fc1p_file) {
#else
static bool read_fc(Reader& r, Bucket& bk) {
#endif
	std::uint32_t fc_hash = r.read_u32_le(); (void)fc_hash;
	auto rd_layer = [&](std::int32_t* bias, int out, std::vector<std::int8_t>& w, int in) {
		int pin = padded(in);
		for (int o = 0; o < out; ++o) bias[o] = r.read_i32_le();
		w.resize((std::size_t)out * pin);
		for (std::size_t i = 0; i < w.size(); ++i) w[i] = (std::int8_t)r.read_u8();
	};
	// 2026-11-07: 持駒 dense 有効時は fc0 の行が padded(FT_DIM + 14) = **1056 B** になる
	//   (= FC0_IN。 学習器の pad32 と同じ)。 無効時は従来どおり 1024。
	rd_layer(bk.fc0_bias, L1_OUT, bk.fc0_w, FT_DIM + HC_DIMS);
	rd_layer(bk.fc1_bias, L3,     bk.fc1_w, L2_IN);
#if defined(P2_L4)
	// 2026-11-07: L4 block は **L2 (fc1) の後・Output (fc2) の前**に在る (学習器 T4 と同順)。
	//   これで 1 bucket は 18,824 -> 18,824 + 4·L4 + L4·pad32(L3) B になる (L4=64 で 23,176)。
	rd_layer(bk.fc1b_bias, L4, bk.fc1b_w, L3);
	rd_layer(bk.fc2_bias,  1,  bk.fc2_w,  L4);
#else
	rd_layer(bk.fc2_bias, 1,      bk.fc2_w, L3);
#endif
	// scrambled ç¹ï½¬ç¹§E¤ç¹§E¢ç¹§E¦ç¹åE§çï½¯ã»AVX2è³E¦è»ã»affine éï½¨): ws[ic*OUT*4 + o*4 + b] = w_row[o*IN_PAD + ic*4+b]
	auto build_ws = [](const std::vector<std::int8_t>& w, WsVec& ws, int in_pad, int nout) {
		ws.resize((std::size_t)(in_pad / 4) * nout * 4);
		for (int o = 0; o < nout; ++o)
			for (int i = 0; i < in_pad; ++i)
				ws[(std::size_t)(i / 4) * nout * 4 + (std::size_t)o * 4 + (i % 4)] = w[(std::size_t)o * in_pad + i];
	};
	build_ws(bk.fc0_w, bk.fc0_ws, FC0_IN, FC0_WS_ROWS);
#if defined(P2_FT_OUT16)
	bk.fc0_ws16.assign((std::size_t) FC0_IN * FC0_WS_ROWS, 0);
	for (int o = 0; o < FC0_WS_ROWS; ++o)
		for (int i = 0; i < FC0_IN; ++i)
			bk.fc0_ws16[(std::size_t) (i / 2) * FC0_WS_ROWS * 2 + (std::size_t) o * 2 + (i % 2)] = bk.fc0_w[(std::size_t) o * FC0_IN + i];
	for (int o = 0; o < L1OUT_MAX; ++o) bk.fc0_bias_m[o] = o < L1_OUT ? bk.fc0_bias[o] * (1 << FT_MS) : 0;
#endif
	build_ws(bk.fc1_w, bk.fc1_ws, L2_PAD, L3);
#if defined(P2_FC1_PREC)
	// P2_FC1_PREC: 新形式の net は fc1 重み 刻み 128 (i8) / fc1 bias 127·4·128 目盛り / fc2 bias 127·4·64 目盛り (fc2 重みは刻み 64 のまま)。
	//   旧形式 (fc1 重み 刻み 64 / bias 127·64 / fc2 bias 127·64) は fc1 重み x2 (i16 に広げるので切られない)・fc1 bias x8・fc2 bias x4 で
	//   同じ目盛りに載せる (活性が細かくなるので旧 exe と同じ値にはならない)。 i32 に収まらない bias があれば読み込みを止める。
	{
		const std::int64_t bm1 = fc1p_file ? 1 : 8, bm2 = fc1p_file ? 1 : 4, wm = fc1p_file ? 1 : 2;
		for (int o = 0; o < L3; ++o) {
			const std::int64_t x = (std::int64_t) bk.fc1_bias[o] * bm1;
			if (x < INT32_MIN || x > INT32_MAX) return false;
			bk.fc1_bias[o] = (std::int32_t) x;
		}
		const std::int64_t x2 = (std::int64_t) bk.fc2_bias[0] * bm2;
		if (x2 < INT32_MIN || x2 > INT32_MAX) return false;
		bk.fc2_bias[0] = (std::int32_t) x2;
		bk.fc1_ws16.assign((std::size_t) L2_PAD * L3, 0);
		for (int o = 0; o < L3; ++o)
			for (int i = 0; i < L2_PAD; ++i)
				bk.fc1_ws16[(std::size_t) (i / 2) * L3 * 2 + (std::size_t) o * 2 + (i % 2)] =
				    (std::int16_t) (bk.fc1_w[(std::size_t) o * L2_PAD + i] * wm);
		for (int i = 0; i < L3; ++i) bk.fc2_w16[i] = bk.fc2_w[i];
	}
#endif
#if defined(P2_L4)
	build_ws(bk.fc1b_w, bk.fc1b_ws, ((L3 + 31) / 32) * 32, L4);
#endif
	return r.ok();
}

bool load_network(const std::string& path) {
	// 🚨 ABI 関門 (2026-10-08): position.cpp が見ている StateInfo の bucket 数と
	//   この翻訳単位の NUM_BUCKETS が一致しているか。 シンボル名に bucket 数が入って
	//   いるので、 食い違うビルドは**そもそも link できない**。 ここの値照合は
	//   「参照が最適化で消えないようにする」ためと、 万一通ったときに黙らせないため。
	//   古い .o が混ざると sizeof(StateInfo) が食い違い、 探索中に state を踏み越える
	//   (readyok は通るので気づけない。 2026-10-07 tsumugi_regime_on.exe)。
	if (P2_STATEINFO_ABI_TAG() != NUM_BUCKETS) {
		sync_cout << "info string P2NNUE: STATEINFO ABI MISMATCH position.cpp="
		          << P2_STATEINFO_ABI_TAG() << " p2nnue=" << NUM_BUCKETS
		          << " — obj を全消ししてビルドし直すこと" << sync_endl;
		return false;
	}
	std::ifstream ifs(path, std::ios::binary);
	if (!ifs) { sync_cout << "info string P2NNUE: cannot open " << path << sync_endl; return false; }
	std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(ifs)),
	                              std::istreambuf_iterator<char>());
	Reader r(buf);

	auto net = std::make_unique<Net>();
	init_shape_table();

	std::uint32_t version = r.read_u32_le();
	std::uint32_t nethash = r.read_u32_le(); (void)nethash;
	std::uint32_t arch_len = r.read_u32_le();
	std::string arch = r.read_string(arch_len);
	if (!r.ok()) { sync_cout << "info string P2NNUE: header read fail" << sync_endl; return false; }
	// P2_FT_I32 (2026-11-18): 新形式の目印 "->I32K<k>:" (FT_KS の注記)。 このビルドの k と一致すれば取り除いて下の parse に流す。
	//   目印が無い (旧形式 = 刻み 254) なら、 FT を読んだ後で重み・bias を k 倍する。 i16 ビルドでは目印を取り除かない
	//   = FT 幅が 0 と読まれて "arch dims parse failed" で止まる。
	int ft_file_k = 1;
	{
		const std::string mk = "->I32K";
		const std::size_t mp = arch.find(mk);
		if (mp != std::string::npos) {
			std::size_t q = mp + mk.size();
			int kk = 0;
			while (q < arch.size() && arch[q] >= '0' && arch[q] <= '9') kk = kk * 10 + (arch[q++] - '0');
			if (q >= arch.size() || arch[q] != ':' || kk != FT_K || FT_K == 1) {
				sync_cout << "info string P2NNUE: FT I32 net (k=" << kk << ", version 0x" << std::hex << version << std::dec
				          << ") needs a -DP2_FT_KS build with k=" << kk << " (this build k=" << FT_K << ")" << sync_endl;
				return false;
			}
			arch.erase(mp + 2, q + 1 - (mp + 2));
			ft_file_k = kk;
		}
	}
	// L1Q128 (2026-11-16): fc0 (最も内側の AffineTransform) の token 名 "AffineTransformL1Q128[" が唯一の目印。
	//   旧 exe はこれを "AffineTransform[" として数えられず arch dims parse failed で必ず止まる (取り違えの関門)。
	//   ここで既存の名前に戻してから下の次元の parse に流す (P2_L4 の 4 token の経路もそのまま通る)。
	//   TL1 / TL2 / TG / MIX16 / HAND_DENSE は L1 の i32 を x64 (8128 目盛り) と仮定しているので、 その build では拒否する。
#if defined(P2_FC1_PREC)
	// P2_FC1_PREC (2026-11-19): fc1 (外から 2 番目の AffineTransform) の token 名 "AffineTransformF1P128[" が新形式の目印 (版 0x7AF32F22)。
	//   旧 exe は "AffineTransform[" を 2 つしか数えられず arch dims parse failed で止まる (取り違えの関門)。 ここで既存の名前に戻す。
	bool fc1p_file = false;
	{
		const std::string f1p = "AffineTransformF1P128[";
		const std::size_t fp = arch.find(f1p);
		if (fp != std::string::npos) {
			int before = 0;
			for (std::size_t p = arch.find("AffineTransform["); p != std::string::npos && p < fp; p = arch.find("AffineTransform[", p + 1)) ++before;
			if (arch.find(f1p, fp + 1) != std::string::npos || before != 1 || version != P2_NET_VERSION_FC1P) {
				sync_cout << "info string P2NNUE: FC1PREC marker must be the 2nd AffineTransform (fc1) only, version 0x7af32f22 (got 0x"
				          << std::hex << version << std::dec << ")" << sync_endl;
				return false;
			}
			arch.replace(fp, f1p.size(), "AffineTransform[");
			fc1p_file = true;
		} else if (version == P2_NET_VERSION_FC1P) {
			sync_cout << "info string P2NNUE: version 0x7af32f22 but no FC1PREC marker" << sync_endl;
			return false;
		}
		sync_cout << "info string P2NNUE: FC1PREC build, net " << (fc1p_file ? "FC1PREC (fc1 w x128)" : "old (fc1 x2 / bias x8 / fc2 bias x4 at load)") << sync_endl;
	}
#endif
	int l1_main_shift = 6;
	{
		const std::string q128 = "AffineTransformL1Q128[";
		const std::size_t qp = arch.find(q128);
		if (qp != std::string::npos) {
			if (arch.find(q128, qp + 1) != std::string::npos
			    || arch.find("AffineTransform[", qp + q128.size()) != std::string::npos) {
				sync_cout << "info string P2NNUE: L1Q128 marker must be the innermost AffineTransform (fc0) only" << sync_endl;
				return false;
			}
#if defined(P2_THREAT_L1) || defined(P2_THREAT_L2) || defined(P2_TG) || defined(P2_RESIDUAL_MIX16)     || defined(P2_RESIDUAL_MIXWIDE16) || defined(P2_HAND_DENSE)
			sync_cout << "info string P2NNUE: L1Q128 net is not supported by this build"
			          << " (TL1/TL2/TG/MIX16/HAND_DENSE assume x64 L1)" << sync_endl;
			return false;
#else
			arch.replace(qp, q128.size(), "AffineTransform[");
			l1_main_shift = 7;
#endif
		}
	}
	bool want_psqt = arch.find("PSQT=") != std::string::npos;
	// B6 可動域 (2026-09-29)。 学習器が "MOB=1," を埋める。
	bool want_mob = arch.find("MOB=1") != std::string::npos;
	// 案1 ブロック射影 (2026-09-30)。 学習器が "ThreatProj=<FT>," を埋める。
	// (3) TL1 (2026-10-01)。 学習器が "ThreatL1=<2K>," を埋める。
	bool want_tl1 = arch.find("ThreatL1=") != std::string::npos;
	// (4) TL2 (2026-10-08)。 学習器が "ThreatL2=<2K>," を埋める。
	//   🚨"ThreatL1=" とも "Threat=" とも別の文字列なので前方一致の事故は起きない。
	bool want_tl2 = arch.find("ThreatL2=") != std::string::npos;
	// (5) TG (2026-10-10)。 学習器が "ThreatGate=<2K>," を埋める。
	//   🚨"Threat=" / "ThreatL1=" / "ThreatL2=" / "ThreatGroups=" とは別の文字列なので
	//     前方一致の事故は起きない。
	bool want_tg = arch.find("ThreatGate=") != std::string::npos;
	bool want_tprj = arch.find("ThreatProj=") != std::string::npos;
	bool want_dual = arch.find("DUAL=") != std::string::npos;
	bool want_threat = arch.find("Threat=") != std::string::npos;
  bool want_threat_factor = arch.find("ThreatFactor16=") != std::string::npos;
  bool want_threat_direct = arch.find("ThreatDirect=") != std::string::npos;
	// 学習時 fv_scale marker ("fv_scale=14")。 learner が書く net には必ず付く。
	// Direct15 の cap を学習時と同一の raw 目盛りで適用するために読む。
	// marker を持たない旧 net は 0 のままにし、 旧挙動へフォールバックする。
	{
		std::size_t fvp = arch.find("fv_scale=");
		if (fvp != std::string::npos) {
			std::size_t p = fvp + 9;
			int v = 0;
			while (p < arch.size() && arch[p] >= '0' && arch[p] <= '9') v = v * 10 + (arch[p++] - '0');
			net->train_fv_scale = v;
		}
	}
	// aux èï½¥èãEmarker(arch ç¸ºE« "AUX=600," ç¸ºå¾â ç¹§å¾ãEè­ï½«èE½¾ç¸ºE« P2AUX section ç¹§åæç¸ºE¤)ç¸²ã»
	// 2026-11-07 (E5): **持駒 14 元 dense** の marker。 学習器は `HandCountDense=14,` を出す。
	// 2026-11-07: marker の**値**も読む。 有無だけだと `HandCountDense=14` の net を
	//   `-DP2_HAND_DENSE=76` の exe が受け入れ、 `rd_layer` が 1120 幅で 1056 幅の
	//   ファイルを読んで**黙って全部ずれる** (★net_exe_mismatch_silent)。
	int file_hc_dims = 0;
	{
		const std::size_t hp = arch.find("HandCountDense=");
		if (hp != std::string::npos) {
			std::size_t q = hp + 15;
			while (q < arch.size() && arch[q] >= '0' && arch[q] <= '9')
				file_hc_dims = file_hc_dims * 10 + (arch[q++] - '0');
		}
	}
	bool want_hand_dense = file_hc_dims > 0;
	bool want_aux_file = arch.find(Aux::AUX_ARCH_MARKER) != std::string::npos;
	// auxcheap marker(arch ç¸ºE« "AUXCHEAP=176," ç¸ºå¾â ç¹§å¾ãEè¿ï½©éEEå¿°èE½¾ç¸ºE« P2AUXC section ç¹§åæç¸ºE¤)ç¸²ã»
	// bullet èï½´è³å¤ï½¾E©: crates/bullet_lib/src/game/inputs/shogi_halfka_hm_auxcheap.rs
	//   AUXCHEAP_ARCH_MARKER = "AUXCHEAP=176," / AUXCHEAP_SUB_MAGIC = "P2AUXC\0\0"ç¸²ã»
	// çª¶E» "AUXCHEAP=" ç¸ºE¯ "AUX=" ç¸ºE¨ç¸ºE¯è»E¥ç¹å»ã»ç¹§E¯ç¹ï½³(é±E¤è®æ·ã»ç¸ºå±âç¸ºã»ç¸²ã»1(STATKD status4ç«ãE)ç¸ºE§
	//   DIM_PER_PERSP=176ç¸²ãç«E248(status4 count-linear)net ç¸ºE¯é«±æ¨ï½ºåé¤ = ç¸ºèï¼E¸ºE§è®æ·ã»ç¸ºè¼ï½ç¸ºã»
	//   want_auxcheap=false ç¸ºE®ç¸ºE¾ç¸ºE¾é©¥éªâ©ç¸ºæ¢§E­E»é¡Eµç¸ºè¼ï½ç¹§ä¹âç¹§âÂâµã»éï½ºé§ã»âè»E¥ç¹éªã£ç¹åæ¡¶ç¸ºã»éãE176 net)ç¸²ã»
	bool want_auxcheap = arch.find("AUXCHEAP=176,") != std::string::npos;
	bool want_endgame_tempo = arch.find("EGTMP=1") != std::string::npos;
	// threat FT é©¥éªâ©ç¸ºã»i16 è«æï½­å¥Â°(marker "ThreatI16=1")ç¸²ãâç¸ºä»£Eç¸ºE°è æºæEi8(è æ¢§å©¿è åé¤)ç¸²ã»
	// è±ï½¨: "ThreatI16=" ç¸ºE¯ "Threat=" ç¸ºE¨ç¸ºE¯è»E¥ç¹å»ã»ç¹§E¯ç¹ï½³(é±E¤è®æ·ã»ç¸ºå±âç¸ºã»ç¸²ã»
	bool threat_i16 = arch.find("ThreatI16=1") != std::string::npos;
	// LayerStack é©ï½¸è¬æ§­Îç¹ï½¼ç¹ãEmarkerç¸²EEH2 net ç¸ºE«ç¸ºE¯ "LSMODE=P4H2:<t0>-<t1>-<t2>-<t3>," ç¸ºå¾Âã»
	// P8H2 net ç¸ºE«ç¸ºE¯ "LSMODE=P8H2:<t0>-<t1>-<t2>-<t3>," ç¸ºå¾¡E»å¥E¥(è¬ãEE§å¸æEèï½¤é©æ¦ãEç¸ºE¯è³E¡ç¹ï½¢ç¹ï½¼ç¹çã»é¨¾ã»ç¸²ã»
	// ç¹èÎç¹ãE-DP2_LSMODE_P4H2 / -DP2_LSMODE_P8H2 + é«¢E¾èï½¤é©æ¦ãE)ç¸ºE¨ net ç¸ºE® marker ç¸ºå¾¡E¸Âé¾E´ç¸ºå±âç¸ºã»âE	// LayerStack ç¸ºE®è«¢ä¸è¢ç¸ºå¾âEç¹§å¾ï½Eè­ï½§ progress8 net ç¹§åç è­E½¹è ä¸ãé±E­ç¹§Â / ç¹ï½¢ç¹ï½¼ç¹çåç¹§ä¼E¼E¸ºã»/ é«¢E¾èï½¤é©è¼ï¼Eç¸ºæºï½ç¸²ã»
	// ç¸ºèï¼E¸ºE§è®æ·ã»ç¸ºå±â»è«¡è²åç¸ºå¶Eç¸²ãEç¹ï½¼ç¹å³¨EE«¢E¾èï½¤é©æ¦ãEç¸ºç¢E¼E¸ºå»ã»è»E¥ç¹éªã£ç¹åæ¡¶ç¸ºã»Âã»
	bool has_p4h2   = arch.find("LSMODE=P4H2") != std::string::npos;
	bool has_p8h2   = arch.find("LSMODE=P8H2") != std::string::npos;
	bool has_lsmode = has_p4h2 || has_p8h2;
	{
		// ç¹èÎç¹çE®å£½çEé«¢E¾èï½¤é©æ¦ãE)ç¸ºä¹ï½è­æº·E¾ã»marker ç¹§å ¤Eµã»â©é¶ä¹â»ç¹§ä¹Âã»
		std::string thr = std::to_string(P2_HAND_VALUE_THRESHOLD[0])
		                + "-" + std::to_string(P2_HAND_VALUE_THRESHOLD[1])
		                + "-" + std::to_string(P2_HAND_VALUE_THRESHOLD[2])
		                + "-" + std::to_string(P2_HAND_VALUE_THRESHOLD[3]);
		if (P2_LS_PROGREGIME) {
			// net と build の (V1 閾値, 分割開始 bucket) が一致していないと bucket の意味が
			// 食い違い、 黙って壊れた評価値で動く (★net_exe_mismatch_silent)。
			std::string want_regime = std::string("LSMODE=") + "PROGREGIME:v1"
			                        + std::to_string(REGIME_V1_TH)
			                        + "-pb" + std::to_string(REGIME_MIN_PB);
			if (arch.find(want_regime) == std::string::npos) {
				sync_cout << "info string P2NNUE: LSMODE MISMATCH build expects marker '"
				          << want_regime
				          << "' but file arch does not contain it. "
				          << "Rebuild with matching -DP2_REGIME_V1_TH / -DP2_REGIME_MIN_PB "
				          << "or use the matching net." << sync_endl;
				return false;
			}
		} else if (P2_LS_P8H2) {
			std::string want = "LSMODE=P8H2:" + thr;
			if (arch.find(want) == std::string::npos) {
				sync_cout << "info string P2NNUE: LSMODE MISMATCH build expects marker '" << want
				          << "' but file arch does not contain it (has_p4h2=" << (int)has_p4h2
				          << " has_p8h2=" << (int)has_p8h2
				          << "). ç¹ï½¢ç¹ï½¼ç¹ãEé«¢E¾èï½¤é©æ¦ãEé©è¼ï¼Eor è­ï½§ netç¸²Eebuild with matching thresholds "
				          << "or use the matching P8H2 net." << sync_endl;
				return false;
			}
		} else if (P2_LS_P4H2) {
			std::string want = "LSMODE=P4H2:" + thr;
			if (arch.find(want) == std::string::npos) {
				sync_cout << "info string P2NNUE: LSMODE MISMATCH build expects marker '" << want
				          << "' but file arch does not contain it (has_p4h2=" << (int)has_p4h2
				          << " has_p8h2=" << (int)has_p8h2
				          << "). ç¹ï½¢ç¹ï½¼ç¹ãEé«¢E¾èï½¤é©æ¦ãEé©è¼ï¼Eor è­ï½§ progress8 netç¸²Eebuild with matching thresholds "
				          << "or use the matching net." << sync_endl;
				return false;
			}
		} else if (!P2_LS_PROGREGIME
		           && arch.find("LSMODE=PROGREGIME") != std::string::npos) {
			// 案 B の net を PROGREGIME 無しの build で読もうとした (逆方向のガード)。
			sync_cout << "info string P2NNUE: LSMODE MISMATCH file is PROGREGIME but build is not. "
			          << "Rebuild with -DP2_LSMODE_PROGREGIME." << sync_endl;
			return false;
		} else if (has_lsmode) {
			sync_cout << "info string P2NNUE: LSMODE MISMATCH file is P4H2/P8H2 but build is legacy progress8. "
			          << "Rebuild with -DP2_LSMODE_P4H2 or -DP2_LSMODE_P8H2 (and matching thresholds)." << sync_endl;
			return false;
		}
	}
	// threat èï½¥èãE net ç¸ºã»threat ç¹§å®ï½¦âµE±ãEEç¹§ã»want_threat)è£E´è·å»ã»ç¸ºE¿ build ç¸ºE® threat capability
	// (P2_HAS_THREAT = P2_ARCH_768T)ç¸ºæï½¿ã»E°å»Âã»
	// ç¬ãEuild ç¸ºã»threat-capable ç¸ºE§ net ç¸ºã»threat è¾ï½¡ç¸ºã»= HalfKA-768 + aux net ç¸ºEªç¸ºE©)ç¸ºE®è£E´è·å»ã»
	//   éªE±è³E¹ç¸ºå¶EE threat é¨çï½·E¯ç¸ºE¯ has_threat=false ç¸ºE®ç¸ºE¾ç¸ºE¾ dormant(runtime ç¸ºE¯ g_net->has_threat ç¸ºE§
	//   é¬§ã»èEç¸ºEªç¸ºE®ç¸ºE§ç¸²âéEè³Â exe(-DP2_ARCH_768T -DP2_AUX)ç¸ºã»threat net ç¸ºE¨ aux net ç¸ºE®è³E¡è­E½¹ç¹§å®ï½ªE­ç¹§âï½ç¸²ã»
	if (want_threat && !P2_HAS_THREAT) {
		sync_cout << "info string P2NNUE: THREAT MISMATCH file needs legacy FT threat but build lacks it." << sync_endl;
		return false;
	}
	if (want_threat_factor && !P2_HAS_THREAT_FACTOR) {
		sync_cout << "info string P2NNUE: THREAT FACTOR MISMATCH file needs profile 13 factor branch." << sync_endl;
		return false;
	}
  if (want_threat_direct && !P2_HAS_THREAT_DIRECT) {
          sync_cout << "info string P2NNUE: THREAT DIRECT MISMATCH file needs profile 14 direct branch." << sync_endl;
          return false;
  }
  if ((want_threat ? 1 : 0) + (want_threat_factor ? 1 : 0) + (want_threat_direct ? 1 : 0) > 1) {
		sync_cout << "info string P2NNUE: invalid net has multiple Threat branch markers." << sync_endl;
		return false;
	}
	// aux marker ç¹§åæç¸ºE¤ net ç¸ºE¯ P2_AUX ç¹èÎç¹å³¨ãç¸ºå±Â°é±E­ç¹§ââEç¸ºã»section ç¹§ã»skip ç¸ºE§ç¸ºéªâç¸ºã»ç¸²ã»
	// é¨¾ã»âEmarker è¾ï½¡ç¸ºã»net ç¹§ã»P2_AUX ç¹èÎç¹å³¨ãé±E­ç¹§Âç¸ºE®ç¸ºE¯è¿E¯(has_aux ç¸ºE¯ section ç¸ºæªâç¸ºä»£Eç¸ºE° falseç¸²ã»
	//  P2_AUX_FORCE ç¸ºE§ zero é©¥éªâ©è·åãEç¸ºå±â» NPS éªåï½¸E¬ç¸ºE«è´E¿ç¸ºå»EEç¸²ã»
	// 2026-11-07 (E5): 持駒 dense は**両方向**で弾く。
	//
	// 🚨🚨これが無いと致命的: HC net の fc0 の行は **1056 B** (= pad32(1024+14))、 現行の exe は
	//   1024 B で読むので `rd_layer` の読み位置が **16 行 x 32 B = 512 B ずれ**、 以後の
	//   fc1 / fc2 / 次の bucket が全部ずれる。 それでも readyok は返るので**黙って壊れた評価値**で動く
	//   (★net_exe_mismatch_silent)。 逆向き (HC exe に 1024 幅の net) も同じ理由で弾く。
	if (want_hand_dense && HC_DIMS == 0) {
		sync_cout << "info string P2NNUE: HANDDENSE MISMATCH file has HandCountDense= marker but build lacks "
		          << "-DP2_HAND_DENSE. Rebuild with -DP2_HAND_DENSE." << sync_endl;
		return false;
	}
	if (!want_hand_dense && HC_DIMS > 0) {
		sync_cout << "info string P2NNUE: HANDDENSE MISMATCH build has -DP2_HAND_DENSE=" << HC_DIMS
		          << " but file has no HandCountDense= marker. Use a hand-dense net." << sync_endl;
		return false;
	}
	if (want_hand_dense && file_hc_dims != HC_DIMS) {
		sync_cout << "info string P2NNUE: HANDDENSE MISMATCH file HandCountDense=" << file_hc_dims
		          << " vs build -DP2_HAND_DENSE=" << HC_DIMS
		          << ". The fc0 row stride would differ; rebuild with the matching value." << sync_endl;
		return false;
	}
	if (want_aux_file && !P2_HAS_AUX) {
		sync_cout << "info string P2NNUE: AUX MISMATCH file has AUX section but build lacks -DP2_AUX. "
		          << "Rebuild with -DP2_AUX." << sync_endl;
		return false;
	}
	// auxcheap marker ç¹§åæç¸ºE¤ net ç¹§å¸æ¼èE½¾è ¢æã³ç¹ï½«ç¹å³¨ãé±E­ç¹§Âç¸ºE¨éªéï½·E´è²å»â©é©¥éªâ©ç¸ºæ¢§E­E»é¡Eµç¸ºè¼ï½ç¹§ã»èEEE¸å¼±ãç¹ï½­ç¸ºE®
	// ç¸ºE¾ç¸ºE¾é§ç©ï½¾E¡ç¸ºç¢EE²ç¹§Â)ç¸ºæºï½ç¸²âµéèã»AUX ç¸ºE®è´æï½³è¼âEèï½£ç¸ºã»ã»éï½ºè«¡è²åç¸ºå¶Eç¸²ã»
	// marker è¾ï½¡ç¸ºã»net ç¸ºE¯è æºæç¸ºE©ç¸ºç¿«EEauxcheap é©¥éªâ©ç¸ºE¯ç¹§E¼ç¹ï½­ç¸ºE®ç¸ºE¾ç¸ºE¾ = è æ¢§å©¿è åé¤)ç¸²ã»
	if (want_auxcheap && !P2_HAS_AUXCHEAP) {
		sync_cout << "info string P2NNUE: AUXCHEAP MISMATCH file has P2AUXC section but build lacks -DP2_AUXCHEAP. "
		          << "Rebuild with -DP2_AUXCHEAP (+ -DLONG_EFFECT_LIBRARY)." << sync_endl;
		return false;
	}
	if (want_endgame_tempo && !P2_HAS_ENDGAME_TEMPO) {
		sync_cout << "info string P2NNUE: EGT MISMATCH file has P2EGTMP section but build lacks "
		          << "-DP2_ENDGAME_TEMPO (+ -DLONG_EFFECT_LIBRARY)." << sync_endl;
		return false;
	}

	// ---- archè­ãEE­æãEç¸ºä¹ï½ç¹§E¢ç¹ï½¼ç¹§E­è°ºE¡èãEEé¾Eªèåæè³ã»----
	// èãE "[73305->1536x2]...AffineTransform[1<-32](...AffineTransform[32<-30](...AffineTransform[16<-3072]..."
	//   FT=1536, L3=32(token0.B), L2_IN=30(token1.B), L1_OUT=16(token2.A)
	{
		auto rd_num = [&](std::size_t& p) -> int {
			int v = 0; while (p < arch.size() && arch[p] >= '0' && arch[p] <= '9') v = v * 10 + (arch[p++] - '0'); return v;
		};
		int ft = 0;
		std::size_t ap = arch.find("->");
		if (ap != std::string::npos) { ap += 2; ft = rd_num(ap); }
		auto affine = [&](int idx, int& A, int& B) -> bool {
			std::size_t p = 0; int cnt = 0; const std::string key = "AffineTransform[";
			while ((p = arch.find(key, p)) != std::string::npos) {
				if (cnt == idx) {
					p += key.size(); A = rd_num(p);
					if (arch.compare(p, 2, "<-") == 0) p += 2; else return false;
					B = rd_num(p); return true;
				}
				++cnt; p += key.size();
			}
			return false;
		};
#if defined(P2_L4)
		// 2026-11-07: **L4 入りの net は AffineTransform token が 4 つ**になる。
		//   新 arch: AffineTransform[1<-L4](ClippedReLU[L4](AffineTransform[L4<-L3](
		//            ClippedReLU[L3](AffineTransform[L3<-L2_IN](SqrClippedReLU[L2_IN](
		//            AffineTransform[L1_OUT<-FT*2](InputSlice[...]))))))) 
		//   🚨`network_hash` / `fc_hash` はどちらも読み捨てている (7389 / 7209) ので、
		//     net x exe の食い違いを弾く門は**ここだけ**。 ★net_exe_mismatch_silent。
		//     旧 net (token 3 つ) をこのビルドで読むと affine(3) が false -> parse failed。
		//     L4 入り net を P2_L4 無しのビルドで読むと token1.B = L4 != L2_IN -> ARCH MISMATCH。
		int a0, b0, a1, b1, a2, b2, a3, b3;
		int f_ft = 0, f_l4 = 0, f_l3 = 0, f_l2in = 0, f_l1 = 0;
		if (ft > 0 && affine(0, a0, b0) && affine(1, a1, b1) && affine(2, a2, b2) && affine(3, a3, b3)) {
			f_ft = ft; f_l4 = b0; f_l3 = b1; f_l2in = b2; f_l1 = a3;
			(void)a0; (void)a1; (void)a2; (void)b3;
		} else {
			sync_cout << "info string P2NNUE: arch dims parse failed"
			          << " (this build is -DP2_L4=" << L4 << " and needs 4 AffineTransform tokens)"
			          << sync_endl;
			return false;
		}
		if (f_l4 != L4) {
			sync_cout << "info string P2NNUE: ARCH MISMATCH file(L4=" << f_l4
			          << ") vs build(L4=" << L4 << ")." << sync_endl;
			return false;
		}
#else
		int a0, b0, a1, b1, a2, b2;
		int f_ft = 0, f_l3 = 0, f_l2in = 0, f_l1 = 0;
		if (ft > 0 && affine(0, a0, b0) && affine(1, a1, b1) && affine(2, a2, b2)) {
			f_ft = ft; f_l3 = b0; f_l2in = b1; f_l1 = a2;
		} else {
			sync_cout << "info string P2NNUE: arch dims parse failed" << sync_endl; return false;
		}
#endif
		// ç¹§E³ç¹ï½³ç¹ä»£ãE¹ï½«è­ãEarch ç¸ºE¨è³Âé¾E´è®æE½¨E¼(è³å ºE¸Âé¾E´ç¸ºEªç¹§å³¨ã³ç¹ï½«ç¹å³¨Eè»ã»å´ç¸ºå»Eç¹§å»â§è«ã»âEç¸²ã»
		if (f_ft != FT_DIM || f_l1 != L1_OUT || f_l3 != L3 || f_l2in != L2_IN) {
			sync_cout << "info string P2NNUE: ARCH MISMATCH file(FT=" << f_ft << " L1=" << f_l1
			          << " L2_IN=" << f_l2in << " L3=" << f_l3 << ") vs build(FT=" << FT_DIM
			          << " L1=" << L1_OUT << " L2_IN=" << L2_IN << " L3=" << L3
			          << "). Rebuild with the matching arch (e.g. P2_ARCH_1024)." << sync_endl;
			return false;
		}
		sync_cout << "info string P2NNUE arch: FT=" << FT_DIM << " L1=" << L1_OUT
		          << " L2_IN=" << L2_IN << " L3=" << L3 << " (PSQT=" << (int)want_psqt
		          << " DUAL=" << (int)want_dual << ")" << sync_endl;
	}

	std::uint32_t ft_hash = r.read_u32_le(); (void)ft_hash;

	// FT (LEB128): 1ç¹æ¶Îç¹ãEãEbias+weighté¨ä»ç²E or 2ç¹æ¶Îç¹ãEãEbias,weight)
	std::vector<std::int16_t> first;
	if (!r.read_compressed_i16(first, 0)) {
		sync_cout << "info string P2NNUE: FT block read fail" << sync_endl; return false;
	}
	const std::size_t wsize = (std::size_t)HALFKA_DIM * FT_DIM;
	// FT_ROW != FT_DIM のときは行ごとに散らして書く (余りは 0)。
	//   FT_ROW == FT_DIM なら 1 回の memcpy と完全に同じ結果。
	auto store_ft = [&](const std::int16_t* src) {
		if (FT_ROW == (std::size_t)FT_DIM) { net->ft_weight.assign(src, wsize); return; }
		net->ft_weight.resize((std::size_t)HALFKA_DIM * FT_ROW);
		std::memset(net->ft_weight.data(), 0, sizeof(std::int16_t) * (std::size_t)HALFKA_DIM * FT_ROW);
		for (std::size_t f = 0; f < (std::size_t)HALFKA_DIM; ++f)
			std::memcpy(net->ft_weight.data() + f * FT_ROW, src + f * (std::size_t)FT_DIM,
			            sizeof(std::int16_t) * (std::size_t)FT_DIM);
	};
	if (first.size() == FT_DIM) {
		net->ft_bias = std::move(first);
		std::vector<std::int16_t> wblk;
		if (!r.read_compressed_i16(wblk, 0) || wblk.size() != wsize) {
			sync_cout << "info string P2NNUE: FT weight block size " << wblk.size() << sync_endl; return false;
		}
		store_ft(wblk.data());
	} else if (first.size() == (std::size_t)FT_DIM + wsize) {
		net->ft_bias.assign(first.begin(), first.begin() + FT_DIM);
		store_ft(first.data() + FT_DIM);
	} else {
		sync_cout << "info string P2NNUE: unexpected FT size " << first.size() << sync_endl; return false;
	}
	if (ft_file_k != FT_K) {
		// P2_FT_KS で旧形式 (刻み 254) の net: 重み・bias を k 倍して k 倍刻みにする (評価値は従来の i16 の exe と同じ)。
		//   i16 に収まらない値があれば読み込みを止める (黙って wrap させない)。
		const int mul = FT_K / ft_file_k;
		auto up = [&](std::int16_t& v) -> bool {
			const int x = (int) v * mul;
			if (x < -32768 || x > 32767) return false;
			v = (std::int16_t) x;
			return true;
		};
		bool okk = true;
		for (auto& v : net->ft_bias) okk = up(v) && okk;
		for (std::size_t i = 0; i < net->ft_weight.size(); ++i) okk = up(net->ft_weight[i]) && okk;
		if (!okk) { sync_cout << "info string P2NNUE: FT x" << mul << " overflows i16" << sync_endl; return false; }
		sync_cout << "info string P2NNUE: FT scaled x" << mul << " (old 254-scale net on FT_KS k=" << FT_K << ")" << sync_endl;
	}
#if defined(P2_FT_I32)
	{
		int mx = 1;
		for (std::size_t i = 0; i < net->ft_weight.size(); ++i) { const int a = net->ft_weight[i] < 0 ? -(int) net->ft_weight[i] : (int) net->ft_weight[i]; if (a > mx) mx = a; }
		g_ft32_group = 32767 / mx < 1 ? 1 : 32767 / mx;
		sync_cout << "info string P2NNUE: FT32 i16 group=" << g_ft32_group << " (max|w|=" << mx << ")" << sync_endl;
	}
#endif
	if (ft_file_k == FT_K && FT_K > 1) {
		sync_cout << "info string P2NNUE: FT I32 net k=" << FT_K << " (version 0x" << std::hex << version << std::dec << ")" << sync_endl;
	}
	if (FT_K > 1 || FT_MS > 0) {
		sync_cout << "info string P2NNUE: FT k=" << FT_K << " out16_m=" << (1 << FT_MS) << " acc="
		          << (sizeof(FtAcc) == 4 ? "i32" : "i16") << (FT_MS > 0 ? " fc0=i16in" : " fc0=u8in") << sync_endl;
	}

	// PSQT (raw i32): bias[NUM_BUCKETS] + weight[HALFKA_DIM*NUM_BUCKETS]
	if (want_psqt) {
		for (int b = 0; b < NUM_BUCKETS; ++b) net->psqt_bias[b] = r.read_i32_le();
		net->psqt_weight.resize((std::size_t)HALFKA_DIM * NUM_BUCKETS);
		for (std::size_t i = 0; i < (std::size_t)HALFKA_DIM * NUM_BUCKETS; ++i)
			net->psqt_weight[i] = r.read_i32_le();
		net->has_psqt = r.ok();
		if (!r.ok()) { sync_cout << "info string P2NNUE: PSQT read fail" << sync_endl; return false; }
	}

	// Threat é©¥éªâ© (raw i8, feature-major [THREAT_DIM * FT_DIM])ç¸²ã»
	// PSQT é¶E´è å¾ãELayerStack èéªÂE®agic è¾ï½¡ç¸ºå±ÂE§ull profile(pid=0)ç¸ºE¯ profile-id prefix è¾ï½¡ç¸ºå±Âã»
	// bullet writer: examples/shogi_layerstack.rs:1983-2011 ç¸ºE¨èE½¾è ¢æÂã»
	g_dt_build = false; // net èéªÎç¹ï½¼ç¹å³¨ãEthreat è¾ï½¡ç¸ºã»net ç¸ºE«è»ã»Eè­ï½¿ç¹§ä¸ï½è£E´è·å»âè¯å¶â´ç¸ºE¦è³Âè­ï½¦éï½½ç¸ºE¨ç¸ºã»
  if (want_threat_factor) {
          int file_profile = 0;
          std::size_t pp = arch.find("ThreatProfile=");
          if (pp != std::string::npos) {
                  std::size_t p = pp + 14;
                  while (p < arch.size() && arch[p] >= '0' && arch[p] <= '9') file_profile = file_profile * 10 + (arch[p++] - '0');
          }
          int dims = 0;
          std::size_t fp = arch.find("ThreatFactor16=");
          if (fp == std::string::npos) { sync_cout << "info string P2NNUE: ThreatFactor16 marker missing" << sync_endl; return false; }
          fp += 15;
          while (fp < arch.size() && arch[fp] >= '0' && arch[fp] <= '9') dims = dims * 10 + (arch[fp++] - '0');
          if (file_profile != 13 || file_profile != Threat::THREAT_PROFILE_ID || dims != THREAT_DIM) {
                  sync_cout << "info string P2NNUE: ThreatFactor16 profile/dimension mismatch" << sync_endl; return false;
          }
  }
  if (want_threat_direct) {
          int file_profile = 0;
          std::size_t pp = arch.find("ThreatProfile=");
          if (pp != std::string::npos) {
                  std::size_t p = pp + 14;
                  while (p < arch.size() && arch[p] >= '0' && arch[p] <= '9') file_profile = file_profile * 10 + (arch[p++] - '0');
          }
          int dims = 0;
          std::size_t dp = arch.find("ThreatDirect=");
          if (dp == std::string::npos) { sync_cout << "info string P2NNUE: ThreatDirect marker missing" << sync_endl; return false; }
          dp += 13;
          while (dp < arch.size() && arch[dp] >= '0' && arch[dp] <= '9') dims = dims * 10 + (arch[dp++] - '0');
          const bool direct15 = Threat::THREAT_PROFILE_ID == Direct15::PROFILE_ID;
          const bool profile_ok = direct15 ? file_profile == Direct15::PROFILE_ID : file_profile == 14;
          // Direct15 は 65,640 (従来) と 131,280 (P5 defended ビットで x2) の両方を受け付ける。
          const bool dims_ok = direct15 ? (dims == Direct15::DIM || dims == 2 * Direct15::DIM)
                                        : (dims > 0 && dims <= DIRECT_THREAT_DIM);
          if (!profile_ok || !dims_ok) {
                  sync_cout << "info string P2NNUE: ThreatDirect profile/dimension mismatch file_profile=" << file_profile << " build_profile=" << Threat::THREAT_PROFILE_ID << " dims=" << dims << " expected_dims=" << (direct15 ? Direct15::DIM : DIRECT_THREAT_DIM) << sync_endl; return false;
          }
  }
if (want_threat) {
		// é«¯E¤èãEprofile marker: pid>0 ç¸ºE® net ç¸ºE¯ arch ç¸ºE« "ThreatProfile=<pid>," ç¹§åæç¸ºE¡ç¸²ã»
		// threat é©¥éªâ© blob ç¸ºE®é¶E´èéªâEu32 LE ç¸ºE® pid prefix ç¸ºå¾¡E»å¥E¥(full=0 ç¸ºE¯ marker/prefix è¾ï½¡ç¸ºã»ç¸²ã»
		// ç¹èÎç¹çã» profile (-DP2_THREAT_PROFILE, è­ï½¢è³ã»0) ç¸ºE¨è³Âé¾E´è ¢ã»E°å»Âã»
		int file_profile = 0;
		{
			std::size_t pp = arch.find("ThreatProfile=");
			if (pp != std::string::npos) {
				std::size_t p = pp + 14;
				while (p < arch.size() && arch[p] >= '0' && arch[p] <= '9')
					file_profile = file_profile * 10 + (arch[p++] - '0');
			}
			if (file_profile != Threat::THREAT_PROFILE_ID) {
				sync_cout << "info string P2NNUE: THREAT PROFILE MISMATCH file(profile=" << file_profile
				          << ") vs build(profile=" << Threat::THREAT_PROFILE_ID
				          << "). Rebuild with -DP2_THREAT_PROFILE=" << file_profile
				          << " (0 = full, define éµâ«æè¿E¯) or use the matching net." << sync_endl;
				return false;
			}
		}
		// arch ç¸ºE® "Threat=<dims>" ç¹§åï½¤æE½¨E¼(profile é©ï½©éï½¨è å¾ãE THREAT_DIM ç¸ºE¨è³Âé¾E´è ¢ã»E°ã»ç¸²ã»
		{
			std::size_t tp = arch.find("Threat=");
			std::size_t p = tp + 7; int dims = 0;
			while (p < arch.size() && arch[p] >= '0' && arch[p] <= '9') dims = dims * 10 + (arch[p++] - '0');
			if (dims != THREAT_DIM) {
				sync_cout << "info string P2NNUE: Threat dims " << dims << " != build " << THREAT_DIM
				          << " (profile=" << Threat::THREAT_PROFILE_ID << ")" << sync_endl;
				return false;
			}
		}
		// pid > 0 ç¸ºE® net ç¸ºE¯é©¥éªâ© blob é¶E´èéªã» u32 LE profile id ç¹§å®ï½ªE­ç¸ºE¿ç¸²â¥rch marker ç¸ºE¨é¯âç²ç¸²ã»
		if (file_profile != 0) {
			std::uint32_t bin_pid = r.read_u32_le();
			if (!r.ok() || (int)bin_pid != file_profile) {
				sync_cout << "info string P2NNUE: Threat profile prefix " << bin_pid
				          << " != arch marker " << file_profile << sync_endl;
				return false;
			}
		}
		// 2026-09-12: threat 表の bucket 群化 marker "ThreatGroups=<G>:<b0>-<b1>-...-<b{NUM_BUCKETS-1}>,"。
		//   有れば群 0..G-1 の表が連結されている。 無ければ従来どおり 1 表。
		int threat_groups = 1;
		{
			std::size_t gp = arch.find("ThreatGroups=");
			if (gp != std::string::npos) {
				std::size_t p = gp + 13;
				int g = 0;
				while (p < arch.size() && arch[p] >= '0' && arch[p] <= '9') g = g * 10 + (arch[p++] - '0');
				bool ok = g >= 2 && g <= NUM_BUCKETS && p < arch.size() && arch[p] == ':';
				for (int b = 0; ok && b < NUM_BUCKETS; ++b) {
					++p; // ':' または '-' を飛ばす
					int v = 0, nd = 0;
					while (p < arch.size() && arch[p] >= '0' && arch[p] <= '9') { v = v * 10 + (arch[p++] - '0'); ++nd; }
					ok = nd > 0 && v < g && p < arch.size() && arch[p] == (b + 1 < NUM_BUCKETS ? '-' : ',');
					if (ok) net->threat_bucket_group[b] = (std::uint8_t)v;
				}
				if (!ok) {
					sync_cout << "info string P2NNUE: ThreatGroups marker parse fail (need " << NUM_BUCKETS
					          << " bucket entries)" << sync_endl;
					return false;
				}
				if (Threat::THREAT_REPRESENTATIVE_ROWS || Threat::THREAT_PROFILE_ID == 11) {
					sync_cout << "info string P2NNUE: ThreatGroups net is not supported by profile 11/12 builds" << sync_endl;
					return false;
				}
				threat_groups = g;
			}
		}
		net->threat_groups = threat_groups;
		const std::size_t tw_count = (std::size_t)THREAT_DIM * THREAT_ACC_DIM * (std::size_t)threat_groups;
		if (threat_i16) {
			// i16 è¿ãE LE 2byte/weightç¸²E¹86 ç¸ºE¯ LE ç¸ºE®ç¸ºæºï½Ememcpy ç¸ºE§é¶E´é±E­ç¸ºE¿(read_bytes)ç¸²ã»
			net->threat_weight_i16.resize(tw_count);
			if (!r.read_bytes(net->threat_weight_i16.data(), tw_count * 2)) {
				sync_cout << "info string P2NNUE: Threat weight(i16) read fail (need " << tw_count * 2
				          << " bytes)" << sync_endl;
				return false;
			}
			net->threat_is_i16 = true;
		} else {
			net->threat_weight.resize(tw_count);
			if (!r.read_bytes(net->threat_weight.data(), tw_count)) {
				sync_cout << "info string P2NNUE: Threat weight read fail (need " << tw_count << " bytes)" << sync_endl;
				return false;
			}
			net->threat_is_i16 = false;
		}
		sync_cout << "info string P2NNUE: Threat weight format = " << (threat_i16 ? "i16" : "i8") << sync_endl;
		if (threat_groups > 1) {
			sync_cout << "info string P2NNUE: ThreatGroups=" << threat_groups << " bucket->group";
			for (int b = 0; b < NUM_BUCKETS; ++b) std::cout << ' ' << (int)net->threat_bucket_group[b];
			std::cout << sync_endl;
		}
		net->has_threat = true;
		// dirty-threat éæ»ãEç¸ºE®è­çæè¹ã»do_move ç¸ºE® hot path ç¸ºE§è»E¤è³å£¹EE1 bool ç¸ºE«é¡E³ç¹§Â)ç¸²ã»
		// threat è¾ï½¡èï½¹ç¹ï½»full è ï½·è»E¶ç¸ºE®ç¸ºE¨ç¸ºéªã»éæ»ãEç¹§E³ç¹§E¹ç¹å»Eç¹§E¼ç¹ï½­ç¸ºE«ç¸ºå¶Eç¸²ã»
		// profile11_full_recompute_default: è³æ»ï½¸E¬ç¸ºE§ç¸ºE¯dirtyéæ»ãE+è¾E®è»ã»âéï½¨ç¹§å»Eç¸²ã»
                // è³E¡éä¹ãfull dual fusedèå´ï½¨è²E®å±ã»è­E½¹ç¸ºç¢E½«å¬Âæºâ¡ç¸ºE£ç¸ºæºâç¹§âÂErofile 11ç¸ºE¯è­ï½¢è³å£¹ãdirtyç¹§å ¤å½è¬èEE ç¸ºEªç¸ºã»Âã»
                // profile 12é²å³¨ã»è æºæç¸ºE©ç¸ºç¿«Edirtyè¾E®è»ã»Eè´E¿éï½¨ç¸ºå¶Eç¸²ã»
#if defined(P2_PROFILE11_DIRTY_DELTA)
                // ABéªEºè­E½­éï½¨: profile11ç¸ºE«ç¹§ãE½¾æºæç¸ºE®dirtyè¾E®è»ã»å½è¬èEEè¬ï½»ç¸ºå¶Âã»
                g_dt_build = P2_HAS_THREAT && !nothreat_env() && !force_full_env() && threat_delta_env();
#else
                g_dt_build = P2_HAS_THREAT && !nothreat_env() && !force_full_env()
                          && threat_delta_env()
                          && Threat::THREAT_PROFILE_ID != 11;
#endif
		// profile 16 (hanging-split): 守り手が動くと「動いていない pair」の h が変わるので、
		// apply_threat_step の affected 集合では足りない (§4.5)。 差分更新は常に無効。
		if (Threat::THREAT_HANG_SPLIT && g_dt_build) {
			g_dt_build = false;
			sync_cout << "info string P2NNUE: hanging-split profile -> threat delta update disabled" << sync_endl;
		}
#if defined(P2_PROBE_BUILD)
		// 計測専用 DT_COST: 群化の無効化を越えて tuple 構築だけ有効にする。
		//   plan.base は wrapper で潰すので評価は full のまま (bit-exact)。
		if (probe_dt_cost()) {
			g_dt_build = P2_HAS_THREAT && !nothreat_env();
			sync_cout << "info string P2NNUE: DT_COST probe -> dt build only (eval unchanged)" << sync_endl;
		} else
#endif
		// 群化 net: 差分更新 (P2_THREAT_DELTA) は止めて、 常に局面の群の表で full 再計算する。
		// 2026-10-07 (S2 -> 撤退): 群一致を base の採用条件に足して解放したうえで NPS を測ったが、
		//   G4 は +1.19pt / -0.94% (どちらも 0 を含む) で撤退線 +1.5% を下回った
		//   (RESULTS-20261007-THREAT-DELTA-GATE.md)。 09-09 の「K=64 で +0.13%」の再現。
		//   → ガードは据え置き。 ただし群 offset を通す修正 (apply_dirty_chain の off と
		//     base の群一致条件) は**正しさの改善なので残してある**。 解放するならこのガードを
		//     外すだけでよく、 表が合わなくなる本来のバグは既に無い。
		if (threat_groups > 1 && g_dt_build) {
			sync_cout << "info string P2NNUE: ThreatGroups net -> threat delta update (P2_THREAT_DELTA) disabled" << sync_endl;
			g_dt_build = false;
		}
		if (want_threat_direct && Threat::THREAT_PROFILE_ID == Direct15::PROFILE_ID) {
                   g_direct15_tables.build();
                   if (g_direct15_tables.dim != Direct15::DIM) {
                           sync_cout << "info string P2NNUE: Direct15 LUT dim " << g_direct15_tables.dim
                                     << " != " << Direct15::DIM << sync_endl;
                           return false;
                   }
           } else
                   g_threat_tables.build(); // legacy Threat and Profile14 Direct LUT.
		build_norm_sq();         // normalize_sq LUT
		// 2026-09-10: threat 列挙が空盤面利きを引くので、 Direct15 を使わない構成でも必ず作る。
		build_direct15_pseudo();   // 25.9KB、 起動時 1 回
		build_s1step();            // S1' step-piece candidate tables (~26KB)
		build_dir_tables();        // 2026-09-16 案B: 方向主導列挙の表 (23KB) + シフト/集合/表の全数自己検査
		if (!g_dirtab.ok) {
			sync_cout << "info string P2NNUE: S1B self-check failed -> load aborted" << sync_endl;
			return false;
		}
		// 有効 mask を必ず出す (本番ビルドで P2_S1B_DEFAULT を付け忘れると黙って旧経路になるため、 PGO 採取ログでも確認する)
		sync_cout << "info string P2NNUE: S1B mask=" << s1b_mask_env() << " (1=knight 2=silver 4=gold-like)" << sync_endl;
		// build è å¾ãE threat_dim è®æE½¨E¼(pair_base é±ä¸æ± == THREAT_DIM)ç¸²ã»
		if (Threat::THREAT_PROFILE_ID != Direct15::PROFILE_ID && g_threat_tables.threat_dim != THREAT_DIM) {
			sync_cout << "info string P2NNUE: threat LUT dim " << g_threat_tables.threat_dim
			          << " != " << THREAT_DIM << sync_endl;
			return false;
		}
	}

	// LayerStacks EEE8
	for (int b = 0; b < NUM_BUCKETS; ++b)
#if defined(P2_FC1_PREC)
		if (!read_fc(r, net->buckets[b], fc1p_file)) {
#else
		if (!read_fc(r, net->buckets[b])) {
#endif
			sync_cout << "info string P2NNUE: layerstack " << b << " read fail" << sync_endl; return false;
		}

	// é¨¾E£éª¼æ¶ã»ç¹ï½­ç¹ï½¼ç¹ãE è·ã»section ç¸ºE® 8byte magic ç¹§å®ï½ªE­ç¸ºE¿ dispatch
	auto magic_eq = [](const std::uint8_t* m, const char* s) {
		return std::memcmp(m, s, 8) == 0;
	};
#if defined(P2_AUXCHEAP)
	// P2AUXC section ç¸ºE®è³Âè­ãå¥³ç¸ºä»£ÂE¨lobal g_auxcheap_w ç¸ºE¸ç¸ºE¯èï½¨ section é±E­è ã»ã»èæº·E¾å¾âEç¸ºE®ç¸ºE¿
	// commit ç¸ºå¶EEé¨¾æ¯ï½¸E­èï½±è¬¨å±ã»é©ï½¨è»ã»EªE­ç¸ºE¿ç¸ºE§è³æº¯E¡å¾¡E¸E­ç¸ºE®é©¥éªâ©ç¹§åï½±å£¹EE¸ºEªç¸ºã»ç¸²ã»
	std::vector<std::int32_t> auxcheap_tmp;
#endif
#if defined(P2_MOB)
	std::vector<std::int16_t> mob_tmp;
#endif
#if defined(P2_THREAT_L1)
	std::vector<std::int16_t> tl1_tmp;
#endif
#if defined(P2_THREAT_L2)
	std::vector<std::int16_t> tl2_tmp;
#endif
#if defined(P2_TG)
	std::vector<std::int16_t> tg_tmp;
#endif
#if defined(P2_THREAT_PROJ)
	// 案1 ブロック射影 (2026-09-30)。 P2_MOB とは独立のガードにする。
	std::vector<std::int16_t> tprj_tmp;
	int tprj_off_tmp = 0;
	std::vector<int> tprj_blk_tmp;
	int tprj_nb_tmp = 1;
#endif
#if defined(P2_ENDGAME_TEMPO)
	std::vector<std::int16_t> endgame_tempo_tmp;
#endif
        while (!r.eof()) {
		std::uint8_t magic[8];
		if (!r.peek_magic8(magic)) break;
		r.read_bytes(magic, 8); // consume
          if (std::memcmp(magic, "P2THRD1", 8) == 0) {
                  if (!want_threat_direct || Threat::THREAT_PROFILE_ID != 14) { sync_cout << "info string P2NNUE: P2THRD1 requires profile 14" << sync_endl; return false; }
                  if (!want_threat_direct) { sync_cout << "info string P2NNUE: ThreatDirect section but profile 14 is disabled" << sync_endl; return false; }
                  const std::uint32_t sec_ver = r.read_u32_le();
                  const std::uint32_t sec_profile = r.read_u32_le();
                  const std::uint32_t sec_dims = r.read_u32_le();
                  const std::uint32_t sec_buckets = r.read_u32_le();
                  if (sec_ver != 1 || sec_profile != 14 || sec_dims > (std::uint32_t)THREAT_DIM || sec_dims == 0 || sec_buckets != (std::uint32_t)NUM_BUCKETS) {
                          sync_cout << "info string P2NNUE: ThreatDirect section header mismatch" << sync_endl; return false;
                  }
                  net->threat_direct_w.resize((std::size_t)NUM_BUCKETS * DIRECT_THREAT_DIM); for (std::size_t i = 0; i < (std::size_t)NUM_BUCKETS * DIRECT_THREAT_DIM; ++i) net->threat_direct_w[i] = 0;
                  std::int64_t threat_direct_max_abs = 0;
                  for (int b = 0; b < NUM_BUCKETS; ++b)
                          for (int f = 0; f < sec_dims; ++f) {
                                  const std::int32_t q = r.read_i32_le();
                                  net->threat_direct_w[(std::size_t)b * DIRECT_THREAT_DIM + f] = q;
                                  const std::int64_t aq = q < 0 ? -std::int64_t(q) : std::int64_t(q);
                                  if (aq > threat_direct_max_abs) threat_direct_max_abs = aq;
                          }
                  net->has_threat_direct = r.ok();
                  if (!net->has_threat_direct) { sync_cout << "info string P2NNUE: ThreatDirect payload read fail" << sync_endl; return false; }
                  // At most 40 non-king attackers target 40 non-king pieces per
                  // perspective. The final difference has two such sums.
                  constexpr std::int64_t DIRECT_I32_ABS_LIMIT = 671088;
                  net->threat_direct_i32_safe = threat_direct_max_abs <= DIRECT_I32_ABS_LIMIT;
                  sync_cout << "info string P2NNUE: ThreatDirect loaded (bucketed scalar, "
                            << (net->threat_direct_i32_safe ? "i32 sum" : "i64 sum") << ")" << sync_endl;
                     } else if (std::memcmp(magic, "P2THRD2", 8) == 0) {
                   if (!want_threat_direct || Threat::THREAT_PROFILE_ID != Direct15::PROFILE_ID) {
                           sync_cout << "info string P2NNUE: P2THRD2 requires profile 15" << sync_endl; return false;
                   }
                   const std::uint32_t sec_ver = r.read_u32_le();
                   const std::uint32_t sec_profile = r.read_u32_le();
                   const std::uint32_t sec_dims = r.read_u32_le();
                   const std::uint32_t sec_buckets = r.read_u32_le();
                   // 形式判別は payload サイズ (section header の dims / buckets) で自己記述的に行う。
                   //   dims    = DIM           : 従来
                   //           = 2*DIM         : P5 defended (被攻撃駒が守られているかで次元 x2。
                   //                             ファイル上は [守備なし DIM | 守備あり DIM] の順)
                   //   buckets = NUM_BUCKETS   : 従来 (stm / nstm 視点で同一テーブル)
                   //           = 2*NUM_BUCKETS : P2 stm/nstm 別テーブル (ファイル上は
                   //                             bucket 0..7 = stm 視点、 8..15 = nstm 視点)
                   // 旧 net (DIM x NUM_BUCKETS) は従来どおり読め、 評価値も bit-exact。
                   const bool file_defended    = sec_dims    == 2u * (std::uint32_t)Direct15::DIM;
                   const bool file_persp_split = sec_buckets == 2u * (std::uint32_t)NUM_BUCKETS;
                   if (!r.ok() || sec_ver != 2 || sec_profile != Direct15::PROFILE_ID
                       || !(sec_dims == (std::uint32_t)Direct15::DIM || file_defended)
                       || !(sec_buckets == (std::uint32_t)NUM_BUCKETS || file_persp_split)) {
                           sync_cout << "info string P2NNUE: Direct15 section header mismatch (dims=" << sec_dims
                                     << " buckets=" << sec_buckets << ")" << sync_endl; return false;
                   }
                   // Direct15 LUT initialization after P2THRD2 header validation.
                   // want_threat_direct can still be false at the earlier architecture
                   // preflight, so initialize here where profile 15 is now confirmed.
                   g_direct15_tables.build();
                   // Profile15 Direct15 requires the perspective normalization LUT before evaluation.
                   build_norm_sq();
                   if (g_direct15_tables.dim != Direct15::DIM) {
                           sync_cout << "info string P2NNUE: Direct15 LUT dim " << g_direct15_tables.dim
                                     << " != " << Direct15::DIM << sync_endl;
                           return false;
                   }
                   // 推論用の空盤面利き表 (attacker 事前フィルタ用) をここで用意する。
                   build_direct15_pseudo();
                   // ファイル形式 (P2THRD1/P2THRD2 のバイト列) は一切変えない。
                   // 一旦そのままの並びで読み、 推論用に dc-minor へ並べ替える。
                   const int         file_buckets = (int)sec_buckets;
                   const int         halves       = file_defended ? 2 : 1;
                   const std::size_t file_dims    = (std::size_t)sec_dims;   // = halves * DIM
                   std::vector<std::int32_t> raw((std::size_t)file_buckets * file_dims);
                   std::int64_t max_abs = 0;
                   for (std::size_t i = 0; i < raw.size(); ++i) {
                           const std::int32_t q = r.read_i32_le();
                           raw[i] = q;
                           const std::int64_t aq = q < 0 ? -std::int64_t(q) : std::int64_t(q);
                           if (aq > max_abs) max_abs = aq;
                   }
                   net->has_threat_direct = r.ok();
                   if (!net->has_threat_direct) { sync_cout << "info string P2NNUE: Direct15 payload read fail" << sync_endl; return false; }

                   // dc-minor 並べ替え (bucket ごと、 defended 半分ごとに独立に適用):
                   //   file idx  = class_base[ac] + dc * A[ac] + r
                   //   infer idx = class_base[ac] + r * CLASS_NB + dc
                   // (ac, dc, r) の全単射なので重みの集合は不変 = 評価値は bit-exact。
                   // メモリ上の並び: [bucket][half][dc-minor DIM]、 bucket stride = halves * DIM。
                   const std::size_t bucket_stride = file_dims;
                   const std::size_t direct_n = (std::size_t)file_buckets * bucket_stride;
                   net->threat_direct_w.resize(direct_n + Direct15::LOAD_PAD);
                   std::memset(net->threat_direct_w.data(), 0,
                               (direct_n + Direct15::LOAD_PAD) * sizeof(std::int32_t));
                   for (int fb = 0; fb < file_buckets; ++fb)
                   for (int h = 0; h < halves; ++h) {
                           const std::int32_t* const src = raw.data() + (std::size_t)fb * file_dims
                                   + (std::size_t)h * Direct15::DIM;
                           std::int32_t* const dst = net->threat_direct_w.data()
                                   + (std::size_t)fb * bucket_stride + (std::size_t)h * Direct15::DIM;
                           for (int ac = 0; ac < Direct15::CLASS_NB; ++ac) {
                                   const int cb = g_direct15_tables.class_base[ac];
                                   const int A  = Direct15::ATTACKS_PER_COLOR[ac];
                                   for (int dc = 0; dc < Direct15::CLASS_NB; ++dc) {
                                           const int sb = g_direct15_tables.pair_base[ac * Direct15::CLASS_NB + dc];
                                           for (int k = 0; k < A; ++k)
                                                   dst[cb + k * Direct15::CLASS_NB + dc] = src[sb + k];
                                   }
                           }
                   }
                   net->direct_bucket_stride = bucket_stride;
                   net->direct_nstm_offset   = file_persp_split ? (std::size_t)NUM_BUCKETS * bucket_stride : 0;
                   net->direct_defended      = file_defended;
                   constexpr std::int64_t DIRECT15_I32_ABS_LIMIT = 600000;
                   net->threat_direct_i32_safe = max_abs <= DIRECT15_I32_ABS_LIMIT;
                   // i16 複製。 範囲外なら i32 経路にフォールバックする (安全弁)。
                   // 2026-09-02 実測: i16 化は逆効果 (+75.5 ns/node、 head で +3.2pt)。
                   // 本 head は 1 eval あたり ~28 回の散在ロードしかなく、 フットプリント
                   // 縮小の利点より符号拡張と別テーブル経由のコストが勝る。 既定は i32 で、
                   // i16 を試したいときだけ P2_DIRECT_W16=1 で有効化する。
                   net->threat_direct_w16_ok = (max_abs <= 32767) && direct_force_w16_env();
                   if (net->threat_direct_w16_ok) {
                           net->threat_direct_w16.resize(direct_n + Direct15::LOAD_PAD);
                           for (std::size_t i = 0; i < direct_n + Direct15::LOAD_PAD; ++i)
                                   net->threat_direct_w16[i] = (std::int16_t)net->threat_direct_w[i];
                   }
                   // 学習時 cap と推論時 cap の一致: cap は raw 単位で cap_cp * train_fv_scale。
                   // 学習側 (shogi_layerstack.rs) は固定 fv_scale で cap を適用するため、
                   // 推論側で実行時 FV_SCALE を掛けると目盛りが 2 倍ズレる (FV_SCALE=28 運用時)。
                   net->direct_cap_raw = (net->train_fv_scale > 0)
                           ? (std::int32_t)(P2_THREAT_DIRECT_CAP_CP * net->train_fv_scale)
                           : 0;
                   if (net->direct_cap_raw == 0 && P2_THREAT_DIRECT_CAP_CP != 0) {
                           sync_cout << "info string P2NNUE: WARNING Direct15 net has no 'fv_scale=' marker "
                                     << "-> falling back to legacy cap (cap_cp x runtime FV_SCALE); "
                                     << "cap may mismatch training scale" << sync_endl;
                   }
                   sync_cout << "info string P2NNUE: Direct15 loaded (king class, cap="
                             << P2_THREAT_DIRECT_CAP_CP << "cp, cap_raw=" << net->direct_cap_raw
                             << " (cap_cp=" << P2_THREAT_DIRECT_CAP_CP
                             << " x train_fv=" << net->train_fv_scale << "), "
                             << (net->threat_direct_i32_safe ? "i32 sum" : "i64 sum")
                             << ", w=" << (net->threat_direct_w16_ok ? "i16" : "i32")
                             << ", max_abs=" << max_abs
                             << ", dims=" << sec_dims << (file_defended ? " (defended x2)" : "")
                             << ", buckets=" << sec_buckets << (file_persp_split ? " (stm/nstm split)" : "")
                             << ")" << sync_endl;
                   } else if (magic_eq(magic, "P2THRF16")) {
		        if (!want_threat_factor) { sync_cout << "info string P2NNUE: ThreatFactor16 section but profile 13 is disabled" << sync_endl; return false; }
		        const std::uint32_t sec_ver = r.read_u32_le();
		        const std::uint32_t sec_profile = r.read_u32_le();
		        const std::uint32_t sec_dims = r.read_u32_le();
		        const std::uint32_t sec_factor_dim = r.read_u32_le();
		        const std::uint32_t sec_buckets = r.read_u32_le();
		        if (!r.ok() || sec_ver != 1 || sec_profile != 13 || sec_dims > (std::uint32_t)THREAT_DIM || sec_dims == 0 ||
		            sec_factor_dim != (std::uint32_t)THREAT_FACTOR_DIM || sec_buckets != (std::uint32_t)NUM_BUCKETS) {
		                sync_cout << "info string P2NNUE: ThreatFactor16 section header mismatch" << sync_endl; return false;
		        }
		        const std::size_t code_count = (std::size_t)THREAT_DIM * THREAT_FACTOR_DIM;
		        net->threat_factor_code.resize(code_count);
		        if (!r.read_bytes(net->threat_factor_code.data(), code_count) ||
		            !r.read_bytes(net->threat_factor_out, sizeof(net->threat_factor_out))) {
		                sync_cout << "info string P2NNUE: ThreatFactor16 section payload read fail" << sync_endl; return false;
		        }
		        			// Collapse the 16-dimensional factor row and its bucket output vector once.
			// The i32 table is accepted only when the original i64 dot product is exact.
			net->threat_factor_proj.resize((std::size_t)NUM_BUCKETS * THREAT_DIM);
			for (int b = 0; b < NUM_BUCKETS; ++b) {
				std::int32_t* const dst = net->threat_factor_proj.data() + (std::size_t)b * THREAT_DIM;
				for (int f = 0; f < THREAT_DIM; ++f) {
					const std::int8_t* const code = net->threat_factor_code.data()
						+ (std::size_t)f * THREAT_FACTOR_DIM;
					std::int64_t dot = 0;
					for (int d = 0; d < THREAT_FACTOR_DIM; ++d)
						dot += (std::int64_t)code[d] * net->threat_factor_out[b][d];
					const std::int32_t packed = (std::int32_t)dot;
					if ((std::int64_t)packed != dot) {
						sync_cout << "info string P2NNUE: ThreatFactor16 projection overflow" << sync_endl;
						return false;
					}
					dst[f] = packed;
				}
			}
			net->has_threat_factor = true;
			sync_cout << "info string P2NNUE: ThreatFactor16 loaded (scalar projection)" << sync_endl;
		} else if (magic_eq(magic, "P2DUAL\0\0")) {
			// è­ï½§ reserve format: version(4) + 32B zero ç¹§ã»skip(èEEE¸ã»)
			r.read_u32_le();
			for (int i = 0; i < 32; ++i) (void)r.read_u8();
		} else if (magic_eq(magic, "P2DUAL3\0")) {
                std::uint32_t ver = r.read_u32_le(); std::uint32_t cnt = r.read_u32_le(); (void)ver;
                if (cnt != 880) { sync_cout << "info string P2NNUE: DUAL3 count " << cnt << sync_endl; return false; }
                net->dual3_w.resize(cnt);
                for (std::uint32_t i = 0; i < cnt; ++i) net->dual3_w[i] = r.read_i16_le();
                net->has_dual3 = r.ok();
                if (!net->has_dual3) {
                        sync_cout << "info string P2NNUE: DUAL3 payload read fail" << sync_endl;
                        return false;
                }
                sync_cout << "info string P2NNUE: Dual3 loaded (880 shared i16 weights)" << sync_endl;
			} else if (magic_eq(magic, "P2CR27\0\0")) {
	                std::uint32_t ver = r.read_u32_le(); std::uint32_t cnt = r.read_u32_le(); (void)ver;
	                if (cnt != 1728) { sync_cout << "info string P2NNUE: CROSS27 count " << cnt << sync_endl; return false; }
	                net->cross27_w.resize(cnt);
	                for (std::uint32_t i = 0; i < cnt; ++i) net->cross27_w[i] = r.read_i16_le();
	                net->has_cross27 = r.ok();
	                if (!net->has_cross27) {
	                        sync_cout << "info string P2NNUE: CROSS27 payload read fail" << sync_endl;
	                        return false;
	                }
	                sync_cout << "info string P2NNUE: CROSS27 loaded (1728 shared i16 weights)" << sync_endl;
        } else if (magic_eq(magic, "P2DUAL2\0")) {
			std::uint32_t ver = r.read_u32_le(); std::uint32_t cnt = r.read_u32_le(); (void)ver;
			if (cnt != 36288) { sync_cout << "info string P2NNUE: DUAL2 count " << cnt << sync_endl; return false; }
			net->dual_w.resize(cnt);
			for (std::uint32_t i = 0; i < cnt; ++i) net->dual_w[i] = r.read_i16_le();
			net->has_dual_psqt = r.ok();
		} else if (magic_eq(magic, "P2DUALZ\0")) {
			std::uint32_t ver = r.read_u32_le(); std::uint32_t cnt = r.read_u32_le(); (void)ver;
			if (cnt != 254016) { sync_cout << "info string P2NNUE: DUALZ count " << cnt << sync_endl; return false; }
			net->dualcross_w.resize(cnt);
			for (std::uint32_t i = 0; i < cnt; ++i) net->dualcross_w[i] = r.read_i16_le();
			net->has_dual_cross = r.ok();
		} else if (magic_eq(magic, "P2PAIR\0\0")) {
			std::uint32_t ver = r.read_u32_le(); std::uint32_t cnt = r.read_u32_le(); (void)ver;
			if (cnt != 3072) { sync_cout << "info string P2NNUE: PAIR count " << cnt << sync_endl; return false; }
			net->pairshape_w.resize(cnt);
			for (std::uint32_t i = 0; i < cnt; ++i) net->pairshape_w[i] = r.read_i16_le();
			net->has_pairshape = r.ok();
		} else if (magic_eq(magic, "P2AUX\0\0\0")) {
			// AUX section: version(u32) + i8 weights [AUX_DIM * FT_DIM](feature-major)ç¸²ã»
			std::uint32_t ver = r.read_u32_le();
			if (ver != Aux::AUX_SECTION_VERSION) {
				sync_cout << "info string P2NNUE: AUX version " << ver << " != "
				          << Aux::AUX_SECTION_VERSION << sync_endl; return false;
			}
			if (!P2_HAS_AUX) { sync_cout << "info string P2NNUE: AUX section but no -DP2_AUX" << sync_endl; return false; }
			const std::size_t aw_count = (std::size_t)AUX_DIM * FT_DIM;
			net->aux_weight.resize(aw_count);
			if (!r.read_bytes(net->aux_weight.data(), aw_count)) {
				sync_cout << "info string P2NNUE: AUX weight read fail (need " << aw_count << " bytes)" << sync_endl;
				return false;
			}
			net->has_aux = r.ok();
			sync_cout << "info string P2NNUE: AUX weight loaded (" << AUX_DIM << " x " << FT_DIM << " i8)" << sync_endl;
		} else if (magic_eq(magic, "P2AUXC\0\0")) {
			// AUXCHEAP PSQT-only section: version(u32 LE = 1) + i32 LE EEE(176EEE)ç¸²ã»
			// count ç¹è¼ãç¹ï½¼ç¹ï½«ç¹èEâç¸ºã»P2AUX ç¸ºE¨è·å¾Â§ç¸ºã»build è³å£½çE176EEE ç¸ºä¹ï½éæãEç¸²EE status4ç«ãE)ç¸²ã»
			// ç¹ï½¬ç¹§E¤ç¹§E¢ç¹§E¦ç¹å»ã» feature-major, bucket è­Âé¨¾ã»= w[local*8 + bucket] ç¸ºE§ç¸²EE+ èï½´
			// g_auxcheap_w[(size_t)local * NUM_BUCKETS + bucket] ç¸ºE¨è·å¾¡E¸Â ç«ãEèç»é¤è³å´ï½¦âãEé¶E´é±E­ç¸ºE¿ç¸²ã»
			// ç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½« QAEE·B=8128 = è­ï½¢èEEPSQT é©¥éªâ©ç¸ºE¨è·å¾¡E¸Âç¸ºE®èã»ÎEeval èåE½½éªÂãE½¯ã»E¸ä¸»E¼ã»
			// (ac_stm[b]-ac_nstm[b])/2 ç¸ºE¯ eval èï½´ç¸ºE®è­ï½¢èE¼ï½¼ä¸ãEç¸ºE¾ç¸ºE¾(ç¸ºèï¼E¸ºE§ /2 ç¸ºå±âç¸ºã»= è ç¢ã¾é©ï½©éï½¨éâµE­E¢)ç¸²ã»
			// bullet writer: examples/shogi_layerstack.rs ç¸ºE® enable_auxcheap section ç¸ºE¨èE½¾è ¢æÂã»
			std::uint32_t ver = r.read_u32_le();
			if (!r.ok() || ver != 1) {
				sync_cout << "info string P2NNUE: AUXCHEAP section version " << ver
				          << " != 1" << sync_endl;
				return false;
			}
#if defined(P2_AUXCHEAP)
			const std::size_t acw_count = (std::size_t)AuxCheap::DIM_PER_PERSP * NUM_BUCKETS; // 176*8=1408 (C1)
			auxcheap_tmp.resize(acw_count);
			// x86 ç¸ºE¯ LE ç¸ºE®ç¸ºæºï½Ememcpy ç¸ºE§é¶E´é±E­ç¸ºE¿(threat i16 ç¸ºE¨è·å¾Â§è´æï½³ã»ç¸²ã»
			if (!r.read_bytes(auxcheap_tmp.data(), acw_count * sizeof(std::int32_t))) {
				sync_cout << "info string P2NNUE: AUXCHEAP weight read fail (need "
				          << acw_count * sizeof(std::int32_t) << " bytes)" << sync_endl;
				return false;
			}
#else
			// é¨¾å£¼E¸E¸ç¸ºE¯ want_auxcheap ç¸ºE®è å¥çè«¡è²åç¸ºE§è»E°é©æE ç¸ºEªç¸ºã»ÂE®arker è¾ï½¡ç¸ºå±ãEsection ç¸ºE°ç¸ºç¬¬æç¸ºE¤
			// è³è´ç´è·ã»net ç¹§ãï½»å¶â²ç¸ºE¦ skip ç¸ºå¸âEè­å®ï½¢Eºè«¡è²å(é©¥éªâ©ç¸ºæ¢§E­E»é¡Eµç¸ºè¼ï½ç¹§ä¹âç¹§ã»ç¸²ã»
			sync_cout << "info string P2NNUE: P2AUXC section but build lacks -DP2_AUXCHEAP" << sync_endl;
			return false;
#endif
		} else if (magic_eq(magic, "P2TL1\0\0\0")) {
			// (3) TL1: version(u32) + K2(u32) + L1_OUT(u32) + NUM_BUCKETS(u32) + i16[bucket][o][c]
			const std::uint32_t ver = r.read_u32_le();
			const std::uint32_t k2 = r.read_u32_le();
			const std::uint32_t l1n = r.read_u32_le();
			const std::uint32_t nb = r.read_u32_le();
			if (!r.ok() || ver != 1) {
				sync_cout << "info string P2NNUE: TL1 section version " << ver << " != 1" << sync_endl;
				return false;
			}
#if defined(P2_THREAT_L1)
			if (k2 != (std::uint32_t)(2 * THREAT_ACC_DIM) || l1n != (std::uint32_t) L1_OUT
			    || nb != (std::uint32_t) NUM_BUCKETS) {
				sync_cout << "info string P2NNUE: TL1 形が違う (K2=" << k2 << " L1=" << l1n
				          << " B=" << nb << ")" << sync_endl;
				return false;
			}
			tl1_tmp.resize((std::size_t) nb * l1n * k2);
			for (std::size_t i = 0; i < tl1_tmp.size(); ++i) tl1_tmp[i] = r.read_i16_le();
			if (!r.ok()) {
				sync_cout << "info string P2NNUE: TL1 weight read fail" << sync_endl; return false;
			}
#else
			// 🚨-DP2_THREAT_L1 が無いビルドで TL1 net を読むと、 重みが無視されて
			//   **黙って別の評価関数**になる (★19)。 警告して読み飛ばす。
			sync_cout << "info string P2NNUE: **TL1 MISMATCH** net has P2TL1 section but build lacks "
			          << "-DP2_THREAT_L1. threat->L1 weights are IGNORED." << sync_endl;
			for (std::size_t i = 0; i < (std::size_t) nb * l1n * k2; ++i) (void) r.read_i16_le();
#endif
		} else if (magic_eq(magic, "P2TL2\0\0\0")) {
			// (4) TL2: version(u32) + K2(u32) + L2_IN(u32) + NUM_BUCKETS(u32) + i16[bucket][j][c]
			const std::uint32_t ver = r.read_u32_le();
			const std::uint32_t k2 = r.read_u32_le();
			const std::uint32_t l2n = r.read_u32_le();
			const std::uint32_t nb = r.read_u32_le();
			// ver 1 = S 64 (shift 7)、 ver 2 = S 1024 (shift 11)。 それ以外はエラー。
			if (!r.ok() || (ver != 1 && ver != 2)) {
				sync_cout << "info string P2NNUE: TL2 section version " << ver << " (1 か 2 のみ)" << sync_endl;
				return false;
			}
#if defined(P2_THREAT_L2)
			g_tl2_ver = ver;
			if (k2 != (std::uint32_t)(2 * THREAT_ACC_DIM) || l2n != (std::uint32_t) L2_IN
			    || nb != (std::uint32_t) NUM_BUCKETS) {
				sync_cout << "info string P2NNUE: TL2 形が違う (K2=" << k2 << " L2_IN=" << l2n
				          << " B=" << nb << ")" << sync_endl;
				return false;
			}
			tl2_tmp.resize((std::size_t) nb * l2n * k2);
			for (std::size_t i = 0; i < tl2_tmp.size(); ++i) tl2_tmp[i] = r.read_i16_le();
			if (!r.ok()) {
				sync_cout << "info string P2NNUE: TL2 weight read fail" << sync_endl; return false;
			}
#else
			// 🚨-DP2_THREAT_L2 が無いビルドで TL2 net を読むと、 重みが無視されて
			//   **黙って別の評価関数**になる (★net_exe_mismatch_silent)。 警告して読み飛ばす。
			sync_cout << "info string P2NNUE: **TL2 MISMATCH** net has P2TL2 section but build lacks "
			          << "-DP2_THREAT_L2. threat->L2 weights are IGNORED." << sync_endl;
			for (std::size_t i = 0; i < (std::size_t) nb * l2n * k2; ++i) (void) r.read_i16_le();
#endif
		} else if (magic_eq(magic, "P2TGS\0\0\0")) {
			// (5) TG: version(u32 = 1) + K2(u32) + SLOTS(u32 = MAIN_DIM) + NUM_BUCKETS(u32)
			//         + i16[bucket][slot][c]   (設計書 3 節)
			const std::uint32_t ver   = r.read_u32_le();
			const std::uint32_t k2    = r.read_u32_le();
			const std::uint32_t slots = r.read_u32_le();
			const std::uint32_t nb    = r.read_u32_le();
			if (!r.ok() || ver != 1) {
				sync_cout << "info string P2NNUE: TG section version " << ver << " != 1" << sync_endl;
				return false;
			}
#if defined(P2_TG)
			if (k2 != (std::uint32_t)(2 * THREAT_ACC_DIM) || slots != (std::uint32_t) MAIN_DIM
			    || nb != (std::uint32_t) NUM_BUCKETS) {
				sync_cout << "info string P2NNUE: TG 形が違う (K2=" << k2 << " SLOTS=" << slots
				          << " B=" << nb << ")" << sync_endl;
				return false;
			}
			tg_tmp.resize((std::size_t) nb * slots * k2);
			for (std::size_t i = 0; i < tg_tmp.size(); ++i) tg_tmp[i] = r.read_i16_le();
			if (!r.ok()) {
				sync_cout << "info string P2NNUE: TG weight read fail" << sync_endl; return false;
			}
#else
			// 🚨TL1 / TL2 は「警告して読み飛ばす」が、 TG は**拒否**する (設計書 3 節 ⑤)。
			//   EXE= の指定ミスで黙って TG 無しの判定が走る事故を防ぐため
			//   (★league_exe_default_silently_wrong_20261003)。
			sync_cout << "info string P2NNUE: **TG MISMATCH** net has P2TGS section but build lacks "
			          << "-DP2_TG. REJECT (黙って TG 無しでは動かさない)." << sync_endl;
			return false;
#endif
		} else if (magic_eq(magic, "P2TPRJ\0\0")) {
			// 案1 ブロック射影: version(u32) + count(u32) + i16 x count (Q15)。
			const std::uint32_t ver = r.read_u32_le();
			const std::uint32_t cnt = r.read_u32_le();
			if (!r.ok() || ver < 1 || ver > 4) {
				sync_cout << "info string P2NNUE: TPRJ section version " << ver << " (1..4 のみ)" << sync_endl;
				return false;
			}
			// version 1 = ずらし無し / 2 = hi 側だけ一定 / 3 = 16 次元ブロックごと。
			std::uint32_t sec_off = 0;
			std::vector<int> sec_blk;
			if (ver == 2) {
				sec_off = r.read_u32_le();
				if (!r.ok()) { sync_cout << "info string P2NNUE: TPRJ offset read fail" << sync_endl; return false; }
				if (sec_off >= (std::uint32_t) THREAT_ACC_DIM) {
					sync_cout << "info string P2NNUE: TPRJ offset " << sec_off << " >= K" << sync_endl; return false;
				}
			} else if (ver >= 3) {
				const std::uint32_t nb = r.read_u32_le();
				if (!r.ok() || nb != (std::uint32_t)(FT_DIM / 16)) {
					sync_cout << "info string P2NNUE: TPRJ block 数 " << nb << " != " << (FT_DIM / 16) << sync_endl;
					return false;
				}
				sec_blk.resize(nb);
				for (std::uint32_t i = 0; i < nb; ++i) {
					const int o = (int) r.read_u8();
					if (o >= THREAT_ACC_DIM) {
						sync_cout << "info string P2NNUE: TPRJ block offset " << o << " >= K" << sync_endl; return false;
					}
					sec_blk[i] = o;
				}
				if (!r.ok()) { sync_cout << "info string P2NNUE: TPRJ block offset read fail" << sync_endl; return false; }
			}
			// version 4: bucket 数。 1-3 は 1 (全 bucket に同じ表を複製する)。
			std::uint32_t sec_nb = 1;
			if (ver >= 4) {
				sec_nb = r.read_u32_le();
				if (!r.ok() || (sec_nb != 1 && sec_nb != (std::uint32_t) NUM_BUCKETS)) {
					sync_cout << "info string P2NNUE: TPRJ bucket 数 " << sec_nb << " は 1 か "
					          << NUM_BUCKETS << " のみ" << sync_endl;
					return false;
				}
			}
#if defined(P2_THREAT_PROJ)
			if (cnt != (std::uint32_t) FT_DIM) {
				sync_cout << "info string P2NNUE: TPRJ count " << cnt << " != " << FT_DIM << sync_endl;
				return false;
			}
			tprj_off_tmp = (int) sec_off;
			tprj_blk_tmp = sec_blk;
			tprj_nb_tmp = (int) sec_nb;
			tprj_tmp.resize((std::size_t) cnt * sec_nb);
			for (std::size_t i = 0; i < tprj_tmp.size(); ++i) tprj_tmp[i] = r.read_i16_le();
			if (!r.ok()) {
				sync_cout << "info string P2NNUE: TPRJ weight read fail" << sync_endl;
				return false;
			}
#else
			// 🚨-DP2_THREAT_PROJ が無いビルドで射影 net を読むと、 重みが無視されて
			//   **黙って別の評価関数**になる (★19 の net x exe 不一致)。 警告して読み飛ばす。
			sync_cout << "info string P2NNUE: **TPRJ MISMATCH** net has P2TPRJ section but build lacks "
			          << "-DP2_THREAT_PROJ. projection weights are IGNORED." << sync_endl;
			for (std::size_t i = 0; i < (std::size_t) cnt * sec_nb; ++i) (void) r.read_i16_le();
#endif
		} else if (magic_eq(magic, "P2MOB\0\0\0")) {
			// B6 可動域: version(u32) + count(u32) + i16 x count。
			// 目盛りは PSQT と同じ QA*QB = 8128 (mob head は psqt_diff と同じ場所に足すため)。
			std::uint32_t ver = r.read_u32_le();
			std::uint32_t cnt = r.read_u32_le();
			if (!r.ok() || ver != 1) {
				sync_cout << "info string P2NNUE: MOB section version " << ver << " != 1" << sync_endl;
				return false;
			}
#if defined(P2_MOB)
			if (cnt != (std::uint32_t) NUM_BUCKETS) {
				sync_cout << "info string P2NNUE: MOB count " << cnt << " != " << NUM_BUCKETS << sync_endl;
				return false;
			}
			mob_tmp.resize(cnt);
			for (std::uint32_t i = 0; i < cnt; ++i) mob_tmp[i] = r.read_i16_le();
			if (!r.ok()) {
				sync_cout << "info string P2NNUE: MOB weight read fail" << sync_endl;
				return false;
			}
#else
			// 🚨-DP2_MOB が無いビルドで MOB net を読むと、 重みが無視されて
			//   **黙って別の評価関数**になる (★19 の net x exe 不一致)。 警告して読み飛ばす。
			sync_cout << "info string P2NNUE: **MOB MISMATCH** net has P2MOB section but build lacks "
			          << "-DP2_MOB. mobility weights are IGNORED." << sync_endl;
			for (std::uint32_t i = 0; i < cnt; ++i) (void) r.read_i16_le();
#endif
		} else if (magic_eq(magic, "P2EGTMP\0")) {
			const std::uint32_t ver = r.read_u32_le();
			const std::uint32_t cnt = r.read_u32_le();
			if (!r.ok() || ver != 1 || cnt != 1024) {
				sync_cout << "info string P2NNUE: EGT header mismatch version=" << ver
				          << " count=" << cnt << sync_endl;
				return false;
			}
#if defined(P2_ENDGAME_TEMPO)
			endgame_tempo_tmp.resize(cnt);
			for (std::uint32_t i = 0; i < cnt; ++i) endgame_tempo_tmp[i] = r.read_i16_le();
			if (!r.ok()) {
				sync_cout << "info string P2NNUE: EGT weight read fail" << sync_endl;
				return false;
			}
#else
			sync_cout << "info string P2NNUE: P2EGTMP section but build lacks -DP2_ENDGAME_TEMPO"
			          << sync_endl;
			return false;
#endif
		} else {
			sync_cout << "info string P2NNUE: unknown section magic, stop" << sync_endl;
			break;
		}
	}
	(void)want_dual;

	// P2_AUX_FORCE: net ç¸ºã»aux section ç¹§åæç¸ºæºâEç¸ºã»E°E´è·å»ãç¹§ã»zero é©¥éªâ©ç¸ºE® aux ç¹ãEã»ç¹æ¶Îç¹§ã»
	// è·åãEç¸ºã»has_aux=true ç¸ºE«ç¸ºå¶EEeval èï½¤ç¸ºE¯è³æ¦ï½¤å³¨ÂâãEè¬ãEé¦æåEéå±ãç¹§E¹ç¹å»ã»ç¸ºE¿éï½ºéãE= Gate1 NPS éªåï½¸E¬éï½¨)ç¸²ã»
	if (P2_HAS_AUX && !net->has_aux && aux_force_env()) {
		const std::size_t aw_count = (std::size_t)AUX_DIM * FT_DIM;
		net->aux_weight.resize(aw_count);
		std::memset(net->aux_weight.data(), 0, aw_count); // zero é©¥éªâ©(eval è³æ¦ï½¤ã»
		net->has_aux = true;
		sync_cout << "info string P2NNUE: AUX FORCED (zero weights, " << AUX_DIM << " x " << FT_DIM
		          << ") for NPS measurement" << sync_endl;
	}
  if (want_threat_factor && !net->has_threat_factor) {
          sync_cout << "info string P2NNUE: ThreatFactor16 section missing" << sync_endl;
          return false;
  }
  if (want_threat_direct && !net->has_threat_direct) {
          sync_cout << "info string P2NNUE: ThreatDirect section missing" << sync_endl;
          return false;
  }

	net->loaded = r.ok();
	if (!net->loaded) { sync_cout << "info string P2NNUE: trailing read error" << sync_endl; return false; }

#if defined(P2_AUXCHEAP)
	// marker ç¸ºã»section ç¸ºE®èE¼æ ç¹§å ¤E´ã»æç¸ºå±â»ç¸ºã»Eç¸ºE®ç¸ºE« magic é¨¾E£éª¼æ¶ãéä¹â½ç¸ºä¹ï½ç¸ºEªç¸ºä¹â²ç¸ºã»ç«ãEè³è´ç´è·ã»netç¸²ã»
	if (want_auxcheap && auxcheap_tmp.empty()) {
		sync_cout << "info string P2NNUE: arch has AUXCHEAP marker but P2AUXC section missing" << sync_endl;
		return false;
	}
	// P2AUXC commit(èï½¨ section é±E­è ã»ã»èæº·E¾å¾ãEç¸ºE¿)ç¸²ãEç¹§E¤ç¹§E¢ç¹§E¦ç¹äº¥éè³Âç¸ºE®ç¸ºæºï½Ememcpy é¶E´èå¶E ç¸²ã»
	if (!auxcheap_tmp.empty()) {
		static_assert(sizeof(g_auxcheap_w)
		              == (std::size_t)AuxCheap::DIM_PER_PERSP * NUM_BUCKETS * sizeof(std::int32_t),
		              "g_auxcheap_w size != P2AUXC section weight count");
		std::memcpy(g_auxcheap_w, auxcheap_tmp.data(), sizeof(g_auxcheap_w));
		g_auxcheap_loaded = true;
		g_auxcheap_inited = true;  // auxcheap_init() ç¸ºE®ç¹§E¼ç¹ï½­èä¹ï½ç¹§å ¤âèE¹è¹ã»è³æ»E¾ç¸ºE¿ç¹§åE½¸é å¶ç¸ºéªE ç¸ºEªç¸ºã»
		build_statkd_wb();         // è³æ»E¾ç¸ºE¿ç¸ºä¹ï½ESTATKD side-weight ç¹ãEã»ç¹æ¶Îç¹§è²ãEè®çE¯ã»count-linear è¬ï½¨é²ä¹ç)
		// è®æE½¨E¼ç¹ï½­ç¹§E°: é«±æ§­ãç¹ï½­é©¥éªâ©è¬¨E°ç¸ºE¨é¨E¶èE½¾èï½¤è­Âèï½§(èï½¨ç¹§E¼ç¹ï½­ net ç¸ºE®è®æ·ã»éï½¨)ç¸²ã»
		std::size_t nz = 0;
		std::int32_t amax = 0;
		for (std::size_t i = 0; i < (std::size_t)AuxCheap::DIM_PER_PERSP * NUM_BUCKETS; ++i) {
			const std::int32_t a = g_auxcheap_w[i] < 0 ? -g_auxcheap_w[i] : g_auxcheap_w[i];
			if (a != 0) ++nz;
			if (a > amax) amax = a;
		}
		sync_cout << "info string P2NNUE: AUXCHEAP weight loaded (" << AuxCheap::DIM_PER_PERSP
		          << " x " << NUM_BUCKETS << " i32, nonzero=" << nz << " absmax=" << amax << ")" << sync_endl;
	} else if (g_auxcheap_loaded) {
		// marker è¾ï½¡ç¸ºã»net ç¸ºE¸ç¸ºE®èéªÎç¹ï½¼ç¹ãE è³æ»E¾ç¸ºE¿ç¹§å ¤E°E´è­½ã»E è æºæç¸ºE®ç¹§E¼ç¹ï½­èEEE¸å¼±âè¬ï½»ç¸ºã»
		// (è°ºE¡èãEeval ç¸ºE® auxcheap_init() ç¸ºã»volatile éï½±è­ï½¥ç¹§E¼ç¹ï½­ç¸ºE§èä¹ï½E¶E´ç¸ºã»ç¸²ã»
		g_auxcheap_loaded = false;
		g_auxcheap_inited = false;
		sync_cout << "info string P2NNUE: AUXCHEAP weights reset (net has no P2AUXC section)" << sync_endl;
	}
#endif

#if defined(P2_THREAT_L1)
	if (want_tl1 && tl1_tmp.empty()) {
		sync_cout << "info string P2NNUE: arch has ThreatL1 marker but P2TL1 section missing" << sync_endl;
		return false;
	}
	if (!tl1_tmp.empty()) {
		std::size_t idx = 0;
		long long nz = 0, amax = 0;
		for (int b = 0; b < NUM_BUCKETS; ++b)
			for (int o = 0; o < L1_OUT; ++o)
				for (int c = 0; c < 2 * THREAT_ACC_DIM; ++c) {
					const std::int16_t v = tl1_tmp[idx++];
					g_tl1_w[b][o][c] = v;
					if (v != 0) ++nz;
					const long long a = v < 0 ? -(long long) v : (long long) v;
					if (a > amax) amax = a;
				}
		g_tl1_loaded = true;
		// 2026-10-08: AVX2 の積和 (tl_dot_2k) は i32 レーンに 16 項ためる。
		//   あふれない上界は 16 * 32767 * |q|max。 |q|max <= 2048 なら 1.07e9 < 2^31 で安全。
		//   学習器は clip ±2 -> round(1024*w) なので必ず満たすが、 net を信用せず検査する。
		g_tl1_simd_ok = (amax <= 2048);
		sync_cout << "info string P2NNUE: TL1 loaded (K2=" << (2 * THREAT_ACC_DIM)
		          << " L1=" << L1_OUT << " B=" << NUM_BUCKETS
		          << " nonzero=" << nz << " max|q|=" << amax
		          << " simd=" << (g_tl1_simd_ok ? 1 : 0) << ")" << sync_endl;
		if (!g_tl1_simd_ok)
			sync_cout << "info string P2NNUE: **TL1 max|q| " << amax
			          << " > 2048** -> AVX2 の積和を使わずスカラで計算する (値は同一)" << sync_endl;
	} else {
		std::memset(g_tl1_w, 0, sizeof(g_tl1_w));
		g_tl1_loaded = false;
		g_tl1_simd_ok = false;
	}
#else
	(void) want_tl1;
#endif

#if defined(P2_THREAT_L2)
	if (want_tl2 && tl2_tmp.empty()) {
		sync_cout << "info string P2NNUE: arch has ThreatL2 marker but P2TL2 section missing" << sync_endl;
		return false;
	}
	if (!tl2_tmp.empty()) {
		std::size_t idx = 0;
		long long nz = 0, amax = 0;
		for (int b = 0; b < NUM_BUCKETS; ++b)
			for (int j = 0; j < L2_IN; ++j)
				for (int c = 0; c < 2 * THREAT_ACC_DIM; ++c) {
					const std::int16_t v = tl2_tmp[idx++];
					g_tl2_w[b][j][c] = v;
					if (v != 0) ++nz;
					const long long a = v < 0 ? -(long long) v : (long long) v;
					if (a > amax) amax = a;
				}
		g_tl2_loaded = true;
		// tl_dot_2k は i32 レーンに 16 項ためる。 あふれない上界は 16 * 32767 * |q|max。
		//   |q|max <= 2048 なら 1.07e9 < 2^31 で安全。 clip ±2 なので本来は ver1 (S=64) で
		//   |q| <= 128、 ver2 (S=1024) で |q| <= 2048 とどちらも内側。 net を信用せず
		//   検査して超えたらスカラに落とす。
		g_tl2_simd_ok = (amax <= 2048);
		sync_cout << "info string P2NNUE: TL2 loaded (ver=" << g_tl2_ver
		          << " S=" << (g_tl2_ver == 2 ? 1024 : 64)
		          << " shift=" << (g_tl2_ver == 2 ? 11 : 7)
		          << " K2=" << (2 * THREAT_ACC_DIM)
		          << " L2_IN=" << L2_IN << " B=" << NUM_BUCKETS
		          << " nonzero=" << nz << " max|q|=" << amax
		          << " simd=" << (g_tl2_simd_ok ? 1 : 0) << ")" << sync_endl;
		// isready 後に ThreatL2Gain の実効値を 1 行出す (配線確認用、 2026-10-09)。
		sync_cout << "info string P2NNUE: ThreatL2Gain=" << g_tl2_gain
		          << " (256 = x1.0)" << sync_endl;
		if (!g_tl2_simd_ok)
			sync_cout << "info string P2NNUE: **TL2 max|q| " << amax
			          << " > 2048** -> AVX2 の積和を使わずスカラで計算する (値は同一)" << sync_endl;
	} else {
		std::memset(g_tl2_w, 0, sizeof(g_tl2_w));
		g_tl2_loaded = false;
		g_tl2_simd_ok = false;
		g_tl2_ver = 1;
	}
#else
	(void) want_tl2;
#endif

#if defined(P2_TG)
	if (want_tg && tg_tmp.empty()) {
		sync_cout << "info string P2NNUE: arch has ThreatGate marker but P2TGS section missing" << sync_endl;
		return false;
	}
	if (!tg_tmp.empty()) {
		std::size_t idx = 0;
		long long nz = 0, amax = 0;
		for (int b = 0; b < NUM_BUCKETS; ++b)
			for (int i = 0; i < MAIN_DIM; ++i)
				for (int c = 0; c < 2 * THREAT_ACC_DIM; ++c) {
					const std::int16_t v = tg_tmp[idx++];
					g_tg_w[b][i][c] = v;
					if (v != 0) ++nz;
					const long long a = v < 0 ? -(long long) v : (long long) v;
					if (a > amax) amax = a;
				}
		g_tg_loaded = true;
		// tl_dot_2k は i32 レーンに 16 項ためる。 あふれない上界は 16 * 32767 * |q|max で、
		//   |q|max <= 2048 なら 1.07e9 < 2^31。 S_g = 1024 / clip |u| <= 2 なら満たすが
		//   net を信用せず検査し、 超えたらスカラに落とす (値は同一)。
		g_tg_simd_ok = (amax <= 2048);
		sync_cout << "info string P2NNUE: TG loaded (S_g=1024 shift=10 K2=" << (2 * THREAT_ACC_DIM)
		          << " SLOTS=" << MAIN_DIM << " B=" << NUM_BUCKETS
		          << " nonzero=" << nz << " max|q|=" << amax
		          << " simd=" << (g_tg_simd_ok ? 1 : 0) << ")" << sync_endl;
		// isready 後に ThreatGateScale の実効値を 1 行出す (配線確認用)。
		sync_cout << "info string P2NNUE: ThreatGateScale=" << g_tg_scale
		          << " (256 = x1.0)" << sync_endl;
		if (!g_tg_simd_ok)
			sync_cout << "info string P2NNUE: **TG max|q| " << amax
			          << " > 2048** -> AVX2 の積和を使わずスカラで計算する (値は同一)" << sync_endl;
	} else {
		std::memset(g_tg_w, 0, sizeof(g_tg_w));
		g_tg_loaded = false;
		g_tg_simd_ok = false;
	}
#else
	(void) want_tg;
#endif

#if defined(P2_THREAT_PROJ)
	if (want_tprj && tprj_tmp.empty()) {
		sync_cout << "info string P2NNUE: arch has ThreatProj marker but P2TPRJ section missing" << sync_endl;
		return false;
	}
	if (!tprj_tmp.empty()) {
		// bucket 別 (nb == NUM_BUCKETS) ならそのまま、 1 本なら全 bucket に複製する。
		for (int b = 0; b < NUM_BUCKETS; ++b) {
			const std::size_t off = (tprj_nb_tmp == 1) ? 0 : (std::size_t) b * FT_DIM;
			for (int i = 0; i < FT_DIM; ++i) g_tprj_w[b][i] = tprj_tmp[off + (std::size_t) i];
		}
		g_tprj_loaded = true;
		g_tprj_offset = tprj_off_tmp;
		// version 1/2 は式から、 version 3 はファイルの値からブロック表を作る。
		if (!tprj_blk_tmp.empty()) {
			for (int b = 0; b < FT_DIM / 16; ++b) g_tprj_blk[b] = tprj_blk_tmp[(std::size_t) b];
		} else {
			const int hb = (FT_DIM / 16) / 2;
			for (int b = 0; b < FT_DIM / 16; ++b) g_tprj_blk[b] = (b >= hb) ? g_tprj_offset : 0;
		}
		// 🚨sync_cout はロックを取るマクロ。 ループ内で複数回呼ぶとデッドロックする (2026-09-29)。
		long long nz = 0, amax = 0;
		for (int b = 0; b < NUM_BUCKETS; ++b)
			for (int i = 0; i < FT_DIM; ++i) {
				if (g_tprj_w[b][i] != 0) ++nz;
				const long long v = g_tprj_w[b][i];
				const long long a = v < 0 ? -v : v;
				if (a > amax) amax = a;
			}
		sync_cout << "info string P2NNUE: TPRJ loaded (FT=" << FT_DIM << " K=" << THREAT_ACC_DIM
		          << " nb=" << tprj_nb_tmp << " offset=" << g_tprj_offset << " blkmax=" << *std::max_element(g_tprj_blk, g_tprj_blk + FT_DIM / 16)
		          << " nonzero=" << nz << " max|q|=" << amax << ")" << sync_endl;
	} else {
		std::memset(g_tprj_w, 0, sizeof(g_tprj_w));
		g_tprj_loaded = false;
		g_tprj_offset = 0;
		std::memset(g_tprj_blk, 0, sizeof(g_tprj_blk));
	}
#else
	// 🚨-DP2_THREAT_PROJ 無しのビルドで射影 net を読んだら、 上の分岐で既に警告している。
	(void) want_tprj;
#endif

#if defined(P2_MOB)
	if (want_mob && mob_tmp.empty()) {
		sync_cout << "info string P2NNUE: arch has MOB marker but P2MOB section missing" << sync_endl;
		return false;
	}
	if (!mob_tmp.empty()) {
		for (int b = 0; b < NUM_BUCKETS; ++b) g_mob_w[b] = (std::int32_t) mob_tmp[b];
		// 🚨sync_cout はロックを取り sync_endl で離すマクロ。 ループ内で複数回呼ぶと
		//   **2 回目で自分のロック待ちになりデッドロックする** (2026-09-29 に踏んだ)。
		//   文字列を先に組み立ててから 1 回だけ出す。
		std::string ws;
		for (int b = 0; b < NUM_BUCKETS; ++b) { ws += " "; ws += std::to_string(g_mob_w[b]); }
		sync_cout << "info string P2NNUE: MOB loaded (w =" << ws << ")" << sync_endl;
	} else {
		std::memset(g_mob_w, 0, sizeof(g_mob_w));
	}
#else
	// 🚨-DP2_MOB 無しのビルドで MOB net を読んだら、 上の分岐で既に警告している。
	(void) want_mob;
#endif

#if defined(P2_ENDGAME_TEMPO)
	if (want_endgame_tempo && endgame_tempo_tmp.empty()) {
		sync_cout << "info string P2NNUE: arch has EGT marker but P2EGTMP section missing" << sync_endl;
		return false;
	}
	if (!endgame_tempo_tmp.empty()) {
		std::size_t p = 0;
		std::size_t nz = 0;
		std::int32_t amax = 0;
		for (int bucket = 0; bucket < EndgameTempo::BUCKET_NB; ++bucket)
			for (int color = 0; color < 2; ++color)
				for (int cls = 0; cls < EndgameTempo::CLASS_NB; ++cls) {
					const std::int32_t value = endgame_tempo_tmp[p++];
					g_endgame_tempo_w[color][cls][bucket] = value;
					const std::int32_t a = value < 0 ? -value : value;
					if (a != 0) ++nz;
					amax = (std::max)(amax, a);
				}
		g_endgame_tempo_loaded = true;
		sync_cout << "info string P2NNUE: EGT weight loaded (1024 i16, nonzero=" << nz
		          << " absmax=" << amax << ")" << sync_endl;
	} else if (g_endgame_tempo_loaded) {
		std::memset(g_endgame_tempo_w, 0, sizeof(g_endgame_tempo_w));
		g_endgame_tempo_loaded = false;
		sync_cout << "info string P2NNUE: EGT weights reset" << sync_endl;
	}
#endif

#if defined(P2_FT_I8)
	build_ft_i8(*net);
#endif
	g_l1_main_shift = l1_main_shift;
	sync_cout << "info string P2NNUE L1 main scale: " << (l1_main_shift == 7 ? 128 : 64)
	          << " (shift " << l1_main_shift << ")" << sync_endl;
	g_net = std::move(net);
	++g_net_generation;  // accumulator cache(finnyç¹ãEã»ç¹æ¶ÎEç¹§è²ãEç¹§E¹ç¹ï½¬ç¹ãEã©ç¸ºE§è¾ï½¡èï½¹è¹ã»
#if defined(P2_FABLE_PROBE) && !defined(P2_FT_I8)
	{   // fable 2026-11-05: FT 重みの値域 (i8 行に縮められるか)。 行ごとの max|w| の分布も出す。
		const std::size_t rows = (std::size_t)HALFKA_DIM;
		std::uint64_t c127 = 0, c255 = 0, c511 = 0, c1023 = 0, c2047 = 0, cbig = 0, rows_le127 = 0, rows_le255 = 0;
		int gmax = 0;
		for (std::size_t f = 0; f < rows; ++f) {
			const std::int16_t* w = ft_row(*g_net, f);
			int rmax = 0;
			for (int k = 0; k < FT_DIM; ++k) {
				int a = w[k] < 0 ? -(int)w[k] : (int)w[k];
				if (a > rmax) rmax = a;
				if (a <= 127) ++c127; else if (a <= 255) ++c255; else if (a <= 511) ++c511; else if (a <= 1023) ++c1023; else if (a <= 2047) ++c2047; else ++cbig;
			}
			if (rmax <= 127) ++rows_le127;
			if (rmax <= 255) ++rows_le255;
			if (rmax > gmax) gmax = rmax;
		}
		int bmax = 0; for (int k = 0; k < FT_DIM; ++k) { int a = g_net->ft_bias[k] < 0 ? -(int)g_net->ft_bias[k] : (int)g_net->ft_bias[k]; if (a > bmax) bmax = a; }
		sync_cout << "info string FABLEFT rows=" << rows << " max=" << gmax << " bias_max=" << bmax
		          << " le127=" << c127 << " le255=" << c255 << " le511=" << c511 << " le1023=" << c1023 << " le2047=" << c2047 << " big=" << cbig
		          << " rows_le127=" << rows_le127 << " rows_le255=" << rows_le255 << sync_endl;
	}
#endif
	sync_cout << "info string P2NNUE loaded: " << path
	          << " psqt=" << g_net->has_psqt << " dual_psqt=" << g_net->has_dual_psqt
	          << " dual_cross=" << g_net->has_dual_cross << " pairshape=" << g_net->has_pairshape
	          << " aux=" << g_net->has_aux
	          << sync_endl;
	return true;
}

bool load_progress(const std::string& path) {
	std::ifstream ifs(path, std::ios::binary);
	if (!ifs) { sync_cout << "info string P2NNUE: cannot open progress " << path << sync_endl; return false; }
	std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(ifs)),
	                              std::istreambuf_iterator<char>());
	if (buf.size() != (std::size_t)PROG_W * sizeof(double)) {
		sync_cout << "info string P2NNUE: progress size mismatch " << buf.size() << sync_endl; return false;
	}
	g_progress.resize(PROG_W);
	for (int i = 0; i < PROG_W; ++i) {
		double d; std::memcpy(&d, &buf[(std::size_t)i * 8], 8);
		g_progress[i] = (float)d;
	}
	g_progress_loaded = true;
	sync_cout << "info string P2NNUE progress loaded: " << path << sync_endl;
	return true;
}

} // namespace YaneuraOu::Eval::P2NNUE

// ===========================================================================
// Eval:: èï½¬é«¢ã»API
// ===========================================================================
namespace YaneuraOu {

namespace {
	OptionsMap* g_options = nullptr;
	std::string g_eval_file = "nn.bin";   // 11-16 YO と同じ既定名 (旧 "q_kda10_A.bin")
	std::string g_eval_dir  = "eval";
	// ★ option の既定値では callback が呼ばれないため、こちらも add_options() の既定と同じ値にすること。
	std::string g_progress_path = "progress.bin";   // 11-16 EvalDir 内の固定名 (旧 "progress/progress.bin" = exe 基準)
#if defined(P2_KH)
	std::string g_kh_path = "kh.bin";   // KHFile (空 = 無効)。 11-16 EvalDir 内の固定名を既定に (旧 空)
#endif
#if defined(P2_OPN)
	std::string g_opn_path = "opn.bin";  // OPNFile (空 = 無効 = 従来と完全一致)。 11-16 EvalDir 内の固定名を既定に (旧 空)
#endif
	bool g_need_load = true;
}

namespace Eval {

// lazy-head ç¹è¼Îç¹§E°(è¬ï½¢éï½¢ç¸ºææ¨ç¹å¼±ã»ç¹çã»è¿E£ç¸ºE§ !PvNode ç¹§åµãç¹ãEã¨)ç¸²ã»
thread_local bool p2_lazy_head = false;
// lazy eval(margin)éï½¨ç¸ºE®è¬ï½¢éï½¢é¯ã»cp)ç¸²ãç¾éï½¢ç¸ºææ¨ç¹å¼±ã»ç¹çã»è¿E£ç¸ºE§ç¹§E»ç¹ãEã¨ç¸²ã»
thread_local int p2_win_a = -32000;
thread_local int p2_win_b =  32000;
// df-pn é§E°ç¸ºE¿ç¹å±Îç¹ï½¼ç¹ãErustè²Eä¾ : per-nodeè·æ¢§æprobe)ç¸²ESIç¹§Eªç¹å±ãç¹ï½§ç¹ï½³ç¸ºE§éªE­è³å£¹Âã»
int  p2_mate_solver_hash_mb = 4;   // rustè·æªï½­ã»4MB df-pn TT)
// ★非同期 df-pn 詰みsolver(別スレッド版, rust mate_solver_thread.rs 移植)。
//   既定 false = 何も起動しないので現行バイナリと bit-exact。
bool p2_mate_solver_async = false;
int  p2_mate_solver_async_max_nodes = 200000;
// NMP verification search(V9.30/rusté¶E¸è ãEç¸²ãéè³å¹Nç¸²EFFç¸ºE§V9.40é¶E¸è ãEè®æE½¨E¼ç¸ºEªç¸ºã»ç¸²ã»
bool p2_nmp_verification = true;
// NMP が「勝ち(詰み)」スコアで beta を超えたときに beta へ丸めて即 cutoff するか(USI: NmpWinCutoff)。
// NAGISA V3.1 準拠。既定 false = 現行バイナリと bit-exact。
bool p2_nmp_win_cutoff = false;
// ç¬ãE PvNode TT-eval èå´ï½©ç©ï½¾E¡(YO/SFè²Eä¾ )ç¸²ãEEè¬ï½¡éï½¨(2026-07-28): èE½¾NAGISA node400k48è»Âç¸ºE§
//   reeval=true 41.7% vs false 33.3% = +8.3pt/+62Elo(pairedè¬¾E¹èãE5/è¬ï½ªè¹ã»0, LOS84%)ç¸²ã»
//   èä¼ï½±Âé«±E¢ç¸ºE§NAGISA/Suisho11ç¸ºç½«eeval=trueç¸ºE®è¬EE8i7g)ç¸ºE«è³Âé¾E´ç¸²ãéè³å¾rue=è¬ï½¡éï½¨ç¸²EEPvReeval=falseç¸ºE§è­ï½§è¬åèç¸²ã»
bool p2_pv_reeval = true;
// INFREP: position.cpp で定義 (既定 false = bit-exact)。
extern bool p2_infrep_history;

void add_options(OptionsMap& options, ThreadPool& /*threads*/) {
	g_options = &options;
	options.add("EvalDir", Option("eval", [](const Option& o) {
		g_eval_dir = std::string(o); g_need_load = true; return std::nullopt; }));
	options.add("EvalFile", Option("nn.bin", /* 11-16 YO と同じ (旧 q_kda10_A.bin) */ [](const Option& o) {
		g_eval_file = std::string(o); g_need_load = true; return std::nullopt; }));
	options.add("LS_PROGRESS_COEFF", Option("progress.bin", /* 11-16 EvalDir 内 (旧 progress/progress.bin) */ [](const Option& o) {
		g_progress_path = std::string(o); g_need_load = true; return std::nullopt; }));
	options.add("LS_BUCKET_MODE", Option("progress8kpabs", [](const Option& o) {
		P2NNUE::g_bucket_mode = std::string(o); return std::nullopt; }));
	// 案 B (Regime LayerStack) の下準備。 root で progress sum / progress bucket / V1 を出す。
	//   V1 = 両玉の 8 近傍のうち**空升**への敵の利き数の和 (= p2volm.h の P2Volm::value)。
	//   🚨目的は「board_effect が長い利きを含むか」を実測で確定すること。
	//     ドキュメントは "ある升の利きの数" としか書いておらず、 python 側の再実装を
	//     推測で合わせると net が黙って壊れる (★net_exe_mismatch_silent)。
	//   既定 0 = 何も出さない = bit-exact。
	options.add("P2RegimeDump", Option(0, 0, 1, [](const Option& o) {
		P2NNUE::g_regime_dump = int(o); return std::nullopt; }));
	options.add("P2QSearchThreatScale", Option(256, 0, 256, [](const Option& o) {
		P2NNUE::g_qs_threat_scale = int(o); return std::nullopt; }));
#if defined(P2_KH)
	// KH の H 部 (浮き駒) の重み (--kh-dump の f32)。 空 (<empty>) = 無効。 相対名は <exe のフォルダ>/<EvalDir>/ 基準 (11-16 YO の nn.bin と同じ作法、 旧 exe のフォルダ基準)。 同名 + ".json" も読む。
	//   🚨読めなければ isready でエラーを出して終了する (黙って KH 無しで走らせない)。
	options.add("KHFile", Option("kh.bin", /* 11-16 EvalDir 内 (旧 空) */ [](const Option& o) {
		g_kh_path = std::string(o); g_need_load = true; return std::nullopt; }));
#endif
#if defined(P2_OPN)
	// ---- OPN (序盤帯専用の副 net、 2026-11-14)。 設計: runs/prereg/DESIGN-20261114-OPNINT-FABLE.md 8.1。
	//   OPNFile: 空 (<empty>) = 無効 (従来と完全一致)。 相対名は <exe のフォルダ>/<EvalDir>/ 基準 (11-16 YO の nn.bin と同じ作法、 旧 exe のフォルダ基準)。 読めなければ isready でエラー終了。
	//   🚨既定値では callback が呼ばれないので、 既定は p2nnue_opn.h の変数の初期値と一致させること。
	options.add("OPNFile", Option("opn.bin", /* 11-16 EvalDir 内 (旧 空) */ [](const Option& o) {
		g_opn_path = std::string(o); g_need_load = true; return std::nullopt; }));
	// λ の上限 (0..256 = 0..1)。 0 = 読み込むが使わない (bit-exact の検証用)。 256 未満 = 本体と OPN の混合。
	options.add("OPN_LambdaMax", Option(256, 0, 256, [](const Option& o) {
		P2NNUE::opn::g_lam_max = int(o); P2NNUE::opn::update_derived(); return std::nullopt; }));
	// 移行区間の終わり (進行度ロジット × 1000)。 −1099 = bucket 1|2 の境 (−1.0986) の内側。 11-18: 上限を −1099 にした (OPN は bucket 0〜1 だけ、 旧上限 −511)。
	options.add("OPN_ProgHi", Option(-1299, /* 11-16 既定を対局時の値へ (旧 -1099) */ -4000, -1099, [](const Option& o) {
		P2NNUE::opn::g_prog_hi = int(o); P2NNUE::opn::update_derived(); return std::nullopt; }));
	// 移行区間の幅 (× 1000)。 0 = 幅 0 の切り替え。
	options.add("OPN_ProgWidth", Option(0, /* 11-16 既定を対局時の値へ (旧 200) */ 0, 2000, [](const Option& o) {
		P2NNUE::opn::g_prog_width = int(o); P2NNUE::opn::update_derived(); return std::nullopt; }));
	// 進行度 bucket 0 / 1 の目盛り k_b (Q10、 1024 = 1.0)。 OPN の raw に掛ける。 11-18: bucket 2 の OPN_Scale2 は削除。
	options.add("OPN_Scale0", Option(1196, /* 11-16 (旧 1024) */ 1, 8192, [](const Option& o) {
		P2NNUE::opn::g_scale[0] = int(o); return std::nullopt; }));
	options.add("OPN_Scale1", Option(1204, /* 11-16 (旧 1024) */ 1, 8192, [](const Option& o) {
		P2NNUE::opn::g_scale[1] = int(o); return std::nullopt; }));
	// OPN_Offset (11-18): OPN の評価に足す定数 [cp、 手番側の視点]。 p2nnue_opn.h の g_offset_cp の注記。 既定 0 = 従来と同じ値。
	options.add("OPN_Offset", Option(0, -200, 200, [](const Option& o) {
		P2NNUE::opn::g_offset_cp = int(o); return std::nullopt; }));
#endif
#if defined(P2_THREAT_L2)
	// ---- TL2 (threat -> L2 入力) の寄与を実行時に定数倍 (2026-10-09)。
	//   t2[] *= gain/256。 既定 256 = 等倍 = 従来と bit-exact (乗算もしない)。
	//   🚨★48: 範囲外の spin はクランプされず破棄されるので 0..4096 と広く取る。
	//   🚨既定値では callback が呼ばれないので、 既定は変数の初期値 (256) と一致させる。
	options.add("ThreatL2Gain", Option(256, 0, 4096, [](const Option& o) {
		P2NNUE::g_tl2_gain = int(o); return std::nullopt; }));
#endif
#if defined(P2_TG)
	// ---- (5) TG (threat による乗法ゲート) の tg[] を実行時に定数倍 (2026-10-10)。
	//   g = clamp(256 + tg * scale/256, 0, 1024)。 既定 256 = 等倍 = TG 無しと bit-exact
	//   (既定では乗算を 1 回もしない)。
	//   🚨★usi_out_of_range_is_discard: 範囲外の spin はクランプされず破棄されるので、
	//     範囲は設計書どおり 0..1024 にしてある。
	//   🚨option の既定値では callback が呼ばれないので、 既定は変数の初期値 (256) と一致させる。
	options.add("ThreatGateScale", Option(256, 0, 1024, [](const Option& o) {
		P2NNUE::g_tg_scale = int(o); return std::nullopt; }));
#endif
	// INFREP (2026-11-16): 優等/劣等局面の判定を対局の履歴にまたがっても返す。 既定 false = 従来と bit-exact。
	options.add("P2InferiorRepHistory", Option(true, /* 11-16 既定を対局時の値へ (旧 false)。 position.cpp の初期値と一致 */ [](const Option& o) {
		p2_infrep_history = bool(o); return std::nullopt; }));
	options.add("P2NoDirect", Option(true, [](const Option& o) {
		P2NNUE::g_p2_no_direct_option = bool(o); return std::nullopt; }));
	// Direct15 の raw 単位 cap を手動上書き (0 = auto = net の fv_scale から確定)。
	// 再ビルド無しで cap だけ変えた A/B を回すための計測用。
	options.add("P2DirectCapRaw", Option(0, 0, 1000000, [](const Option& o) {
		P2NNUE::g_p2_direct_cap_raw_option = (std::int32_t)int(o); return std::nullopt; }));
	options.add("FV_SCALE", Option(28, 0, 1024, [](const Option& o) {
		P2NNUE::g_fv_scale = int(o); return std::nullopt; }));
#if defined(P2_MATPRICE)
	// 駒割の値付け補正。 実測 612。 🚨★48: 範囲外は破棄されるので範囲を広く取る。
	options.add("MAT_PRICE_ADJ", Option(0, -4000, 4000, [](const Option& o) {
		P2NNUE::g_mat_price_adj = int(o); return std::nullopt; }));
#endif
	// ---- 振り飛車の様式項 (2026-10-05)。 既定 0 = 無効 (本番と bit-exact)
	//   🚨★48: 範囲外の spin はクランプされず破棄されるので範囲を広く取る。
	//   🚨option の既定値では callback が呼ばれないので、 既定は変数の初期値と一致させる。
	// ---- 🎯片側版 (2026-10-22)。 自分の角が**盤上**にある枚数に cp/枚 を付ける。
	//   🚨★usi_out_of_range_is_discard: 範囲外の spin は**クランプされず破棄される**ので広く取る。
	//   🎯`OwnBoardBias` は定数の下駄を消す用。 実測の平均が 0.494/0.504 なので **G/2** を推奨。
#if 0   // 🏁OwnKnightAdv の option (2026-10-28 棄却済みとして無効化)
	// ---- 🎯桂を攻めに使う棋風 (2026-10-22)。 自分の桂が 1 回以上跳ねた枚数に cp/枚。
	//   🎯`OwnKnightAdvBias` は定数の下駄を消す用。 実測の平均が 0.631 なので **G*0.63** を推奨。
	options.add("OwnKnightAdv", Option(0, -512, 512, [](const Option& o) {
		P2NNUE::g_knightadv = int(o); return std::nullopt; }));
	options.add("OwnKnightAdvBias", Option(0, -512, 512, [](const Option& o) {
		P2NNUE::g_knightadv_bias = int(o); return std::nullopt; }));
#endif  // 🏁OwnKnightAdv の option
	options.add("OwnBoardB", Option(0, -512, 512, [](const Option& o) {
		P2NNUE::g_ownboard_b = int(o); return std::nullopt; }));
	options.add("OwnBoardBias", Option(0, -512, 512, [](const Option& o) {
		P2NNUE::g_ownboard_bias = int(o); return std::nullopt; }));
	// ---- 🎯片側項 K1 / K2 (2026-10-25 / fable 設計)。 既定 0 で bit-exact。
	//   🎯`*Bias` は定数の下駄を消す用。 実測の平均は K1 **1.06 枚** / K2 **2.09 枚**。
	//   🚨★usi_out_of_range_is_discard: 範囲外の spin は**クランプされず破棄される**ので広く取る。
	options.add("OwnKingDefGS", Option(0, -512, 512, [](const Option& o) {
		P2NNUE::g_kdef = int(o); return std::nullopt; }));
	options.add("OwnKingDefGSBias", Option(0, -512, 512, [](const Option& o) {
		P2NNUE::g_kdef_bias = int(o); return std::nullopt; }));
#if 0   // 🏁K2 の option (2026-10-28 棄却済みとして無効化)
	options.add("OwnCampGS", Option(0, -512, 512, [](const Option& o) {
		P2NNUE::g_campgs = int(o); return std::nullopt; }));
	options.add("OwnCampGSBias", Option(0, -512, 512, [](const Option& o) {
		P2NNUE::g_campgs_bias = int(o); return std::nullopt; }));
#endif  // 🏁K2 の option
#if 0   // 🏁K★ + MatCorr の option (2026-10-28 棄却済みとして無効化)
	// ---- 🎯K★ (2026-10-25 / fable 設計)。 **root でだけ指す手を選び直す**。
	//   探索も評価も変えないので、 損は 1 手あたり Margin 以下に上から抑えられる。
	//   🚨既定 0 で bit-exact。 単位は **Value** (cp ではない / net 依存の換算を避ける)。
	options.add("RootStyleMargin", Option(0, 0, 2000, [](const Option& o) {
		P2NNUE::g_rootstyle_margin = int(o); return std::nullopt; }));
	options.add("RootStyleKind", Option(1, 1, 2, [](const Option& o) {
		P2NNUE::g_rootstyle_kind = int(o); return std::nullopt; }));
	// ---- 駒種ごとの値付け補正 (2026-10-21)。 **棋風のてこ**。 単位 cp/枚、 既定 0 で bit-exact。
	//   🚨★usi_out_of_range_is_discard: 範囲外の spin は**クランプされず破棄される**ので広めに取る。
	options.add("MatCorrP", Option(0, -512, 512, [](const Option& o) {   // 歩
		P2NNUE::g_matcorr[PAWN] = int(o); P2NNUE::matcorr_refresh(); return std::nullopt; }));
	options.add("MatCorrL", Option(0, -512, 512, [](const Option& o) {   // 香
		P2NNUE::g_matcorr[LANCE] = int(o); P2NNUE::matcorr_refresh(); return std::nullopt; }));
	options.add("MatCorrN", Option(0, -512, 512, [](const Option& o) {   // 桂
		P2NNUE::g_matcorr[KNIGHT] = int(o); P2NNUE::matcorr_refresh(); return std::nullopt; }));
	options.add("MatCorrS", Option(0, -512, 512, [](const Option& o) {   // 銀
		P2NNUE::g_matcorr[SILVER] = int(o); P2NNUE::matcorr_refresh(); return std::nullopt; }));
	options.add("MatCorrB", Option(0, -512, 512, [](const Option& o) {   // 角
		P2NNUE::g_matcorr[BISHOP] = int(o); P2NNUE::matcorr_refresh(); return std::nullopt; }));
	options.add("MatCorrR", Option(0, -512, 512, [](const Option& o) {   // 飛
		P2NNUE::g_matcorr[ROOK] = int(o); P2NNUE::matcorr_refresh(); return std::nullopt; }));
	options.add("MatCorrG", Option(0, -512, 512, [](const Option& o) {   // 金
		P2NNUE::g_matcorr[GOLD] = int(o); P2NNUE::matcorr_refresh(); return std::nullopt; }));
	options.add("MatCorrPP", Option(0, -512, 512, [](const Option& o) {   // と
		P2NNUE::g_matcorr[PRO_PAWN] = int(o); P2NNUE::matcorr_refresh(); return std::nullopt; }));
	options.add("MatCorrPL", Option(0, -512, 512, [](const Option& o) {   // 成香
		P2NNUE::g_matcorr[PRO_LANCE] = int(o); P2NNUE::matcorr_refresh(); return std::nullopt; }));
	options.add("MatCorrPN", Option(0, -512, 512, [](const Option& o) {   // 成桂
		P2NNUE::g_matcorr[PRO_KNIGHT] = int(o); P2NNUE::matcorr_refresh(); return std::nullopt; }));
	options.add("MatCorrPS", Option(0, -512, 512, [](const Option& o) {   // 成銀
		P2NNUE::g_matcorr[PRO_SILVER] = int(o); P2NNUE::matcorr_refresh(); return std::nullopt; }));
	options.add("MatCorrPB", Option(0, -512, 512, [](const Option& o) {   // 馬
		P2NNUE::g_matcorr[HORSE] = int(o); P2NNUE::matcorr_refresh(); return std::nullopt; }));
	options.add("MatCorrPR", Option(0, -512, 512, [](const Option& o) {   // 龍
		P2NNUE::g_matcorr[DRAGON] = int(o); P2NNUE::matcorr_refresh(); return std::nullopt; }));
#endif  // 🏁K★ + MatCorr の option
	options.add("FuribishaBonus", Option(0, -1024, 1024, [](const Option& o) {
		P2NNUE::g_furi_bonus = int(o); return std::nullopt; }));
	options.add("FuribishaFadeStart", Option(30, 0, 512, [](const Option& o) {
		P2NNUE::g_furi_fade_start = int(o); return std::nullopt; }));
	options.add("FuribishaFadeEnd", Option(60, 0, 512, [](const Option& o) {
		P2NNUE::g_furi_fade_end = int(o); return std::nullopt; }));
	// ---- 右玉の様式項 (2026-10-05)。 既定 0 = 無効 (本番と bit-exact)
	options.add("MigigyokuBonus", Option(0, -1024, 1024, [](const Option& o) {
		P2NNUE::g_migi_bonus = int(o); return std::nullopt; }));
	options.add("MigigyokuFadeStart", Option(30, 0, 512, [](const Option& o) {
		P2NNUE::g_migi_fade_start = int(o); return std::nullopt; }));
	options.add("MigigyokuFadeEnd", Option(70, 0, 512, [](const Option& o) {
		P2NNUE::g_migi_fade_end = int(o); return std::nullopt; }));
	options.add("MigigyokuKakugawari", Option(1, 0, 1, [](const Option& o) {
		P2NNUE::g_migi_kakugawari = int(o); return std::nullopt; }));
	// 🚨後手の右玉に乗せる G を別に持つ (2026-10-18)。 **-9999 = 先手と同値** (従来と bit-exact)。
	//   floodgate の局単位・engine 別の実測で、 右玉の値段は先手 +2〜+19.7pt /
	//   後手 -0.5〜-38.7pt と**符号が割れる**。 反対称のままだと損な側も押してしまう。
	//   🚨範囲外の spin は**クランプされず破棄される** (★usi_out_of_range_is_discard) ので
	//     番兵 -9999 を min に入れてある。
	options.add("MigigyokuBonusWhite", Option(-9999, -9999, 1024, [](const Option& o) {
		P2NNUE::g_migi_bonus_white = int(o); return std::nullopt; }));
	// 相掛かり角交換右玉の下駄 (2026-10-13)。 0 = 無効 (従来と bit-exact)。
	//   発火は ①双方が角を手駒 ②2 筋と 8 筋に歩が 1 枚も無い ③片方だけ右玉 ④ply < FadeEnd。
	options.add("AigakariMigigyokuBonus", Option(0, -1024, 1024, [](const Option& o) {
		P2NNUE::g_aiga_bonus = int(o); return std::nullopt; }));
	options.add("AigakariMigigyokuFadeStart", Option(30, 0, 512, [](const Option& o) {
		P2NNUE::g_aiga_fade_start = int(o); return std::nullopt; }));
	options.add("AigakariMigigyokuFadeEnd", Option(70, 0, 512, [](const Option& o) {
		P2NNUE::g_aiga_fade_end = int(o); return std::nullopt; }));
	// 後手の相掛かり右玉に乗せる G。 **-9999 = 先手と同値** (従来と bit-exact)。
	options.add("AigakariMigigyokuBonusWhite", Option(-9999, -9999, 1024, [](const Option& o) {
		P2NNUE::g_aiga_bonus_white = int(o); return std::nullopt; }));
	// ---- 囲いの様式項 (2026-10-06)。 既定 0 = 無効 (本番と bit-exact)
	options.add("CastleBonus", Option(0, -1024, 1024, [](const Option& o) {
		P2NNUE::g_castle_bonus = int(o); return std::nullopt; }));
	options.add("CastleShape", Option(8766, 0, 19682, [](const Option& o) {
		P2NNUE::g_castle_shape = int(o); return std::nullopt; }));
	options.add("CastleFadeStart", Option(20, 0, 512, [](const Option& o) {
		P2NNUE::g_castle_fade_start = int(o); return std::nullopt; }));
	options.add("CastleFadeEnd", Option(90, 0, 512, [](const Option& o) {
		P2NNUE::g_castle_fade_end = int(o); return std::nullopt; }));
	// èæEåª FV_SCALE(è»ã»èèE¾é²ã»: è åEä¹¢ç¸ºE¯èE¸ï¼E¸ºã»FV(éE³¨EèE²ç¸ºã»ç¸²â½E¸E­é¶E¤èï½¥é«¯éªã» FV_SCALEç¸²ã»
	//   FV_SCALE_OPENING==FV_SCALE(è­ï½¢è³ã»20==20)ç¸ºEªç¹§çï½¾æºæç¸ºE®è³å£½çè¬åèç¸ºE¨ bit-exactç¸²ã»
	//   PROG_LO/HI ç¸ºE¯ progress sum(ç«ãE2.0ç¸²ã»2.0)ç¸ºE® 1/100 èåE½½éªã»é«¢E¾èï½¤ç¸²ã»
	options.add("FV_SCALE_OPENING", Option(26, 0, 1024, /* 11-16 既定を対局時の値へ (旧 0) */ [](const Option& o) {
		P2NNUE::g_fv_scale_opening = int(o); return std::nullopt; }));
	options.add("FV_SCALE_PROG_LO", Option(-110, -400, 400, [](const Option& o) {
		P2NNUE::g_fv_prog_lo = int(o); return std::nullopt; }));
	options.add("FV_SCALE_PROG_HI", Option(0, -400, 400, [](const Option& o) {
		P2NNUE::g_fv_prog_hi = int(o); return std::nullopt; }));
	// ç¬ãEç¹§E¢ç¹ï½³ç¹§E«ç¹ï½¼èæEåªFV: é¨ãä¹¢ç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½«(0=è¾ï½¡èï½¹=è³E­é¶E¤èï½¤ç¸ºE«éï½½è èãè æºæEç¹§E¢ç¹ï½³ç¹§E«ç¹ï½¼ç¸ºE¨ bit-exact)ç¸²ã»0 ç¸ºE§è³E­é¶E¤ç«å ¤Eµãä¹¢ revertç¸²ã»
	//   è åEä¹¢20ç«åE¸E­é¶E¤(FV_SCALE)ç«å ¤Eµãä¹¢(FV_SCALE_ENDGAME)ç¸²E°ver-optimism ç¹§åE½¸E­é¶E¤ç¸ºE°ç¸ºå¤§E¼E·ç¸ºä¸ã ç¹ï½³ç¹å±E é¨ãä¹¢ç¸ºE¯è±ï½ºé¹Âé©¥å´ï½¦æ¶Âã»
	options.add("FV_SCALE_ENDGAME", Option(28, 0, 1024, /* 11-16 既定を対局時の値へ (旧 0) */ [](const Option& o) {
		P2NNUE::g_fv_scale_endgame = int(o); return std::nullopt; }));
	// 2026-09-13: bucket 別 FV_SCALE。 "28,28,30,28,28,28,28,28" のようにカンマ区切りで 8 個。
	//   0 または省略した bucket は従来の 3 点補間に従う。 空文字 (既定) なら従来と完全に同じ。
	// 2026-09-13: SPSA で調整できるよう、 bucket 別 FV を 1 個ずつの整数 option にも出す。
	//   0 = 従来の 3 点補間に従う (既定)。 FV_SCALE_BUCKETS と両方指定した場合は後から設定したほうが勝つ。
	// 2026-11-19 FVNODE: true なら FV_SCALE_B0..B7 を**探索中の各局面の** bucket で使う (根で固定しない)。 既定 false = 従来と bit-exact。
	options.add("FV_SCALE_PER_NODE", Option(false, [](const Option& o) {
		P2NNUE::g_fv_per_node = bool(o); return std::nullopt; }));
	for (int b = 0; b < 8; ++b) {
		options.add("FV_SCALE_B" + std::to_string(b), Option(0, 0, 1024, [b](const Option& o) {
			P2NNUE::g_fv_scale_bucket[b] = int(o);
			P2NNUE::g_fv_bucket_any = false;
			for (int k = 0; k < 8; ++k) if (P2NNUE::g_fv_scale_bucket[k] > 0) P2NNUE::g_fv_bucket_any = true;
			return std::nullopt; }));
	}
#if defined(P2_BUCKET_OFFSET)
	// P2_BUCKET_OFFSET (11-19): bucket ごとの片寄りの補正 [cp、 手番側]。 既定 0 = 従来と bit-exact (g_bucket_off_cp の注記)。
	for (int b = 0; b < 8; ++b) {
		options.add("P2_BUCKET_OFFSET_B" + std::to_string(b), Option(0, -200, 200, [b](const Option& o) {
			P2NNUE::g_bucket_off_cp[b] = int(o);
			P2NNUE::select_bucket_off(P2NNUE::g_bucket_off_use_sente);
			return std::nullopt; }));
	}
	// SIDEOFF2 (11-21 案 A): 我々が優勢なら後手でも SENTE 表 [表示 cp]。 既定 0 = 無効 (g_sente_switch_cp の注記)。
	options.add("P2_SENTE_TABLE_SWITCH_CP", Option(0, 0, 100000, [](const Option& o) {
		P2NNUE::g_sente_switch_cp = int(o); return std::nullopt; }));
	options.add("P2_SENTE_TABLE_SWITCH_LOG", Option(false, [](const Option& o) {
		P2NNUE::g_sente_switch_log = bool(o); return std::nullopt; }));
	// SIDEOFF (11-21): 先手番 root 用の表。 既定 -999 (= BUCKET_OFF_UNSET) は未設定で、 その bucket は P2_BUCKET_OFFSET_Bb を使う。
	for (int b = 0; b < 8; ++b) {
		options.add("P2_BUCKET_OFFSET_SENTE_B" + std::to_string(b),
		            Option(P2NNUE::BUCKET_OFF_UNSET, P2NNUE::BUCKET_OFF_UNSET, 200, [b](const Option& o) {
			P2NNUE::g_bucket_off_sente[b] = int(o);
			P2NNUE::select_bucket_off(P2NNUE::g_bucket_off_use_sente);
			return std::nullopt; }));
	}
#endif
	options.add("FV_SCALE_BUCKETS", Option("", [](const Option& o) {
		std::string v = std::string(o);
		for (int i = 0; i < 8; ++i) P2NNUE::g_fv_scale_bucket[i] = 0;
		P2NNUE::g_fv_bucket_any = false;
		int idx = 0; std::string cur;
		for (size_t i = 0; i <= v.size() && idx < 8; ++i) {
			if (i == v.size() || v[i] == ',') {
				if (!cur.empty()) {
					int x = std::atoi(cur.c_str());
					if (x > 0 && x <= 1024) { P2NNUE::g_fv_scale_bucket[idx] = x; P2NNUE::g_fv_bucket_any = true; }
				}
				cur.clear(); ++idx;
			} else if (v[i] != ' ') cur += v[i];
		}
		return std::nullopt; }));
	// è³E­é¶E¤ç«å ¤Eµãä¹¢ é¬æºä¿£ç¸ºE®è³é¡ï½«E¯(progress sum ç¸ºE® 1/100)ç¸²ãE¼E¹§çï½¶ã»ãè³æãEç¸ºE«é¨ãä¹¢ç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½«ç¸²ã»
	// 2026-09-14 時間管理の止め時係数。 既定値で従来と bit-exact。
	//   use_time_management() のときだけ効くので、 go nodes の測定には無影響。
	options.add("SPSA_TM_FE_PREV_MULT",  Option(SPSA::TM_FE_PREV_MULT,  0, 5000, [](const Option& o) {
		SPSA::TM_FE_PREV_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_TM_FE_ITER_MULT",  Option(SPSA::TM_FE_ITER_MULT,  0, 3000, [](const Option& o) {
		SPSA::TM_FE_ITER_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_TM_INSTAB_MULT",   Option(SPSA::TM_INSTAB_MULT,   0, 40000, [](const Option& o) {
		SPSA::TM_INSTAB_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_TM_STABLE_CENTER", Option(SPSA::TM_STABLE_CENTER, 0, 4000, [](const Option& o) {
		SPSA::TM_STABLE_CENTER = int(o); return std::nullopt; }));
	options.add("SPSA_TM_EFFORT_MULT",   Option(SPSA::TM_EFFORT_MULT,   100, 1000, [](const Option& o) {
		SPSA::TM_EFFORT_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_TM_INCDEPTH_PCT",  Option(SPSA::TM_INCDEPTH_PCT,  100, 1000, [](const Option& o) {
		SPSA::TM_INCDEPTH_PCT = int(o); return std::nullopt; }));
	options.add("FV_SCALE_PROG_END", Option(130, -400, 400, [](const Option& o) {
		P2NNUE::g_fv_prog_end = int(o); return std::nullopt; }));
	// root eval(cp)ç¸ºå¾ï¼E¸ºE®èï½¤èï½¥è³ç¿«âç¹§è¾ÂE²é¦æï½ºE¦ç¸ºE«é«¢E¢ç¹§ä¸ï½ç¸ºå¤ï½µãä¹¢ç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½«(=èE½ªèï½¢è­ãã» revert)ç¸²ãéè³ã»2000=è¾ï½¡èï½¹ç¸²ã»
	options.add("FV_SCALE_ENDGAME_EVAL", Option(32000, -32000, 32000, [](const Option& o) {
		P2NNUE::g_fv_endgame_eval = int(o); return std::nullopt; }));
	// é¦E¨éï½ºèEçç¹ï½ªç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½«èï½ºè²EEFVç¸²ã»=OFF(è æ¢§å©¿è åé¤)ç¸²ã»0 ç¸ºE§ score cp ç¹§è²æ¸è²E¶ãç¹§E±ç¹ï½¼ç¹ï½«ç¸ºE«é¨E±è³Âé¦E¨éï½ºç¸²ã»
	options.add("FV_SCALE_DISPLAY", Option(28, 0, 1024, /* 11-16 既定を対局時の値へ (旧 0) */ [](const Option& o) {
		P2NNUE::g_fv_display = int(o); return std::nullopt; }));
	// lazy head é§ç©ï½¾E¡ç¹ï½¢ç¹ï½¼ç¹ãEEloè±ç¢ï½¼ã»çE: 0=fullè´E³èEE/ 1=lazy-head(é«±é¸Véµâ«æE / 2=margin lazyç¸²ã»
	options.add("P2LazyMode", Option(1, 0, 2, [](const Option& o) {
		P2NNUE::g_lazy_mode = int(o); return std::nullopt; }));
	options.add("P2LazyMargin", Option(96, 0, 4096, [](const Option& o) {
		P2NNUE::g_lazy_margin = int(o); return std::nullopt; }));
	// lazy-threat(éå°E¼E¼): 0=off(threat è³æãEé§ç©ï½¾E¡) / >0=margin M(cp)ç¸²E£ase èï½® eval ç¸ºã»
	// è¬ï½¢éï½¢é¯ã»[alpha-M, beta+M] ç¸ºE®èæ¶ã» far node ç¸ºE§ threat èEEE¸å¼±Eè³E¸ç¸ºæâEéµâ«æENPSè·å°E¸ç¿«Âã»æ¼ bit-exact)ç¸²ã»
	options.add("P2LazyThreatMargin", Option(0, 0, 4096, [](const Option& o) {
		P2NNUE::g_lazy_threat_margin = int(o); return std::nullopt; }));
	// MateSolverHashMB: df-pn の作業メモリ[MB]。非同期版 UseMateSolverAsync が使う。
	options.add("MateSolverHashMB", Option(4, 1, 4096, [](const Option& o) {
		p2_mate_solver_hash_mb = int(o); return std::nullopt; }));
	// 非同期 df-pn 詰みsolver(別スレッド版)。
	// UseMateSolverAsync: 既定 false。true で go のたびに root から別スレッドで df-pn を回す。
	// ★Threads の値とは独立に起動する(Threads=1 でも 1 コア余分に使う)。
	// MateSolverAsyncMaxNodes: 1 go あたりのノード上限。
	// df-pn の作業メモリは MateSolverHashMB を流用する。
	options.add("UseMateSolverAsync", Option(false, [](const Option& o) {
		p2_mate_solver_async = bool(o); return std::nullopt; }));
	options.add("MateSolverAsyncMaxNodes", Option(200000, 10000, 100000000, [](const Option& o) {
		p2_mate_solver_async_max_nodes = int(o); return std::nullopt; }));
	// NMP verification search: true=V9.30/rusté¶E¸è ãEè®æE½¨E¼ç¸ºãE½E / false=V9.40é¶E¸è ãEè®æE½¨E¼ç¸ºEªç¸ºã»ç¸²ã»
	options.add("NmpVerification", Option(true, [](const Option& o) {
		p2_nmp_verification = bool(o); return std::nullopt; }));
	// NmpWinCutoff: true=NAGISA V3.1 相当。null move が詰みスコアで beta を超えたら beta で cutoff。
	//               false(既定)=本家 9.40/9.60 相当。証明されていない詰みスコアでは cutoff しない。
	options.add("NmpWinCutoff", Option(false, [](const Option& o) {
		p2_nmp_win_cutoff = bool(o); return std::nullopt; }));
	// ç¬ãE PvNode ç¸ºE§ TT-eval ç¹§åE½¿E¡éï½¨ç¸ºå¸âE evaluate() èå´ï½©ç©ï½¾E¡(YOè³æ»ï½¸E¬R70éãEç¸²ãEEè¬ï½¡éï½¨ç¹§ã»â´è­ï½¢è³å¾rueç¸²ã»
	options.add("P2PvReeval", Option(true, [](const Option& o) {
		p2_pv_reeval = bool(o); return std::nullopt; }));

	// ---- SPSA tuning parameters (rust rshogi-usi è²Eä¾ , Phase 1) ----
	// ç¬ãEefault ç¸ºE¯ C++(SF16éï½»ç¹§ã»ã»ç¸ºã»Eéåºç¾éï½¢)ç¸ºE®è¿´E¾é¦æÂE¤(spsa_params.h)ç¸²E´etoptionè­ãã»ç¸ºE¿è¬åèèçå§ç¸²ã»
	using namespace YaneuraOu::SPSA;
	options.add("SPSA_LMR_TABLE_COEFF", Option(LMR_TABLE_COEFF, 1024, 8192, [](const Option& o) {
		SPSA::LMR_TABLE_COEFF = int(o); return std::nullopt; }));
	// è åEä¹¢ regime(FV_OPENING ç¹§åE½½E¿ç¸ºã»E½å²å¶EèEçç¸ºE® LMR è«ãçEç¸²ãE½¸E­é¶E¤èï½¤=è³ç¿«ã» SPSA_LMR_TABLE_COEFFç¸²ã»
	// FV ç¸ºE¨è·å¾¡E¸Âç¸ºE®é¬æºä¿£èï£°èEEt(FV_SCALE_PROG_LO/HI)ç¸ºE§é¨¾E£èè¼ÂEPENING==è³E­é¶E¤èï½¤ ç¸ºEªç¹§è¥æ¨é¦å¾âE bit-exactç¸²ã»
	options.add("SPSA_LMR_TABLE_COEFF_OPENING", Option(LMR_TABLE_COEFF, 1024, 8192, [](const Option& o) {
		P2NNUE::g_lmr_coeff_opening = int(o); return std::nullopt; }));
	// reduction() の3定数。NAGISA V3.1 は式が完全に同一で値のみ 798/396/1076 (我々 757/218/1200)。
	// 既定は現行値のまま = bit-exact。setoption で NAGISA 値に切り替えて A/B できる。
	options.add("SPSA_LMR_REDUCTION_DELTA_MULT", Option(LMR_REDUCTION_DELTA_MULT, 0, 2048, [](const Option& o) {
		SPSA::LMR_REDUCTION_DELTA_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_REDUCTION_NOIMPROVING", Option(LMR_REDUCTION_NOIMPROVING, 0, 1024, [](const Option& o) {
		SPSA::LMR_REDUCTION_NOIMPROVING = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_REDUCTION_BASE", Option(LMR_REDUCTION_BASE, 0, 4096, [](const Option& o) {
		SPSA::LMR_REDUCTION_BASE = int(o); return std::nullopt; }));
	// Step18 (LMR スキップ時の完全探索) の深さ削減閾値。NAGISA V3.1 は 2725/4752 (我々 3212/4784)。
	options.add("SPSA_LMR_STEP18_THRESHOLD1", Option(LMR_STEP18_THRESHOLD1, 0, 8192, [](const Option& o) {
		SPSA::LMR_STEP18_THRESHOLD1 = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_STEP18_THRESHOLD2", Option(LMR_STEP18_THRESHOLD2, 0, 8192, [](const Option& o) {
		SPSA::LMR_STEP18_THRESHOLD2 = int(o); return std::nullopt; }));
	options.add("SPSA_CHECK_EXT", Option(CHECK_EXT, 0, 1, [](const Option& o) {
		SPSA::CHECK_EXT = int(o); return std::nullopt; }));
	options.add("SPSA_CHECK_EXT_PLY_LIMIT_MULT", Option(CHECK_EXT_PLY_LIMIT_MULT, 1, 8, [](const Option& o) {
		SPSA::CHECK_EXT_PLY_LIMIT_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_STAT_BONUS_DEPTH_MULT", Option(STAT_BONUS_DEPTH_MULT, 0, 2048, [](const Option& o) {
		SPSA::STAT_BONUS_DEPTH_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_STAT_BONUS_OFFSET", Option(STAT_BONUS_OFFSET, -4096, 4096, [](const Option& o) {
		SPSA::STAT_BONUS_OFFSET = int(o); return std::nullopt; }));
	options.add("SPSA_STAT_BONUS_MAX", Option(STAT_BONUS_MAX, 1, 8192, [](const Option& o) {
		SPSA::STAT_BONUS_MAX = int(o); return std::nullopt; }));
	options.add("SPSA_STAT_BONUS_TT_BONUS", Option(STAT_BONUS_TT_BONUS, -4096, 4096, [](const Option& o) {
		SPSA::STAT_BONUS_TT_BONUS = int(o); return std::nullopt; }));
	options.add("SPSA_STATBONUS_PREV_DIV", Option(STATBONUS_PREV_DIV, 0, 256, [](const Option& o) {
		SPSA::STATBONUS_PREV_DIV = int(o); return std::nullopt; }));
	options.add("SPSA_STAT_MALUS_DEPTH_MULT", Option(STAT_MALUS_DEPTH_MULT, 0, 4096, [](const Option& o) {
		SPSA::STAT_MALUS_DEPTH_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_STAT_MALUS_OFFSET", Option(STAT_MALUS_OFFSET, -4096, 4096, [](const Option& o) {
		SPSA::STAT_MALUS_OFFSET = int(o); return std::nullopt; }));
	options.add("SPSA_STAT_MALUS_MAX", Option(STAT_MALUS_MAX, 1, 8192, [](const Option& o) {
		SPSA::STAT_MALUS_MAX = int(o); return std::nullopt; }));
	options.add("SPSA_FUTILITY_MARGIN_BASE", Option(FUTILITY_MARGIN_BASE, 0, 1024, [](const Option& o) {
		SPSA::FUTILITY_MARGIN_BASE = int(o); return std::nullopt; }));
	options.add("SPSA_FUTILITY_MARGIN_TT_BONUS", Option(FUTILITY_MARGIN_TT_BONUS, 0, 512, [](const Option& o) {
		SPSA::FUTILITY_MARGIN_TT_BONUS = int(o); return std::nullopt; }));
	options.add("SPSA_FUTILITY_IMPROVING_SCALE", Option(FUTILITY_IMPROVING_SCALE, 0, 4096, [](const Option& o) {
		SPSA::FUTILITY_IMPROVING_SCALE = int(o); return std::nullopt; }));
	options.add("SPSA_FUTILITY_OPP_WORSENING_SCALE", Option(FUTILITY_OPP_WORSENING_SCALE, 0, 4096, [](const Option& o) {
		SPSA::FUTILITY_OPP_WORSENING_SCALE = int(o); return std::nullopt; }));
	options.add("SPSA_RAZORING_BASE", Option(RAZORING_BASE, 0, 4096, [](const Option& o) {
		SPSA::RAZORING_BASE = int(o); return std::nullopt; }));
	options.add("SPSA_RAZORING_DEPTH2", Option(RAZORING_DEPTH2, 0, 4096, [](const Option& o) {
		SPSA::RAZORING_DEPTH2 = int(o); return std::nullopt; }));
	// RAZV (2026-09-28): V >= この値の局面で razoring を止める。 0 = 無効 (bit-exact)。
	// 🚨範囲外の値は**クランプされず破棄される** (usioption.cpp:216) ので梯子の値を全部含む範囲にする。
	options.add("SPSA_RAZV_VMIN", Option(RAZV_VMIN, 0, 64, [](const Option& o) {
		SPSA::RAZV_VMIN = int(o); return std::nullopt; }));
	// NALLOC (2026-09-28): V bin ごとのノード配分の重み x1024。 既定 1024 = 一様 = bit-exact。
	// 🚨範囲外は破棄される (usioption.cpp:216) ので、 梯子で使う値を全部含む範囲にする。
	options.add("SPSA_NALLOC_W0", Option(NALLOC_W0, 64, 8192, [](const Option& o) {
		SPSA::NALLOC_W0 = int(o); return std::nullopt; }));
	options.add("SPSA_NALLOC_W1", Option(NALLOC_W1, 64, 8192, [](const Option& o) {
		SPSA::NALLOC_W1 = int(o); return std::nullopt; }));
	options.add("SPSA_NALLOC_W2", Option(NALLOC_W2, 64, 8192, [](const Option& o) {
		SPSA::NALLOC_W2 = int(o); return std::nullopt; }));
	options.add("SPSA_NALLOC_W3", Option(NALLOC_W3, 64, 8192, [](const Option& o) {
		SPSA::NALLOC_W3 = int(o); return std::nullopt; }));
	options.add("SPSA_NALLOC_W4", Option(NALLOC_W4, 64, 8192, [](const Option& o) {
		SPSA::NALLOC_W4 = int(o); return std::nullopt; }));
	options.add("SPSA_NALLOC_WMIN", Option(NALLOC_WMIN, 64, 1024, [](const Option& o) {
		SPSA::NALLOC_WMIN = int(o); return std::nullopt; }));
	options.add("SPSA_NALLOC_PLACEBO", Option(NALLOC_PLACEBO, 0, 1, [](const Option& o) {
		SPSA::NALLOC_PLACEBO = int(o); return std::nullopt; }));
	options.add("SPSA_NMP_MARGIN_DEPTH_MULT", Option(NMP_MARGIN_DEPTH_MULT, 0, 256, [](const Option& o) {
		SPSA::NMP_MARGIN_DEPTH_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_NMP_MARGIN_IMPROVING_BONUS", Option(NMP_MARGIN_IMPROVING_BONUS, 0, 1024, [](const Option& o) {
		SPSA::NMP_MARGIN_IMPROVING_BONUS = int(o); return std::nullopt; }));
	options.add("SPSA_NMP_MARGIN_OFFSET", Option(NMP_MARGIN_OFFSET, -4096, 4096, [](const Option& o) {
		SPSA::NMP_MARGIN_OFFSET = int(o); return std::nullopt; }));
	options.add("SPSA_NMP_REDUCTION_BASE", Option(NMP_REDUCTION_BASE, 1, 32, [](const Option& o) {
		SPSA::NMP_REDUCTION_BASE = int(o); return std::nullopt; }));
	options.add("SPSA_NMP_REDUCTION_DEPTH_DIV", Option(NMP_REDUCTION_DEPTH_DIV, 1, 32, [](const Option& o) {
		SPSA::NMP_REDUCTION_DEPTH_DIV = int(o); return std::nullopt; }));

	// ---- SPSA tuning parameters Phase 2 (history / correction history) ----
	// history è»æEè¹éÂE¤(clearè­ãâè¿è´ä¸E
	options.add("SPSA_MAIN_HIST_INIT", Option(MAIN_HIST_INIT, -2048, 2048, [](const Option& o) {
		SPSA::MAIN_HIST_INIT = int(o); return std::nullopt; }));
	options.add("SPSA_CAPTURE_HIST_INIT", Option(CAPTURE_HIST_INIT, -4096, 4096, [](const Option& o) {
		SPSA::CAPTURE_HIST_INIT = int(o); return std::nullopt; }));
	options.add("SPSA_CONT_HIST_INIT", Option(CONT_HIST_INIT, -4096, 4096, [](const Option& o) {
		SPSA::CONT_HIST_INIT = int(o); return std::nullopt; }));
	options.add("SPSA_PAWN_HIST_INIT", Option(PAWN_HIST_INIT, -4096, 4096, [](const Option& o) {
		SPSA::PAWN_HIST_INIT = int(o); return std::nullopt; }));
	options.add("SPSA_LOW_PLY_HIST_INIT", Option(LOW_PLY_HIST_INIT, -2048, 2048, [](const Option& o) {
		SPSA::LOW_PLY_HIST_INIT = int(o); return std::nullopt; }));
	// correction history è·è²E®é´ã¾ç¸ºE¿
	options.add("SPSA_CORR_PCV_WEIGHT", Option(CORR_PCV_WEIGHT, 0, 32768, [](const Option& o) {
		SPSA::CORR_PCV_WEIGHT = int(o); return std::nullopt; }));
	options.add("SPSA_CORR_MICV_WEIGHT", Option(CORR_MICV_WEIGHT, 0, 32768, [](const Option& o) {
		SPSA::CORR_MICV_WEIGHT = int(o); return std::nullopt; }));
	options.add("SPSA_CORR_NONPAWN_WEIGHT", Option(CORR_NONPAWN_WEIGHT, 0, 32768, [](const Option& o) {
		SPSA::CORR_NONPAWN_WEIGHT = int(o); return std::nullopt; }));
	options.add("SPSA_CORR_CNT_WEIGHT", Option(CORR_CNT_WEIGHT, 0, 32768, [](const Option& o) {
		SPSA::CORR_CNT_WEIGHT = int(o); return std::nullopt; }));
	// correction history è­ï½´è­E½°è«ãçE
	options.add("SPSA_CORR_HIST_NONPAWN", Option(CORR_HIST_NONPAWN, 0, 1024, [](const Option& o) {
		SPSA::CORR_HIST_NONPAWN = int(o); return std::nullopt; }));
	options.add("SPSA_CORR_HIST_MINOR", Option(CORR_HIST_MINOR, 0, 1024, [](const Option& o) {
		SPSA::CORR_HIST_MINOR = int(o); return std::nullopt; }));
	options.add("SPSA_CORR_HIST_CONT_SS2", Option(CORR_HIST_CONT_SS2, 0, 1024, [](const Option& o) {
		SPSA::CORR_HIST_CONT_SS2 = int(o); return std::nullopt; }));
	options.add("SPSA_CORR_HIST_CONT_SS4", Option(CORR_HIST_CONT_SS4, 0, 1024, [](const Option& o) {
		SPSA::CORR_HIST_CONT_SS4 = int(o); return std::nullopt; }));

	// ---- SPSA tuning parameters Phase 3 (continuation history / pawnç¹ï½»low-ply multiplier) ----
	options.add("SPSA_CONT_HISTORY_WEIGHT_1", Option(CONT_HISTORY_WEIGHT_1, -4096, 4096, [](const Option& o) {
		SPSA::CONT_HISTORY_WEIGHT_1 = int(o); return std::nullopt; }));
	options.add("SPSA_CONT_HISTORY_WEIGHT_2", Option(CONT_HISTORY_WEIGHT_2, -4096, 4096, [](const Option& o) {
		SPSA::CONT_HISTORY_WEIGHT_2 = int(o); return std::nullopt; }));
	options.add("SPSA_CONT_HISTORY_WEIGHT_3", Option(CONT_HISTORY_WEIGHT_3, -4096, 4096, [](const Option& o) {
		SPSA::CONT_HISTORY_WEIGHT_3 = int(o); return std::nullopt; }));
	options.add("SPSA_CONT_HISTORY_WEIGHT_4", Option(CONT_HISTORY_WEIGHT_4, -4096, 4096, [](const Option& o) {
		SPSA::CONT_HISTORY_WEIGHT_4 = int(o); return std::nullopt; }));
	options.add("SPSA_CONT_HISTORY_WEIGHT_5", Option(CONT_HISTORY_WEIGHT_5, -4096, 4096, [](const Option& o) {
		SPSA::CONT_HISTORY_WEIGHT_5 = int(o); return std::nullopt; }));
	options.add("SPSA_CONT_HISTORY_WEIGHT_6", Option(CONT_HISTORY_WEIGHT_6, -4096, 4096, [](const Option& o) {
		SPSA::CONT_HISTORY_WEIGHT_6 = int(o); return std::nullopt; }));
	options.add("SPSA_CONT_HISTORY_NEAR_PLY_OFFSET", Option(CONT_HISTORY_NEAR_PLY_OFFSET, -1024, 1024, [](const Option& o) {
		SPSA::CONT_HISTORY_NEAR_PLY_OFFSET = int(o); return std::nullopt; }));
	options.add("SPSA_CONT_HISTORY_MULTIPLIER", Option(CONT_HISTORY_MULTIPLIER, 0, 2048, [](const Option& o) {
		SPSA::CONT_HISTORY_MULTIPLIER = int(o); return std::nullopt; }));
	options.add("SPSA_LOW_PLY_HISTORY_MULTIPLIER", Option(LOW_PLY_HISTORY_MULTIPLIER, 0, 2048, [](const Option& o) {
		SPSA::LOW_PLY_HISTORY_MULTIPLIER = int(o); return std::nullopt; }));
	options.add("SPSA_PAWN_HISTORY_POS_MULTIPLIER", Option(PAWN_HISTORY_POS_MULTIPLIER, 0, 2048, [](const Option& o) {
		SPSA::PAWN_HISTORY_POS_MULTIPLIER = int(o); return std::nullopt; }));
	options.add("SPSA_PAWN_HISTORY_NEG_MULTIPLIER", Option(PAWN_HISTORY_NEG_MULTIPLIER, 0, 2048, [](const Option& o) {
		SPSA::PAWN_HISTORY_NEG_MULTIPLIER = int(o); return std::nullopt; }));

	// ---- è­E½¹é©¥æE¬ è¬¨E°(policy index) : quiet move ordering ç¹åEÎ¦ç¹ãEè³æ»E½¨ã» ----
	// POLICY_HINT_SCALE=0(è­ï½¢è³ã»ç¸ºE§è æºæè¬åèç¸ºE¨è³æãEè³Âé¾E´(bit-exact)ç¸²E¦val ç¸ºE«ç¸ºE¯é«±æ¨£æªè³å¼±Âã»
	options.add("POLICY_HINT_SCALE", Option(POLICY_HINT_SCALE, 0, 1024, [](const Option& o) {
		SPSA::POLICY_HINT_SCALE = int(o); return std::nullopt; }));
	options.add("POLICY_DEBUG", Option(POLICY_DEBUG, 0, 1, [](const Option& o) {
		SPSA::POLICY_DEBUG = int(o); return std::nullopt; }));

	// ---- 右玉誘導 (MovePicker) : 既定 0 = 従来と bit-exact。 🚨評価には非関与。
	//   ユーザー指示 2026-10-20「fable 設計そのまま実装で、 下駄は sweep します」。
	//   🚨★usi_out_of_range_is_discard: 範囲外は**黙って破棄**されるので 0..4096 と広く取る
	//     (SPSA が端に寄ったときに無言で既定値へ戻るのを防ぐ)。
	//   1024 で +16384 = 王手ボーナス相当 / 100 で +1600 = POLICY_HINT=100 相当。
	options.add("MIGI_MOVEORDER_SCALE", Option(SPSA::MIGI_MOVEORDER_SCALE, 0, 4096, [](const Option& o) {
		SPSA::MIGI_MOVEORDER_SCALE = int(o); return std::nullopt; }));
	options.add("MIGI_MOVEORDER_DEBUG", Option(SPSA::MIGI_MOVEORDER_DEBUG, 0, 1, [](const Option& o) {
		SPSA::MIGI_MOVEORDER_DEBUG = int(o); return std::nullopt; }));
	// 🚨序盤の下限 (既定 0 = 従来どおり下限なし)。 定跡の出口は実測で中央 ply 23 なので
	//   30 前後にすると「定跡を抜けた後だけ押す」形になる。
	options.add("MIGI_MOVEORDER_PLY_MIN", Option(0, 0, 512, [](const Option& o) {
		P2NNUE::g_migi_mo_ply_min = int(o); return std::nullopt; }));

	// ---- Drop Continuation History : é¬§åé ç¸ºE¡ quiet move ordering ç¹åEÎ¦ç¹ãEè³æ»E½¨ã» ----
	// DROP_CONT_HIST_SCALE=0(è­ï½¢è³ã»ç¸ºE§è æºæè¬åèç¸ºE¨è³æãEè³Âé¾E´(bit-exact)ç¸²E¦val ç¸ºE«ç¸ºE¯é«±æ¨£æªè³å¼±Âã»
	options.add("DROP_CONT_HIST_SCALE", Option(SPSA::DROP_CONT_HIST_SCALE, 0, 1024, [](const Option& o) {
		SPSA::DROP_CONT_HIST_SCALE = int(o); return std::nullopt; }));

	// ---- è¬âEâ é¬§åï½§åºã» Correction History : NNUEé«±ååªé§ç©ï½¾E¡ç¸ºE®éï½»é¨E±è¾E®ç¹§åæç¸ºE¡é¬§åæé¼å»ãé¬æï½­E£(è³æ»E½¨ã» ----
	// CORR_HAND_WEIGHT=0(è­ï½¢è³ã»ç¸ºE§è æºæè¬åèç¸ºE¨è³æãEè³Âé¾E´(bit-exact)ç¸²ENè­ãã» pawné¶E¸è ãE~9536)ç¸²ã»
	options.add("CORR_HAND_WEIGHT", Option(SPSA::CORR_HAND_WEIGHT, 0, 32768, [](const Option& o) {
		SPSA::CORR_HAND_WEIGHT = int(o); return std::nullopt; }));

	// ---- History Confidence LMR : è»E¥è±E½´3è¬ä»ãEç¸ºæéEé¨E¦è¿E·=é¬®åE½¿E¡é¬E½¼ç¸ºE®ç¸ºE¨ç¸ºéªâ¡ç¸ºã»reduction é±E¿è¬¨E´ç¹§è²ï½¼E·ç¹§âï½Eè³æ»E½¨ã» ----
	// HIST_CONFIDENCE_SCALE=0(è­ï½¢è³ã»ç¸ºE§è æºæè¬åèç¸ºE¨è³æãEè³Âé¾E´(bit-exact)ç¸²ENè­ãã»è»æEèåµE£ã»100ç¸²ã»00ç¸²ã»
	// NOTE 2026-10-17: the default is 150 (spsa_params.h:251), NOT 0.
	options.add("HIST_CONFIDENCE_SCALE", Option(SPSA::HIST_CONFIDENCE_SCALE, 0, 4096, [](const Option& o) {
		SPSA::HIST_CONFIDENCE_SCALE = int(o); return std::nullopt; }));

	// ---- Uncertainty-linked LMR (é¨E¬1ç¹ä»£ã£ç¹ãE : é§ç©ï½¾E¡è«¤E¥èçE±Âé«±E¢ç¸ºE®è³Âé©ï½¨ quiet ç¹§ã»plyè ï½±ç¹§âï½Eè³æ»E½¨ã» ----
	// UNCERT_LMR_THRESHOLD=0(è­ï½¢è³ã»ç¸ºE§è æºæè¬åèç¸ºE¨è³æãEè³Âé¾E´(bit-exact)ç¸²ãæEèï½¤ç¸ºE¯è³æ»ï½¸E¬ volatility è»ã»E¸ã»ãè±ï½ºè³å£¹Âã»
	options.add("UNCERT_LMR_THRESHOLD", Option(SPSA::UNCERT_LMR_THRESHOLD, 0, 30000, [](const Option& o) {
		SPSA::UNCERT_LMR_THRESHOLD = int(o); return std::nullopt; }));

	// ---- Threat Continuation History (Phase B) : é¼ã»E¨âµæé¼å»ãç¸ºE® quiet response ç¹§è²ï½­E¦éåEã»èï£°è½¤E¹(è³æ»E½¨ã» ----
	// THREAT_HISTORY_SCALE=0(è­ï½¢è³ã»ç¸ºE§è æºæè¬åèç¸ºE¨è³æãEè³Âé¾E´(bit-exact)ç¸²ã»0 ç¸ºE§è­çæEèï£°éå±ã» history/8 èï½ºè³ã»ç¸²ã»
	options.add("THREAT_HISTORY_SCALE", Option(SPSA::THREAT_HISTORY_SCALE, 0, 1024, [](const Option& o) {
		SPSA::THREAT_HISTORY_SCALE = int(o); return std::nullopt; }));
	options.add("THREAT_MIN_DEPTH", Option(SPSA::THREAT_MIN_DEPTH, 0, 32, [](const Option& o) {
		SPSA::THREAT_MIN_DEPTH = int(o); return std::nullopt; }));
	// è¬ï½¡éï½¨ ThreatClass ç¸ºE® bitmask(è­ï½¢è³ã»x7E=èï½¨ç¹§E¯ç¹ï½©ç¹§E¹)ç¸²ã»aptureé«¯E¤èãE0x76, é©ªE­ç¸ºã»ÑE½âãEç¸ºE¿=0x36ç¸²ã»
	options.add("THREAT_CLASS_MASK", Option(SPSA::THREAT_CLASS_MASK, 0, 127, [](const Option& o) {
		SPSA::THREAT_CLASS_MASK = int(o); return std::nullopt; }));
	// Threat depth ç¹§E²ç¹ï½¼ç¹ãEè­ï½¢è³ã»=èï½¨ç¹å¼±ã»ç¹ãEç¸²ãï½®ä¹ï½è±ºE±ç¸ºã»=ç¸ºèãEèï½¤ç¸ºE®ç¹å¼±ã»ç¹å³¨â¡ç¸ºå¤§åè½¤E¹/è­ï½´è­E½°ç¸²ãï½µã»ã®ç¹ï½¼ç¹å³¨ã»ç¹æ´ãè¬å¤§å®ç¸²ã»
	options.add("THREAT_MIN_APPLY_DEPTH", Option(SPSA::THREAT_MIN_APPLY_DEPTH, 0, 64, [](const Option& o) {
		SPSA::THREAT_MIN_APPLY_DEPTH = int(o); return std::nullopt; }));
	// Threat eval ç¹§E²ç¹ï½¼ç¹ãEè­ï½¢è³ã»2000=è¾ï½¡èï½¹)ç¸²E´s->staticEval<ç¸ºèãEèï½¤=è³æ¦èç¸ºEªç¹å¼±ã»ç¹å³¨â¡ç¸ºå¤§åè½¤E¹/è­ï½´è­E½°ç¸²ãEèï½¢è­ãã»è¾ï½¡éï½¨ç¹§E®ç¹ï½£ç¹ï½³ç¹æ¶Îè¬å¤§å®ç¸²ã»
	options.add("THREAT_MAX_EVAL", Option(SPSA::THREAT_MAX_EVAL, -32000, 32000, [](const Option& o) {
		SPSA::THREAT_MAX_EVAL = int(o); return std::nullopt; }));
	// Threat èE½¦éãEè­ï½´è­E½°)ç¹§E²ç¹ï½¼ç¹å»ÂãE½¸ç¿«ã»é©ï½©éï½¨ç¹§E²ç¹ï½¼ç¹ãETHREAT_MIN_APPLY_DEPTH/THREAT_MAX_EVAL)ç¸ºä¹ï½è»ã»å±¬ç¸ºå±âE¹§ãEEç¸ºE§ç¸²ã»
	// è­ï½¢è³å£¹ã»è·æÂE¤=è æºæç¸ºE¨ bit-exactç¸²ãE½ºã»E¡ç¹§ä¹âEè´E¿ç¸ºã»E§ä¹ï½ç¹§å¥çç¸ºä¹ï½ç¹ãEã»ç¹æ¶Îç¹§å®ãç¸ºE¦ç¹§å³¨Eç¹§ã»ç¸ºæ´ã»è»ã»TT probe è E±ãNPSè´æ¹E¸ã»ç¸²ã»
	options.add("THREAT_LEARN_MIN_DEPTH", Option(SPSA::THREAT_LEARN_MIN_DEPTH, 0, 64, [](const Option& o) {
		SPSA::THREAT_LEARN_MIN_DEPTH = int(o); return std::nullopt; }));
	options.add("THREAT_LEARN_MAX_EVAL", Option(SPSA::THREAT_LEARN_MAX_EVAL, -32000, 32000, [](const Option& o) {
		SPSA::THREAT_LEARN_MAX_EVAL = int(o); return std::nullopt; }));

	// ---- 分類A のうち USI 未登録だった 8 個(変数は spsa_params.h に既存) ----
	options.add("SPSA_LMR_RESEARCH_DEEPER_THRESHOLD", Option(LMR_RESEARCH_DEEPER_THRESHOLD, 0, 96, [](const Option& o) {
		SPSA::LMR_RESEARCH_DEEPER_THRESHOLD = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_RESEARCH_SHALLOWER_THRESHOLD", Option(LMR_RESEARCH_SHALLOWER_THRESHOLD, 0, 18, [](const Option& o) {
		SPSA::LMR_RESEARCH_SHALLOWER_THRESHOLD = int(o); return std::nullopt; }));
	options.add("SPSA_CAPTURE_SEE_PRUNING_DEPTH_MULT", Option(CAPTURE_SEE_PRUNING_DEPTH_MULT, 0, 334, [](const Option& o) {
		SPSA::CAPTURE_SEE_PRUNING_DEPTH_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_CAPTURE_SEE_PRUNING_HISTORY_MULT", Option(CAPTURE_SEE_PRUNING_HISTORY_MULT, 0, 68, [](const Option& o) {
		SPSA::CAPTURE_SEE_PRUNING_HISTORY_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_FUTILITY_CORRECTION_DIV", Option(FUTILITY_CORRECTION_DIV, 1, 183406, [](const Option& o) {
		SPSA::FUTILITY_CORRECTION_DIV = int(o); return std::nullopt; }));
#if defined(P2_SRCH_BUNDLE_A)
	// 束 A (L2): MovePicker が使う magic number も一緒に作り直す。
	options.add("SPSA_GOOD_CAPTURE_SEE_DIV", Option(GOOD_CAPTURE_SEE_DIV, 1, 36, [](const Option& o) {
		SPSA::GOOD_CAPTURE_SEE_DIV = int(o);
		BundleA::goodCaptureSeeDiv = BundleA::SDiv32::make(SPSA::GOOD_CAPTURE_SEE_DIV);
		return std::nullopt; }));
#else
	options.add("SPSA_GOOD_CAPTURE_SEE_DIV", Option(GOOD_CAPTURE_SEE_DIV, 1, 36, [](const Option& o) {
		SPSA::GOOD_CAPTURE_SEE_DIV = int(o); return std::nullopt; }));
#endif
	options.add("SPSA_QUIET_PARTIAL_SORT_DEPTH_MULT", Option(QUIET_PARTIAL_SORT_DEPTH_MULT, -7120, 0, [](const Option& o) {
		SPSA::QUIET_PARTIAL_SORT_DEPTH_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_ALLNODE_COEFF", Option(LMR_ALLNODE_COEFF, 0, 546, [](const Option& o) {
		SPSA::LMR_ALLNODE_COEFF = int(o); return std::nullopt; }));
	options.add("SPSA_UPDATE_ALL_SEARCHED_COUNT_NUM", Option(UPDATE_ALL_SEARCHED_COUNT_NUM, 0, 8192, [](const Option& o) {
		SPSA::UPDATE_ALL_SEARCHED_COUNT_NUM = int(o); return std::nullopt; }));
	options.add("SPSA_UPDATE_ALL_NORM_NUM", Option(UPDATE_ALL_NORM_NUM, 0, 4096, [](const Option& o) {
		SPSA::UPDATE_ALL_NORM_NUM = int(o); return std::nullopt; }));
	options.add("SPSA_QS_CHECKS", Option(QS_CHECKS, 0, 1, [](const Option& o) {
		SPSA::QS_CHECKS = int(o); return std::nullopt; }));
	options.add("SPSA_FINAL_PV_COMPLETED", Option(FINAL_PV_COMPLETED, 0, 1, [](const Option& o) {
		SPSA::FINAL_PV_COMPLETED = int(o); return std::nullopt; }));
	options.add("SPSA_QUIET_MALUS_DECAY_NUM", Option(QUIET_MALUS_DECAY_NUM, 512, 1024, [](const Option& o) {
		SPSA::QUIET_MALUS_DECAY_NUM = int(o); return std::nullopt; }));

	// ---- SPSA tuning parameters Phase 4 : NAGISA V3.1 TUNABLE_PARAM 対応(分類B 93個) ----
	// 既定値は我々の現行リテラル値。setoption しなければ従来と bit-exact。
	options.add("SPSA_QSEARCH_SEE_PRUNING", Option(QSEARCH_SEE_PRUNING, -146, 0, [](const Option& o) {
		SPSA::QSEARCH_SEE_PRUNING = int(o); return std::nullopt; }));
	options.add("SPSA_QSEARCH_MOVE_COUNT_PRUNING", Option(QSEARCH_MOVE_COUNT_PRUNING, 0, 4, [](const Option& o) {
		SPSA::QSEARCH_MOVE_COUNT_PRUNING = int(o); return std::nullopt; }));
	options.add("SPSA_CORR_HIST_BONUS_BESTMOVE", Option(CORR_HIST_BONUS_BESTMOVE, 0, 24, [](const Option& o) {
		SPSA::CORR_HIST_BONUS_BESTMOVE = int(o); return std::nullopt; }));
	options.add("SPSA_CORR_HIST_BONUS_NOBESTMOVE", Option(CORR_HIST_BONUS_NOBESTMOVE, 0, 34, [](const Option& o) {
		SPSA::CORR_HIST_BONUS_NOBESTMOVE = int(o); return std::nullopt; }));
	options.add("SPSA_CORR_HIST_BONUS_SCALE", Option(CORR_HIST_BONUS_SCALE, 0, 2138, [](const Option& o) {
		SPSA::CORR_HIST_BONUS_SCALE = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_CAPTURE_BONUS", Option(FAILLOW_CAPTURE_BONUS, 0, 2036, [](const Option& o) {
		SPSA::FAILLOW_CAPTURE_BONUS = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_BONUS_SCALE_BASE", Option(FAILLOW_BONUS_SCALE_BASE, -464, 0, [](const Option& o) {
		SPSA::FAILLOW_BONUS_SCALE_BASE = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_BONUS_STATSCORE_DIV", Option(FAILLOW_BONUS_STATSCORE_DIV, 1, 216, [](const Option& o) {
		SPSA::FAILLOW_BONUS_STATSCORE_DIV = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_BONUS_DEPTH_MULT", Option(FAILLOW_BONUS_DEPTH_MULT, 0, 118, [](const Option& o) {
		SPSA::FAILLOW_BONUS_DEPTH_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_BONUS_DEPTH_MAX", Option(FAILLOW_BONUS_DEPTH_MAX, 0, 908, [](const Option& o) {
		SPSA::FAILLOW_BONUS_DEPTH_MAX = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_BONUS_MOVECOUNT", Option(FAILLOW_BONUS_MOVECOUNT, 0, 338, [](const Option& o) {
		SPSA::FAILLOW_BONUS_MOVECOUNT = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_BONUS_MOVECOUNT_TH", Option(FAILLOW_BONUS_MOVECOUNT_TH, 0, 16, [](const Option& o) {
		SPSA::FAILLOW_BONUS_MOVECOUNT_TH = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_BONUS_EVAL", Option(FAILLOW_BONUS_EVAL, 0, 290, [](const Option& o) {
		SPSA::FAILLOW_BONUS_EVAL = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_BONUS_EVAL_MARGIN", Option(FAILLOW_BONUS_EVAL_MARGIN, 0, 220, [](const Option& o) {
		SPSA::FAILLOW_BONUS_EVAL_MARGIN = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_BONUS_PREVEVAL", Option(FAILLOW_BONUS_PREVEVAL, 0, 308, [](const Option& o) {
		SPSA::FAILLOW_BONUS_PREVEVAL = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_BONUS_PREVEVAL_MARGIN", Option(FAILLOW_BONUS_PREVEVAL_MARGIN, 0, 146, [](const Option& o) {
		SPSA::FAILLOW_BONUS_PREVEVAL_MARGIN = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_SCALED_DEPTH_MULT", Option(FAILLOW_SCALED_DEPTH_MULT, 0, 270, [](const Option& o) {
		SPSA::FAILLOW_SCALED_DEPTH_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_SCALED_OFFSET", Option(FAILLOW_SCALED_OFFSET, 0, 160, [](const Option& o) {
		SPSA::FAILLOW_SCALED_OFFSET = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_SCALED_MAX", Option(FAILLOW_SCALED_MAX, 0, 2800, [](const Option& o) {
		SPSA::FAILLOW_SCALED_MAX = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_CONTHIST_MULT", Option(FAILLOW_CONTHIST_MULT, 0, 442, [](const Option& o) {
		SPSA::FAILLOW_CONTHIST_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_MAINHIST_MULT", Option(FAILLOW_MAINHIST_MULT, 0, 470, [](const Option& o) {
		SPSA::FAILLOW_MAINHIST_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_FAILLOW_PAWNHIST_MULT", Option(FAILLOW_PAWNHIST_MULT, 0, 580, [](const Option& o) {
		SPSA::FAILLOW_PAWNHIST_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_IIR_MIN_DEPTH", Option(IIR_MIN_DEPTH, 0, 12, [](const Option& o) {
		SPSA::IIR_MIN_DEPTH = int(o); return std::nullopt; }));
	options.add("SPSA_IIR_PRIOR_REDUCTION_MAX", Option(IIR_PRIOR_REDUCTION_MAX, 0, 6, [](const Option& o) {
		SPSA::IIR_PRIOR_REDUCTION_MAX = int(o); return std::nullopt; }));
	options.add("SPSA_CONT_CORR_HIST_INIT", Option(CONT_CORR_HIST_INIT, 0, 12, [](const Option& o) {
		SPSA::CONT_CORR_HIST_INIT = int(o); return std::nullopt; }));
	options.add("SPSA_QSEARCH_FUTILITY_BASE_MARGIN", Option(QSEARCH_FUTILITY_BASE_MARGIN, 0, 656, [](const Option& o) {
		SPSA::QSEARCH_FUTILITY_BASE_MARGIN = int(o); return std::nullopt; }));
	options.add("SPSA_TTMOVE_HISTORY_BONUS", Option(TTMOVE_HISTORY_BONUS, 0, 1316, [](const Option& o) {
		SPSA::TTMOVE_HISTORY_BONUS = int(o); return std::nullopt; }));
	options.add("SPSA_TTMOVE_HISTORY_MALUS", Option(TTMOVE_HISTORY_MALUS, -1724, 0, [](const Option& o) {
		SPSA::TTMOVE_HISTORY_MALUS = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_STEP18_NO_TTMOVE_ADD", Option(LMR_STEP18_NO_TTMOVE_ADD, 0, 2442, [](const Option& o) {
		SPSA::LMR_STEP18_NO_TTMOVE_ADD = int(o); return std::nullopt; }));
	options.add("SPSA_POST_LMR_CONTHIST_BONUS", Option(POST_LMR_CONTHIST_BONUS, 0, 2000, [](const Option& o) {
		SPSA::POST_LMR_CONTHIST_BONUS = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_STATSCORE_MULT", Option(LMR_STATSCORE_MULT, 0, 746, [](const Option& o) {
		SPSA::LMR_STATSCORE_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_CAPTURE_STATSCORE_PIECEVALUE_MULT", Option(LMR_CAPTURE_STATSCORE_PIECEVALUE_MULT, 0, 1732, [](const Option& o) {
		SPSA::LMR_CAPTURE_STATSCORE_PIECEVALUE_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_TTMOVE_REDUCTION", Option(LMR_TTMOVE_REDUCTION, 0, 3452, [](const Option& o) {
		SPSA::LMR_TTMOVE_REDUCTION = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_CUTOFFCNT_BASE", Option(LMR_CUTOFFCNT_BASE, 0, 502, [](const Option& o) {
		SPSA::LMR_CUTOFFCNT_BASE = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_CUTOFFCNT_GT2", Option(LMR_CUTOFFCNT_GT2, 0, 2980, [](const Option& o) {
		SPSA::LMR_CUTOFFCNT_GT2 = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_CUTOFFCNT_ALLNODE", Option(LMR_CUTOFFCNT_ALLNODE, 0, 1274, [](const Option& o) {
		SPSA::LMR_CUTOFFCNT_ALLNODE = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_TTCAPTURE_ADD", Option(LMR_TTCAPTURE_ADD, 0, 3086, [](const Option& o) {
		SPSA::LMR_TTCAPTURE_ADD = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_CUTNODE_ADD", Option(LMR_CUTNODE_ADD, 0, 4558, [](const Option& o) {
		SPSA::LMR_CUTNODE_ADD = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_CUTNODE_NO_TTMOVE_ADD", Option(LMR_CUTNODE_NO_TTMOVE_ADD, 0, 3020, [](const Option& o) {
		SPSA::LMR_CUTNODE_NO_TTMOVE_ADD = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_MOVECOUNT_MULT", Option(LMR_MOVECOUNT_MULT, 0, 130, [](const Option& o) {
		SPSA::LMR_MOVECOUNT_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_CORRECTION_DIV", Option(LMR_CORRECTION_DIV, 1, 56294, [](const Option& o) {
		SPSA::LMR_CORRECTION_DIV = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_BASE_OFFSET", Option(LMR_BASE_OFFSET, 0, 2064, [](const Option& o) {
		SPSA::LMR_BASE_OFFSET = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_TTPV_BASE", Option(LMR_TTPV_BASE, 0, 4472, [](const Option& o) {
		SPSA::LMR_TTPV_BASE = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_TTPV_PVNODE", Option(LMR_TTPV_PVNODE, 0, 2408, [](const Option& o) {
		SPSA::LMR_TTPV_PVNODE = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_TTPV_TTVALUE_GT_ALPHA", Option(LMR_TTPV_TTVALUE_GT_ALPHA, 0, 1810, [](const Option& o) {
		SPSA::LMR_TTPV_TTVALUE_GT_ALPHA = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_TTPV_TTDEPTH", Option(LMR_TTPV_TTDEPTH, 0, 1870, [](const Option& o) {
		SPSA::LMR_TTPV_TTDEPTH = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_TTPV_TTDEPTH_CUTNODE", Option(LMR_TTPV_TTDEPTH_CUTNODE, 0, 2542, [](const Option& o) {
		SPSA::LMR_TTPV_TTDEPTH_CUTNODE = int(o); return std::nullopt; }));
	options.add("SPSA_SE_TRIPLE_MARGIN_BASE", Option(SE_TRIPLE_MARGIN_BASE, 0, 232, [](const Option& o) {
		SPSA::SE_TRIPLE_MARGIN_BASE = int(o); return std::nullopt; }));
	options.add("SPSA_SE_TRIPLE_MARGIN_PVNODE", Option(SE_TRIPLE_MARGIN_PVNODE, 0, 798, [](const Option& o) {
		SPSA::SE_TRIPLE_MARGIN_PVNODE = int(o); return std::nullopt; }));
	options.add("SPSA_SE_TRIPLE_MARGIN_NOTTCAPTURE", Option(SE_TRIPLE_MARGIN_NOTTCAPTURE, 0, 436, [](const Option& o) {
		SPSA::SE_TRIPLE_MARGIN_NOTTCAPTURE = int(o); return std::nullopt; }));
	options.add("SPSA_SE_TRIPLE_MARGIN_TTPV", Option(SE_TRIPLE_MARGIN_TTPV, 0, 238, [](const Option& o) {
		SPSA::SE_TRIPLE_MARGIN_TTPV = int(o); return std::nullopt; }));
	options.add("SPSA_SE_TRIPLE_MARGIN_PLY", Option(SE_TRIPLE_MARGIN_PLY, 0, 76, [](const Option& o) {
		SPSA::SE_TRIPLE_MARGIN_PLY = int(o); return std::nullopt; }));
	options.add("SPSA_SE_DOUBLE_MARGIN_BASE", Option(SE_DOUBLE_MARGIN_BASE, -8, 6, [](const Option& o) {
		SPSA::SE_DOUBLE_MARGIN_BASE = int(o); return std::nullopt; }));
	options.add("SPSA_SE_DOUBLE_MARGIN_PVNODE", Option(SE_DOUBLE_MARGIN_PVNODE, 0, 390, [](const Option& o) {
		SPSA::SE_DOUBLE_MARGIN_PVNODE = int(o); return std::nullopt; }));
	options.add("SPSA_SE_DOUBLE_MARGIN_NOTTCAPTURE", Option(SE_DOUBLE_MARGIN_NOTTCAPTURE, 0, 492, [](const Option& o) {
		SPSA::SE_DOUBLE_MARGIN_NOTTCAPTURE = int(o); return std::nullopt; }));
	options.add("SPSA_SE_DOUBLE_MARGIN_TTMOVEHIST", Option(SE_DOUBLE_MARGIN_TTMOVEHIST, 0, 1662, [](const Option& o) {
		SPSA::SE_DOUBLE_MARGIN_TTMOVEHIST = int(o); return std::nullopt; }));
	options.add("SPSA_SE_DOUBLE_MARGIN_PLY", Option(SE_DOUBLE_MARGIN_PLY, 0, 88, [](const Option& o) {
		SPSA::SE_DOUBLE_MARGIN_PLY = int(o); return std::nullopt; }));
	options.add("SPSA_SE_CORRVAL_DIV", Option(SE_CORRVAL_DIV, 1, 556130, [](const Option& o) {
		SPSA::SE_CORRVAL_DIV = int(o); return std::nullopt; }));
	options.add("SPSA_SE_SINGULAR_BETA_BASE", Option(SE_SINGULAR_BETA_BASE, 0, 146, [](const Option& o) {
		SPSA::SE_SINGULAR_BETA_BASE = int(o); return std::nullopt; }));
	options.add("SPSA_SE_SINGULAR_BETA_TTPV", Option(SE_SINGULAR_BETA_TTPV, 0, 234, [](const Option& o) {
		SPSA::SE_SINGULAR_BETA_TTPV = int(o); return std::nullopt; }));
	options.add("SPSA_SE_SINGULAR_BETA_DIV", Option(SE_SINGULAR_BETA_DIV, 1, 66, [](const Option& o) {
		SPSA::SE_SINGULAR_BETA_DIV = int(o); return std::nullopt; }));
	options.add("SPSA_SEE_PRUNING_QUIET_MULT", Option(SEE_PRUNING_QUIET_MULT, -82, 0, [](const Option& o) {
		SPSA::SEE_PRUNING_QUIET_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_FUTILITY_PARENT_LMRDEPTH_MULT", Option(FUTILITY_PARENT_LMRDEPTH_MULT, 0, 240, [](const Option& o) {
		SPSA::FUTILITY_PARENT_LMRDEPTH_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_FUTILITY_PARENT_ALPHA_BONUS", Option(FUTILITY_PARENT_ALPHA_BONUS, 0, 154, [](const Option& o) {
		SPSA::FUTILITY_PARENT_ALPHA_BONUS = int(o); return std::nullopt; }));
	options.add("SPSA_CONTHIST_PRUNING_DEPTH_MULT", Option(CONTHIST_PRUNING_DEPTH_MULT, -7822, 0, [](const Option& o) {
		SPSA::CONTHIST_PRUNING_DEPTH_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_CONTHIST_PRUNING_MAINHIST_MULT", Option(CONTHIST_PRUNING_MAINHIST_MULT, 0, 144, [](const Option& o) {
		SPSA::CONTHIST_PRUNING_MAINHIST_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_CONTHIST_PRUNING_LMRDEPTH_DIV", Option(CONTHIST_PRUNING_LMRDEPTH_DIV, 1, 7186, [](const Option& o) {
		SPSA::CONTHIST_PRUNING_LMRDEPTH_DIV = int(o); return std::nullopt; }));
	// 2026-09-27: 上限 766 で x2.0 (984) が無言に切られていた (= x1.56)。
	//   margin 群の中でここだけ未探索だったため上限を上げる。
	options.add("SPSA_CAPTURE_FUTILITY_BASE", Option(CAPTURE_FUTILITY_BASE, 0, 3000, [](const Option& o) {
		SPSA::CAPTURE_FUTILITY_BASE = int(o); return std::nullopt; }));
	// 序盤 (progress <= FV_SCALE_PROG_LO) の capture futility 基準。
	//   0 = 無効 (SPSA_CAPTURE_FUTILITY_BASE に追従 = 従来と bit-exact)。
	//   上限は本体と同じ 3000 (lim3000 系 exe のみ。 旧 exe は範囲外を破棄する)。
	options.add("SPSA_CAPFUT_OPENING", Option(0, 0, 3000, [](const Option& o) {
		P2NNUE::g_capfut_opening = int(o); return std::nullopt; }));
	options.add("SPSA_CAPTURE_FUTILITY_LMRDEPTH_MULT", Option(CAPTURE_FUTILITY_LMRDEPTH_MULT, 0, 810, [](const Option& o) {
		SPSA::CAPTURE_FUTILITY_LMRDEPTH_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_CAPTURE_FUTILITY_CAPTHIST_MULT", Option(CAPTURE_FUTILITY_CAPTHIST_MULT, 0, 290, [](const Option& o) {
		SPSA::CAPTURE_FUTILITY_CAPTHIST_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_LMR_TTPV_INCREASE", Option(LMR_TTPV_INCREASE, 0, 2026, [](const Option& o) {
		SPSA::LMR_TTPV_INCREASE = int(o); return std::nullopt; }));
	options.add("SPSA_SMALL_PROBCUT_MARGIN", Option(SMALL_PROBCUT_MARGIN, 0, 484, [](const Option& o) {
		SPSA::SMALL_PROBCUT_MARGIN = int(o); return std::nullopt; }));
	options.add("SPSA_PROBCUT_MARGIN", Option(PROBCUT_MARGIN, 0, 256, [](const Option& o) {
		SPSA::PROBCUT_MARGIN = int(o); return std::nullopt; }));
	options.add("SPSA_PROBCUT_IMPROVING_MARGIN", Option(PROBCUT_IMPROVING_MARGIN, 0, 158, [](const Option& o) {
		SPSA::PROBCUT_IMPROVING_MARGIN = int(o); return std::nullopt; }));
	options.add("SPSA_HINDSIGHT_EVAL_SUM_TH", Option(HINDSIGHT_EVAL_SUM_TH, 0, 396, [](const Option& o) {
		SPSA::HINDSIGHT_EVAL_SUM_TH = int(o); return std::nullopt; }));
	options.add("SPSA_STATIC_EVAL_PAWNHIST_MULT_Q8", Option(STATIC_EVAL_PAWNHIST_MULT_Q8, 0, 8192 * (SPSA::SCORE_SCALE / 256), [](const Option& o) {
		SPSA::STATIC_EVAL_PAWNHIST_MULT_Q8 = int(o); return std::nullopt; }));
	options.add("SPSA_STATIC_EVAL_CLAMP_LO_Q8", Option(STATIC_EVAL_CLAMP_LO_Q8, -262144 * (SPSA::SCORE_SCALE / 256), 0, [](const Option& o) {
		SPSA::STATIC_EVAL_CLAMP_LO_Q8 = int(o); return std::nullopt; }));
	options.add("SPSA_STATIC_EVAL_CLAMP_HI_Q8", Option(STATIC_EVAL_CLAMP_HI_Q8, 0, 262144 * (SPSA::SCORE_SCALE / 256), [](const Option& o) {
		SPSA::STATIC_EVAL_CLAMP_HI_Q8 = int(o); return std::nullopt; }));
	options.add("SPSA_STATIC_EVAL_OFFSET_Q8", Option(STATIC_EVAL_OFFSET_Q8, 0, 65536 * (SPSA::SCORE_SCALE / 256), [](const Option& o) {
		SPSA::STATIC_EVAL_OFFSET_Q8 = int(o); return std::nullopt; }));
	options.add("SPSA_STATIC_EVAL_MAINHIST_MULT_Q8", Option(STATIC_EVAL_MAINHIST_MULT_Q8, 0, 8192 * (SPSA::SCORE_SCALE / 256), [](const Option& o) {
		SPSA::STATIC_EVAL_MAINHIST_MULT_Q8 = int(o); return std::nullopt; }));
	options.add("SPSA_TT_PREV_CONTHIST_MALUS", Option(TT_PREV_CONTHIST_MALUS, -2354, 0, [](const Option& o) {
		SPSA::TT_PREV_CONTHIST_MALUS = int(o); return std::nullopt; }));
	options.add("SPSA_TT_QUIET_BONUS_DEPTH_MULT", Option(TT_QUIET_BONUS_DEPTH_MULT, 0, 260, [](const Option& o) {
		SPSA::TT_QUIET_BONUS_DEPTH_MULT = int(o); return std::nullopt; }));
	options.add("SPSA_TT_QUIET_BONUS_OFFSET", Option(TT_QUIET_BONUS_OFFSET, 0, 174, [](const Option& o) {
		SPSA::TT_QUIET_BONUS_OFFSET = int(o); return std::nullopt; }));
	options.add("SPSA_TT_QUIET_BONUS_MAX", Option(TT_QUIET_BONUS_MAX, 0, 2010, [](const Option& o) {
		SPSA::TT_QUIET_BONUS_MAX = int(o); return std::nullopt; }));
	options.add("SPSA_ASPIRATION_DELTA_BASE", Option(ASPIRATION_DELTA_BASE, 0, 46, [](const Option& o) {
		SPSA::ASPIRATION_DELTA_BASE = int(o); return std::nullopt; }));
	// p2shogi 2026-11-01: aspiration 窓の thread 幅。 既定 1 = 従来と bit-exact。
	// p2shogi 2026-11-01: continuation history の thread 間共有。 既定 0 = bit-exact。
	// p2shogi 2026-11-01: SF 61c6871 の移植。 odd thread の NMP reduction を -1。 既定 0。
	options.add("SPSA_NMP_THREAD_PARITY", Option(NMP_THREAD_PARITY, 0, 1, [](const Option& o) {
		SPSA::NMP_THREAD_PARITY = int(o); return std::nullopt; }));
	// p2shogi 2026-11-02: M1 遅延 (ABDADA-lite)。 既定 0 = bit-exact。
	//   残り深さがこの値以上の節点で、 2 手目以降の「他 thread が読んでいる子局面」への手を
	//   後回しにする。 後回しは末尾で元の moveCount のまま必ず読む (探索の完全性は不変)。
	options.add("SPSA_SMP_DEFER_MIN_DEPTH", Option(SMP_DEFER_MIN_DEPTH, 0, 32, [](const Option& o) {
		SPSA::SMP_DEFER_MIN_DEPTH = int(o); return std::nullopt; }));
	// p2shogi 2026-11-02: M2 helper の反復ずらし (SF9 skipSize/skipPhase の復活)。 既定 0。
	options.add("SPSA_SMP_HELPER_SKIP", Option(SMP_HELPER_SKIP, 0, 1, [](const Option& o) {
		SPSA::SMP_HELPER_SKIP = int(o); return std::nullopt; }));
	// p2shogi 2026-11-02: M3 根の YBWC。 既定 0 = bit-exact。
	//   長兄 (1 手目) を master が読み終えてから兄弟手を空いた thread に配る。
	//   共有 alpha を使うので Lazy SMP の投機の税 (重複 92-95%) を払わない。
	//   🚨Threads>1 かつ MultiPV==1 かつ searchmoves 無しのときだけ有効になる。
	options.add("SPSA_ROOT_YBWC", Option(ROOT_YBWC, 0, 1, [](const Option& o) {
		SPSA::ROOT_YBWC = int(o); return std::nullopt; }));
	// p2shogi 2026-11-02: M4 木の内部での仕事の委譲。 既定 0 = bit-exact。
	//   深い節の 2 手目以降を空いている thread に提示し、 helper が置換表を温める。
	//   master は必ず全手を自分で読むので**正しさは helper に依存しない**。
	options.add("SPSA_YBWC_MIN_DEPTH", Option(YBWC_MIN_DEPTH, 0, 32, [](const Option& o) {
		SPSA::YBWC_MIN_DEPTH = int(o); return std::nullopt; }));
	options.add("SPSA_SHARED_CONT_HIST", Option(SHARED_CONT_HIST, 0, 1, [](const Option& o) {
		SPSA::SHARED_CONT_HIST = int(o); return std::nullopt; }));
	options.add("SPSA_ASPIRATION_THREAD_STEP", Option(ASPIRATION_THREAD_STEP, 1, 12, [](const Option& o) {
		SPSA::ASPIRATION_THREAD_STEP = int(o); return std::nullopt; }));
	options.add("SPSA_ASPIRATION_MSS_DIV", Option(ASPIRATION_MSS_DIV, 1, 13584, [](const Option& o) {
		SPSA::ASPIRATION_MSS_DIV = int(o); return std::nullopt; }));
	options.add("SPSA_MP_GOOD_QUIET_THRESHOLD", Option(MP_GOOD_QUIET_THRESHOLD, -28000, 0, [](const Option& o) {
		SPSA::MP_GOOD_QUIET_THRESHOLD = int(o); return std::nullopt; }));
	options.add("SPSA_MP_LOW_PLY_HISTORY_MULT_Q8", Option(MP_LOW_PLY_HISTORY_MULT_Q8, 0, 8192 * (SPSA::SCORE_SCALE / 256), [](const Option& o) {
		SPSA::MP_LOW_PLY_HISTORY_MULT_Q8 = int(o); return std::nullopt; }));
	options.add("SPSA_MP_QUIET_MAINHIST_MULT_Q8", Option(MP_QUIET_MAINHIST_MULT_Q8, 0, 4096 * (SPSA::SCORE_SCALE / 256), [](const Option& o) {
		SPSA::MP_QUIET_MAINHIST_MULT_Q8 = int(o); return std::nullopt; }));
	options.add("SPSA_MP_QUIET_PAWNHIST_MULT_Q8", Option(MP_QUIET_PAWNHIST_MULT_Q8, 0, 4096 * (SPSA::SCORE_SCALE / 256), [](const Option& o) {
		SPSA::MP_QUIET_PAWNHIST_MULT_Q8 = int(o); return std::nullopt; }));
	options.add("SPSA_MP_QUIET_CHECK_SEE_TH", Option(MP_QUIET_CHECK_SEE_TH, -150, 0, [](const Option& o) {
		SPSA::MP_QUIET_CHECK_SEE_TH = int(o); return std::nullopt; }));
	options.add("SPSA_MP_QUIET_CHECK_BONUS", Option(MP_QUIET_CHECK_BONUS, 0, 32768, [](const Option& o) {
		SPSA::MP_QUIET_CHECK_BONUS = int(o); return std::nullopt; }));
	options.add("SPSA_MP_CAPTURE_PIECEVALUE_MULT_Q8", Option(MP_CAPTURE_PIECEVALUE_MULT_Q8, 0, 8192 * (SPSA::SCORE_SCALE / 256), [](const Option& o) {
		SPSA::MP_CAPTURE_PIECEVALUE_MULT_Q8 = int(o); return std::nullopt; }));

	// ---- Phase 5 : Stockfish 2026-09 master 未移植手法の移植 (SF port, 2026-09-12) ----
	// すべて既定 OFF。OFF なら探索は従来と bit-exact。変数と原文の対応は spsa_params.h を参照。
	// 値は setoption 時にここで SPSA:: グローバルへコピーされ、探索中に Options は引かない。
	// #2 TT secondary aging / TT penalize
	options.add("SF_TT_AGING", Option(false, [](const Option& o) {
		SPSA::SF_TT_AGING = bool(o); return std::nullopt; }));
	options.add("SF_TT_PENALIZE", Option(false, [](const Option& o) {
		SPSA::SF_TT_PENALIZE = bool(o); return std::nullopt; }));
	// #4 go ごとの mainHistory 減衰 (× NUM/1024)
	options.add("SF_MAINHIST_DECAY", Option(false, [](const Option& o) {
		SPSA::SF_MAINHIST_DECAY = bool(o); return std::nullopt; }));
	options.add("SF_MAINHIST_DECAY_NUM", Option(SF_MAINHIST_DECAY_NUM, 0, 1024, [](const Option& o) {
		SPSA::SF_MAINHIST_DECAY_NUM = int(o); return std::nullopt; }));
	// #5 CMHC
	options.add("SF_CMHC", Option(false, [](const Option& o) {
		SPSA::SF_CMHC = bool(o); return std::nullopt; }));
	// #6 multicut 時の correction history 更新
	options.add("SF_MULTICUT_CORR", Option(false, [](const Option& o) {
		SPSA::SF_MULTICUT_CORR = bool(o); return std::nullopt; }));
	options.add("SF_MULTICUT_CORR_MULT", Option(SF_MULTICUT_CORR_MULT, 0, 1024, [](const Option& o) {
		SPSA::SF_MULTICUT_CORR_MULT = int(o); return std::nullopt; }));
	// #7(a) 深さ依存 LMR history 除数 (配列 × SCALE/1024。1024 = SF 生値、既定 1269 = 3712/2995)
	options.add("SF_LMR_DEPTH_DIV", Option(false, [](const Option& o) {
		SPSA::SF_LMR_DEPTH_DIV = bool(o); return std::nullopt; }));
	options.add("SF_LMR_DEPTH_DIV_SCALE", Option(SF_LMR_DEPTH_DIV_SCALE, 256, 4096, [](const Option& o) {
		SPSA::SF_LMR_DEPTH_DIV_SCALE = int(o); SPSA::sf_rescale(); return std::nullopt; }));
	// #7(b) ttMove の reduction 減算を cutoffCnt>1 の else-if にする
	options.add("SF_LMR_TTMOVE_ELSEIF", Option(false, [](const Option& o) {
		SPSA::SF_LMR_TTMOVE_ELSEIF = bool(o); return std::nullopt; }));
	// #7(c) LMR の alpha 項 r += MULT*clamp(alpha - eval, -64, 96) (SF_EVAL_SCALE で換算)
	options.add("SF_LMR_ALPHA_TERM", Option(false, [](const Option& o) {
		SPSA::SF_LMR_ALPHA_TERM = bool(o); return std::nullopt; }));
	options.add("SF_LMR_ALPHA_MULT", Option(SF_LMR_ALPHA_MULT, 0, 32, [](const Option& o) {
		SPSA::SF_LMR_ALPHA_MULT = int(o); SPSA::sf_rescale(); return std::nullopt; }));
	// #7(d) futility pruning の連続化 (深さ上限 DEPTH_MAX、既定 19)
	options.add("SF_FUTILITY_V2", Option(false, [](const Option& o) {
		SPSA::SF_FUTILITY_V2 = bool(o); return std::nullopt; }));
	options.add("SF_FUTILITY_V2_DEPTH_MAX", Option(SF_FUTILITY_V2_DEPTH_MAX, 1, 64, [](const Option& o) {
		SPSA::SF_FUTILITY_V2_DEPTH_MAX = int(o); return std::nullopt; }));
	// eval 単位の SF 定数 (#7c の clamp 幅、#7d の futilityMult / corr 除数) の一括換算係数。
	// 1024 = SF 値そのまま (やねうら王 master と同じ 1:1 移植)。候補: 647 (=48/76), 918 (=1/1.115)。
	options.add("SF_EVAL_SCALE", Option(SF_EVAL_SCALE, 256, 4096, [](const Option& o) {
		SPSA::SF_EVAL_SCALE = int(o); SPSA::sf_rescale(); return std::nullopt; }));
	// 2026-11-22 release defaults (final_s29g_backs_sw500). USI default and current value both become these.
	static const std::pair<const char*, const char*> release_defaults[] = {
		{"SPSA_CAPTURE_FUTILITY_BASE", "590"},
		{"AigakariMigigyokuBonusWhite", "0"},
		{"MigigyokuBonusWhite", "0"},
		{"SPSA_FUTILITY_MARGIN_BASE", "73"},
		{"SPSA_FUTILITY_MARGIN_TT_BONUS", "50"},
		{"SPSA_RAZORING_BASE", "345"},
		{"SPSA_RAZORING_DEPTH2", "377"},
		{"SPSA_NMP_MARGIN_DEPTH_MULT", "9"},
		{"SPSA_NMP_MARGIN_OFFSET", "132"},
		{"SPSA_CAPTURE_FUTILITY_LMRDEPTH_MULT", "446"},
		{"SPSA_FUTILITY_PARENT_ALPHA_BONUS", "44"},
		{"SPSA_FUTILITY_PARENT_LMRDEPTH_MULT", "52"},
		{"SPSA_HINDSIGHT_EVAL_SUM_TH", "180"},
		{"SPSA_PROBCUT_MARGIN", "170"},
		{"SPSA_QSEARCH_FUTILITY_BASE_MARGIN", "189"},
		{"SPSA_SEE_PRUNING_QUIET_MULT", "-35"},
		{"SPSA_SE_SINGULAR_BETA_BASE", "65"},
		{"SPSA_SMALL_PROBCUT_MARGIN", "161"},
		{"OPN_ProgHi", "-1946"},
		{"SPSA_CAPTURE_SEE_PRUNING_DEPTH_MULT", "169"},
		{"SPSA_CAPTURE_SEE_PRUNING_HISTORY_MULT", "34"},
		{"FV_SCALE_PER_NODE", "true"},
		{"FV_SCALE_B0", "26"},
		{"FV_SCALE_B1", "26"},
		{"FV_SCALE_B2", "27"},
		{"FV_SCALE_B3", "28"},
		{"FV_SCALE_B4", "28"},
		{"FV_SCALE_B5", "28"},
		{"FV_SCALE_B6", "28"},
		{"FV_SCALE_B7", "28"},
		{"P2_BUCKET_OFFSET_B0", "-39"},
		{"P2_BUCKET_OFFSET_B1", "-14"},
		{"P2_BUCKET_OFFSET_B2", "-27"},
		{"P2_BUCKET_OFFSET_B3", "19"},
		{"P2_BUCKET_OFFSET_B4", "4"},
		{"P2_BUCKET_OFFSET_B5", "-12"},
		{"P2_BUCKET_OFFSET_B6", "10"},
		{"P2_BUCKET_OFFSET_B7", "29"},
		{"P2_BUCKET_OFFSET_SENTE_B0", "-18"},
		{"P2_BUCKET_OFFSET_SENTE_B1", "-3"},
		{"P2_BUCKET_OFFSET_SENTE_B2", "-32"},
		{"P2_BUCKET_OFFSET_SENTE_B3", "-5"},
		{"P2_BUCKET_OFFSET_SENTE_B4", "16"},
		{"P2_BUCKET_OFFSET_SENTE_B5", "-15"},
		{"P2_BUCKET_OFFSET_SENTE_B6", "-4"},
		{"P2_BUCKET_OFFSET_SENTE_B7", "15"},
		{"P2_SENTE_TABLE_SWITCH_CP", "500"},
		{"SPSA_LMR_TABLE_COEFF_OPENING", "2763"},
	};
	for (const auto& kv : release_defaults)
		options.set_release_default(kv.first, kv.second);
}

void load_eval() {
	// 11-16: ファイル 4 種 (EvalFile / LS_PROGRESS_COEFF / KHFile / OPNFile) をやねうら王の nn.bin と同じ作法で解決する。
	//   相対名 = <exe のフォルダ>/<EvalDir>/<名前> (YO の evaluate_nnue.cpp と同じ Path::Combine 2 段)、
	//   絶対パス = そのまま (Path::Combine が IsAbsolute を素通しする)。 空 = 無効 (進行度 / KH / OPN)。
	//   読めなければ YO の nn.bin と同じく Error! を出して終了する (黙って別の評価で走らせない)。
	const std::string eval_dir = Path::Combine(Directory::GetBinaryFolder(), g_eval_dir);
	auto resolve = [&](const std::string& f) { return f.empty() ? f : Path::Combine(eval_dir, f); };
	auto die = [](const char* opt, const std::string& p) {
		sync_cout << "info string Error! " << opt << ": failed to read " << p << sync_endl;
		std::fprintf(stderr, "Error! %s: failed to read %s\n", opt, p.c_str());
		std::exit(EXIT_FAILURE);
	};
	{
		const std::string path = resolve(g_eval_file);
		if (!P2NNUE::load_network(path)) die("EvalFile", path);
	}
	if (!g_progress_path.empty()) {
		const std::string path = resolve(g_progress_path);
		if (!P2NNUE::load_progress(path)) die("LS_PROGRESS_COEFF", path);
	}
#if defined(P2_KH)
	P2NNUE::kh::load(resolve(g_kh_path));   // 失敗時は kh::load が Error! を出して終了する
#endif
#if defined(P2_OPN)
	P2NNUE::opn::load(resolve(g_opn_path),  // 失敗時は opn::load が Error! を出して終了する
	    P2NNUE::g_progress_loaded, P2NNUE::g_lazy_threat_margin, P2NNUE::g_lazy_mode);
#endif
	g_need_load = false;
}

Value evaluate(const Position& pos) {
	R2T_TICK(); R2T_SCOPE(R_EVAL);
	return Value(P2NNUE::evaluate_internal(pos, false));
}

Value compute_eval(const Position& pos) {
	R2T_TICK(); R2T_SCOPE(R_EVAL);
	return Value(P2NNUE::evaluate_internal(pos, false));
}

// 2026-11-05 (fable) P2_FT_PREFETCH_LINES: do_move の末尾で、 この局面の evaluate が読むことになる
//   FT 行 (差分 add/sub の行、 persp 2 つ分) の先頭数ラインを prefetch する。
//   狙い: ensure_perspective (acc 更新) は 2KB 行を DRAM から流し込む memory-bound 区間で、 評価 1 回
//   あたり ~5.9K cycle (probe 実測、 eval の 40-55%)。 do_move から evaluate までの間 (TT probe /
//   mate_1ply / 指し手生成) に行の先頭を取り寄せておけば DRAM 待ちの一部が重なる。
//   🚨🚨2026-11-10 訂正: 上の「~5.9K cycle / eval の 40-55%」は**再現しない**。
//     P2_FABLE_PROBE で中盤 5 局面 x 1M ノード (eval 3,109,888 回) を測った実測は:
//       T_acc  **2,844 cycle (28%)**  / T_fwd 6,334 (63%) / T_heads 884 (9%) / eval 全体 10,062
//       T_acc の分布: <1K 3.2% / 1-2K 43.7% / 2-4K 43.3% / 4-8K 7.4% / >=8K 2.4%
//       -> **T_acc < 2,000 の eval が 46.9%**
//     記録 runs/prereg/VERDICT-G0-20261110.log。 この誤った数字を前提に
//     「列挙を別スレッドへ逃がして acc の DRAM 待ちに隠す」案 (G0) を立て、 実測で閉じた。
//     🚨**コメントの数字を設計の前提にしないこと** (ログの在る実測だけを使う)。
//   bit-exact の根拠: prefetch はアーキテクチャ状態 (レジスタ・メモリの値) を一切変えない。
//   守るべき不変条件: 行 index の計算は ensure_perspective / refresh_perspective と同じ式
//   (kb * PIECE_INPUTS + pack_bonapiece(bp, mirror))。 違っていても値は変わらない (無駄になるだけ)。
//   玉が動いた persp は Finny の refresh 経路 (行の集合は差分では決まらない) なので触らない。
#if defined(P2_FT_PREFETCH_LINES)
void p2_prefetch_ft_rows(const Position& pos) {
	const P2NNUE::Net* net = P2NNUE::g_net.get();
	if (net == nullptr || !net->loaded) return;
	const StateInfo* st = pos.state();
	const Eval::DirtyPiece& dp = st->dirtyPiece;
	if (dp.dirty_num == 0) return;
	constexpr int LINES = P2_FT_PREFETCH_LINES;
	constexpr int ROW_LINES = (int)((P2NNUE::FT_ROW_BYTES + 63) / 64);
	constexpr int N = LINES < ROW_LINES ? LINES : ROW_LINES;
	for (Color persp : {BLACK, WHITE}) {
		if (P2NNUE::friend_king_moved(st, persp)) {
#if defined(P2_FINNY_PREFETCH)
			// **案 5 (設計 fable 2026-11-06)**: 玉が動いた persp は FT 行の差分では決まらず
			//   Finny の refresh 経路に落ちる。 refresh_perspective が最初に読むのは
			//   `g_acc_cache.e[persp][kb][mirror]` の `valid` と `bp[]` (160 B = 3 line) と
			//   acc の先頭 (2 KB の頭)。 ここで触っておけば do_move -> evaluate の間に重なる。
			//   🚨値は読まないので **bit-exact** (prefetch はアーキテクチャ状態を変えない)。
			//   不変条件: 索引は refresh_perspective と同じ (king_bucket / is_hm_mirror)。
			//     違っていても値は変わらない (無駄になるだけ)。
			const Square kmv = pos.square<KING>(persp);
			const P2NNUE::AccCacheEntry& en =
				P2NNUE::g_acc_cache.e[persp][P2NNUE::king_bucket(kmv, persp)]
				                     [P2NNUE::is_hm_mirror(kmv, persp) ? 1 : 0];
			const char* const bpp = reinterpret_cast<const char*>(&en.bp[0]);
			_mm_prefetch(bpp, _MM_HINT_T0);
			_mm_prefetch(bpp + 64, _MM_HINT_T0);
			_mm_prefetch(bpp + 128, _MM_HINT_T0);
			const char* const acp = reinterpret_cast<const char*>(&en.acc[0]);
			_mm_prefetch(acp, _MM_HINT_T0);
			_mm_prefetch(acp + 64, _MM_HINT_T0);
#endif
			continue;
		}
		const Square ksq = pos.square<KING>(persp);
		const int kb = P2NNUE::king_bucket(ksq, persp);
		const bool mirror = P2NNUE::is_hm_mirror(ksq, persp);
		for (int i = 0; i < dp.dirty_num; ++i) {
			const Eval::ChangedBonaPiece& cp = dp.changed_piece[i];
			const int ob = (persp == BLACK) ? (int)cp.old_piece.fb : (int)cp.old_piece.fw;
			const int nb = (persp == BLACK) ? (int)cp.new_piece.fb : (int)cp.new_piece.fw;
			if (ob != BONA_PIECE_ZERO) {
				const char* row = reinterpret_cast<const char*>(P2NNUE::ft_row_bytes(*net, (std::size_t)kb * P2NNUE::PIECE_INPUTS + P2NNUE::pack_bonapiece(ob, mirror)));
				for (int l = 0; l < N; ++l) _mm_prefetch(row + 64 * l, _MM_HINT_T0);
			}
			if (nb != BONA_PIECE_ZERO) {
				const char* row = reinterpret_cast<const char*>(P2NNUE::ft_row_bytes(*net, (std::size_t)kb * P2NNUE::PIECE_INPUTS + P2NNUE::pack_bonapiece(nb, mirror)));
				for (int l = 0; l < N; ++l) _mm_prefetch(row + 64 * l, _MM_HINT_T0);
			}
		}
	}
}
#endif

void p2_build_dirty_threats_eager(const Position& pos) {
#if defined(P2_DT_EAGER)
	// do_move ����ɌĂ΂��B dirty-threat �^�v���͎��_��ˑ� (���̏�/��) �Ȃ̂� 1 ��ł悢�B
	// ���������� lazy �łƓ����ɂ��邪�A �u�e�� threat acc ���L�����v�͖��Ȃ�
	// (�����҂̂� chain �f��̌����̂���)�B
	if (!P2NNUE::g_dt_build) return;
	StateInfo* const st = pos.state();
	if (st->p2_dt_state != P2NNUE::DT_UNBUILT) return;
	if (st->previous == nullptr || st->dirtyPiece.dirty_num == 0) return;
	P2NNUE::build_dirty_threats_now(pos);
#else
	(void)pos;
#endif
}

void evaluate_with_no_return(const Position& pos) {
	R2T_SCOPE(R_PRIME);
	// è°ºE¡ç¹å¼±ã»ç¹å³¨ã»è¾E®è»ã»å³©è­E½°ç¸ºE®è¥ï½·è½¤E¹ç¸ºE¨ç¸ºEªç¹§ä¹ï½ç¸ºã»Ââ¥ccumulator ç¹§å®ï½¨è²E®ç²E¸å»â©ç¸ºE«ç¸ºå±â»ç¸ºç¿«E¥ç¸²ã»
	if (P2NNUE::g_net && P2NNUE::g_net->loaded) {
#if defined(P2_OPN)
		// OPN (2026-11-14): λ > 0 の帯では OPN のアキュムレータを作っておく (子の差分の基点)。
		//   λ = 256 (OPN だけ) の帯では本体と threat の priming をしない (その node では本体を使わない)。
		//   🚨OPN 有効時はここで ensure_progress が走る (進行度の float の積み方が変わる。 OPN 有効時は bit-exact を求めない)。
		if (P2NNUE::opn::g_active) {
			const int opn_pb  = P2NNUE::progress_bucket(pos);
			const int opn_lam = P2NNUE::opn::lambda(opn_pb, pos.state()->p2_prog_sum);
#if defined(P2_OPN_F2) && !defined(P2_OPN_F2_EAGER_PRIME)
			// F2-5 (2026-11-15 Fable): OPN の acc はここでは作らない。 子が評価されるときに opn::ensure_perspective が
			//   親の acc を作って残す (p2nnue_opn.h の F2-5)。 acc の値は経路に依らないので評価値は変わらない。
			if (false) {
#else
			if (opn_lam > 0) {
#endif
				P2NNUE::opn::ensure_both(pos);
#if defined(P2_OPN_PROBE)
				++P2NNUE::opn::probe::prime_opn;
#endif
			}
			if (opn_lam == 256) {
#if defined(P2_OPN_PROBE)
				++P2NNUE::opn::probe::prime_skip_main;
#endif
				return;
			}
		}
#endif
#if defined(P2_FABLE_PROBE)
		P2NNUE::PrimeProbe::in_prime = true;    // 2026-11-11 (PRIMEW)
		P2NNUE::ensure_accumulator(pos);
		P2NNUE::PrimeProbe::in_prime = false;
#else
		P2NNUE::ensure_accumulator(pos);
#endif
		// threat acc ç¹§ãEEè®å¥âèEºè½¤E¹ç¸ºE¨ç¸ºå±â»éªè²E®ã»priming)ç¸²E¦val ç¸ºæä»ç¸ºE°ç¹§å¾âEç¸ºã»ã®ç¹ï½¼ç¹ãEéEºçéEã»ç¸ºE§
		// threat acc ç¹§ã»valid ç¸ºE«ç¸ºå±â»ç¸ºç¿«E¥ç¸ºèâEç¸ºE§ç¸²âï½­ä»ï½­E«ç¹å¼±ã»ç¹ãEeval ç¸ºE® delta é©ï½©éï½¨éEEEè³ç¿«E¡ç¸²ã»
		// full èå´ï½¨è²E®ã»~24é¦å¤ã»persp ç¸ºE®ç¹ï½©ç¹ï½³ç¹Âç¹ï£°é¦çï½ªE­ç¸ºE¿)ç¸ºE®é¬E½»è ï½¦ç¹§åE½¸ä¹ï¿¡ç¹§ä¹Âã»
		// eval èï½¤ç¸ºE¯è³Âè»ã»E¤å³¨Eç¹§å³¨âç¸ºã»threat acc ç¸ºE¯ç¸ºE©ç¸ºèãéªè²E®å±E ç¸ºE¦ç¹§ã»bit-exact è·å¾¡E¸Â)ç¸²ã»
		// lazy-threat è­çæè­ãEP2LazyThreatMargin>0)ç¸ºE¯ far node ç¸ºE§ threat é¾Eªè´èï½E		// éµâ«æç¸ºå¶Eè­E½¹é©¥æ´ã»ç¸ºæºï½Epriming ç¸ºå±âç¸ºã»Âã»
		// 2026-09-15 (A1 NOPRIME): priming は「子局面の threat 差分更新 (dirty-threat chain) の基点を
		//   作る」ためにある。 差分更新が無効 (g_dt_build == false: 群化 net / P2_THREAT_FULL / P2_THREAT_DELTA
		//   未設定) のときは、 次の eval で threat acc は必ずこの局面の群の表で full 再計算されるので、
		//   ここで計算しても一度も再利用されない (= 純粋な無駄)。 g_dt_build が真のときだけ priming する。
		//   ★USE_LAZY_EVALUATE 下の安全性: evaluate() を呼ばない node (TT hit && !PvNode) から do_move した
		//     子局面でも、 threat acc の読み手は evaluate_internal → ensure_threat_both (無効なら計算) だけ。
		//     g_dt_build == false では resolve_threat_plan が親の valid を差分の基点に使わない
		//     (build_dirty_threats_now は g_dt_build 条件付き、 通常手の子は p2_dt_state == DT_UNBUILT で chain 打ち切り)。
		//     null move の子 (dt_state == DT_OK, dt_count 0) だけは親が valid なら親の acc を memcpy するが、
		//     null move は盤面・持ち駒・玉位置が同一なので full 再計算と同一の値 (int16 wrapping 加算は順序非依存)。
		//     → priming の有無で評価値・探索木は変わらない (bit-exact)。
		if (P2NNUE::P2_HAS_THREAT && P2NNUE::g_net->has_threat && !P2NNUE::nothreat_env()
		    && !P2NNUE::force_full_env() && P2NNUE::g_lazy_threat_margin == 0
		    // 2026-10-07 S0 (§1.2): P2_THREAT_PRIME で常に priming、 P2_THREAT_NOPRIME で常に切る。
		    //   NOPRIME が勝つ。 既定ではどちらも false なので従来どおり g_dt_build だけで決まる。
		    //   🚨priming は ensure_progress が走る node の集合を変えるので prod と bit-exact で
		    //     なくなる (S0 で確定)。 差分を解放するなら priming は入れないこと (§1.3)。
		    && (P2NNUE::g_dt_build || P2NNUE::probe_prime()) && !P2NNUE::probe_noprime()) {
			P2NNUE::ensure_threat_both(pos);
		}
	}
}

void print_eval_stat(Position& pos) {
	P2NNUE::evaluate_internal(pos, true);
}

// go é«¢å¥E§åºå¾ç¸ºE«ç¹ï½«ç¹ï½¼ç¹äº¥E±Âé«±E¢ rootPos ç¸ºE® progress ç¸ºä¹ï½Eeff_fv / eff_lmr ç¹§ã»èæ§­â¡ç¸ºéï½¢Eºè³å£¹E ç¸²âµç¾éï½¢è³E­ç¸ºE¯èï½ºè³å£¹âE¹§ä¹Âã»
// FV ç¸ºE¨ LMR ç¸ºE¯è·å¾¡E¸Âç¸ºE®é¬æºä¿£èï£°èEEt(progress_t)ç¹§è²ãEè­å³¨E ç¸ºE¦è³æãEç¹ï½­ç¹ãEãç¹§E¹ç¹ãEã£ç¹å±âè»ã»å´ç¹§ä¸ï½ç¸²ã»
// ç¸ºèï½èE¥é«¯éªã»è¬ï½¢éï½¢è³E­ evaluate()(dump=false)ç¸ºE¯è¿´E¾è»Âé«±E¢(ç¹ï½ªç¹ï½¼ç¹ãEç¸ºE§ç¸ºE¯ç¸ºEªç¸ºã»g_root_eff_fv ç¹§åE½½E¿ç¸ºã»Âã»
// LMR reduction table ç¸ºE¯ start_searching ç¸ºã»p2_root_lmr_coeff() ç¸ºE®è«ãçEç¸ºE§èè´E§çï½¯å³¨âE¹§ä¹Âã»
#if defined(P2_BUCKET_OFFSET)
// SIDEOFF2 (11-21): 探索の終わり (main thread、 bestmove を返す直前) に root の探索値を記録する (g_sente_switch_cp の注記)。
//   disp_cp は表示 cp (詰みは ±P2_PREV_MATE_CP)、 valid = false なら記録を消す。
void p2_note_root_result(const Position& rootPos, bool valid, int disp_cp) {
	P2NNUE::g_prev_root_valid = valid;
	P2NNUE::g_prev_root_color = rootPos.side_to_move();
	P2NNUE::g_prev_root_ply   = rootPos.game_ply();
	P2NNUE::g_prev_root_cp    = disp_cp;
}
// SIDEOFF2: isready (YaneuraOuEngine::clear) で記録を消す。
void p2_root_result_reset() { P2NNUE::g_prev_root_valid = false; }
int p2_prev_mate_cp() { return P2NNUE::P2_PREV_MATE_CP; }
#endif

void p2_search_begin(const Position& rootPos) {
	// 🎯片側版 OwnBoardB の「自分」= **root の手番** (2026-10-22)。
	//   1 局の中では我々の色は固定なので、 これで評価は (局面, 自色) の純関数になる。
	//   🚨🚨本番 (USI_Ponder=true) では連続 go / ponder で色が入れ替わりうるので
	//     **TT key の salt が必須**。 判定器は ponder を使わず毎局 isready で TT を消す。
	P2NNUE::g_own_color = rootPos.side_to_move();
#if defined(P2_BUCKET_OFFSET)
	// SIDEOFF (11-21): root 色で bucket 補正の表 (先手番 = SENTE 表 / 後手番 = 既存の表) を確定する。 探索中は固定。
	// SIDEOFF2 (11-21 案 A): SWITCH_CP > 0 なら、 後手番でも直前の手の root 探索値 (表示 cp) が SWITCH_CP 以上なら SENTE 表 (g_sente_switch_cp の注記)。
	{
		const Color rc = rootPos.side_to_move();
		bool use_sente = (rc == BLACK);
		if (P2NNUE::g_sente_switch_cp > 0 && rc == WHITE) {
			// 直前の記録が同じ色で game_ply が +2 (通常 / ponder hit) か +0 (ponder miss: 外れた ponder 探索の root と同じ手数) のとき同じ対局とみなす。
			const int dply = rootPos.game_ply() - P2NNUE::g_prev_root_ply;
			const bool same_game = P2NNUE::g_prev_root_valid && P2NNUE::g_prev_root_color == rc && (dply == 2 || dply == 0);
			if (same_game && P2NNUE::g_prev_root_cp >= P2NNUE::g_sente_switch_cp) use_sente = true;
			if (P2NNUE::g_sente_switch_log)
				sync_cout << "info string P2SWITCH ply " << rootPos.game_ply() << " prev_valid " << P2NNUE::g_prev_root_valid
				          << " prev_ply " << P2NNUE::g_prev_root_ply << " prev_cp " << P2NNUE::g_prev_root_cp
				          << " same_game " << same_game << " th " << P2NNUE::g_sente_switch_cp
				          << " table " << (use_sente ? "SENTE" : "GOTE") << sync_endl;
		}
		P2NNUE::select_bucket_off(use_sente);
	}
#endif
	// 🎯色が決まったら TT の salt を更新する (片側版が全部 0 なら 0 のまま)。
	P2NNUE::update_tt_salt();

#if defined(P2_THREAT_STATS_BUILD)
        // é¨E±éªäº¥E°ãçç¹èÎç¹å³¨ãç¸ºE¯ç¸²âçãèæ·oç¸ºE®é¨ãE½ºã»E¾å¾âEè°ºE¡goç¸ºE¸èï½¥ç¹§çå³©èéªã»é«®ã»E¨å»EèEºç¸ºå¶Âã»
        // é¨¾å£¼E¸E¸ç¹èÎç¹å³¨âç¸ºE¯è³Âè»ã»E®ä¹ï½ç¸ºå£¹ÂâãEèæ·oç¸ºE§ç¸ºE¯é¨Eºé¨E±éªå»EèEºç¸ºè¼âEç¸ºã»Âã»
        static bool g_threat_stats_reported_search = false;
        if (P2NNUE::threat_stats_env() && g_threat_stats_reported_search)
                P2NNUE::g_threat_stats.print();
        g_threat_stats_reported_search = true;
#endif
	// 案 B の下準備: V1 のダンプ (P2RegimeDump=1 のときだけ)。 2026-10-07。
	//   学習器側の python 再実装と突き合わせて V1 の定義を確定するために使う。
	if (P2NNUE::g_regime_dump) {
		P2NNUE::ensure_progress(rootPos);
		const int v1 = YaneuraOu::P2Volm::value(rootPos);
		sync_cout << "info string P2REGIME v1=" << v1
		          << " prog=" << rootPos.state()->p2_prog_sum
		          << " ply=" << rootPos.game_ply()
		          << " sfen=" << rootPos.sfen() << sync_endl;
	}
	float prog = 0.0f;
	const bool have = P2NNUE::g_progress_loaded;
	if (have) {
		// rootPos ç¸ºE® progress sum ç¹§å®ï½¨è²E®ã»è­ï½ªéªè²E®å±âç¹§è¾â¦èï½¶è¾E®è»ã»ç¹è¼Îéªè²E®ã»ç¸²ã»
		P2NNUE::ensure_progress(rootPos);
		prog = rootPos.state()->p2_prog_sum;
	}
	// ç¬ãEE£æºä¿£èï£°èEEt ç¹§ã»èæ§­â¡ç¸ºé¡ï½¨è²E®å±E ç¸²EV ç¸ºE¨ LMR ç¸ºE§èï½±è­å³¨âE¹§ä¹Âã»
	const float t = have ? P2NNUE::progress_t(prog) : 0.0f;

	// FV: 3ç¹§E¢ç¹ï½³ç¹§E«ç¹ï½¼(è åEä¹¢ç«åE¸E­é¶E¤ç«å ¤Eµãä¹¢)ç¹§ã»progress ç¸ºä¹ï½éEºè³å£¹Âã»
	P2NNUE::g_root_eff_fv = P2NNUE::effective_fv_scale(prog);
	// bucket 別 FV: root の bucket に対応する値が設定されていればそれを使う (探索中は固定)。
	if (P2NNUE::g_fv_bucket_any) {
		const int rb = P2NNUE::get_layer_stack_index(rootPos);
		if (rb >= 0 && rb < 8 && P2NNUE::g_fv_scale_bucket[rb] > 0)
			P2NNUE::g_root_eff_fv = P2NNUE::g_fv_scale_bucket[rb];
	}
	P2NNUE::g_root_fv_valid = true;
	// ç¬ãEval ç¹å»Îç¹§E¬ç¹ï½¼: èE½ªèï½¢(root eval >= FV_SCALE_ENDGAME_EVAL)ç¸ºEªç¹§è¾ÂE²é¦æï½ºE¦ç¸ºE«é«¢E¢ç¹§ä¸ï½ç¸ºå¤ï½µãä¹¢ç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½«ç¸ºE¸ revertç¸²ã»
	//   é¨ãä¹¢è­çæEendgame>0)ç¸ºä¹â½é«¢E¾èï½¤è­çæE<32000)è­ãã»ç¸ºE¿ç¸²ãoot ç¹§ã»èæ«E½©ç©ï½¾E¡ç¸ºå±â»è»E¤è³ã»è¿´E¾ eff_fv ç¸ºE§é§ç©ï½¾E¡)ç¸²ã»
	if (P2NNUE::g_fv_scale_endgame > 0 && P2NNUE::g_fv_endgame_eval < 32000) {
		int rootEval = P2NNUE::evaluate_internal(rootPos, false);  // è¿´E¾ g_root_eff_fv ç¸ºE§ root ç¹§å®ï½©ç©ï½¾E¡(è¬EåEä¹ã)
		if (rootEval >= P2NNUE::g_fv_endgame_eval)
			P2NNUE::g_root_eff_fv = P2NNUE::g_fv_scale_endgame;
	}
	// FVNODE: 各局面の bucket で FV を選ぶ。 g_root_eff_fv は表示 (p2_display_rescale_cp) の換算にだけ使われる。
	if (P2NNUE::g_fv_per_node) {
		P2NNUE::rebuild_fv_node();
		P2NNUE::g_root_eff_fv = P2NNUE::g_fv_node[P2NNUE::progress_bucket(rootPos)];
	}

	// LMR: è³E­é¶E¤èï½¤ = è­ï½¢èEESPSA::LMR_TABLE_COEFFç¸²ãE½¾æ¢§å©¿è åé¤(opening==mid)/è­ï½ªç¹ï½­ç¹ï½¼ç¹ç»å¾ç¸ºE¯è³E­é¶E¤èï½¤ç¸²ã»
	const int lmr_mid = SPSA::LMR_TABLE_COEFF;
	if (P2NNUE::g_lmr_coeff_opening == lmr_mid || !have)
		P2NNUE::g_root_eff_lmr = lmr_mid;
	else
		P2NNUE::g_root_eff_lmr = P2NNUE::lerp_round(P2NNUE::g_lmr_coeff_opening, lmr_mid, t);

	// capfut: 序盤値 = 0 か中盤値と同値なら本体に追従 (= 従来と bit-exact)
	const int capfut_mid = SPSA::CAPTURE_FUTILITY_BASE;
	if (P2NNUE::g_capfut_opening == 0 || P2NNUE::g_capfut_opening == capfut_mid || !have)
		p2_root_capfut_base = capfut_mid;
	else
		p2_root_capfut_base = P2NNUE::lerp_round(P2NNUE::g_capfut_opening, capfut_mid, t);

	P2NNUE::g_root_fv_valid = true;
#if defined(P2_OPN)
	// OPN (2026-11-14 設計書 6.2): OPN だけの node が続いた後で本体が要るとき、 遡りが対局の履歴まで行かないように
	//   root で本体と OPN の両方を計算しておく。 値は経路に依らない (i16 wrapping の和) ので結果は変わらない。
	if (P2NNUE::opn::g_active && P2NNUE::g_net && P2NNUE::g_net->loaded) {
		P2NNUE::ensure_accumulator(rootPos);
		P2NNUE::opn::ensure_both(rootPos);
	}
#endif
}

// start_searching ç¸ºã»reductions[] èè´E§çï½¯å³¨âè´E¿ç¸ºã»ÂâÎç¹ï½¼ç¹äº¥å´è³å£¹ã» LMR è«ãçEç¸²ã»
int p2_root_lmr_coeff() { return P2NNUE::g_root_eff_lmr; }

// é¦E¨éï½ºèEçE èã»ÎEcp(= raw/eff_fv ç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½«)ç¹§è²æ¸è²EEFV(FV_SCALE_DISPLAY)ç¹§E¹ç¹§E±ç¹ï½¼ç¹ï½«ç¸ºE¸è¬ å¸·E®å±Âã»
//   FV_SCALE_DISPLAY==0(è­ï½¢è³ã»ç¹§ã»g_root_eff_fv è­ï½ªéï½ºè³å£½å¾ç¸ºE¯ç¸ºæ´ã»ç¸ºE¾ç¸ºE¾éæâEè æ¢§å©¿è åé¤)ç¸²ã»
//   é¨¾å£¼E¸E¸ cp èï½¤ç¸ºE®ç¸ºE¿ç¸ºE«é©ï½©éï½¨(mate/none ç¸ºE¯è»E¼ç¸ºE³èE½ºç¸ºæãEç¸ºE§é«¯E¤èåE¸å»â©)ç¸²ãç¾éï½¢èã»Îç¹ï½­ç¹§E¸ç¹ãEãç¸ºE«ç¸ºE¯é«±æ¨£æªè³å¼±Âã»
int p2_display_rescale_cp(int cp) {
	const int disp = P2NNUE::g_fv_display;
	const int eff = P2NNUE::g_root_eff_fv;
	if (disp <= 0 || eff <= 0) return cp;
	// displayed = cp * eff / disp (èå¸¶æè æ³ãE)ç¸²E¤p,eff<=è¬¨E°è³ã»ã»1024 ç¸ºE§ int ç¸ºE«è¿å¼±âªç¹§ä¹Âã»
	long long num = (long long)cp * eff;
	long long r = (num >= 0) ? (num + disp / 2) / disp : (num - disp / 2) / disp;
	return (int)r;
}

#if defined(P2_SRCH_BUNDLE_A)
// 束 A (L5): 探索の node 入口で書く TLS が、 今の設定で評価関数に読まれるか (= 結果に効くか)。
//   lazyHead : p2_lazy_head は evaluate_internal の do_heads 判定で P2LazyMode == 1 のときだけ読まれる。
//              do_heads が切るのは pairshape と dual_eval (dual_psqt / dual_cross) だけで、
//              それらを持たない net では値が 0 で同じ (dual_eval は dual_w / dualcross_w が空なら 0)。
//   win      : p2_win_a/b は lazy-threat (P2LazyThreatMargin > 0) と margin lazy (P2LazyMode == 2) の
//              判定でだけ読まれる。
//   qsFlag   : g_in_qsearch は `g_in_qsearch ? g_qs_threat_scale : 256` の形でだけ読まれるので、
//              P2QSearchThreatScale == 256 なら値に依らない。
//   🚨net 未読込のときは安全側 (読まれる) にする。 探索外の evaluate (USI の eval コマンドや
//     p2_search_begin の root eval) は直前の探索が最後に書いた値を読むが、 その値に依るのは
//     上の条件が真のときだけで、 そのときは探索が従来どおり書くので同じになる
//     (例外は、 偽の設定で go したあと真の設定に変えて最初に読む 1 回。 本番は設定を変えない)。
void p2_srch_tls_needs(bool& lazyHead, bool& win, bool& qsFlag) {
	const P2NNUE::Net* net   = P2NNUE::g_net.get();
	const bool         heads = net == nullptr || net->has_pairshape || net->has_dual_psqt
	                        || net->has_dual_cross || !net->dual_w.empty() || !net->dualcross_w.empty();
	lazyHead = P2NNUE::g_lazy_mode == 1 && heads;
	win      = P2NNUE::g_lazy_threat_margin > 0 || P2NNUE::g_lazy_mode == 2;
	qsFlag   = P2NNUE::g_qs_threat_scale != 256;
}

#endif
} // namespace Eval
} // namespace YaneuraOu

#endif // EVAL_P2NNUE
