#ifndef MOVEPICK_H_INCLUDED
#define MOVEPICK_H_INCLUDED

#include "config.h"
#if defined(USE_MOVE_PICKER)

#include <array>
//#include <cassert>
#include <cmath>
//#include <cstdint>
//#include <cstdlib>
#include <limits>
//#include <type_traits>

#include "history.h"
#include "movegen.h"
#include "types.h"
#include "position.h"

namespace YaneuraOu {

// -----------------------
//   PolicyIndex(方針指数)
// -----------------------

/// 局面ごとの「方針指数」。attack=攻めるべき度、castle=囲う/受けるべき度、
/// wait=様子見(手待ち・整備)すべき度。いずれも概ね 0..1000 のスケール。
/// eval には一切加算せず、MovePicker の quiet move ordering の微調整にのみ使う
/// (探索順序の誘導のみ = minimax 値を変えない)。QUIET_INIT で per-node 1 回だけ計算し、
/// ヒープ割り当ては行わない(int + bitboard 演算のみ)。
struct PolicyIndex {
    int attack = 0;  // 敵玉を攻める指し手を優先したい度合い
    int castle = 0;  // 自玉を囲う/受ける指し手を優先したい度合い
    int wait   = 0;  // 悪化しにくい手待ち・整備の指し手を優先したい度合い
};

// -----------------------
//   [設計メモ] 方針指数 × LMR(今回は実装しない)
// -----------------------
// 将来案: 方針に「合致」する quiet move の late move reduction(r)をわずかに減らし、
// 方針に反する手の r をわずかに増やす。例(search 側で per-move 判定):
//   - attack 高 かつ その手が「敵玉へ近づく/敵陣へ/成り」   → r -= small
//   - castle 高 かつ その手が「玉移動/金銀寄せ/自玉近傍打ち」 → r -= small
//   - いずれの方針にも合致しない quiet                       → r += small(任意)
// ordering ヒントと同様に minimax 値は変えず、探索の深さ配分だけを方針で偏らせる案。
// ただし LMR は探索木の形を直接変え selfplay 検証の分散が大きいため、ordering ヒントの
// 棋力効果を先に確認してから慎重に導入する(今回はコメントのみ、実装は保留)。

// -----------------------
//   MovePicker
// -----------------------

// 指し手オーダリング器


// MovePicker class is used to pick one pseudo-legal move at a time from the
// current position. The most important method is next_move(), which returns a
// new pseudo-legal move each time it is called, until there are no moves left,
// when Move::none() is returned. In order to improve the efficiency of the
// alpha-beta algorithm, MovePicker attempts to return the moves which are most
// likely to get a cut-off first.
//
// MovePickerクラスは、現在の局面から、(呼び出し)一回につきpseudo legalな指し手を一つ取り出すのに用いる。
// 最も重要なメソッドはnext_move()であり、これは、新しいpseudo legalな指し手を呼ばれるごとに返し、
// (返すべき)指し手が無くなった場合には、Move::none()を返す。
// alpha beta探索の効率を改善するために、MovePickerは最初に(早い段階のnext_move()で)カットオフ(beta cut)が
// 最も出来そうな指し手を返そうとする。
//
class MovePicker {
   public:
    // このクラスは指し手生成バッファが大きいので、コピーして使うような使い方は禁止。
    MovePicker(const MovePicker&)            = delete;
    MovePicker& operator=(const MovePicker&) = delete;

    // 通常探索(main search)と静止探索から呼び出されるとき用のコンストラクタ。
    MovePicker(const Position&              pos_,
               Move                         ttMove_,
               Depth                        depth_,
               const ButterflyHistory*      mh,
               const LowPlyHistory*         lph,
               const CapturePieceToHistory* cph,
               const PieceToHistory**       ch,
               const SharedHistories*       sh,
               int ply_,
               const DropContHistory*       dch_,        // 駒打ち専用history(OFF時 nullptr)
               int                          dropPrevType_,// 直前手の駒種(前手無しは -1)
               const ThreatHistory*         thh_,        // Threat History(OFF/脅威なし時 nullptr)
               uint32_t                     threatCtx_   // 脅威文脈ハッシュ
#if !STOCKFISH
              ,bool generate_all_legal_moves
#endif
    );

    // 通常探索時にProbCutの処理から呼び出されるのコンストラクタ。
    // SEEの値がth以上となるcaptureの指してだけを生成する。
    // threshold_ = 直前に取られた駒の価値。これ以下の捕獲の指し手は生成しない。
    // capture_or_pawn_promotion()に該当する指し手しか返さない。
    MovePicker(const Position&, Move ttMove_, int threshold_, const CapturePieceToHistory*
#if STOCKFISH
#else
    , bool generate_all_legal_moves
#endif
    );

    // 呼び出されるごとに新しいpseudo legalな指し手をひとつ返す。
    // 指し手が尽きればMove::none()が返る。
    // 置換表の指し手(ttMove)を返したあとは、それを取り除いた指し手を返す。
    Move next_move();

    // next_move()で、quietな指し手をskipするためのフラグをセットする。
    void skip_quiet_moves();

   private:
    template<typename Pred>
    Move select(Pred);

    // 指し手のオーダリング用
    // GenType == CAPTURES : 捕獲する指し手のオーダリング
    // GenType == QUIETS   : 捕獲しない指し手のオーダリング
    // GenType == EVASIONS : 王手回避の指し手のオーダリング
    template<GenType T>
    ExtMove* score(const MoveList<T>&);
#if defined(P2_SRCH_BUNDLE_A) && !defined(P2_MP_PREFETCH)
    // 束 A (L1): quiet 採点の速い経路 (movepick.cpp)。 本番で常に偽の機能をすべて外した版。
    template<GenType T>
    [[gnu::noinline]] ExtMove* score_quiets_fast(const MoveList<T>&);
#endif

    // range-based forを使いたいので。
    // 現在の指し手から終端までの指し手が返る。
    ExtMove* begin() { return cur; }
    ExtMove* end() { return endCur; }

    const Position& pos;

    // コンストラクタで渡されたhistroyのポインタを保存しておく変数。
    const ButterflyHistory*      mainHistory;
    const LowPlyHistory*         lowPlyHistory;
    const CapturePieceToHistory* captureHistory;
    const PieceToHistory**       continuationHistory;
	const SharedHistories*       sharedHistory;

    // 駒打ち専用 Drop Continuation History(OFF時 nullptr)と、直前手の駒種(前手無しは -1)。
    const DropContHistory*       dropContHistory = nullptr;
    int                          dropPrevType    = -1;

    // Threat Continuation History(OFF/脅威なし時 nullptr)と、脅威文脈ハッシュ。
    const ThreatHistory*         threatHistory   = nullptr;
    uint32_t                     threatCtx       = 0;

    // 置換表の指し手(コンストラクタで渡される)
    Move ttMove;

    // cur            : 次に返す指し手
    // endCur         : 生成された指し手の末尾
    // endBadCapture  : BadCaptureの終端(captureの指し手を試すフェイズでのendMovesから後方に向かって悪い捕獲の指し手を移動させていく時に用いる)
    // endCaptures    : capturesの終端
    // endGenerated   : 生成された終端
    ExtMove *cur, *endCur, *endBadCaptures, *endCaptures, *endGenerated;

    // 指し手生成の段階
    int stage;

    // ProbCut用の指し手生成に用いる、直前の指し手で捕獲された駒の価値
    int threshold;

    // コンストラクタで渡された探索深さ
    Depth depth;

    // コンストラクタで渡されたrootからの手数
    int ply;

    // next_move()で、quietな指し手をskipするかのフラグ
    bool skipQuiets = false;

    // 方針指数(QUIET_INIT で per-node 1 回だけ計算。POLICY_HINT_SCALE==0 のときは未計算=ゼロ)。
    PolicyIndex policy;

    // 指し手生成バッファ
    // 最大合法手の数 = 593
    // cf. https://www.nara-wu.ac.jp/math/personal/shinoda/bunki.html
    ExtMove moves[MAX_MOVES];

#if STOCKFISH
#else
    // Position::pseudo_legal()に渡す第2パラメーター。
    // 📝 やねうら王では、MovePickerのコンストラクタでもらう。
    bool generate_all_legal_moves;
#endif
};

}  // namespace YaneuraOu

#endif // defined(USE_MOVE_PICKER)
#endif // #ifndef MOVEPICK_H_INCLUDED

