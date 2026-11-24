#include "movepick.h"
#if defined(USE_MOVE_PICKER)

#include <algorithm>
//#include <cassert>
#include <iterator>
#include <utility>

#include "bitboard.h"
#include "position.h"
#include "misc.h"         // sync_cout(POLICY_DEBUG 出力用)
#include "spsa_params.h"  // SPSA::POLICY_HINT_SCALE / POLICY_DEBUG
#if defined(P2_SRCH_BUNDLE_A)
#include "srch_bundle_a.h"
#endif

namespace YaneuraOu {
using namespace Eval; // Eval::PieceValue

namespace {

// -----------------------
//   Q8 固定小数の丸め検算 (NAGISA V3.1 movepick.cpp の static_assert 相当)
// -----------------------
// Q8 化しても既定値では切り捨て結果が変わらないことをコンパイル時に確認する。
// (a) 「最後に 1 回だけ SCORE_SCALE で割る」形は、分子が SCORE_SCALE の倍数なので厳密に一致。
static_assert(SPSA::SCORE_SCALE * 7183 / SPSA::SCORE_SCALE == 7183);
static_assert(SPSA::SCORE_SCALE * -7183 / SPSA::SCORE_SCALE == -7183);
// (b) lowPlyHistory の「SCORE_SCALE*(1+ply) で割る」形は有理数が同一なので切り捨ても一致。
static_assert(8 * SPSA::SCORE_SCALE * 7183 / (SPSA::SCORE_SCALE * 5) == 8 * 7183 / 5);
static_assert(8 * SPSA::SCORE_SCALE * -7183 / (SPSA::SCORE_SCALE * 5) == 8 * -7183 / 5);

// -----------------------
//   方針指数(policy index)の計算(per-node 1 回、軽量ヒューリスティック)
// -----------------------

/// c 側の手駒の攻撃ポテンシャル概算(打ち込みの攻撃力を重み付き枚数で)。
inline int hand_attack_weight(const Position& pos, Color c) {
    const Hand h = pos.hand_of(c);
    return hand_count(h, ROOK) * 4 + hand_count(h, BISHOP) * 3 + hand_count(h, GOLD) * 2
         + hand_count(h, SILVER) * 2 + hand_count(h, KNIGHT) * 1 + hand_count(h, LANCE) * 1
         + hand_count(h, PAWN) * 1;
}

/// 玉 sq の 5x5(自身含む最大25升)近傍 Bitboard を返す。
/// 初回呼び出し時に 81 升ぶんを1回だけ構築し、以後はテーブル参照(per-node は AND+popcount のみ)。
/// C++11 の magic static によりスレッドセーフに1回初期化される(ヒープ確保は初回のみ・以後ゼロ)。
inline const Bitboard& king_zone5(Square sq) {
    static const std::array<Bitboard, SQ_NB> table = [] {
        std::array<Bitboard, SQ_NB> t;
        for (int s = 0; s < SQ_NB; ++s) {
            const int f0 = (int)file_of((Square)s);
            const int r0 = (int)rank_of((Square)s);
            Bitboard bb = Bitboard(ZERO);
            for (int df = -2; df <= 2; ++df)
                for (int dr = -2; dr <= 2; ++dr) {
                    const int f = f0 + df, r = r0 + dr;
                    if (f >= 0 && f < 9 && r >= 0 && r < 9)
                        bb |= Bitboard((File)f | (Rank)r);
                }
            t[s] = bb;
        }
        return t;
    }();
    return table[sq];
}

/// 局面の方針指数(attack/castle/wait)を計算する。eval には関与せず ordering ヒント専用。
/// ★利き生成(attackers_to)は使わず、玉 5x5 zone の駒 proximity(bitboard AND + popcount)で近似:
///   attack = 敵玉危険度(敵玉 zone の自軍駒 + 手駒 - 敵守備薄) - castle/2、
///   castle = 自玉危険度(自玉 zone の相手駒 + 自守備薄)、
///   wait   = どちらも切迫していない残差。全て概ね 0..1000 にクランプ。
inline PolicyIndex compute_policy_index(const Position& pos) {
    PolicyIndex pi;
    const Color  us   = pos.side_to_move();
    const Color  them = ~us;
    const Square ok   = pos.square<KING>(us);
    const Square ek   = pos.square<KING>(them);

    const Bitboard ourPieces   = pos.pieces(us);
    const Bitboard theirPieces = pos.pieces(them);
    const Bitboard ez          = king_zone5(ek);  // 敵玉 5x5
    const Bitboard oz          = king_zone5(ok);  // 自玉 5x5

    // 攻め: 敵玉 zone の自軍駒数 + 手駒 + (敵の守備駒が少ないほど高い)。
    const int ourNearEnemyKing = (ez & ourPieces).pop_count();
    const int enemyDefenders   = (ez & theirPieces).pop_count();  // 敵玉自身を含む
    const int ourHand          = hand_attack_weight(pos, us);

    // 受け: 自玉 zone の相手駒数 + (自軍の守備駒が少ないほど高い)。
    const int enemyNearOurKing = (oz & theirPieces).pop_count();
    const int ourDefenders     = (oz & ourPieces).pop_count();    // 自玉自身を含む

    int castle = enemyNearOurKing * 180 + (std::max)(0, 4 - ourDefenders) * 60;
    castle     = (std::min)(castle, 1000);

    int attack = ourNearEnemyKing * 160 + ourHand * 22 + (std::max)(0, 4 - enemyDefenders) * 30;
    attack -= castle / 2;  // 自玉が危険なら攻めは抑制。
    attack = (std::max)(0, (std::min)(attack, 1000));

    int wait = (std::max)(0, 600 - attack - castle);

    pi.attack = attack;
    pi.castle = castle;
    pi.wait   = wait;
    return pi;
}


// -----------------------
//   指し手オーダリング
// -----------------------

// 指し手を段階的に生成するために現在どの段階にあるかの状態を表す定数
enum Stages: int {

	// -----------------------------------------------------
	//   王手がかっていない通常探索時用の指し手生成
	// -----------------------------------------------------

	MAIN_TT,					// 置換表の指し手を返すフェーズ
	CAPTURE_INIT,				// (CAPTURESの指し手生成)
	GOOD_CAPTURE,				// 捕獲する指し手(CAPTURES_PRO_PLUS)を生成して指し手を一つずつ返す。ただし、SEE値の悪い手(=BAD_CAPTURE)は後回し。
	QUIET_INIT,					// (QUIETの指し手生成)
	GOOD_QUIET,					// CAPTURES_PRO_PLUSで生成しなかった指し手を生成して、一つずつ返す。
	BAD_CAPTURE,				// 捕獲する  悪い指し手(将棋においてそこまで悪い手とは限らない)
	BAD_QUIET,                  // 捕獲しない悪い指し手(これはBAD_CAPTUREのあとに処理する)

	// 将棋ではBAD_CAPTUREをQUIET_の前にやったほうが良いという従来説は以下の実験データにより覆った。
	//  r300, 2585 - 62 - 2993(46.34% R - 25.46)[2016/08/19]
	// b1000, 1051 - 43 - 1256(45.56% R - 30.95)[2016/08/19]

	// -----------------------------------------------------
	//   王手がかっているときの静止探索/通常探索の指し手生成
	// -----------------------------------------------------

	EVASION_TT,					// 置換表の指し手を返すフェーズ
	EVASION_INIT,				// (EVASIONSの指し手を生成)
	EVASION,					// 回避する指し手(EVASIONS)を生成した指し手を一つずつ返す

	// -----------------------------------------------------
	//   通常探索のProbCutの処理のなかから呼び出される用
	// -----------------------------------------------------

	PROBCUT_TT,					// 置換表の指し手を返すフェーズ
	PROBCUT_INIT,				// (PROBCUTの指し手を生成)
	PROBCUT,					// 直前の指し手での駒の価値を上回る駒取りの指し手のみを生成するフェーズ

	// -----------------------------------------------------
	//   静止探索時用の指し手生成
	// -----------------------------------------------------

	QSEARCH_TT,					// 置換表の指し手を返すフェーズ
	QCAPTURE_INIT,				// (QCAPTUREの指し手生成)
	QCAPTURE,					// 捕獲する指し手 + 歩を成る指し手を一手ずつ返す
	QCHECK_INIT,				// 2026-10-17: (静かな王手の生成) QS_CHECKS==0 なら入らない
	QCHECK,						// 駒を取らない王手を一手ずつ返す
};

/*
状態遷移の順番は、
	王手がかかっていない時。
		通常探索時 : MAIN_TT    → CAPTURE_INIT  → GOOD_CAPTURE → GOOD_QUIET → BAD_CAPTURE → BAD_QUIET
		静止探索時 : QSEARCH_TT → QCAPTURE_INIT → QCAPTURE
		ProbCut時  : PROBCUT_TT → PROBCUT_INIT  → PROBCUT

		静止探索では、captureする指し手しか生成しないので単にスコア順に並び替えて順番に返せば良い。

	王手がかかっている時。
		通常探索、静止探索共通 : EVASION_TT → EVASION_INIT → EVASION

*/

// -----------------------
//   partial insertion sort
// -----------------------

// Sort moves in descending order up to and including
// a given limit. The order of moves smaller than the limit is left unspecified.
//
// partial_insertion_sort()は指し手を与えられたlimitより、ExtMove::valueが大きいものだけを降順でソートする。
// limitよりも小さい値の指し手の順序については、不定。(sortしたときに末尾のほうに移動する)
// 将棋だと指し手の数が多い(ことがある)ので、数が多いときは途中で打ち切ったほうがいいかも。
// 現状、全体時間の6.5～7.5%程度をこの関数で消費している。
// (長い時間思考させるとこの割合が増えてくる)
void partial_insertion_sort(ExtMove* begin, ExtMove* end, int limit) {

	for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; ++p)
		if (p->value >= limit)
		{
			ExtMove tmp = *p, *q;
			*p          = *++sortedEnd;
			for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
				// ■ 備考
				// ここ、ExtMove同士の operator<()を呼び出している。
				// これはExtMove::valueを比較するように定義されている。
				*q = *(q - 1);
			*q = tmp;
		}
}

} // end of namespace

// 指し手オーダリング器

// Constructors of the MovePicker class. As arguments, we pass information
// to decide which class of moves to emit, to help sorting the (presumably)
// good moves first, and how important move ordering is at the current node.

// MovePicker constructor for the main search and for the quiescence search

// MovePickerクラスのコンストラクタ。引数として、どの種類の手を生成するかを決定するための情報、
// どの手を優先的に（おそらく良い手を）ソートするか、そして現在のノードで手順の順序がどれほど重要かを渡します。

MovePicker::MovePicker(const Position&              p,
                       Move                         ttm,
                       Depth                        d,
                       const ButterflyHistory*      mh,
                       const LowPlyHistory*         lph,
                       const CapturePieceToHistory* cph,
                       const PieceToHistory**       ch,
                       const SharedHistories*       sh,
                       int                          pl,
                       const DropContHistory*       dch_,
                       int                          dropPrevType_,
                       const ThreatHistory*         thh_,
                       uint32_t                     threatCtx_
#if !STOCKFISH
                       ,bool generate_all_legal_moves
#endif
                       ) :
    pos(p),
    mainHistory(mh),
    lowPlyHistory(lph),
    captureHistory(cph),
    continuationHistory(ch),
    sharedHistory(sh),
    dropContHistory(dch_),
    dropPrevType(dropPrevType_),
    threatHistory(thh_),
    threatCtx(threatCtx_),
    ttMove(ttm),
    depth(d),
    ply(pl)
#if STOCKFISH
#else
    ,
    generate_all_legal_moves(generate_all_legal_moves)
#endif
{
    // 次の指し手生成の段階
    // 王手がかかっているなら王手回避のフェーズへ。さもなくばQSEARCHのフェーズへ。

#if STOCKFISH
	if (pos.in_check())
        // 王手がかかっているなら回避手
		stage = EVASION_TT + !(ttm && pos.pseudo_legal(ttm));

    else
        // 王手がかかっていないなら通常探索用/静止探索の指し手生成
        // ⇨ 通常探索から呼び出されたのか、静止探索から呼び出されたのかについてはdepth > 0 によって判定できる。
        stage = (depth > 0 ? MAIN_TT : QSEARCH_TT)
              + !(ttm && pos.pseudo_legal(ttm));
#else
	// 🌈 やねうら王では、pos.pseudo_legal()にgenerate_all_legal_movesを渡してやる必要がある。

    if (pos.in_check())
        // 王手がかかっているなら回避手
        stage = EVASION_TT + !(ttm && pos.pseudo_legal(ttm, generate_all_legal_moves));

    else
        // 王手がかかっていないなら通常探索用/静止探索の指し手生成
        // ⇨ 通常探索から呼び出されたのか、静止探索から呼び出されたのかについてはdepth > 0 によって判定できる。
        stage = (depth > 0 ? MAIN_TT : QSEARCH_TT)
              + !(ttm && pos.pseudo_legal(ttm, generate_all_legal_moves));
#endif

	// ⇨ Stockfish 16のコード、ttm(置換表の指し手)は無条件でこのMovePickerが返す1番目の指し手としているが、これだと
    //    TTの指し手だけで千日手になってしまうことがある。これは、将棋ではわりと起こりうる。
    //    対策としては、qsearchで千日手チェックをしたり、SEEが悪いならskipするなど。
    //  ※　ここでStockfish 14のころのように置換表の指し手に条件をつけるのは良くなさげ。(V7.74l3 と V7.74mとの比較)
    //  →　ただし、その場合、qsearch()で千日手チェックが必要になる。
    //    qsearchでの千日手チェックのコストが馬鹿にならないので、
    //    ⇓このコードを有効にして、qsearch()での千日手チェックをやめた方が得。

    // recaptureの制約なくす。(ただし、やねうら王ではttmは何らかの制約を課す)
    // →　この制約入れないと、TTの指し手だけで16手超えで循環されてしまうとqsearch()で
    //    is_repetition()入れたところで永久ループになる。

    // 置換表の指し手を優遇するコード。
    // depth > -5なら、TT優先。depth <= -5でもcaptureである制約。
    // 単にcapture()にするより、この制約にしたほうが良さげ。(V775a7 vs V775a8)

    // ↓これ参考に変更したほうがよさげ？

    /*
	stage = (pos.in_check() ? EVASION_TT : QSEARCH_TT) +
		!(ttm
			&& (pos.in_check() || depth > -5 || pos.capture(ttm))
			&& pos.pseudo_legal(ttm));
*/

    // 置換表の指し手があるならそれを最初に試す。ただしpseudo_legalでなければならない。
    // 置換表の指し手がないなら、次のstageから開始する。
}

// MovePicker constructor for ProbCut: we generate captures with Static Exchange
// Evaluation (SEE) greater than or equal to the given threshold.

// 通常探索のProbCutのためのMovePickerコンストラクタ
// : 与えられた閾値以上の静的交換評価（SEE）を持つキャプチャを生成します。
// th = 枝刈りのしきい値
// ⇨ SEEの値がth以上となるcaptureの指し手(歩の成りは含む)だけを生成する。

MovePicker::MovePicker(const Position& p, Move ttm, int th, const CapturePieceToHistory* cph
#if STOCKFISH
#else
    , bool generate_all_legal_moves
#endif

                       ) :
    pos(p),
    captureHistory(cph),
    ttMove(ttm),
    threshold(Value(th))
#if STOCKFISH
#else
    ,generate_all_legal_moves(generate_all_legal_moves)
#endif

    {

    // ProbCutから呼び出されているので王手はかかっていないはず。
    ASSERT_LV3(!pos.in_check());

    // ProbCutにおいて、SEEが与えられたthresholdの値以上の指し手のみ生成する。
    // (置換表の指し手も、この条件を満たさなければならない)
    // 置換表の指し手がないなら、次のstageから開始する。

    stage = PROBCUT_TT
      + !(
        ttm
#if STOCKFISH        
        && pos.capture_stage(ttm)
        && pos.pseudo_legal(ttm) 
#else
        && pos.capture(ttm)
        // 注意 : ⇑ ProbCutの指し手生成(PROBCUT_INIT)で、
        // 歩の成りも生成するなら、ここはcapture_or_pawn_promotion()、しないならcapture()にすること。
        // ただし、TTの指し手は優遇した方が良い可能性もある。
        && pos.pseudo_legal(ttm, generate_all_legal_moves) 
#endif
	    );
    // ⇨ qsearch()のTTと同様、置換表の指し手に関してはsee_geの条件、
    // つけないほうがいい可能性があるが、やってみたら良くなかった。(V774v2 vs V774v3)
}

// Assigns a numerical value to each move in a list, used for sorting.
// Captures are ordered by Most Valuable Victim (MVV), preferring captures
// with a good history. Quiets moves are ordered using the history tables.

// 各手に数値を割り当ててリストをソートします。
// キャプチャは最も価値のある駒（MVV）に基づいて順序付けられ、
// 良好な履歴を持つキャプチャが優先されます。
// 静かな手は履歴テーブルを使用して順序付けられます。

// QUIETS、EVASIONS、CAPTURESの指し手のオーダリングのためのスコアリング。似た処理なので一本化。
#if defined(P2_SRCH_BUNDLE_A) && !defined(P2_MP_PREFETCH)
// 束 A (L1): quiet 採点の速い経路。 score<QUIETS / QUIETS_ALL> から、 次の条件がすべて成り立つ
//   node でだけ呼ばれる (本番設定では常に成り立つ。 option は fixed_options.txt / floodgate toml):
//     policyScale <= 0      (POLICY_HINT_SCALE 既定 0)
//     !useDropHist          (DROP_CONT_HIST_SCALE 既定 0)
//     threatHistory == null (本番 toml THREAT_HISTORY_SCALE = 0)
//     migiBonus <= 0        (MIGI_MOVEORDER_SCALE 既定 0)
//     ply >= LOW_PLY_HISTORY_SIZE (lowPlyHistory の項が付かない)
//   このとき元のループで残る項は mainHistory / pawnHistory / contHist 5 本 / 王手ボーナスだけで、
//   ここではそれを**同じ式・同じ足し順**で計算する (= 採点値は完全に同一。 VERIFY で 1 手ごとに照合)。
//   違いは、 基底ポインタと係数を node ごとに 1 回だけ読んでローカル const に持つこと。
//   元のループでは ExtMove (int) への書き込みが global の係数や history の基底と別名になりうると
//   見なされ、 手ごとに読み直されていた。 noinline にしてこの小さなループだけでレジスタを割り当てる。
//   不変条件: ループ中に係数 (SPSA::*) と history の表は書き換わらない (採点は読むだけ)。
template<GenType Type>
ExtMove* MovePicker::score_quiets_fast(const MoveList<Type>& ml) {
    const Color           us         = pos.side_to_move();
    const auto&           mainHist   = (*mainHistory)[us];
    const auto&           pawnHist   = sharedHistory->pawn_entry(pos);
    const PieceToHistory& ch0        = *continuationHistory[0];
    const PieceToHistory& ch1        = *continuationHistory[1];
    const PieceToHistory& ch2        = *continuationHistory[2];
    const PieceToHistory& ch3        = *continuationHistory[3];
    const PieceToHistory& ch5        = *continuationHistory[5];
    const int             mainMul    = SPSA::MP_QUIET_MAINHIST_MULT_Q8;
    const int             pawnMul    = SPSA::MP_QUIET_PAWNHIST_MULT_Q8;
    const int             checkTh    = SPSA::MP_QUIET_CHECK_SEE_TH;
    const int             checkBonus = SPSA::MP_QUIET_CHECK_BONUS;

    ExtMove* it = cur;
    for (auto move : ml)
    {
        ExtMove& m = *it++;
        m          = move;

        const Square    to = m.to_sq();
        const Piece     pc = pos.moved_piece(m);
        const PieceType pt = type_of(pc);

        int quietScoreQ8 = mainMul * int(mainHist[m.raw()]);
        quietScoreQ8 += pawnMul * int(pawnHist[pc][to]);
        quietScoreQ8 += SPSA::SCORE_SCALE * int(ch0[pc][to]);
        quietScoreQ8 += SPSA::SCORE_SCALE * int(ch1[pc][to]);
        quietScoreQ8 += SPSA::SCORE_SCALE * int(ch2[pc][to]);
        quietScoreQ8 += SPSA::SCORE_SCALE * int(ch3[pc][to]);
        quietScoreQ8 += SPSA::SCORE_SCALE * int(ch5[pc][to]);
        quietScoreQ8 += ((pos.check_squares(pt) & to) && pos.see_ge(m, checkTh))
                      * SPSA::SCORE_SCALE * checkBonus;

        m.value = quietScoreQ8 / SPSA::SCORE_SCALE;
    }
    return it;
}
#endif

template<GenType Type>
ExtMove* MovePicker::score(const MoveList<Type>& ml) {

#if STOCKFISH
	static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "Wrong type");
#else
    static_assert(Type == CAPTURES || Type == CAPTURES_ALL || Type == QUIETS || Type == QUIETS_ALL
                    || Type == EVASIONS || Type == EVASIONS_ALL,
                  "Wrong type");
#endif

	Color us = pos.side_to_move();

	// 方針指数ヒント用の per-node 前計算。POLICY_HINT_SCALE==0(既定)なら一切計算せず
	// 従来挙動と完全一致(bit-exact)。QUIETS 以外の Type では未使用。
	int      policyScale       = 0;
	// 右玉誘導 (MIGI_MOVEORDER_SCALE) の per-node 値。 🚨scale==0 なら 0 のまま = bit-exact。
	//   ユーザー決定 2026-10-20「右玉の誘導は評価でなく MovePicker で / 先後とも押す」。
	int      migiBonus         = 0;   // 「右玉を完成させる玉の手」に足す値 (0 = 押さない)
	int      migiBoosted       = 0;   // MIGI_MOVEORDER_DEBUG 用 (加点した手の数)
	Square   policyOurKing     = SQ_NB;
	Square   policyEnemyKing   = SQ_NB;
	Bitboard policyEnemyPawnAtk = Bitboard(ZERO);

	// Drop Continuation History 用の per-node 前計算(DROP_CONT_HIST_SCALE==0 なら完全ゼロコスト)。
	int    dropScale     = 0;
	Square dropEnemyKing = SQ_NB;
	bool   useDropHist   = false;

	// 自分より価値の安い駒で当たりになっているか

#if STOCKFISH
    [[maybe_unused]] Bitboard threatByLesser[KING + 1];
	if constexpr (Type == QUIETS)
#else
    if constexpr (Type == QUIETS || Type == QUIETS_ALL)
#endif
	{
		// 方針指数の per-node 計算(scale>0 のときのみ = scale==0 は完全ゼロコスト)。
		policyScale = SPSA::POLICY_HINT_SCALE;

		// Drop Continuation History の per-node 前計算(policy とは独立。scale==0 ならゼロコスト)。
		dropScale = SPSA::DROP_CONT_HIST_SCALE;
		if (dropScale > 0 && dropContHistory != nullptr && dropPrevType >= 0)
		{
			useDropHist   = true;
			dropEnemyKing = pos.square<KING>(~us);
		}

		// ★右玉誘導の per-node 前計算 (2026-10-20)。 評価には足さず ordering にだけ使う。
		//   🚨🚨**policyScale のブロックの外に置くこと**。 2026-10-20 に内側に書いてしまい、
		//     POLICY_HINT_SCALE が既定 0 なので**一度も実行されず**、 陽性対照で
		//     bestmove が 1 手も変わらなかった (機構が死んでいるように見えた)。
		//   migigyoku_build_weight は「飛車は右・玉はまだ右でない」局面でのみ 0 以外を返し、
		//   🚨**既に右玉なら 0** (= 維持はしない / 崩すべきときに崩せるようにする)。
		//   角換わり条件と fade は評価側の option (MigigyokuKakugawari /
		//   MigigyokuFadeStart / MigigyokuFadeEnd) を共用する。
		//   SCALE=1024, weight=256 で +16384 = 王手ボーナス (16492) と同程度。
		//   SCALE==0 なら関数呼び出しもしない = ゼロコスト (bit-exact)。
		if (SPSA::MIGI_MOVEORDER_SCALE > 0)
		{
			const int migiW = Eval::P2NNUE::migigyoku_build_weight(pos, us);   // 0..256
			migiBonus = SPSA::MIGI_MOVEORDER_SCALE * migiW / 16;
			// root 局面 (ply==0) で発火を trace (MIGI_MOVEORDER_DEBUG 時のみ)。 関門の観測点。
			if (SPSA::MIGI_MOVEORDER_DEBUG && ply == 0)
				sync_cout << "info string migi_moveorder weight=" << migiW
				          << " bonus=" << migiBonus << sync_endl;
		}

		if (policyScale > 0)
		{
			policy            = compute_policy_index(pos);
			policyOurKing     = pos.square<KING>(us);
			policyEnemyKing   = pos.square<KING>(~us);
			policyEnemyPawnAtk = (~us == BLACK) ? pos.attacks_by<BLACK, PAWN>()
			                                    : pos.attacks_by<WHITE, PAWN>();
			// root 局面(ply==0)の方針指数を trace 出力(POLICY_DEBUG 時)。
			if (SPSA::POLICY_DEBUG && ply == 0)
				sync_cout << "info string policy attack=" << policy.attack
				          << " castle=" << policy.castle << " wait=" << policy.wait << sync_endl;
		}
#if STOCKFISH
        threatByLesser[PAWN]   = 0;
        threatByLesser[KNIGHT] = threatByLesser[BISHOP] = pos.attacks_by<PAWN>(~us);
        threatByLesser[ROOK] =
          pos.attacks_by<KNIGHT>(~us) | pos.attacks_by<BISHOP>(~us) | threatByLesser[KNIGHT];
        threatByLesser[QUEEN] = pos.attacks_by<ROOK>(~us) | threatByLesser[ROOK];
        threatByLesser[KING]  = 0;

#else

#if 0
		// 🌈　Stockfishのコードを忠実に実装すると将棋ではたくさんの利きを計算しなくてはならないので
		//     非常に計算コストが高くなる。ここでは歩による当たりになっている駒だけ考える。

		// 歩による脅威だけ。
		// squares threatened by pawns
		threatenedByPawn = (~us == BLACK) ? pos.attacks_by<BLACK, PAWN>() : pos.attacks_by<WHITE, PAWN>();

		// 歩以外の自駒で、相手の歩の利きにある駒
		threatened =  (pos.pieces(us,PAWN).andnot(pos.pieces(us))                 & threatenedByPawn );
#endif
		// →　やってみたが強くならないのでコメントアウトする。[2022/04/26]

#endif
	}

#if defined(P2_SRCH_BUNDLE_A) && !defined(P2_MP_PREFETCH)
    // 束 A (L1): 本番で常に偽の機能がすべて無効なら、 速い経路 (score_quiets_fast) で採点する。
    //   per-node の前計算 (上のブロック) は両経路で共通なので、 ここで分けても採点値は同一。
#if defined(P2_SRCH_BUNDLE_A_VERIFY)
    // VERIFY: 速い経路の結果を退避しておき、 元のループの結果と末尾で照合する。
    int      bundleFastVal[MAX_MOVES];
    ExtMove* bundleFastEnd = nullptr;
#endif
    if constexpr (Type == QUIETS || Type == QUIETS_ALL)
    {
        if (policyScale <= 0 && !useDropHist && threatHistory == nullptr && migiBonus <= 0
            && ply >= LOW_PLY_HISTORY_SIZE)
        {
#if defined(P2_SRCH_BUNDLE_A_VERIFY)
            bundleFastEnd = score_quiets_fast(ml);
            for (ExtMove* p = cur; p != bundleFastEnd; ++p)
                bundleFastVal[p - cur] = p->value;
#else
            return score_quiets_fast(ml);
#endif
        }
    }
#endif
    ExtMove* it = cur;
    // 2026-11-06 **パイプライン prefetch** の先読み位置 (設計 fable / 実装 opus)。
    //   手の配列 ml は採点より前に全部出来ているので、 PF_AHEAD 手先の history の行を先に触れる。
#if defined(P2_MP_PREFETCH)
    constexpr int PF_AHEAD = 4;
    auto       pf     = ml.begin();
    const auto pf_end = ml.end();
    for (int pfk = 0; pfk < PF_AHEAD && pf != pf_end; ++pfk) ++pf;
#endif
    for (auto move : ml)
    {
        ExtMove& m = *it++;
        m          = move;

		const Square    from          = m.from_sq();
		const Square    to            = m.to_sq();
		const Piece     pc            = pos.moved_piece(m);
		const PieceType pt            = type_of(pc);
		const Piece     capturedPiece = pos.piece_on(to);

		if constexpr (Type == CAPTURES)
		{
			// Position::see()を用いると遅い。単に取る駒の価値順に調べたほうがパフォーマンス的にもいい。
			// 歩が成る指し手もあるのでこれはある程度優先されないといけない。
			// CAPTURE系である以上、打つ指し手は除外されている。
			// CAPTURES_PRO_PLUSで生成しているので歩の成る指し手が混じる。
			
			// MVV-LVAだが、将棋ではLVAあんまり関係なさげだが(複数の駒である1つの駒が取れるケースがチェスより少ない)、
			// Stockfish 9に倣いMVV + captureHistoryで処理する。

			// ここに来るCAPTURESに歩の成りを含めているので、捕獲する駒(pos.piece_on(to_sq(m)))がNO_PIECEで
			// ある可能性については考慮しておく必要がある。
			// → Eval::CapturePieceValuePlusPromote()を用いて計算。
			// → しかしこのあとsee_ge()の引数に使うのだが、see_ge()ではpromotionの価値を考慮してないので、
			//    ここでpromotionの価値まで足し込んでしまうとそこと整合性がとれなくなるのか…。

			// ★Q8 固定小数化 (NAGISA V3.1 と同一の定式化。2026-09-03)
			//   旧: m.value = captureHistory + MULT * PieceValue
			//   新: captureScoreQ8 = SCORE_SCALE * captureHistory + MULT_Q8 * PieceValue
			//       m.value = captureScoreQ8 / SCORE_SCALE   (除算は最後の 1 回だけ)
			//   既定 MULT_Q8 = 7 * SCORE_SCALE なので分子は SCORE_SCALE の倍数 = bit-exact。
			//   オーバーフロー: captureHistory は ±約 1.1e4、 PieceValue は約 1.5e3 なので
			//   512 * 1.1e4 + 7168 * 1.5e3 ≒ 1.7e7 で int に十分収まる。
			const int captureScoreQ8 =
			  SPSA::SCORE_SCALE * int((*captureHistory)[pc][to][type_of(capturedPiece)])
			  + SPSA::MP_CAPTURE_PIECEVALUE_MULT_Q8 * int(Eval::PieceValue[capturedPiece]);
			m.value = captureScoreQ8 / SPSA::SCORE_SCALE;

			// →　係数を掛けてるのは、
			// このあと、GOOD_CAPTURE で、
			//	return pos.see_ge(*cur, Value(-cur->value))
			// のようにしてあって、要するにsee_ge()の時のスケール(PieceValue)に変換するため。
			// 
			// Stockfishとは駒点が異なるので、この部分の係数を調整する必要がある。
			//
		}
#if STOCKFISH
		else if constexpr (Type == QUIETS)
#else
        else if constexpr (Type == QUIETS || Type == QUIETS_ALL)
#endif
		{
			// 2026-11-06 **パイプライン prefetch** (設計 fable / 実装 opus)。
			//   quiet 1 手の採点で引く random line は contHist **x5**
			//   (表 **53.7 MB** [実測 sizeof] / PieceToHistory 5,184 B / 行 pc = 162 B)
			//   + mainHistory (256 KB)。 contHist は部分木の根で表が切り替わるので cold。
			//   -> **PF_AHEAD 手先の行**を先に触っておく。
			//   🚨値は読まないので **bit-exact** (prefetch に副作用は無い)。
			//   🚨一括で増やすのは逆効果 (FT の pfall は -6.9%)。 ここは 1 手につき 6 本だけ。
#if defined(P2_MP_PREFETCH)
			if (pf != pf_end)
			{
				const auto   nm  = *pf;
				const Square nto = nm.to_sq();
				const Piece  npc = pos.moved_piece(nm);
				prefetch((void*)&(*continuationHistory[0])[npc][nto]);
				prefetch((void*)&(*continuationHistory[1])[npc][nto]);
				prefetch((void*)&(*continuationHistory[2])[npc][nto]);
				prefetch((void*)&(*continuationHistory[3])[npc][nto]);
				prefetch((void*)&(*continuationHistory[5])[npc][nto]);
				prefetch((void*)&(*mainHistory)[us][nm.raw()]);
				++pf;
			}
#endif

			// 駒を取らない指し手をオーダリングする。
			// ここ、歩以外の成りも含まれているのだが…。
			// →　指し手オーダリングは、quietな指し手の間での優劣を付けたいわけで、
			//    駒を成るような指し手はどうせevaluate()で大きな値がつくからそっちを先に探索することになる。

			// ★Q8 固定小数化 (NAGISA V3.1 と同一の定式化。2026-09-03)
			//   quietScoreQ8 に「実効値 × SCORE_SCALE」を積み上げ、最後に 1 回だけ
			//   SCORE_SCALE で割る。Q8 化するのは mainHistory / pawnHistory の 2 係数のみで、
			//   他の項は SCORE_SCALE を掛けて単位を合わせる (実効値は不変)。
			//   途中に丸めは入らないので (Σ SCORE_SCALE*x) / SCORE_SCALE == Σ x が厳密に成立し、
			//   既定値では旧挙動と bit-exact。
			//   オーバーフロー: 各 history は高々 ±3e4、 SCORE_SCALE=512 で ±1.5e7。
			//   7 項 + 王手ボーナス (512*32768 = 1.7e7) を足しても 1.3e8 程度で int に収まる。
			int quietScoreQ8  = SPSA::MP_QUIET_MAINHIST_MULT_Q8 * int((*mainHistory)[us][m.raw()]);
			quietScoreQ8 += SPSA::MP_QUIET_PAWNHIST_MULT_Q8
			              * int(sharedHistory->pawn_entry(pos)[pc][to]);
			quietScoreQ8 += SPSA::SCORE_SCALE * int((*continuationHistory[0])[pc][to]);
			quietScoreQ8 += SPSA::SCORE_SCALE * int((*continuationHistory[1])[pc][to]);
			quietScoreQ8 += SPSA::SCORE_SCALE * int((*continuationHistory[2])[pc][to]);
			quietScoreQ8 += SPSA::SCORE_SCALE * int((*continuationHistory[3])[pc][to]);
			quietScoreQ8 += SPSA::SCORE_SCALE * int((*continuationHistory[5])[pc][to]);

			// 駒打ち専用 Drop Continuation History(useDropHist は QUIETS かつ scale>0 かつ前手ありのときのみ真)。
			// pt = type_of(pc) = 打つ駒種、to = 打つ升。既存 contHist とは別テーブルで独立に加算。
			// ★/400 は Q8 化前の実効値のまま計算してから SCORE_SCALE を掛ける (丸めを変えないため)。
			if (useDropHist && m.is_drop())
				quietScoreQ8 += SPSA::SCORE_SCALE
				              * ((*dropContHistory)[drop_cont_index(us, pt, PieceType(dropPrevType), to, dropEnemyKing)]
				                 * dropScale / 400);

			// Threat Continuation History(threatHistory!=nullptr = THREAT_HISTORY_SCALE>0 かつ脅威あり)。
			// threatMove 文脈で成功しやすい quiet response を薄く加点(/8 固定)。TT手/成り response は対象外。
			// ★/8 も Q8 化前の実効値のまま計算してから SCORE_SCALE を掛ける。
			if (threatHistory && m != ttMove && !m.is_promote())
				quietScoreQ8 += SPSA::SCORE_SCALE
				              * ((*threatHistory)[threat_history_index(threatCtx, pc, to)] / 8);

			// bonus for checks
			// ★MP_QUIET_CHECK_SEE_TH / MP_QUIET_CHECK_BONUS は NAGISA でも Q8 化されていない。
			quietScoreQ8 += ((pos.check_squares(pt) & to)
			                 && pos.see_ge(m, SPSA::MP_QUIET_CHECK_SEE_TH))
			              * SPSA::SCORE_SCALE * SPSA::MP_QUIET_CHECK_BONUS;

			m.value = quietScoreQ8 / SPSA::SCORE_SCALE;

			// これ、効果があるのか検証したほうが良さげ。

			// ★方針指数ヒント(POLICY_HINT_SCALE>0 のときのみ、探索順序の微調整)。
			//   eval には加算せず m.value(ordering スコア)にのみ薄く混ぜる。history(±数千)や
			//   check bonus(16384) を圧倒しないよう上限を設ける。TT/capture/killer は別 stage で不変。
			if (policyScale > 0)
			{
				int pol = 0;
				// 攻め: 敵玉へ近づく / 敵陣(成れる段)へ入る / 成る quiet 手。
				if (policy.attack > 0)
				{
					const bool towardEnemyKing =
					  !m.is_drop() && dist(to, policyEnemyKing) < dist(from, policyEnemyKing);
					const bool intoEnemyCamp = canPromote(us, to);
					if (towardEnemyKing || intoEnemyCamp || m.is_promote())
						pol += policy.attack;
				}
				// 受け: 玉移動 / 金銀を自玉へ寄せる / 自玉近傍への駒打ち。
				if (policy.castle > 0)
				{
					const bool kingMove = (pt == KING);
					const bool reinforce = !m.is_drop() && (pt == GOLD || pt == SILVER)
					                     && dist(to, policyOurKing) < dist(from, policyOurKing);
					const bool dropNearKing = m.is_drop() && dist(to, policyOurKing) <= 2;
					if (kingMove || reinforce || dropNearKing)
						pol += policy.castle;
				}
				// 手待ち: 相手歩の利きに自駒を晒さない(悪化しにくい)整備手。
				if (policy.wait > 0 && !(policyEnemyPawnAtk & to))
					pol += policy.wait;

				if (pol > 1000)
					pol = 1000;                    // 上限クランプ
				m.value += pol * policyScale / 50; // POLICY_HINT_SCALE=100 で最大 +2000 程度
			}

			// ★右玉誘導 (2026-10-20)。 玉を**右玉の完成位置** (先手 2-3 筋 / 後手 7-8 筋) に
			//   近づける quiet な玉の手だけに加点する。
			//   🚨🚨2026-10-20 修正: 旧実装は FILE_4 / FILE_6 (= 先手 1-4 筋) を使っていた。
			//     平手の先手玉は 5 筋なので **4 筋へ動くだけで条件を満たし**、 しかも
			//     p2nnue 側の完成判定も同じ閾値だったので**そこでボーナスが消えて止まった**。
			//     実測: 右玉率が 基準 4.0% -> **0.0% (0/63 局)** に下がった。
			//   -> 閾値は p2nnue.cpp の **migi_mo_done** と揃える (先手 FILE_3 以下 /
			//      後手 FILE_7 以上)。 **片方を変えたら両方変えること**。
			//   🚨migiBonus>0 になるのは「飛車は右・玉は未完成・角換わり・fade 内」の局面だけ。
			//   🚨玉は打てないので is_drop() は実質不要だが、 意図を明示するために残す。
			//   🚨🚨2026-10-20 **再修正**: 条件を「完成位置に**入る**手」から
			//     「**右に近づく**手」に変えた。 玉は 1 マスずつしか動かないので、
			//     玉 5 筋から完成位置 (2-3 筋) へは **2-3 手**かかる。
			//     「入る手」だけに絞ると**途中の手が押されず誘導が働かない**
			//     (陽性局面が 300 件中 **0 件**になって発覚した)。
			//   -> 加点は「右に近づく手」/ 完成の判定 (p2nnue の migi_mo_done) は
			//      「2-3 筋」のまま。 これで 5->4->3 と押し続け、 3 筋で止まる。
			if (migiBonus > 0 && pt == KING && !m.is_drop()
			    && ((us == BLACK) ? (file_of(to) < file_of(from))
			                      : (file_of(to) > file_of(from))))
			{
				m.value += migiBonus;
				++migiBoosted;
			}

#if STOCKFISH
			// penalty for moving to a square threatened by a lesser piece
			// or bonus for escaping an attack by a lesser piece.

			// 格下の駒に脅かされているマスに移動する際のペナルティ  
			// または格下の駒による攻撃から逃れる際のボーナス

			//  📓 移動元の駒が安い駒で当たりになっている場合、
			//      移動させることでそれを回避できるなら価値を上げておく。

            int v = 20 * (bool(threatByLesser[pt] & from) - bool(threatByLesser[pt] & to));
            m.value += PieceValue[pt] * v;

			// → Stockfishのコードそのままは書けない。
#endif

			// lowPlyHistoryも加算
			// ★Q8 固定小数化 (NAGISA V3.1 と同一の定式化)。
			//   旧: m.value += MULT * lph / (1 + ply)
			//   新: m.value += MULT_Q8 * lph / (SCORE_SCALE * (1 + ply))
			//   MULT_Q8 = MULT * SCORE_SCALE なので有理数として完全に同一 → 切り捨ても同一 (bit-exact)。
			//   除数は元から (1+ply) で idiv なので追加コストなし。
			//   オーバーフロー: lph は高々 ±8192、 MULT_Q8 は最大 8192 → 6.7e7 で int に収まる。
			if (ply < LOW_PLY_HISTORY_SIZE)
                m.value += SPSA::MP_LOW_PLY_HISTORY_MULT_Q8 * int((*lowPlyHistory)[ply][m.raw()])
                           / (SPSA::SCORE_SCALE * (1 + ply));
			
		}
		else // Type == EVASIONS || EVASIONS_ALL
		{
			// 王手回避の指し手をスコアリングする。
			if (pos.capture_stage(m))
				m.value = PieceValue[capturedPiece] + (1 << 28);
			/* 📓 捕獲する指し手に関しては簡易SEE + MVV/LVA
				  
				  被害が小さいように、LVA(価値の低い駒)を動かして取ることを
				  優先されたほうが良いので駒に価値の低い順に番号をつける。
				  そのためのテーブル。

				  💡 LVA = Least Valuable Aggressor。cf.MVV-LVA

				ここ、moved_piece_before()を用いるのが正しい。
				そうしておかないと、同じto,fromである角成りと角成らずの2つの指し手がある時、
				moved_piece_after()だと、角成りの方は、取られた時の損失が多いとみなされてしまい、
				オーダリング上、後回しになってしまう。

				⇑これは、captureの指し手のスコアがそうでない指し手のスコアより
				常に大きくなるようにするための下駄履き。
				
				　captureの指し手の方がそうでない指し手より稀なので、この下駄履きは、captureの時にしておく。
			*/

			else
				// それ以外の指し手に関してはhistoryの値の順番
				m.value = (*mainHistory)[us][m.raw()] + (*continuationHistory[0])[pc][to];
		}
	}
#if defined(P2_SRCH_BUNDLE_A) && defined(P2_SRCH_BUNDLE_A_VERIFY) && !defined(P2_MP_PREFETCH)
    if (bundleFastEnd)
    {
        BundleA::check(BundleA::V_L1_QUIET, bundleFastEnd - cur, it - cur);
        for (ExtMove* p = cur; p != it; ++p)
            BundleA::check(BundleA::V_L1_QUIET, bundleFastVal[p - cur], p->value);
    }
#endif
    return it;
}

// Returns the next move satisfying a predicate function.
// This never returns the TT move, as it was emitted before.

// 条件を満たす次の手を返します。
// この関数は、トランスポジションテーブル（TT）の手は既に出力されているため、決して返しません。

// ※　この関数の返し値は同時にthis->moveにも格納されるので活用すると良い。filterのなかでも
//   この変数にアクセスできるので、指し手によってfilterするかどうかを選べる。

template<typename Pred>
Move MovePicker::select(Pred filter) {

	for (; cur < endCur; ++cur)
		// filter()のなかで*curにアクセスして判定するのでfilter()は引数を取らない。
		if (*cur != ttMove && filter())
			return *cur++;

	return Move::none();
}

// This is the most important method of the MovePicker class. We emit one
// new pseudo-legal move on every call until there are no more moves left,
// picking the move with the highest score from a list of generated moves.

// これはMovePickerクラスで最も重要なメソッドです。呼び出すたびに、新しい擬似合法手を1つ生成し、
// すべての手が尽きるまで続けます。生成された手のリストから、最も高いスコアを持つ手を選びます。

// ※ 置換表の指し手(ttMove)を返したあとは、それを取り除いた指し手を返す。
// skipQuiets : これがtrueだとQUIETな指し手は返さない。
Move MovePicker::next_move() {

	// 💡 good Quietの閾値
	const int goodQuietThreshold = SPSA::MP_GOOD_QUIET_THRESHOLD;

top:
    switch (stage)
    {

    // 置換表の指し手を返すフェーズ
    case MAIN_TT :
    case EVASION_TT :
    case QSEARCH_TT :
    case PROBCUT_TT :
        ++stage;
        return ttMove;

    // 置換表の指し手を返したあとのフェーズ
    case CAPTURE_INIT :
    case PROBCUT_INIT :
    case QCAPTURE_INIT : {

#if STOCKFISH
        MoveList<CAPTURES> ml(pos);

        cur = endBadCaptures = moves;

        // 駒を捕獲する指し手に対してオーダリングのためのスコアをつける
        endCur = endCaptures = score<CAPTURES>(ml);
#else
        if (generate_all_legal_moves)
        {
            MoveList<CAPTURES_ALL> ml(pos);
            cur = endBadCaptures = moves;
            endCur = endCaptures = score<CAPTURES_ALL>(ml);
        }
        else
        {
            MoveList<CAPTURES> ml(pos);
            cur = endBadCaptures = moves;
            endCur = endCaptures = score<CAPTURES>(ml);
        }
#endif

        // captureの指し手はそんなに数多くないので全数ソートで問題ないし、全数ソートした方が良い。
        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());

        ++stage;
        goto top;
    }

	// 置換表の指し手を返したあとのフェーズ
	// (killer moveの前のフェーズなのでkiller除去は不要)
	case GOOD_CAPTURE:
		if (select([&]() {
				// moveは駒打ちではないからsee()の内部での駒打ちは判定不要だが…。
#if defined(P2_SRCH_BUNDLE_A)
				if (pos.see_ge(*cur, BundleA::vdiv(BundleA::goodCaptureSeeDiv, -cur->value,
				                                   SPSA::GOOD_CAPTURE_SEE_DIV, BundleA::V_L2_GOODCAP)))
#else
				if (pos.see_ge(*cur, -cur->value / SPSA::GOOD_CAPTURE_SEE_DIV))
#endif
					return true;
				std::swap(*endBadCaptures++, *cur);
				// 損をする捕獲する指し手はあとのほうで試行されるようにendBadCapturesに移動させる
				return false;
			}))
			return *(cur - 1);

		++stage;
		[[fallthrough]];

	// 駒を捕獲しない指し手を生成してオーダリング
	case QUIET_INIT:

		if (!skipQuiets)
		{
			/*
			moves          : バッファの先頭
			endBadCaptures : movesから(endBadCaptures - 1) までに bad capturesの指し手が格納されている。

				■ quiet(captures)の指し手を生成した直後。

				|--- 指し手生成用のバッファ ---------------------------------------|
				|    quiet(captures)                        |  未使用のバッファ    |
				|------------------------------------------------------------------|
				↑                                          ↑
				moves = endBadCaptures                    endMoves


				■ quiet(captures)の指し手を生成して、curポインタをインクリメントしながら読み進めているとき。

				|--- 指し手生成用のバッファ ---------------------------------------|
				|   badCaptures |  quiet(captures)           |  未使用のバッファ   |
				|------------------------------------------------------------------|
				↑             ↑              ↑            ↑
				moves      endBadCaptures      cur         endMoves


				■ quiet(captures)の指し手をpartial_sortで並び替え後にある程度スコアがいい指し手を返したとき

				残りの悪いquietの指し手の先頭をbeginBadQuietsにして、curはmovesに移動。これで、badCaptureを処理する。
				そのあと、beginBadQuietsを処理する。


				|--- 指し手生成用のバッファ ---------------------------------------|
				|   badCaptures |  quiet(captures)           |  未使用のバッファ   |
				|------------------------------------------------------------------|
				↑             ↑              ↑            ↑
				moves      endBadCaptures    beginBadQuiets endMoves
				 ↑cur

			*/

#if STOCKFISH
            MoveList<QUIETS> ml(pos);

            endCur = endGenerated = score<QUIETS>(ml);
#else
			if (generate_all_legal_moves)
			{
                MoveList<QUIETS_ALL> ml(pos);
                endCur = endGenerated = score<QUIETS_ALL>(ml);
			}
			else
			{
                MoveList<QUIETS> ml(pos);
                // 駒を捕獲しない指し手に対してオーダリングのためのスコアをつける
                endCur = endGenerated = score<QUIETS>(ml);
			}
            // ⚠ ここ⇑、CAPTURE_INITで生成した指し手に歩の成りの指し手が
			//     含まれているなら、それを除外しなければならない。
#endif						

			/*
				📓

					指し手を部分的にソートする。depthに線形に依存する閾値で。
					(depthが低いときに真面目に全要素ソートするのは無駄だから)
			
					将棋では平均合法手は100手程度。(以前は80手程度だったが、
					AI同士の対局では終局までの平均手数が伸びたので相対的に
					終盤が多くなり、終盤は手駒を持っていることが多いから、
					そのため平均合法手が増えた。)

					また、合法手の最大は、593手。
			 
					それに対して、チェスの平均合法手は40手、合法手の最大は、218手と言われている。
			
					insertion sortの計算量は、O(n^2) で、将棋ではわりと悩ましいところ。
					sortする個数が64以上などはquick sortに切り替えるなどした方がいい可能性もある。
			*/


			partial_insertion_sort(cur, endCur, SPSA::QUIET_PARTIAL_SORT_DEPTH_MULT * depth);
		}

		++stage;
		[[fallthrough]];

		// 駒を捕獲しない指し手を返す。
		// (置換表の指し手とkillerの指し手は返したあとなのでこれらの指し手は除外する必要がある)
		// ※　これ、指し手の数が多い場合、AVXを使って一気に削除しておいたほうが良いのでは..
	case GOOD_QUIET:
        if (!skipQuiets && select([&]() { return cur->value > goodQuietThreshold; }))
            return *(cur - 1);

		// Prepare the pointers to loop over the bad captures
		// bad capturesの指し手を返すためにポインタを準備する。
		// 📝　bad capturesの先頭を指すようにする。これは指し手生成バッファの先頭からの領域を再利用している。

		cur    = moves;
        endCur = endBadCaptures;

		++stage;
		[[fallthrough]];

		// see()が負の指し手を返す。
	case BAD_CAPTURE:
		if (select([]() { return true; }))
			return *(cur - 1);

		// Prepare the pointers to loop over the bad quiets
		// 悪いquietの手をループするためのポインタを準備します

        cur    = endCaptures;
		endCur = endGenerated;

		++stage;
		[[fallthrough]];

	case BAD_QUIET:
		if (!skipQuiets)
            return select([&]() { return cur->value <= goodQuietThreshold; });

		return Move::none();

	// 王手回避手の生成
    case EVASION_INIT : {
#if STOCKFISH
        MoveList<EVASIONS> ml(pos);

        cur    = moves;
        endCur = endGenerated = score<EVASIONS>(ml);
#else
		if (generate_all_legal_moves)
		{
            MoveList<EVASIONS_ALL> ml(pos);
            cur    = moves;
            endCur = endGenerated = score<EVASIONS_ALL>(ml);
		}
		else
		{
            MoveList<EVASIONS> ml(pos);
            cur    = moves;
			// 王手を回避する指し手に対してオーダリングのためのスコアをつける
            endCur = endGenerated = score<EVASIONS>(ml);
		}
#endif

        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());

        ++stage;
        [[fallthrough]];
    }
		// 王手回避の指し手を返す
	case EVASION:
		// そんなに数は多くないはずだから、オーダリングがベストのスコアのものを選択する
		return select([]() { return true; });

		// 2026-10-17: 静止探索の駒取りが尽きたところ。
		//   SPSA::QS_CHECKS が 1 で、 かつ qsearch の**入口の 1 層** (depth >= DEPTH_QS) のときだけ
		//   「駒を取らない王手」の段へ進む。 既定 0 なら Move::none() を返して従来と bit-exact。
	case QCAPTURE:
		if (select([]() { return true; }))
			return *(cur - 1);
		if (!(SPSA::QS_CHECKS && depth >= DEPTH_QS))
			return Move::none();
		++stage;
		[[fallthrough]];

	case QCHECK_INIT: {
		// 順序付けはしない (score<> が QUIET_CHECKS を受け付けないための最小形)。
		cur = moves;
		ExtMove* w = moves;
		if (generate_all_legal_moves)
		{
			MoveList<QUIET_CHECKS_ALL> ml(pos);
			for (const auto& m : ml) { *w = m; w->value = 0; ++w; }
		}
		else
		{
			MoveList<QUIET_CHECKS> ml(pos);
			for (const auto& m : ml) { *w = m; w->value = 0; ++w; }
		}
		endCur = w;
		++stage;
		goto top;
	}

	case QCHECK:
		return select([]() { return true; });

		// PROBCUTの指し手を返す
	case PROBCUT:
		return select([&]() { return pos.see_ge(*cur, threshold); });
		// threadshold以上のSEE値で、ベストのものを一つずつ返す

	default:
		UNREACHABLE;
		return Move::none();
	}

	ASSERT(false);
	return Move::none(); // Silence warning
}

void MovePicker::skip_quiet_moves() { skipQuiets = true; }

} // namespace YaneuraOu

#endif // defined(USE_MOVE_PICKER)
