#ifndef BOOKAWARE_H_INCLUDED
#define BOOKAWARE_H_INCLUDED

// ============================================================================
//  BookAware: 定跡反映型探索 (定跡を「指す」のではなく「避ける / 向かう」ために使う)
//  設計 runs/prereg/DESIGN-20260925-BOOKAWARE.md   2026-09-25
//
//  root だけで、 各 root 手の**子局面**が
//    avoid  定跡に載っていれば  -B     (相手が min-max 済み = 準備がある土俵を避ける)
//    prefer 定跡に載っていれば  +B     (我々が誘導したい形へ向かう)
//  を探索値に足して argmax を取る。 探索本体・評価関数・TT・枝刈りには触らない。
//
//  🚨方針 (ユーザー 2026-09-25):
//    ・定跡知識を「避ける」ために使うのは OK
//    ・定跡手をそのまま**再生する目的では使わない** (USI_OwnBook=false のまま。
//      定跡の指し手・評価値・PV は一切読まない。 **載っているかどうか (membership) だけ**)
//    ・**有力な手が定跡手しかなければ最終的に指す**
//      ⇔ max v(定跡手) - max v(非定跡手) > B  ⇔ 上の argmax そのもの
//    → つまみは B 1 本だけ。 B=0 で現行と bit-exact。
//
//  🚨MemoryBook / BookMoveSelector には触らない。 それらの find() は
//    IgnoreBookPly=false のとき ply 欄の不一致で全ミスする (book.cpp:790) ので、
//    membership 専用の二分探索を自前で持つ。
// ============================================================================

#include "../types.h"

// 🏁🏁**2026-10-27: 理論離脱 (BookAware) は不採用**。 凍結の exe には**含めない**。
//   根拠 (設計 runs/prereg/DESIGN-20261027-THEORYEXIT-V3.md):
//     ①cp→pt の換算が 5 倍甘かった — 同ノード 200K・互角の対局では **1 pt ≈ 5 cp (s≈120)**。
//       真の損 45.5cp は 1.9pt ではなく **≈9pt**。
//     ②「理論の外」は**値の等高線**だった — 逸脱手の 80% は ybb の登録手 (上から 2-3 番目) で、
//       ペタショック自身が「中央 +69 / 平均 +85cp 悪い」と値を付けた手。
//       「子が ybb に無い手」≡「≥約 70cp 悪い手」なので、 **安く降りる手は構造的に存在しない**。
//     実測 161 対で **-8.1pt** (2*SE 11.1) / 候補 5 本 (B↓ / Attempts=1 / ybb 2 番手 /
//     prefer 限定 / ゲート強化) は全部「1 回の損 ≈ -7pt 以上」対「利益 ≤ +1.5pt」の比を変えない。
//
//   🎯**コードは残す** (再現と記録のため / アピール文書で参照する)。 **コンパイルしないだけ**。
//   ✅再開するときは下の `#if 0 &&` の `0 &&` を外す。
//   🚨撤去の理由は「害があるから」ではなく (既定 0 で bit-exact 300/300 を確認済)、
//     **凍結する exe に使わない機構を入れない** (i-cache / PGO のプロファイルを汚さない) ため。
#if 0 && defined(USE_SFEN_PACKER)
	#define USE_BOOK_AWARE
#endif

#if defined(USE_BOOK_AWARE)

#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "../position.h"
#include "../usi.h"

namespace YaneuraOu {

namespace BookAware {

// 定跡の役割。
enum class Role {
	Avoid,   // 相手の準備がある = 避ける (-B)
	Prefer,  // 我々が向かいたい形   (+B)
};

// 1 本の定跡が持つ属性。 §6.2 で「本ごと」に置くと決めたものだけ。
struct BookAttr {
	Role        role   = Role::Avoid;
	// side: COLOR_NB のときは both (手番を問わない)。
	Color       side   = COLOR_NB;
	int         maxply = 40;
	std::string path;
};

// 定跡 1 本の membership 判定子。
class BookSet {
public:
	explicit BookSet(const BookAttr& a) : attr(a) {}
	virtual ~BookSet() {}

	// この局面が載っているか。 **これしか使わない** (指し手も評価値も読まない)。
	virtual bool contains(const PackedSfen& ps) const = 0;

	// 読み込んだ局面数 (ログ用。 ybb は全 record 数)。
	virtual u64 size() const = 0;

	// "ybb" / "mem" (ログ用)。
	virtual const char* mode() const = 0;

	// この定跡が手番 us に対して有効か。
	bool active_for(Color us) const { return attr.side == COLOR_NB || attr.side == us; }

	const BookAttr attr;
};

// 「相手が定跡を辿っている証拠」のゲート (設計 §15)。
//   🚨ユーザー指摘「相手が定跡手順じゃない場合、 逆効果になっている」への対応。
//   root が定跡に載っていることは**相手が準備している証拠にならない** (我々自身が定跡なしで
//   96% の局を 60 手目まで定跡内で指す)。 相手が自力で考えているときに avoid を効かせると、
//   「−B まで悪い手を選ぶ損」だけが残る。
//   → **相手の消費時間**で判定する。 定跡から指す相手は即応する。
//   τ = 0.2 × ref (ref = 増分 / 秒読み / 残り時間の 1/50)。 実測で定跡あり群と無し群が完全に分離。
struct GateState {
	bool      have_last = false;   // 前回の観測があるか
	TimePoint last_them = 0;       // 前回観測時の相手の残り時間
	int       obs       = 0;       // 観測できた「相手の 1 手」の区間数
	bool      booking   = false;   // 現在ゲートが開いているか
	// 🎯相手の手数で索引する (2026-10-25)。 ponder miss 直後の `go ponder` は
	//   相手の時計が前回と同じ値なので、 `go` が来るたびに観測すると `spent = inc` を
	//   見て**「遅い」と誤判定**する。 k == last_k+1 のときだけ観測する。
	int       last_k    = -1;      // 時計に反映済みの相手の手数
	int       streak    = 0;       // 連続して「即指し」と観測した回数
	// 🚨2 パス目は毎回 +50% の時間を食うので回数を上限で抑える (2026-10-25)
	int       attempts  = 0;       // この対局で 2 パス目を回した回数

	// 🚨(E) 2026-09-26: 逸脱は 1 局に 1 回だけ。
	//   利益は「相手が定跡を抜ける」時点で一度確定するので、 繰り返してもコストが積むだけ。
	//   実測: 1 回の逸脱の真の損は平均 25cp で、 B の値にほとんど依存しない。
	bool      deviated        = false;  // この対局で既に定跡から逸脱したか

	void reset() { *this = GateState(); }
};

// 🎯🎯**v2 の受け渡し箱** (2026-10-27 / 設計 DESIGN-20261027-THEORYEXIT-V2.md §3.1)。
//   `probe` の中で「定跡手 m0 が引けた」ことと「出口集合 E」を決め、 探索側がそれを拾う。
//   🚨v1 は定跡手 m0 を**一度も候補にしていなかった** (比較対象が素の探索手 m1 だった) ため、
//     ゲートが開いている間ずっと ybb を捨てる = 事実上の「定跡なしの腕」になっていた。
//     v2 は m0 を既定にし、 出口手 m2 を `v0 - v2 <= B` のときだけ 1 局 1 回選ぶ。
struct PendingOffer {
	bool              live = false;   // 探索側がまだ拾っていない提案があるか
	Move              m0   = Move::none();   // 定跡手 (既定として指す手)
	Value             v0   = VALUE_ZERO;     // その定跡手の**定跡値** (探索せずに指すときの score)
	std::vector<Move> exit;           // 出口集合 E (子が avoid の外 / prefer の中)

	void clear() { live = false; m0 = Move::none(); v0 = VALUE_ZERO; exit.clear(); }
};

// 本体。 USI オプションと N 本の BookSet を持つ。
class BookAwareManager {
public:
	// USI オプションを追加する。 定跡が増えても増えない。
	void add_options(OptionsMap& o);

	// 対局開始時 (isready → clear) にゲートの観測をリセットする。
	void reset_gate() { gate.reset(); }

	// go / go ponder ごとに、 annotate の直前に呼ぶ。 相手の消費時間からゲートを更新する。
	// us = root の手番 (我々)。
	// 🚨`game_ply` が要るので署名を変えた (2026-10-25)。 呼び出し元も直すこと。
	void on_go(const Search::LimitsType& limits, Color us, int game_ply);

	// 🎯2 パス選択 (設計 §19)。
	//   🚨旧実装 (探索値に bias を足す) は **α-β を汚染して「損は高々 B」が壊れた**
	//     (実測: B=0 対 B=60 で 48 手中 42 手が不変条件違反、 最大 349cp の手を指した)。
	//     正しい規則は「普通に探索 → 定跡外だけで再探索 → 差が B 以内なら逸脱」。
	//     両方とも汚染されていない値なので損が構造上 B 以内になる。
	//
	// 🎯v2: `probe` が定跡手 m0 を決めた直後に呼ぶ。 前提 (設計 §3.1 の 1 行目) を全部ここで見て、
	//   通れば pending に (m0, E) を積んで **true** を返す (probe は定跡手を返さず探索へ回す)。
	//   通らなければ false (従来どおり nodes 0 で m0 を指す)。
	//   🚨`m0 ∈ E` (定跡手自身が理論の外へ出る) と `E が空` は false。 試行にも数えない。
	//   🚨root が理論の中かは **book2 が当たった事実**で判定する。 KeySet の membership は使わない
	//     (FlippedBook の反転ヒットで食い違うため / 設計 §7.1)。
	bool offer_book_move(Position& rootPos, Move m0, Value v0);

	// 🎯v2: 探索側が pending を 1 回だけ受け取る。 受け取ったら箱は空になる。
	bool take_pending(Move& m0, std::vector<Move>& exit, Value& v0);

	// 🎯v2: 受け取った提案を**探索せずに捨てた**とき (外部 stop) に呼ぶ。
	//   試行に数えない (設計 §3.5-②)。 本番は `USI_Ponder=true` なので、 数えると
	//   ponder miss が続くだけで AttemptsMax に達して機構が止まる。
	void uncount_attempt() { if (gate.attempts > 0) --gate.attempts; }

	// 2 パス目の結果を受けて、 逸脱するかを決める。 true なら m2 を指す。
	//   v0 = 定跡手 m0 の探索値 / v2 = 出口手 m2 の探索値。 どちらも PV の exact 値。
	bool accept_deviation(int v0, int v2);

	int  bias() const { return bias_value; }
	bool has_deviated() const { return gate.deviated; }

	// リストファイルを読んで定跡を開く。 isready (Search::clear) から呼ぶ。
	// 同じリスト・同じ bias なら再読み込みしない。
	void read_books();

	// B != 0 かつ有効な定跡が 1 本以上あるか。
	bool enabled() const { return bias_value != 0 && !books.empty(); }



private:
	std::vector<std::unique_ptr<BookSet>> books;

	// 読み込み済みの設定 (再読み込みの要否判定に使う)。
	std::string loaded_list;
	int         loaded_maxply = 0;

	// B を内部 Value 単位にしたもの。 0 なら無効。
	int bias_value = 0;

	// 相手が定跡を辿っている証拠のゲート (§15)。
	GateState gate;

	// 🎯v2: probe → 探索 の受け渡し箱 (1 回の go に高々 1 つ)。
	PendingOffer pending;

	// 実験用: ゲートを常に開いたものとして扱う。 既定 false。
	// 🚨棋力のつまみではない。 本番 toml には書かない。 同ノードの測定ハーネス用。
	bool assume_booking = false;
	// 🎯この手数より前は開けない (★book_match_needs_nobook_control:
	//   ply<=20 の即指しは定跡なしでも起きる / 信号は ply 31 以降)。
	int  min_ply        = 0;
	// 🚨試行回数の上限。 **v2 で 4 -> 2 に下げた** (fable レビュー §2 / 2026-10-27)。
	//   後手 10 分+2 秒の 1 手の予算は optimum ≈ 14 s。 1 試行 = パス A (通常) + パス B (1/2)
	//   + 「本来 nodes 0 だった定跡手が探索になる」分 = **1.5 手分**。
	//   Max=4 だと最悪 6 手分 = -3.7pt で設計の利益を食い潰す。
	//   Max=2 なら 期待 -0.7〜-0.8pt / 最悪 7% で -1.0pt / 分岐率 81% (受理率 p=.56)。
	static constexpr int AttemptsMax = 2;

	OptionsMap* options = nullptr;
};

// グローバルに 1 つ。 探索と ThreadPool から参照する。
extern BookAwareManager manager;

} // namespace BookAware

} // namespace YaneuraOu

#endif // USE_BOOK_AWARE
#endif // BOOKAWARE_H_INCLUDED
