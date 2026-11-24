// ============================================================================
//  BookAware: 定跡反映型探索の実装。 2026-09-25
//  設計 runs/prereg/DESIGN-20260925-BOOKAWARE.md §3 / §5 / §6
// ============================================================================

#include "bookaware.h"

#if defined(USE_BOOK_AWARE)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <sstream>

#include "../search.h"
#include "../thread.h"
#include "../movegen.h"

namespace YaneuraOu {
namespace BookAware {

BookAwareManager manager;

namespace {

// ---- ybb (YANE-BINBOOK-V1) の membership 専用リーダ -------------------------
// 🚨book.cpp の reader は MemoryBook::find() 経由で、 IgnoreBookPly=false のとき
//   ply 欄の不一致で全ミスする (book.cpp:790)。 ここでは ply を**見ない**。
//   moves 領域も読まない (seek が 1 回減る)。

constexpr std::array<char, 16> YbbMagic     = {'Y','A','N','E','-','B','I','N','B','O','O','K','-','V','1','\0'};
constexpr u64                  YbbHeaderSize      = 32;
constexpr u64                  YbbIndexRecordSize = 44;

bool read_u64_le(std::istream& is, u64& value) {
	std::array<unsigned char, 8> b{};
	is.read(reinterpret_cast<char*>(b.data()), b.size());
	if (!is)
		return false;
	value = 0;
	for (int i = 7; i >= 0; --i) { value <<= 8; value |= b[size_t(i)]; }
	return true;
}

// index レコードの packed_sfen (先頭 32 バイト) だけを読む。
bool read_packed_at(std::fstream& fs, u64 record_index, PackedSfen& out) {
	fs.clear();
	fs.seekg(std::streamoff(YbbHeaderSize + record_index * YbbIndexRecordSize), std::ios::beg);
	if (!fs)
		return false;
	fs.read(reinterpret_cast<char*>(out.data), 32);
	return bool(fs);
}

// 🎯**KeySet**: `.keys` (packed_sfen 32B の FNV-1a 64 を昇順に並べた u64 配列) を
//   **全部メモリに読んで二分探索**する。 2026-10-25。
//   🚨ybb は HDD にあり on-the-fly は cold 最悪 206 ms。 1 判断 ≈ 75 lookup なので
//     最悪 15 秒 = 持ち時間 3 分の対局では使えない (実測)。 96 MB をメモリに置く。
//   🚨鍵は **packed_sfen のハッシュ** (手数を含まない)。
//     `BookExit` の fnv1a64(sfen 文字列) とは**別物**なので混ぜない。
//   器: `tools/build_book_keys.py` / 関門 G1 で python の ybb membership と全一致を確認済み。
class KeySet : public BookSet {
public:
	KeySet(const BookAttr& a, std::vector<u64>&& ks) : BookSet(a), keys(std::move(ks)) {}

	bool contains(const PackedSfen& ps) const override {
		return std::binary_search(keys.begin(), keys.end(), fnv1a64_bytes(ps.data, 32));
	}
	u64 size() const override { return (u64) keys.size(); }
	const char* mode() const override { return "keys"; }

	static u64 fnv1a64_bytes(const void* p, std::size_t n) {
		const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
		u64 h = 0xCBF29CE484222325ULL;
		for (std::size_t i = 0; i < n; ++i) {
			h ^= (u64) b[i];
			h *= 0x100000001B3ULL;
		}
		return h;
	}

private:
	std::vector<u64> keys;   // 昇順・重複なし
};

class YbbSet : public BookSet {
public:
	YbbSet(const BookAttr& a, std::fstream&& fs, u64 n)
		: BookSet(a), index_fs(std::move(fs)), record_count(n) {}

	bool contains(const PackedSfen& ps) const override {
		// index は packed_sfen の昇順に並んでいる前提 (ybb の仕様)。
		// 🚨fstream は const でないので mutable。 main thread からしか呼ばない (§5.2)。
		u64 left = 0, right = record_count;
		PackedSfen e{};
		while (left < right) {
			u64 mid = left + (right - left) / 2;
			if (!read_packed_at(index_fs, mid, e))
				return false;
			int c = std::memcmp(ps.data, e.data, 32);
			if (c < 0)       right = mid;
			else if (c > 0)  left  = mid + 1;
			else             return true;
		}
		return false;
	}

	u64         size() const override { return record_count; }
	const char* mode() const override { return "ybb"; }

private:
	mutable std::fstream index_fs;
	u64                  record_count = 0;
};

// ---- DB2016 テキストを ply で絞ってメモリに載せる ---------------------------

class MemSet : public BookSet {
public:
	MemSet(const BookAttr& a, std::unordered_set<PackedSfen, PackedSfenHash>&& s)
		: BookSet(a), set(std::move(s)) {}

	bool contains(const PackedSfen& ps) const override { return set.find(ps) != set.end(); }
	u64         size() const override { return u64(set.size()); }
	const char* mode() const override { return "mem"; }

private:
	std::unordered_set<PackedSfen, PackedSfenHash> set;
};

// DB2016 でメモリに載せる上限 (§6.3 の安全弁)。 超えたら .ybb への変換を促して無効化する。
constexpr size_t MemSetLimit = 2'000'000;

void info_str(const std::string& s) { sync_cout << "info string " << s << sync_endl; }

// 🚨このエンジンは -fno-exceptions でビルドされるので std::stoi は使えない
//   (try/catch が書けず、 変換失敗時に abort する)。 自前で符号なし整数をパースする。
//   成功時のみ out に書き、 true を返す。
bool parse_int(const std::string& s, int& out) {
	size_t i = s.find_first_not_of(" \t\r\n");
	if (i == std::string::npos)
		return false;
	bool neg = false;
	if (s[i] == '+' || s[i] == '-') { neg = (s[i] == '-'); ++i; }
	if (i >= s.size() || s[i] < '0' || s[i] > '9')
		return false;
	long long v = 0;
	for (; i < s.size() && s[i] >= '0' && s[i] <= '9'; ++i) {
		v = v * 10 + (s[i] - '0');
		if (v > 1'000'000'000)      // 桁あふれを防ぐ。 ply も maxply もこの範囲で十分。
			return false;
	}
	// 数字のあとに空白以外が続くなら不正とみなす。
	if (s.find_first_not_of(" \t\r\n", i) != std::string::npos)
		return false;
	out = int(neg ? -v : v);
	return true;
}

// ファイルの形式を見て BookSet を作る (§6.3: 形式が読み方を一意に決める)。
std::unique_ptr<BookSet> open_book(const BookAttr& attr) {
	std::fstream fs(attr.path, std::ios::in | std::ios::binary);
	if (!fs) {
		info_str("bookaware error: cannot open " + attr.path);
		return nullptr;
	}

	// ---- 🎯`.keys` か? (先頭 16 バイトの magic / 2026-10-25)
	{
		std::array<char, 16> km{};
		fs.read(km.data(), km.size());
		static const std::array<char, 16> KeysMagic = {
			'Y','A','N','E','-','B','O','O','K','K','E','Y','S','1','\0','\0' };
		if (fs && km == KeysMagic) {
			u64 n = 0, rsv = 0;
			if (!read_u64_le(fs, n) || !read_u64_le(fs, rsv)) {
				info_str("bookaware error: broken keys header " + attr.path);
				return nullptr;
			}
			std::vector<u64> ks(n);
			if (n) {
				fs.read(reinterpret_cast<char*>(ks.data()), std::streamsize(n * 8));
				if (!fs) {
					info_str("bookaware error: short keys body " + attr.path);
					return nullptr;
				}
			}
			// 🚨昇順でないと binary_search が黙って外す。 ここで確かめる。
			if (!std::is_sorted(ks.begin(), ks.end())) {
				info_str("bookaware error: keys not sorted " + attr.path);
				return nullptr;
			}
			info_str("bookaware: keys " + std::to_string(n) + " from " + attr.path);
			return std::make_unique<KeySet>(attr, std::move(ks));
		}
		fs.clear();
		fs.seekg(0, std::ios::beg);
	}

	// ---- ybb か? (先頭 16 バイトの magic)
	std::array<char, 16> magic{};
	fs.read(magic.data(), magic.size());
	if (fs && magic == YbbMagic) {
		u64 record_count = 0, flags = 0;
		if (!read_u64_le(fs, record_count) || !read_u64_le(fs, flags)) {
			info_str("bookaware error: broken ybb header " + attr.path);
			return nullptr;
		}
		// membership しか使わないので flags は解釈しない (moves 領域を読まないため)。
		return std::make_unique<YbbSet>(attr, std::move(fs), record_count);
	}

	// ---- DB2016 テキストか?
	fs.clear();
	fs.seekg(0, std::ios::beg);
	std::string line;
	if (!std::getline(fs, line) || line.rfind("#YANEURAOU-DB2016", 0) != 0) {
		info_str("bookaware error: unknown format (need .ybb or #YANEURAOU-DB2016) " + attr.path);
		return nullptr;
	}

	// sfen 行だけを拾い、 末尾の ply が maxply 以下のものを packed にして載せる。
	std::unordered_set<PackedSfen, PackedSfenHash> set;
	Position   pos;
	StateInfo  si;
	PackedSfen ps{};
	size_t     scanned = 0, over = 0;
	while (std::getline(fs, line)) {
		if (line.rfind("sfen ", 0) != 0)
			continue;
		++scanned;

		// "sfen <board> <turn> <hand> <ply>" の末尾が ply。
		size_t sp = line.find_last_of(' ');
		if (sp == std::string::npos)
			continue;
		int ply = 0;
		if (!parse_int(line.substr(sp + 1), ply))
			continue;
		if (ply > attr.maxply)
			continue;

		// "sfen " を取り除いて Position に食わせる。
		pos.set(line.substr(5), &si);
		pos.sfen_pack(ps);
		set.insert(ps);

		if (set.size() > MemSetLimit) { over = 1; break; }
	}
	if (over) {
		info_str("bookaware error: too large, convert to .ybb " + attr.path);
		return nullptr;
	}
	(void)scanned;
	return std::make_unique<MemSet>(attr, std::move(set));
}

} // namespace

// ---------------------------------------------------------------------------

void BookAwareManager::add_options(OptionsMap& o) {
	options = &o;

	// 唯一のつまみ。 cp 単位。 0 = 無効 (現行と bit-exact)。
	o.add("BookAwareBias", Option(0, 0, 2000));

	// 定跡リストファイルのパス。 空なら無効。
	o.add("BookAwareList", Option(""));

	// 各定跡の maxply の既定値。 実験用であって棋力のつまみではない。
	o.add("BookAwareMaxPly", Option(40, 0, 1024));

	// 🚨実験用。 ゲート (§15) を常に開いたものとして扱う。
	//   同ノード対局では時間情報が無く booking は常に false になるので、
	//   「届いているか」の測定 (§13.4) を再現するためだけに使う。
	//   **棋力のつまみではない。 本番 toml には書かない。**
	o.add("BookAwareAssumeBooking", Option(false));
	// 🎯この手数より前はゲートを開けない (2026-10-25)。
	//   🚨★book_match_needs_nobook_control: **ply<=20 の即指しは定跡なしでも起きる**。
	//     信号は ply 31 以降の連続した即指し。 既定 0 = 従来どおり (bit-exact)。
	o.add("BookAwareMinPly", Option(0, 0, 512));
}

// 「相手が定跡を辿っている証拠」のゲートを更新する (設計 §15.2)。
//   go / go ponder ごとに main thread から 1 回だけ呼ばれる。
void BookAwareManager::on_go(const Search::LimitsType& limits, Color us, int game_ply) {
	if (options != nullptr) {
		assume_booking = bool((*options)["BookAwareAssumeBooking"]);
		min_ply        = int((*options)["BookAwareMinPly"]);
	}
	const Color them = ~us;

	// 時計が無い探索 (go nodes / depth / infinite / movetime / mate) では観測できない。
	//   -> `BookAwareAssumeBooking` が true ならゲートを開く (判定で使う)。
	if (!limits.use_time_management()) {
		gate.booking = assume_booking;
		return;
	}

	// ---- τ: この秒数以下なら「即指し」= 相手はまだ定跡の中
	//   🚨🚨**既定の 800 ms は電竜戦で壊れる** (2026-10-25 に判明)。
	//     CSA の相手消費時間は**整数秒**でしか返ってこない (session.rs:1349) ので、
	//     T=1 秒は 1000 ms と観測される。 τ=800 だと **T1 が全部「遅い」**になる。
	//   ✅監査の定義 (T<=1) に揃えて τ = max(**1000**, ref/5) の **inclusive** にする。
	TimePoint ref = limits.inc[them];
	if (ref <= 0) ref = limits.byoyomi[them];
	if (ref <= 0) ref = std::max<TimePoint>(1000, limits.time[them] / 50);
	constexpr TimePoint TauFloorMs = 1000;
	const TimePoint tau = std::max<TimePoint>(TauFloorMs, ref / 5);

	// ---- 🎯**相手の手数で索引する** (2026-10-25)
	//   🚨🚨既存は「go が来るたび」観測していたので、 ponder miss のあとの `go ponder` で
	//     **相手の時計が前回と同じ値**のまま渡ってきて `spent = inc` を観測し、
	//     フィッシャーでは「遅い」、 秒読みでは「速い」と**誤判定**していた。
	//   root は常に我々の手番なので、 相手が指した手数は game_ply から決まる。
	//   `go ponder` は我々の手の直後なので、 相手の時計にはまだ 1 手反映されていない。
	const int n_opp = (us == BLACK) ? (game_ply - 1) / 2 : game_ply / 2;
	const int k     = n_opp - (limits.ponderMode ? 1 : 0);
	const TimePoint now = limits.time[them];

	if (k == gate.last_k) {
		// 新しい観測は無い。 状態を維持する (ここで誤判定が入っていた)。
	} else if (k == gate.last_k + 1 && gate.have_last) {
		const TimePoint spent = gate.last_them + limits.inc[them] - now;
		// 秒読み中 (持ち時間 0 + 秒読み) は消費が読めないので「遅い」側に倒す
		const bool byoyomi_only = (limits.time[them] == 0 && limits.byoyomi[them] > 0);
		if (spent < 0 || byoyomi_only) {
			gate.streak = 0;
		} else {
			++gate.obs;
			const bool fast = (spent <= tau);   // 🚨inclusive
			gate.streak = fast ? gate.streak + 1 : 0;
		}
		gate.last_k    = k;
		gate.last_them = now;
		gate.have_last = true;
	} else {
		// 手戻し / 新対局 / 初回。 観測を捨ててやり直す。
		gate.streak    = 0;
		gate.last_k    = k;
		gate.last_them = now;
		gate.have_last = true;
	}

	// ---- 🎯判定: **ply が窓に入っていて、 即指しが 2 回続いた**とき開く
	//   🚨★book_match_needs_nobook_control: ply<=20 の即指しは**定跡なしでも起きる**ので、
	//     「開幕 1 手目で book_user を確定」は捨てた。 信号は **ply 31 以降の連続した即指し**。
	//   🚨誤判定は**偽陽性側に倒す** (代償は 1 局 1 回 <= B の譲りで上から抑えられる)。
	gate.booking = (game_ply >= min_ply) && (gate.streak >= 2);
}

void BookAwareManager::read_books() {
	if (options == nullptr)
		return;

	const int         bias_cp = int((*options)["BookAwareBias"]);
	const std::string list    = std::string((*options)["BookAwareList"]);
	const int         maxply  = int((*options)["BookAwareMaxPly"]);

	// B=0 のときは**ファイルも開かない** (§5.3: bit-exact を保証する)。
	if (bias_cp == 0 || list.empty()) {
		books.clear();
		loaded_list.clear();
		bias_value = 0;
		return;
	}

	bias_value = int(USIEngine::cp_to_value(bias_cp));

	// 同じ設定なら開き直さない (isready は対局ごとに来る)。
	if (list == loaded_list && maxply == loaded_maxply && !books.empty())
		return;

	books.clear();
	loaded_list   = list;
	loaded_maxply = maxply;

	std::ifstream ifs(list);
	if (!ifs) {
		info_str("bookaware error: cannot open list " + list);
		return;
	}

	// リストファイルからの相対パスを解決するための base ディレクトリ。
	std::string base;
	{
		size_t p = list.find_last_of("/\\");
		if (p != std::string::npos)
			base = list.substr(0, p + 1);
	}

	std::string line;
	while (std::getline(ifs, line)) {
		// コメント・空行を飛ばす。
		size_t h = line.find('#');
		if (h != std::string::npos)
			line = line.substr(0, h);
		if (line.find_first_not_of(" \t\r\n") == std::string::npos)
			continue;

		std::istringstream is(line);
		std::string        role_s, path_s;
		if (!(is >> role_s >> path_s)) {
			info_str("bookaware error: bad line: " + line);
			continue;
		}

		BookAttr attr;
		if (role_s == "avoid")       attr.role = Role::Avoid;
		else if (role_s == "prefer") attr.role = Role::Prefer;
		else { info_str("bookaware error: role must be avoid/prefer: " + line); continue; }

		// 相対パスなら base を前置する。
		attr.path = (path_s.find(':') == std::string::npos && path_s[0] != '/' && path_s[0] != '\\')
		              ? base + path_s : path_s;
		attr.maxply = maxply;

		// key=value (順不同)。 未知の key は警告して無視する
		// (将来 weight= 等を足しても古い exe が落ちないように)。
		std::string kv;
		bool        bad = false;
		while (is >> kv) {
			size_t eq = kv.find('=');
			if (eq == std::string::npos) { info_str("bookaware warn: ignored token '" + kv + "'"); continue; }
			std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
			if (k == "side") {
				if (v == "black")      attr.side = BLACK;
				else if (v == "white") attr.side = WHITE;
				else if (v == "both")  attr.side = COLOR_NB;
				else { info_str("bookaware error: side must be black/white/both: " + line); bad = true; break; }
			}
			else if (k == "maxply") {
				if (!parse_int(v, attr.maxply)) { info_str("bookaware error: bad maxply: " + line); bad = true; break; }
			}
			else
				info_str("bookaware warn: unknown key '" + k + "' ignored");
		}
		if (bad)
			continue;

		auto bs = open_book(attr);
		if (!bs)
			continue;   // 読めない行は**その行だけ**無効にして続行する (対局を落とさない)。

		std::ostringstream os;
		os << "bookaware loaded role=" << role_s
		   << " path=" << attr.path
		   << " entries=" << bs->size()
		   << " mode=" << bs->mode()
		   << " side=" << (attr.side == COLOR_NB ? "both" : (attr.side == BLACK ? "black" : "white"))
		   << " maxply=" << attr.maxply;
		info_str(os.str());

		books.push_back(std::move(bs));
	}

	if (books.empty())
		info_str("bookaware: no book loaded, disabled");
}

// 🏁**v1 の `block_book2` はここにあったが v2 で削除した** (2026-10-27)。
//   「ゲートが開いている間ずっと ybb を引かない」= 定跡手 m0 を指す経路が無く、
//   事実上「定跡なしの腕」だった。 v2 は `offer_book_move` (下) が m0 を既定にする。

// 🎯🎯**v2 の入口** (2026-10-27 / 設計 DESIGN-20261027-THEORYEXIT-V2.md §3.1)。
//   `probe` が book2 (ybb) で定跡手 m0 を決めた直後に呼ばれる。
//   前提を全部ここで見て、 通れば pending に (m0, E) を積んで true を返す
//   (probe は定跡手を返さず探索へ回る)。
//
//   🚨🚨**v1 からの変更点**: v1 の `block_book2` は「ゲートが開いていれば**毎手** ybb を
//     引かない」だけで、 定跡手 m0 を指す経路が存在しなかった。 その結果
//     ・1 局あたり 2.4 手 (逸脱 1 + 「探索したが理論の中に留まった」1.39) で
//       **1 手あたり中央 30cp** を払っていた (真の損 491 局面の実測)
//     ・97.6% の分岐のうち 29% は「理論の中の別手」= 利益ゼロで損だけ
//     v2 は **m0 を既定**にし、 出口手 m2 を `v0 - v2 <= B` のときだけ 1 局 1 回選ぶ。
//     損は構造上「1 局 1 回 ≤ B」に収まる。
//
//   🚨root が理論の中かは **book2 が当たった事実**で判定する (KeySet の membership は使わない)。
//     `FlippedBook=true` では engine が先後反転局面でもヒットするので、 python 側の
//     素朴な照合と食い違う (★2026-10-25)。
bool BookAwareManager::offer_book_move(Position& rootPos, Move m0, Value v0) {
	pending.clear();
	if (!enabled() || gate.deviated || !m0.is_ok())
		return false;
	if (!(gate.booking || assume_booking))
		return false;

	const Color us  = rootPos.side_to_move();
	const int   ply = rootPos.game_ply();

	// 🚨**MinPly** (fable レビュー D / 2026-10-27)。 v1 の `block_book2` は見ていたが
	//   v2 で落ちていた。 実 TC は `gate.booking` が ply>=min_ply を含むので無事だが、
	//   `AssumeBooking=true` (測定ハーネス) では素通りしてしまう。
	if (ply < min_ply)
		return false;

	// 🚨**試行回数の上限** (2026-10-25)。 `deviated` は成功 1 回を止めるが、
	//   ゲートが開いている間は毎手 2 パス目を回せてしまう。 2 パス目は毎回
	//   **+50% の時間** (movetime = elapsed/2) を食うので、 窓 31-60 の我々の手 <=15 を
	//   全部使うと最悪 +7.5 手分。 4 回に抑えると +2 手分。
	if (gate.attempts >= AttemptsMax)
		return false;

	std::vector<BookSet*> avoid;
	std::vector<BookSet*> prefer_sets;
	for (auto& b : books)
		if (b->active_for(us) && ply <= b->attr.maxply) {
			if (b->attr.role == Role::Avoid)  avoid.push_back(b.get());
			else                              prefer_sets.push_back(b.get());
		}
	if (avoid.empty())
		return false;

	PackedSfen ps{};
	auto in_book = [&](void) {
		rootPos.sfen_pack(ps);
		for (auto* b : avoid)
			if (b->contains(ps)) return true;
		return false;
	};

	// 🚨v2: `in_book()` による root の判定は**しない**。 ここへ来た時点で book2 が当たっている
	//   = root は理論の中、 というのが呼び出し側の契約 (設計 §7.1)。

	// 🎯**prefer (= 我々の 478) にある子は「降りた」扱い** (2026-10-25)。
	//   478 と ybb の重なりは **2.8%** (実測) なので出番はほぼ無いが、
	//   transposition で 478 に戻る手を「理論の中」と誤判定しないための保険。
	auto in_prefer = [&](void) {
		if (prefer_sets.empty())
			return false;
		rootPos.sfen_pack(ps);
		for (auto* b : prefer_sets)
			if (b->contains(ps)) return true;
		return false;
	};

	StateInfo si;

	// 🎯定跡手 m0 自身が理論の外へ出るなら、 機構の出番は無い。 m0 を指して終わり。
	//   🚨試行 (attempts) には**数えない** — 時間を 1 秒も使っていないため。
	rootPos.do_move(m0, si);
	const bool m0_in   = in_book();
	const bool m0_pref = in_prefer();
	rootPos.undo_move(m0);
	if (!m0_in || m0_pref)
		return false;

	// 出口集合 E = 定跡外へ出る合法手 (prefer に入る手も「降りた」扱いで入れる)。
	std::vector<Move> exit;
	for (const auto& ml : MoveList<LEGAL_ALL>(rootPos)) {
		const Move m = Move(ml);
		rootPos.do_move(m, si);
		const bool child_in = in_book();
		const bool child_pr = in_prefer();
		rootPos.undo_move(m);
		if (!child_in || child_pr)
			exit.push_back(m);
	}
	if (exit.empty())        // 降りる先が無い -> m0 を指して終わり (試行に数えない)
		return false;

	++gate.attempts;
	pending.live = true;
	pending.m0   = m0;
	pending.v0   = v0;      // 🚨探索せずに m0 を指すときの score に使う (入れないと -∞ が出る)
	pending.exit = std::move(exit);

	std::ostringstream os;
	os << "bookaware offer attempt=" << gate.attempts << " ply=" << ply
	   << " exit=" << pending.exit.size() << " m0=" << USIEngine::move(m0);
	info_str(os.str());
	return true;
}

// 🎯v2: 探索側が pending を 1 回だけ受け取る。 受け取ったら箱は空になる。
bool BookAwareManager::take_pending(Move& m0, std::vector<Move>& exit, Value& v0) {
	if (!pending.live)
		return false;
	m0   = pending.m0;
	v0   = pending.v0;
	exit = pending.exit;
	pending.clear();
	return true;
}

// 🎯2 パス選択の 2 段目。
//   🚨**v2 で比較対象が変わった** (2026-10-27): v0 = **定跡手 m0** の探索値 /
//     v2 = 出口手 m2 の探索値。 v1 (素の探索手) ではない。
//     これで初めて「譲るのは定跡手に対して B 以内」という保証が言葉どおりになる。
//   差が B 以内なら逸脱する。 これが「有力な手が定跡手しかなければ指す」の実装。
bool BookAwareManager::accept_deviation(int v0, int v2) {
	const int v1 = v0;   // 以下の式は v1 の名前で書かれている (意味は定跡手の値)
	// 🎯優勢なら定跡を外さない (設計 §20、 ユーザー要求)。
	//   B_eff = B * clamp(1 - cp1/V0, 0, 1)、 V0 = 150cp。
	//   cp1 >= V0 なら B_eff = 0 = 絶対に外さない。
	//   🚨これは**測定で得られた改善ではなく方針の表明**。 実測では v1 と逸脱の真の損に
	//     相関が無く (Pearson +0.15 / Spearman -0.06、 n=33)、 ロジスティック換算では
	//     コストと利益が同じように増減するので費用対効果の比は v1 に依存しない。
	//     採用理由は「互角のときに差別化し、 勝っているときは賭けない」という方針と、
	//     実戦で序盤に +100 を超えるのが 4% しかない = 害がほぼ無いこと。
	//   ⚠劣勢側で B を広げることはしない (ロジスティックは対称で、 本当に悪い定跡筋なら
	//     v1-v2 が自然に小さくなって逸脱が起きるため)。
	constexpr int V0_CP = 150;
	const int cp1 = int(USIEngine::to_cp(Value(v1)));
	const int cp2 = int(USIEngine::to_cp(Value(v2)));
	const int b_cp = int(USIEngine::to_cp(Value(bias_value)));
	const int b_eff = (cp1 >= V0_CP) ? 0
	                : (cp1 <= 0)     ? b_cp
	                                 : int(std::lround(double(b_cp) * (1.0 - double(cp1) / V0_CP)));

	std::ostringstream os;
	if (cp1 - cp2 <= b_eff) {
		gate.deviated = true;
		os << "bookaware deviate attempt=" << gate.attempts
		   << " cp0=" << cp1 << " cp2=" << cp2 << " gap=" << (cp1 - cp2)
		   << " <= B_eff=" << b_eff << " (B=" << b_cp << ", V0=" << V0_CP << ")"
		   << " 以後この対局では避けない";
		info_str(os.str());
		return true;
	}
	os << "bookaware keep-book attempt=" << gate.attempts
	   << " cp0=" << cp1 << " cp2=" << cp2 << " gap=" << (cp1 - cp2)
	   << " > B_eff=" << b_eff << " (B=" << b_cp << ", V0=" << V0_CP << ")"
	   << (cp1 >= V0_CP ? " ★優勢なので外さない" : "");
	info_str(os.str());
	return false;
}

} // namespace BookAware
} // namespace YaneuraOu

#endif // USE_BOOK_AWARE
