#include "../../types.h"

#if defined (YANEURAOU_ENGINE)

// -----------------------
//  繧・・縺・ｉ邇・讓呎ｺ匁爾邏｢驛ｨ
// -----------------------

// 險域ｸｬ雉・侭鄂ｮ縺榊ｴ : https://github.com/yaneurao/YaneuraOu/wiki/%E6%8E%A2%E7%B4%A2%E9%83%A8%E3%81%AE%E8%A8%88%E6%B8%AC%E8%B3%87%E6%96%99

// -----------------------
//   includes
// -----------------------

#include "../../search.h"

#include <sstream>
#include <algorithm>	// std::binary_search / std::sort (BookExit)
#include <cstdint>	// std::uint64_t (BookExit)
#include <fstream>
#include <iomanip>
#include <cmath>	// std::log(),std::pow(),std::round()
#include <cstring>	// memset()
#include <atomic>	// UNCERT_LMR_MEASURE 縺ｮ險域ｸｬ繧ｫ繧ｦ繝ｳ繧ｿ
#include <thread>	// std::thread (p2shogi: df-pn 隧ｰ縺ｿsolver蟆ら畑繧ｹ繝ｬ繝・ラ)
#include <memory>	// std::unique_ptr
#include <vector>	// std::vector

#include "yaneuraou-search.h"
#include "../../position.h"
#include "../../p2volm.h"	// P2Volm::value / bin (2026-09-28)
#include "../../thread.h"
#include "../../misc.h"
#include "../../tt.h"
#include "../../book/book.h"
#include "../../book/bookaware.h"	// 🎯BookAware (定跡反映型探索、 2026-09-25)
#include "../../movepick.h"
#include "../../usi.h"
#include "../../mate/mate.h"
#include "../../tune.h"
#include "../../spsa_params.h"	// SPSA謗｢邏｢繝√Η繝ｼ繝九Φ繧ｰ繝代Λ繝｡繝ｼ繧ｿ(rust貅匁侠, EVAL_P2NNUE譎・

namespace YaneuraOu {

// ===== 定跡回避 (BookExit)。 2026-10-06。 設計 runs/prereg/DESIGN-20261006-BOOKEXIT.md =====
//   🎯floodgate の我々の 392 局を相手の定跡の深さで層別すると、
//     **相手定跡 >60 手の局で勝率 0.073 / rate 期待比 -22.8pt** (母集団 n=1,600 で -29.0pt)。
//     角換わりで相手定跡 <=60 手なら残差 ≈ 0 = 負け筋は「角換わり」ではなく「深い定跡」。
//   🚨様式項 (評価に下駄) にしない: 振り飛車 G=320 は +82cp で対局 -36pt だった。
//     **root で指す手を選び直すだけ**なら探索の質は変わらず、 代償は 1 局 1 回の Margin だけ。
//   🚨鍵は Zobrist ではなく **fnv1a64(盤面 + ' ' + 手番 + ' ' + 手駒)**。
//     python 側 (tools/build_bookexit_set.py) と同じ実装にそろえてある。 手数は入れない。
namespace BookExit {

static std::vector<std::uint64_t> g_set;   // 昇順。 二分探索で引く
static int  g_margin   = 0;                // 0 = 無効 (既定)
static int  g_max_ply  = 60;
static bool g_loaded   = false;

static inline std::uint64_t fnv1a64(const std::string& s) {
	std::uint64_t h = 0xCBF29CE484222325ULL;
	for (unsigned char c : s) {
		h ^= (std::uint64_t) c;
		h *= 0x100000001B3ULL;
	}
	return h;
}

// sfen の先頭 3 要素 (盤面 / 手番 / 手駒) だけで鍵を作る。 手数カウンタは入れない。
static inline std::uint64_t key_of(const Position& pos) {
	const std::string sf = pos.sfen();
	std::size_t a = sf.find(' ');
	if (a == std::string::npos) return 0;
	std::size_t b = sf.find(' ', a + 1);
	if (b == std::string::npos) return 0;
	std::size_t c = sf.find(' ', b + 1);
	const std::string head = (c == std::string::npos) ? sf : sf.substr(0, c);
	return fnv1a64(head);
}

static inline bool in_set(const Position& pos) {
	if (g_set.empty()) return false;
	const std::uint64_t k = key_of(pos);
	return std::binary_search(g_set.begin(), g_set.end(), k);
}

// 回避集合をファイルから読む (u64 の昇順配列)。 空パスなら無効化。
static void load(const std::string& path) {
	g_set.clear();
	g_loaded = false;
	if (path.empty()) return;
	std::ifstream f(path, std::ios::binary);
	if (!f) { sync_cout << "info string BookExit: ファイルが開けない " << path << sync_endl; return; }
	f.seekg(0, std::ios::end);
	const std::streamoff sz = f.tellg();
	f.seekg(0, std::ios::beg);
	if (sz <= 0 || (sz % 8) != 0) { sync_cout << "info string BookExit: ファイルの大きさが 8 の倍数でない" << sync_endl; return; }
	g_set.resize((std::size_t)(sz / 8));
	f.read(reinterpret_cast<char*>(g_set.data()), sz);
	if (!std::is_sorted(g_set.begin(), g_set.end())) std::sort(g_set.begin(), g_set.end());
	g_loaded = true;
	sync_cout << "info string BookExit: " << g_set.size() << " 局面を読んだ" << sync_endl;
}

}  // namespace BookExit



using namespace Search;
using namespace Eval;  // Eval::PieceValue

// -------------------
// 決 Uncertainty-linked LMR (隨ｬ1繝代ャ繝・
// -------------------

// nodeEvalVolatility(=|ss->staticEval + (ss-1)->staticEval|)縺ｮ髦ｲ蠕｡逧・け繝ｩ繝ｳ繝嶺ｸ企剞縲・
// 隱ｿ謨ｴ繝代Λ繝｡繝ｼ繧ｿ縺ｧ縺ｯ縺ｪ縺・る撼decisive縺ｪstaticEval縺ｮ蜥後・逅・ｫ紋ｸ・2*VALUE_TB 譛ｪ貅縺縺後・
// int螳牙・蝓溘°縺､ binning(>=1024縺ｯ蜷御ｸbin)縺ｫ蠖ｱ髻ｿ縺励↑縺・崋螳壼､縺ｨ縺励※ VALUE_TB 譛ｪ貅縺ｮ 30000 繧剃ｽｿ縺・・
constexpr int UNCERT_LMR_MAX_DELTA = 30000;

#ifdef UNCERT_LMR_MEASURE
// 蛻・ｸ・ｨ域ｸｬ(FEATURE_COUNT逶ｸ蠖・縲６NCERT_LMR_MEASURE 螳夂ｾｩ繝薙Ν繝峨〒縺ｮ縺ｿ譛牙柑縲・
// 髱槫ｮ夂ｾｩ繝薙Ν繝峨〒縺ｯ荳蛻・・繧ｳ繝ｼ繝・繧ｪ繝ｼ繝舌・繝倥ャ繝峨ｒ谿九＆縺ｪ縺・謗｢邏｢邨先棡縺ｯ荳榊､・縲・
namespace {
struct UncertLmrMeasure {
    std::atomic<uint64_t> validVolNodes{0};   // validEvalVolatility=true 縺ｨ縺ｪ縺｣縺・node 謨ｰ
    std::atomic<uint64_t> candidateQuiet{0};  // LMR驕ｩ逕ｨ蜿ｯ閭ｽ(髢ｾ蛟､莉･螟悶・蜈ｨ譚｡莉ｶ謌千ｫ・quiet 謨ｰ
    std::atomic<uint64_t> bin[6]{};           // volatility bin: 0-63/64-127/128-255/256-511/512-1023/1024+
    std::atomic<uint64_t> fired{0};           // 螳滄圀縺ｫ r 繧・ply貂帙ｉ縺励◆謨ｰ
    std::atomic<uint64_t> firedFailHigh{0};   // 貂帙ｉ縺励◆謇九′ reduced 謗｢邏｢縺ｧ value>alpha
    std::atomic<uint64_t> firedReSearch{0};   // 貂帙ｉ縺励◆謇九′ full-depth 蜀肴爾邏｢縺ｫ閾ｳ縺｣縺滓焚
    void candidate(int vol) {
        candidateQuiet.fetch_add(1, std::memory_order_relaxed);
        int b = vol < 64 ? 0 : vol < 128 ? 1 : vol < 256 ? 2 : vol < 512 ? 3 : vol < 1024 ? 4 : 5;
        bin[b].fetch_add(1, std::memory_order_relaxed);
    }
};
UncertLmrMeasure g_uncertMeasure;
}  // namespace
#endif

// -------------------
// 決 Threat Continuation History (Phase B: threatMove 譁・ц縺ｧ縺ｮ quiet response 蟄ｦ鄙・
// -------------------

// threatMove(逶ｸ謇九・閼・ｨ∵焔)繧・蝗槭□縺大・鬘槭☆繧九Ｈives_check 縺ｯ蜻ｼ縺ｳ蜃ｺ縺怜・縺ｧ1蝗槭□縺題ｨ育ｮ励＠縺ｦ貂｡縺吶・
// pos 縺ｯ null 螻髱｢(逶ｸ謇区焔逡ｪ)縲・ingZone 縺ｮ蟇ｾ雎｡邇峨・ ~side_to_move() = 閼・ｨ√ｒ蜿励￠繧句・(=謇狗分pass縺励◆蛛ｴ)縺ｮ邇峨・
inline ThreatClass classify_threat(const Position& pos, Move m, bool givesCheck) {
    if (m.is_drop() && givesCheck)  return ThreatClass::DropCheck;
    if (givesCheck)                 return ThreatClass::Check;
    if (pos.capture_stage(m))       return ThreatClass::Capture;
    if (m.type_of() == PROMOTION)   return ThreatClass::Promotion;
    if (dist(m.to_sq(), pos.square<KING>(~pos.side_to_move())) <= 2)
                                    return ThreatClass::KingZone;
    return ThreatClass::Other;
}

#ifdef THREAT_STATS
namespace {
struct ThreatStatsCounters {
    std::atomic<uint64_t> nmNoCutoff{0}, adopt{0};   // 蜿門ｾ苓ｩｦ陦・nullValue<beta) / threatMove 謗｡逕ｨ
    std::atomic<uint64_t> cls[7]{};                  // 謗｡逕ｨ threatMove 縺ｮ ThreatClass 蛻･
    std::atomic<uint64_t> posUpd{0}, negUpd{0};      // Threat History 豁｣/雋 譖ｴ譁ｰ謨ｰ
    std::atomic<uint64_t> updRefs{0}, updCollisions{0};  // 譖ｴ譁ｰindex蜿ら・謨ｰ / 蛻･繝輔Ν繧ｭ繝ｼ陦晉ｪ∵焚
};
ThreatStatsCounters g_threatStats;
// 譖ｴ譁ｰ index 縺ｫ譛蠕後↓蜈･縺｣縺・32bit 繝輔Ν繧ｭ繝ｼ縲ょ挨繝輔Ν繧ｭ繝ｼ縺悟酔 slot 縺ｫ譚･縺溘ｉ陦晉ｪ√→縺励※險域焚(Threads=1險域ｸｬ逕ｨ)縲・
uint32_t g_threatTag[THREAT_HISTORY_SIZE] = {0};
}  // namespace
#endif

// -------------------
// 決 繧・・縺・ｉ邇狗峡閾ｪ霑ｽ蜉
// -------------------

// 決 tune.py縺ｫ繧医▲縺ｦ閾ｪ蜍慕噪縺ｫTUNE縺吶ｋ螟画焚螳｣險縺後％縺謎ｻ･荳九↓霑ｽ蜉縺輔ｌ繧九・
// 統 tune.py縺ｨ縺ｯ縲√ヱ繝ｩ繝｡繝ｼ繧ｿ繝ｼ閾ｪ蜍戊ｪｿ謨ｴ繝輔Ξ繝ｼ繝繝ｯ繝ｼ繧ｯ縺ｮ繧ｹ繧ｯ繝ｪ繝励ヨ縺ｧ縺ゅｋ縲・
//     https://github.com/yaneurao/YaneuraOu-ScriptCollection/tree/main/SPSA
//                            %%TUNE_DECLARATION%%


// 縺薙・讒矩菴薙Γ繝ｳ繝舌・縺ｫ蟇ｾ蠢懊☆繧九お繝ｳ繧ｸ繝ｳ繧ｪ繝励す繝ｧ繝ｳ繧堤函繧・☆
void SearchOptions::add_options(OptionsMap& options) {
    // 蠑輔″蛻・￠縺ｾ縺ｧ縺ｮ譛螟ｧ謇区焚縲・56謇九Ν繝ｼ繝ｫ縺ｮ縺ｨ縺阪↓256繧定ｨｭ螳壹☆繧九→濶ｯ縺・・縺ｪ繧臥┌蛻ｶ髯舌・
    /*
		涛 0縺瑚ｨｭ螳壹＆繧後※縺・◆繧峨∝ｼ輔″蛻・￠縺ｪ縺励□縺九ｉ100000縺御ｻ｣蜈･縺輔ｌ繧九％縺ｨ縺ｫ縺ｪ縺｣縺ｦ縺・ｋ縲・
		    (谿九ｊ謇区焚繧定ｨ育ｮ励☆繧区凾縺ｫ譯√≠縺ｵ繧後☆繧九→濶ｯ縺上↑縺・・縺ｧint_max縺ｫ縺ｯ縺励※縺・↑縺・

			蛻晄焔(76豁ｩ縺ｨ縺・縺・謇狗岼縺ｧ縺ゅｋ縲・
			1謇狗岼繧呈欠縺吝燕縺ｮ螻髱｢縺ｯPosition::game_ply() == 1縺ｧ縺ゅｋ縲・

			縺昴＠縺ｦ256謇区欠縺輔ｌ縺滓凾轤ｹ(257謇狗岼縺ｮ螻髱｢縺ｧ謖・☆讓ｩ蛻ｩ縺後≠繧九％縺ｨ縲・
			繧ｵ繝ｼ繝舌・縺九ｉ257謇狗岼縺ｮ螻髱｢縺ｯ繧・▲縺ｦ縺薙↑縺・ｂ縺ｮ縺ｨ縺吶ｋ)縺ｧ蠑輔″蛻・￠縺縺ｨ縺励◆繧・
			257謇狗岼(繧呈欠縺吝燕縺ｮ螻髱｢)縺ｯ縲“ame_ply() == 257縺ｧ縺ゅｋ縲・
			縺薙ｌ縺後∝ｼ輔″蛻・￠謇ｱ縺・→縺・≧縺薙→縺ｫ縺ｪ繧九・

			pos.game_ply() > max_moves_to_draw
		縲	縺ｧ(縺九▽縲∬ｩｰ縺ｿ縺ｧ縺ｪ縺代ｌ縺ｰ)蠑輔″蛻・￠縺ｨ縺・≧縺薙→縺ｫ縺ｪ繧九・

			縺薙・蠑輔″蛻・￠縺ｮ謇ｱ縺・↓縺､縺・※縺ｯ縲∽ｻ･荳九・險倅ｺ九′隧ｳ縺励＞縲・
			螟壹￥縺ｮ蟆・｣九た繝輔ヨ縺ｧ256謇九Ν繝ｼ繝ｫ縺ｮ螳溯｣・′繝舌げ縺｣縺ｦ縺・ｋ莉ｶ
			https://yaneuraou.yaneu.com/2021/01/13/incorrectly-implemented-the-256-moves-rule/

	*/
    options.add("MaxMovesToDraw", Option(0, 0, 100000, [&](const Option& o) {
                    // 縺薙ｌ0縺ｮ譎ゅ∽ｽ輔ｉ縺玖ｨｭ螳壹＠縺ｦ縺翫°縺ｪ縺・→謗｢邏｢驛ｨ縺ｧ縺薙・謇区焚繧定ｶ・∴縺滓凾縺ｫ
                    // 蠑輔″蛻・￠謇ｱ縺・↓縺励※縺励∪縺・・縺ｧ縲∫┌髯仙､ｧ縺ｿ縺溘＞縺ｪ螳壽焚縺ｮ險ｭ螳壹′蠢・ｦ√・
                    max_moves_to_draw = int(o);
                    if (max_moves_to_draw == 0)
                        max_moves_to_draw = 100000;
                    return std::nullopt;
                }));

    //  PV縺ｮ蜃ｺ蜉帙・謚大宛縺ｮ縺溘ａ縺ｫ蜑榊屓蜃ｺ蜉帶凾髢薙°繧峨・髢馴囈繧呈欠螳壹〒縺阪ｋ縲・
    options.add("PvInterval", Option(300, 0, 100000000, [&](const Option& o) {
                    pv_interval = s64(o);
                    return std::nullopt;
                }));

    // ---- 定跡回避 (BookExit)。 2026-10-06。 設計 runs/prereg/DESIGN-20261006-BOOKEXIT.md
    //   既定 (Margin=0 / ファイル無し) で**完全に無効** = 既存 exe と bit-exact。
    //   🚨Margin は **Value 単位** (cp との換算は net 依存)。 ★48: 範囲外 spin は破棄されるので広く取る。
    options.add("BookExitMargin", Option(0, 0, 2000, [](const Option& o) {
                    BookExit::g_margin = int(o);
                    return std::nullopt;
                }));
    options.add("BookExitMaxPly", Option(60, 0, 512, [](const Option& o) {
                    BookExit::g_max_ply = int(o);
                    return std::nullopt;
                }));
    options.add("BookExitFile", Option("", [](const Option& o) {
                    BookExit::load(std::string(o));
                    return std::nullopt;
                }));

    // 讀懆ｨ弱Δ繝ｼ繝臥畑縺ｮPV繧貞・蜉帙☆繧九Δ繝ｼ繝・
    options.add("ConsiderationMode", Option(false, [&](const Option& o) {
                    consideration_mode = o;
                    return std::nullopt;
                }));

    // fail low/high縺ｮ縺ｨ縺阪↓PV繧貞・蜉帙☆繧九°縺ｩ縺・°縲・
    options.add("OutputFailLHPV", Option(true, [&](const Option& o) {
                    outout_fail_lh_pv = o;
                    return std::nullopt;
                }));

    // 縺吶∋縺ｦ縺ｮ蜷域ｳ墓焔繧堤函謌舌☆繧九・縺・
    options.add("GenerateAllLegalMoves", Option(false, [&](const Option& o) {
                    generate_all_legal_moves = o;
                    return std::nullopt;
                }));

    // 蜈･邇峨Ν繝ｼ繝ｫ
    options.add("EnteringKingRule",
                Option(EKR_STRINGS, EKR_STRINGS[EKR_27_POINT], [&](const Option& o) {
                    enteringKingRule = to_entering_king_rule(o);
                    return std::nullopt;
                }));
}


void SearchManager::pre_start_searching(YaneuraOuWorker& worker) {
    // ､・Stockfish縺ｮThreadPool::start_thinking()縺ｫ縺ゅ▲縺滉ｻ･荳九・蛻晄悄蛹悶ｒ縺薙■繧峨↓遘ｻ蜍輔＆縺帙◆縲・

    stopOnPonderhit /* = stop = abortedSearch */ = false;
    ponder                                       = worker.limits.ponderMode;
    increaseDepth                                = true;
}

// 諤晁・お繝ｳ繧ｸ繝ｳ縺ｮ霑ｽ蜉繧ｪ繝励す繝ｧ繝ｳ繧定ｨｭ螳壹☆繧九・
// 庁 Stockfish縺ｧ縺ｯ縲・ngine::Engine()縺ｧ陦後▲縺ｦ縺・ｋ縲・
#if defined(P2_SMP_EVALCHECK)
// P2_SMP_EVALCHECK (2026-11-21 / 診断ビルド専用、 本番には入らない): Threads > 1 で helper の評価が main と違って TT を汚していないかの計装。
//   E: Worker::evaluate の値を P2SMP_SAMPLE 回に 1 回、 同じスレッドで sfen から作り直した局面 (accumulator を最初から作る) の評価と比べる。
//   R: search / qsearch の非 PV node で TT の eval (補正前の静的評価) をそのまま使うとき、 同じく作り直した局面の評価と比べる。
//   B: env P2SMP_SFENS の局面集を各 Worker が iterative_deepening の開始時 (phase 0) と rootDepth 10 の開始時 (phase 1) に評価し、
//      main (thread 0) の値と比べる (探索中の他スレッドの p2_search_begin などと並行に走る状態で測る)。
//   結果は go の終わりに main が info string P2SMP で出し、 計数を 0 に戻す。
namespace Eval { namespace P2NNUE { extern thread_local bool g_in_qsearch; } }
namespace P2SmpCheck {
constexpr int NT = 8;
static std::atomic<std::uint64_t> e_n[NT], e_bad[NT], r_n[NT], r_bad[NT];
static std::atomic<int> n_print{0};
static thread_local std::uint32_t tick = 0;
static std::uint32_t sample = 8;
static std::vector<std::string> sfens;
static std::vector<int> res[2][NT];
static bool res_done[2][NT];
static thread_local Position*  fpos = nullptr;
static thread_local StateInfo* fsi  = nullptr;
static struct Init {
	Init() {
		if (const char* e = std::getenv("P2SMP_SAMPLE")) sample = std::max(1, std::atoi(e));
		if (const char* f = std::getenv("P2SMP_SFENS")) {
			std::ifstream is(f); std::string l;
			while (std::getline(is, l)) {
				while (!l.empty() && (l.back() == '\r' || l.back() == ' ')) l.pop_back();
				if (l.rfind("sfen ", 0) == 0) l = l.substr(5);
				if (!l.empty()) sfens.push_back(l);
			}
		}
	}
} init_;
static inline int tid(size_t t) { return int(std::min<size_t>(t, NT - 1)); }
// 作り直した局面で評価する (スレッドごとに 1 回だけ確保。 診断ビルドなので開放しない)。
static Value fresh_eval(const std::string& sfen) {
	if (fpos == nullptr) { fpos = new Position(); fsi = new StateInfo(); }
	fpos->set(sfen, fsi);
	return Eval::evaluate(*fpos);
}
static void bad(const char* kind, size_t t, const Position& pos, int v, int r) {
	if (n_print.fetch_add(1) < 12)
		sync_cout << "info string P2SMP_BAD " << kind << " t " << t << " v " << v << " ref " << r
		          << " ply " << pos.game_ply() << " sfen " << pos.sfen() << sync_endl;
}
static inline void check_e(const Position& pos, Value v, size_t t) {
	if (++tick % sample) return;
	const Value r = fresh_eval(pos.sfen());
	e_n[tid(t)]++;
	if (r != v) { e_bad[tid(t)]++; bad("E", t, pos, int(v), int(r)); }
}
static inline void check_r(const Position& pos, Value ttEval, size_t t) {
	if (++tick % sample) return;
	const Value r = fresh_eval(pos.sfen());
	r_n[tid(t)]++;
	if (r != ttEval) { r_bad[tid(t)]++; bad("R", t, pos, int(ttEval), int(r)); }
}
static void batch(int phase, size_t t) {
	if (sfens.empty()) return;
	// node 入口の TLS を root / 単発 eval と同じ既定値にして測る (終わったら戻す)。
	const bool lh = Eval::p2_lazy_head; const int wa = Eval::p2_win_a, wb = Eval::p2_win_b;
	const bool qs = Eval::P2NNUE::g_in_qsearch;
	Eval::p2_lazy_head = false; Eval::p2_win_a = -32000; Eval::p2_win_b = 32000; Eval::P2NNUE::g_in_qsearch = false;
	auto& v = res[phase][tid(t)];
	v.clear();
	for (auto& s : sfens) v.push_back(int(fresh_eval(s)));
	res_done[phase][tid(t)] = true;
	Eval::p2_lazy_head = lh; Eval::p2_win_a = wa; Eval::p2_win_b = wb; Eval::P2NNUE::g_in_qsearch = qs;
}
static void report(size_t nthreads) {
	const int n = int(std::min<size_t>(nthreads, NT));
	for (int t = 0; t < n; ++t) {
		sync_cout << "info string P2SMP t " << t << " E " << e_bad[t].load() << "/" << e_n[t].load()
		          << " R " << r_bad[t].load() << "/" << r_n[t].load();
		for (int p = 0; p < 2; ++p) {
			if (!res_done[p][t] || !res_done[p][0]) { std::cout << " B" << p << " -"; continue; }
			int nb = 0;
			for (size_t i = 0; i < res[p][t].size(); ++i) nb += (res[p][t][i] != res[p][0][i]);
			std::cout << " B" << p << " " << nb << "/" << res[p][t].size();
		}
		std::cout << sync_endl;
		e_n[t] = e_bad[t] = r_n[t] = r_bad[t] = 0;
	}
	for (int p = 0; p < 2; ++p) for (int t = 0; t < NT; ++t) res_done[p][t] = false;
	n_print = 0;
}
}  // namespace P2SmpCheck
#define P2SMP_CHECK_R(pos, ev) P2SmpCheck::check_r(pos, ev, threadIdx)
#else
#define P2SMP_CHECK_R(pos, ev) ((void) 0)
#endif

#if defined(P2_OPTIMISM)
// P2_OPTIMISM (2026-11-21 / ユーザー「試したい！」): Stockfish の optimism 型の「自分と相手の非対称」。
//   読みの中の各局面で、 手番側が root の手番側 (自分) なら p2Opt[us]、 相手なら p2Opt[~us] を静的評価に足す (手番側視点)。
//   p2Opt[us]  = P2_OPT_US_CP   + dyn
//   p2Opt[~us] = P2_OPT_THEM_CP - dyn
//   dyn (SF 方式) = P2_OPT_DYN_SCALE * avg / (|avg| + P2_OPT_DYN_DIV)、 avg = 各反復の開始時の rootMoves[pvIdx].averageScore
//     (SF search.cpp の optimism[us] = 137 * avg / (|avg| + 91) / optimism[~us] = -optimism[us])。 avg が未設定 (-VALUE_INFINITE) なら dyn = 0。
//   混ぜ方は単純加算 (SF の駒の量で混ぜる式は入れない)。 加算する場所は to_corrected_static_eval の手前 (P2_OPT_ADD)。
//   🎯TT に入れるのは補正前の unadjustedStaticEval のまま (ttWriter.write の引数は変えていない) なので、 TT の eval は非対称で汚れない。
//     読み出した ttData.eval にも同じ場所で足す。 TT の value (探索値) には SF と同じく optimism 込みの値が入る。
//   既定 (US/THEM/SCALE が 0) なら p2Opt は常に 0 で、 + 0 は値を変えない = 従来と bit-exact。
//   P2_OPT_LOG = true なら main thread が反復ごとに info string P2OPT を出す (既定 false = 出さない)。
static int  g_p2opt_us        = 0;
static int  g_p2opt_them      = 0;
static int  g_p2opt_dyn_scale = 0;
static int  g_p2opt_dyn_div   = 91;
static bool g_p2opt_log       = false;
#define P2_OPT_ADD(v) Value((v) + p2Opt[pos.side_to_move()])
#else
#define P2_OPT_ADD(v) (v)
#endif

void YaneuraOuEngine::add_options() {

	// 東 蝓ｺ譛ｬ險ｭ螳・base class縺ｮadd_options()繧貞他縺ｳ蜃ｺ縺励※繧・ｋ)

	Engine::add_options();

	// 東 縺薙・謗｢邏｢驛ｨ縺檎畑縺・ｋ繧ｪ繝励す繝ｧ繝ｳ縺ｮ霑ｽ蜉縲・

#if STOCKFISH
    options.add(  //
        "USI_Hash", Option(16, 1, MaxHashMB, [this](const Option& o) {
            set_tt_size(o);
            return std::nullopt;
        }));
#else
	// 決 繧・・縺・ｉ邇九〒縺ｯ縲‥efault蛟､繧・024縺ｫ螟画峩縲・
    options.add(  //
        "USI_Hash", Option(1024, 1, MaxHashMB, [this](const Option& o) {
            set_tt_size(o);
            return std::nullopt;
        }));
#endif

	// 縺昴・螻髱｢縺ｧ縺ｮ荳贋ｽ康蛟九・蛟呵｣懈焔繧定ｪｿ縺ｹ繧区ｩ溯・
    // 竍ｨ縲縺薙ｌMAX_MOVES縺ｧ蜊∝・縲・
    options.add("MultiPV", Option(1, 1, MAX_MOVES));

    options.add("DrawValueBlack", Option(-2, -30000, 30000));   // 2026-11-24 public default: neutral (was -53 for Denryu rule)  // 11-16 既定を対局時の値へ (旧 -2)
    options.add("DrawValueWhite", Option(-2, -30000, 30000));   // 2026-11-24 public default: neutral (was +41 for Denryu rule)   // 11-16 既定を対局時の値へ (旧 -2)

    // 謚穂ｺ・せ繧ｳ繧｢
    options.add("ResignValue", Option(99999, 0, 99999));

    // BookExitTimeFix (2026-11-17): 探索を飛ばした手 (定跡・宣言勝ち・合法手なし・投了) の後で、 次の手の時間配分が
    //   「前回評価 = -VALUE_INFINITE」を読んで fallingEval を下限に張り付け、 定跡を抜けた最初の 1 手の考慮時間が約 4 割減る欠陥の修正。
    //   true = 修正あり (探索を飛ばしたときは前回評価を VALUE_INFINITE = 「前回評価なし」にする)。 false = 従来と完全一致。
    options.add("BookExitTimeFix", Option(true));

#if defined(P2_OPTIMISM)
    // P2_OPTIMISM (11-21): 自分と相手の非対称 (g_p2opt_* の注記)。 既定 0 = 従来と bit-exact。
    options.add("P2_OPT_US_CP", Option(0, -500, 500, [](const Option& o) { g_p2opt_us = int(o); return std::nullopt; }));
    options.add("P2_OPT_THEM_CP", Option(0, -500, 500, [](const Option& o) { g_p2opt_them = int(o); return std::nullopt; }));
    options.add("P2_OPT_DYN_SCALE", Option(0, 0, 1000, [](const Option& o) { g_p2opt_dyn_scale = int(o); return std::nullopt; }));
    options.add("P2_OPT_DYN_DIV", Option(91, 1, 10000, [](const Option& o) { g_p2opt_dyn_div = int(o); return std::nullopt; }));
    options.add("P2_OPT_LOG", Option(false, [](const Option& o) { g_p2opt_log = bool(o); return std::nullopt; }));
#endif

#if defined(P2_KHCORR)
    // 2026-11-12 KHCORR (案①): 玉近傍 x 浮き駒 鍵の correction history。 既定 0 = OFF (bit-exact)。
    //   ON の目安は CORR_PCV_WEIGHT (6540) と同じ桁。
    options.add("CORR_KH_WEIGHT", Option(SPSA::CORR_KH_WEIGHT, 0, 32768, [](const Option& o) {
                    SPSA::CORR_KH_WEIGHT = int(o);
                    return std::nullopt;
                }));
    options.add("CORR_HIST_KH", Option(SPSA::CORR_HIST_KH, 0, 1024, [](const Option& o) {
                    SPSA::CORR_HIST_KH = int(o);
                    return std::nullopt;
                }));
#endif
#if defined(P2_KHGATE)
    // 2026-11-12 KHGATE: PESS (静的評価の固定の悲観補正) / SELEXT (荒れた node の LMR 緩和・枝刈り停止)。 既定 0 = OFF (bit-exact)。
    options.add("PESS_HANG", Option(SPSA::PESS_HANG, 0, 200, [](const Option& o) {
                    SPSA::PESS_HANG = int(o);
                    return std::nullopt;
                }));
    options.add("PESS_ZA", Option(SPSA::PESS_ZA, 0, 200, [](const Option& o) {
                    SPSA::PESS_ZA = int(o);
                    return std::nullopt;
                }));
    options.add("PESS_HANG_THEM", Option(SPSA::PESS_HANG_THEM, 0, 200, [](const Option& o) {
                    SPSA::PESS_HANG_THEM = int(o);
                    return std::nullopt;
                }));
    options.add("PESS_ZA_THEM", Option(SPSA::PESS_ZA_THEM, 0, 200, [](const Option& o) {
                    SPSA::PESS_ZA_THEM = int(o);
                    return std::nullopt;
                }));
    options.add("SELEXT_LMR", Option(SPSA::SELEXT_LMR, 0, 3, [](const Option& o) {
                    SPSA::SELEXT_LMR = int(o);
                    return std::nullopt;
                }));
    options.add("SELEXT_NOPRUNE", Option(SPSA::SELEXT_NOPRUNE, 0, 1, [](const Option& o) {
                    SPSA::SELEXT_NOPRUNE = int(o);
                    return std::nullopt;
                }));
    options.add("SELEXT_HANG_TH", Option(SPSA::SELEXT_HANG_TH, 1, 16, [](const Option& o) {
                    SPSA::SELEXT_HANG_TH = int(o);
                    return std::nullopt;
                }));
    options.add("SELEXT_ZA_TH", Option(SPSA::SELEXT_ZA_TH, 1, 24, [](const Option& o) {
                    SPSA::SELEXT_ZA_TH = int(o);
                    return std::nullopt;
                }));
#endif

	// 東 SearchOptions縺檎畑縺・ｋ繧ｪ繝励す繝ｧ繝ｳ縺ｮ霑ｽ蜉

	manager.search_options.add_options(options);

	// 東 TimeManagement縺檎畑縺・ｋ繧ｪ繝励す繝ｧ繝ｳ縺ｮ霑ｽ蜉

    manager.tm.add_options(options);

	// 東 螳夊ｷ｡縺檎畑縺・ｋ繧ｪ繝励す繝ｧ繝ｳ縺ｮ霑ｽ蜉

    book.add_options(options);

#if defined(USE_BOOK_AWARE)
	// 🎯BookAware: 定跡を「指す」のではなく「避ける / 向かう」ために使う。
	//   オプションは 3 個だけ (BookAwareBias / BookAwareList / BookAwareMaxPly)。
	//   定跡が増えてもオプションは増えない (リストファイルに行を足す)。
	BookAware::manager.add_options(options);
#endif

    // 庁  莉･荳九・險ｭ螳壹・縺・■縲・isready"縺ｮ繧ｿ繧､繝溘Φ繧ｰ縺ｧoptions縺九ｉ蛟､繧貞叙蠕励☆繧九ｂ縺ｮ縺ｫ髢｢縺励※縺ｯ
    //      event handler縺ｯ險ｭ螳壹＠縺ｪ縺・・

    // 蠑輔″蛻・￠繧貞女縺大・繧後ｋ繧ｹ繧ｳ繧｢
    // 豁ｩ繧・00縺ｨ縺吶ｋ縲ゆｾ九∴縺ｰ縲√％縺ｮ蛟､繧・-100縺ｫ縺吶ｋ縺ｨ蠑輔″蛻・￠縺ｮ螻髱｢縺ｯ隧穂ｾ｡蛟､縺・-100縺ｨ縺ｿ縺ｪ縺輔ｌ繧九・

    // 蜊・律謇九〒縺ｮ蠑輔″蛻・￠繧貞屓驕ｿ縺励ｄ縺吶￥縺ｪ繧九ｈ縺・↓縲√ョ繝輔か繝ｫ繝亥､繧・2縲・
    // 縺｡縺ｪ縺ｿ縺ｫ縲・2縺ｫ縺励※縺ゅｋ縺ｮ縺ｯ縲・
    //  int draw_value = Options["DrawValueBlack"] * PawnValue / 100; 縺ｧPawnValue縺・00繧医ｊ蟆上＆縺・・縺ｧ
    // 1縺縺ｨ蛻・ｊ謐ｨ縺ｦ繧峨ｌ縺ｦ縺励∪縺・°繧峨〒縺ゅｋ縲・

    // Stockfish縺ｧ縺ｯ"Contempt"縺ｨ縺・≧繧ｨ繝ｳ繧ｸ繝ｳ繧ｪ繝励す繝ｧ繝ｳ縺ｧ縺ゅ▲縺溘′縲∝・蠕後・蛹ｺ蛻･縺後▽縺代ｉ繧後↑縺・＠縲・
    // 蛻・°繧翫▼繧峨＞縺ｮ縺ｧ螟画峩縺励◆縲・


    // 東 Stockfish縺ｫ縺ｯ縺ゅｋ縺後√ｄ縺ｭ縺・ｉ邇九〒縺ｯ繧ｵ繝昴・繝医＠縺ｪ縺・が繝励す繝ｧ繝ｳ
#if STOCKFISH
	// 蠑ｱ縺上☆繧九◆繧√↓隱ｿ謨ｴ縺吶ｋ縲・0縺ｪ繧画焔蜉貂帙↑縺励・縺梧怙蠑ｱ縲・
	options.add("Skill Level", Option(20, 0, 20));

	options.add("Move Overhead", Option(10, 0, 5000));

	// nodes as time繝｢繝ｼ繝峨・
	// 繝溘Μ遘偵≠縺溘ｊ縺ｮ繝弱・繝画焚繧定ｨｭ螳壹☆繧九Ｈo繧ｳ繝槭Φ繝峨〒btime縺後√％縺薙〒險ｭ螳壹＠縺溷､縺ｫ謗帙￠邂励＆繧後◆繝弱・繝画焚繧呈爾邏｢縺ｮ荳企剞縺ｨ縺吶ｋ縲・
	// 0繧呈欠螳壹☆繧後・nodes as time繝｢繝ｼ繝峨〒縺ｯ縺ｪ縺・・
	// 萓・ 600knps縺ｮPC蜍穂ｽ懊ｒ繧ｷ繝溘Η繝ｬ繝ｼ繝医☆繧九↑繧峨・600繧呈欠螳壹☆繧九・
	// 統 繧・・縺・ｉ邇九〒縺ｯ縲√％縺ｮ讖溯・縲√し繝昴・繝医＠縺ｪ縺・％縺ｨ縺ｫ縺吶ｋ縲・UI縺後←縺・○繧ｵ繝昴・繝医＠縺ｦ縺・↑縺・・縺ｧ窶ｦ縲・
	options.add("nodestime", Option(0, 0, 10000));

	options.add("UCI_Chess960", Option(false));

	options.add("UCI_LimitStrength", Option(false));

	// Elo繝ｬ繝ｼ繝・ぅ繝ｳ繧ｰ繧呈欠螳壹＠縺ｦ譽句鴨隱ｿ謨ｴ縺吶ｋ縺溘ａ縺ｮ繧ｨ繝ｳ繧ｸ繝ｳ繧ｪ繝励す繝ｧ繝ｳ縲・
	options.add("UCI_Elo",
		Option(Stockfish::Search::Skill::LowestElo, Stockfish::Search::Skill::LowestElo,
			Stockfish::Search::Skill::HighestElo));

	options.add("UCI_ShowWDL", Option(false));

	// 庁 Syzygy縺ｨ縺ｯ縲∫ｵら乢縺ｮTablebases髢｢騾｣縲ょｰ・｣九〒縺ｯ菴ｿ縺医↑縺・・

	options.add(  //
		"SyzygyPath", Option("", [](const Option& o) {
			Tablebases::init(o);
			return std::nullopt;
			}));

	options.add("SyzygyProbeDepth", Option(1, 1, 100));

	options.add("Syzygy50MoveRule", Option(true));

	options.add("SyzygyProbeLimit", Option(7, 0, 7));

#endif

	// 決 tune.py縺ｫ繧医▲縺ｦ縺薙％莉･荳九↓閾ｪ蜍慕噪縺ｫ繧ｨ繝ｳ繧ｸ繝ｳ繧ｪ繝励す繝ｧ繝ｳ縺瑚ｿｽ蜉縺輔ｌ繧九・
	//                      %%TUNE_OPTIONS%%


	// 繧ゅ＠TUNE()繝槭け繝ｭ縺ｧ譁ｰ縺溘↓繝代Λ繝｡繝ｼ繧ｿ繝ｼ繧定ｿｽ蜉縺励◆縺ｪ繧峨√◎繧後ｒ蜿肴丐縺輔○繧九・
	Tune::init(options);
}

// "isready"縺ｮ繧ｿ繧､繝溘Φ繧ｰ縺ｧ縺ｮ蛻晄悄蛹門・逅・・
void YaneuraOuEngine::isready() {

    // M1 遅延 (ABDADA-lite) の busy 表を空にする   2026-11-02
    //   enter / leave は RAII で対になっているので探索後は原則空だが、
    //   way の奪い合いで**取り残された印**が少しずつ溜まる。 溜まると偽陽性が増えて
    //   「全部後回し」になるので、 isready で確実に空に戻す (256 KB の memset = 数十 µs)。
    //   🚨正しさには影響しない (偽陽性は余計に後回しにするだけ)。 性能の保険。
    smpBusy.clear();

    // M3 根の YBWC: 世代を無効化してカウンタを 0 に戻す。
    //   🚨isready を跨いで古い世代が残ると、 次の go の最初の publish 前に
    //     all_taken_done() が古い印を見て止まる恐れがある。
    rootSplit.reset();
    rootSplit.clear_counters();

    // M4: 枠を全部空にする。 isready を跨いで TAKEN が残ると提示できなくなる。
    deleg.clear();

	// 決 繧・・縺・ｉ邇狗峡閾ｪ繧ｪ繝励す繝ｧ繝ｳ縺ｮ蜀・ｮｹ繧定ｨｭ螳壹↑縺ｩ縺ｫ蜿肴丐縺輔○繧九・

#if defined(USE_CLASSIC_EVAL)
    // 東 譌ｧ隧穂ｾ｡髢｢謨ｰ縺ｯ縲（sready縺ｫ蟇ｾ縺励※蜻ｼ縺ｳ蜃ｺ縺吶・
    //     隧穂ｾ｡髢｢謨ｰ繝代Λ繝｡繝ｼ繧ｿ繝ｼ縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ蜃ｦ逅・′蠢・ｦ√・
    Engine::run_heavy_job([] { Eval::load_eval(); });

    // 蛻晄悄蛹悶ち繧､繝溘Φ繧ｰ縺後↑縺九▲縺溘・縺ｧ縲・
    // 縺薙・繧ｿ繧､繝溘Φ繧ｰ縺ｧ蟷ｳ謇九・螻髱｢縺ｫ蛻晄悄蛹悶＠縺ｦ縺翫￥縲・
    // 庁 縺薙ｌ繧偵＠縺ｦ縺翫°縺ｪ縺・→繝・ヰ繝・げ縺ｮ譎ゅ↓縲・isready"縺ｮ縺ゅ→
    //     "position"繧ｳ繝槭Φ繝峨ｒ騾∽ｿ｡縺励↑縺・→螻髱｢縺御ｸ肴ｭ｣縺ｧ關ｽ縺｡縺ｦ髱｢蛟偵・
    states = StateListPtr(new std::deque<StateInfo>(1));
    pos.set(StartSFEN, &states->back());
#endif

	// 東 蝓ｺ譛ｬ險ｭ螳・base class縺ｮisready()繧貞他縺ｳ蜃ｺ縺励※繧・ｋ)
	//Engine::isready();
	// ､・tt縺ｮclear縺ｫ蜈育ｫ九■縲ゝhreadPool縺後せ繝ｬ繝・ラ繧堤畑諢上＠縺ｦ縺上ｌ縺ｦ縺・↑縺・→縺・￠縺ｪ縺・・

	// 繧ｨ繝ｳ繧ｸ繝ｳ險ｭ螳壹・繧ｹ繝ｬ繝・ラ謨ｰ繧貞渚譏縺輔○繧九・
    resize_threads();

	// 鄂ｮ謠幄｡ｨ縺ｮclear
	tt.clear(threads);

	// Stockfish縺ｮThreadPool::clear()縺ｫ縺ゅ▲縺溘さ繝ｼ繝峨・
	clear();

	// 螳夊ｷ｡縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ
    book.read_book();

#if defined(USE_BOOK_AWARE)
	// 🎯BookAware: リストファイルを読んで N 本の定跡を開く。
	//   BookAwareBias=0 (既定) のときはファイルを開かない = bit-exact。
	BookAware::manager.read_books();
#endif

	// 決 tune.py縺ｫ繧医▲縺ｦ縺薙％莉･荳九↓閾ｪ蜍慕噪縺ｫ繧ｨ繝ｳ繧ｸ繝ｳ繧ｪ繝励す繝ｧ繝ｳ縺瑚ｿｽ蜉縺輔ｌ繧九・
    //                      %%TUNE_ISREADY%%


    sync_cout << "readyok" << sync_endl;
}

// Stockfish縺ｮThreadPool::clear()縺ｫ縺ゅ▲縺溘ｂ縺ｮ縲・
// 統 isready()縺九ｉ蜻ｼ縺ｳ蜃ｺ縺輔ｌ繧九ょｯｾ螻髢句ｧ区凾縺ｫ螳溯｡後＆繧後ｋ縲・
void YaneuraOuEngine::clear()
{
#if defined(USE_BOOK_AWARE)
	// 🎯BookAware: 対局ごとにゲートの観測をリセットする (clear は isready から呼ばれる)。
	BookAware::manager.reset_gate();
#endif

    // ､・莉･荳九・蛻晄悄蛹悶・縲ヾtockfish縺ｮThreadPool::clear()縺ｫ縺ゅ▲縺溘ｂ縺ｮ縲・
    //     繧・・縺・ｉ邇九〒縺ｯ縲√％繧後・Engine豢ｾ逕歡lass縺ｧ陦後≧縲・

    // These two affect the time taken on the first move of a game:
    // 縺薙ｌ繧・縺､縺ｯ縲√ご繝ｼ繝縺ｮ譛蛻昴・謇九↓縺九°繧区凾髢薙↓蠖ｱ髻ｿ縺吶ｋ縲・

    main_manager()->bestPreviousAverageScore = VALUE_INFINITE;
    main_manager()->previousTimeReduction    = 0.85;

    main_manager()->callsCnt          = 0;
    main_manager()->bestPreviousScore = VALUE_INFINITE;
#if defined(P2_BUCKET_OFFSET)
    // SIDEOFF2 (11-21): isready で直前の手の root 探索値の記録を消す (P2_SENTE_TABLE_SWITCH_CP)。
    Eval::p2_root_result_reset();
#endif
#if STOCKFISH
    main_manager()->originalTimeAdjust = -1;
#else
    main_manager()->lastGamePly = 0;
#endif
    main_manager()->tm.clear();
}


// 決 "ponderhit"縺ｫ蟇ｾ縺吶ｋ蜃ｦ逅・・
void YaneuraOuEngine::set_ponderhit(bool b) {

	// 統 ponderhit縺励◆縺ｮ縺ｧ縲√ｄ縺ｭ縺・ｉ邇九〒縺ｯ縲・
    //     迴ｾ蝨ｨ譎ょ綾繧稚imer class縺ｫ菫晏ｭ倥＠縺ｦ縺翫￥蠢・ｦ√′縺ゅｋ縲・
	// 庁 ponder繝輔Λ繧ｰ繧貞､画峩縺吶ｋ蜑阪↓縺薙■繧峨ｒ蜈医↓螳溯｡後＠縺ｪ縺・→
	//     ponder繝輔Λ繧ｰ繧定ｦ九※ponderhitTime繧貞盾辣ｧ縺励※髢馴＆縺｣縺溯ｨ育ｮ励ｒ縺励※縺励∪縺・・
    manager.tm.ponderhitTime = now();

	manager.ponder           = b;
}

// 繧ｹ繝ｬ繝・ラ謨ｰ繧貞渚譏縺輔○繧矩未謨ｰ
void YaneuraOuEngine::resize_threads() {
    // 庁 Engine::resize_threads()繧貞盾閠・↓譖ｸ縺上・

    // 東 謗｢邏｢縺ｮ邨ゆｺ・ｒ蠕・▽
    threads.wait_for_search_finished();

    // 東 繧ｹ繝ｬ繝・ラ謨ｰ縺ｮ繝ｪ繧ｵ繧､繧ｺ

    auto worker_factory = [&](SharedState& sharedState, const ThreadIds& ids)
	{

		auto p = make_unique_large_page<Search::YaneuraOuWorker>(
			// Worker縺ｮ繧ｳ繝ｳ繧ｹ繝医Λ繧ｯ繧ｿ縺梧ｸ｡縺励※谺ｲ縺励＞繧ゅ・縲・
			sharedState,
			ids,
			// 霑ｽ蜉縺ｧYaneuraOuEngine縺九ｉ繧ゅｉ縺・◆縺・ｂ縺ｮ
			*this
		);

		return LargePagePtr<Worker>(p.release());  // Worker* 縺ｫ upcast
    };

    threads.set(numaContext.get_numa_config(), {options, threads, tt, sharedHists /*, networks*/ },
                updateContext , options["Threads"], worker_factory);

	// 鄂ｮ謠幄｡ｨ縺ｮ蜑ｲ繧雁ｽ薙※
	set_tt_size(options["USI_Hash"]);
 
    // 東 NUMA縺ｮ險ｭ螳・

    // 繧ｹ繝ｬ繝・ラ縺ｮ逕ｨ縺・ｋ隧穂ｾ｡髢｢謨ｰ繝代Λ繝｡繝ｼ繧ｿ繝ｼ縺梧ｭ｣縺励＞NUMA縺ｫ螻槭☆繧九ｈ縺・↓縺吶ｋ
    threads.ensure_network_replicated();
}

// 鄂ｮ謠幄｡ｨ縺ｮ蜑ｲ繧雁ｽ薙※
void YaneuraOuEngine::set_tt_size(size_t mb){
	wait_for_search_finished();
	tt.resize(mb, threads);
}

// 鄂ｮ謠幄｡ｨ縺ｮ菴ｿ逕ｨ邇・ｒ霑斐☆縲・
// 決 繧・・縺・ｉ邇九〒縺ｯ豢ｾ逕歡lass蛛ｴ縺ｧget_hashfull()繧貞ｮ溯｣・☆繧九・
int YaneuraOuEngine::get_hashfull(int maxAge) const
{
	return tt.hashfull(maxAge);
}

namespace {

constexpr size_t QSEARCH_PSV_CHUNK_RECORDS = 65536;

struct QSearchPsvStats {
    u64 records       = 0;
    u64 replaced      = 0;
    u64 decodeErrors  = 0;
    u64 illegalPv     = 0;
    u64 maxLeafPly    = 0;

    void merge(const QSearchPsvStats& rhs) {
        records      += rhs.records;
        replaced     += rhs.replaced;
        decodeErrors += rhs.decodeErrors;
        illegalPv    += rhs.illegalPv;
        maxLeafPly    = std::max(maxLeafPly, rhs.maxLeafPly);
    }
};

// PsvRecord 1莉ｶ縺ｫ蟇ｾ縺励※ qsearch<PV>() 繧貞ｮ溯｡後＠縲・
// 蠕励ｉ繧後◆PV繧貞ｮ滄圀縺ｫ騾ｲ繧√◆leaf node縺ｮ螻髱｢縺ｧrecord.sfen繧堤ｽｮ謠帙☆繧九・
// score/game_result縺ｯPSV縺ｮ隕冗ｴ・↓蜷医ｏ縺帙※縲〕eaf蛛ｴ縺ｮ謇狗分隕也せ縺ｫ縺ｪ繧九ｈ縺・ｿ・ｦ√↑繧臥ｬｦ蜿ｷ蜿崎ｻ｢縺吶ｋ縲・

// 2026-11-05: **蒸留用のラベル付け直し** (env P2_RELABEL_ALPHA=0..100 で有効)。
//   狙い: threat 入力を持たない plain net に、 現職 (threat あり) net の評価関数を
//   真似させる。 fgb_tr は既に qsearch 末端の局面なので (★qsearch_leaf_data)
//   **局面は動かさず** score だけを混ぜる:
//       new = ((100-a) * 元ラベル + a * 現職 net の静的 eval) / 100
//   a=100 で完全な自己蒸留、 a=0 で無変更 (= 検算に使える)。
//   🚨移せるのは「ラベルの滑らかさ」だけ。 threat は駒の**対** (2 次) の関係で、
//     plain net の FT は 1 次の和なので**対の情報自体は原理的に移らない**。
static int relabel_alpha() {
    static const int v = []() {
        const char* s = std::getenv("P2_RELABEL_ALPHA");
        if (s == nullptr)
            return -1;
        const int a = std::atoi(s);
        return (a < 0 || a > 100) ? -1 : a;
    }();
    return v;
}

// ラベルだけ差し替える (局面・手番・game_result・gamePly はそのまま)。
bool relabel_psv_record(PsvRecord& record, int alpha, QSearchPsvStats& stats) {
    Position  pos;
    StateInfo si;
    if (pos.set_from_packed_sfen(record.sfen, &si, false, record.gamePly).is_not_ok())
    {
        ++stats.decodeErrors;
        return false;
    }
    const int ev    = int(Eval::evaluate(pos));
    const int mixed = ((100 - alpha) * int(record.score) + alpha * ev) / 100;
    record.score    = s16(std::max(-32000, std::min(32000, mixed)));
    ++stats.replaced;
    return true;
}

bool qsearch_psv_record(YaneuraOuWorker& worker, PsvRecord& record, QSearchPsvStats& stats) {
    ++stats.records;

    // 蒸留モード: 局面は動かさず score だけ付け直す。
    {
        const int alpha = relabel_alpha();
        if (alpha >= 0)
            return relabel_psv_record(record, alpha, stats);
    }

    Position  pos;
    StateInfo si;

    if (pos.set_from_packed_sfen(record.sfen, &si, false, record.gamePly).is_not_ok())
    {
        ++stats.decodeErrors;
        return false;
    }

    PVMoves pv;
    worker.qsearch_pv(pos, pv);

    if (pv.empty())
        return true;

    std::vector<StateInfo> states(pv.size());
    size_t                 leafPly = 0;

    for (Move move : pv)
    {
        if (!move.is_ok() || !(pos.pseudo_legal_s<true>(move) && pos.legal(move)))
        {
            ++stats.illegalPv;
            return false;
        }

        pos.do_move(move, states[leafPly]);
        ++leafPly;
    }

    pos.sfen_pack(record.sfen);

    const int gamePly = pos.game_ply();
    record.gamePly =
      static_cast<u16>(std::min(gamePly, int((std::numeric_limits<u16>::max)())));

    // PsvRecord縺ｮscore/game_result縺ｯ謇狗分蛛ｴ隕也せ縺ｪ縺ｮ縺ｧ縲・
    // leaf縺ｾ縺ｧ螂・焚謇矩ｲ繧薙□蝣ｴ蜷医・隕也せ繧貞渚霆｢縺吶ｋ縲・
    if (leafPly & 1)
    {
        record.score = record.score == (std::numeric_limits<s16>::min)()
                       ? (std::numeric_limits<s16>::max)()
                       : s16(-record.score);
        record.game_result = s8(-record.game_result);
    }

    // root螻髱｢逕ｨ縺ｮbest move縺ｯleaf螻髱｢縺ｧ縺ｯ騾壼ｸｸ蜷域ｳ輔〒縺ｪ縺・・
    record.move = Move16::none().to_u16();

    ++stats.replaced;
    stats.maxLeafPly = std::max<u64>(stats.maxLeafPly, leafPly);
    return true;
}

} // namespace

// USI諡｡蠑ｵ繧ｳ繝槭Φ繝・"qsearch_psv input.psv output.psv [workers]" 縺ｮ螳滉ｽ薙・
// .psv(PsvRecord蛻・繧団hunk蜊倅ｽ阪〒隱ｭ縺ｿ霎ｼ縺ｿ縲∝推繝ｬ繧ｳ繝ｼ繝峨・螻髱｢繧・
// qsearch縺ｮPV leaf node縺ｫ鄂ｮ謠帙＠縺ｦ縲∝酔縺榔SV蠖｢蠑上〒outputPath縺ｸ譖ｸ縺榊・縺吶・
// qsearch閾ｪ菴薙ｒ荳ｦ蛻怜喧縺吶ｋ縺ｮ縺ｧ縺ｯ縺ｪ縺上∫峡遶九＠縺蘖SV繝ｬ繧ｳ繝ｼ繝峨ｒ譌｢蟄倥・謗｢邏｢worker縺ｸ蛻・・縺吶ｋ縲・
bool YaneuraOuEngine::qsearch_psv(const std::string& inputPath,
                                  const std::string& outputPath,
                                  size_t             workerCount,
                                  std::string&       message) {

#if !defined(USE_SFEN_PACKER)
    message = "qsearch_psv requires USE_SFEN_PACKER.";
    return false;
#else
    if (inputPath.empty() || outputPath.empty())
    {
        message = "usage: qsearch_psv input.psv output.psv [workers]";
        return false;
    }

    if (inputPath == outputPath)
    {
        message = "input and output path must be different.";
        return false;
    }

    wait_for_search_finished();

    if (threads.empty())
        resize_threads();

    if (threads.empty())
    {
        message = "no search worker is available.";
        return false;
    }

    if (workerCount == 0)
        workerCount = threads.num_threads();
    else if (workerCount > threads.num_threads())
    {
        message = "requested workers exceed current Threads option.";
        return false;
    }

    std::ifstream input(inputPath, std::ios::binary);
    if (!input)
    {
        message = "failed to open input file: " + inputPath;
        return false;
    }

    std::ofstream output(outputPath, std::ios::binary);
    if (!output)
    {
        message = "failed to open output file: " + outputPath;
        return false;
    }

    std::vector<PsvRecord> records(QSEARCH_PSV_CHUNK_RECORDS);
    QSearchPsvStats                 total;
    u64                             nextProgress = 1000000;

    const auto recordBytes = std::streamsize(sizeof(PsvRecord));
    const auto chunkBytes  = std::streamsize(records.size() * sizeof(PsvRecord));

    while (true)
    {
        input.read(reinterpret_cast<char*>(records.data()), chunkBytes);
        const std::streamsize bytesRead = input.gcount();

        if (bytesRead == 0)
            break;

        if ((bytesRead % recordBytes) != 0)
        {
            message = "truncated psv record at end of input.";
            return false;
        }

        const size_t count = size_t(bytesRead / recordBytes);

        std::atomic<size_t>         nextRecord(0);
        std::vector<QSearchPsvStats> localStats(workerCount);

        for (size_t threadId = 0; threadId < workerCount; ++threadId)
        {
            threads.run_on_thread(threadId, [&, threadId]() {
                auto* worker =
                  static_cast<YaneuraOuWorker*>(threads.threads[threadId]->worker.get());

                while (true)
                {
                    const size_t index = nextRecord.fetch_add(1, std::memory_order_relaxed);
                    if (index >= count)
                        break;

                    qsearch_psv_record(*worker, records[index], localStats[threadId]);
                }
            });
        }

        for (size_t threadId = 0; threadId < workerCount; ++threadId)
            threads.wait_on_thread(threadId);

        for (const auto& stats : localStats)
            total.merge(stats);

        output.write(reinterpret_cast<const char*>(records.data()), bytesRead);
        if (!output)
        {
            message = "failed to write output file: " + outputPath;
            return false;
        }

        if (total.records >= nextProgress)
        {
            sync_cout << "info string qsearch_psv processed " << total.records << " records"
                      << sync_endl;
            nextProgress += 1000000;
        }
    }

    if (input.bad())
    {
        message = "failed to read input file: " + inputPath;
        return false;
    }

    std::ostringstream ss;
    ss << "qsearch_psv done: records=" << total.records
       << " replaced=" << total.replaced
       << " decode_errors=" << total.decodeErrors
       << " illegal_pv=" << total.illegalPv
       << " max_leaf_ply=" << total.maxLeafPly
       << " workers=" << workerCount;
    message = ss.str();
    return total.decodeErrors == 0 && total.illegalPv == 0;
#endif
}

// utility functions

void YaneuraOuEngine::trace_eval() const {
    StateListPtr trace_states(new std::deque<StateInfo>(1));
    Position     p;
#if STOCKFISH
	p.set(pos.fen(), options["UCI_Chess960"], &trace_states->back());
#else
    p.set(pos.sfen(),&trace_states->back());
#endif
    verify_networks();

    //sync_cout << "\n" << Eval::trace(p, *networks) << sync_endl;
	// TODO 縺ゅ→縺ｧ
#if defined(EVAL_P2NNUE)
	// P2NNUE: 髱咏噪隧穂ｾ｡縺ｮ蜀・ｨｳ繧貞・蜉帙☆繧・讀懆ｨｼ逕ｨ)縲・
	Eval::print_eval_stat(p);
#endif
}

// 迴ｾ蝨ｨ縺ｮ螻髱｢縺ｮ隧穂ｾ｡蛟､繧貞・蜉帙☆繧九・
Value YaneuraOuEngine::evaluate() const {
	verify_networks();
    return Eval::evaluate(pos); // cp縺ｫ螟画鋤縺吶ｋ縺具ｼ溘∪縺ゅ＞縺・°窶ｦ縲・
}

// 荳ｦ蛻玲爾邏｢縺ｫ縺翫＞縺ｦ荳逡ｪ濶ｯ縺・晁・ｒ縺励◆thread縺ｮ驕ｸ蜃ｺ縲・
// 庁 Stockfish縺ｧ縺ｯ ThreadPool::get_best_thread()縺ｫ逶ｸ蠖薙☆繧九ｂ縺ｮ縲・

YaneuraOuWorker* YaneuraOuWorker::get_best_thread() const {

    // M3 根の YBWC: 根の分業は論理的に 1 つの探索なので、 完全な rootMoves を持つ
    //   master を必ず返す。 helper の rootMoves は長兄も飛ばした手も埋まっていないので
    //   投票に使えない (source/smp_root_ybwc.h の不変条件)。
    if (rootSplit)
        return const_cast<YaneuraOuWorker*>(this);

    auto& threads = this->threads.threads;

    Thread* bestThread = threads.front().get();
    Value   minScore   = VALUE_NONE;

    std::unordered_map<Move, int64_t, Move::MoveHash> votes(
      2 * std::min(threads.size(), bestThread->worker->rootMoves.size()));

    // Find the minimum score of all threads
    for (auto&& th : threads)
        minScore = std::min(minScore, th->worker->rootMoves[0].score);

    // Vote according to score and depth, and select the best thread
    auto thread_voting_value = [minScore](Thread* th) {
        // Worker縺九ｉ豢ｾ逕溘＆縺帙※縺・ｋ縺ｮ縺ｧdynamic_cast縺励※縺励∪縺・・
		// 笞 RTTI繧呈怏蜉ｹ蛹悶＠縺ｦ縺・↑縺・・縺ｧ蜊倥↓static_cast縺ｫ縺吶ｋ縲・
        auto worker = static_cast<Search::YaneuraOuWorker*>(th->worker.get());
        return (th->worker->rootMoves[0].score - minScore + 14) * int(worker->completedDepth);
    };

    for (auto&& th : threads)
        votes[th->worker->rootMoves[0].pv[0]] += thread_voting_value(th.get());

    for (auto&& th : threads)
    {
        const auto bestThreadScore = bestThread->worker->rootMoves[0].score;
        const auto newThreadScore  = th->worker->rootMoves[0].score;

        const auto& bestThreadPV = bestThread->worker->rootMoves[0].pv;
        const auto& newThreadPV  = th->worker->rootMoves[0].pv;

        const auto bestThreadMoveVote = votes[bestThreadPV[0]];
        const auto newThreadMoveVote  = votes[newThreadPV[0]];

        const bool bestThreadInProvenWin = is_win(bestThreadScore);
        const bool newThreadInProvenWin  = is_win(newThreadScore);

        const bool bestThreadInProvenLoss =
          bestThreadScore != -VALUE_INFINITE && is_loss(bestThreadScore);
        const bool newThreadInProvenLoss =
          newThreadScore != -VALUE_INFINITE && is_loss(newThreadScore);

        // We make sure not to pick a thread with truncated principal variation
        const bool betterVotingValue =
          thread_voting_value(th.get()) * int(newThreadPV.size() > 2)
          > thread_voting_value(bestThread) * int(bestThreadPV.size() > 2);

        if (bestThreadInProvenWin)
        {
            // Make sure we pick the shortest mate / TB conversion
            if (newThreadScore > bestThreadScore)
                bestThread = th.get();
        }
        else if (bestThreadInProvenLoss)
        {
            // Make sure we pick the shortest mated / TB conversion
            if (newThreadInProvenLoss && newThreadScore < bestThreadScore)
                bestThread = th.get();
        }
        else if (newThreadInProvenWin || newThreadInProvenLoss
                 || (!is_loss(newThreadScore)
                     && (newThreadMoveVote > bestThreadMoveVote
                         || (newThreadMoveVote == bestThreadMoveVote && betterVotingValue))))
            bestThread = th.get();
    }

    // Thread縺ｫ蟇ｾ縺励※worker縺悟ｾ励ｉ繧後ｋ縺九ｉ縲√％繧後・YaneuraOuWorker*縺ｪ縺ｮ縺ｧdynamic_cast縺励※霑斐☆縲・
    // 笞 RTTI繧呈怏蜉ｹ蛹悶＠縺ｦ縺・↑縺・・縺ｧ蜊倥↓static_cast縺ｫ縺吶ｋ縲・
    //return dynamic_cast<YaneuraOuWorker*>(bestThread->worker.get());
    return static_cast<YaneuraOuWorker*>(bestThread->worker.get());
}

// -----------------------------------------------------------
// 東 縺薙％莉･髯阪・Stockfish縺ｮsearch.cpp繧貞盾閠・↓縺励↑縺後ｉ譖ｸ縺上・東
// -----------------------------------------------------------


#if STOCKFISH
namespace Stockfish {
// 庁 縺薙・繝輔ぃ繧､繝ｫ縺ｮ蜀帝ｭ縺ｧnamespace YaneuraOu { 縺ｨ譖ｸ縺・※縺・ｋ縺ｮ縺ｧ繧ｳ繝｡繝ｳ繝医い繧ｦ繝医・

// 蟆・｣九〒縺ｯ縲ゝablebases縺ｯ逕ｨ縺・↑縺・・
namespace TB = Tablebases;

void syzygy_extend_pv(const OptionsMap& options,
	const Search::LimitsType& limits,
	Stockfish::Position& pos,
	Stockfish::Search::RootMove& rootMove,
	Value& v);

using namespace Search;
// 庁 蜀帝ｭ縺ｧ譖ｸ縺・◆縺ｮ縺ｧ繧ｳ繝｡繝ｳ繝医い繧ｦ繝医・
#endif

namespace {

// -----------------------
//   謗｢邏｢逕ｨ縺ｮ隧穂ｾ｡蛟､陬懈紛
// -----------------------

// 庁 縺ゅｋnode縺ｧ逕滓・縺励◆謖・＠謇九↓bonus繧剃ｸ弱∴繧九◆繧√↓縲√◎縺ｮnode縺ｧ逕滓・縺励◆謖・＠謇九ｒ濶ｯ縺・・↓菫晏ｭ倥＠縺ｦ縺翫￥驟榊・縺ｮcapacity縲・
constexpr int SEARCHEDLIST_CAPACITY = 32;
constexpr int mainHistoryDefault    = 0;
using SearchedList                  = ValueList<Move, SEARCHEDLIST_CAPACITY>;

// (*Scalers):
// The values with Scaler asterisks have proven non-linear scaling.
// They are optimized to time controls of 180 + 1.8 and longer,
// so changing them or adding conditions that are similar requires
// tests at these types of time controls.

// (*Scaler) All tuned parameters at time controls shorter than
// optimized for require verifications at longer time controls

// ・遺ｻ繧ｹ繧ｱ繝ｼ繝ｩ繝ｼ縺ｫ縺､縺・※・・
// 繧｢繧ｹ繧ｿ繝ｪ繧ｹ繧ｯ・・・我ｻ倥″縺ｮ繧ｹ繧ｱ繝ｼ繝ｩ繝ｼ蛟､縺ｯ縲・撼邱壼ｽ｢繧ｹ繧ｱ繝ｼ繝ｪ繝ｳ繧ｰ縺ｧ縺ゅｋ縺薙→縺檎｢ｺ隱阪＆繧後※縺・∪縺吶・
// 縺薙ｌ繧峨・蛟､縺ｯ縲・80・・.8・・80遘抵ｼ・.8遘貞刈邂暦ｼ我ｻ･荳翫・謖√■譎る俣縺ｧ譛驕ｩ蛹悶＆繧後※縺・∪縺吶・
// 縺励◆縺後▲縺ｦ縲√％繧後ｉ繧貞､画峩縺励◆繧翫∝酔讒倥・譚｡莉ｶ繧定ｿｽ蜉縺励◆繧翫☆繧句ｴ蜷医・縲・
// 縺薙・遞ｮ縺ｮ謖√■譎る俣縺ｧ縺ｮ繝・せ繝医′蠢・ｦ√〒縺吶・

// ・遺ｻ繧ｹ繧ｱ繝ｼ繝ｩ繝ｼ・画怙驕ｩ蛹悶＆繧後◆謖√■譎る俣繧医ｊ繧ら洒縺・凾髢楢ｨｭ螳壹〒隱ｿ謨ｴ縺輔ｌ縺・
// 縺吶∋縺ｦ縺ｮ繝代Λ繝｡繝ｼ繧ｿ縺ｫ縺､縺・※縺ｯ縲√ｈ繧企聞縺・戟縺｡譎る俣縺ｧ縺ｮ讀懆ｨｼ縺悟ｿ・ｦ√〒縺吶・

#if defined(P2_KHCORR) || defined(P2_KHGATE)
// 2026-11-12 KHCORR (DESIGN-20261112-OPTIMISM 案① / Fable 5.2) の鍵。 数え上げ部分 (kh_attacked_by / kh_zone / kh_hang) は KHGATE と共通。
//   定義の正本 = runs/sibling/sib_feats.py の zaE / hang (利きは玉の利きを含み、 遮蔽込み・x-ray なし)。
//   zaE[c]  = 色 c の玉の 8 近傍の**空升**への、 敵の利き数の和 (各空升の敵の利きの枚数を足す)。
//   hang[c] = 色 c の玉以外の駒のうち、 敵の利きがあり味方の利きが無いものの数。
//   鍵 (手番 us 視点) = ((min(zaE[us],3)*4 + min(zaE[them],3))*3 + min(hang[us],2))*3 + min(hang[them],2) ∈ [0,144)。
//   本ビルドは LONG_EFFECT_LIBRARY が無いので利きは都度計算する:
//     利きの和集合 = 歩は pawn_attacks_bb で一括 + 歩以外 (玉を含む) を effects_from で 1 枚ずつ (先後で約 2x20 回)。
//     zaE は「近傍の空升 かつ 敵の利きがある升」だけ attackers_to の popcount を取り、 cap に届いたら打ち切る。
namespace {

// 色 c の全駒 (玉を含む) の利きの和集合。 occ = 盤上の全駒 (遮蔽込み・x-ray なし)。
Bitboard kh_attacked_by(const Position& pos, Color c, const Bitboard& occ) {
    Bitboard a = c == BLACK ? pawn_attacks_bb<BLACK>(pos.pieces(BLACK, PAWN))
                            : pawn_attacks_bb<WHITE>(pos.pieces(WHITE, PAWN));
    Bitboard b = pos.pieces(c) ^ pos.pieces(c, PAWN);
    while (b)
    {
        const Square s = b.pop();
        a |= effects_from(pos.piece_on(s), s, occ);
    }
    return a;
}

// 色 kc の玉の 8 近傍の空升への、 敵の利き数の和 (zaE)。 cap 以上になったら cap を返す。
int kh_zone(const Position& pos, Color kc, const Bitboard& enemyAtk, const Bitboard& occ, int cap) {
    Bitboard z = occ.andnot(kingEffect(pos.square<KING>(kc)) & enemyAtk);
    int      n = 0;
    while (z && n < cap)
        n += pos.attackers_to(~kc, z.pop(), occ).pop_count();
    return std::min(n, cap);
}

// 色 c の浮き駒 (hang) の数。 atk[] = kh_attacked_by の先後 2 枚。
int kh_hang(const Position& pos, Color c, const Bitboard* atk) {
    const Bitboard mine = pos.pieces(c) ^ pos.pieces(c, KING);
    return atk[c].andnot(mine & atk[~c]).pop_count();
}

}  // namespace
#endif  // P2_KHCORR || P2_KHGATE

#if defined(P2_KHCORR)
namespace {

// KHCORR の鍵 (0..143)。 重み 0 のときは呼ばれない。
int kh_corr_key(const Position& pos) {
    const Color    us   = pos.side_to_move();
    const Color    them = ~us;
    const Bitboard occ  = pos.pieces();
    const Bitboard atk[COLOR_NB] = {kh_attacked_by(pos, BLACK, occ), kh_attacked_by(pos, WHITE, occ)};
    const int      zU   = kh_zone(pos, us, atk[them], occ, 3);
    const int      zT   = kh_zone(pos, them, atk[us], occ, 3);
    const int      hU   = std::min(kh_hang(pos, us, atk), 2);
    const int      hT   = std::min(kh_hang(pos, them, atk), 2);
    return ((zU * 4 + zT) * 3 + hU) * 3 + hT;
}

#if defined(P2_KHCORR_DEBUG)
// 検算用 (sib_feats.py との突き合わせ): root 局面の生の zaE / hang (先手, 後手 の順) と鍵を出す。
void kh_debug_print(const Position& pos) {
    const Bitboard occ = pos.pieces();
    const Bitboard atk[COLOR_NB] = {kh_attacked_by(pos, BLACK, occ), kh_attacked_by(pos, WHITE, occ)};
    sync_cout << "info string khkey zaE " << kh_zone(pos, BLACK, atk[WHITE], occ, 1000) << ' '
              << kh_zone(pos, WHITE, atk[BLACK], occ, 1000) << " hang " << kh_hang(pos, BLACK, atk) << ' '
              << kh_hang(pos, WHITE, atk) << " key " << kh_corr_key(pos) << sync_endl;
}
#endif

}  // namespace
#endif  // P2_KHCORR (kh_corr_key)

#if defined(P2_KHGATE)
// 2026-11-12 KHGATE (取り合い・玉の危険局面での楽観の補い / runs/prereg/VERDICT-OPTIMISM-P0-20261112.log)。
//   zaE / hang の定義は上の KHCORR と同じ (共通の kh_attacked_by / kh_zone / kh_hang を使う)。
//   PESS   = 静的評価に固定の悲観補正: -(PESS_HANG*min(hang_us,3) + PESS_ZA*min(zaE_us,3))
//                                     +(PESS_HANG_THEM*min(hang_them,3) + PESS_ZA_THEM*min(zaE_them,3))  [cp, 手番視点]
//   SELEXT = hang_us >= SELEXT_HANG_TH または zaE_us >= SELEXT_ZA_TH の node (= 荒れた node) で
//            LMR の r を SELEXT_LMR ply 減らす / SELEXT_NOPRUNE=1 なら Step 8 (futility) と Step 9 (NMP) を止める。
//   全 option 0 (既定) なら kh_gate_eval は呼ばれない (bit-exact)。
//   費用を抑える工夫:
//     - 手番側の量だけなら敵の利きの和集合 1 枚だけを作る (味方の和集合は作らない)。
//     - hang_us は「敵の利きがある自駒」だけ attackers_to(us) で守りを調べる (典型 0-3 升)。
//     - zaE は cap で打ち切る。 SELEXT だけなら zaE が閾値に届いた時点で hang を数えない。
//     - 相手側 (THEM) の量は THEM の option が非 0 のときだけ数える。
//     - search は 1 node 1 回 (Step 6 で計算して staticEval / 枝刈り / LMR で共有)。
namespace {

struct KhGate {
    int  pess  = 0;      // 静的評価から引く量 (cp)
    bool rough = false;  // SELEXT の対象 node か
};

inline bool kh_pess_on() {
    return (SPSA::PESS_HANG | SPSA::PESS_ZA | SPSA::PESS_HANG_THEM | SPSA::PESS_ZA_THEM) != 0;
}
inline bool kh_selext_on() { return (SPSA::SELEXT_LMR | SPSA::SELEXT_NOPRUNE) != 0; }

// 手番側の浮き駒数 (cap 打ち切り)。 enemyAtk = 敵の利きの和集合。 kh_hang(pos, us, atk) と同じ値 (cap まで)。
int kh_hang_us(const Position& pos, Color us, const Bitboard& enemyAtk, const Bitboard& occ, int cap) {
    Bitboard t = (pos.pieces(us) ^ pos.pieces(us, KING)) & enemyAtk;
    int      n = 0;
    while (t && n < cap)
        if (!pos.attackers_to(us, t.pop(), occ))
            ++n;
    return n;
}

KhGate kh_gate_eval(const Position& pos, bool wantPess, bool wantRough) {
    KhGate         g;
    const Color    us   = pos.side_to_move();
    const Color    them = ~us;
    const Bitboard occ  = pos.pieces();
    const Bitboard atkThem = kh_attacked_by(pos, them, occ);

    const int zaTh   = std::max(SPSA::SELEXT_ZA_TH, 1);
    const int hangTh = std::max(SPSA::SELEXT_HANG_TH, 1);
    const int zCap   = std::max(wantPess ? 3 : 0, wantRough ? zaTh : 0);
    const int zU     = kh_zone(pos, us, atkThem, occ, zCap);

    if (!wantPess)
    {
        g.rough = zU >= zaTh || kh_hang_us(pos, us, atkThem, occ, hangTh) >= hangTh;
        return g;
    }

    const int hCap = std::max(3, wantRough ? hangTh : 0);
    const int hU   = kh_hang_us(pos, us, atkThem, occ, hCap);
    g.rough        = wantRough && (zU >= zaTh || hU >= hangTh);
    g.pess         = SPSA::PESS_HANG * std::min(hU, 3) + SPSA::PESS_ZA * std::min(zU, 3);

    if (SPSA::PESS_HANG_THEM | SPSA::PESS_ZA_THEM)
    {
        const Bitboard atk[COLOR_NB] = {us == BLACK ? kh_attacked_by(pos, BLACK, occ) : atkThem,
                                        us == WHITE ? kh_attacked_by(pos, WHITE, occ) : atkThem};
        const int      zT = kh_zone(pos, them, atk[us], occ, 3);
        const int      hT = std::min(kh_hang(pos, them, atk), 3);
        g.pess -= SPSA::PESS_HANG_THEM * hT + SPSA::PESS_ZA_THEM * zT;
    }
    return g;
}

}  // namespace
#endif  // P2_KHGATE

// KHGATE のマクロ: PESS は search() の khGate / qsearch() の khPess を引く。
// KHGATE SELEXT_NOPRUNE: 荒れた node では Step 8 (futility) / Step 9 (NMP) を止める。 search() の Step 6 の khGate を参照する。
#if defined(P2_KHGATE)
#define P2_KHGATE_PRUNE_OK && !(khGate.rough && SPSA::SELEXT_NOPRUNE)
#define P2_KHGATE_QPESS(v) Value((v) - khPess)
#define P2_KHGATE_PESS(v) Value((v) - khGate.pess)
#else
#define P2_KHGATE_PESS(v) (v)
#define P2_KHGATE_PRUNE_OK
#define P2_KHGATE_QPESS(v) (v)
#endif

int correction_value(const YaneuraOuWorker& w, const Position& pos, const Stack* const ss) {
    const Color us    = pos.side_to_move();
    const auto  m     = (ss - 1)->currentMove;
    const auto& shared = w.sharedHistory;
    const int   pcv    = shared.pawn_correction_entry(pos).at(us).pawn;
    const int   micv   = shared.minor_piece_correction_entry(pos).at(us).minor;
    const int   wnpcv  = shared.nonpawn_correction_entry<WHITE>(pos).at(us).nonPawnWhite;
    const int   bnpcv  = shared.nonpawn_correction_entry<BLACK>(pos).at(us).nonPawnBlack;
    // 蟆・｣狗畑: 謖√■鬧呈ｧ区・ correction(CORR_HAND_WEIGHT=0 縺ｮ譌｢螳壹↑繧牙ｯ・ｸ・=bit-exact)縲・
    // ★NPS: CORR_HAND_WEIGHT は既定 0 の実行時グローバルなので、 コンパイラはこのロードを消せない。
    //   correctionHistory は 65536 x スレッド数 x 20B あり、 これは毎ノード 5 本目の乱数索引だった。
    //   書き込み側 (update_correction_history) は既に if で守られている。 0 倍なので bit-exact。
    const int   hcv    = SPSA::CORR_HAND_WEIGHT ? shared.hand_correction_entry(pos).at(us).hand : 0;
    const auto  cntcv =
      m.is_ok() ? (*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()]
                + (*(ss - 4)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()]
                 : 8;

#if defined(P2_KHCORR)
    // KHCORR: 重み 0 (既定) なら鍵の計算も表の読みも走らせない (bit-exact)。
    const int   khcv   = SPSA::CORR_KH_WEIGHT ? shared.kh_correction_entry(kh_corr_key(pos)).at(us).kh : 0;
    return SPSA::CORR_PCV_WEIGHT * pcv + SPSA::CORR_MICV_WEIGHT * micv + SPSA::CORR_NONPAWN_WEIGHT * (wnpcv + bnpcv) + SPSA::CORR_CNT_WEIGHT * cntcv + SPSA::CORR_HAND_WEIGHT * hcv + SPSA::CORR_KH_WEIGHT * khcv;
#else
    return SPSA::CORR_PCV_WEIGHT * pcv + SPSA::CORR_MICV_WEIGHT * micv + SPSA::CORR_NONPAWN_WEIGHT * (wnpcv + bnpcv) + SPSA::CORR_CNT_WEIGHT * cntcv + SPSA::CORR_HAND_WEIGHT * hcv;
#endif
}

// Add correctionHistory value to raw staticEval and guarantee evaluation
// does not hit the tablebase range.
// correctionHistory 縺ｮ蛟､繧・raw staticEval 縺ｫ蜉縺医・
// 隧穂ｾ｡蛟､縺後ユ繝ｼ繝悶Ν繝吶・繧ｹ縺ｮ遽・峇縺ｫ蜈･繧峨↑縺・％縺ｨ繧剃ｿ晁ｨｼ縺吶ｋ縲・
Value to_corrected_static_eval(const Value v, const int cv) {
	return std::clamp(v + cv / 131072, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
}

void update_correction_history(const Position&          pos,
                               Stack* const             ss,
                               Search::YaneuraOuWorker& workerThread,
                               const int                bonus) {
    const Move  m  = (ss - 1)->currentMove;
    const Color us = pos.side_to_move();

    const int nonPawnWeight = SPSA::CORR_HIST_NONPAWN;

    auto&         shared        = workerThread.sharedHistory;

    shared.pawn_correction_entry(pos).at(us).pawn << bonus;
    shared.minor_piece_correction_entry(pos).at(us).minor << bonus * SPSA::CORR_HIST_MINOR / 128;
    shared.nonpawn_correction_entry<WHITE>(pos).at(us).nonPawnWhite << bonus * nonPawnWeight / 128;
    shared.nonpawn_correction_entry<BLACK>(pos).at(us).nonPawnBlack << bonus * nonPawnWeight / 128;
    // 蟆・｣狗畑: 謖√■鬧呈ｧ区・ correction 縺ｮ譖ｴ譁ｰ(CORR_HAND_WEIGHT>0=ON 縺ｮ縺ｨ縺阪・縺ｿ譖ｸ縺・縲・
    if (SPSA::CORR_HAND_WEIGHT)
        shared.hand_correction_entry(pos).at(us).hand << bonus * SPSA::CORR_HIST_HAND / 128;
#if defined(P2_KHCORR)
    // KHCORR: 読み側と同じく重み 0 なら書かない (鍵の計算も走らせない)。
    if (SPSA::CORR_KH_WEIGHT)
        shared.kh_correction_entry(kh_corr_key(pos)).at(us).kh << bonus * SPSA::CORR_HIST_KH / 128;
#endif

    if (m.is_ok())
    {
        const Square to = m.to_sq();
        const Piece  pc = pos.piece_on(to);
        (*(ss - 2)->continuationCorrectionHistory)[pc][to] << bonus * SPSA::CORR_HIST_CONT_SS2 / 128;
        (*(ss - 4)->continuationCorrectionHistory)[pc][to] << bonus * SPSA::CORR_HIST_CONT_SS4 / 128;
    }
}


// Add a small random component to draw evaluations to avoid 3-fold blindness
// 3蝗槫酔荳螻髱｢縺ｫ縺ｪ繧狗峇轤ｹ・・-fold blindness・峨ｒ蝗樣∩縺吶ｋ縺溘ａ縲∬ｩ穂ｾ｡繧貞ｼ輔″蛻・￠譁ｹ蜷代↓隱伜ｰ弱☆繧句ｰ上＆縺ｪ繝ｩ繝ｳ繝繝謌仙・繧定ｿｽ蜉縺吶ｋ

// 庁 蠑輔″蛻・￠譎ゅ・隧穂ｾ｡蛟､VALUE_DRAW(0)縺ｮ莉｣繧上ｊ縺ｫﾂｱ1縺ｮ荵ｱ謨ｰ縺ｿ縺溘＞縺ｪ縺ｮ繧剃ｸ弱∴繧九・
//     nodes : 迴ｾ蝨ｨ縺ｮ謗｢邏｢node謨ｰ(荵ｱ謨ｰ縺ｮseed莉｣繧上ｊ縺ｫ逕ｨ縺・ｋ)

// 統 繝√ぉ繧ｹ縺ｧ縺ｯ縲∝ｼ輔″蛻・￠縺・.5蜍晄桶縺・↑縺ｮ縺ｧ蠑輔″蛻・￠蝗樣∩縺ｮ縺溘ａ縺ｮ蟾･螟ｫ縺後＠縺ｦ縺ゅ▲縺ｦ縲・
//     莉･荳九・繧医≧縺ｫvalue_draw縺ｫ謠ｺ繧峨℃繧貞刈邂励☆繧九％縺ｨ縺ｫ繧医▲縺ｦ謗｢邏｢繧貞崋螳壼喧縺励↑縺・蜷後§node繧・
//     謗｢邏｢縺励▽縺･縺代※蜊・律謇九↓縺励※縺励∪縺・・繧貞屓驕ｿ)蟾･螟ｫ縺後≠繧九・
//     蟆・｣九・蝣ｴ蜷医∵勸騾壹・蜊・律謇九→騾｣邯夂視謇九・蜊・律謇九→蜉｣遲牙ｱ髱｢縺ｫ繧医ｋ蜊・律謇・蠕ｪ迺ｰ・・縺ｨ縺九≠繧九・縺ｧ縺薙ｌ蟆主・縺吶ｋ縺ｮ縺｡繧・▲縺ｨ雖後・
//     竍ｨ  TODO : 繧ゅ≧縺｡繧・▲縺ｨ縺ｩ縺・↓縺九☆繧九・
// ---------------------------------------------------------------------------
// 2026-09-08 診断計測 (P2_SEARCH_STATS=1 のときだけ info string で出す)。
//   実戦条件 go nodes 400000 で NAGISA と 2-3 倍の探索効率差、 seldepth-depth 11.0 対 8.0。
//   qsearch の過剰展開か extension の過剰発動かを切り分けるための一時計測。
//   ホットパスへの追加は relaxed な atomic インクリメント 1 命令のみ。
// ---------------------------------------------------------------------------
namespace P2Stat {
std::atomic<u64> main_nodes{0}, qs_nodes{0};
std::atomic<u64> se_g1{0}, se_g2{0}, se_g3{0}, se_g4{0}, se_g5{0}, se_g6{0};
std::atomic<u64> tt_probe{0}, tt_hit{0}, deep_node{0}, mv_loop{0};
std::atomic<u64> ext_p1{0}, ext_p2{0}, ext_p3{0}, ext_m2{0}, ext_m3{0}, se_try{0};
// ★NPS: 関数内 static は呼ぶたびガード変数の acquire ロードと分岐が要る。
//   探索の hot path で 9 箇所踏むので、 計測ビルド (-DP2_SEARCH_STATS) 以外では
//   constexpr false にして呼び出しごと消す。
#if defined(P2_SEARCH_STATS)
inline bool enabled() {
    static const bool v = std::getenv("P2_SEARCH_STATS") != nullptr;
    return v;
}
#else
constexpr bool enabled() { return false; }
#endif
inline void reset() {
    se_g1=0; se_g2=0; se_g3=0; se_g4=0; se_g5=0; se_g6=0; tt_probe=0; tt_hit=0; deep_node=0; mv_loop=0;

    main_nodes = 0; qs_nodes = 0;
    ext_p1 = 0; ext_p2 = 0; ext_p3 = 0; ext_m2 = 0; ext_m3 = 0; se_try = 0;
}
inline void dump() {
    sync_cout << "info string P2FUNNEL g1=" << se_g1.load() << " g2=" << se_g2.load()
              << " g3=" << se_g3.load() << " g4=" << se_g4.load() << " g5=" << se_g5.load()
              << " g6=" << se_g6.load() << " ttprobe=" << tt_probe.load()
              << " tthit=" << tt_hit.load() << " ttmove=" << deep_node.load() << " mvloop=" << mv_loop.load() << sync_endl;

    u64 m = main_nodes.load(), q = qs_nodes.load();
    u64 tot = m + q;
    sync_cout << "info string P2STAT main=" << m << " qs=" << q
              << " qs_ratio=" << (tot ? (100.0 * double(q) / double(tot)) : 0.0)
              << " se_try=" << se_try.load()
              << " ext+1=" << ext_p1.load() << " ext+2=" << ext_p2.load()
              << " ext+3=" << ext_p3.load()
              << " ext-2=" << ext_m2.load() << " ext-3=" << ext_m3.load() << sync_endl;
}
}  // namespace P2Stat

// P2Volm (局面の「揺れ」V) は p2volm.h へ移した (2026-09-28、 thread.cpp からも使うため)

#ifdef P2_PRUNE_STATS
// 2026-09-17 計測専用ビルド (-DP2_PRUNE_STATS): 枝刈り・打ち切りごとに「発火回数」と、 env P2_PRUNE_VERIFY=N (N>0) のとき
// 1/N を読み直した「見落とし」を数える。 本番ビルドには入らない。 Threads=1 前提 (読み直し中の同 ply ガードはグローバル)。
//   ノード単位 (razor/rfp/nmp/probcut/smallpc): 同じノードを枝刈り無しで search し直し、 判定 (>=beta / <=alpha) が外れたら見落とし。
//   手単位 (capfut/capsee/conthist/parentfut/quietsee): その手を newDepth・null window で読み、 alpha を超えたら見落とし。
namespace P2PruneStat {
constexpr int NK = 10;
inline const char* NAME[NK] = {"razor", "rfp", "nmp", "probcut", "smallpc", "capfut", "capsee", "conthist", "parentfut", "quietsee"};
inline std::atomic<u64> fired[NK], vn[NK], vm[NK];
// 2026-09-28 VOLM F1: RFP の見落とし率が V (玉 8 近傍の空升への敵利き数の和) で単調に上がるか。
//   V の定義は runs/rescore/empty_split.py:47-66 と同じ: 両玉について 8 近傍のうち**駒が乗っていない升**
//   だけを見て、 その玉の**敵**の利き数 (board_effect = step+ray、 do_move 差分維持) を足す。
//   ★23 により**否定の関門としてだけ**使う (単調に上がらなければ VOLM 中止)。
constexpr int NV = 5;   // 0 / 1-2 / 3-4 / 5-6 / >=7
inline std::atomic<u64> firedV[NK][NV], vnV[NK][NV], vmV[NK][NV];
// 2026-09-28 VOLM ノード節約側: 読み直しで消費したノード数 = 「刈って浮かせた分」を V bin 別に積む。
//   🚨NPS -> Elo は校正済み (2 倍 = +70 Elo、 対数則) なので **ここだけは計算で期待値が出る**。
inline std::atomic<u64> savedV[NK][NV];
inline int cur_vbin = -1;
// 🎯🎯2026-10-20 **bucket 跨ぎの層別** (ユーザー指示「候補 A の関門やって」)。
//   何を測るか: 親子で LayerStack の bucket が違うノード (= 評価関数が切り替わる境界) で
//   **刈りの誤り率 (vm/vn)** が非跨ぎより高いか。
//   🎯これは「手が変わる率」ではなく**刈りの誤りそのもの**なので**向きが見える**
//     (★capfut_opening_is_a_lottery: 変化率はどんな摂動でも 3 割なので情報を持たない)。
//   🚨**既定 0 = 従来の V 層別**なので既存の測定と互換。 環境変数で切り替える。
//   🚨NV=5 の配列をそのまま使い **bin 0 (非跨ぎ) / 1 (跨ぎ)** だけ埋める (ABI を触らない)。
inline int ply_bucket[MAX_PLY + 10];   // 計測専用。 static eval を取った ply の bucket
inline int vbin_mode() {
    static const int v = [] { const char* e = std::getenv("P2_PRUNE_VBIN_MODE"); return e ? std::atoi(e) : 0; }();
    return v;
}
// 跨ぎなら 1 / 非跨ぎなら 0。 どちらか不明 (-1) なら **0 (非跨ぎ扱い)**。
//   🚨王手ノードは evaluate を呼ばないので bucket が未計算になる。 そこを跨ぎに数えると
//     「王手が多い局面ほど跨ぎ」という別の量を測ってしまうので、 非跨ぎに寄せる。
inline int xbin(const Position& pos, int ply) {
    const int cur = Eval::P2NNUE::progress_bucket_if_computed(pos);
    if (cur < 0 || ply <= 0)
        return 0;
    const int par = ply_bucket[ply - 1];
    if (par < 0)
        return 0;
    return (cur != par) ? 1 : 0;
}
inline int vbin(int v) { return v <= 0 ? 0 : v <= 2 ? 1 : v <= 4 ? 2 : v <= 6 ? 3 : 4; }
inline int volm_v(const Position& pos) { return YaneuraOu::P2Volm::value(pos); }   // 本番実装を使う
// 🚨2026-10-20 位置を vbin/volm_v の後ろに移した (前は未宣言でコンパイルエラー)
// 🚨🚨2026-10-20 **交絡の対照** (P2_PRUNE_VBIN_MODE=2)。
//   bucket の跨ぎは「駒が減った」= 取り合いと相関する。 取り合いのノードは
//   そもそも誤刈り率が高いので、 跨ぎの x1.9 が **「跨ぎ」ではなく「取った直後」**を
//   測っているだけという交絡がありうる (★control_must_use_production_path)。
//   mode 2 は層別の鍵を**「直前の手が駒を取った」**に差し替える。
//   -> mode 1 と mode 2 で同じ比が出たら、 bucket 跨ぎは独自の情報を持たない。
inline int capbin(const Position& pos) {
    const StateInfo* st = pos.state();
    return (st != nullptr && st->capturedPiece != NO_PIECE) ? 1 : 0;
}
// 🚨🚨2026-10-20 **第 2 の交絡: 玉の移動**。
//   `progress_full()` は `g_progress[sq_k * FE_OLD_END]` を駒リストで引くので、
//   **玉が動くと表ごと切り替わる** = 取っていなくても bucket が変わりやすい。
//   玉の移動直後のノードは玉の安全度の都合で元から誤刈り率が違いうるので、
//   mode 5 (玉が動いた直後) と mode 6 (跨ぎ かつ 取っていない かつ 玉も動いていない) で潰す。
inline int kingmove_prev(const Position& pos, Move m) {
    if (!m.is_ok() || m == Move::null() || m.is_drop())
        return 0;
    return (type_of(pos.piece_on(m.to_sq())) == KING) ? 1 : 0;
}
inline int cur_bin(const Position& pos, int ply, Move prev) {
    switch (vbin_mode())
    {
    case 1:  return xbin(pos, ply);
    case 2:  return capbin(pos);                 // 対照: 取った直後か
    case 3:  return xbin(pos, ply) & capbin(pos);  // 跨ぎかつ取った直後
    case 4:  return (xbin(pos, ply) && !capbin(pos)) ? 1 : 0;  // 跨ぎだが取っていない
    case 5:  return kingmove_prev(pos, prev);    // 対照: 玉が動いた直後か
    // 🎯純粋な bucket 跨ぎ (取ってもいない / 玉も動いていない)
    case 6:  return (xbin(pos, ply) && !capbin(pos) && !kingmove_prev(pos, prev)) ? 1 : 0;
    // 🎯🎯**最後の対照 (3 値)**: 「境界を**越えた**」のか「境界**付近**の局面」なのか。
    //   0 = 同 bucket かつ閾値から遠い / 1 = **同 bucket だが閾値の近く** / 2 = 跨ぎ
    //   bin1 が bin0 と同じなら「越えたこと」が原因 = 機構になる。
    //   bin1 も上がっているなら「付近の局面がそういう局面」= 進行度の話で既知のゼロ。
    case 7:
    {
        if (xbin(pos, ply))
            return 2;
        // 🚨`near` は Windows の互換マクロなので識別子に使えない
        const int nearTh = Eval::P2NNUE::progress_near_threshold_if_computed(pos);
        return (nearTh == 1) ? 1 : 0;
    }
    // 🚨🚨**第 3 の交絡: bucket そのもの**。 閾値の間隔は 0.85/0.59/0.51/0.51/0.59/0.85 なので
    //   同じ delta でも**中央の bucket ほど「近く」に入りやすい** = 「近く」は中盤に偏る。
    //   中盤のノードは元から誤刈り率が高いので、 bucket を揃えて比べ直す必要がある。
    //   mode 8 = bucket/2 (0..3) の基礎率 / mode 9 = **bucket を固定**して 遠い/近く/跨ぎ。
    case 8:
    {
        const int b = Eval::P2NNUE::progress_bucket_if_computed(pos);
        return (b < 0) ? -1 : (b >> 1);        // 0..3
    }
    case 9:
    {
        const int b = Eval::P2NNUE::progress_bucket_if_computed(pos);
        static const int only = [] {
            const char* e = std::getenv("P2_PRUNE_BUCKET_ONLY");
            return e ? std::atoi(e) : 3;
        }();
        if (b < 0 || b != only)
            return -1;                          // この bucket 以外は数えない
        if (xbin(pos, ply))
            return 2;
        return (Eval::P2NNUE::progress_near_threshold_if_computed(pos) == 1) ? 1 : 0;
    }
    default: return vbin(volm_v(pos));
    }
}
inline u64  counter = 0;
inline int  noprune_ply = -1;
inline int rate() {
    static const int v = [] { const char* e = std::getenv("P2_PRUNE_VERIFY"); return e ? std::atoi(e) : 0; }();
    return v;
}
inline bool blocked(int ply) { return ply == noprune_ply; }
inline bool want(int k) {
    fired[k].fetch_add(1, std::memory_order_relaxed);
    if (cur_vbin >= 0) firedV[k][cur_vbin].fetch_add(1, std::memory_order_relaxed);
    return rate() > 0 && (counter++ % u64(rate())) == 0;
}
inline void verify(int k, bool miss) {
    vn[k].fetch_add(1, std::memory_order_relaxed);
    if (miss) vm[k].fetch_add(1, std::memory_order_relaxed);
    if (cur_vbin >= 0)
    {
        vnV[k][cur_vbin].fetch_add(1, std::memory_order_relaxed);
        if (miss) vmV[k][cur_vbin].fetch_add(1, std::memory_order_relaxed);
    }
}
inline void reset() { for (int i = 0; i < NK; ++i) { fired[i] = 0; vn[i] = 0; vm[i] = 0; }
                      for (int i = 0; i <= MAX_PLY + 9; ++i) ply_bucket[i] = -1;
                      for (int j = 0; j < NK; ++j) for (int i = 0; i < NV; ++i) { firedV[j][i] = 0; vnV[j][i] = 0; vmV[j][i] = 0; savedV[j][i] = 0; } }
inline void dump() {
    for (int i = 0; i < NK; ++i)
        sync_cout << "info string P2PRUNE " << NAME[i] << " fired=" << fired[i].load() << " vn=" << vn[i].load()
                  << " vm=" << vm[i].load() << sync_endl;
    for (int j = 0; j < NK; ++j)
        for (int i = 0; i < NV; ++i)
            sync_cout << "info string P2VOLM " << NAME[j] << " vbin=" << i << " fired=" << firedV[j][i].load()
                      << " vn=" << vnV[j][i].load() << " vm=" << vmV[j][i].load()
                      << " saved=" << savedV[j][i].load() << sync_endl;
    // 🎯跨ぎの層別で測ったときは、 bin 0 = 非跨ぎ / bin 1 = 跨ぎ と明示して出す
    if (vbin_mode())
        for (int j = 0; j < NK; ++j)
        {
            const u64 n0 = vnV[j][0].load(), m0 = vmV[j][0].load();
            const u64 n1 = vnV[j][1].load(), m1 = vmV[j][1].load();
            sync_cout << "info string P2XBIN " << NAME[j]
                      << " same_vn=" << n0 << " same_vm=" << m0
                      << " cross_vn=" << n1 << " cross_vm=" << m1
                      << " same_rate=" << (n0 ? double(m0) / double(n0) : -1.0)
                      << " cross_rate=" << (n1 ? double(m1) / double(n1) : -1.0)
                      << " fired_same=" << firedV[j][0].load()
                      << " fired_cross=" << firedV[j][1].load() << sync_endl;
        }
}
}  // namespace P2PruneStat
#define P2_PRUNE_NOT_BLOCKED && !P2PruneStat::blocked(ss->ply)
// ノード単位: 同じノードを枝刈り無しで読み直す (NonPV なので null window のまま)
#define P2_PRUNE_NODE_VERIFY(k, missExpr) \
    P2PruneStat::cur_vbin = P2PruneStat::cur_bin(pos, ss->ply, (ss - 1)->currentMove); \
    if (P2PruneStat::want(k) && !PvNode) \
    { \
        const int p2old_         = P2PruneStat::noprune_ply; \
        const u64 p2n0_          = threads.nodes_searched(); \
        const int p2vb_          = P2PruneStat::cur_vbin;   /* 🚨再帰で上書きされるので退避 */ \
        P2PruneStat::noprune_ply = ss->ply; \
        const Value v_           = search<NonPV>(pos, ss, alpha, beta, depth, cutNode); \
        P2PruneStat::noprune_ply = p2old_; \
        P2PruneStat::cur_vbin    = p2vb_; \
        if (p2vb_ >= 0) \
            P2PruneStat::savedV[k][p2vb_].fetch_add(threads.nodes_searched() - p2n0_, std::memory_order_relaxed); \
        P2PruneStat::verify(k, missExpr); \
    }
// 手単位: 枝刈りした手を newDepth で読んで alpha を超えるか
#define P2_PRUNE_MOVE_VERIFY(k) \
    P2PruneStat::cur_vbin = P2PruneStat::cur_bin(pos, ss->ply, (ss - 1)->currentMove); \
    if (P2PruneStat::want(k)) \
    { \
        const u64 p2n0_ = threads.nodes_searched(); \
        const int p2vb_ = P2PruneStat::cur_vbin;   /* 🚨再帰で上書きされるので退避 */ \
        do_move(pos, move, st, givesCheck, ss); \
        const Value v_ = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode); \
        undo_move(pos, move); \
        P2PruneStat::cur_vbin = p2vb_; \
        if (p2vb_ >= 0) \
            P2PruneStat::savedV[k][p2vb_].fetch_add(threads.nodes_searched() - p2n0_, std::memory_order_relaxed); \
        P2PruneStat::verify(k, v_ > alpha); \
    }
#else
#define P2_PRUNE_NOT_BLOCKED
#define P2_PRUNE_NODE_VERIFY(k, missExpr)
#define P2_PRUNE_MOVE_VERIFY(k)
#endif

#ifdef P2_LMR_STATS
// 2026-09-17 計測専用ビルド (-DP2_LMR_STATS): LMR の条件別に「削った / 浅い探索が alpha 超え / 深さを戻しても alpha 超え /
// (標本) 浅い探索で alpha 以下だったが深さを戻すと alpha 超え = 見落とし」を数える。 本番ビルドには入らない。
// env P2_LMR_VERIFY=N (N>0) で、 浅い探索が fail low した手の 1/N を newDepth で読み直す (探索木が変わるので別 run で使う)。
namespace P2LmrStat {
constexpr int NC = 7;  // 0 無条件 / 1 cutNode(ttMove有) / 2 cutNode(ttMove無) / 3 ttCapture / 4 cutoffCnt==2 / 5 cutoffCnt>2 / 6 allNode&cutoffCnt>1
inline const char* NAME[NC] = {"none", "cutnode_tt", "cutnode_nott", "ttcapture", "cutoff2", "cutoff3p", "allnode_co"};
inline std::atomic<u64> n[NC], red[NC], fh[NC], rs[NC], cf[NC], vn[NC], vm[NC], rsum[NC];
inline std::atomic<u64> sample{0};
inline int verify_rate() {
    static const int v = [] { const char* e = std::getenv("P2_LMR_VERIFY"); return e ? std::atoi(e) : 0; }();
    return v;
}
inline void reset() {
    for (int i = 0; i < NC; ++i) { n[i] = 0; red[i] = 0; fh[i] = 0; rs[i] = 0; cf[i] = 0; vn[i] = 0; vm[i] = 0; rsum[i] = 0; }
}
inline void dump() {
    for (int i = 0; i < NC; ++i)
        sync_cout << "info string P2LMR " << NAME[i] << " n=" << n[i].load() << " red=" << red[i].load() << " r=" << rsum[i].load()
                  << " fh=" << fh[i].load() << " rs=" << rs[i].load() << " cf=" << cf[i].load()
                  << " vn=" << vn[i].load() << " vm=" << vm[i].load() << sync_endl;
}
}  // namespace P2LmrStat
#endif

Value value_draw(size_t nodes) { return VALUE_DRAW - 1 + Value(nodes & 0x2); }

Value value_to_tt(Value v, int ply);
Value value_from_tt(Value v, int ply /*, int r50c */);
void  update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus);
void  update_quiet_histories(const Position&                pos,
                             Stack*                         ss,
                             Search::YaneuraOuWorker& workerThread,
                             Move                           move,
                             int                            bonus);

// 統 32縺ｯ縲〈uietsSearched縲〈uietsSearched縺ｮ譛螟ｧ謨ｰ縲ゅ◎縺ｮnode縺ｧ逕滓・縺励◆QUIETS/CAPTURES縺ｮ謖・＠謇九ｒ濶ｯ縺・・↓菫晄戟縺励※縺ゅｋ縲・
//     bonus繧貞刈轤ｹ縺吶ｋ縺ｨ縺阪↓縺薙ｌ繧峨・謖・＠謇九↓蟇ｾ縺励※陦後≧縲・
//     Stockfish 16縺ｧ縺ｯquietsSearched縺ｮ驟榊・繧ｵ繧､繧ｺ縺啓64]縺九ｉ[32]縺ｫ縺ｪ縺｣縺溘・
//     蟆・｣九〒縺ｯ繝上ぜ繝ｬ縺ｮquiet縺ｮ謖・＠謇九′螟ｧ驥上↓縺ゅｋ縺ｮ縺ｧ縲√◎繧後′繝吶せ繝医→縺ｯ髯舌ｉ縺ｪ縺・・
// 投 豈碑ｼ・＠縺溘→縺薙ｍ縲・4繧医ｊ32縺ｮ譁ｹ縺後ｏ縺壹°縺ｫ濶ｯ縺九▲縺溘・縺ｧ縲√→繧翫≠縺医★32縺ｫ縺励※縺翫￥縲・V7.73m縺ｨV7.73m2縺ｨ縺ｮ豈碑ｼ・

void update_all_stats(const Position&          pos,
                      Stack*                   ss,
                      Search::YaneuraOuWorker& workerThread,
                      Move                     bestMove,
                      Square                   prevSq,
                      SearchedList&            quietsSearched,
                      SearchedList&            capturesSearched,
                      Depth                    depth,
                      Move                     ttMove,
                      uint32_t                 threatCtx,
                      bool                     threatValid,
                      bool                     betaCutoff,
                      bool                     pvNode);


bool is_shuffling(Move move, Stack* const ss, const Position& pos) {
#if STOCKFISH
    if (pos.capture_stage(move) || pos.rule50_count() < 11)
        return false;
    if (pos.state()->pliesFromNull <= 6 || ss->ply < 18)
        return false;
    return move.from_sq() == (ss - 2)->currentMove.to_sq()
        && (ss - 2)->currentMove.from_sq() == (ss - 4)->currentMove.to_sq();
#else
    // 蟆・｣九↓縺ｯ50謇九Ν繝ｼ繝ｫ縺後↑縺上・ｧ呈遠縺｡縺ｯ縲悟ｾ蠕ｩ謇九阪〒縺ｯ縺ｪ縺・・縺ｧ髯､螟悶☆繧九・
    if (pos.capture_stage(move) || move.is_drop())
        return false;

    if (pos.state()->pliesFromNull <= 6 || ss->ply < 18)
        return false;

    const Move move2 = (ss - 2)->currentMove;
    const Move move4 = (ss - 4)->currentMove;

    if (!move2.is_ok() || !move4.is_ok() || move2.is_drop() || move4.is_drop())
        return false;

    return move.from_sq() == move2.to_sq() && move2.from_sq() == move4.to_sq();
#endif
}


}  // namespace

// 庁 繧・・縺・ｉ邇九〒縺ｯ縲仝orker繧呈ｴｾ逕溘＆縺帙※譖ｸ縺上％縺ｨ縺ｫ縺励◆縺ｮ縺ｧ縲√％縺ｮ繧ｳ繝ｼ繝峨・縲∵ｴｾ逕歡lass縺ｧ縺ゅｋYaneuraOuWorker縺ｮ繧ｳ繝ｳ繧ｹ繝医Λ繧ｯ繧ｿ縺ｧ譖ｸ縺上・
#if STOCKFISH
Search::Worker::Worker(SharedState& sharedState,
	std::unique_ptr<ISearchManager> sm,
	size_t                          threadId,
	NumaReplicatedAccessToken       token) :
	// Unpack the SharedState struct into member variables
	threadIdx(threadId),
	numaAccessToken(token),
	manager(std::move(sm)),
	options(sharedState.options),
	threads(sharedState.threads),
	tt(sharedState.tt),
	networks(sharedState.networks),
	refreshTable(networks[token]) {
	clear();
}
#endif

Search::YaneuraOuWorker::YaneuraOuWorker(SharedState& sharedState,
#if STOCKFISH
                                         std::unique_ptr<ISearchManager>,
                                         size_t                    threadIdx,
                                         size_t                    numaThreadIdx,
                                         size_t                    numaTotal,
                                         NumaReplicatedAccessToken token,
#else
                                         const ThreadIds& ids,
#endif
										 YaneuraOuEngine&          engine) :
	// Unpack the SharedState struct into member variables
    sharedHistory(sharedState.sharedHistories.at(ids.numaAccessToken.get_numa_index())),
	// 谿九ｊ縺ｯ蝓ｺ蠎苗lass縺ｧ縺ゅｋSearch::Worker縺ｮ縺ｻ縺・〒unpack縺吶ｋ縲・
	Search::Worker(sharedState, ids),
	engine(engine),
	manager(engine.manager)
{

    //clear();

	// ､・ThreadPool::resize_thread()竊探hreadPool::set()縺ｧThreadPool::clear()縺悟他縺ｳ蜃ｺ縺輔ｌ縺ｦ縲・
	//     縺昴・縺ｪ縺九〒Worker::clear()縺悟他縺ｳ蜃ｺ縺輔ｌ繧九°繧峨√％縺薙〒蜻ｼ縺ｳ蜃ｺ縺吝ｿ・ｦ√・縺ｪ縺・→諤昴≧縲・
}

void Search::YaneuraOuWorker::ensure_network_replicated() {
	// TODO : 縺ゅ→縺ｧ隧穂ｾ｡髢｢謨ｰ繧貞ｮ溯｣・☆繧九・
	#if 0
    // Access once to force lazy initialization.
    // We do this because we want to avoid initialization during search.
    (void) (networks[numaAccessToken]);
	#endif
}

void Search::YaneuraOuWorker::pre_start_searching() {

	if (is_mainthread())
        // 決 Stockfish縺ｧthread.cpp縺ｫ縺ゅ▲縺溷・譛溷喧縺ｮ荳驛ｨ縺ｯSearchManager::pre_start_searching()縺ｫ遘ｻ蜍輔＆縺帙◆縲・
        main_manager()->pre_start_searching(*this);

    // 統 Stockfish縺ｧ縺ｯThreadPool::start_thinking()縺ｧ陦後▲縺ｦ縺・ｋ縺後・
    //     繧・・縺・ｉ邇九〒縺ｯ縲∵ｴｾ逕歡lass縺ｮpre_start_thinking()莉･髯阪〒陦後≧縲・
#if STOCKFISH || defined(EVAL_P2NNUE)
    nmpMinPly       = 0;
#endif
    bestMoveChanges = 0;
    rootDepth = completedDepth = 0;

    // 蜷Цorker縺ｮPosition::set_ekr()繧貞他縺ｳ蜃ｺ縺励※蜈･邇峨Ν繝ｼ繝ｫ繧貞渚譏縺輔○繧句ｿ・ｦ√′縺ゅｋ縲・
    auto& search_options = main_manager()->search_options;
    rootPos.set_ekr(search_options.enteringKingRule);

	// 決 蜈･邇牙ｮ｣險縺後〒縺阪ｋ縺ｪ繧詠ootMoves縺ｫ霑ｽ蜉縺吶ｋ縲・
	//    縺薙ｌ縺ｯ縲［ain thread縺ｧ縺縺題｡後∴縺ｰ濶ｯ縺・・main thread縺ｫ螻槭☆繧妓orker縺ｮrootMoves縺ｫ縺輔∴霑ｽ蜉縺輔ｌ縺ｦ縺・ｌ縺ｰ濶ｯ縺・・縺ｧ)
	if (is_mainthread())
    {
        // 決  螳｣險蜍昴■縺ｧ縺阪ｋ縺ｪ繧峨〉ootMoves縺ｫ霑ｽ蜉縺吶ｋ縲・
		// 笞  竊代・set_erk()繧偵＠縺溘≠縺ｨ縺ｧ縺ｪ縺・→Position::DeclarationWin()縺梧ｭ｣蟶ｸ縺ｫ讖溯・縺励↑縺・・
        auto bestMove = rootPos.DeclarationWin();
        if (bestMove != Move::none())
        {
            // ､・searchmoves縺梧欠螳壹＆繧後※縺・※
            //     縺昴％縺ｫ螳｣險蜍昴■縺後↑縺・凾縺ｫ螳｣險蜍昴■縺ｯ縺ｧ縺阪ｋ縺ｮ縺銀ｦ・・
            //     縺ｧ縺阪↑縺・→荳堺ｾｿ縺ｪ豌励・蟆代＠縺吶ｋ縺九ｉ縲・
            //     螳｣險蜍昴■縺ｧ縺阪ｋ縺ｪ繧峨・縺､縺ｭ縺ｫMove::win()繧定ｿｽ蜉縺励※縺翫￥縲・

            auto it_move = std::find(rootMoves.begin(), rootMoves.end(), bestMove);
            if (it_move == rootMoves.end()) // 霑ｽ蜉縺輔ｌ縺ｦ縺・↑縺・
                rootMoves.emplace_back(Move::win());
        }
    }
}

#if defined(EVAL_P2NNUE)
// ============================================================================
//   非同期 df-pn 詰みsolver スレッド
//   (rust: crates/p2-search/src/mate_solver_thread.rs の C++ 移植)
// ============================================================================
//
// ★設計方針: main search に完全非干渉。
//   - root 局面の sfen から「自前の Position + StateInfo」を別スレッドで組み直す。
//     YaneuraOu の Position は StateInfo へのポインタ(Position::st)を持つので、
//     main search の Position を共有することも単純コピーすることも禁止。
//   - main thread への通知は std::atomic<bool> mateFound と
//     std::atomic<uint32_t> mateMove の 2 word だけ。
//   - search()/qsearch() のホットパスには一切分岐を足していない。起動/停止は
//     go の入口と出口(= 探索の外側)でのみ行う。
//   - USI オプション UseMateSolverAsync(既定 false) が off のときは何も起動しない
//     ので、現行バイナリと bit-exact。
//
// ★Threads の値とは独立に起動する(ユーザ判断)。Threads=1 でも起動して 1 コアを
//   余分に使う。探索スレッドを 1 本削って詰み探索に回すようなことはしない。
//
// 探索内で同期的に df-pn を呼ぶ旧実装 (USI: UseMateSolverThread) は 2026-09-03 に削除済み。
// 発火ゼロかつ TT を汚すため。詰み探索はこの非同期版に一本化した。
namespace {

class P2MateSolverAsync
{
   public:
    P2MateSolverAsync() = default;

    // スレッドリーク厳禁。必ず停止要求 → join してから死ぬ。
    ~P2MateSolverAsync() {
        request_stop();
        join();
    }

    P2MateSolverAsync(const P2MateSolverAsync&)            = delete;
    P2MateSolverAsync& operator=(const P2MateSolverAsync&) = delete;

    // root 局面(sfen)から df-pn スレッドを起動する。
    //   max_nodes : 1 go あたりの総ノード上限(USI: MateSolverAsyncMaxNodes)
    //   hash_mb   : df-pn の作業メモリ[MB](USI: MateSolverHashMB を流用)
    void start(const std::string& root_sfen, u64 max_nodes, size_t hash_mb) {
        // solver は main thread 側で構築する。停止要求 dfpn_stop() を外から
        // 呼べるように、main thread が実体へのポインタを保持する必要があるため。
        solver =
          std::make_unique<Mate::Dfpn::MateDfpnSolver>(Mate::Dfpn::DfpnSolverType::Node32bit);

        // 作業メモリ。MateSolverHashMB[MB] を流用するが、ノード予算を使い切れる
        // だけの Node 数が確保できないと out of memory で解けなくなるので、
        // 足りない場合はノード数指定で確保する。
        // (Node32bit の Node は 16 bytes 固定 : mate_dfpn.hpp の Node 定義参照)
        constexpr u64 DFPN_NODE_SIZE = 16;
        const u64     nodes_by_mb    = (u64) hash_mb * 1024 * 1024 / DFPN_NODE_SIZE;
        if (nodes_by_mb >= max_nodes)
            solver->alloc(hash_mb);
        else
            solver->alloc_by_nodes_limit((size_t) max_nodes);

        solver->dfpn_stop(false);

        stop.store(false, std::memory_order_relaxed);
        mateFound.store(false, std::memory_order_relaxed);
        mateMove.store(0, std::memory_order_relaxed);

        // ⚠ このビルドは -fno-exceptions なので try/catch は使えない。
        //    スレッド生成に失敗した場合は std::terminate になるが、
        //    これはメモリ確保失敗と同等の致命的状態なので容認する。
        th = std::thread([this, root_sfen, max_nodes]() { worker_body(root_sfen, max_nodes); });
    }

    // 停止要求(非ブロッキング)。
    void request_stop() {
        stop.store(true, std::memory_order_release);
        if (solver)
            // df-pn の内部探索ループを協調停止させる(mate_dfpn.hpp の !stop 判定)。
            // ⚠ MateDfpnSolver::dfpn_stop() は「入れ物」の stop を書くだけで impl に
            //    伝わらないバグがあったので mate.h 側で impl へ転送するよう修正済み。
            solver->dfpn_stop(true);
    }

    // スレッドの終了を待つ。
    void join() {
        if (th.joinable())
            th.join();
    }

    // 詰み手(Move の生値)。未発見なら 0。
    uint32_t mate_move_raw() const {
        return mateFound.load(std::memory_order_acquire) ? mateMove.load(std::memory_order_relaxed)
                                                         : 0;
    }

   private:
    void worker_body(const std::string& root_sfen, u64 max_nodes) {
        // ★Position は StateInfo へのポインタを持つので、sfen から完全に別個体として
        //   組み直す。Position/StateInfo とも巨大なのでスレッドスタックではなく heap に置く。
        auto pos = std::make_unique<Position>();
        auto si  = std::make_unique<StateInfo>();
        if (pos->set(root_sfen, si.get()))
            return;  // sfen の parse に失敗 → 何もしない

        // 最大手数の上限。rust 版の max_depth = 31 に合わせる。
        solver->set_max_game_ply(pos->game_ply() + 31);

        // rust 版と同じ反復: max(10000, MaxNodes/8) から始めて、見つからなければ
        // 倍々に増やして MaxNodes で打ち止め。
        u64 current = std::max<u64>(10000, max_nodes / 8);

        while (!stop.load(std::memory_order_relaxed))
        {
            Move m = solver->mate_dfpn(*pos, current);

            if (stop.load(std::memory_order_relaxed))
                return;

            if (m == Move::null())
                return;  // 不詰が証明された → これ以上探索しても無意味

            if (m != Move::none())
            {
                // 詰み発見。2 word だけで main thread に通知する。
                // (mateMove を先に書いてから release で mateFound を立てる)
                mateMove.store(m.to_u32(), std::memory_order_relaxed);
                mateFound.store(true, std::memory_order_release);
                return;
            }

            if (current >= max_nodes)
                return;  // ノード予算を使い切った

            current = std::min(max_nodes, current * 2);
        }
    }

    std::thread                                 th;
    std::unique_ptr<Mate::Dfpn::MateDfpnSolver> solver;
    std::atomic<bool>                           stop{false};
    std::atomic<bool>                           mateFound{false};
    std::atomic<uint32_t>                       mateMove{0};
};

}  // namespace
#endif  // defined(EVAL_P2NNUE)

void Search::YaneuraOuWorker::start_searching() {
    if (P2Stat::enabled() && is_mainthread()) P2Stat::reset();
#if defined(P2_KHCORR_DEBUG)
    if (is_mainthread()) kh_debug_print(rootPos);
#endif
#ifdef P2_LMR_STATS
    if (is_mainthread()) P2LmrStat::reset();
#endif
#ifdef P2_PRUNE_STATS
    if (is_mainthread()) P2PruneStat::reset();
#endif

#if defined(USE_SFNN)
    // 謗｢邏｢縺ｮ蛻晏屓evaluate()縺ｧ縺ｯ螻髱｢縺ｮ蟾ｮ蛻・峩譁ｰ縺後〒縺阪↑縺・・縺ｧ
    // accumulator縺ｮ蛻晏屓繝輔Λ繧ｰ繧偵そ繝・ヨ縺吶ｋ縲・
    accumulatorStack.reset();
#endif

    // 蜑榊屓縺ｮgo縺ｧ蠕励◆PV縺ｯ莉雁屓縺ｮ謗｢邏｢縺ｧ縺ｯ菴ｿ縺医↑縺・・縺ｧ縲∝推Worker縺ｮ謗｢邏｢髢句ｧ区凾縺ｫ遐ｴ譽・☆繧九・
    lastIterationPV.clear();

#if defined(EVAL_P2NNUE)
    // 笘・虚逧・FV_SCALE / LMR: 繝ｫ繝ｼ繝亥ｱ髱｢ rootPos 縺ｮ progress 縺九ｉ eff_fv 縺ｨ eff_lmr 繧・
    //   1蝗槭□縺醍｢ｺ螳壹＠縲√％縺ｮ謗｢邏｢(go)縺ｮ蜈ｨ繝弱・繝峨〒蝗ｺ螳壹☆繧九ょ・ Worker 縺悟酔荳 rootPos
    //   progress 縺九ｉ蜷御ｸ蛟､繧定ｨ育ｮ励☆繧九・縺ｧ遶ｶ蜷医＠縺ｦ繧ょ､縺ｯ荳閾ｴ縺吶ｋ(Threads=1 縺ｪ繧牙腰邏比ｻ｣蜈･)縲・
    Eval::p2_search_begin(rootPos);
#if defined(P2_SRCH_BUNDLE_A)
    bundle_a_begin();
#endif
    // 繝ｫ繝ｼ繝亥崋螳壹・ LMR 菫よ焚縺ｧ縲√％縺ｮ Worker 縺ｮ reductions[] table 繧貞・讒狗ｯ峨☆繧九・
    //   (clear() 縺ｨ蜷御ｸ蠑上・MR_OPENING==荳ｭ逶､蛟､ 縺ｪ繧牙酔荳菫よ焚=bit-exact 縺ｫ table 蜀肴ｧ狗ｯ・
    {
        const int lmr_coeff = Eval::p2_root_lmr_coeff();
        for (size_t i = 1; i < reductions.size(); ++i)
            reductions[i] = int(lmr_coeff / 128.0 * std::log(i));
    }
#endif

    // Non-main threads go directly to iterative_deepening()
    // 繝｡繧､繝ｳ繧ｹ繝ｬ繝・ラ莉･螟悶・逶ｴ謗･ iterative_deepening() 縺ｸ騾ｲ繧

    if (!is_mainthread())
    {
        iterative_deepening();
        return;
    }

    // 東 縺薙％莉･荳九・繧ｳ繝ｼ繝峨・縲［ain thread縺ｧ"go"縺ｫ蟇ｾ縺励※螳溯｡後＆繧後ｋ縲・
    //     "go"縺ｮ縺斐→縺ｫ蛻晄悄蛹悶＠縺ｪ縺・→縺・￠縺ｪ縺・ｂ縺ｮ縺ｯ縺薙％縺ｧ陦後≧縲・

    // iterative_deepening()蜀・〒譛邨１V繧偵☆縺ｧ縺ｫ蜃ｺ蜉帙＠縺溘°縲・
    bool uciPvSent = false;

    // 東 莉雁屓縺ｮ諤晁・凾髢薙・險ｭ螳壹・
    //     縺薙ｌ縺ｯ縲｝onderhit縺励◆譎ゅ↓ponderhit縺ｫ繝代Λ繝｡繝ｼ繧ｿ繝ｼ縺御ｻ倬囂縺励※縺・ｌ縺ｰ
    //     蜀崎ｨ育ｮ励☆繧九☆繧句ｿ・ｦ∵ｧ縺後≠繧九・縺ｧ縲√＞縺壹ｌ縺ｫ縺帙ｈ蜻ｼ縺ｳ蜃ｺ縺励※縺翫￥蠢・ｦ√′縺ゅｋ縲・
    // 庁 繧・・縺・ｉ邇九〒縺ｯ縲｛riginalTimeAdjust縺ｯ逕ｨ縺・↑縺・・

#if STOCKFISH
    main_manager()->tm.init(limits, rootPos.side_to_move(), rootPos.game_ply(), options,
                            main_manager()->originalTimeAdjust);
#else
    auto& mainManager = *main_manager();

    mainManager.tm.init(limits, rootPos.side_to_move(), rootPos.game_ply(), options,
                        mainManager.search_options.max_moves_to_draw);
#endif

    // 東 鄂ｮ謠幄｡ｨ縺ｮTTEntry縺ｮ荳紋ｻ｣繧帝ｲ繧√ｋ縲・
    // 統 sub thread縺悟虚縺丞燕縺ｧ縺ゅｋ縺薙・繧ｿ繧､繝溘Φ繧ｰ縺ｧ鄂ｮ謠幄｡ｨ縺ｮ荳紋ｻ｣繧帝ｲ繧√ｋ縺ｹ縺阪〒縺ゅｋ縲・
    //     cf. Call TT.new_search() earlier. : https://github.com/official-stockfish/Stockfish/commit/ebc563059c5fc103ca6d79edb04bb6d5f182eaf5

    // 鄂ｮ謠幄｡ｨ縺ｮ荳紋ｻ｣繧ｫ繧ｦ繝ｳ繧ｿ繝ｼ繧帝ｲ繧√ｋ(繧ｯ繝ｪ繧｢縺ｧ縺ｯ縺ｪ縺・
    tt.new_search();

#if STOCKFISH
#else
    // 決 繧・・縺・ｉ邇句崋譛峨・蛻晄悄蛹・決

    auto& search_options = mainManager.search_options;

    // PV縺瑚ｩｰ縺ｾ繧九・繧呈椛蛻ｶ縺吶ｋ縺溘ａ縺ｫ縲∝燕蝗槫・蜉帶凾蛻ｻ繧定ｨ倬鹸縺励※縺翫￥縲・
    search_options.lastPvInfoTime = limits.startTime;

    // PV縺ｮ蜃ｺ蜉幃俣髫納ms]
    // go infinite縺ｯShogiGUI縺ｪ縺ｩ縺ｮ讀懆ｨ弱Δ繝ｼ繝峨〒蜍穂ｽ懊＆縺帙※縺・ｋ縺ｨ閠・∴繧峨ｌ繧九・縺ｧ
    // 縺薙・蝣ｴ蜷医・縲￣V繧呈ｯ主屓蜃ｺ蜉帙＠縺ｪ縺・→隱ｭ縺ｿ遲九′蜃ｺ蜉帙＆繧後↑縺・％縺ｨ縺後≠繧九・
    search_options.computed_pv_interval =
      (limits.infinite || limits.disablePvInterval || search_options.consideration_mode)
        ? 0
        : search_options.pv_interval;

    // 決 蠑輔″蛻・￠縺ｮ繧ｹ繧ｳ繧｢

    // 蠑輔″蛻・￠譎ゅ・蛟､縺ｨ縺励※迴ｾ蝨ｨ縺ｮ謇狗分縺ｫ蠢懊§縺溷､繧定ｨｭ螳壹＠縺ｦ繧・ｋ縲・
    Color us         = rootPos.side_to_move();
    int   draw_value = (int) ((us == BLACK ? options["DrawValueBlack"] : options["DrawValueWhite"])
                            * Eval::PawnValue / 100);

    // 謗｢邏｢縺ｮleaf node縺ｧ縺ｯ縲∫嶌謇狗分(root_color != side_to_move)縺ｧ縺ゅｋ蝣ｴ蜷医・+draw_value縺ｧ縺ｯ縺ｪ縺上・draw_value繧定ｨｭ螳壹＠縺ｦ繧・ｉ縺ｪ縺・→髱槫ｯｾ遘ｰ縺ｪ謗｢邏｢縺ｨ縺ｪ縺｣縺ｦ濶ｯ縺上↑縺・・
    // 萓・ 閾ｪ蛻・・蠑輔″蛻・￠繧貞享縺｡謇ｱ縺・□縺ｨ諤昴▲縺ｦ謗｢邏｢縺励※縺・ｋ縺ｪ繧峨∫嶌謇九・縲∝ｼ輔″蛻・￠繧定ｲ縺代→縺ｿ縺ｪ縺励※縺上ｌ縺ｪ縺・→髱槫ｯｾ遘ｰ縺ｫ縺ｪ繧九・
    drawValueTable[REPETITION_DRAW][us]  = +draw_value;
    drawValueTable[REPETITION_DRAW][~us] = -draw_value;

    // 莉雁屓縲・壼ｸｸ謗｢邏｢繧偵＠縺溘°縺ｮ繝輔Λ繧ｰ
    // 縺薙・繝輔Λ繧ｰ縺荊rue縺ｪ繧・螳夊ｷ｡縺ｫhit縺励◆繧・謇玖ｩｰ繧√ｒ逋ｺ隕九＠縺溘ｊ縺励◆縺ｮ縺ｧ)謗｢邏｢繧偵せ繧ｭ繝・・縺励◆縲・
    bool search_skipped = true;

    // 決 ponder逕ｨ縺ｮ謖・＠謇九・蛻晄悄蛹・
    //    繧・・縺・ｉ邇九〒縺ｯ縲｝onder縺ｮ謖・＠謇九′縺ｪ縺・→縺阪∽ｸ縺､蜑阪・iteration縺ｮ縺ｨ縺阪・PV荳翫・(逶ｸ謇九・)謖・＠謇九ｒ逕ｨ縺・ｋ縺ｨ縺・≧迢ｬ閾ｪ莉墓ｧ倥・
    //     Stockfish譛ｬ螳ｶ繧ゅ％縺・☆繧九∋縺阪□縺ｨ諤昴≧縲・
    mainManager.ponder_candidate = Move::none();

    // 謗｢邏｢縺帙★縺ｫ謖・＠謇九ｒ霑斐☆縺ｨ縺阪・謖・＠謇・
    Book::ProbeResult probeResult;

#if defined(EVAL_P2NNUE)
    // 非同期 df-pn 詰みsolver(USI: UseMateSolverAsync, 既定 false)。
    // ⚠ この下に goto SKIP_SEARCH があるため、宣言は必ずラベルより前(=ここ)に置く。
    //    デストラクタで stop+join するので、どの経路で抜けてもスレッドは回収される。
    P2MateSolverAsync mateAsync;
#endif

#if defined(USE_BOOK_AWARE)
    // 🎯🎯**理論離脱 v2** (DESIGN-20261027-THEORYEXIT-V2.md §3.1)。
    //   probe が「定跡手 m0 を既定にして探索へ回す」と決めたときの受け皿。
    //   ⚠上の mateAsync と同じ理由で、 goto SKIP_SEARCH を跨ぐため**ここで宣言する**。
    Move              ba_m0     = Move::none();   // 定跡手 (棄却ならこれを指す)
    std::vector<Move> ba_exit;                    // 出口集合 E (パス A の rootMoves)
    bool              ba_active = false;          // この go で v2 が動いているか
    LimitsType        ba_saved_limits;            // パス A の前の limits (必ず戻す)
    // 🚨probe が返した定跡手の評価値。 探索せずに m0 を指すときの `score` に使う
    //   (入れないと -VALUE_INFINITE のまま出て評価値が壊れる / 2026-10-27)。
    Value             ba_book_value = VALUE_ZERO;
#endif


#if defined(SHOGI24)
    // ---------------------
    //    蟆・｣句ｶ讌ｽ驛ｨ24蟇ｾ遲・
    // ---------------------

    // 逶ｸ謇狗脂縺悟叙繧後ｋ縺ｪ繧牙叙繧九・
    //
    // 逶ｸ謇狗脂縺悟叙繧後ｋ螻髱｢縺ｯ縲・逶ｴ蜑阪〒邇区焔謾ｾ鄂ｮ縺後≠縺｣縺溘→縺・≧縺薙→縺縺九ｉ)髱槫粋豕募ｱ髱｢縺ｧ縲・
    // 蟆・｣区園縺ｧ縺ｯ縺昴・繧医≧縺ｪ螻髱｢縺九ｉ縺ｮ蟇ｾ螻髢句ｧ九・縺ｧ縺阪↑縺・′縲ヾhogiGUI縺ｧ縺ｯ蟇ｾ螻髢句ｧ九〒縺阪ｋ縲・
    //
    // 縺ｾ縺溘∝ｰ・｣句ｶ讌ｽ驛ｨ24縺ｧ邇区焔謾ｾ鄂ｮ縺ｮ螻髱｢繧剃ｽ懊ｋ縺薙→縺後〒縺阪ｋ縺ｮ縺ｧ縲・
    // 逶ｸ謇狗脂縺悟叙繧後ｋ縺薙→縺後≠繧九・
    //
    // 繧・∴縺ｫ縲∝叙繧後ｋ縺ｪ繧牙叙繧区欠縺玲焔繧呈欠縺帙◆縺ｻ縺・′濶ｯ縺・・
    //
    // 蜿り・虚逕ｻ : https://www.youtube.com/watch?v=8nwJcKH0x0c

    auto their_king = rootPos.square<KING>(~us);
    auto our_piece  = rootPos.attackers_to(their_king) & rootPos.pieces(us);
    // 謨ｵ邇峨↓蛻ｩ縺・※縺・ｋ閾ｪ鬧偵′縺ゅｋ縺ｪ繧峨√◎繧後ｒ遘ｻ蜍輔＆縺帙※蜍昴■縲・
    if (our_piece)
    {
        Square from = our_piece.pop();
        Square to   = their_king;
        Move16 m16  = make_move16(from, to);
        Move   m    = rootPos.to_move(m16);

        // 邇峨ｒ蜿悶ｋ謖・＠謇九・captures縺ｧ逕滓・縺輔ｌ縺ｦ縺・↑縺・庄閭ｽ諤ｧ縺後≠繧九・
        // 莉墓婿縺後↑縺・・縺ｧ縲〉ootMoves[0]繧呈嶌縺肴鋤縺医ｋ縺薙→縺ｫ縺吶ｋ縲・

        // 邇峨〒邇峨ｒ蜿悶ｋ謇九・rootMoves縺ｫ蜷ｫ縺ｾ繧後↑縺・・縺ｧ縲∝ｴ蜷医↓繧医▲縺ｦ縺ｯ縲√◎繧後＠縺区欠縺玲焔縺後↑縺・ｴ蜷医↓縲・
        // rootMoves.size() == 0縺縺代←縲∫脂縺ｧ邇峨ｒ蜿悶ｋ謖・＠謇九□縺代′縺ゅｋ縺薙→縺ｯ襍ｷ縺薙ｊ蠕励ｋ縲・
        // (縺薙・逅・罰縺九ｉ縲∫脂繧貞叙繧句愛螳壹・縲∝粋豕墓焔縺後↑縺・愛螳壹ｈ繧雁・縺ｫ縺励↑縺代ｌ縺ｰ縺ｪ繧峨↑縺・

        probeMove.bestmove  = m;
        probeMove.bestscore = mate_in(1);
        goto SKIP_SEARCH;
    }
#endif

    // 笨・迢ｬ閾ｪ霑ｽ蜉縺薙％縺ｾ縺ｧ縲・
#endif

    // ---------------------
    // 蜷域ｳ墓焔縺後↑縺・↑繧峨％縺薙〒謚穂ｺ・
    // ---------------------

    if (rootMoves.empty())
    {
        // root縺ｧ謖・＠謇九′縺ｪ縺・= (蟆・｣九□縺ｨ)隧ｰ縺ｿ縺ｮ螻髱｢縺ｧ縺ゅｋ

        probeResult.bestmove  = Move::resign();
        probeResult.bestscore = mated_in(1);

		// 庁 縺薙・縺ゅ→rootMoves[0]縺ｫ繧｢繧ｯ繧ｻ繧ｹ縺励※縲√い繧ｯ繧ｻ繧ｹ驕募渚縺ｫ縺ｪ繧九・繧帝亟縺舌◆繧√・
        rootMoves.emplace_back(Move::none());

#if STOCKFISH
        main_manager()->updates.onUpdateNoMoves(
          {0, {rootPos.checkers() ? -VALUE_MATE : VALUE_DRAW, rootPos}});
        // 庁 繝√ぉ繧ｹ縺縺ｨ邇区焔縺輔ｌ縺ｦ縺・↑縺・↑繧牙ｼ輔″蛻・￠縺縺後∝ｰ・｣九□縺ｨ縺､縺ｭ縺ｫ雋縺代・
#else
		// 繧・・縺・ｉ邇九〒縺ｯ縲√％縺ｮ縺ゅ→縲｝robeResult繧堤畑縺・ｋ譎ら畑縺ｫPV繧貞・蜉帙☆繧九・縺ｧ縲√◎縺薙〒陦後≧縲・
#endif

        goto SKIP_SEARCH;
    }

    // ---------------------
    //    螳｣險蜍昴■蛻､螳・
    // ---------------------

    {
        // 螳｣險蜍昴■縺ｪ繧峨◎縺ｮ謖・＠謇九ｒ驕ｸ謚槭・
        // 邇区焔縺後°縺九▲縺ｦ縺・※繧ゅ∝屓驕ｿ縺励↑縺後ｉ繝医Λ繧､縺吶ｋ縺薙→繧ゅ≠繧九・縺ｧ邇区焔縺後°縺九▲縺ｦ縺・ｈ縺・′
        // Position::DeclarationWin()縺ｧ蛻､螳壹＠縺ｦ濶ｯ縺・・
        // 1謇玖ｩｰ繧√・縲√％縺薙〒縺ｯ蛻､螳壹＠縺ｪ縺・・
        // (MultiPV縺ｮ縺ｨ縺阪↓1謇玖ｩｰ繧√ｒ隕九▽縺代◆縺九ｉ縺ｨ險縺｣縺ｦ謗｢邏｢繧堤ｵゆｺ・＠縺溘￥縺ｪ縺・°繧峨・

        auto bestMove = rootPos.DeclarationWin();
        if (bestMove != Move::none())
        {
            probeResult.bestmove  = bestMove;
            probeResult.bestscore = mate_in(1);
            goto SKIP_SEARCH;
        }
    }

    // ---------------------
    //     螳夊ｷ｡縺ｮ驕ｸ謚樣Κ
    // ---------------------

    probeResult = engine.book.probe(rootPos, main_manager()->updates);
    if (probeResult.bestmove)
        goto SKIP_SEARCH;

#if defined(USE_BOOK_AWARE)
    // 🎯v2: probe が offer を積んだかをここで受け取る。 絞り込みは探索の**後**で行う。
    //   🚨🚨**`limits.searchmoves` ではできない** (2026-10-27 に G3' で発覚)。
    //     `rootMoves` は `ThreadPool::start_thinking` (thread.cpp:521) が
    //     **search() を呼ぶ前**に `limits.searchmoves` を読んで作る。
    //     ここは既に search() の中なので、 今から searchmoves を書き換えても手遅れで、
    //     パス A が全合法手のまま走る (G3' で「受理したのに子が ybb の中」39/48 として出た)。
    //   ✅パス A / パス B はどちらも「探索が終わってから rootMoves を絞って回し直す」形にする。
    ba_active = BookAware::manager.take_pending(ba_m0, ba_exit, ba_book_value);
    if (ba_active)
    {
        ba_saved_limits = limits;
        // 🎯🎯**パス A = この go の本探索そのもの** (予算は通常 = 設計 §3.1)。
        //   🚨`limits.searchmoves` は使えない (`start_thinking` が search() を呼ぶ前に読むため) が、
        //     **`rootMoves` は Worker のメンバ**なので、 探索が始まる前のここで直接絞って
        //     helper へ配り直せる。 探索後に回し直す必要は無い (fable レビュー A / 2.5x -> 1.5x)。
        //   🚨helper は start_thinking の custom job を終えて idle なので run_custom_job は即時に通る。
        //   🚨main 自身に run_custom_job を投げるとデッドロックする (main は自分のジョブの中)。
        rootMoves.clear();
        for (const Move m : ba_exit)
            rootMoves.emplace_back(m);
        for (auto it = threads.begin(); it != threads.end(); ++it)
        {
            Thread* tp = it->get();
            if (tp == threads.main_thread())
                continue;
            tp->run_custom_job([this, tp]() { tp->worker->rootMoves = this->rootMoves; });
        }
        threads.wait_for_search_finished();
    }
#endif

    // ---------------------
    //    騾壼ｸｸ縺ｮ諤晁・・逅・
    // ---------------------

#if defined(EVAL_P2NNUE)
    // 非同期 df-pn 詰みsolver の起動(探索の外側なのでホットパスに分岐は増えない)。
    // ponder 中は root 局面が「予想局面」なので起動しない。
    // ★Threads の値とは無関係に起動する(Threads=1 でも 1 コア余分に使う)。
    if (Eval::p2_mate_solver_async && !limits.ponderMode)
        mateAsync.start(rootPos.sfen(), (u64) Eval::p2_mate_solver_async_max_nodes,
                        (size_t) Eval::p2_mate_solver_hash_mb);
#endif

    threads.start_searching();  // start non-main threads
    // 統 main莉･螟悶・縺吶∋縺ｦ縺ｮthread繧帝幕蟋九☆繧九・
    //    main莉･螟悶・thread縺茎tart_searching()繧帝幕蟋九☆繧九・
    //    start_searching()縺ｮ蜈磯ｭ縺ｫ縺ｯ縲［ain thread莉･螟悶〒縺ゅｌ縺ｰ蜊ｳ蠎ｧ縺ｫ
    //    iterative_deepning()繧貞他縺ｳ蜃ｺ縺吶ｈ縺・↓縺ｪ縺｣縺ｦ縺・ｋ縺ｮ縺ｧ縲√％繧後↓繧医ｊ荳ｦ蛻玲爾邏｢縺碁幕蟋九〒縺阪ｋ縲・

    uciPvSent = iterative_deepening();  // main thread start searching
    // 庁 main thread繧ゆｸｦ蛻玲爾邏｢縺ｫ蜉繧上ｋ縲・

    // When we reach the maximum depth, we can arrive here without a raise of
    // threads.stop. However, if we are pondering or in an infinite search,
    // the UCI protocol states that we shouldn't print the best move before the
    // GUI sends a "stop" or "ponderhit" command. We therefore simply wait here
    // until the GUI sends one of those commands.

    // 譛螟ｧ豺ｱ縺輔↓蛻ｰ驕斐＠縺溘→縺阪》hreads.stop 縺檎匱逕溘○縺壹↓縺薙％縺ｫ蛻ｰ驕斐☆繧九％縺ｨ縺後≠繧九・
    // 縺励°縺励√・繝ｳ繝繝ｪ繝ｳ繧ｰ荳ｭ繧・┌髯先爾邏｢荳ｭ縺ｮ蝣ｴ蜷医ゞCI繝励Ο繝医さ繝ｫ縺ｧ縺ｯ
    // GUI 縺・"stop" 縺ｾ縺溘・ "ponderhit" 繧ｳ繝槭Φ繝峨ｒ騾√ｋ縺ｾ縺ｧ
    // best move 繧貞・蜉帙☆縺ｹ縺阪〒縺ｯ縺ｪ縺・→縺輔ｌ縺ｦ縺・ｋ縲・
    // 縺励◆縺後▲縺ｦ縲√％縺薙〒縺ｯ蜊倡ｴ斐↓ GUI 縺九ｉ縺薙ｌ繧峨・繧ｳ繝槭Φ繝峨′騾√ｉ繧後※縺上ｋ縺ｮ繧貞ｾ・▽縲・

    // 統 譛螟ｧdepth豺ｱ縺輔↓蛻ｰ驕斐＠縺溘→縺阪↓縲√％縺薙∪縺ｧ螳溯｡後′蛻ｰ驕斐☆繧九′縲・
    //     縺ｾ縺threads.stop縺檎函縺倥※縺・↑縺・ゅ＠縺九＠縲｝onder荳ｭ繧・“o infinite縺ｫ繧医ｋ謗｢邏｢縺ｮ蝣ｴ蜷医・
    //     USI(UCI)繝励Ο繝医さ繝ｫ縺ｧ縺ｯ縲・stop"繧・ponderhit"繧ｳ繝槭Φ繝峨ｒGUI縺九ｉ騾√ｉ繧後※縺上ｋ縺ｾ縺ｧbest move繧貞・蜉帙＠縺ｦ縺ｯ縺ｪ繧峨↑縺・・
    //     縺昴ｌ繧・∴縲∝腰縺ｫ縺薙％縺ｧGUI縺九ｉ縺昴ｌ繧峨・縺・★繧後°縺ｮ繧ｳ繝槭Φ繝峨′騾√ｉ繧後※縺上ｋ縺ｾ縺ｧ蠕・▽縲・
    //
    //     1. "stop"縺碁√ｉ繧後※縺阪◆繧欝hreads.stop == true縺ｫ縺ｪ繧九・
    //     2. "ponderhit"縺碁√ｉ繧後※縺阪◆繧欝hreads.ponder == false縺ｫ縺ｪ繧・
    //       縺ｮ縺ｧ縲√◎繧後ｒ蠕・▽縲・
    //
    //     縺｡縺ｪ縺ｿ縺ｫStockfish縺ｮ縺ｻ縺・√％縺薙・繧ｳ繝ｼ繝峨↓髟ｷ繧峨￥蜷梧悄荳翫・繝舌げ縺後≠縺｣縺溘・
    //     繧・・縺・ｉ邇九・縺ｻ縺・・縲√°縺ｪ繧頑掠縺上°繧峨％縺ｮ讒矩縺ｧ譖ｸ縺・※縺・◆縲ょｾ後↓Stockfish縺後％縺ｮ譖ｸ縺肴婿縺ｫ霑ｽ髫上＠縺溘・

    // 譎ｮ騾壹↓謗｢邏｢縺励◆縺ｮ縺ｧskip縺励◆縺九・繝輔Λ繧ｰ繧断alse縺ｫ縺吶ｋ縲・
    // 庁繧・・縺・ｉ邇狗峡閾ｪ
    search_skipped = false;

SKIP_SEARCH:

    while (!threads.stop && (main_manager()->ponder || limits.infinite))
    {
        // Busy wait for a stop or a ponder reset
        // stop 縺・ponder reset 繧貞ｾ・▽髢薙・繝薙ず繝ｼ繧ｦ繧ｧ繧､繝・

        //	縺薙■繧峨・諤晁・・邨ゅｏ縺｣縺ｦ縺・ｋ繧上￠縺縺九ｉ縲√≠繧狗ｨ句ｺｦ邏ｰ縺九￥蠕・▲縺ｦ繧ょ撫鬘後↑縺・・
        // (諤晁・・縺溘ａ縺ｫ縺ｯ險育ｮ苓ｳ・ｺ舌ｒ菴ｿ縺｣縺ｦ縺・↑縺・・縺ｧ縲・
        Tools::sleep(1);
        // 笞 Stockfish縺ｮ繧ｳ繝ｼ繝峨√％縺薙｜usy wait縺ｫ縺ｪ縺｣縺ｦ縺・ｋ縺後√＆縺吶′縺ｫ縺昴ｌ縺ｯ濶ｯ縺上↑縺・→諤昴≧縲・


        /*
			涛 縺薙％縺ｧPV繧貞・蜉帙＠縺溘⊇縺・′縺・＞縺九ｂ・・

				ponder荳ｭ/go infinite荳ｭ縺ｧ縺ゅ▲縺ｦ繧ゅ√％縺薙↓謚懊￠縺ｦ縺阪※縺・ｋ莉･荳翫・
				蜈ｨ謗｢邏｢繧ｹ繝ｬ繝・ラ縺ｮ蛛懈ｭ｢縺檎｢ｺ隱阪〒縺阪◆譎らせ縺ｧPV縺ｯ蜃ｺ蜉帙＠縺溘⊇縺・′濶ｯ縺・→諤昴≧縲・

				"go infinite"縺ｮ蝣ｴ蜷医∬ｩｰ縺ｿ繧堤匱隕九＠縺ｦ繧ゅ◎繧後′ponder繝輔Λ繧ｰ縺ｮ隗｣髯､繧貞ｾ・▲縺ｦ縺九ｉ縺縺ｨ縲・
				PV繧定ｿ斐☆縺ｮ縺碁≦繧後ｋ縲・"stop"縺梧擂繧九∪縺ｧ霑斐○縺ｪ縺・

				// 縲縺薙％縺ｧ縺吶∋縺ｦ縺ｮ謗｢邏｢繧ｹ繝ｬ繝・ラ縺悟●豁｢縺励※縺・ｋ縺ｪ繧峨・譛邨１V繧貞・蜉帙＠縺ｦ繧・ｋ縲・
				if (!output_final_pv_done
					&& Threads.search_finished()) // 蜈ｨ謗｢邏｢繧ｹ繝ｬ繝・ラ縺梧爾邏｢繧貞ｮ御ｺ・＠縺ｦ縺・ｋ
					output_final_pv();
		*/
    }

    // Stop the threads if not already stopped (also raise the stop if
    // "ponderhit" just reset threads.ponder)
    // 縺ｾ縺蛛懈ｭ｢縺励※縺・↑縺代ｌ縺ｰ繧ｹ繝ｬ繝・ラ繧貞●豁｢縺吶ｋ・・ponderhit" 縺ｫ繧医ｊ threads.ponder 縺・
    // 繝ｪ繧ｻ繝・ヨ縺輔ｌ縺溷ｴ蜷医ｂ stop 繧堤匱逕溘＆縺帙ｋ・・
    threads.stop = true;

#if defined(EVAL_P2NNUE)
    // 非同期 df-pn 詰みsolver にも停止要求を出す(join は下でまとめて行う)。
    mateAsync.request_stop();
#endif

    // Wait until all threads have finished
    // 縺吶∋縺ｦ縺ｮ繧ｹ繝ｬ繝・ラ縺檎ｵゆｺ・☆繧九・繧貞ｾ・▽
    // 庁 髢句ｧ九＠縺ｦ縺・↑縺代ｌ縺ｰ縺・↑縺・〒讒九ｏ縺ｪ縺・・

    threads.wait_for_search_finished();

#if defined(USE_BOOK_AWARE)
	// 🎯🎯**理論離脱 v2 のパス B** (設計 runs/prereg/DESIGN-20261027-THEORYEXIT-V2.md §3.1)。
	//   ここは全スレッドが停止済みで bestmove 出力前。
	//     パス A (直前に終わった探索) = **出口集合 E だけ**を通常予算で探索 → m2, v2
	//     パス B (ここで回す)         = **定跡手 m0 だけ**を 1/2 予算で探索 → v0
	//     accept_deviation(v0, v2) が受理なら m2、 棄却なら **m0 を指す**。
	//   🚨🚨**v1 との違い**: v1 は比較対象が「素の探索手 m1」で、 棄却しても m1 を指していた。
	//     定跡手 m0 を指す経路が無かったので、 ゲートが開いている間ずっと ybb を捨てていた
	//     (= 事実上「定跡なしの腕」/ 1 局 2.4 手 × 中央 30cp の損)。
	//     v2 は m0 が既定なので、 損は構造上「1 局 1 回 ≤ B」に収まる。
	if (ba_active)
	{
		// 🎯直前に終わった探索が**パス A** (rootMoves は take_pending の直後に E へ絞ってある)。
		//   ここで回すのは**パス B (m0 だけ / 1/2 予算)** だけ。

		// 指定した手だけを rootMoves に据えて探索し直す。 予算は budget_nodes / budget_ms で与える。
		//   🚨main 自身に run_custom_job を投げると**デッドロックする** (main は既に自分のジョブの
		//     中にいるので searching が false にならない)。 helper (index 1 以降) だけに配る。
		auto rerun = [&](const std::vector<Move>& ms, u64 budget_nodes, TimePoint budget_ms) {
			rootMoves.clear();
			for (const Move m : ms)
				rootMoves.emplace_back(m);
			if (rootMoves.empty())
				return;
			limits = ba_saved_limits;
			if (ba_saved_limits.nodes)
				limits.nodes = threads.nodes_searched() + std::max<u64>(1000, budget_nodes);
			else
			{
				// 🚨🚨**v1 のバグ** (設計 §3.5-①): `limits.movetime = elapsed/2` と書くと、
				//   停止判定は `elapsed >= movetime` で `elapsed` は**元の go からの経過**なので
				//   既に超えており、 追加パスが depth 1 で即停止して値が偽になる。
				//   ✅**絶対時刻で与える**。 ponder で待っていた時間は予算に入れない。
				auto& tm = main_manager()->tm;
				limits.movetime = tm.elapsed_time() + std::max<TimePoint>(50, budget_ms);
				// 🚨前のパスが設定した終了予定時刻が残っていると条件 5 で即停止する。
				tm.search_end = 0;
			}
			for (auto it = threads.begin(); it != threads.end(); ++it)
			{
				Thread* tp = it->get();
				if (tp == threads.main_thread())
					continue;
				// 🚨helper の `start_searching()` は `pre_start_searching()` を呼ばないので、
				//   rootDepth/completedDepth/bestMoveChanges/nmpMinPly が前のパスの値のまま残る。
				//   ここで明示的にリセットする (fable レビュー B-2)。
				tp->run_custom_job([this, tp]() {
					tp->worker->rootMoves = this->rootMoves;
					tp->worker->pre_start_searching();
				});
			}
			threads.wait_for_search_finished();
			// main 自身は start_searching() を再入できないので手で戻す。
			rootDepth = 0; completedDepth = 0; selDepth = 0;
			bestMoveChanges = 0; nmpMinPly = 0;
			threads.stop = false;
			threads.start_searching();   // helper を起こす
			iterative_deepening();       // main
			threads.stop = true;
			threads.wait_for_search_finished();
			limits = ba_saved_limits;
		};

		// 定跡手 m0 を rootMoves に据える (棄却・スキップ時はこれを指す)。
		//   🚨🚨**score を必ず入れる** (2026-10-27 に 48 局すべての最終評価値が -51 固定になって発覚)。
		//     `RootMove(Move)` は **pv に手を入れるだけ**で `score` は `-VALUE_INFINITE` のまま
		//     (search.h:103 / 133)。 そのまま bestmove にすると
		//       ・USI の `score cp` が壊れた固定値になる (jsonl / CSA の評価値が無意味に)
		//       ・`bestPreviousScore` を通じて**次の手の aspiration 窓**が壊れる
		//       ・`get_best_thread()` の minScore が -∞ になり helper の票が暴れる (fable 指摘 B の裏口)
		//     -> 呼ぶ側が持っている値 (パス B の v0 / 無ければ定跡値) を渡す。
		auto set_book_move = [&](Value sc) {
			rootMoves.clear();
			rootMoves.emplace_back(ba_m0);
			rootMoves[0].score         = sc;
			rootMoves[0].uciScore      = sc;
			rootMoves[0].previousScore = sc;
			rootMoves[0].averageScore  = sc;
			rootMoves[0].selDepth      = 1;
		};

		// 🚨🚨**外から止められた直後は追加パスを回さない** (設計 §3.5-②)。
		//   ponder miss の `stop` の後に探索を始めると bestmove が予算ぶん遅れる。
		//   🚨`goto` は変数の初期化を跨げないので**ネストで書く** (2026-10-27 にビルドが落ちた)。
		if (threads.external_stop)
		{
			// 🎯探索を 1 回もしていないので**試行に数えない** (設計 §3.5-② / fable レビュー C)。
			//   本番は `USI_Ponder=true` なので、 数えると miss が続くだけで Max に達して機構が止まる。
			BookAware::manager.uncount_attempt();
			// 🚨外部 stop で何も読めていないので、 **probe が返した定跡値**を使う。
			set_book_move(ba_book_value);
		}
		else
		{
			// ---- パス A は**この go の本探索そのもの**。 上 (take_pending の直後) で
			//      rootMoves を E に絞ってあるので、 直前に終わった探索の結果がそれ。
			const RootMoves exit_result = rootMoves;
			const Value     v2          = rootMoves.empty() ? -VALUE_INFINITE : rootMoves[0].score;

			if (exit_result.empty() || is_decisive(v2) || v2 == -VALUE_INFINITE)
			{
				// 出口が読めなかった -> 定跡手に戻すだけ (損ゼロ)。 値は probe の定跡値。
				set_book_move(ba_book_value);
			}
			else
			{
				// ---- パス B: 定跡手 m0 だけを**1/2 予算**で探索 → v0
				rerun(std::vector<Move>{ba_m0},
				      ba_saved_limits.nodes / 2,
				      main_manager()->tm.elapsed_time_from_ponderhit() / 2);

				const Value v0 = rootMoves.empty() ? -VALUE_INFINITE : rootMoves[0].score;
				if (!rootMoves.empty() && !is_decisive(v0) && v0 != -VALUE_INFINITE
				    && BookAware::manager.accept_deviation(int(v0), int(v2)))
					rootMoves = exit_result;   // 受理 → 理論の外へ出る手 m2 を指す
				else if (rootMoves.empty())
					set_book_move(ba_book_value);   // 保険
				// 棄却ならここは m0 のまま = **パス B が探索した rootMoves** なので
				// score も pv も正規の探索結果が入っている (set_book_move は呼ばない)。
			}
		}
	}
#endif

#if defined(EVAL_P2NNUE)
    // 非同期 df-pn 詰みsolver を必ず join する(スレッドリーク厳禁)。
    mateAsync.join();
#endif

// 庁 繧・・縺・ｉ邇九〒縺ｯ縲］pmsec繧偵し繝昴・繝医＠縺ｪ縺・・
#if STOCKFISH
    // When playing in 'nodes as time' mode, subtract the searched nodes from
    // the available ones before exiting.
    // 'nodes as time' 繝｢繝ｼ繝峨〒繝励Ξ繧､縺励※縺・ｋ蝣ｴ蜷医∫ｵゆｺ・☆繧句燕縺ｫ
    // 菴ｿ逕ｨ蜿ｯ閭ｽ縺ｪ繝弱・繝画焚縺九ｉ謗｢邏｢貂医∩繝弱・繝画焚繧貞ｷｮ縺怜ｼ輔￥縲・

    // 統 'nodes as time'繝｢繝ｼ繝峨→縺ｯ縲∵凾髢薙→縺励※nodes繧堤畑縺・ｋ繝｢繝ｼ繝・
    //     譎る俣蛻・ｌ縺ｮ蝣ｴ蜷医∬ｲ縺ｮ謨ｰ縺ｫ縺ｪ繧翫≧繧九・
    // 笞 蟆・｣九・蝣ｴ蜷医∫ｧ定ｪｭ縺ｿ縺後≠繧九・縺ｧ遘定ｪｭ縺ｿ繧り・・縺励↑縺・→縺・￠縺ｪ縺・・
    //     Time.availableNodes += Limits.inc[us] + Limits.byoyomi[us] - Threads.nodes_searched();
    //     縺ｿ縺溘＞縺ｪ蜃ｦ逅・′蠢・ｦ√°・溷ｰ・｣九→逶ｸ諤ｧ縺瑚憶縺上↑縺・・縺ｧ縺薙・讖溯・縲∫┌蜉ｹ蛹悶☆繧九・

    if (limits.npmsec)
        main_manager()->tm.advance_nodes_time(threads.nodes_searched()
                                              - limits.inc[rootPos.side_to_move()]);
#endif

    // 東 謖・＠謇九ｒGUI縺ｫ霑斐☆ 東

    // Lazy SMP縺ｮ邨先棡繧貞叙繧雁・縺・

#if STOCKFISH
    // 荳ｦ蛻玲爾邏｢縺励◆縺・■縺ｮbest縺ｪ邨先棡繧剃ｿ晄戟縺励※縺・ｋthread
    // 縺ｾ縺嗾his繧貞・繧後※縺翫＞縺ｦ縲∝ｮ夊ｷ｡繧定ｿ斐☆譎ゅ↑縺ｩ縺ｯthis縺ｮ縺ｾ縺ｾ縺ｫ縺吶ｋ繧ｳ繝ｼ繝峨ｒ驕ｩ逕ｨ縺吶ｋ縲・
    Worker* bestThread = this;

    Skill skill =
      Skill(options["Skill Level"], options["UCI_LimitStrength"] ? int(options["UCI_Elo"]) : 0);
    // TODO : Skill縺ｮ蟆主・縺ｯ縺ゅ→縺ｧ讀懆ｨ弱☆繧九・
    //  ､・ 縺昴ｌ縺ｫ縺励※繧ゅが繝励す繝ｧ繝ｳ縺・縺､繧ょ｢励∴繧九・雖後□縺ｪ窶ｦ縲・

#else
    YaneuraOuWorker* bestThread = this;

    Skill skill = Skill(20, 0);
#endif

    // 🚨🚨**理論離脱 v2 のときは helper の投票で決定を覆させない** (fable レビュー B-1 / 2026-10-27)。
    //   投票の重みは `(score - minScore + 14) * completedDepth` (776-856)。
    //   v2 が m0 を据えたとき main の score は -VALUE_INFINITE なので minScore も -∞ になり、
    //   helper の票が巨大になって **helper の rootMoves[0] が bestmove になってしまう**。
    //   具体的には ①external_stop スキップで素の探索手 ②パス A が decisive のとき B の検査なしで m2
    //   ③受理時も票次第で m0 が出て deviated だけ消費する。
    //   🚨G3'/G3''/G7 は Threads=1 で走ったので見えなかった。 **本番 toml は Threads=2**。
    //   決定は main の rootMoves に入っているので、 ここは素通りさせる。
    if (int(options["MultiPV"]) == 1 && !limits.depth && !limits.mate && !skill.enabled()
        && rootMoves[0].pv[0] != Move::none()
#if defined(USE_BOOK_AWARE)
        && !ba_active
#endif
        )
#if STOCKFISH
        bestThread = threads.get_best_thread()->worker.get();
#else
        // 庁 繧・・縺・ｉ邇九〒縺ｯ縲“et_best_thread()縺ｯ縲ゝhreadPool縺九ｉ縺薙・class縺ｫ遘ｻ蜍輔＆縺帙◆縲・
        bestThread = get_best_thread();
#endif

    // 谺｡蝗槭・謗｢邏｢縺ｮ縺ｨ縺阪↓菴輔ｉ縺倶ｽｿ縺医ｋ縺ｮ縺ｧ繝吶せ繝医↑謖・＠謇九・隧穂ｾ｡蛟､繧剃ｿ晏ｭ倥＠縺ｦ縺翫￥縲・
    // BookExitTimeFix (2026-11-17。 Kisou (kisou-public) の 2026-11-13 の修正 start_searching() 末尾の写し):
    //   定跡・宣言勝ち・合法手なしで探索を飛ばしたときは rootMoves[0].score / averageScore が既定値の -VALUE_INFINITE のまま
    //   (RootMove(Move) は score を入れない) なので、 そのまま保存すると次の探索 (= 定跡を抜けた最初の 1 手) で
    //   fallingEval が「-∞ からの急上昇」と見て時間係数を下限 (0.5688) に張り付ける。 飛ばしたときは「前回評価なし」
    //   (VALUE_INFINITE = 対局開始時と同じ扱い) にする。
    //   効くのは search_skipped のときだけ。 定跡を使わず探索する手 (go nodes / go depth の関門・定跡なしの対局) では
    //   search_skipped == false なので従来と同じ値を保存する。 option が false なら常に従来どおり。
    if (search_skipped && bool(options["BookExitTimeFix"]))
    {
        main_manager()->bestPreviousScore        = VALUE_INFINITE;
        main_manager()->bestPreviousAverageScore = VALUE_INFINITE;
    }
    else
    {
        main_manager()->bestPreviousScore        = bestThread->rootMoves[0].score;
        main_manager()->bestPreviousAverageScore = bestThread->rootMoves[0].averageScore;
    }
#if defined(P2_BUCKET_OFFSET)
    // SIDEOFF2 (11-21): 返す bestmove の root 探索値を記録する (次の go の P2_SENTE_TABLE_SWITCH_CP の判定用)。
    //   表示 cp = info score cp と同じ換算 (p2_display_rescale_cp(to_cp(v)))。 未設定の値 (探索を飛ばした手など) は記録を消す。
    {
        const Value v  = bestThread->rootMoves[0].score;
        const bool  ok = v != -VALUE_INFINITE && v != VALUE_INFINITE && v != VALUE_NONE;
        const int   cp = !ok ? 0
                       : std::abs(v) >= VALUE_MATE_IN_MAX_PLY ? (v > 0 ? Eval::p2_prev_mate_cp() : -Eval::p2_prev_mate_cp())
                       : Eval::p2_display_rescale_cp(USIEngine::to_cp(v));
        Eval::p2_note_root_result(rootPos, ok, cp);
    }
#endif
#if defined(P2_SMP_EVALCHECK)
    P2SmpCheck::report(threads.size());
#endif
#if !STOCKFISH
	// 谺｡蝗槭↓謇狗分縺御ｻ雁屓縺ｨ逡ｰ縺ｪ繧九°繧呈､懷・縺吶ｋ縺溘ａ縺ｫgame_ply()繧剃ｿ晏ｭ倥＠縺ｦ縺翫￥縲・
    main_manager()->lastGamePly = rootPos.game_ply();
#endif

#ifdef UNCERT_LMR_MEASURE
    // Uncertainty-linked LMR 蛻・ｸ・ｨ域ｸｬ縺ｮ邏ｯ遨阪Ξ繝昴・繝・go 縺斐→縺ｫ蜃ｺ蜉帙｜ench 縺ｧ縺ｯ譛邨り｡・邱剰ｨ・縲・
    {
        auto&    m  = g_uncertMeasure;
        uint64_t vv = m.validVolNodes.load(), cq = m.candidateQuiet.load();
        uint64_t b0 = m.bin[0].load(), b1 = m.bin[1].load(), b2 = m.bin[2].load(),
                 b3 = m.bin[3].load(), b4 = m.bin[4].load(), b5 = m.bin[5].load();
        auto pct = [cq](uint64_t x) { return cq ? 100.0 * double(x) / double(cq) : 0.0; };
        std::ostringstream os;
        os << "UNCERTLMR validVolNodes=" << vv << " candQuiet=" << cq
           << " bin0[0-63]=" << b0 << "(" << pct(b0) << "%)"
           << " bin1[64-127]=" << b1 << "(" << pct(b1) << "%)"
           << " bin2[128-255]=" << b2 << "(" << pct(b2) << "%)"
           << " bin3[256-511]=" << b3 << "(" << pct(b3) << "%)"
           << " bin4[512-1023]=" << b4 << "(" << pct(b4) << "%)"
           << " bin5[1024+]=" << b5 << "(" << pct(b5) << "%)"
           << " >=64=" << (b1 + b2 + b3 + b4 + b5) << "(" << pct(b1 + b2 + b3 + b4 + b5) << "%)"
           << " >=128=" << (b2 + b3 + b4 + b5) << "(" << pct(b2 + b3 + b4 + b5) << "%)"
           << " >=256=" << (b3 + b4 + b5) << "(" << pct(b3 + b4 + b5) << "%)"
           << " >=512=" << (b4 + b5) << "(" << pct(b4 + b5) << "%)"
           << " >=1024=" << b5 << "(" << pct(b5) << "%)"
           << " fired=" << m.fired.load() << " firedFailHigh=" << m.firedFailHigh.load()
           << " firedReSearch=" << m.firedReSearch.load();
        sync_cout << "info string " << os.str() << sync_endl;
    }
#endif

#ifdef THREAT_STATS
    // Threat Continuation History Phase B 縺ｮ邏ｯ遨阪Ξ繝昴・繝・go 縺斐→縲｜ench 縺ｧ縺ｯ譛邨り｡・邱剰ｨ・縲・
    {
        auto&    T   = g_threatStats;
        uint64_t tri = T.nmNoCutoff.load(), adopt = T.adopt.load();
        auto pctT = [tri](uint64_t x) { return tri ? 100.0 * double(x) / double(tri) : 0.0; };
        std::ostringstream os;
        os << "THREATSTATS tries=" << tri << " adopt=" << adopt << "(" << pctT(adopt) << "%/tries)"
           << " cls[DropCheck]=" << T.cls[1].load() << " cls[Check]=" << T.cls[2].load()
           << " cls[Capture]=" << T.cls[3].load() << " cls[Promotion]=" << T.cls[4].load()
           << " cls[KingZone]=" << T.cls[5].load() << " cls[Other]=" << T.cls[6].load()
           << " posUpd=" << T.posUpd.load() << " negUpd=" << T.negUpd.load();
        sync_cout << "info string " << os.str() << sync_endl;

        // 繝・・繝悶Ν繝舌う繧｢繧ｹ襍ｰ譟ｻ(縺薙・ Worker 縺ｮ threatHistory 蜈ｨ entry)+ 譖ｴ譁ｰ index 陦晉ｪ∫紫縲・
        int64_t nz = 0, pos = 0, neg = 0, satHi = 0, satLo = 0, sumPos = 0, sumNeg = 0, usedSlots = 0;
        for (size_t i = 0; i < THREAT_HISTORY_SIZE; ++i)
        {
            int v = threatHistory[i];
            if (v != 0) { nz++; if (v > 0) { pos++; sumPos += v; } else { neg++; sumNeg += -v; } }
            if (v >= 8000) satHi++;
            else if (v <= -8000) satLo++;
            if (g_threatTag[i] != 0) usedSlots++;
        }
        uint64_t refs = T.updRefs.load(), col = T.updCollisions.load();
        std::ostringstream os2;
        os2 << "THREATBIAS updRefs=" << refs << " collisions=" << col << "("
            << (refs ? 100.0 * double(col) / double(refs) : 0.0) << "%)"
            << " usedSlots=" << usedSlots << "/" << THREAT_HISTORY_SIZE << "("
            << 100.0 * double(usedSlots) / double(THREAT_HISTORY_SIZE) << "%)"
            << " nonzero=" << nz << " pos=" << pos << " neg=" << neg << " satHi=" << satHi
            << " satLo=" << satLo << " avgAll=" << double(sumPos - sumNeg) / double(THREAT_HISTORY_SIZE)
            << " sumPos=" << sumPos << " sumNeg=" << sumNeg;
        sync_cout << "info string " << os2.str() << sync_endl;
    }
#endif

#if STOCKFISH
    // Send again PV info if we have a new best thread
    // 譁ｰ縺励＞繝吶せ繝医せ繝ｬ繝・ラ縺後≠繧後・縲∝・蠎ｦPV諠・ｱ繧帝∽ｿ｡縺吶ｋ
    if (bestThread != this)
        main_manager()->pv(*bestThread, threads, tt, bestThread->completedDepth);

    // ､・縺薙ｓ縺ｪ縺ｫPV蜃ｺ蜉帙☆繧九・螂ｽ縺阪§繧・↑縺・・縺ｧ逵∫払縲・
    //     縺溘□縺励∽ｸ蠎ｦ繧１V繧貞・蜉帙＠縺ｦ縺・↑縺・↑繧峨∝・蜉帙☆縺ｹ縺阪□縺ｨ諤昴≧縲・

#else
    // iterative_deepening()蜀・〒縲∵治逕ｨ縺吶ｋ譛邨１V繧偵☆縺ｧ縺ｫ蜃ｺ蜉帙＠縺ｦ縺・ｋ縺ｪ繧峨・
    // bestmove逶ｴ蜑阪↓蜷後§PV繧帝㍾隍・＠縺ｦ蜃ｺ蜉帙☆繧句ｿ・ｦ√・縺ｪ縺・・
    //
    // 縺溘□縺励￣V縺恵estmove 1謇九□縺代〒邨ゅｏ縺｣縺ｦ縺・ｋ蝣ｴ蜷医・
    // extract_ponder_from_tt() 縺檎ｽｮ謠幄｡ｨ繧аonder_candidate縺九ｉponder謇九ｒ諡ｾ縺｣縺ｦ縲・
    // PV繧・bestmove + pondermove 縺ｫ莨ｸ縺ｰ縺吶％縺ｨ縺後≠繧九・
    // 縺薙・蝣ｴ蜷医∵怙蠕後↓蜃ｺ蜉帶ｸ医∩縺ｮPV縺ｨ蜀・ｮｹ縺悟､峨ｏ繧九・縺ｧ縲「ciPvSent繧断alse縺ｫ謌ｻ縺励※
    // 荳九・PV蜃ｺ蜉帛・逅・↓蜈･繧峨○繧九・
    //
    // search_skipped縺ｮ蝣ｴ蜷医・縲∝ｮ夊ｷ｡繝ｻ螳｣險蜍昴■繝ｻ蜷域ｳ墓焔縺ｪ縺励↑縺ｩ縺ｧ騾壼ｸｸ謗｢邏｢繧偵＠縺ｦ縺翫ｉ縺壹・
    // rootMoves縺ｧ縺ｯ縺ｪ縺術robeResult縺九ｉ閾ｪ蜑阪〒PV繧呈ｧ狗ｯ峨☆繧九・縺ｧ縲√％縺ｮponder謚ｽ蜃ｺ縺ｯ陦後ｏ縺ｪ縺・・
    if (!search_skipped && bestThread->rootMoves[0].pv.size() == 1
        && bestThread->rootMoves[0].extract_ponder_from_tt(tt, rootPos,
                                                           main_manager()->ponder_candidate))
        uciPvSent = false;

    // bestmove繧定ｿ斐☆蜑阪↓迴ｾ蝨ｨ縺ｮPV繧貞・蜉帙☆繧区擅莉ｶ縺ｯ谺｡縺ｮ縺・★繧後°縲・
    //
    // 1. iterative_deepening()縺ｧ譛邨１V繧偵∪縺蜃ｺ蜉帙＠縺ｦ縺・↑縺・・
    // 2. Lazy SMP縺ｧ縲∵怙邨よ治逕ｨ縺吶ｋbestThread縺稽ain thread(this)縺ｧ縺ｯ縺ｪ縺・・
    //
    // 2. 縺ｮ蝣ｴ蜷医（terative_deepening()荳ｭ縺ｫ蜃ｺ蜉帶ｸ医∩縺縺｣縺溘→縺励※繧ゅ・
    // 縺昴ｌ縺ｯmain thread縺ｮPV縺ｪ縺ｮ縺ｧ縲∵治逕ｨ縺輔ｌ縺歟estThread縺ｮPV繧呈隼繧√※蜃ｺ蜉帙☆繧九・
    if (!uciPvSent || bestThread != this)
    {
        if (search_skipped)
        {
            // search_skipped縺ｮ縺ｨ縺阪・縲∬・蜑阪〒PV繧呈ｧ狗ｯ峨☆繧九・
            // 庁 縺薙・縺ｨ縺阪〉ootMoves縺ｮ諠・ｱ繧剃ｽｿ繧上↑縺・ｈ縺・↓縺励◆縺・・

            InfoFull info;
            info.depth     = 0;
            info.selDepth  = 0;
            info.multiPV   = 1;
            info.score     = probeResult.bestscore;
            TimePoint time = std::max(TimePoint(1), mainManager.tm.elapsed_time());
            info.timeMs    = time;
            info.nodes     = 0;
            info.nps       = 0;
            std::string pv = probeResult.bestmove.to_usi_string();
            if (probeResult.pondermove)
                pv += " " + probeResult.pondermove.to_usi_string();
            info.pv       = pv;
            info.hashfull = tt.hashfull();
            mainManager.updates.onUpdateFull(info);
        }
        else
        {
            main_manager()->pv(*bestThread, threads, tt, bestThread->completedDepth, true);
        }
    }

    // 決 謚穂ｺ・せ繧ｳ繧｢縺瑚ｨｭ螳壹＆繧後※縺・※縲∵ｭｩ縺ｮ萓｡蛟､繧・00縺ｨ縺励※豁｣隕丞喧縺励◆蛟､縺後◎繧後ｒ荳句屓繧九↑繧画兜莠・・
    //    縺溘□縺怜ｮ夊ｷ｡縺ｮ謖・＠謇九↓hit縺励◆蝣ｴ蜷医↑縺ｩ縺ｯrootMoves[0].score == -VALUE_INFINITE縺ｫ縺ｪ縺｣縺ｦ縺・ｋ縺ｮ縺ｧ縺昴ｌ縺ｯ髯､螟悶・
    auto resign_value = (int) options["ResignValue"];
    if (bestThread->rootMoves[0].score != -VALUE_INFINITE
        && USIEngine::to_cp(bestThread->rootMoves[0].score) <= -resign_value)
    {
        // 謗｢邏｢縺茎kip縺輔ｌ縺滓桶縺・↓縺励※縲〉esign繧堤ｩ阪・縲・
        search_skipped = true;
        probeResult    = Book::ProbeResult(Move::resign());
    };

#endif

    // 繝・ヰ繝・げ逕ｨ縺ｫ(繧ｮ繝ｪ繧ｮ繝ｪ縺ｾ縺ｧ諤晁・〒縺阪※縺・ｋ縺九ｒ遒ｺ隱阪☆繧九◆繧√↓)邨碁℃譎る俣繧貞・蜉帙＠縺ｦ縺ｿ繧九・
    /*
    auto& tm = main_manager()->tm;
    sync_cout << "info string elapsed time           = " << tm.elapsed_time() << "\n"
              << "info string elapsed_from_ponderhit = " << now() - tm.ponderhitTime << sync_endl;
	*/

    std::string ponder;

#if STOCKFISH
    if (bestThread->rootMoves[0].pv.size() > 1
        || bestThread->rootMoves[0].extract_ponder_from_tt(tt, rootPos))
        ponder = UCIEngine::move(bestThread->rootMoves[0].pv[1], rootPos.is_chess960());

    auto bestmove = UCIEngine::move(bestThread->rootMoves[0].pv[0], rootPos.is_chess960());
#else

	std::string bestmove;
    if (search_skipped)
    {
        bestmove = probeResult.bestmove.to_usi_string();
        if (probeResult.pondermove)
	        ponder   = probeResult.pondermove.to_usi_string();
	}
    else
    {
		// 決 extract_ponder_from_tt()縺ｫ
		//     ponder_candidate繧呈ｸ｡縺励※縲｝onder縺ｮ謖・＠謇九ｒ縺ｲ縺ｭ繧雁・縺吶・
        if (bestThread->rootMoves[0].pv.size() > 1
            || bestThread->rootMoves[0].extract_ponder_from_tt(tt, rootPos,
                                                               main_manager()->ponder_candidate))
            ponder = USIEngine::move(bestThread->rootMoves[0].pv[1]);

        // ---- 定跡回避 (BookExit)。 2026-10-06。 **root でだけ指す手を選び直す**
        //   既定 (BookExitMargin=0 / ファイル無し) では 1 命令も挙動を変えない。
        //   🚨score は Value 単位。 option も Value 単位 (cp との換算は net 依存なので混同を避ける)。
        Move be_pick = bestThread->rootMoves[0].pv[0];
        if (BookExit::g_margin > 0 && !BookExit::g_set.empty()
            && rootPos.game_ply() <= BookExit::g_max_ply && BookExit::in_set(rootPos))
        {
            const Value bestScore = bestThread->rootMoves[0].score;
            Value       bestOut   = -VALUE_INFINITE;
            Move        pickOut   = Move::none();
            StateInfo   si;
            for (const auto& rm : bestThread->rootMoves)
            {
                if (rm.score == -VALUE_INFINITE) continue;          // 読んでいない手は使わない
                if (bestScore - rm.score > (Value) BookExit::g_margin) continue;  // 譲りすぎ
                Position& p = const_cast<Position&>(rootPos);
                p.do_move(rm.pv[0], si);
                const bool still_in = BookExit::in_set(p);
                p.undo_move(rm.pv[0]);
                if (still_in) continue;                             // 外れていない
                if (rm.score > bestOut) { bestOut = rm.score; pickOut = rm.pv[0]; }
            }
            if (pickOut != Move::none() && pickOut != be_pick)
            {
                sync_cout << "info string BookExit: 定跡から外れる "
                          << USIEngine::move(be_pick) << " -> " << USIEngine::move(pickOut)
                          << " (譲った score " << (int)(bestScore - bestOut) << ")" << sync_endl;
                be_pick = pickOut;
            }
        }
#if 0   // 🏁K★ (RootStyle) 2026-10-28 棄却: MultiPV=1 では非 PV の score が -∞ で発火 0
        // ---- 🎯K★ 片側項を root だけで効かせる (RootStyleMargin)。 2026-10-25。
        //   最善から Margin 以内の手のうち**様式量が最大の手**を選ぶ。
        //   🎯fail-soft の bound は真値の上界なので、 損は **1 手あたり Margin 以下**。
        //   🚨既定 (0) では 1 比較で抜ける = bit-exact。
        if (Eval::P2NNUE::g_rootstyle_margin > 0)
        {
            const Value bestScore = bestThread->rootMoves[0].score;
            int         bestCnt   = -1;
            Value       bestCntScore = -VALUE_INFINITE;
            Move        pickSty  = Move::none();
            StateInfo   si2;
            for (const auto& rm : bestThread->rootMoves)
            {
                if (rm.score == -VALUE_INFINITE) continue;   // 読んでいない手は使わない
                if (bestScore - rm.score > (Value) Eval::P2NNUE::g_rootstyle_margin) continue;
                Position& p = const_cast<Position&>(rootPos);
                p.do_move(rm.pv[0], si2);
                const int cnt = Eval::P2NNUE::rootstyle_count(p);
                p.undo_move(rm.pv[0]);
                // 🚨同じ枚数なら**元の順 (= score の良い順)** を保つため > で比べる
                if (cnt > bestCnt || (cnt == bestCnt && rm.score > bestCntScore))
                {
                    bestCnt = cnt; bestCntScore = rm.score; pickSty = rm.pv[0];
                }
            }
            if (pickSty != Move::none() && pickSty != be_pick)
            {
                sync_cout << "info string RootStyle: " << USIEngine::move(be_pick)
                          << " -> " << USIEngine::move(pickSty)
                          << " (kind " << Eval::P2NNUE::g_rootstyle_kind
                          << " count " << bestCnt
                          << " / 譲った score " << (int)(bestScore - bestCntScore) << ")" << sync_endl;
                be_pick = pickSty;
            }
        }
#endif  // 🏁K★ (RootStyle)
        bestmove = USIEngine::move(be_pick);


    }

	/*
		涛
			USI繝励Ο繝医さ繝ｫ縺ｫ縺翫＞縺ｦ縲｜estmove縺ｮ譎ゅ↓霑斐☆ponder縺ｯ縲∝ｸｸ縺ｫ霑斐＠縺ｦ蝠城｡後↑縺・・

			繧ｨ繝ｳ繧ｸ繝ｳ繧ｪ繝励す繝ｧ繝ｳ縺ｮ"USI_Ponder"縺荊rue縺ｧ縺ゅｌ縺ｰ縲；UI縺ｯ谺｡縺ｫ逶ｸ謇九・謇狗分縺ｧ"go ponder"繧帝∽ｿ｡縺励※
			諤晁・お繝ｳ繧ｸ繝ｳ縺ｫponder縺ｧ諤晁・＆縺帙ｋ縲・

			縺､縺ｾ繧翫・USI_Ponder"縺ｯ諤晁・お繝ｳ繧ｸ繝ｳ縺ｮ縺溘ａ縺ｧ縺ｯ縺ｪ縺上；UI蛛ｴ縺ｮ縺溘ａ縺ｮ繧ｪ繝励す繝ｧ繝ｳ縺ｧ縺ゅｋ縲・
	*/

#endif

#if defined(EVAL_P2NNUE)
    // 非同期 df-pn が詰みを見つけていたら bestmove をその手で上書きする。
    // ★上書き前に root 局面での合法性を必ず検証する。
    //   - dfpn スレッドは別 Position で探索しているので、Move 上位 16bit(移動後の駒)は
    //     そのまま信用せず、下位 16bit から rootPos.to_move() で組み直す。
    //   - pseudo_legal_s<true>() + legal() を通らなければ上書きしない。
    // ponder は付けない(詰み手のあとの相手の応手を読んでいないため)。
    {
        const uint32_t mateRaw = mateAsync.mate_move_raw();
        if (mateRaw != 0)
        {
            const Move mateMove = rootPos.to_move(Move(mateRaw).to_move16());
            if (mateMove.is_ok() && rootPos.pseudo_legal_s<true>(mateMove)
                && rootPos.legal(mateMove))
            {
                bestmove = USIEngine::move(mateMove);
                ponder.clear();
                sync_cout << "info string mate solver async: bestmove overridden by "
                          << bestmove << sync_endl;
            }
        }
    }
#endif

    if (P2Stat::enabled()) P2Stat::dump();
#ifdef P2_LMR_STATS
    P2LmrStat::dump();
#endif
#ifdef P2_PRUNE_STATS
    P2PruneStat::dump();
#endif
    // M1 遅延 (D2): 機構が **実際に発火したか** を直接確かめる。  2026-11-02
    //   🚨nodes/time の CI は「効果がない」と「配線されていない」を区別できない。
    //     defer=0 なら後者。 既定では SMP_DEFER_MIN_DEPTH=0 なので何も出ない。
    if (SPSA::SMP_DEFER_MIN_DEPTH > 0)
    {
        u64 probe = 0, defer = 0, read = 0;
        for (auto&& th : threads)
        {
            auto* w = static_cast<Search::YaneuraOuWorker*>(th->worker.get());
            probe += w->smpBusyProbe;
            defer += w->smpDeferCnt;
            read  += w->smpDeferRead;
        }
        sync_cout << "info string SMP_INFO probe=" << probe << " defer=" << defer
                  << " read=" << read << " rate="
                  << (probe ? double(defer) / double(probe) : -1.0) << sync_endl;
    }

    // M3 根の YBWC (配線と発火を直接数える / ★abdada_firing_rate_is_structural)
    //   🚨taken=0 なら helper が 1 手も取れていない = 配線が切れている。
    //   🎯wait_ms は helper が長兄を待った時間。 Amdahl の直列部分の実測値。
    if (rootSplit)
        sync_cout << "info string YBWC_INFO gen=" << ybwcGen
                  << " taken=" << rootSplit->cntTaken.load()
                  << " done=" << rootSplit->cntDone.load()
                  << " giveback=" << rootSplit->cntGiveBack.load()
                  << " import=" << rootSplit->cntImport.load()
                  << " wait_ms=" << (rootSplit->cntWaitUs.load() / 1000) << sync_endl;

    // M4 仕事の委譲 (配線と発火を直接数える)
    //   🚨claim=0 なら helper が 1 件も取れていない = 配線が切れている。
    //   🎯full は枠が足りなかった回数、 idle は helper の空振り。 両方が大きいなら
    //     「提示が間に合っていない」= 委譲の粒度が細かすぎる。
    if (deleg)
        sync_cout << "info string DELEG_INFO offer=" << deleg->nOffer.load()
                  << " full=" << deleg->nFull.load()
                  << " claim=" << deleg->nClaim.load()
                  << " finish=" << deleg->nFinish.load()
                  << " idle=" << deleg->nIdle.load() << sync_endl;

    main_manager()->updates.onBestmove(bestmove, ponder);
}

// Main iterative deepening loop. It calls search()
// repeatedly with increasing depth until the allocated thinking time has been
// consumed, the user stops the search, or the maximum search depth is reached.

// 繝｡繧､繝ｳ縺ｮ蜿榊ｾｩ豺ｱ蛹悶Ν繝ｼ繝励Ｔearch() 繧堤ｹｰ繧願ｿ斐＠蜻ｼ縺ｳ蜃ｺ縺励・
// 險ｭ螳壹＆繧後◆諤晁・凾髢薙ｒ菴ｿ縺・・繧九°縲√Θ繝ｼ繧ｶ繝ｼ縺梧爾邏｢繧貞●豁｢縺吶ｋ縺九・
// 譛螟ｧ謗｢邏｢豺ｱ蠎ｦ縺ｫ蛻ｰ驕斐☆繧九∪縺ｧ蜃ｦ逅・ｒ邯壹￠繧九・

// 統 謗｢邏｢譛ｬ菴薙ゆｸｦ蛻怜喧縺励※縺・ｋ蝣ｴ蜷医√％縺薙′蜷дhread縺ｮ謗｢邏｢縺ｮentry point縲・
//     Lazy SMP縺ｪ縺ｮ縺ｧ縲∫ｽｮ謠幄｡ｨ繧貞・譛峨＠縺ｪ縺後ｉ縺昴ｌ縺槭ｌ縺ｮ繧ｹ繝ｬ繝・ラ縺悟享謇九↓謗｢邏｢縺励※縺・ｋ縺縺代・

// ============================================================================
//  M3 根の YBWC                                                     2026-11-02
// ============================================================================

// **自分の rootMoves を master の手順に揃える** (不変条件 I-2)。
//   各 thread は自分の rootMoves を自分のスコアで並べ替えるので、 放っておくと
//   「3 番目の手」が thread ごとに別物になり、 添字で引く共有表が壊れる。
//   O(n) で済ませるため、 master の手順を走査しながら前から詰めていく。
void Search::YaneuraOuWorker::ybwc_reorder_root_moves() {
    const int n = std::min<int>(rootSplit->n_moves(), int(rootMoves.size()));
    for (int i = 0; i < n; ++i)
    {
        const Move want = rootSplit->move_at(i);
        if (rootMoves[i].pv[0] == want)
            continue;
        for (int j = i + 1; j < int(rootMoves.size()); ++j)
            if (rootMoves[j].pv[0] == want)
            {
                std::swap(rootMoves[i], rootMoves[j]);
                break;
            }
    }
}

// ============================================================================
//  M4 木の内部での仕事の委譲                                        2026-11-02
// ============================================================================

// **委譲された 1 手を読む。** root からの手順を再生して節まで降り、 その手を
//   master が渡した窓・深さで読む。 **返り値は使わない** — 目的は置換表を温めることで、
//   master が末尾で同じ手を読むときに当たればよい (source/smp_deleg.h)。
//   🚨盤の再生に失敗したら (手順が合法でない = 盤がずれている) 何もせずに戻る。
//     master は必ず自分で読むので、 ここで諦めても探索結果は変わらない。
void Search::YaneuraOuWorker::ybwc_run_deleg(const DelegSlot& d, Stack* ss) {
    if (d.pathLen < 0 || d.pathLen > DelegSlot::PATH_CAP || d.move == Move::none())
        return;

    Position& pos    = rootPos;
    int       pushed = 0;
    curPathLen       = 0;   // helper は root から再生する

    // root からの手順を再生する。 do_move が ss->currentMove と
    //   continuationHistory を設定してくれるので、 文脈はこれで復元される。
    for (int j = 0; j < d.pathLen; ++j)
    {
        const Move m = d.path[j];
        if (!m.is_ok() || !pos.pseudo_legal_s<true>(m) || !pos.legal(m))
            break;
        (ss + j)->inCheck = pos.in_check();
        do_move(pos, m, delegSt[j], (ss + j));
        ++pushed;
    }

    // 🚨切り分け用の段階スイッチ (環境変数 P2_YBWC_STAGE)。
    //   1 = 手順の再生だけ (探索を呼ばない) / 2 以上 (既定) = 再生 + 探索。
    //   並行機構は落ちたときに「どの段で落ちたか」が分からないと直せないので残す。
    static const int stage = [] {
        const char* e = std::getenv("P2_YBWC_STAGE");
        return e ? std::atoi(e) : 2;
    }();

    if (stage >= 2 && pushed == d.pathLen && d.move.is_ok()
        && pos.pseudo_legal_s<true>(d.move) && pos.legal(d.move))
    {
        // 深さの文脈を master に合わせる (ss->ply > rootDepth 等の式で使われる)
        rootDepth              = d.pathLen + std::max(d.depth, 1);
        // 🚨`reduction()` は rootDelta で割る (:7224)。 専従 helper は aspiration ループを
        //   通らないので 0 のままで、 設定を忘れるとゼロ除算で即死する。
        rootDelta              = std::max(d.rootDelta, 1);
#if defined(P2_SRCH_BUNDLE_A)
        nonPvDeltaTerm = SPSA::LMR_REDUCTION_DELTA_MULT / rootDelta;
#endif
        (ss + pushed)->inCheck = pos.in_check();
        do_move(pos, d.move, delegSt[pushed], (ss + pushed));

        // 🚨呼び出し側が用意しなければならない子節の項目を埋める。
        //   `pv` は PV 節で `ss->pv->update(...)` (:6227) に参照されるので、
        //   **nullptr を入れて NonPV だけで読む**。 2026-11-02 にここを忘れて
        //   ACCESS_VIOLATION で死んだ。
        //   `excludedMove` は SE の再帰で呼び出し側が入れる項目なので明示的に空にする
        //   (古い値が残っていると helper が勝手に除外探索を始める)。
        Stack* const hs = ss + pushed + 1;
        hs->pv          = nullptr;
        hs->excludedMove = Move::none();

        const Value a = Value(d.alpha);
        // 🎯**null window の NonPV だけで読む**。 目的は置換表を温めることなので、
        //   全窓の読み直しは master に任せる (PV 節の pv 配線を helper に持ち込まない)。
        Value v = -search<NonPV>(pos, hs, -(a + 1), -a, d.depth, !d.cutNode);
        (void) v;   // 🎯値は捨てる。 置換表に残るのが目的

        undo_move(pos, d.move);
    }

    // 再生した分だけ正確に巻き戻す
    for (int j = pushed - 1; j >= 0; --j)
        undo_move(pos, d.path[j]);
}

// **helper が読んだ結果を master の rootMoves に回収する**。
//   単スレッドと同じ規則を master の手順で前から当てる:
//     値が今の最善を上回れば PV 手として score/uciScore/pv を書き、 最善を更新。
//     上回らなければ score = -VALUE_INFINITE (単スレッドが PV 以外に与える値と同じ)。
//   🚨自分が読んだ手には当てない (ybwcMine)。 二重に当てると自分の最善を壊す。
//   🚨読めなかった手 (DONE でない) は -VALUE_INFINITE のままにする。 長兄は master が
//     必ず読んでいるので最善手は必ず存在する = **正しさは helper に依存しない**。
Value Search::YaneuraOuWorker::ybwc_import(Value bestValue) {
    const int n = std::min<int>(rootSplit->n_moves(), int(rootMoves.size()));
    for (int i = 1; i < n; ++i)
    {
        if (ybwcMine[i])
            continue;
        RootMove& rm = rootMoves[i];
        if (!rootSplit->is_done(ybwcGen, i))
        {
            rm.score = -VALUE_INFINITE;
            continue;
        }
        const Value v = rootSplit->score_of(i);
        rootSplit->cntImport.fetch_add(1, std::memory_order_relaxed);
        rm.averageScore =
          rm.averageScore != -VALUE_INFINITE ? (v + rm.averageScore) / 2 : v;
        rm.meanSquaredScore = rm.meanSquaredScore != -VALUE_INFINITE * VALUE_INFINITE
                              ? (v * std::abs(v) + rm.meanSquaredScore) / 2
                              : v * std::abs(v);
        if (v > bestValue)
        {
            bestValue          = v;
            rm.score           = rm.uciScore = v;
            rm.scoreLowerbound = rm.scoreUpperbound = false;
            rm.pv.resize(1);
            const int len = rootSplit->pv_len(i);
            for (int j = 1; j < len; ++j)
                rm.pv.push_back(rootSplit->pv_at(i, j));
        }
        else
            rm.score = -VALUE_INFINITE;
    }
    return bestValue;
}

// **helper の反復**。 自分で深さを進めず、 master が publish した世代に従う (I-1)。
//   ① 新しい世代を待つ
//   ② 長兄が終わるのを待つ (I-4。 ここが Amdahl の直列部分)
//   ③ master の手順・深さ・窓で search<Root> を呼ぶ。 兄弟手の取得は手ループの中
//   🚨helper の rootMoves は報告に使わない (get_best_thread が master を返す)。
void Search::YaneuraOuWorker::ybwc_helper_loop(Stack* ss) {
    int       lastGen = 0;
    DelegSlot job;
    while (!threads.stop)
    {
        // --- M4: まず委譲された仕事を探す (こちらが本命)
        if (deleg)
        {
            std::uint32_t jg = 0;
            const int     sl = deleg->claim(job, jg);
            if (sl >= 0)
            {
                ybwc_run_deleg(job, ss);
                deleg->finish(sl, jg);
                continue;
            }
            deleg->nIdle.fetch_add(1, std::memory_order_relaxed);
            if (!rootSplit)
            {
                std::this_thread::yield();
                continue;
            }
        }
        if (!rootSplit)
            break;

        const int g = rootSplit->gen();
        if (g == 0 || g == lastGen)
        {
            std::this_thread::yield();
            continue;
        }

        // ② 長兄を待つ。 待った時間は計測する (天井の実測値になる)
        const TimePoint t0 = now();
        while (!rootSplit->eldest_done(g) && !threads.stop && rootSplit->gen() == g)
            std::this_thread::yield();
        rootSplit->cntWaitUs.fetch_add(std::uint64_t(std::max<TimePoint>(now() - t0, 0)) * 1000,
                                       std::memory_order_relaxed);
        if (threads.stop || rootSplit->gen() != g)
            continue;

        lastGen = g;
        ybwcGen = g;
        ybwcIdx = -1;
        ybwc_reorder_root_moves();

        rootDepth = rootSplit->depth();
        pvIdx     = rootSplit->pv_idx();
        Value a   = rootSplit->alpha();
        Value b   = rootSplit->beta();
        rootDelta = b - a;
#if defined(P2_SRCH_BUNDLE_A)
        nonPvDeltaTerm = SPSA::LMR_REDUCTION_DELTA_MULT / rootDelta;
#endif
        selDepth  = 0;

        search<Root>(rootPos, ss, a, b, rootDepth, false);

        // 取ったまま読めなかった手があれば必ず返す (I-5)
        if (ybwcIdx >= 0)
        {
            rootSplit->give_back(g, ybwcIdx);
            rootSplit->cntGiveBack.fetch_add(1, std::memory_order_relaxed);
            ybwcIdx = -1;
        }
        completedDepth = rootDepth;
    }
    ybwcGen = 0;
}

bool Search::YaneuraOuWorker::iterative_deepening() {

    // 繧ゅ＠閾ｪ蛻・′繝｡繧､繝ｳ繧ｹ繝ｬ繝・ラ縺ｧ縺ゅｋ縺ｪ繧盈ainThread縺ｫmain_manager縺ｮ繝昴う繝ｳ繧ｿ繧剃ｻ｣蜈･縲・
    // 閾ｪ蛻・′繧ｵ繝悶せ繝ｬ繝・ラ縺ｮ縺ｨ縺阪・縲√％繧後・nullptr縺ｫ縺ｪ繧九・
    SearchManager* mainThread = (is_mainthread() ? main_manager() : nullptr);

#if !STOCKFISH
    // 繧・・縺・ｉ邇九〒縺ｯ謗｢邏｢繧ｪ繝励す繝ｧ繝ｳ縺ｯ縲［ain_manager縺梧戟縺｣縺ｦ縺・ｋ縲・
    SearchOptions& search_options = main_manager()->search_options;

    // 蜷Цorker縺ｮPosition::set_ekr()繧貞他縺ｳ蜃ｺ縺励※蜈･邇峨Ν繝ｼ繝ｫ繧貞渚譏縺輔○繧句ｿ・ｦ√′縺ゅｋ縲・
    rootPos.set_ekr(search_options.enteringKingRule);
#endif

    PVMoves pv;

    // 謗｢邏｢縺ｮ螳牙ｮ壽ｧ繧定ｩ穂ｾ｡縺吶ｋ縺溘ａ縺ｫ蜑榊屓縺ｮiteration譎ゅ・best PV繧定ｨ倬鹸縺励※縺翫￥縲・
    Depth lastBestMoveDepth = 0;
    Value lastBestScore     = -VALUE_INFINITE;
    PVMoves lastBestPV;
    lastBestPV.push_back(Move::none());

    // alpha,beta         : aspiration search縺ｮ遯薙・遽・峇(alpha,beta)
    // delta              : apritation search縺ｧ遯薙ｒ蜍輔°縺吝､ｧ縺阪＆delta
    // us                 : 縺薙・螻髱｢縺ｮ謇狗分蛛ｴ
    // timeReduction      : 隱ｭ縺ｿ遲九′螳牙ｮ壹＠縺ｦ縺・ｋ縺ｨ縺阪↓譎る俣繧堤洒邵ｮ縺吶ｋ縺溘ａ縺ｮ菫よ焚縲・
    // totBestMoveChanges : 逶ｴ霑代〒bestMove縺悟､牙喧縺励◆蝗樊焚縺ｮ邨ｱ險医りｪｭ縺ｿ遲九・螳牙ｮ壼ｺｦ縺ｮ逶ｮ螳峨↓縺吶ｋ縲・
    // iterIdx            : 蜿榊ｾｩ豺ｱ蛹悶・譎ゅ↓1蝗槭＃縺ｨ縺ｮbest value繧剃ｿ晏ｭ倥☆繧九◆繧√・驟榊・縺ｸ縺ｮindex縲・縺九ｉ3縺ｾ縺ｧ縺ｮ蛟､繧偵→繧九・
    Value  alpha, beta;
    Value  bestValue     = -VALUE_INFINITE;
    Color  us            = rootPos.side_to_move();
#if defined(P2_SMP_EVALCHECK)
    P2SmpCheck::batch(0, threadIdx);
#endif
#if defined(P2_OPTIMISM)
    // P2_OPTIMISM: 静的な非対称だけで始める (dyn は各反復の開始時に足す)。
    p2Opt[us]  = Value(g_p2opt_us);
    p2Opt[~us] = Value(g_p2opt_them);
#endif
    double timeReduction = 1, totBestMoveChanges = 0;
    int    delta, iterIdx                        = 0;

    // 統 Stockfish 14縺ｮ鬆・・縲∝渚蠕ｩ豺ｱ蛹悶・iteration縺梧ｵ・＞縺・■縺ｯaspiration search繧剃ｽｿ繧上★
    //     謗｢邏｢遯薙ｒ (-VALUE_INFINITE , +VALUE_INFINITE)縺ｨ縺励※縺・◆縺後ヾtockfish 16縺ｧ縺ｯ縲・
    //     豬・＞縺・■縺九ｉaspiration search繧剃ｽｿ縺・ｈ縺・↓縺ｪ縺｣縺溘・縺ｧ縲∥lpha,beta縺ｮ蛻晄悄蛹悶・縺薙％縺ｧ繧・ｉ縺ｪ縺上↑縺｣縺溘・

    // Allocate stack with extra size to allow access from (ss - 7) to (ss + 2):
    // (ss - 7) is needed for update_continuation_histories(ss - 1) which accesses (ss - 6),
    // (ss + 2) is needed for initialization of cutOffCnt.

    // (ss-7)縺九ｉ(ss+2)縺ｸ縺ｮ繧｢繧ｯ繧ｻ繧ｹ繧定ｨｱ蜿ｯ縺吶ｋ縺溘ａ縺ｫ縲∬ｿｽ蜉縺ｮ繧ｵ繧､繧ｺ縺ｧ繧ｹ繧ｿ繝・け繧貞牡繧雁ｽ薙※縺ｾ縺・
    // (ss-7)縺ｯupdate_continuation_histories(ss-1, ...)縺ｮ縺溘ａ縺ｫ蠢・ｦ√〒縺ゅｊ縲√％繧後・(ss-6)縺ｫ繧｢繧ｯ繧ｻ繧ｹ縺励∪縺・
    // (ss+2)縺ｯstatScore縺ｨkillers縺ｮ蛻晄悄蛹悶・縺溘ａ縺ｫ蠢・ｦ√〒縺・

    // 庁 continuationHistory縺ｮ縺溘ａ縲・ss-7)縺九ｉ(ss+2)縺ｾ縺ｧ縺ｫ繧｢繧ｯ繧ｻ繧ｹ縺励◆縺・・縺ｧ菴吝・縺ｫ遒ｺ菫昴＠縺ｦ縺翫￥縲・

    // 統 stack縺ｯ縲∝・鬆ｭ10蛟九ｒ蛻晄悄蛹悶＠縺ｦ縺翫￠縺ｰ蜊∝・縲・
    //     縺昴・縺ゅ→縺ｯsearch()縺ｮ蜈磯ｭ縺ｧss+1,ss+2繧帝←螳懷・譛溷喧縺励※縺・￥縲・
    //     RootNode縺ｯss->ply == 0縺後◎縺ｮ譚｡莉ｶ縲・ss->ply縺ｯsearch_thread_init縺ｧ蛻晄悄蛹悶＆繧後※縺・ｋ)

    Stack  stack[MAX_PLY + 10] = {};
    Stack* ss                  = stack + 7;

    // counterMoves繧地ullptr縺ｫ蛻晄悄蛹悶☆繧九・縺ｧ縺ｯ縺ｪ縺蒐O_PIECE縺ｮ縺ｨ縺阪・蛟､繧堤分蜈ｵ縺ｨ縺励※逕ｨ縺・ｋ縲・
    for (int i = 7; i > 0; --i)
    {
        (ss - i)->continuationHistory =
          &contHistRoot[0][0][NO_PIECE][0];  // Use as a sentinel

        (ss - i)->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];
        (ss - i)->staticEval                    = VALUE_NONE;
    }

    // Stack(謗｢邏｢逕ｨ縺ｮ讒矩菴・荳翫・ply(謇区焚)縺ｯ莠句燕縺ｫ蛻晄悄蛹悶＠縺ｦ縺翫￠縺ｰ謗｢邏｢譎ゅ↓莉｣蜈･縺吶ｋ蠢・ｦ√′縺ｪ縺・・
    for (int i = 0; i <= MAX_PLY + 2; ++i)
        (ss + i)->ply = i;

    // 譛蝟・ｿ懈焔蛻・Principal Variation)
    ss->pv = &pv;

    if (mainThread)
    {
#if !STOCKFISH
		// 決 Stochastic Ponder縺ｧ縺ｯ蜑榊屓縺ｨ蛻･謇狗分縺ｫ縺ｪ繧九・縺ｧ縲√％縺ｮ縺ｨ縺阪・
		//     bestPreviousScore縺ｨpreviousAverageScore繧貞渚霆｢縺輔○繧句ｿ・ｦ√′縺ゅｋ縲・
		if ((mainThread->lastGamePly - rootPos.game_ply()) & 1)
		{
            if (mainThread->bestPreviousScore != VALUE_INFINITE)
	            mainThread->bestPreviousScore *= -1;

            if (mainThread->bestPreviousAverageScore != VALUE_INFINITE)
	            mainThread->bestPreviousAverageScore *= -1;
		}
#endif

        if (mainThread->bestPreviousScore == VALUE_INFINITE)
            mainThread->iterValue.fill(VALUE_ZERO);
        else
            mainThread->iterValue.fill(mainThread->bestPreviousScore);
    }

    // MultiPV
    // 庁 bestmove縺ｨ縺励※縺励％縺ｮ螻髱｢縺ｮ荳贋ｽ康蛟九ｒ謗｢邏｢縺吶ｋ讖溯・

    size_t multiPV = size_t(options["MultiPV"]);

#if STOCKFISH
    Skill skill(options["Skill Level"], options["UCI_LimitStrength"] ? int(options["UCI_Elo"]) : 0);
    // ､・竊代％繧後〒繧ｨ繝ｳ繧ｸ繝ｳ繧ｪ繝励す繝ｧ繝ｳ2縺､繧ょ｢励∴繧九・繧・□縺ｪ窶ｦ縲よｰ励′蜷代＞縺溘ｉ繧ｵ繝昴・繝医☆繧九％縺ｨ縺ｫ縺吶ｋ縲・
#else
    Skill skill = Skill(20, 0);
#endif

    // When playing with strength handicap enable MultiPV search that we will
    // use behind-the-scenes to retrieve a set of possible moves.

    // 蠑ｷ縺輔ワ繝ｳ繝・ぅ繧ｭ繝｣繝・・莉倥″縺ｧ繝励Ξ繧､縺吶ｋ縺ｨ縺阪・縲｀ultiPV謗｢邏｢繧呈怏蜉ｹ縺ｫ縺吶ｋ縲・
    // 縺薙ｌ縺ｫ繧医ｊ縲∬｣丞・縺ｧ隍・焚縺ｮ蛟呵｣懈焔繧貞叙蠕励〒縺阪ｋ繧医≧縺ｫ縺吶ｋ縲・

    // 統 SkillLevel縺梧怏蜉ｹ(險ｭ螳壹＆繧後◆蛟､縺・0譛ｪ貅)縺ｮ縺ｨ縺阪・縲｀ultiPV = 4縺ｧ謗｢邏｢縲・

    if (skill.enabled())
        multiPV = std::max(multiPV, size_t(4));

    // 庁 multiPV縺ｮ蛟､縺ｯ縲√％縺ｮ螻髱｢縺ｧ縺ｮ蜷域ｳ墓焔縺ｮ謨ｰ繧剃ｸ雁屓縺｣縺ｦ縺ｯ縺ｪ繧峨↑縺・・
    multiPV = std::min(multiPV, rootMoves.size());

    // -----------------------
    // M1 遅延 (ABDADA-lite) を使うかをここで 1 回決める   2026-11-02
    // -----------------------
    //   不変条件 I-5: 既定 (SMP_DEFER_MIN_DEPTH=0) / Threads=1 / MultiPV>1 では
    //     ポインタを nullptr にして **1 命令も実行しない** (= bit-exact)。
    //   🚨MultiPV>1 を外すのは pvIdx と後回しの moveCount の整合を考えたくないため。
    //   設計: runs/prereg/DESIGN-20261102-SMP-DEFER.md §5.3 (A)
    smpBusy = (SPSA::SMP_DEFER_MIN_DEPTH > 0 && threads.size() > 1 && multiPV == 1)
                ? &engine.smpBusy
                : nullptr;
    // D2 のカウンタは go ごとに 0 から数える (worker 自身の値なので排他不要)
    smpBusyProbe = smpDeferCnt = smpDeferRead = 0;

    // -----------------------
    // M3 根の YBWC を使うかをここで 1 回決める   2026-11-02
    // -----------------------
    //   不変条件 I-6: 既定 (ROOT_YBWC=0) / Threads=1 / MultiPV>1 / searchmoves 指定時は
    //     ポインタを nullptr にして **1 命令も実行しない** (= bit-exact)。
    //   🚨searchmoves を外すのは、 thread ごとに rootMoves の中身が変わる可能性を
    //     排除するため (共有表を添字で引くので、 全 thread が同じ手集合でなければならない)。
    //   🚨M1 (遅延) とは**排他**。 phase2 の再読み込みで同じ添字を取り直すと
    //     M1 の不変条件 I-1 (後回しの手は必ず読む) が壊れるため、 YBWC が勝つ。
    rootSplit = (SPSA::ROOT_YBWC && threads.size() > 1 && multiPV == 1
                 && limits.searchmoves.empty())
                  ? &engine.rootSplit
                  : nullptr;
    if (rootSplit)
        smpBusy = nullptr;
    ybwcGen = 0;
    ybwcIdx = -1;

    // -----------------------
    // M4 木の内部での仕事の委譲   2026-11-02
    // -----------------------
    //   不変条件 I-5: 既定 (YBWC_MIN_DEPTH=0) / Threads=1 では nullptr = 1 命令も実行しない。
    //   🚨M1 (遅延) とは排他。 後回しの枠 (deferred[]) を共有するので、 両方が同じ枠に
    //     押すと M1 の不変条件 I-1 (後回しは必ず読む) の管理が二重になる。
    deleg = (SPSA::YBWC_MIN_DEPTH > 0 && threads.size() > 1) ? &engine.deleg : nullptr;
    if (deleg)
    {
        smpBusy = nullptr;
        if (mainThread)
            deleg->clear_counters();
    }
    if (rootSplit && mainThread)
        rootSplit->clear_counters();

    // ---------------------
    //   蜿榊ｾｩ豺ｱ蛹悶・繝ｫ繝ｼ繝・
    // ---------------------

    // 蜿榊ｾｩ豺ｱ蛹悶・謗｢邏｢豺ｱ縺輔′豺ｱ縺上↑縺｣縺ｦ陦後▲縺ｦ縺・ｋ縺九・繝√ぉ繝・け逕ｨ縺ｮ繧ｫ繧ｦ繝ｳ繧ｿ繝ｼ
    // 縺薙ｌ縺悟｢励∴縺ｦ縺・↑縺・凾縲∝酔縺俶ｷｱ縺輔ｒ蜀榊ｺｦ謗｢邏｢縺励※縺・ｋ縺薙→縺ｫ縺ｪ繧九・fail high縺礼ｶ壹￠縺ｦ縺・ｋ)
    // 庁 縺ゅ∪繧雁酔縺俶ｷｱ縺輔〒縺､縺｣縺九∴縺ｦ縺・ｋ譎ゅ・縲∥spiration window縺ｮ蟷・ｒ螟ｧ縺阪￥縺励※繧・ｋ縺ｪ縺ｩ縺励※蝗樣∩縺吶ｋ蠢・ｦ√′縺ゅｋ縲・
    int searchAgainCounter = 0;

    // 蜿榊ｾｩ豺ｱ蛹門・縺ｧ縲∫樟蝨ｨ縺ｮiteration縺ｮ譛邨１V繧竪UI縺ｸ蜃ｺ蜉帶ｸ医∩縺九・
    bool uciPvSent = false;

    // 庁 lowPlyHistory縺ｯ縲∬ｩｦ蜷磯幕蟋区凾縺ｫ1蝗槭□縺代〒縺ｯ縺ｪ縺上・go"縺ｮ蠎ｦ縺ｫ蛻晄悄蛹悶＠縺溘⊇縺・′蠑ｷ縺・・
    lowPlyHistory.fill(SPSA::LOW_PLY_HIST_INIT);

    // ★SF port #4 (SF search.cpp:327-329, commit ced9f69834 → 現行は ×729/1024):
    //   "go" ごとに mainHistory の全要素を × SF_MAINHIST_DECAY_NUM / 1024 に減衰させる
    //   (前局面の履歴を弱めて持ち越す)。探索開始時に 1 回 (2×65536 要素) なので NPS には無関係。
    //   既定 OFF (SPSA::SF_MAINHIST_DECAY=false) なら何もしない = bit-exact。
    if (SPSA::SF_MAINHIST_DECAY)
        for (Color c : {BLACK, WHITE})
            for (int i = 0; i < UINT_16_HISTORY_SIZE; i++)
                mainHistory[c][i] = int16_t(int(mainHistory[c][i]) * SPSA::SF_MAINHIST_DECAY_NUM / 1024);

    // Iterative deepening loop until requested to stop or the target depth is reached
    // 隕∵ｱゅ′縺ゅｋ縺九√∪縺溘・逶ｮ讓呎ｷｱ蠎ｦ縺ｫ驕斐☆繧九∪縺ｧ蜿榊ｾｩ豺ｱ蛹悶Ν繝ｼ繝励ｒ螳溯｡後＠縺ｾ縺・

    // 統 rootDepth縺ｯ縺薙・thread縺ｮ蜿榊ｾｩ豺ｱ蛹悶〒縺ｮ謗｢邏｢荳ｭ縺ｮ豺ｱ縺輔・
    //     limits.depth("go"繧ｳ繝槭Φ繝峨・譎ゅ↓"depth"縺ｨ縺励※謖・ｮ壹＆繧後◆謗｢邏｢豺ｱ縺・縺梧欠螳壹＆繧後※縺・ｋ譎ゅ・縲・
    //     main thread縺ｮrootDepth縺後◎繧後ｒ雜・∴縺滓凾轤ｹ縺ｧ縺薙・繝ｫ繝ｼ繝励ｒ謚懊￠繧九・
    //     (main thread縺梧栢縺代ｋ縺ｨthreads.stop == true縺ｫ縺ｪ繧九・縺ｧ縺昴・縺ゅ→sub thread縺ｯ蜍晄焔縺ｫ縺薙・繝ｫ繝ｼ繝励ｒ謚懊￠繧・

    // M3 根の YBWC: helper は自分で深さを進めず master の世代に従う (I-1)。
    //   🚨既定では rootSplit == nullptr なのでこの分岐は無い。
    if ((rootSplit || deleg) && !mainThread)
    {
        ybwc_helper_loop(ss);
        return false;
    }

    while (++rootDepth < MAX_PLY && !threads.stop
           && !(limits.depth && mainThread && rootDepth > limits.depth))
    {
        /*
		涛 Stockfish9縺ｫ縺ｯslave thread繧知ain thread繧医ｊ蜈郁｡後＆縺帙ｋ繧ｳ繝ｼ繝峨′縺薙％縺ｫ縺ゅ▲縺溘′縲・
			Stockfish10縺ｧ蟒・ｭ｢縺輔ｌ縺溘・

			縺薙ｌ縺ｫ繧医ｊ遏ｭ縺・凾髢・菴弱＞rootDepth)縺ｧ縺ｯ謗｢邏｢蜉ｹ邇・′謔ｪ蛹悶＠縺ｦ蠑ｱ縺上↑縺｣縺溘・
			縺薙ｌ縺ｯ縲〉ootDepth縺悟ｰ上＆縺・→縺阪・helper thread縺後⊇縺ｨ繧薙←謗｢邏｢縺ｫ蟇・ｸ弱＠縺ｪ縺・◆繧√〒縺ゅｋ縲・
			縺励°縺羊ootDepth縺碁ｫ倥￥縺ｪ縺｣縺ｦ縺阪◆縺ｨ縺阪↓縺ｯ莠区ュ縺檎焚縺ｪ縺｣縺ｦ縺・※縲［ain thread繧医ｊdepth + 3縺ｨ縺九〒
			隱ｿ縺ｹ縺ｦ縺・ｋhelper thread縺後≠縺｣縺溘→縺励※繧ゅ∵爾邏｢縺梧遠縺｡蛻・ｉ繧後ｋ逶ｴ蜑阪↓縺翫＞縺ｦ縺ｯ縲・
			縺昴ｌ縺ｯmain thread縺ｮ謗｢邏｢縺ｫ蟇・ｸ弱＠縺ｦ縺・ｋ縺ｨ縺ｯ險縺・屮縺・◆繧√∫┌鬧・↓縺ｪ繧九・

			謚倩｡ｷ譯医→縺励※縲〉ootDepth縺御ｽ弱＞譎ゅ↓helper thread繧知ain thread繧医ｊ蜈郁｡後＆縺帙ｋ(鬮倥＞depth縺ｫ縺吶ｋ)
			繧ｳ繝ｼ繝芽・菴薙・蜈･繧後◆縺ｻ縺・′縺・＞縺九ｂ遏･繧後↑縺・・
		*/

        // ------------------------
        // Lazy SMP縺ｮ縺溘ａ縺ｮ蛻晄悄蛹・
        // ------------------------

        // Age out PV variability metric and signal the start of a new iteration.
        // PV螟牙虚繝｡繝医Μ繝・け繧貞商縺上＠縲∵眠縺励＞蜿榊ｾｩ縺ｮ髢句ｧ九ｒ遉ｺ縺吶・

        // 統 bestMove縺悟､牙喧縺励◆蝗樊焚繧定ｨ倬鹸縺励※縺・ｋ縺後∝渚蠕ｩ豺ｱ蛹悶・荳紋ｻ｣縺御ｸ縺､騾ｲ繧縺ｮ縺ｧ縲・
        //     蜿､縺・ｸ紋ｻ｣縺ｮ諠・ｱ縺ｨ縺励※驥阪∩繧剃ｽ弱￥縺励※縺・￥縲・

        if (mainThread)
        {
            totBestMoveChanges /= 2;
            uciPvSent = false;
        }

        // Save the last iteration's scores before the first PV line is searched and
        // all the move scores except the (new) PV are set to -VALUE_INFINITE.

        // 譛蛻昴・PV繝ｩ繧､繝ｳ繧呈爾邏｢縺吶ｋ蜑阪↓蜑榊屓繧､繝・Ξ繝ｼ繧ｷ繝ｧ繝ｳ縺ｮ繧ｹ繧ｳ繧｢繧剃ｿ晏ｭ倥＠縲・
        // ・域眠縺励＞・臼V 莉･螟悶・縺吶∋縺ｦ縺ｮ謇九・繧ｹ繧ｳ繧｢繧・-VALUE_INFINITE 縺ｫ險ｭ螳壹☆繧九・

        // 庁 aspiration window search縺ｮ縺溘ａ縺ｫ蜿榊ｾｩ豺ｱ蛹悶・蜑榊屓縺ｮiteration縺ｮ繧ｹ繧ｳ繧｢繧偵さ繝斐・縺励※縺翫￥

        for (RootMove& rm : rootMoves)
            rm.previousScore = rm.score;

        // ､・蟆・｣九〒縺ｯ縺薙ｌ菴ｿ繧上↑縺上※縺・＞繧医≧縺ｪ・・

        //size_t pvFirst = 0;
        //pvLast         = 0;

        // 庁 謗｢邏｢豺ｱ縺輔ｒ蠅励ｄ縺吶°縺ｮ繝輔Λ繧ｰ縺掲alse縺ｪ繧峨∝酔縺俶ｷｱ縺輔ｒ謗｢邏｢縺励◆縺薙→縺ｫ縺ｪ繧九・縺ｧ縲・
        //     searchAgainCounter繧ｫ繧ｦ繝ｳ繧ｿ繝ｼ繧・蠅励ｄ縺・
#if STOCKFISH
        if (!threads.increaseDepth)
#else
        if (!main_manager()->increaseDepth)
#endif
            searchAgainCounter++;

        // MultiPV loop. We perform a full root search for each PV line
        // MultiPV縺ｮloop縲・ultiPV縺ｮ縺溘ａ縺ｫ縺薙・螻髱｢縺ｮ蛟呵｣懈焔繧誰蛟矩∈蜃ｺ縺吶ｋ縲・
        for (pvIdx = 0; pvIdx < multiPV; ++pvIdx)
        {
            // 統 chess縺ｧ縺ｯtbRank縺ｮ蜃ｦ逅・′蠢・ｦ√ｉ縺励＞縲ょｰ・｣九〒縺ｯ髢｢菫ゅ↑縺輔￡縺ｪ縺ｮ縺ｧ繧ｳ繝｡繝ｳ繝医い繧ｦ繝医・
            //     tbRank縺悟酔縺伜､縺ｮ縺ｨ縺薙ｍ縺ｾ縺ｧ縺励°sort縺励↑縺上※濶ｯ縺・ｉ縺励＞縲・
            //     (縺昴％莉･髯阪・縲∵・繧峨°縺ｫ謔ｪ縺・欠縺玲焔縺ｪ縺ｮ縺ｧ)

#if STOCKFISH
            if (pvIdx == pvLast)
            {
                pvFirst = pvLast;
                for (pvLast++; pvLast < rootMoves.size(); pvLast++)
                    if (rootMoves[pvLast].tbRank != rootMoves[pvFirst].tbRank)
                        break;
            }
#else

            // ､・蟆・｣九□縺ｨtbRank縺ｯ蟶ｸ縺ｫ蜷後§縺ｨ縺ｿ縺ｪ縺帙ｋ縺ｮ縺ｧ縲・
            //     pvLast縺ｯrootMoves.size()縺ｫ縺ｪ繧九∪縺ｧ
            //     繧､繝ｳ繧ｯ繝ｪ繝｡繝ｳ繝医＆繧後ｋ縺九ｉ縲∵ｬ｡縺ｮ繧医≧縺ｫ蜊倡ｴ泌喧縺ｧ縺阪ｋ縲・

            size_t pvFirst = pvIdx;
            pvLast         = rootMoves.size();

#endif

            // Reset UCI info selDepth for each depth and each PV line
            // 縺昴ｌ縺槭ｌ縺ｮdepth縺ｨPV line縺ｫ蟇ｾ縺吶ｋUSI info縺ｧ蜃ｺ蜉帙☆繧虐elDepth
            selDepth = 0;

            // ------------------------
            // Aspiration window search
            // ------------------------

            /*
				涛 謗｢邏｢遯薙ｒ迢ｭ繧√※縺薙・遽・峇縺ｧ謗｢邏｢縺励※縲√％縺ｮ遯薙・遽・峇縺ｮscore縺瑚ｿ斐▲縺ｦ縺阪◆繧峨Λ繝・く繝ｼ縲√∩縺溘＞縺ｪ謗｢邏｢縲・

					謗｢邏｢縺梧ｵ・＞縺ｨ縺阪・ (-VALUE_INFINITE,+VALUE_INFINITE)縺ｮ遽・峇縺ｧ謗｢邏｢縺吶ｋ縲・
					謗｢邏｢豺ｱ縺輔′荳螳壻ｻ･荳翫≠繧九↑繧牙燕蝗槭・蜿榊ｾｩ豺ｱ蛹悶・iteration譎ゅ・譛蟆丞､縺ｨ譛螟ｧ蛟､
					繧医ｊ蟆代＠蟷・ｒ蠎・￡縺溘＄繧峨＞縺ｮ謗｢邏｢遯薙ｒ繝・ヵ繧ｩ繝ｫ繝医→縺吶ｋ縲・

					Reset aspiration window starting size
					aspiration window縺ｮ髢句ｧ九し繧､繧ｺ繧偵Μ繧ｻ繝・ヨ縺吶ｋ

					aspiration window縺ｮ蟷・
					邊ｾ蠎ｦ縺ｮ濶ｯ縺・ｩ穂ｾ｡髢｢謨ｰ縺ｪ繧峨・縺薙・蟷・ｒ蟆上＆縺上☆繧九→謗｢邏｢蜉ｹ邇・′荳翫′繧九・縺縺後・
					邊ｾ蠎ｦ縺ｮ謔ｪ縺・ｩ穂ｾ｡髢｢謨ｰ縺縺ｨ縺薙・蟷・ｒ蟆上＆縺上＠縺吶℃繧九→蜀肴爾邏｢縺悟｢励∴縺ｦ謗｢邏｢蜉ｹ邇・′菴惹ｸ九☆繧九・
					繧・・縺・ｉ邇九・KPP隧穂ｾ｡髢｢謨ｰ縺ｧ縺ｯ35・・0縺舌ｉ縺・′繝吶せ繝医・
					繧・・縺・ｉ邇九・KPPT(Apery WCSC26)縺ｧ縺ｯStockfish縺ｮ縺ｾ縺ｾ(18莉倩ｿ・縺後・繧ｹ繝医・
					繧ゅ▲縺ｨ邊ｾ蠎ｦ縺ｮ鬮倥＞隧穂ｾ｡髢｢謨ｰ繧堤畑諢上☆縺ｹ縺阪・
					縺薙・蛟､縺ｯStockfish10縺ｧ縺ｯ20縺ｫ螟画峩縺輔ｌ縺溘・
					Stockfish 12(NNUE繧貞ｰ主・縺励◆)縺ｧ縺ｯ17縺ｫ螟画峩縺輔ｌ縺溘・
					Stockfish 12.1縺ｧ縺ｯ16縺ｫ螟画峩縺輔ｌ縺溘・
					Stockfish 16縺ｧ縺ｯ10縺ｫ螟画峩縺輔ｌ縺溘・
					Stockfish 17縺ｧ縺ｯ 5縺ｫ螟画峩縺輔ｌ縺溘・

				庁 蟆・｣九〒縺ｯStockfish繧医ｊ蟆代＠鬮倥ａ縺瑚憶縺輔◎縺・・
			*/

            // Reset aspiration window starting size
            // aspiration window縺ｮ髢句ｧ九し繧､繧ｺ繧偵Μ繧ｻ繝・ヨ縺吶ｋ縲・
            delta     = SPSA::ASPIRATION_DELTA_BASE + SPSA::ASPIRATION_THREAD_STEP * (int)(threadIdx % 8)
                      + std::abs(rootMoves[pvIdx].meanSquaredScore) / SPSA::ASPIRATION_MSS_DIV;
            Value avg = rootMoves[pvIdx].averageScore;
            alpha     = std::max(avg - delta, -VALUE_INFINITE);
            beta      = std::min(avg + delta,  VALUE_INFINITE);

#if 0
            // Adjust optimism based on root move's averageScore
            // 繝ｫ繝ｼ繝域焔縺ｮ averageScore 縺ｫ蝓ｺ縺･縺・※讌ｽ隕ｳ蠎ｦ繧定ｪｿ謨ｴ縺吶ｋ
			optimism[us]  = 137 * avg / (std::abs(avg) + 91);
            optimism[~us] = -optimism[us];
#endif
#if defined(P2_SMP_EVALCHECK)
            if (pvIdx == 0 && rootDepth == 10) P2SmpCheck::batch(1, threadIdx);
#endif
#if defined(P2_OPTIMISM)
            // P2_OPTIMISM (SF 方式の動的部分): この反復の root の平均評価から dyn を決める。 SCALE = 0 (既定) なら何もしない。
            if (g_p2opt_dyn_scale != 0)
            {
                const int dyn = (avg == -VALUE_INFINITE)
                                  ? 0
                                  : int(g_p2opt_dyn_scale * int(avg) / (std::abs(int(avg)) + g_p2opt_dyn_div));
                p2Opt[us]  = Value(g_p2opt_us + dyn);
                p2Opt[~us] = Value(g_p2opt_them - dyn);
                if (g_p2opt_log && mainThread)
                    sync_cout << "info string P2OPT depth " << rootDepth << " pv " << pvIdx << " avg " << int(avg)
                              << " dyn " << dyn << " us " << int(p2Opt[us]) << " them " << int(p2Opt[~us]) << sync_endl;
            }
#endif

			// ､・縺薙・optimism縺ｯ縲ヾtockfish縺ｮNNUE隧穂ｾ｡髢｢謨ｰ縺ｧ菴輔ｄ繧我ｽｿ縺｣縺ｦ縺・ｋ繧医≧縺ｪ縺ｮ縺縺娯ｦ縲・
            //     TODO : 縺ゅ→縺ｧ讀懆ｨ弱☆繧九・

            // Start with a small aspiration window and, in the case of a fail
            // high/low, re-search with a bigger window until we don't fail
            // high/low anymore.

            // 蟆上＆縺ｪaspiration window縺ｧ髢句ｧ九＠縺ｦ縲’ail high/low縺ｮ縺ｨ縺阪↓縲’ail high/low縺ｫ縺ｪ繧峨↑縺・ｈ縺・↓縺ｪ繧九∪縺ｧ
            // 螟ｧ縺阪↑window縺ｧ蜀肴爾邏｢縺吶ｋ縲・

            // fail high縺励◆蝗樊焚
            // 庁 fail high縺励◆蝗樊焚蛻・□縺第爾邏｢depth繧剃ｸ九￡縺ｦ繧・ｋ縺ｻ縺・′蠑ｷ縺・ｉ縺励＞縲・
            int failedHighCnt = 0;

            while (true)
            {
                // Adjust the effective depth searched, but ensure at least one
                // effective increment for every four searchAgain steps (see issue #2717).

                // 螳滄圀縺ｫ謗｢邏｢縺励◆豺ｱ縺輔ｒ隱ｿ謨ｴ縺吶ｋ縺後・
                // searchAgain 繧ｹ繝・ャ繝・蝗槭＃縺ｨ縺ｫ蟆代↑縺上→繧・蝗槭・譛牙柑縺ｪ蠅怜・縺後≠繧九ｈ縺・↓縺吶ｋ
                // ・・ssue #2717 繧貞盾辣ｧ・峨・

                // fail high縺吶ｋ縺斐→縺ｫdepth繧剃ｸ九￡縺ｦ縺・￥蜃ｦ逅・
                Depth adjustedDepth =
                  std::max(1, rootDepth - failedHighCnt - 3 * (searchAgainCounter + 1) / 4);
                rootDelta = beta - alpha;
#if defined(P2_SRCH_BUNDLE_A)
                nonPvDeltaTerm = SPSA::LMR_REDUCTION_DELTA_MULT / rootDelta;
#endif

                // M3 根の YBWC: この世代を publish する (master だけ)
                if (rootSplit)
                {
                    // 🚨前の世代で取られたままの手が無くなるのを待つ。 待たずに次を
                    //   publish すると、 古い世代の helper が新しい表へ書いて結果が混ざる。
                    if (ybwcGen)
                        while (!rootSplit->all_taken_done(ybwcGen) && !threads.stop)
                            std::this_thread::yield();
                    const int n = int(rootMoves.size());
                    for (int i = 0; i < n; ++i)
                    {
                        ybwcMoves[i] = rootMoves[i].pv[0];
                        ybwcMine[i]  = false;
                    }
                    ++ybwcGen;
                    ybwcIdx = -1;
                    rootSplit->publish(ybwcGen, n, ybwcMoves, alpha, beta, adjustedDepth,
                                       int(pvIdx));
                }

                bestValue = search<Root>(rootPos, ss, alpha, beta, adjustedDepth, false);

                // M3 根の YBWC: helper の結果を回収する
                if (rootSplit)
                {
                    // 🚨待ちは**有界**にする。 間に合わなかった手は読まれなかったものとして
                    //   扱えばよく (score = -VALUE_INFINITE)、 正しさは保たれる。
                    const TimePoint tw = now();
                    while (!rootSplit->all_taken_done(ybwcGen) && !threads.stop
                           && now() - tw < 200)
                        std::this_thread::yield();
                    rootSplit->close(ybwcGen);
                    bestValue = ybwc_import(bestValue);
                }

                // Bring the best move to the front. It is critical that sorting
                // is done with a stable algorithm because all the values but the
                // first and eventually the new best one is set to -VALUE_INFINITE
                // and we want to keep the same order for all the moves except the
                // new PV that goes to the front. Note that in the case of MultiPV
                // search the already searched PV lines are preserved.

                // 譛蝟・焔繧貞・鬆ｭ縺ｫ謖√▲縺ｦ縺上ｋ縲ゅ％繧後・髱槫ｸｸ縺ｫ驥崎ｦ√〒縺ゅｊ縲・
                // 螳牙ｮ壹た繝ｼ繝医い繝ｫ繧ｴ繝ｪ繧ｺ繝繧剃ｽｿ縺・ｿ・ｦ√′縺ゅｋ縲・
                // 縺ｪ縺懊↑繧峨∵怙蛻昴・謇九♀繧医・譁ｰ縺励＞譛蝟・焔莉･螟悶・縺吶∋縺ｦ縺ｮ蛟､縺ｯ
                // -VALUE_INFINITE 縺ｫ險ｭ螳壹＆繧後ｋ縺溘ａ縲・
                // 譁ｰ縺励＞PV縺悟・鬆ｭ縺ｫ譚･繧倶ｻ･螟悶・縲√☆縺ｹ縺ｦ縺ｮ謇九・鬆・ｺ上ｒ邯ｭ謖√＠縺溘＞縺九ｉ縺ｧ縺ゅｋ縲・
                // MultiPV謗｢邏｢縺ｮ蝣ｴ蜷医√☆縺ｧ縺ｫ謗｢邏｢貂医∩縺ｮPV繝ｩ繧､繝ｳ縺ｯ菫晄戟縺輔ｌ繧狗せ縺ｫ豕ｨ諢上・

                // 統 縺昴ｌ縺槭ｌ縺ｮ謖・＠謇九↓蟇ｾ縺吶ｋ繧ｹ繧ｳ繧｢繝ｪ繝ｳ繧ｰ縺檎ｵゅｏ縺｣縺溘・縺ｧ荳ｦ縺ｹ譖ｿ縺医♀縺上・
                //    荳縺､逶ｮ縺ｮ謖・＠謇倶ｻ･螟悶・-VALUE_INFINITE縺瑚ｿ斐ｋ莉墓ｧ倥↑縺ｮ縺ｧ荳ｦ縺ｹ譖ｿ縺医・縺溘ａ縺ｫ螳牙ｮ壹た繝ｼ繝医ｒ
                //    逕ｨ縺・↑縺・→蜑榊屓縺ｮ蜿榊ｾｩ豺ｱ蛹悶・邨先棡縺ｫ繧医▲縺ｦ蠕励◆荳ｦ縺ｳ鬆・ｒ螟峨∴縺ｦ縺励∪縺・％縺ｨ縺ｫ縺ｪ繧九・縺ｧ縺ｾ縺壹＞縲・

                std::stable_sort(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast);

                // If search has been stopped, we break immediately. Sorting is
                // safe because RootMoves is still valid, although it refers to
                // the previous iteration.

                // 謗｢邏｢縺悟●豁｢縺輔ｌ縺ｦ縺・ｋ蝣ｴ蜷医・逶ｴ縺｡縺ｫ break 縺吶ｋ縲・
                // RootMoves 縺ｯ蜑榊屓縺ｮ繧､繝・Ξ繝ｼ繧ｷ繝ｧ繝ｳ繧貞盾辣ｧ縺励※縺・ｋ縺御ｾ晉┯縺ｨ縺励※譛牙柑縺ｪ縺ｮ縺ｧ縲・
                // 繧ｽ繝ｼ繝医・螳牙・縺ｧ縺ゅｋ縲・

                if (threads.stop)
                    break;

                // When failing high/low give some update before a re-search. To avoid
                // excessive output that could hang GUIs like Fritz 19, only start
                // at nodes > 10M (rather than depth N, which can be reached quickly)

                // fail high / low縺ｮ譎ゅ↓縺ｯ縲∝・謗｢邏｢蜑阪↓菴輔ｉ縺九・諠・ｱ繧貞・蜉帙☆繧九・
                // 縺溘□縺励：ritz 19 縺ｪ縺ｩ縺ｮ GUI 縺後ヵ繝ｪ繝ｼ繧ｺ縺吶ｋ縺ｮ繧帝∩縺代ｋ縺溘ａ縲・
                // 豺ｱ縺・N 縺ｧ縺ｯ縺ｪ縺上・0M 繝弱・繝峨ｒ雜・∴縺ｦ縺九ｉ蜃ｺ蜉帙ｒ蟋九ａ繧九・

                // 庁 蟆・｣区園縺ｮ繧ｳ繝ｳ繧ｽ繝ｼ繝ｫ縺瑚ｩｰ縺ｾ繧九・繧剃ｺ磯亟縺吶ｋ縺溘ａ縺ｫ蜃ｺ蜉帙ｒ蟆代＠謚大宛縺吶ｋ縲・
                //    縺ｾ縺溘“o infinite縺ｮ縺ｨ縺阪・縲∵､懆ｨ弱Δ繝ｼ繝峨°繧我ｽｿ逕ｨ縺励※縺・ｋ繧上￠縺ｧ縲￣V縺ｯ蠢・★蜃ｺ蜉帙☆繧九・

                if (mainThread && multiPV == 1 && (bestValue <= alpha || bestValue >= beta)
                    && nodes > 10000000

#if !SOTCKFISH
                    // 決 莉･荳九ｄ縺ｭ縺・ｉ邇狗峡閾ｪ諡｡蠑ｵ
                    && (rootDepth < 3
                        || search_options.lastPvInfoTime + search_options.computed_pv_interval
                             <= now())
                    // outout_fail_lh_pv縺掲alse縺ｪ繧映ail high/fail low縺ｮ縺ｨ縺阪・PV繧貞・蜉帙＠縺ｪ縺・・
                    && search_options.outout_fail_lh_pv
#endif
                )
#if STOCKFISH
                    main_manager()->pv(*this, threads, tt, rootDepth);
#else
                {
                    main_manager()->pv(*this, threads, tt, rootDepth);
                    // 譛蠕後↓PV繧貞・蜉帙＠縺滓凾蛻ｻ繧呈ｼ邏阪＠縺ｦ縺翫￥縲・
                    search_options.lastPvInfoTime = now();
                }
#endif

                // In case of failing low/high increase aspiration window and re-search,
                // otherwise exit the loop.
                if (bestValue <= alpha)
                {
                    beta  = alpha;
                    alpha = std::max(bestValue - delta, -VALUE_INFINITE);

                    failedHighCnt = 0;
                    if (mainThread)
                        mainThread->stopOnPonderhit = false;
                }
                else if (bestValue >= beta)
                {
                    alpha = std::max(beta - delta, alpha);
                    beta  = std::min(bestValue + delta, VALUE_INFINITE);
                    ++failedHighCnt;
                }
                else
                    break;

                delta += delta / 3;

                assert(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
            }

            // Sort the PV lines searched so far and update the GUI
            // 縺薙ｌ縺ｾ縺ｧ縺ｫ謗｢邏｢縺励◆PV繝ｩ繧､繝ｳ繧偵た繝ｼ繝医＠縲；UI繧呈峩譁ｰ縺吶ｋ

            // 庁 MultiPV縺ｮ蛟呵｣懈焔繧偵せ繧ｳ繧｢鬆・↓蜀榊ｺｦ荳ｦ縺ｳ譖ｿ縺医※縺翫￥縲・
            //    (莠檎分逶ｮ縺縺ｨ諤昴▲縺ｦ縺・◆縺ｻ縺・・謖・＠謇九・縺ｻ縺・′隧穂ｾ｡蛟､縺瑚憶縺・庄閭ｽ諤ｧ縺後≠繧九・縺ｧ窶ｦ)

            std::stable_sort(rootMoves.begin() /* + pvFirst */, rootMoves.begin() + pvIdx + 1);

            if (mainThread
                && (!threads.stop && (pvIdx + 1 == multiPV || nodes > 10000000))
                // A fail-low/high root line is an incomplete aspiration result. Do not emit it as PV.
                && !rootMoves[0].scoreLowerbound && !rootMoves[0].scoreUpperbound
                // A thread that aborted search can have mated-in/TB-loss PV and
                // score that cannot be trusted, i.e. it can be delayed or refuted
                // if we would have had time to fully search other root-moves. Thus
                // we suppress this output and below pick a proven score/PV for this
                // thread (from the previous iteration).

                // 謗｢邏｢繧剃ｸｭ譁ｭ縺励◆繧ｹ繝ｬ繝・ラ縺ｯ縲∬ｩｰ縺ｿ逶ｴ蜑阪・PV繧УB謳榊､ｱ縺ｮPV縺翫ｈ縺ｳ
                // 菫｡鬆ｼ縺ｧ縺阪↑縺・せ繧ｳ繧｢繧呈戟縺､蜿ｯ閭ｽ諤ｧ縺後≠繧九・
                // 縺､縺ｾ繧翫∽ｻ悶・繝ｫ繝ｼ繝医Β繝ｼ繝悶ｒ螳悟・縺ｫ謗｢邏｢縺吶ｋ譎る俣縺後≠繧後・縲・
                // 驕・ｻｶ縺励◆繧雁渚險ｼ縺輔ｌ縺溘ｊ縺吶ｋ縺九ｂ縺励ｌ縺ｪ縺・・
                // 縺励◆縺後▲縺ｦ縲√％縺ｮ蜃ｺ蜉帙ｒ謚大宛縺励∽ｻ･荳九〒縺薙・繧ｹ繝ｬ繝・ラ縺ｫ蟇ｾ縺励※
                // ・亥燕蝗槭・蜿榊ｾｩ縺九ｉ・芽ｨｼ譏取ｸ医∩縺ｮ繧ｹ繧ｳ繧｢・襲V繧帝∈謚槭☆繧九・

                && !(threads.abortedSearch && is_loss(rootMoves[0].uciScore))
#if !STOCKFISH
                // PV縺ｮ蜃ｺ蜉幃俣髫斐ｒ雜・∴縺ｦ縺・ｋ縲・
                && search_options.lastPvInfoTime + search_options.computed_pv_interval <= now()
#endif
            )
#if STOCKFISH
                main_manager()->pv(*this, threads, tt, rootDepth);
#else
            {
                main_manager()->pv(*this, threads, tt, rootDepth);
                // 譛蠕後↓PV繧貞・蜉帙＠縺滓凾蛻ｻ繧呈ｼ邏阪＠縺ｦ縺翫￥縲・
                search_options.lastPvInfoTime = now();
                uciPvSent = (pvIdx + 1 == multiPV);
            }
#endif


            if (threads.stop)
                break;

        }  // multi pv loop

        if (!threads.stop)
        {
            completedDepth  = rootDepth;

            if (lastIterationPV.empty() || rootMoves[0].pv[0] != lastIterationPV[0])
                lastBestMoveDepth = rootDepth;

            lastIterationPV = rootMoves[0].pv;
        }

        // A mated-in/TB-loss score from an aborted search cannot be trusted: the loss
        // could be delayed or refuted upon exploring the remaining root-moves.
        // Thus here we roll back to the score from the previous iteration.

        // 荳ｭ譁ｭ縺輔ｌ縺滓爾邏｢縺ｧ蠕励◆隧ｰ縺ｿ/TB雋縺代・繧ｹ繧ｳ繧｢縺ｯ菫｡鬆ｼ縺ｧ縺阪↑縺・・
        // 谿九ｊ縺ｮ繝ｫ繝ｼ繝域焔繧定ｪｿ縺ｹ繧九→縲√◎縺ｮ雋縺代′蟒ｶ縺ｳ縺溘ｊ蜿崎ｨｼ縺輔ｌ縺溘ｊ縺吶ｋ蜿ｯ閭ｽ諤ｧ縺後≠繧九・
        // 縺昴・縺溘ａ縲√％縺薙〒縺ｯ蜑榊屓縺ｮ蜿榊ｾｩ豺ｱ蛹悶〒蠕励◆繧ｹ繧ｳ繧｢縺ｫ謌ｻ縺吶・

        if (threads.abortedSearch && rootMoves[0].score != -VALUE_INFINITE
            && is_loss(rootMoves[0].score))
        {
            // Bring the last best move to the front for best thread selection.
            // 譛蠕後↓蠕励ｉ繧後◆譛蝟・焔繧貞・鬆ｭ縺ｫ遘ｻ蜍輔＠縲∵怙驕ｩ縺ｪ繧ｹ繝ｬ繝・ラ驕ｸ謚槭・縺溘ａ縺ｫ蛯吶∴繧九・
            // 庁 move_to_front()縺ｯ縲∬ｦ九▽縺代◆繧ゅ・繧貞・鬆ｭ縺ｫ遘ｻ蜍輔＆縺帙∝・縺ｮ蜈磯ｭ縺九ｉ縺昴％縺ｾ縺ｧ縺ｯ1縺､蠕梧婿縺ｫ縺壹ｉ縺吶・

            Utility::move_to_front(rootMoves, [&lastBestPV = std::as_const(lastBestPV)](
                                                const auto& rm) { return rm == lastBestPV[0]; });
            rootMoves[0].pv    = lastBestPV;
            rootMoves[0].score = rootMoves[0].uciScore = lastBestScore;

            if (mainThread && lastBestPV[0] != Move::none())
            {
#if STOCKFISH
                uciPvSent = true;
#else
                // 蜑榊屓iteration縺ｮPV縺ｸ繝ｭ繝ｼ繝ｫ繝舌ャ繧ｯ縺励◆縺縺代〒縲√◎縺ｮPV繧旦SI縺ｸ蜃ｺ蜉帙＠縺溘→縺ｯ髯舌ｉ縺ｪ縺・・
                // 繧・・縺・ｉ邇九〒縺ｯPV蜃ｺ蜉幃俣髫斐↓繧医▲縺ｦ騾比ｸｭPV縺梧椛蛻ｶ縺輔ｌ繧九◆繧√√％縺薙〒true縺ｫ縺吶ｋ縺ｨ
                // bestmove逶ｴ蜑阪・譛邨１V蜃ｺ蜉帙′繧ｹ繧ｭ繝・・縺輔ｌ繧九％縺ｨ縺後≠繧九・
                uciPvSent = false;
#endif
            }
        }
        else if (!threads.stop && rootMoves[0].pv[0] != lastBestPV[0])
        {
            lastBestPV        = rootMoves[0].pv;
            lastBestScore     = rootMoves[0].score;
            lastBestMoveDepth = rootDepth;
        }

        // ､・謗｢邏｢豺ｱ縺輔′縲［ate縺ｮ謇区焚縺ｮ2蛟堺ｻ･荳翫↓縺ｪ縺｣縺溘ｉ縲√◎繧後ｈ繧顔洒縺・ｩｰ縺ｿ縺・
        //     隕九▽縺九ｋ縺薙→縺ｯ遞縺縺励∵爾邏｢繧貞●豁｢縺輔○縺ｦ濶ｯ縺・→諤昴≧縲・繧・・縺・ｉ邇狗峡閾ｪ)
        // 庁 multi_pv縺ｮ縺ｨ縺阪・荳縺､縺ｮpv縺ｧ隧ｰ縺ｿ繧定ｦ九▽縺代◆縺縺代〒縺ｯ蛛懈ｭ｢縺吶ｋ縺ｮ縺ｯ濶ｯ縺上↑縺・・縺ｧ
        //     譌ｩ譛溽ｵゆｺ・・multiPV == 1縺ｮ縺ｨ縺阪・縺ｿ陦後↑縺・・

        if (multiPV == 1)
        {
            // 蜍昴■繧定ｪｭ縺ｿ縺阪▲縺ｦ縺・ｋ縺ｮ縺ｫ蟆・｣区園縺ｮ陦ｨ遉ｺ縺瑚ｿｽ縺・▽縺九★縺ｫ縲∝ｰ・｣区園縺後ヵ繝ｪ繝ｼ繧ｺ縺励※縺・※縲√◎縺ｮ髢薙・譎る俣繝ｭ繧ｹ縺ｧ
            // 譎る俣蛻・ｌ縺ｧ雋縺代ｋ縺薙→縺後≠繧九・
            // mate繧定ｪｭ縺ｿ縺阪▲縺溘→縺阪√◎縺ｮmate縺ｮ2.5蛟堺ｻ･荳翫（teration繧貞屓縺励※繧ゆｻ墓婿縺ｪ縺・ｰ励′縺吶ｋ縺ｮ縺ｧ謗｢邏｢繧呈遠縺｡蛻・ｋ繧医≧縺ｫ縺吶ｋ縲・
            // 竍ｨ 縺ゅ→縲〉oot縺ｧ1謇玖ｩｰ繧∝他縺ｳ蜃ｺ縺輔↑縺上＠縺溘・縺ｧ縲√◎縺ｮ蠖ｱ髻ｿ繧ゅ≠縺｣縺ｦ縲〃ALUE_MATE == bestValue縺ｫ縺ｪ繧九％縺ｨ縺ｯ縺ゅｋ縺九ｉ縲√％縺ｮ譎ゅ・
            //   rootDepth == 1縺ｧ謗｢邏｢繧堤ｵゆｺ・＆繧後※縺ｯ蝗ｰ繧九ゅｂ縺・ｰ代＠蜈医∪縺ｧ隱ｿ縺ｹ縺ｦ谺ｲ縺励＞縺ｮ縺ｧ縲・2縺励※縺翫￥縲・
            if (!limits.mate && bestValue >= VALUE_MATE_IN_MAX_PLY
                && (VALUE_MATE - bestValue + 2) * 5 / 2 < (Value) (rootDepth))
                break;

            // 隧ｰ縺ｾ縺輔ｌ繧句ｽ｢縺ｫ縺､縺・※繧ょ酔讒倥ゅ％縺｡繧峨・mate縺ｮ2.5蛟堺ｻ･荳翫（teration繧貞屓縺励◆縺ｪ繧画爾邏｢繧呈遠縺｡蛻・ｋ縲・
            if (!limits.mate && bestValue <= VALUE_MATED_IN_MAX_PLY
                && (bestValue - (-VALUE_MATE) + 2) * 5 / 2 < (Value) (rootDepth))
                break;
        }

        if (!mainThread)
            continue;

        // 決 ponder逕ｨ縺ｮ謖・＠謇九→縺励※縲・謇狗岼縺ｮ謖・＠謇九ｒ菫晏ｭ倥＠縺ｦ縺翫￥縲・
        //     縺薙ｌ縺稽ain thread縺ｮ繧ゅ・縺縺代〒縺・＞縺九←縺・°縺ｯ繧医￥繧上°繧峨↑縺・′縲・
        //     縺ｨ繧翫≠縺医★縲∫┌縺・ｈ繧翫・繧ｷ縺繧阪≧縲・繧・・縺・ｉ邇狗峡閾ｪ諡｡蠑ｵ)

        if (rootMoves[0].pv.size() > 1)
            mainThread->ponder_candidate = rootMoves[0].pv[1];

        // Have we found a "mate in x"?
        // x謇玖ｩｰ繧√ｒ逋ｺ隕九＠縺溘・縺具ｼ・

        // 庁 UCI縺ｧ縺ｯ"go mate 5"縺ｮ繧医≧縺ｫmate縺ｮ縺ゅ→謇区焚縺碁√ｉ繧後※縺上ｋ莉墓ｧ倥・
        //     USI縺ｧ縺ｯ"go mate"縺ｮ縺ゅ→縺ｯ諤晁・凾髢薙′繧・▲縺ｦ縺上ｋ縺ｮ縺ｧ縲∵掠譛溘Μ繧ｿ繧､繧｢縺ｧ縺阪↑縺・・
#if STOCKFISH
        if (limits.mate && rootMoves[0].score == rootMoves[0].usiScore
            && ((rootMoves[0].score >= VALUE_MATE_IN_MAX_PLY
                 && VALUE_MATE - rootMoves[0].score <= 2 * limits.mate)
                || (rootMoves[0].score != -VALUE_INFINITE
                    && rootMoves[0].score <= VALUE_MATED_IN_MAX_PLY
                    && VALUE_MATE + rootMoves[0].score <= 2 * limits.mate)))
            threads.stop = true;
#endif

        // If the skill level is enabled and time is up, pick a sub-optimal best move
        // 繧ｹ繧ｭ繝ｫ繝ｬ繝吶Ν縺梧怏蜉ｹ縺ｧ縲√°縺､譎る俣蛻・ｌ縺ｮ蝣ｴ蜷医∵怙驕ｩ縺ｧ縺ｪ縺・・繧ｹ繝医Β繝ｼ繝悶ｒ驕ｸ縺ｶ縲・

        if (skill.enabled() && skill.time_to_pick(rootDepth))
            skill.pick_best(rootMoves, multiPV);

        // Use part of the gained time from a previous stable move for the current move
        // 逶ｴ蜑阪・螳牙ｮ壹＠縺滓焔縺ｧ蠕励◆譎る俣縺ｮ荳驛ｨ繧堤樟蝨ｨ縺ｮ謇九↓菴ｿ縺・・

        for (auto&& th : threads)
        {
            auto yw = toYaneuraOuWorker(th->worker);
            totBestMoveChanges += yw->bestMoveChanges;
            yw->bestMoveChanges = 0;
        }

        // Do we have time for the next iteration? Can we stop searching now?
        // 谺｡縺ｮ蜿榊ｾｩ繧定｡後≧譎る俣縺ｯ縺ゅｋ縺具ｼ滉ｻ翫☆縺先爾邏｢繧呈ｭ｢繧√ｉ繧後ｋ縺具ｼ・
#if STOCKFISH
        if (limits.use_time_management() && !threads.stop && !mainThread->stopOnPonderhit)
#else
        // 統 繧・・縺・ｉ邇九・蝣ｴ蜷医《earch_end縺瑚ｨｭ螳壹＆繧後※縺・ｋ譎ゅ・繧ゅ≧邨ゆｺ・′遒ｺ螳壹＠縺ｦ縺・ｋ縺ｮ縺ｧ
        //     縺薙・邨ゆｺ・メ繧ｧ繝・け繧定｡後≧縺ｮ縺ｯ辟｡鬧・〒縺ゅｋ縲・
        if (limits.use_time_management() && !threads.stop && !mainThread->stopOnPonderhit
            && !mainThread->tm.search_end)
#endif
        {
            uint64_t nodesEffort =
              rootMoves[0].effort * 100000 / std::max(size_t(1), size_t(nodes));

            double fallingEval =
              (11.325 + (SPSA::TM_FE_PREV_MULT / 1000.0) * (mainThread->bestPreviousAverageScore - bestValue)
               + (SPSA::TM_FE_ITER_MULT / 1000.0) * (mainThread->iterValue[iterIdx] - bestValue))
              / 100.0;
            fallingEval = std::clamp(fallingEval, 0.5688, 1.5698);

            // If the bestMove is stable over several iterations, reduce time accordingly
            // bestMove 縺瑚､・焚蝗槭・繧､繝・Ξ繝ｼ繧ｷ繝ｧ繝ｳ縺ｧ螳牙ｮ壹＠縺ｦ縺・ｋ蝣ｴ蜷医√◎繧後↓蠢懊§縺ｦ譎る俣繧堤洒邵ｮ縺吶ｋ

            double k      = 0.5189;
            double center = lastBestMoveDepth + SPSA::TM_STABLE_CENTER / 100.0;
            timeReduction = 0.723 + 0.79 / (1.104 + std::exp(-k * (completedDepth - center)));
            double reduction =
              (1.455 + mainThread->previousTimeReduction) / (2.2375 * timeReduction);
            double bestMoveInstability = 1.04 + (SPSA::TM_INSTAB_MULT / 10000.0) * totBestMoveChanges / threads.size();
			double highBestMoveEffort  = completedDepth >= 10 && nodesEffort >= 92425 ? SPSA::TM_EFFORT_MULT / 1000.0 : 1.0;

            double totalTime = mainThread->tm.optimum() * fallingEval * reduction
                             * bestMoveInstability * highBestMoveEffort;

            // Cap used time in case of a single legal move for a better viewer experience
            // 隕冶・閠・ｽ馴ｨ薙ｒ蜷台ｸ翫＆縺帙ｋ縺溘ａ縲∝粋豕墓焔縺・縺､縺縺代・蝣ｴ蜷医↓菴ｿ逕ｨ譎る俣繧剃ｸ企剞縺ｧ蛻ｶ髯舌☆繧・

            if (rootMoves.size() == 1)
                totalTime = std::min(502.0, totalTime);
	            // ､・繧・・縺・ｉ邇九〒縺ｯ縺薙％0縺ｧ繧り憶縺・ｈ縺・↑窶ｦ・・

#if STOCKFISH
            auto elapsedTime = elapsed();
#else
            // 統 繧・・縺・ｉ邇九〒縺ｯ縲｀ainManager縺ｮTimer class縺ｮ縺ｻ縺・°繧臥ｵ碁℃譎る俣繧偵ｂ繧峨≧縲・
            auto elapsedTime = mainThread->tm.elapsed_time();
#endif

            // Stop the search if we have exceeded the totalTime or maximum
            // totalTime 縺ｾ縺溘・ maximum 繧定ｶ・∴縺溷ｴ蜷医∵爾邏｢繧貞●豁｢縺吶ｋ

#if STOCKFISH
            if (elapsedTime > std::min(totalTime, double(mainThread->tm.maximum())))
            {
                // 蛛懈ｭ｢譚｡莉ｶ繧呈ｺ縺溘＠縺・

                // 統 蟆・｣九・蝣ｴ蜷医√ヵ繧｣繝・す繝｣繝ｼ繝ｫ繝ｼ繝ｫ縺ｧ縺ｯ縺ｪ縺・・縺ｧ縺薙・譎らせ縺ｧ繧よ怙蟆乗晁・凾髢灘・縺縺代・
                //     諤晁・ｒ邯咏ｶ壹＠縺溘⊇縺・′蠕励↑縺ｮ縺ｧ縲∵晁・・菴薙・邯咏ｶ壹＠縺ｦ縲√く繝ｪ縺ｮ濶ｯ縺・凾髢薙↓縺ｪ縺｣縺溘ｉ
                //     check_time()縺ｫ縺ｦ蛛懈ｭ｢縺吶ｋ縲・

                // If we are allowed to ponder do not stop the search now but
                // keep pondering until the GUI sends "ponderhit" or "stop".

                if (mainThread->ponder)
                    mainThread->stopOnPonderhit = true;
                else
                    threads.stop = true;
            }
            else
                threads.increaseDepth = mainThread->ponder || elapsedTime <= totalTime * (SPSA::TM_INCDEPTH_PCT / 1000.0);

#else
            /*
				 笞 繧・・縺・ｉ邇九〒縺ｯ縲√％縺ｮ縺ｸ繧謎ｻ慕ｵ・∩縺檎焚縺ｪ繧九・
			         遘貞腰菴阪〒蛻・ｊ荳翫￡縺溘＞縺ｮ縺ｧ縲》m.search_end縺ｾ縺ｧ
				     謖√■譎る俣繧剃ｽｿ縺・・繧翫◆縺・・

					 threads.stop = true;
					 縺ｮ莉｣繧上ｊ縺ｫ
					 mainThread->tm.set_search_end(elapsedTime);
					 繧貞他縺ｳ蜃ｺ縺吶％縺ｨ縲・
			*/

			if (elapsedTime > std::min(totalTime, double(mainThread->tm.maximum())))
            {
				if (mainThread->ponder)
                    mainThread->stopOnPonderhit = true;
                else
                    // 蛛懈ｭ｢譚｡莉ｶ繧呈ｺ縺溘＠縺ｦ縺ｯ縺・ｋ縺ｮ縺縺後∫ｧ偵・蛻・ｊ荳翫￡縺ｯ陦後≧縲・
                    mainThread->tm.set_search_end(elapsedTime);
            }
            else
                main_manager()->increaseDepth =
                  mainThread->ponder || elapsedTime <= totalTime * (SPSA::TM_INCDEPTH_PCT / 1000.0);
#endif
        }

        mainThread->iterValue[iterIdx] = bestValue;
        iterIdx                        = (iterIdx + 1) & 3;
    }

    if (!mainThread)
        return false;

    mainThread->previousTimeReduction = timeReduction;

    // If the skill level is enabled, swap the best PV line with the sub-optimal one
    if (skill.enabled())
        std::swap(rootMoves[0],
                  *std::find(rootMoves.begin(), rootMoves.end(),
                             skill.best ? skill.best : skill.pick_best(rootMoves, multiPV)));

    return uciPvSent;
}

void Search::YaneuraOuWorker::do_move(Position & pos, const Move move, StateInfo& st,
                                        Stack* const ss) {
    do_move(pos, move, st, pos.gives_check(move), ss);
}


void YaneuraOuWorker::do_move(
  Position& pos, const Move move, StateInfo& st, const bool givesCheck, Stack* const ss) {

    bool capture = pos.capture_stage(move);

    // M4: root からの手順を積む。 既定 (deleg == nullptr) では何もしない。
    //   🚨ss からは取れない (ss = stack + 7 なので ss - ply は ply <= 7 でしか有効でなく、
    //     それより深い節では配列の外を読む。 2026-11-02 の最初の実装がこれで即死した)。
    if (deleg != nullptr && curPathLen < MAX_PLY)
        curPath[curPathLen++] = move;

    // Preferable over fetch_add to avoid locking instructions
	// 繝ｭ繝・け蜻ｽ莉､繧帝∩縺代ｋ縺溘ａ縲’etch_add 繧医ｊ繧ゅ％縺｡繧峨・譁ｹ縺梧悍縺ｾ縺励＞縲・

	nodes.store(nodes.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);

#if defined(EVAL_SFNN)

    // accumulatorStack繧堤畑縺・ｋ螳溯｣・・

    DirtyBoardData dirtyBoardData = pos.do_move(move, st, givesCheck, &tt);
    accumulatorStack.push(dirtyBoardData);

#else

    pos.do_move(move, st, givesCheck, &tt);

#endif

    if (ss != nullptr)
    {
        // currentMove(迴ｾ蝨ｨ謗｢邏｢荳ｭ縺ｮ謖・＠謇・縺ｮ譖ｴ譁ｰ
        ss->currentMove = move;

#if STOCKFISH
        ss->currentMove = move;
        ss->continuationHistory =
          &contHistRoot[ss->inCheck][capture][dirtyBoardData.dp.pc][move.to_sq()];
        ss->continuationCorrectionHistory =
          &continuationCorrectionHistory[dirtyBoardData.dp.pc][move.to_sq()];
#else
        // 繧・・縺・ｉ邇九→Stockfish縺ｧ縺ｯ縲．irtyPiece縺ｮ讒矩縺碁＆縺・＠縲・
        // 繧・・縺・ｉ邇九〒縺ｯ縲∫ｧｻ蜍募ｾ後・鬧・謌舌ｊ縺ｮ謖・＠謇九↑繧画・繧企ｧ・繧剃ｽｿ縺・◆縺・・
        Piece dp_pc             = pos.moved_piece(move);
        ss->continuationHistory = &contHistRoot[ss->inCheck][capture][dp_pc][move.to_sq()];
        ss->continuationCorrectionHistory = &continuationCorrectionHistory[dp_pc][move.to_sq()];
#endif
    }
}

void YaneuraOuWorker::do_null_move(Position& pos, StateInfo& st) {
    // M4: null move は再生できないので Move::none() を積む (提示時に弾く)
    if (deleg != nullptr && curPathLen < MAX_PLY)
        curPath[curPathLen++] = Move::none();
    pos.do_null_move(st, tt);
}

void YaneuraOuWorker::undo_move(Position& pos, const Move move) {

    // M4: 積んだ手順を戻す
    if (deleg != nullptr && curPathLen > 0)
        --curPathLen;


	pos.undo_move(move);

#if defined(EVAL_SFNN)
    //accumulatorStack.pop();
#endif
}

void YaneuraOuWorker::undo_null_move(Position& pos) {
    if (deleg != nullptr && curPathLen > 0)
        --curPathLen;
    pos.undo_null_move();
}


// Reset histories, usually before a new game
// 螻･豁ｴ繧偵Μ繧ｻ繝・ヨ縺吶ｋ縲る壼ｸｸ縺ｯ譁ｰ縺励＞繧ｲ繝ｼ繝縺ｮ蜑阪↓螳溯｡後＆繧後ｋ縲・
void YaneuraOuWorker::clear() {

	// TODO : 縺ゅ→縺ｧ隱ｿ謨ｴ縺吶ｋ縲ＱawnHistory.fill(-1238@);繧ゅ・
	mainHistory.fill(SPSA::MAIN_HIST_INIT);
    captureHistory.fill(SPSA::CAPTURE_HIST_INIT);
    dropContHistory.fill(0);  // 鬧呈遠縺｡蟆ら畑 Drop Continuation History
    threatHistory.fill(0);    // Threat Continuation History (Phase B)

    // Each thread is responsible for clearing their part of shared history
    sharedHistory.correctionHistory.clear_range(0, numaThreadIdx, numaTotal);
    sharedHistory.pawnHistory.clear_range(SPSA::PAWN_HIST_INIT, numaThreadIdx, numaTotal);

	ttMoveHistory = 0;

    for (auto& to : continuationCorrectionHistory)
        for (auto& h : to)
            h.fill(SPSA::CONT_CORR_HIST_INIT);

    //     縺ｻ縺ｨ繧薙←縺ｮ螻･豁ｴ繧ｨ繝ｳ繝医Μ縺後＞縺壹ｌ縺ｫ縺帙ｈ蠕後〒雋縺ｫ縺ｪ繧九◆繧√・
    //     髢句ｧ句､繧偵梧ｭ｣縺励＞縲肴婿蜷代↓蟆代＠繧ｷ繝輔ヨ縺輔○繧九◆繧√∬ｲ縺ｮ謨ｰ縺ｧ蝓九ａ縺ｦ縺・ｋ縲・
    //     縺薙・蜉ｹ譫懊・縲∵ｷｱ蠎ｦ縺梧ｷｱ縺上↑繧九⊇縺ｩ阮・ｌ繧九・縺ｧ縲・聞譎る俣諤晁・＆縺帙ｋ譎ゅ↓縺ｯ
    //     縺ゅ∪繧頑э蜻ｳ縺後↑縺・′縲∫┌鬧・〒縺ｯ縺ｪ縺・ｉ縺励＞縲・
    //     cf. Tweak history initialization : https://github.com/official-stockfish/Stockfish/commit/7d44b43b3ceb2eebc756709432a0e291f885a1d2

	// p2shogi 2026-11-01: 参照先を決める (clear() は isready / usinewgame ごとに呼ばれるので
	//   option の変更がここで反映される)。 既定 0 なら自分の配列 = 従来と bit-exact。
	//   🚨確保は SharedHistories の構築時に済んでいる (遅延確保は thread 並列で data race)。
	//   🚨共有時は全 thread が同じ領域を同じ値で埋める。 値が同一なので競合は無害で、
	//     usinewgame のときだけなので区間分割の複雑さを足さない (意図的な簡略化)。
	contHistRoot = SPSA::SHARED_CONT_HIST ? sharedHistory.contHist->h : continuationHistory;

	for (bool inCheck : {false, true})
        for (StatsType c : {NoCaptures, Captures})
            for (auto& to : contHistRoot[inCheck][c])
                for (auto& h : to)
                    h.fill(SPSA::CONT_HIST_INIT);

	// reductions table縺ｮ蛻晄悄蛹・縺薙ｌ縺ｯWorker縺斐→縺梧戟縺､繧医≧縺ｫ螟画峩縺輔ｌ縺・
    for (size_t i = 1; i < reductions.size(); ++i)
        reductions[i] = int(SPSA::LMR_TABLE_COEFF / 128.0 * std::log(i));

	// 統 lowPlyHistory縺ｮ蛻晄悄蛹悶・縲∝ｯｾ螻縺斐→縺ｧ縺ｯ縺ｪ縺上∝ｱ髱｢縺斐→("go"縺ｮ縺斐→)縺ｫ螟画峩縺輔ｌ縺溘・

#if defined(EVAL_SFNN)
    refreshTable.clear(networks[numaAccessToken]);
#endif
}

// qsearch<PV>()繧探T hit縺ｪ縺玲桶縺・〒蜊倡峡蜻ｼ縺ｳ蜃ｺ縺励☆繧九◆繧√・繝倥Ν繝代・縲・
// qsearch蜀・〒Stack::pv縺ｫ讒狗ｯ峨＆繧後◆PV繧恥v縺ｸ霑斐☆縺ｮ縺ｧ縲∝他縺ｳ蜃ｺ縺怜・縺ｯ縺薙・PV繧壇o_move()縺励※
// qsearch縺ｮleaf node螻髱｢繧貞ｾ励ｉ繧後ｋ縲る壼ｸｸ謗｢邏｢縺ｮinfo pv陦ｨ遉ｺ縺ｮ繧医≧縺ｫTT繧定ｾｿ縺｣縺ｦPV繧貞ｻｶ髟ｷ縺励↑縺・・
Value YaneuraOuWorker::qsearch_pv(Position& pos, PVMoves& pv) {

#if defined(USE_SFNN)
    // 謗｢邏｢髢句ｧ区凾縺ｨ蜷梧ｧ倥∝・蝗枡valuate()縺ｧ蟾ｮ蛻・峩譁ｰ繧剃ｽｿ繧上↑縺・憾諷九↓謌ｻ縺吶・
    accumulatorStack.reset();
#endif

    nodes.store(0, std::memory_order_relaxed);
    selDepth = 0;
    lowPlyHistory.fill(SPSA::LOW_PLY_HIST_INIT);
    pv.clear();

    // iterative_deepening()縺ｨ蜷後§菴咏區縺､縺拘tack蛻晄悄蛹悶・
    Stack  stack[MAX_PLY + 10] = {};
    Stack* ss                  = stack + 7;

    for (int i = 7; i > 0; --i)
    {
        (ss - i)->continuationHistory =
          &contHistRoot[0][0][NO_PIECE][0];
        (ss - i)->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];
        (ss - i)->staticEval                    = VALUE_NONE;
        (ss - i)->currentMove                   = Move::none();
    }

    for (int i = 0; i <= MAX_PLY + 2; ++i)
        (ss + i)->ply = i;

    ss->pv          = &pv;
    ss->currentMove = Move::none();
    ss->staticEval  = VALUE_NONE;
    ss->ttPv        = false;
    ss->ttHit       = false;

    pos.set_ekr(main_manager()->search_options.enteringKingRule);

    return qsearch<PV, false>(pos, ss, -VALUE_INFINITE, VALUE_INFINITE);
}

// -----------------------
//      騾壼ｸｸ謗｢邏｢
// -----------------------

// 庁  search()縺ｯ縲∵怙蛻昴（terative_deepening()縺ｮ縺ｪ縺九°繧牙他縺ｳ蜃ｺ縺輔ｌ繧九・
//     縺薙ｌ縺ｯ莉ｮ諠ｳ髢｢謨ｰ縺ｫ縺ｯ縺ｪ縺｣縺ｦ縺・↑縺・・縺ｧ縲∽ｻｮ諠ｳ髢｢謨ｰ蜻ｼ縺ｳ蜃ｺ縺励・overhead縺ｯ縺ｪ縺・・

// Main search function for both PV and non-PV nodes
// PV , non-PV node蜈ｱ逕ｨ縺ｮ繝｡繧､繝ｳ縺ｮ謗｢邏｢髢｢謨ｰ縲・

// cutNode : LMR縺ｧ謔ｪ縺昴≧縺ｪ謖・＠謇九↓蟇ｾ縺励※reduction驥上ｒ蠅励ｄ縺冢ode

#if defined(P2_SRCH_BUNDLE_A)
// 束 A (L6): search<NonPV> の呼び出し口。
//   search<NonPV> は入口で prologue (大きなフレームの __chkstk を含む) を済ませてから
//   `depth <= 0` を見て qsearch<NonPV> に行く。 呼び出し側で先に振り分けて、 その往復を省く。
//   意味は search の冒頭と同じ: 統計 (P2Stat) を数え、 qsearch<NonPV>(pos, ss, alpha, beta) を返す。
//   search の入口の TLS 書き込み (L5) は、 qsearch<NonPV> の入口が同じ値 (lazy_head = !PvNode = true /
//   alpha / beta) で書き直すので、 ここで書かなくても評価関数から見える値は同じ。
//   P2_CORR_PREFETCH の prefetch は値を変えないので省いてよい。
inline Value YaneuraOuWorker::search_nonpv(
  Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode) {
    if (depth <= 0)
    {
        if (P2Stat::enabled()) P2Stat::main_nodes.fetch_add(1, std::memory_order_relaxed);
#if defined(P2_SRCH_BUNDLE_A_VERIFY)
        BundleA::check(BundleA::V_L6_DISPATCH, 0, 0);
#endif
        return qsearch<NonPV>(pos, ss, alpha, beta);
    }
    return search<NonPV>(pos, ss, alpha, beta, depth, cutNode);
}

#endif
template<NodeType nodeType>
Value YaneuraOuWorker::search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode)
{
    if (P2Stat::enabled()) P2Stat::main_nodes.fetch_add(1, std::memory_order_relaxed);
    // -----------------------
    //     node縺ｮ遞ｮ鬘・
    // -----------------------

	// PV node縺ｧ縺ゅｋ縺・
	// 庁 root node縺ｯ蠢・★PV node縺ｫ蜷ｫ縺ｾ繧後ｋ縲・

    constexpr bool PvNode   = nodeType != NonPV;

	// root node縺ｧ縺ゅｋ縺・
    constexpr bool rootNode = nodeType == Root;

	// allNode縺ｧ縺ゅｋ縺九・
    // 統 allNode縺ｨ縺ｯ縲￣vNode縺ｧ繧ゅ↑縺縦utNode縺ｧ繧ゅ↑縺・ode縺ｮ縺薙→縲・
    //     allNode縺ｨ縺ｯ縲√ご繝ｼ繝譛ｨ謗｢邏｢縺ｧ縲∝・縺ｦ縺ｮ蟄舌ヮ繝ｼ繝峨ｒ隧穂ｾ｡縺吶ｋ蠢・ｦ√′縺ゅｋ繝弱・繝峨・縺薙→縲・

    const bool     allNode  = !(PvNode || cutNode);

#if defined(EVAL_P2NNUE)
    // 2026-11-06 **correction history の prefetch** (設計 fable / 実装 opus)。
    //   correction_value (:966) は node ごとに pawn / minor / nonPawn W,B の **4 本**を引く。
    //   表は 65,536 x 20 B = 1.3 MB で L3 に乗る大きさだが、 TT 512 MB が L3 を絶えず流すので
    //   実際には DRAM から来ている可能性が高い。 node の入口で触っておけば
    //   TT probe の遅延 (300〜500 cycle) に重ねられる。
    //   🚨値は読まないので **bit-exact** (prefetch に副作用は無い)。
    //   🚨一括で増やすのは逆効果 (FT の pfall は -6.9%)。 ここは node につき 4 本だけ。
#if defined(P2_CORR_PREFETCH)
    prefetch((void*)&sharedHistory.pawn_correction_entry(pos));
    prefetch((void*)&sharedHistory.minor_piece_correction_entry(pos));
    prefetch((void*)&sharedHistory.nonpawn_correction_entry<WHITE>(pos));
    prefetch((void*)&sharedHistory.nonpawn_correction_entry<BLACK>(pos));
#endif
#endif

#if defined(EVAL_P2NNUE)
    // lazy-head: 髱霸V繝弱・繝峨〒縺ｯ隧穂ｾ｡髢｢謨ｰ縺ｮ head 鄒､繧堤怐逡･(rust 蜷梧婿驥・ 霑台ｼｼ縺ｧ鬮倬溷喧)縲・
#if defined(P2_SRCH_BUNDLE_A)
    // 束 A (L5): 評価関数がこの TLS を読まない設定 (tlsNeed* = false) なら書き込みを省く。
    //   判定と根拠は p2nnue.cpp の p2_srch_tls_needs()。 go の開始時に決め、 探索中は不変。
    if (tlsNeedLazyHead)
        Eval::p2_lazy_head = !PvNode;
    if (tlsNeedWin)
    {
        Eval::p2_win_a = (int) alpha;
        Eval::p2_win_b = (int) beta;
    }
#if defined(P2_SRCH_BUNDLE_A_VERIFY)
    bundle_a_verify_tls();
#endif
#else
    Eval::p2_lazy_head = !PvNode;
    // lazy eval(margin): 謗｢邏｢遯薙ｒ eval 縺ｫ莨昴∴繧九・
    Eval::p2_win_a = (int)alpha; Eval::p2_win_b = (int)beta;
#endif
#endif

    // Dive into quiescence search when the depth reaches zero
    // 谿九ｊ謗｢邏｢豺ｱ縺輔′1謇区悴貅縺ｧ縺ゅｋ縺ｪ繧臥樟蝨ｨ縺ｮ螻髱｢縺ｮ縺ｾ縺ｾ髱呎ｭ｢謗｢邏｢繧貞他縺ｳ蜃ｺ縺・
    if (depth <= 0)
        return qsearch<PvNode ? PV : NonPV>(pos, ss, alpha, beta);

    // Limit the depth if extensions made it too large
    // 諡｡蠑ｵ縺ｫ繧医▲縺ｦ豺ｱ縺輔′螟ｧ縺阪￥縺ｪ繧翫☆縺弱◆蝣ｴ蜷医∵ｷｱ縺輔ｒ蛻ｶ髯舌＠縺ｾ縺・

    depth = std::min(depth, MAX_PLY - 1);

	// 統 谺｡縺ｮ謖・＠謇九〒蠑輔″蛻・￠縺ｫ謖√■霎ｼ繧√※縺九▽縲｜eta縺悟ｼ輔″蛻・￠縺ｮ繧ｹ繧ｳ繧｢繧医ｊ菴弱＞縺ｪ繧・
    //     譌ｩ譛滓椌蛻医ｊ縺悟ｮ滓命縺ｧ縺阪ｋ縲・
    // ､・蟆・｣九□縺ｨ縺ゅ∪繧雁鴻譌･謇九′襍ｷ縺薙ｉ縺ｪ縺・・縺ｧ蜉ｹ譫懊′縺ｪ縺輔￡縲よ治逕ｨ縺励↑縺・・

#if STOCKFISH
    // Check if we have an upcoming move that draws by repetition
    // 逶ｴ霑代・謇九′郢ｰ繧願ｿ斐＠縺ｫ繧医ｋ蠑輔″蛻・￠縺ｫ縺ｪ繧九°繧堤｢ｺ隱阪＠縺ｾ縺・

	if (!rootNode && alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply))
    {
        alpha = value_draw(nodes);
        if (alpha >= beta)
            return alpha;
    }
#endif

	ASSERT_LV3(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    ASSERT_LV3(PvNode || (alpha == beta - 1));
    ASSERT_LV3(0 < depth && depth < MAX_PLY);
    // IID繧貞性繧√￣vNode縺ｧ縺ｯcutNode縺ｧ蜻ｼ縺ｳ蜃ｺ縺輔↑縺・・
    ASSERT_LV3(!(PvNode && cutNode));

	// -----------------------
    //     螟画焚螳｣險
    // -----------------------

	// pv : 縺薙・node縺九ｉ縺ｮPV line(隱ｭ縺ｿ遲・
    // st : do_move()縺吶ｋ縺ｨ縺阪↓蠢・ｦ・

    PVMoves   pv;
	StateInfo st;

	// posKey       : 縺薙・node縺ｮhash key

	Key        posKey;

    // move			: MovePicker縺九ｉ1謇九★縺､繧ゅｉ縺・→縺阪・荳譎ょ､画焚
    // excludedMove	: singular extemsion縺ｮ縺ｨ縺阪↓髯､螟悶☆繧区欠縺玲焔
    // bestMove		: 縺薙・node縺ｮbest move

	Move  move, excludedMove, bestMove;

	// extension	: 蟒ｶ髟ｷ縺吶ｋ豺ｱ縺・
	// newDepth		: 譁ｰ縺励＞depth(谿九ｊ謗｢邏｢豺ｱ縺・

	Depth extension, newDepth;

    // bestValue	: 縺薙・node縺ｮbest縺ｪ謗｢邏｢繧ｹ繧ｳ繧｢
    // value		: 謗｢邏｢繧ｹ繧ｳ繧｢繧貞女縺大叙繧倶ｸ譎ょ､画焚
    // eval			: 縺薙・node縺ｮ髱咏噪隧穂ｾ｡蛟､(縺ｮ隕狗ｩ阪ｊ)
    // maxValue     : table base probe縺ｫ逕ｨ縺・ｋ縲を沒・蟆・｣九□縺ｨ逕ｨ縺・↑縺・・
    // probCutBeta  : prob cut縺吶ｋ譎ゅ・beta縺ｮ蛟､縲・
#if STOCKFISH
	Value bestValue, value, eval, maxValue, probCutBeta;
#else
	Value bestValue, value, eval, probCutBeta;
#endif
	// givesCheck			: move縺ｫ繧医▲縺ｦ邇区焔縺ｫ縺ｪ繧九・縺・
	// improving			: 逶ｴ蜑阪・node縺九ｉ隧穂ｾ｡蛟､縺御ｸ翫′縺｣縺ｦ縺阪※縺・ｋ縺ｮ縺・
	//   縺薙・繝輔Λ繧ｰ繧貞推遞ｮ譫晏・繧翫・margin縺ｮ豎ｺ螳壹↓逕ｨ縺・ｋ
	//   cf. Tweak probcut margin with 'improving' flag : https://github.com/official-stockfish/Stockfish/commit/c5f6bd517c68e16c3ead7892e1d83a6b1bb89b69
	//   cf. Use evaluation trend to adjust futility margin : https://github.com/official-stockfish/Stockfish/commit/65c3bb8586eba11277f8297ef0f55c121772d82c
	// priorCapture         : 1縺､蜑阪・螻髱｢縺ｯ鬧偵ｒ蜿悶ｋ謖・＠謇九°・・
	// opponentWorsening    : 逶ｸ謇九・迥ｶ豕√′謔ｪ蛹悶＠縺ｦ縺・ｋ縺九・繝輔Λ繧ｰ

	bool  givesCheck, improving, priorCapture, opponentWorsening;

	// capture              : move縺碁ｧ偵ｒ謐慕佐縺吶ｋ謖・＠謇九ｂ縺励￥縺ｯ豁ｩ繧呈・繧区焔縺ｧ縺ゅｋ縺・
    // ttCapture			: 鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇九′capture縺吶ｋ謖・＠謇九〒縺ゅｋ縺・

	bool  capture, ttCapture;

	// priorReduction       : 1謇句燕縺ｮ螻髱｢縺ｧ縺ｮreduction縺ｮ驥・
	// movedPiece           :  move縺ｫ繧医▲縺ｦ遘ｻ蜍輔＆縺帙ｋ鬧・

	int   priorReduction;
    Piece movedPiece;

	// capturesSearched : 縺薙・node縺ｧ逕滓・縺励◆縲｀ovePicker縺九ｉ蜿門ｾ励＠縺滄ｧ偵ｒ謐慕佐縺吶ｋ謖・＠謇九ｒ鬆・分縺ｫ譬ｼ邏阪☆繧・ (+豁ｩ縺ｮ謌舌ｊ)
    // quietsSearched   : 縺薙・node縺ｧ逕滓・縺励◆縲｀ovePicker縺九ｉ蜿門ｾ励＠縺滄ｧ偵ｒ謐慕佐縺励↑縺・欠縺玲焔繧帝・分縺ｫ譬ｼ邏阪☆繧・-豁ｩ縺ｮ謌舌ｊ)

    SearchedList capturesSearched;
    SearchedList quietsSearched;

	// -----------------------
    // Step 1. Initialize node
	// Step 1. 繝弱・繝峨・蛻晄悄蛹・
    // -----------------------

	//     node縺ｮ蛻晄悄蛹・

    ss->inCheck        = pos.checkers();
    priorCapture       = pos.captured_piece();
    Color us           = pos.side_to_move();
    ss->moveCount      = 0;
    bestValue          = -VALUE_INFINITE;
#if STOCKFISH
    maxValue           =  VALUE_INFINITE;
    // 統 蟆・｣九〒縺ｯtable probe菴ｿ縺｣縺ｦ縺・↑縺・・縺ｧmaxValue縺ｯ髢｢菫ゅ↑縺・・
#else
    // 繧・・縺・ｉ邇区爾邏｢縺ｧ霑ｽ蜉縺励◆諤晁・お繝ｳ繧ｸ繝ｳ繧ｪ繝励す繝ｧ繝ｳ
    auto& search_options = main_manager()->search_options;
#endif

    // 蜑榊屓縺ｮ蜿榊ｾｩ豺ｱ蛹悶〒蠕励◆PV line繧定ｾｿ縺｣縺ｦ縺・ｋ縺九・
    // Stockfish縺ｧ縺ｯ縲√％縺ｮline荳翫〒IIR縺ｨquiet shallow pruning繧呈椛蛻ｶ縺励※縲・
    // 蜑榊屓縺ｮPV繧呈ｵ・＞譫晏・繧翫〒螢翫＠縺ｫ縺上￥縺励※縺・ｋ縲・
    ss->followPV = rootNode
                || ((ss - 1)->followPV && static_cast<size_t>(ss->ply - 1) < lastIterationPV.size()
                    && (ss - 1)->currentMove == lastIterationPV[ss->ply - 1]);

#if defined(USE_CLASSIC_EVAL) && defined(USE_LAZY_EVALUATE)
	// 統 谺｡縺ｮnode縺ｫ陦後￥縺ｾ縺ｧ縺ｫevaluate()縺菊valuate_with_no_return()繧貞他縺ｳ蜃ｺ縺吶％縺ｨ繧剃ｿ晁ｨｼ縺励※
	//     evaluate蜀・・蟾ｮ蛻・ｨ育ｮ励ｒ菫・＆縺ｪ縺代ｌ縺ｰ縺ｪ繧峨↑縺・・
    bool evaluated = false;
    auto evaluate  = [&](Position& pos) {
        evaluated = true;
        return this->evaluate(pos);
    };
    auto do_move = [&](Position & pos, Move move, StateInfo& st, bool givesCheck, Stack* ss) {
        if (!evaluated)
        {
            evaluated = true;
            Eval::evaluate_with_no_return(pos);
        }
        this->do_move(pos, move, st, givesCheck, ss);
    };

	// ､・蜷後§蜷榊燕縺ｧ蜻ｼ縺ｳ蛻・￠縺ｧ縺阪↑縺・・縺ｧ縲・
	//     縺薙■繧峨ｒ蜷榊燕繧・do_move_ 縺ｫ縺吶ｋ縲・
    auto do_move_ = [&](Position & pos, Move move, StateInfo& st, Stack* ss) {
        if (!evaluated)
        {
            evaluated = true;
            Eval::evaluate_with_no_return(pos);
        }
        this->do_move(pos, move, st, ss);
    };
    auto do_null_move = [&](Position& pos, StateInfo& st) {
        if (!evaluated)
        {
            evaluated = true;
            Eval::evaluate_with_no_return(pos);
        }
        this->do_null_move(pos, st);
    };
#else
    auto do_move_ = [&](Position& pos, Move move, StateInfo& st, Stack* ss) { this->do_move(pos, move, st, ss); };
#endif

	// 東 Timer縺ｮ逶｣隕・

    // Check for the available remaining time
    // 谿九ｊ縺ｮ蛻ｩ逕ｨ蜿ｯ閭ｽ縺ｪ譎る俣繧堤｢ｺ隱阪＠縺ｾ縺・
    // 庁 縺薙ｌ縺ｯ繝｡繧､繝ｳ繧ｹ繝ｬ繝・ラ縺ｮ縺ｿ縺瑚｡後↑縺・・
	if (is_mainthread())
        main_manager()->check_time(*this);


    // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
	// selDepth諠・ｱ繧竪UI縺ｫ騾∽ｿ｡縺吶ｋ縺溘ａ縺ｫ菴ｿ逕ｨ縺励∪縺呻ｼ・elDepth縺ｯ1縺九ｉ繧ｫ繧ｦ繝ｳ繝医＠縲｝ly縺ｯ0縺九ｉ繧ｫ繧ｦ繝ｳ繝医＠縺ｾ縺呻ｼ・
    if (PvNode && selDepth < ss->ply + 1)
        selDepth = ss->ply + 1;

	// -----------------------
    //  RootNode莉･螟悶〒縺ｮ蜃ｦ逅・
    // -----------------------

    if (!rootNode)
    {
        // -----------------------
        // Step 2. Check for aborted search and immediate draw
        // Step 2. 荳ｭ譁ｭ縺輔ｌ縺滓爾邏｢縺翫ｈ縺ｳ蜊ｳ譎ょｼ輔″蛻・￠繧堤｢ｺ隱阪＠縺ｾ縺・
        // -----------------------

		// 謗｢邏｢縺ｮ荳ｭ譁ｭ縺ｨ縲∝ｼ輔″蛻・￠縺ｫ縺､縺・※繝√ぉ繝・け縺吶ｋ
        // 騾｣邯夂視謇九↓繧医ｋ蜊・律謇九√♀繧医・騾壼ｸｸ縺ｮ蜊・律謇九∝━遲牙ｱ髱｢繝ｻ蜉｣遲牙ｱ髱｢縲・

        // 騾｣邯夂視謇九↓繧医ｋ蜊・律謇九↓蟇ｾ縺励※draw_value()縺ｯ縲∬ｩｰ縺ｿ縺ｮ繧ｹ繧ｳ繧｢繧定ｿ斐☆縺ｮ縺ｧ縲〉oot縺九ｉ縺ｮ謇区焚繧定・・縺励◆繧ｹ繧ｳ繧｢縺ｫ螟画鋤縺吶ｋ蠢・ｦ√′縺ゅｋ縲・
        // 縺昴％縺ｧ縲」alue_from_tt()縺ｧ螟画鋤縺励※縺九ｉ霑斐☆縺ｮ縺梧ｭ｣隗｣縲・

        // 謨吝ｸｫ螻髱｢逕滓・譎ゅ↓縺ｯ縲√％繧後ｒ繧ｪ繝輔↓縺励◆縺ｻ縺・′濶ｯ縺・°繧ら衍繧後↑縺・・
        // 縺溘□縺励√◎縺ｮ縺ｨ縺阪〒縺ゅ▲縺ｦ繧る｣邯夂視謇九・蜊・律謇九・蛻､螳壹・譛牙柑縺ｫ縺励※縺翫￥縲・

        // 竊偵蜆ｪ遲牙ｱ髱｢繝ｻ蜉｣遲牙ｱ髱｢縺ｯroot繧医ｊ驕｡縺｣縺ｦ蛻､螳壹＠縺ｪ縺・・
        // (USI縺ｧ蜃ｺ蜉帙☆繧玖ｩ穂ｾ｡蛟､縺祁ALUE_SUPERIOR縺ｫ縺ｪ繧九・縺ｯ縺｡繧・▲縺ｨ雖後□縺励・
        // 縲蜆ｪ遲牙ｱ髱｢縺ｫ遯∝・縺吶ｋ縺九ｉ縺ｨ險縺｣縺ｦ蜊ｳ隧ｰ縺ｿ繧帝・′縺吶・繧ゅ■繧・▲縺ｨ雖・
        // cf. https://github.com/yaneurao/YaneuraOu/issues/264

		// 譛螟ｧ謇区焚繧定ｶ・∴縺ｦ縺・ｋ縲√ｂ縺励￥縺ｯ蜊・律謇九∝●豁｢蜻ｽ莉､縺梧擂縺ｦ縺・ｋ縲・

#if STOCKFISH
        if (threads.stop.load(std::memory_order_relaxed) || pos.is_draw(ss->ply)
            || ss->ply >= MAX_PLY)
            return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos)
                                                        : value_draw(nodes);
#else

		auto draw_type = pos.is_repetition(ss->ply, true);
        if (draw_type != REPETITION_NONE)
        { 
            if (draw_type == REPETITION_DRAW)
				// 騾壼ｸｸ縺ｮ蜊・律謇九・譎ゅ・繧・ｉ縺弱ｒ謖√◆縺帙ｋ縲・
				// 庁 蠑輔″蛻・￠縺ｮ繧ｹ繧ｳ繧｢v縺ｯ abs(vﾂｱ1) <= VALUE_MAX_EVAL縺ｧ縺ゅｋ縺薙→縺御ｿ晁ｨｼ縺輔ｌ縺ｦ縺・ｋ縺ｮ縺ｧ縲・
				//     value_from_tt()縺ｧ縺ｮ螟画鋤縺ｯ荳崎ｦ√・
                return draw_value(draw_type, pos.side_to_move()) + value_draw(nodes);
            else
	            return value_from_tt(draw_value(draw_type, pos.side_to_move()), ss->ply);
        }

		// 東 蟆・｣九〒縺ｯ謇区焚繧定ｶ・∴縺溘ｉ辟｡譚｡莉ｶ縺ｧ蠑輔″蛻・￠謇ｱ縺・・
        if (threads.stop.load(std::memory_order_relaxed) || ss->ply >= MAX_PLY
            || pos.game_ply() > search_options.max_moves_to_draw)
            return draw_value(REPETITION_DRAW, pos.side_to_move()) + value_draw(nodes);
#endif

		/*
		統 蛯呵・

			256謇九Ν繝ｼ繝ｫ縺ｧ
			1. 256謇狗岼縺ｮ螻髱｢縺ｧ蛻､螳壹ｒ陦後≧蝣ｴ蜷医・縲・
				縲瑚ｩｰ縺ｾ縺輔ｌ縺ｦ縺・↑縺・√°縺､縲・｣邯夂視謇九・蜊・律謇九′謌千ｫ九＠縺ｦ縺・↑縺・阪↑繧峨・縲・
				蠑輔″蛻・￠縺ｨ縺励※return縺励※濶ｯ縺・・

			2. 257謇狗岼縺ｮ螻髱｢縺ｧ蛻､螳壹ｒ陦後≧蝣ｴ蜷医・縲・
				縺薙・螻髱｢縺ｫ蛻ｰ驕斐＠縺溘→縺・≧縺薙→縺ｯ縲・56謇狗岼縺ｮ螻髱｢縺ｧ蜷域ｳ墓焔縺後≠縺｣縺溘→縺・≧縺薙→縺縺九ｉ縲・
				蠑輔″蛻・￠縺ｨ縺励※return縺励※濶ｯ縺・・

			縺ｨ縺・≧縺薙→縺ｫ縺ｪ繧九・.縺ｮ譁ｹ蠑上〒繧・▲縺溘⊇縺・′縲・56謇狗岼縺ｮ螻髱｢縺ｧ謖・＠謇狗函謌舌→縺・
			1謇九★縺､隧ｦ縺励※縺・￥縺ｮ縺ｨ縺九′荳ｸ縺斐→遶ｯ謚倥ｌ繧九・縺ｧ謗｢邏｢蜉ｹ邇・・濶ｯ縺・・縺縺後√さ繝ｼ繝峨′隍・尅縺ｫ縺ｪ繧九・

			2.縺ｮ譁ｹ蠑上〒繧・▲縺ｦ縺・＞縺ｪ繧峨・縲・57謇狗岼縺ｮ螻髱｢縺ｪ繧牙腰縺ｫreturn縺吶ｋ縺縺代〒貂医・縲・
			謗｢邏｢蜉ｹ邇・・蟆代＠謔ｪ縺・′縲√さ繝ｼ繝峨・讌ｵ繧√※繧ｷ繝ｳ繝励Ν縺ｫ縺ｪ繧九・

			縺ｾ縺溘・56謇九Ν繝ｼ繝ｫ縺ｧ256謇狗岼縺ｧ縺ゅｋ謖・＠謇九ｒ謖・＠縺ｦ縲・57謇狗岼縺ｮ螻髱｢縺ｧ騾｣邯夂視謇九・蜊・律謇九′
			謌千ｫ九☆繧九→縺阪・縺薙ｌ縺ｯ髱槫粋豕墓焔謇ｱ縺・ｒ縺吶∋縺阪□縺ｨ諤昴≧縲・

			縺縺九ｉ縲・.縺ｮ譁ｹ蠑上〒蛻､螳壹☆繧九→縺阪・縲√％縺ｮ騾｣邯夂視謇九・蜊・律謇句愛螳壹ｒ蜈医↓繧・▲縺ｦ縺九ｉ縲・
			257謇狗岼縺ｮ螻髱｢縺ｧ縺ゅｋ縺九・蛻､螳壹ｒ陦後≧蠢・ｦ√′縺ゅｋ縲・

			荳願ｨ倥・繧ｳ繝ｼ繝峨・縲√◎縺・↑縺｣縺ｦ縺・ｋ縲・
		*/

		// -----------------------
		// Step 3. Mate distance pruning.
		// Step 3. 隧ｰ縺ｿ縺ｾ縺ｧ縺ｮ謇区焚縺ｫ繧医ｋ譫晏・繧・
		// -----------------------

        // Step 3. Mate distance pruning. Even if we mate at the next move our score
        // would be at best mate_in(ss->ply + 1), but if alpha is already bigger because
        // a shorter mate was found upward in the tree then there is no need to search
        // because we will never beat the current alpha. Same logic but with reversed
        // signs apply also in the opposite condition of being mated instead of giving
        // mate. In this case, return a fail-high score.

		// 繧ｹ繝・ャ繝・. 隧ｰ縺ｿ縺ｾ縺ｧ縺ｮ謇区焚縺ｫ繧医ｋ譫晏・繧翫ゅ◆縺ｨ縺域ｬ｡縺ｮ謇九〒繝｡繧､繝医＠縺ｦ繧ゅ√せ繧ｳ繧｢縺ｯ譛螟ｧ縺ｧ
        // mate_in(ss->ply + 1)縺ｨ縺ｪ繧翫∪縺吶ゅ＠縺九＠縲√ｂ縺預lpha縺後☆縺ｧ縺ｫ縺昴ｌ莉･荳翫〒縺ゅｌ縺ｰ縲・
        // 繝・Μ繝ｼ荳翫〒繧医ｊ遏ｭ縺・ｩｰ縺ｿ縺瑚ｦ九▽縺九▲縺溘％縺ｨ繧呈э蜻ｳ縺励√％繧御ｻ･荳頑爾邏｢縺吶ｋ蠢・ｦ√・縺ゅｊ縺ｾ縺帙ｓ縲・
        // 迴ｾ蝨ｨ縺ｮalpha繧剃ｸ雁屓繧九％縺ｨ縺ｯ縺ｧ縺阪↑縺・°繧峨〒縺吶・
        // 騾・↓縲∬ｩｰ縺ｿ縺輔ｌ繧句ｴ蜷医ｂ蜷梧ｧ倥・繝ｭ繧ｸ繝・け縺碁←逕ｨ縺輔ｌ縺ｾ縺吶′縲∫ｬｦ蜿ｷ縺碁・↓縺ｪ繧翫∪縺吶・
        // 縺薙・蝣ｴ蜷医’ail high繧定ｿ斐＠縺ｾ縺吶・

		/*
		   統 蛯呵・
        
			   root縺九ｉ5謇狗岼縺ｮ螻髱｢縺縺ｨ縺励※縲√％縺ｮnode縺ｮ繧ｹ繧ｳ繧｢縺・謇倶ｻ･蜀・〒
			   隧ｰ縺ｾ縺吶→縺阪・繧ｹ繧ｳ繧｢繧剃ｸ雁屓繧九％縺ｨ繧ゅ↑縺・＠縲・
			   5謇倶ｻ･蜀・〒隧ｰ縺ｾ縺輔○繧九→縺阪・繧ｹ繧ｳ繧｢繧剃ｸ句屓繧九％縺ｨ繧ゅ↑縺・・
			   縺昴％縺ｧ縲∥lpha , beta縺ｮ蛟､繧偵∪縺壹％縺ｮ遽・峇縺ｫ陬懈ｭ｣縺励◆縺ゅ→縲・
			   alpha縺恵eta蛟､繧定ｶ・∴縺ｦ縺・ｋ縺ｪ繧叡eta cut縺吶ｋ縲・
		*/

        alpha = std::max(mated_in(ss->ply), alpha);
        beta  = std::min(mate_in(ss->ply + 1), beta);
        if (alpha >= beta)
            return alpha;
    }

	// -----------------------
    //  謗｢邏｢Stack縺ｮ蛻晄悄蛹・
    // -----------------------

	// root縺九ｉ縺ｮ謇区焚
    ASSERT_LV3(0 <= ss->ply && ss->ply < MAX_PLY);

	// 蜑阪・謖・＠謇九〒遘ｻ蜍輔＆縺帙◆蜈医・蜊・岼
    // null move縺ｮ縺ｨ縺阪・is_ok() == false縺ｪ縺ｮ縺ｧSQ_NONE縺ｨ縺吶ｋ縲・
    Square prevSq  = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;
    bestMove       = Move::none();
    priorReduction = (ss - 1)->reduction;
    (ss - 1)->reduction = 0;
    ss->statScore       = 0;
    (ss + 2)->cutoffCnt = 0;

	// -----------------------
    // Step 4. Transposition table lookup.
    // Step 4. 鄂ｮ謠幄｡ｨ縺ｮ蜿ら・
    // -----------------------

	// 縺薙・node縺ｧ謗｢邏｢縺九ｉ髯､螟悶☆繧区欠縺玲焔縲Ｔs->excludedMove縺ｮ繧ｳ繝斐・縲・
    excludedMove                   = ss->excludedMove;

	/*
		統
			excludedMove縺後≠繧・singular extension譎・縺ｯ縲・
			蜑榊屓縺ｮ蜈ｨ謗｢邏｢縺ｮ鄂ｮ謠幄｡ｨ縺ｮ蛟､繧剃ｸ頑嶌縺阪☆繧矩Κ蛻・爾邏｢縺ｮ繧ｹ繧ｳ繧｢縺ｯ
			谺ｲ縺励￥縺ｪ縺・・縺ｧ縲‘xcluded move縺後≠繧句ｴ蜷医↓縺ｯ逡ｰ縺ｪ繧吃ash key繧堤畑縺・※
			逡ｰ縺ｪ繧亀TEntry繧定ｪｭ縺ｿ譖ｸ縺阪☆縺ｹ縺阪□縺ｨ諤昴≧縺後・
			Stockfish 16縺ｧ縲∝酔縺狼TEntry繧堤畑縺・ｋ繧医≧縺ｫ縺ｪ縺｣縺溘・
			(縺溘□縺容xcluded move縺後≠繧区凾縺ｫ謗｢邏｢縺励◆邨先棡縺ｯTTEntry縺ｫsave縺励↑縺・
			縺､縺ｾ繧翫｝robe縺励※諠・ｱ縺縺大茜逕ｨ縺吶ｋ諢溘§縺ｮ繧医≧縺縲よュ蝣ｱ縺ｯ菴ｿ縺医ｋ縺ｨ縺・≧縺薙→縺ｪ縺ｮ縺繧阪≧縺銀ｦ縲・

			posKey = excludedMove == Move::none() ? pos.hash_key() : pos.hash_key() ^ HASH_KEY(make_key(excludedMove));
			竊代％縺ｮ縺ｨ縺膏os.key()縺ｮbit0繧堤ｴ螢翫☆繧九％縺ｨ縺ｯ險ｱ縺輔ｌ縺ｪ縺・・縺ｧ縲［ake_key()縺ｧbit0縺ｯ繧ｯ繝ｪ繧｢縺励※縺翫￥縲・
			excludedMove縺勲ove::none()縺ｮ譎ゅ・key繧貞､画峩縺励※縺ｯ縺ｪ繧峨↑縺・・

			竊鉄tockfish 16縺ｧ逡ｰ縺ｪ繧亀TEntry繧剃ｽｿ繧上↑縺・ｈ縺・↓縺ｪ縺｣縺ｦ谺｡縺ｮ繧医≧縺ｫ蜊倡ｴ泌喧縺輔ｌ縺溘・
			   cf. https://github.com/official-stockfish/Stockfish/commit/8d3457a9966f8c744ab7f8536be408196ccd8af9
	*/

	/*
		涛
			excluded move縺ｫ縺､縺・※隧ｳ縺励￥縲・

			singular extension縺ｨ縺ｯ縲∫ｽｮ謠幄｡ｨ縺九ｉ諡ｾ縺｣縺ｦ縺阪◆謖・＠謇九□縺代′縺吶％縺ｶ繧九ｈ繧阪＠縺・欠縺玲焔縺ｧ縺ゅｋ譎ゅ・
			荳譛ｬ驕薙・螟牙喧縺縺九ｉ縲√％縺ｮ謖・＠謇九・繧ゅ▲縺ｨ蟒ｶ髟ｷ縺励※縺ゅ￡繧医≧縺ｨ縺・≧縺薙→縺ｧ縺ゅｋ縲るｧ偵・縺溘□謐ｨ縺ｦ縺ｪ縺ｩ縺ｧ
			謖・＠謇九ｒ蠑輔″莨ｸ縺ｰ縺吶ｈ縺・↑豌ｴ蟷ｳ邱壼柑譫懊ｒ謚代∴繧句ｽｹ蜑ｲ繧ゅ≠繧九・縺溘・繧・

			縺縺九ｉ縲∫ｽｮ謠幄｡ｨ縺ｮ謖・＠謇九ｒ髯､螟悶＠縺ｦ蜷後§node縺ｧ謗｢邏｢縺励↑縺翫☆蠢・ｦ√′縺ゅｋ縲・
			縺薙・譎ゅ・謗｢邏｢縺ｫ縺翫￠繧狗ｽｮ謠幄｡ｨ縺ｫ蜈・≠縺｣縺滓欠縺玲焔繧弾xcluded move縺ｨ蜻ｼ縺ｶ縲・

			縺､縺ｾ繧翫√％縺ｮ譎ゅ・謗｢邏｢邨先棡縺ｯ縲‘xcluded move繧帝勁螟悶＠縺ｦ蠕励ｉ繧後◆謗｢邏｢邨先棡縺ｪ縺ｮ縺ｧ縲・
			蜷後§TTEntry(鄂ｮ謠幄｡ｨ縺ｮ繧ｨ繝ｳ繝医Μ繝ｼ)縺ｫ譖ｸ縺榊・縺吶・縺ｯ縺翫°縺励＞繧上￠縺ｧ縺ゅｋ縲・

			縺縺九ｉexcluded move縺後≠繧区凾縺ｯ縲∝ｱ髱｢縺ｮhash key繧偵√％縺ｮexcluded move繧・
			閠・・縺励◆hash key縺ｫ螟画峩縺励※蛻･縺ｮTTEntry繧堤畑縺・ｋ繧医≧縺ｫ縺励※縺・◆縲・

			縺昴・繧ｳ繝ｼ繝峨′荳翫・ pos.hash_key() ^ HASH_KEY(make_key(excludedMove) 縺ｮ驛ｨ蛻・〒縺ゅｋ縲・
			(make_key縺ｯexcludedMove繧痴eed縺ｨ縺吶ｋ逍台ｼｼ荵ｱ謨ｰ繧堤函謌舌☆繧・

			縺ｨ縺薙ｍ縺後√％繧後ｒStockfish縺ｮ荳翫・commit縺ｯ縲∝ｻ・ｭ｢縺吶ｋ縺ｨ縺・≧縺ｮ縺ｧ縺ゅｋ縲・

			繝｡繝ｪ繝・ヨ縺ｨ縺励※縺ｯ縲［ake_key縺ｧ蜈ｨ辟ｶ驕輔≧TTEntry繧定ｦ九↓陦後￥縺ｨCPU縺ｮcache縺ｫmiss hit縺吶ｋ縺ｮ縺ｧ縲・
			縺昴％縺ｧ驕・￥縺ｪ繧九・縺縺後∝酔縺狼TEntry繧定ｦ九↓陦後￥縺ｪ繧峨・俣驕輔＞縺ｪ縺修PU cache縺ｫhit縺吶ｋ縺ｨ縺・≧繧ゅ・縺ｧ縺ゅｋ縲・
			縺ｾ縺溘∝・繧ｨ繝ｳ繝医Μ繝ｼ縺ｮ蛟､縺ｮ縺・■縲《taticEval(evaluate()縺励◆譎ゅ・蛟､)縺舌ｉ縺・・菴ｿ縺医ｋ繧薙§繧・・・溘→縺・≧縺薙→縺ｧ縺ゅｋ縲・

			繝・Γ繝ｪ繝・ヨ縺ｨ縺励※縺ｯ縲√％縺ｮ譎ゅ・謗｢邏｢邨先棡繧偵◎縺ｮTTEntry縺ｫ菫晏ｭ倥＠縺ｦ縺励∪縺・→縺昴ｌ縺ｯexcluded move縺後↑縺・凾縺ｮ
			謗｢邏｢邨先棡縺ｨ縺励※縺ｯ豁｣縺励￥縺ｪ縺・・縺ｧ縲√％縺ｮ繧医≧縺ｪ菫晏ｭ倥・縺ｧ縺阪↑縺・→縺・≧縺薙→縺ｧ縺ゅｋ縲・
			縺昴ｌ縺ｫ繧医ｊ縲∵ｬ｡蝗槭ｂ蜷後§excluded move縺ｧsingular extension縺吶ｋ譎ゅ↓莉雁屓縺ｮ謗｢邏｢邨先棡縺梧ｴｻ逕ｨ縺ｧ縺阪↑縺・
			縺ｨ縺・≧縺ｮ縺ｯ縺ゅｋ縲・

			縺昴・縺ｩ縺｡繧峨′蠕励↑縺ｮ縺九→縺・≧縺薙→縺ｮ繧医≧縺ｧ縺ゅｋ縲・
	*/

    posKey                         = pos.key();
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey, pos);
    if (P2Stat::enabled()) {
        P2Stat::tt_probe.fetch_add(1, std::memory_order_relaxed);
        if (ttHit) P2Stat::tt_hit.fetch_add(1, std::memory_order_relaxed);
        if (ttData.move) P2Stat::deep_node.fetch_add(1, std::memory_order_relaxed);
    }

    // Need further processing of the saved data
    // 菫晏ｭ倥＆繧後※縺・◆繝・・繧ｿ縺ｮ縺輔ｉ縺ｪ繧句・逅・′蠢・ｦ√〒縺・

	ss->ttHit    = ttHit;

	/*
		統
			鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇・
			鄂ｮ謠幄｡ｨ縺ｫhit縺励↑縺代ｌ縺ｰMove::none()
			RootNode縺ｧ縺ゅｋ縺ｪ繧峨・MultiPV縺ｪ縺ｩ縺ｧ繧・迴ｾ蝨ｨ豕ｨ逶ｮ縺励※縺・ｋ1謇九□縺代′繝吶せ繝医・謖・＠謇九→莉ｮ螳壹〒縺阪ｋ縺九ｉ縲・
			縺昴ｌ縺檎ｽｮ謠幄｡ｨ縺ｫ縺ゅ▲縺溘ｂ縺ｮ縺ｨ縺励※謖・＠謇九ｒ騾ｲ繧√ｋ縲・
	
		笞
	
		    TT縺ｫMove::win()繧よ欠縺玲焔縺ｨ縺励※譖ｸ縺榊・縺吝ｴ蜷医・縲・
		    tte->move()縺ｫ縺ｯMove::win()繧ょ性縺ｾ繧後※縺・ｋ蜿ｯ閭ｽ諤ｧ縺後≠繧九・
		    縺薙・譎ゅ｝os.to_move(MOVE_WIN) == Move::win()縺ｪ縺ｮ縺ｧ縲》tMove == Move::win()縺ｨ縺ｪ繧九・
		    竍ｨ 迴ｾ迥ｶ縺ｮ繧・・縺・ｉ邇九〒縺ｯ縲｀ove::win()縺ｯTT縺ｫ譖ｸ縺榊・縺輔↑縺・・
	*/

	ttData.move  = rootNode ? rootMoves[pvIdx].pv[0]
                 : ttHit    ? ttData.move
                            : Move::none();

	// 鄂ｮ謠幄｡ｨ荳翫・繧ｹ繧ｳ繧｢
    // 庁 鄂ｮ謠幄｡ｨ縺ｫhit縺励↑縺代ｌ縺ｰVALUE_NONE

    // singular search縺ｨIID縺ｨ縺ｮ繧ｹ繝ｬ繝・ラ遶ｶ蜷医ｒ閠・・縺励※縲》tValue , ttMove縺ｮ鬆・〒蜿悶ｊ蜃ｺ縺輔↑縺・→縺・￠縺ｪ縺・ｉ縺励＞縲・
    // cf. More robust interaction of singular search and iid : https://github.com/official-stockfish/Stockfish/commit/16b31bb249ccb9f4f625001f9772799d286e2f04

	ttData.value =
          ttHit ? value_from_tt(ttData.value, ss->ply /*, pos.rule50_count()*/) : VALUE_NONE;

	// 統 鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇九↓pseudo_legal縺ｧ縺ｯ縺ｪ縺・欠縺玲焔縺梧ｷｷ縺倥▲縺ｦ縺・◆繧峨・
	//     縺昴ｌ縺ｯ蜈亥ｾ後・螻髱｢繧帝俣驕輔∴縺溽ｽｮ謠幄｡ｨEntry縺ｫ譖ｸ縺榊・縺励※縺励∪縺｣縺ｦ縺・ｋ繝舌げ縲・
	ASSERT_LV3(pos.legal_promote(ttData.move));

    ss->ttPv     = excludedMove ? ss->ttPv : PvNode || (ttHit && ttData.is_pv);

	/*
		鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇九′capture縺ｧ縺ゅｋ縺九・
		鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇九′capture縺ｪ繧蛾ｫ倥＞遒ｺ邇・〒縺薙・謖・＠謇九′繝吶せ繝医↑縺ｮ縺ｧ縲∽ｻ悶・謖・＠謇九ｒ
		縺昴ｓ縺ｪ縺ｫ隱ｭ縺ｾ縺ｪ縺上※繧ょ､ｧ荳亥､ｫ縲ゅ↑縺ｮ縺ｧ縲√％縺ｮnode縺ｮ縺吶∋縺ｦ縺ｮ謖・＠謇九・reduction繧貞｢励ｄ縺吶・

		縺薙％縲…apture_or_promotion()縺ｨ縺議apture_or_pawn_promotion()縺ｨ縺玖牡縲・､峨∴縺ｦ縺ｿ縺溘′縲・
		迴ｾ蝨ｨ縺ｧ縺ｯ縲…apture()縺ｫ縺吶ｋ縺ｮ縺瑚憶縺・ｈ縺・□縲・2022/04/13]
		竊偵謐慕佐縺吶ｋ謖・＠謇九〒荳逡ｪ蟆上＆縺・ｾ｡蛟､荳頑・縺ｯ豁ｩ縺ｮ謐慕佐(+ 2*PAWN_VALUE)縺ｪ縺ｮ縺ｧ縺薙ｌ縺舌ｉ縺・・蟾ｮ縺ｫ縺ｪ繧九ｂ縺ｮ
			豁ｩ縺ｮ謌舌ｊ縲・ｦ吶・謌舌ｊ縲∵｡ゅ・謌舌ｊ縲縺舌ｉ縺・・隱ｿ縺ｹ縺ｦ繧り憶縺輔◎縺・↑窶ｦ縲・
		竊・Stockfish縺ｧcapture_stage()縺ｫ縺ｪ縺｣縺ｦ縺・ｋ縺ｨ縺薙ｍ縺ｯ縺昴ｌ縺ｫ蛟｣縺・％縺ｨ縺ｫ縺励◆縲・2023/11/05]
	*/

    ttCapture    = ttData.move && pos.capture_stage(ttData.move);

    // At this point, if excluded, skip straight to step 6, static eval. However,
    // to save indentation, we list the condition in all code between here and there.
	// 縺薙・譎らせ縺ｧ縲・勁螟悶＆繧後◆蝣ｴ蜷医・縲√せ繝・ャ繝・縺ｮ髱咏噪隧穂ｾ｡縺ｫ逶ｴ謗･騾ｲ縺ｿ縺ｾ縺吶・
    // 縺励°縺励√う繝ｳ繝・Φ繝医ｒ貂帙ｉ縺吶◆繧√↓縲√％縺薙°繧峨◎縺薙∪縺ｧ縺ｮ繧ｳ繝ｼ繝牙・縺ｮ譚｡莉ｶ繧偵☆縺ｹ縺ｦ險倩ｼ峨＠縺ｦ縺・∪縺吶・

	// 統 陬懆ｶｳ
    //
    //    鄂ｮ謠幄｡ｨ縺ｫhit縺励↑縺九▲縺滓凾縺ｯ縲￣V node縺ｮ縺ｨ縺阪□縺奏tPv縺ｨ縺励※謇ｱ縺・・
    //    縺薙ｌss->ttPV縺ｫ菫晏ｭ倥＠縺ｦ繧九￠縺ｩ縲《ingular縺ｮ蛻､螳夂ｭ峨〒search繧痴s+1縺ｧ縺ｯ縺ｪ縺峻s縺ｧ蜻ｼ縺ｳ蜃ｺ縺吶％縺ｨ縺後≠繧翫・
    //    縺昴・縺ｨ縺阪↓ss->ttPv縺檎ｴ螢翫＆繧後ｋ縲ゅ↑縺ｮ縺ｧ縲∫ｴ螢翫＠縺昴≧縺ｪ縺ｨ縺阪・逶ｴ蜑阪↓繝ｭ繝ｼ繧ｫ繝ｫ螟画焚縺ｫ菫晏ｭ倥☆繧九さ繝ｼ繝峨′譖ｸ縺・※縺ゅｋ縲・

    // At non-PV nodes we check for an early TT cutoff
    // 髱霸V繝弱・繝峨〒縺ｯ縲∵掠譛溘・TT繧ｫ繝・ヨ繧ｪ繝輔ｒ遒ｺ隱阪☆繧・

	// 統  PV node縺ｧ縺ｯ鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇九〒縺ｯ譫晏・繧翫＠縺ｪ縺・・
	//      PV node縺ｯ縺斐￥繧上★縺九＠縺九↑縺・＠縲・㍾隕√↑螟牙喧縺縺九ｉ譫晏・繧翫＠縺溘￥縺ｪ縺・・

	if (!PvNode
		&& !excludedMove
		&& ttData.depth > depth - (ttData.value <= beta) // 鄂ｮ謠幄｡ｨ縺ｫ逋ｻ骭ｲ縺輔ｌ縺ｦ縺・ｋ謗｢邏｢豺ｱ縺輔・縺ｻ縺・′豺ｱ縺上※
        && is_valid(ttData.value)   // Can happen when !ttHit or when access race in probe()
							        // !ttHit縺ｮ蝣ｴ蜷医ｄprobe()縺ｧ縺ｮ繧｢繧ｯ繧ｻ繧ｹ遶ｶ蜷域凾縺ｫ逋ｺ逕溘☆繧句庄閭ｽ諤ｧ縺後≠繧翫≧繧九・
        && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER))
        && (cutNode == (ttData.value >= beta) || depth > 5))

		/*
		統 隗｣隱ｬ
		
			ttValue縺御ｸ狗阜(逵溘・隧穂ｾ｡蛟､縺ｯ縺薙ｌ繧医ｊ螟ｧ縺阪＞)繧ゅ＠縺上・繧ｸ繝｣繧ｹ繝医↑蛟､縺ｧ縲√°縺､ttValue >= beta雜・∴縺ｪ繧叡eta cut縺輔ｌ繧・
			ttValue縺御ｸ顔阜(逵溘・隧穂ｾ｡蛟､縺ｯ縺薙ｌ繧医ｊ蟆上＆縺・縺縺後》te->depth()縺ｮ縺ｻ縺・′depth繧医ｊ豺ｱ縺・→縺・≧縺薙→縺ｯ縲・
			莉雁屓縺ｮ謗｢邏｢繧医ｊ縺溘￥縺輔ｓ謗｢邏｢縺励◆邨先棡縺ｮ縺ｯ縺壹↑縺ｮ縺ｧ縲∽ｻ雁屓繧医ｊ縺ｯ譫晏・繧翫′逕倥＞縺ｯ縺壹□縺九ｉ縲√◎縺ｮ蛟､繧剃ｿ｡鬆ｼ縺励※
			縺薙・縺ｾ縺ｾ縺薙・蛟､縺ｧreturn縺励※濶ｯ縺・・
	*/
    {
        // If ttMove is quiet, update move sorting heuristics on TT hit
        // ttMove縺繋uiet縺ｮ謖・＠謇九〒縺ゅｋ蝣ｴ蜷医∫ｽｮ謠幄｡ｨ繝偵ャ繝域凾縺ｫ謖・＠謇九・繧ｽ繝ｼ繝育畑繝偵Η繝ｼ繝ｪ繧ｹ繝・ぅ繧ｯ繧ｹ繧呈峩譁ｰ縺励∪縺吶・

        if (ttData.move && ttData.value >= beta)
        {
			/*
				統 蛯呵・
		
					鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇九〒beta cut縺瑚ｵｷ縺阪◆縺ｮ縺ｧ縺ゅｌ縺ｰ縲√％縺ｮ謖・＠謇九ｒkiller遲峨↓逋ｻ骭ｲ縺吶ｋ縲・
					謐慕佐縺吶ｋ謖・＠謇九°謌舌ｋ謖・＠謇九〒縺ゅｌ縺ｰ縺薙ｌ縺ｯ(capture縺ｧ逕滓・縺吶ｋ謖・＠謇九↑縺ｮ縺ｧ)killer繧呈峩譁ｰ縺吶ｋ萓｡蛟､縺ｯ縺ｪ縺・・
		
					縺溘□縺礼ｽｮ謠幄｡ｨ縺ｮ謖・＠謇九↓縺ｯ縲”ash陦晉ｪ√↓繧医ｊpseudo-leagl縺ｧ縺ｪ縺・欠縺玲焔縺ｧ縺ゅｋ蜿ｯ閭ｽ諤ｧ縺後≠繧九・
					update_quiet_stats()縺ｧ縲√％縺ｮ謖・＠謇九・遘ｻ蜍募・縺ｮ鬧偵ｒ蜿門ｾ励＠縺ｦCounter Move縺ｨ縺吶ｋ縺後・
					縺昴ｌ縺後％縺ｮ螻髱｢縺ｮ謇狗分蛛ｴ縺ｮ鬧偵〒縺ｯ縺ｪ縺・％縺ｨ縺後≠繧九・縺ｧ繧・∴縺ｫ縺薙％縺ｧpseudo_legal縺ｮ繝√ぉ繝・け繧偵＠縺ｦ縲・
					Counter Move縺ｫ蜈域焔縺ｮ謖・＠謇九→縺励※蠕梧焔縺ｮ謖・＠謇九′逋ｻ骭ｲ縺輔ｌ繧九ｈ縺・↑莠区・繧貞屓驕ｿ縺励※縺・ｋ縲・
					縺昴・譎ゅ↓陦後ｏ繧後ｋ隱､縺｣縺湮ｲcut(譫晏・繧・縺ｯ險ｱ螳ｹ縺ｧ縺阪ｋ縲・non PV縺ｧ逕溘§繧九％縺ｨ縺ｪ縺ｮ縺ｧ縺昴％縺ｾ縺ｧ謗｢邏｢縺ｫ蟇ｾ縺励※謔ｪ縺・ｽｱ髻ｿ縺後↑縺・
					cf. https://yaneuraou.yaneu.com/2021/08/17/about-the-yaneuraou-bug-that-appeared-in-the-long-match/

					繧・・縺・ｉ邇九〒縺ｯttMove縺勲ove::win()縺ｧ縺ゅｋ縺薙→縺ｯ縺ゅｊ縺・ｋ縺ｮ縺ｧ豕ｨ諢上′蠢・ｦ√・
					is_ok(m)==false縺ｮ譎ゅ￣osition::to_move(m)縺稽繧偵◎縺ｮ縺ｾ縺ｾ蟶ｰ縺吶％縺ｨ縺ｯ菫晁ｨｼ縺輔ｌ縺ｦ縺・ｋ縲・
					縺昴・縺溘ａttMove縺勲ove::win()縺ｧ縺ゅｊ縺・ｋ縲ゅ％繧後・stat縺ｮupdate繧偵＆繧後ｋ縺ｨ蝗ｰ繧九・縺ｧis_ok()縺ｧ蠑ｾ縺丞ｿ・ｦ√′縺ゅｋ縲・
					is_ok()縺ｯ縲》tMove == Move::none()縺ｮ譎ゅ・false縺ｪ縺ｮ縺ｧ縺薙・譚｡莉ｶ繧堤怐逡･縺ｧ縺阪ｋ縲・
					竍ｨ縲Move::win()譖ｸ縺榊・縺吶・縲∫ｭ九′濶ｯ縺上↑縺・＠縲∝・邇芽・菴薙′雜・Ξ繧｢繧ｱ繝ｼ繧ｹ縺ｪ縺ｮ縺ｧ譽句鴨縺ｫ蠖ｱ髻ｿ縺励↑縺・＠縲√％繧後ｄ繧√ｋ縺薙→縺ｫ縺吶ｋ縲・

					If ttMove is quiet, update move sorting heuristics on TT hit (~2 Elo)
					鄂ｮ謠幄｡ｨ縺ｫhit縺励◆譎ゅ↓縲》tMove縺繋uiet縺ｮ謖・＠謇九〒縺ゅｋ縺ｪ繧峨∵欠縺玲焔荳ｦ縺ｳ譖ｿ縺・euristics(quiet_stats縺ｮ縺薙→)繧呈峩譁ｰ縺吶ｋ縲・
			*/

            // Bonus for a quiet ttMove that fails high
            // fail high縺励◆quiet縺ｪquiet縺ｪ(鬧偵ｒ蜿悶ｉ縺ｪ縺・ttMove(鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇・縺ｫ蟇ｾ縺吶ｋ繝懊・繝翫せ

            if (!ttCapture)
                update_quiet_histories(pos, ss, *this, ttData.move,
                                       std::min(SPSA::TT_QUIET_BONUS_DEPTH_MULT * depth
                                                - SPSA::TT_QUIET_BONUS_OFFSET,
                                                SPSA::TT_QUIET_BONUS_MAX));

            // Extra penalty for early quiet moves of the previous ply
            // 1謇句燕縺ｮ譌ｩ縺・凾轤ｹ縺ｮquiet縺ｮ謖・＠謇九↓蟇ｾ縺吶ｋ霑ｽ蜉縺ｮ繝壹リ繝ｫ繝・ぅ

            // 庁 1謇句燕縺勲ove::null()縺ｧ縺ゅｋ縺薙→繧定・・縺吶ｋ蠢・ｦ√′縺ゅｋ縲・

            if (prevSq != SQ_NONE && (ss - 1)->moveCount <= 4 && !priorCapture)
                update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq,
                                              SPSA::TT_PREV_CONTHIST_MALUS);
        }

        // Partial workaround for the graph history interaction problem
        // For high rule50 counts don't produce transposition table cutoffs.

		// 繧ｰ繝ｩ繝募ｱ･豁ｴ縺ｮ逶ｸ莠剃ｽ懃畑蝠城｡後↓蟇ｾ縺吶ｋ驛ｨ蛻・噪縺ｪ蝗樣∩遲・
        // rule50繧ｫ繧ｦ繝ｳ繝医′鬮倥＞蝣ｴ蜷医・縲√ヨ繝ｩ繝ｳ繧ｹ繝昴ず繧ｷ繝ｧ繝ｳ繝・・繝悶Ν縺ｫ繧医ｋ繧ｫ繝・ヨ繧ｪ繝輔ｒ陦後＞縺ｾ縺帙ｓ縲・

        // 笞 蟆・｣九〒縺ｯ髢｢菫ゅ・縺ｪ縺・Ν繝ｼ繝ｫ縺ｪ縺ｮ縺ｧ辟｡隕悶＠縺ｦ濶ｯ縺・′縲｝os.rule50_count < 96 縺・繝√ぉ繧ｹ縺ｮ)騾壼ｸｸ縺ｮ迥ｶ諷九↑縺ｮ縺ｧ縲・
        //     if謌千ｫ区凾縺ｮreturn縺ｯ縺励↑縺代ｌ縺ｰ縺ｪ繧峨↑縺・・

        // ､・豈碑ｼ・＠縺ｦ縺ｿ縺溘′縲√％繧後ｒ譛牙柑縺ｫ縺吶ｋ縺ｨR10縺舌ｉ縺・ｼｱ縺上↑繧九▲縺ｽ縺・・
        //     蜈ｨ菴薙ｒ隱ｿ謨ｴ縺励※縺九ｉ縺ｾ縺溯・∴繧九・
#if 0
		if (pos.rule50_count() < 96)
        {
            // TODO : 蟆・｣九〒繧ゅ％縺ｮ蜃ｦ逅・ｿ・ｦ√↑縺ｮ縺具ｼ・

            if (depth >= 8 && ttData.move
#if STOCKFISH
				&& pos.pseudo_legal(ttData.move)
#else
                && pos.pseudo_legal(ttData.move, search_options.generate_all_legal_moves)
#endif
				&& pos.legal(ttData.move)
                && !is_decisive(ttData.value))
            {
#if STOCKFISH
                pos.do_move(ttData.move, st);
                Key nextPosKey                             = pos.key();
#else
                pos.do_move(ttData.move, st);
                HASH_KEY nextPosKey                        = pos.hash_key();
#endif
				auto [ttHitNext, ttDataNext, ttWriterNext] = tt.probe(nextPosKey, pos);
                pos.undo_move(ttData.move);

                // Check that the ttValue after the tt move would also trigger a cutoff
                if (!is_valid(ttDataNext.value))
                    return ttData.value;
                if ((ttData.value >= beta) == (-ttDataNext.value >= beta))
                    return ttData.value;
            }
            else
                return ttData.value;
        }
#else
        return ttData.value;
#endif
    }
    // ★SF port #2 penalize (SF search.cpp:913-921, commit 319d61effd) "No cutoff, but why?":
    //   cutoff できなかった理由が「十分な depth はあるが bound が窓と不整合 (EXACT でない)」だけなら、
    //   そのエントリは今後も役に立たないので格納 depth を 1 下げる (次回の置換を促す)。
    //   is_valid(ttData.value) は ttHit を含意する (!ttHit なら VALUE_NONE を入れている) ので、
    //   penalize が別局面用の置換候補エントリに触れることはない。
    //   既定 OFF (SPSA::SF_TT_PENALIZE=false) なら非 PV ノードで分岐 1 つのみ。
    else if (!PvNode && SPSA::SF_TT_PENALIZE && !excludedMove
             && ttData.depth > depth - (ttData.value <= beta) && is_valid(ttData.value)
             && ttData.bound != BOUND_EXACT
             && (ttData.bound & (ttData.value >= beta ? BOUND_UPPER : BOUND_LOWER)) && depth > 5)
    {
        // If such a mismatch is the only reason cutoff failed, the TT entry is now useless
        ttWriter.penalize(1);
    }

	// -----------------------
    //     螳｣險蜍昴■
    // -----------------------

    // Step 5. Tablebases probe
    // 笞 Stockfish縺ｮStep 5.縺ｮ繧ｳ繝ｼ繝峨・tablebase(邨ら乢繝・・繧ｿ繝吶・繧ｹ)縺ｧ蟆・｣九↓縺ｯ髢｢菫ゅ↑縺・・縺ｧ蜑ｲ諢帙・

#if STOCKFISH
    if (!rootNode && !excludedMove && tbConfig.cardinality)
    {
        int piecesCount = pos.count<ALL_PIECES>();

        if (piecesCount <= tbConfig.cardinality
            && (piecesCount < tbConfig.cardinality || depth >= tbConfig.probeDepth)
            && pos.rule50_count() == 0 && !pos.can_castle(ANY_CASTLING))
        {
            TB::ProbeState err;
            TB::WDLScore   wdl = Tablebases::probe_wdl(pos, &err);

            // Force check of time on the next occasion
            if (is_mainthread())
                main_manager()->callsCnt = 0;

            if (err != TB::ProbeState::FAIL)
            {
                // Preferable over fetch_add to avoid locking instructions
                tbHits.store(tbHits.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);

                int drawScore = tbConfig.useRule50 ? 1 : 0;

                Value tbValue = VALUE_TB - ss->ply;

                // Use the range VALUE_TB to VALUE_TB_WIN_IN_MAX_PLY to score
                value = wdl < -drawScore ? -tbValue
                      : wdl > drawScore  ? tbValue
                                         : VALUE_DRAW + 2 * wdl * drawScore;

                Bound b = wdl < -drawScore ? BOUND_UPPER
                        : wdl > drawScore  ? BOUND_LOWER
                                           : BOUND_EXACT;

                if (b == BOUND_EXACT || (b == BOUND_LOWER ? value >= beta : value <= alpha))
                {
                    ttWriter.write(posKey, value_to_tt(value, ss->ply), ss->ttPv, b,
                                   std::min(MAX_PLY - 1, depth + 6), Move::none(), VALUE_NONE,
                                   tt.generation());

                    return value;
                }

                if (PvNode)
                {
                    if (b == BOUND_LOWER)
                        bestValue = value, alpha = std::max(alpha, bestValue);
                    else
                        maxValue = value;
                }
            }
        }
    }
#else

	// 縺薙ｌ縺ｯ蟆・｣九↓縺ｯ縺ｪ縺・′縲∝ｰ・｣九↓縺ｯ莉｣繧上ｊ縺ｫ螳｣險蜍昴■縺ｨ縺・≧縺ｮ縺後≠繧九・
    // 螳｣險蜍昴■縺ｨ1謇玖ｩｰ繧√□縺ｨ1謇玖ｩｰ繧√・譁ｹ縺悟悸蛟堤噪縺ｫ螟壹＞縺ｮ縺ｧ縲√∪縺・謇玖ｩｰ繧∝愛螳壹ｒ陦後≧縲・

    // 決 莉･荳九・縲√ｄ縺ｭ縺・ｉ邇狗峡閾ｪ縺ｮ繧ｳ繝ｼ繝・決

    // -----------------------
    //    1謇玖ｩｰ縺ｿ縺具ｼ・
    // -----------------------

	/*
		統 ttHit縺励※縺・ｋ譎ゅ・縲√☆縺ｧ縺ｫ1謇玖ｩｰ繧√・螳｣險蜍昴■縺ｮ蛻､螳壹・螳御ｺ・＠縺ｦ縺・ｋ縺ｯ縺壹↑縺ｮ縺ｧ陦後ｏ縺ｪ縺・・
			excludedMove縺後≠繧区凾縺ｯttHit縺励※ttMove縺後≠繧九・縺壹□縺後∫ｽｮ謠幄｡ｨ螢翫＆繧後ｋ繧ｱ繝ｼ繧ｹ縺後≠繧九・縺ｧ
			蠢ｵ縺ｮ縺溘ａ繝√ぉ繝・け縺ｯ縺励↑縺・→縺・￠縺ｪ縺・・

			!rootnode縺ｧ縺ｯ縺ｪ縺・PvNode縺ｮ譁ｹ縺後＞縺・°繧ゑｼ・
			(PvNode縺ｧ縺ｯ鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇九ｒ菫｡逕ｨ縺励※縺ｯ縺・￠縺ｪ縺・→縺・≧譁ｹ驥昴↑繧・

			excludedMove縺後≠繧区凾縺ｫ縺ｯ譛ｬ蠖薙・縲√◎繧後ｒ髯､螟悶＠縺ｦ隧ｰ縺ｿ謗｢邏｢繧偵☆繧句ｿ・ｦ√′縺ゅｋ縺後・
			隧ｰ縺ｿ縺後≠繧句ｴ蜷医・縲《ingular extension縺ｮ蛻､螳壹・蜑阪∪縺ｧ縺ｫbeta cut縺吶ｋ縺ｮ縺ｧ縲∫ｵ仙ｱ縲・
			隧ｰ縺ｿ縺後≠繧九・縺ｫexcludedMove縺瑚ｨｭ螳壹＆繧後※縺・ｋ縺ｨ縺・≧縺薙→縺ｯ縺ゅｊ縺医↑縺・・
			繧医▲縺ｦ縲√憩xcludedMove縺ｯ險ｭ螳壹＆繧後※縺・↑縺・肴凾縺縺題ｩｰ縺ｿ縺後≠繧九°縺ｩ縺・°繧定ｪｿ縺ｹ繧後・濶ｯ縺上・
			縺薙・譚｡莉ｶ繧定ｩｰ縺ｿ謗｢邏｢縺ｮ莠句燕譚｡莉ｶ縺ｫ霑ｽ蜉縺吶ｋ縺薙→縺後〒縺阪ｋ縲・
		
			縺溘□縺励‘xcludedMove縺後≠繧区凾縲《ingular extension縺ｮ莠句燕譚｡莉ｶ繧呈ｺ縺溘☆縺ｯ縺壹〒縲・
			singular extension縺ｯttMove縺悟ｭ伜惠縺吶ｋ縺薙→縺後◎縺ｮ譚｡莉ｶ縺ｫ蜷ｫ縺ｾ繧後ｋ縺九ｉ縲・
			ss->ttHit == true縺ｫ縺ｪ縺｣縺ｦ縺・ｋ縺ｯ縺壹・
		
			RootNode縺ｧ縺ｯ1謇玖ｩｰ繧∝愛螳壹√ｄ繧・％縺励￥縺ｪ繧九・縺ｧ繧・ｉ縺ｪ縺・・RootMoves縺ｮ蜈･繧梧崛縺育ｭ峨′逋ｺ逕溘☆繧九・縺ｧ)
			鄂ｮ謠幄｡ｨ縺ｫhit縺励◆縺ｨ縺阪ｂ1謇玖ｩｰ繧∝愛螳壹・縺吶〒縺ｫ陦後ｏ繧後※縺・ｋ縺ｨ諤昴ｏ繧後ｋ縺ｮ縺ｧ縺薙・蝣ｴ蜷医ｂ縺ｯ縺励ｇ繧九・
			depth縺ｮ谿九ｊ縺後≠繧狗ｨ句ｺｦ縺ｪ縺・→縲・謇玖ｩｰ繧√・縺ｩ縺・○縺薙・縺ゅ→縺吶＄縺ｫ隕九▽縺代※縺励∪縺・ｏ縺代〒1謇玖ｩｰ繧√ｒ
			隕九▽縺代◆縺ｨ縺阪・繝ｪ繧ｿ繝ｼ繝ｳ(隕玖ｿ斐ｊ)縺悟ｰ代↑縺・・
			縺溘□縲・撕豁｢謗｢邏｢縺ｧ蜈･繧後※縺・ｋ莉･荳翫‥epth == 1縺ｧ繧・謇玖ｩｰ繧√ｒ蛻､螳壹＠縺溘⊇縺・′繧医＆縺偵〒縺ｯ縺ゅｋ縲・
	*/

#if defined(EVAL_P2NNUE)
	// 決 rust貅匁侠: rust alpha_beta.rs 縺ｮ mate_1ply probe 縺ｨ蜷御ｸ繧ｲ繝ｼ繝医↓謠・∴繧九・
	//    rust譚｡莉ｶ: pv_node && depth>=6 && game_ply>=30 && !tt_hit && excluded辟｡ && !in_check縲・
	//    (v940譌｢螳壹・蜈ｨ繝弱・繝峨〒逋ｺ蜍輔☆繧九′縲〉ust縺ｯPV繝ｻ荳ｭ逶､繝ｻdepth>=6縺ｮ縺ｿ縺ｫ邨槭▲縺ｦ縺・ｋ)
	if (!rootNode && PvNode && depth >= 6 && pos.game_ply() >= 30 && !ttHit && !excludedMove)
#else
	if (!rootNode && !ttHit
		// TODO : 縺薙・譚｡莉ｶ蠢・ｦ√↑縺ｮ縺具ｼ・
        && !excludedMove
    )
#endif
    {
        if (!ss->inCheck)
        {
            move = Mate::mate_1ply(pos);
            int p2_mate_dist = 1;
            // 深い詰みは非同期の df-pn 詰みsolver (USI: UseMateSolverAsync) が担当する。
            //     蜷・ヮ繝ｼ繝峨〒縺ｯ讓呎ｺ悶・ 1謇玖ｩｰ繧√・縺ｿ(邁｡譏・mate_odd_ply probe 縺ｯ謦､蜴ｻ)縲・

            if (move != Move::none())
            {
                /*
					､・1謇玖ｩｰ繧√せ繧ｳ繧｢縺ｪ縺ｮ縺ｧ遒ｺ螳溘↓value > alpha縺ｪ縺ｯ縺壹・
					    1謇玖ｩｰ繧√・谺｡縺ｮnode縺ｧ隧ｰ繧縺ｨ縺・≧隗｣驥・

					笞 縺薙・縺ｨ縺阪｜estValue繧致alue_to_tt()縺ｧ螟画鋤縺励※縺ｯ縺ｪ繧峨↑縺・・
					    mate_in()縺ｯroot縺九ｉ縺ｮ險育ｮ励＆繧後◆隧ｰ縺ｿ縺ｮ繧ｹ繧ｳ繧｢縺ｧ縺ゅｋ縺九ｉ縲√％縺ｮ縺ｾ縺ｾ鄂ｮ謠幄｡ｨ縺ｫ譬ｼ邏阪＠縺ｦ繧医＞縲・
                
                		縺ゅｋ縺・・縲〃ALUE_MATE - 1繧致alue_to_tt()縺ｧ螟画鋤縺励◆繧ゅ・繧堤ｽｮ謠幄｡ｨ縺ｫ譬ｼ邏阪☆繧九°縺縺後・
					    縺昴・蝣ｴ蜷医〉eturn縺ｧ霑斐☆蛟､繧堤畑諢上☆繧九・縺ｫ蜀榊ｺｦ螟画鋤縺悟ｿ・ｦ√↓縺ｪ繧九・縺ｧ縺昴≧縺・≧譖ｸ縺肴婿縺ｯ濶ｯ縺上↑縺・・
				*/

                bestValue = mate_in(ss->ply + p2_mate_dist);

                ASSERT_LV3(pos.legal_promote(move));
                if (!excludedMove)
                    ttWriter.write(posKey, bestValue, ss->ttPv, BOUND_EXACT,
                                    std::min(MAX_PLY - 1, depth + 6), move, VALUE_NONE,
                                    tt.generation());

				// 笞 excludedMove縺後≠繧九→縺阪・鄂ｮ謠幄｡ｨ縺ｫ譖ｸ縺榊・縺輔↑縺・Ν繝ｼ繝ｫ縺ｫ縺ｪ縺｣縺ｦ縺・ｋ縺ｮ縺ｧ縲・
                //     縺薙・竊代・譚｡莉ｶ蠑上′蠢・ｦ√↑縺ｮ縺ｧ豕ｨ諢上・

				/*
				   統 
                縲	 縲占ｨ域ｸｬ雉・侭 39.縲・mate1ply縺ｮ謖・＠謇九ｒ隕九▽縺代◆譎ゅ↓鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇九〒beta
					  cut縺吶ｋ譎ゅ→蜷後§蜃ｦ逅・ｒ縺吶ｋ縲・

                      蜈・ｼ溷ｱ髱｢縺ｧ縺薙・mate縺ｮ謖・＠謇九′繧医＞謖・＠謇九≠繧句庄閭ｽ諤ｧ縺後≠繧九・縺ｧ 縺薙％縺ｧttMove縺ｧbeta
                      cut縺吶ｋ譎ゅ→蜷梧ｧ倥・蜃ｦ逅・ｒ陦後≧縺ｨ遏ｭ縺・凾髢薙〒縺ｯ繧上★縺九↓蠑ｷ縺上↑繧九▲縺ｽ縺・・縺縺・
                      髟ｷ縺・凾髢薙〒險域ｸｬ縺ｧ縺阪ｋ蟾ｮ縺ｧ縺ｯ縺ｪ縺九▲縺溘・縺ｧ蜑企勁縲・

					庁
						1謇玖ｩｰ繧√ｒ逋ｺ隕九＠縺滓凾縺ｫ縲《ave()縺ｧdepth繧偵←縺ｮ繧医≧縺ｫ險ｭ螳壹☆縺ｹ縺阪°蝠城｡後↓縺､縺・※縲・

						蜊ｳ隧ｰ縺ｿ縺ｯ邨ｶ蟇ｾ縺ｧ縺ゅｊ縲｀AX_PLY縺ｮ豺ｱ縺輔〒謗｢邏｢縺励◆譎ゅ・邨先棡縺ｨ蜷後§縺ｧ縺ゅｋ縺九ｉ縲・
						莉･蜑阪・MAX_PLY縺ｫ縺励※縺・◆縺ｮ縺縺後√ｈ縺剰・∴縺溘ｉ縲∝叉隧ｰ縺ｿ縺後≠繧九↑繧我ｸ贋ｽ阪ヮ繝ｼ繝峨〒
						譫晏・繧翫′逋ｺ逕溘＠縺ｦ縺薙・繝弱・繝峨↓縺ｯ縺ｻ縺ｼ蜀崎ｨｪ蝠上＠縺ｪ縺・→閠・∴繧峨ｌ繧九・縺ｧ縺薙ｓ縺ｪ繧ゅ・縺・
						鄂ｮ謠幄｡ｨ縺ｫ谿九▲縺ｦ縺・ｋ萓｡蛟､縺ｫ荵上＠縺上√∪縺溘｀AX_PLY縺ｫ縺励※縺励∪縺・→縲・
						TTEntry縺ｮreplacement strategy荳翫‥epth縺悟､ｧ縺阪↑TTEntry縺ｯ縺九↑繧翫・蜆ｪ蜈亥ｺｦ縺ｫ縺ｪ繧・
						縺・▽縺ｾ縺ｧ繧Ｓeplacement縺輔ｌ縺ｪ縺・・

						縺薙ｓ縺ｪ諠・ｱ縲〕ost縺励◆縺ｨ縺薙ｍ縺ｧ1謇玖ｩｰ繧√↑繧盈ate1ply()縺ｧ荳謇九ｂ騾ｲ繧√★縺ｫ蠕励ｉ繧後ｋ諠・ｱ縺ｧ縺ゅｊ縲・
						譛蜆ｪ蜈医↓replace縺吶∋縺控TEntry縺ｫ繧る未繧上ｉ縺壹〒縺ゅｋ縲・

						縺九→險縺｣縺ｦDEPTH_NONE縺ｫ縺吶ｋ縺ｨtt->depth()縺・0 縺ｫ縺ｪ縺｣縺ｦ縺励∪縺・∵椌蛻医ｊ縺後＆繧後↑縺上↑繧九・
						縺昴％縺ｧ縲‥epth + 6 縺舌ｉ縺・′繝吶せ繝医〒縺ゅｋ繧医≧縺縲・
				*/

				return bestValue;
            }
        }
	}

	// -----------------------
    //      螳｣險蜍昴■縺具ｼ・
    // -----------------------

	// 鄂ｮ謠幄｡ｨ縺ｫhit縺励※縺・↑縺・→縺阪・螳｣險蜍昴■縺ｮ蛻､螳壹ｒ縺ｾ縺繧・▲縺ｦ縺・↑縺・→縺・≧縺薙→縺ｪ縺ｮ縺ｧ莉雁屓繧・ｋ縲・
    // PvNode縺ｧ縺ｯ鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇九ｒ菫｡逕ｨ縺励※縺ｯ縺・￠縺ｪ縺・・縺ｧ豈主屓繧・ｋ縲・
    if (!ttData.move || PvNode)
    {
        // 庁 邇区焔縺後°縺九▲縺ｦ繧医≧縺後°縺九▲縺ｦ縺ｾ縺・′縲∝ｮ｣險蜍昴■縺ｮ蛻､螳壹・豁｣縺励＞縲・
        //     (繝医Λ繧､繝ｫ繝ｼ繝ｫ縺ｮ縺ｨ縺咲視謇九ｒ蝗樣∩縺励↑縺後ｉ蜈･邇峨☆繧九％縺ｨ縺ｯ縺ゅｊ縺・ｋ縺ｮ縺ｧ)
        //     繝医Λ繧､繝ｫ繝ｼ繝ｫ縺ｮ縺ｨ縺阪％縺薙〒霑斐▲縺ｦ縺上ｋ縺ｮ縺ｯ16bit縺ｮmove縺縺後∫ｽｮ謠幄｡ｨ縺ｫ譬ｼ邏阪☆繧九↓縺ｯ蝠城｡後↑縺・・
        move = pos.DeclarationWin();
        if (move != Move::none())
        {
            bestValue = mate_in(
                ss->ply + 1);  // 1謇玖ｩｰ繧√↑縺ｮ縺ｧ縺薙・谺｡縺ｮnode縺ｧ(謖・＠謇九′縺ｪ縺上↑縺｣縺ｦ)隧ｰ繧縺ｨ縺・≧隗｣驥・

            ASSERT_LV3(pos.legal_promote(move));
            /*
				if (!excludedMove)
					ttWriter.write(posKey, bestValue, ss->ttPv, BOUND_EXACT,
						std::min(MAX_PLY - 1, depth + 6),
						is_ok(move) ? move : MOVE_NONE, // MOVE_WIN縺ｯMOVE_NONE縺ｨ縺励※譖ｸ縺榊・縺吶・
						VALUE_NONE, TT.generation());
			*/

			/*
				統  is_ok(m)縺ｯ縲｀OVE_WIN縺ｧ縺ｯ縺ｪ縺・壼ｸｸ縺ｮ謖・＠謇・繝医Λ繧､繝ｫ繝ｼ繝ｫ縺ｮ譎ゅ・51邇峨・繧医≧縺ｪ謖・＠謇・縺ｯ
					 縺昴・謖・焔繧堤ｽｮ謠幄｡ｨ縺ｫ譖ｸ縺榊・縺吶→縺・≧蜃ｦ逅・・
            
					 probe()縺ｧttData.move == MOVE_WIN縺ｮ繧ｱ繝ｼ繧ｹ繧貞ｮ悟・縺ｫ閠・・縺吶ｋ縺ｮ縺ｯ髱槫ｸｸ縺ｫ髮｣縺励＞縲・
             縲		 MOVE_WIN縺ｮ蛟､縺ｯ縲∫ｽｮ謠幄｡ｨ縺ｫ譖ｸ縺榊・縺輔↑縺・⊇縺・′縺・＞縺ｨ諤昴≧縲・
            
					 縺ｾ縺溘∫ｽｮ謠幄｡ｨ縺ｫ繧上＊縺ｨ譖ｸ縺崎ｾｼ縺ｾ縺ｪ縺・％縺ｨ縺ｫ繧医▲縺ｦ縲∵ｬ｡縺ｫ縺薙・node縺ｫ險ｪ蝠上＠縺溘→縺阪↓
					 蜀榊ｺｦ蜈･邇牙愛螳壹ｒ陦後＞縲∵椌蛻医ｊ縺輔ｌ繧九％縺ｨ繧呈悄蠕・＠縺ｦ縺・ｋ縲・
			*/

            return bestValue;
        }
        // ､・1謇玖ｩｰ繧√→螳｣險蜍昴■縺後↑縺九▲縺溘・縺ｧ縺薙・譎らせ縺ｧ繧Ｔave()縺励◆縺ｻ縺・′縺・＞繧医≧縺ｪ豌励′縺励↑縺上ｂ縺ｪ縺・・
    }

#endif // STOCKFISH

// 2026-09-03: 探索内で同期的に df-pn を呼ぶ詰みsolver を削除した。
//   - 実測で発火ゼロ (16局面、うち15局面がゲート条件 game_ply>=30 を満たすのに
//     on/off で depth も EBF も完全一致)。ゲート !ttHit && ss->ply<=2 && depth>=12 が狭すぎた。
//   - 発火した場合は ttWriter.write(..., depth+6, ...) で TT を汚し探索木を歪める。
//     rust 側 (crates/p2-search/src/mate_solver_thread.rs) が「探索内 integration は
//     alpha_beta の読み筋を歪め逆効果」として撤退した形と同じ。
//   - 代替は非同期版 UseMateSolverAsync (main search 非干渉、bestmove 上書きのみ)。

	// -----------------------
    // Step 6. Static evaluation of the position
    // Step 6. 螻髱｢縺ｮ髱咏噪縺ｪ隧穂ｾ｡
    // -----------------------

    Value      unadjustedStaticEval = VALUE_NONE;
    const auto correctionValue      = correction_value(*this, pos, ss);

#if defined(P2_KHGATE)
    // KHGATE: 1 node 1 回だけ数えて PESS (staticEval) / SELEXT (Step 8,9 / LMR) で共有する。
    //   PESS は王手中は使わない (王手中は staticEval を作らないので既存の補正と同じ扱い)。 全 option 0 なら何も走らない。
    KhGate khGate;
    {
        const bool wantPess  = kh_pess_on() && !ss->inCheck && !excludedMove;
        const bool wantRough = kh_selext_on();
        if (wantPess | wantRough)
            khGate = kh_gate_eval(pos, wantPess, wantRough);
    }
#endif

    // Uncertainty-linked LMR (隨ｬ1繝代ャ繝・: 隧穂ｾ｡諤･螟牙ｺｦ縲Ｎoves loop / SE 蜑阪↓1蠎ｦ縺縺醍ｮ怜・縺励Ο繝ｼ繧ｫ繝ｫ騾驕ｿ縺吶ｋ縲・
    // goto moves_loop(inCheck) 繧定ｷｨ縺舌◆繧√∝・譛溷喧莉倥″螳｣險縺ｯ縺薙％(繝ｩ繝吶Ν/goto 繧医ｊ蜑・縺ｫ鄂ｮ縺上・
    int  nodeEvalVolatility  = 0;
    bool validEvalVolatility = false;

    // Threat Continuation History (Phase B): Null Move 縺ｧ謗｡蜿悶＠縺・threatMove 縺ｮ譁・ц繝上ャ繧ｷ繝･縲・
    // goto moves_loop 繧定ｷｨ縺舌◆繧∝・譛溷喧莉倥″螳｣險繧偵％縺薙↓鄂ｮ縺・inCheck 繝弱・繝峨・ threatValid=false 縺ｮ縺ｾ縺ｾ)縲・
    uint32_t nodeThreatCtx = 0;
    bool     threatValid   = false;

	if (ss->inCheck)
    {
        // Skip early pruning when in check
        // 邇区焔縺後°縺九▲縺ｦ縺・ｋ縺ｨ縺阪・縲‘arly pruning(譌ｩ譛滓椌蛻医ｊ)繧偵せ繧ｭ繝・・縺吶ｋ

		ss->staticEval = eval = (ss - 2)->staticEval;
        improving             = false;
        goto moves_loop;
    }
    else if (excludedMove)
        unadjustedStaticEval = eval = ss->staticEval;
    /*
		統  excludedMove縺後≠繧九→縺阪・縲√％縺ｮ螻髱｢縺ｮ諠・ｱ繧探T縺ｫ菫晏ｭ倥＠縺ｦ縺ｯ縺ｪ繧峨↑縺・・
			 (蜷御ｸ螻髱｢縺ｧ逡ｰ縺ｪ繧菊xcludedMove繧呈戟縺､螻髱｢縺悟酔縺鷲ashkey繧呈戟縺､縺ｮ縺ｧ諠・ｱ縺ｮ荳雋ｫ諤ｧ縺後↑縺上↑繧九・
			 竍ｨ 逡ｰ縺ｪ繧菊xcludedMove縺ｫ蟇ｾ縺励※逡ｰ縺ｪ繧吃ashkey繧呈戟縺ｦ縺ｰ縺・＞縺ｮ縺縺・
			   萓・ auto posKey = pos.hash_key() ^ make_key(excludedMove);
			 (莉･蜑阪・Stockfish縺ｮ繝舌・繧ｸ繝ｧ繝ｳ縺ｧ縺ｯ縺昴・繧医≧縺ｫ縺励※縺・◆)
			  迴ｾ蝨ｨ縺ｮ繧ｳ繝ｼ繝峨・縺ｻ縺・′蠑ｷ縺・ｈ縺・〒縲√◎縺ｮ繧ｳ繝ｼ繝峨・蜿悶ｊ髯､縺九ｌ縺溘・
	*/
    else if (ss->ttHit)
    {
        // Never assume anything about values stored in TT
        // TT縺ｫ譬ｼ邏阪＆繧後※縺・ｋ蛟､縺ｫ髢｢縺励※菴輔ｂ莉ｮ螳壹・縺励↑縺・

        // 庁 鄂ｮ謠幄｡ｨ縺ｫhit縺励◆縺ｪ繧峨∬ｩ穂ｾ｡蛟､縺瑚ｨ倬鹸縺輔ｌ縺ｦ縺・ｋ縺ｯ縺壹□縺九ｉ縲√◎繧後ｒ蜿悶ｊ蜃ｺ縺励※縺翫￥縲・
        //     縺ゅ→縺ｧ鄂ｮ謠幄｡ｨ縺ｫ譖ｸ縺崎ｾｼ繧縺ｨ縺阪↓縺薙・蛟､繧剃ｽｿ縺医ｋ縺励∝推遞ｮ譫晏・繧翫・縺薙・隧穂ｾ｡蛟､繧偵・繝ｼ繧ｹ縺ｫ陦後↑縺・°繧峨・

		unadjustedStaticEval = ttData.eval;
        if (!is_valid(unadjustedStaticEval))
            unadjustedStaticEval = evaluate(pos);
#if defined(EVAL_P2NNUE)
        // 笘・ P2NNUE: 譌｢螳壹・ PvNode 縺ｧ繧・ttData.eval 繧剃ｿ｡逕ｨ(YANEURAOU_ENGINE_NNUE 譛ｪ螳夂ｾｩ繧・∴
        //   荳九・ R70 蛻・ｲ舌′譛ｪ繧ｳ繝ｳ繝代う繝ｫ)縲１2PvReeval=true 縺ｧ YO/SF 貅匁侠縺ｫ PvNode 繧貞・隧穂ｾ｡縲・
        else if (PvNode && Eval::p2_pv_reeval)
            unadjustedStaticEval = evaluate(pos);
#endif

#if !STOCKFISH
#if defined(YANEURAOU_ENGINE_NNUE)
#if defined(USE_LAZY_EVALUATE)
        // 決 縺薙ｌ譖ｸ縺九↑縺・→R70縺舌ｉ縺・ｼｱ縺上↑繧九・
        else if (PvNode)
        {
			unadjustedStaticEval = evaluate(pos);
#ifdef P2_PRUNE_STATS   // 🚨P2PruneStat はこの中にある (本番ビルドでは消える)
			// 🎯計測用: この ply の bucket を覚える (跨ぎ判定の親側になる)。
			//   🚨evaluate した直後なので StateInfo に計算済み = 読んでも bit-exact を崩さない。
			P2PruneStat::ply_bucket[ss->ply] = Eval::P2NNUE::progress_bucket_if_computed(pos);
#endif

			/*
				 ､・: 縺薙％縺ｧevaluate() 縺悟ｿ・医↑逅・罰縺後ｈ縺上ｏ縺九ｉ縺ｪ縺・・
				       Stockfish縺ｫ縺ｯ辟｡縺・・縺ｫ窶ｦ縲ゅ↑縺懊↑縺ｮ縺銀ｦ縲・

				lazy_evaluate()縺ｫ螟峨∴繧・                       竍ｨ 蠑ｱ縺・V8.38dev-n7)
				unadjustedStaticEval縺ｫ莉｣蜈･縺帙★縺ｫ蜊倥↓evaluate() 竍ｨ 迚ｹ縺ｫ蠑ｱ縺上・縺ｪ縺輔◎縺・V838dev_n8)
				邨占ｫ也噪縺ｫ縺ｯ縲¨NUE縺ｮevaluate_with_no_return()縺ｨevaluate()縺ｮ謖吝虚縺碁＆縺・・縺悟次蝗縺｣縺ｽ縺・・
				縺薙ｌ縺ｯ縺ゅ″縺ｾ縺帙ｓ繧鞘ｦ縲・
			*/
        }
#else
        // lazy eval繧剃ｽｿ繧上↑縺・↑繧峨√％縺ｮ譎らせ縺ｧ縺ｩ縺・↓縺九＠縺ｦ縺翫￥縲・
        else
			unadjustedStaticEval = evaluate(pos);
#ifdef P2_PRUNE_STATS   // 🚨P2PruneStat はこの中にある (本番ビルドでは消える)
			// 🎯計測用: この ply の bucket を覚える (跨ぎ判定の親側になる)。
			//   🚨evaluate した直後なので StateInfo に計算済み = 読んでも bit-exact を崩さない。
			P2PruneStat::ply_bucket[ss->ply] = Eval::P2NNUE::progress_bucket_if_computed(pos);
#endif
#endif
#endif
#endif

        if (!PvNode && is_valid(ttData.eval) && unadjustedStaticEval == ttData.eval) P2SMP_CHECK_R(pos, ttData.eval);
        ss->staticEval = eval = to_corrected_static_eval(P2_OPT_ADD(P2_KHGATE_PESS(unadjustedStaticEval)), correctionValue);

        // ttValue can be used as a better position evaluation
        // ttValue 縺ｯ縲√ｈ繧願憶縺・ｱ髱｢隧穂ｾ｡縺ｨ縺励※菴ｿ逕ｨ縺ｧ縺阪ｋ

		/*
			統 ttValue縺ｮ縺ｻ縺・′縺薙・螻髱｢縺ｮ隧穂ｾ｡蛟､縺ｮ隕狗ｩ阪ｂ繧翫→縺励※驕ｩ蛻・〒縺ゅｋ縺ｪ繧峨◎繧後ｒ謗｡逕ｨ縺吶ｋ縲・

				1. ttValue > evaluate()縺ｧ縺九▽縲》tValue縺沓OUND_LOWER縺ｪ繧峨∫悄縺ｮ蛟､縺ｯ縺薙ｌ繧医ｊ螟ｧ縺阪＞縺ｯ縺壹□縺九ｉ縲・
				  eval縺ｨ縺励※ttValue繧呈治逕ｨ縺励※濶ｯ縺・・

				2. ttValue < evaluate()縺ｧ縺九▽縲》tValue縺沓OUND_UPPER縺ｪ繧峨∫悄縺ｮ蛟､縺ｯ縺薙ｌ繧医ｊ蟆上＆縺・・縺壹□縺九ｉ縲・
				  eval縺ｨ縺励※ttValue繧呈治逕ｨ縺励◆縺ｻ縺・′縺薙・螻髱｢縺ｫ蟇ｾ縺吶ｋ隧穂ｾ｡蛟､縺ｮ隕狗ｩ阪ｊ縺ｨ縺励※驕ｩ蛻・〒縺ゅｋ縲・
		*/

        if (is_valid(ttData.value)
            && (ttData.bound & (ttData.value > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttData.value;
    }
    else
    {
        unadjustedStaticEval = evaluate(pos);

        ss->staticEval = eval = to_corrected_static_eval(P2_OPT_ADD(P2_KHGATE_PESS(unadjustedStaticEval)), correctionValue);

        // Static evaluation is saved as it was before adjustment by correction history
        // 髱咏噪隧穂ｾ｡縺ｯ縲∬｣懈ｭ｣螻･豁ｴ縺ｫ繧医ｋ隱ｿ謨ｴ縺瑚｡後ｏ繧後ｋ蜑阪・迥ｶ諷九〒菫晏ｭ倥＆繧後ｋ縲・

		/*
		  統 excludedMove縺後≠繧区凾縺ｯ縲√％繧後ｒ鄂ｮ謠幄｡ｨ縺ｫ菫晏ｭ倥☆繧九・縺ｯ蜊ｱ縺ｪ縺・・
              cf . Add / remove leaves from search tree ttPv : https://github.com/official-stockfish/Stockfish/commit/c02b3a4c7a339d212d5c6f75b3b89c926d33a800
              荳翫・譁ｹ縺ｫ縺ゅｋ else if (excludedMove) 縺ｧ縺薙・譚｡莉ｶ縺ｯ髯､螟悶＆繧後※縺・ｋ縲・
		*/

        ttWriter.write(posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_UNSEARCHED, Move::none(),
                       unadjustedStaticEval, tt.generation());

		// 縺ｩ縺・○豈始ode隧穂ｾ｡髢｢謨ｰ繧貞他縺ｳ蜃ｺ縺吶・縺ｧ縲‘val縺ｮ蛟､縺ｫ縺昴ｓ縺ｪ縺ｫ萓｡蛟､縺ｯ縺ｪ縺・・縺縺後［ate_1ply()繧・
        // 螳溯｡後＠縺溘→縺・≧險ｼ縺ｫ縺ｯ縺ｪ繧九・縺ｧ諢丞袖縺後≠繧九・
    }

	// -----------------------
    //   eval繝吶・繧ｹ縺ｮ譫晏・繧・
    // -----------------------

    // Use static evaluation difference to improve quiet move ordering
    // 髱咏噪隧穂ｾ｡縺ｮ蟾ｮ繧貞茜逕ｨ縺励※縲・撕縺九↑謇九・鬆・ｺ丈ｻ倥￠繧呈隼蝟・＠縺ｾ縺吶・

	/*
		統 螻髱｢縺ｮ髱咏噪隧穂ｾ｡蛟､(eval)縺悟ｾ励ｉ繧後◆縺ｮ縺ｧ縲∽ｻ･荳九〒縺ｯ縺薙・隧穂ｾ｡蛟､繧堤畑縺・※蜷・ｨｮ譫晏・繧翫ｒ陦後↑縺・・
		    邇区焔縺ｮ縺ｨ縺阪・縺薙％縺ｫ縺ｯ縺薙↑縺・・荳翫・inCheck縺ｮ縺ｪ縺九〒MOVES_LOOP縺ｫ遯∝・縲・

			is_ok()縺ｯMove::null()縺九・繝√ぉ繝・け縲・
			1謇句燕縺ｧMove::null()縺ｧ縺ｯ縺ｪ縺上∫視謇九′縺九°縺｣縺ｦ縺翫ｉ縺壹・ｧ偵ｒ蜿悶ｋ謖・＠謇九〒縺ｯ縺ｪ縺九▲縺溘↑繧俄ｦ縲・
	*/


    if (((ss - 1)->currentMove).is_ok() && !(ss - 1)->inCheck && !priorCapture)
    {
        // ★Q8 固定小数化 (NAGISA V3.1 と同一の定式化。2026-09-03)
        //   旧: evalDiff = clamp(-(prevSE+SE), LO, HI) + OFFSET;  hist << evalDiff * MULT
        //   新: evalDiffQ8 = clamp(-(prevSE+SE) * SCORE_SCALE, LO_Q8, HI_Q8) + OFFSET_Q8
        //       hist << evalDiffQ8 * MULT_Q8 / SCORE_SCALE_SQ
        //   既定値は旧値 × SCORE_SCALE なので分子は SCORE_SCALE_SQ の倍数となり、
        //   切り捨ての差が出ず旧挙動と bit-exact。除算は最後の 1 回だけ (途中に丸めなし)。
        //
        //   オーバーフロー: staticEval は高々 ±32000 程度なので
        //   -(prevSE+SE) * SCORE_SCALE は最大でも 6.4e4 * 512 = 3.3e7 で int に収まる。
        //   clamp 後の evalDiffQ8 は最大 |LO_Q8| + OFFSET_Q8 ≒ 3.3e5 (SCALE=512 時)。
        //   これに MULT_Q8 (最大 1.6e4) を掛けると 5.5e9 で int を溢れるため、
        //   積だけ int64_t を経由する (このパスはノードあたり 1 回で除算はシフト化される)。
        const int evalDiffQ8 =
          std::clamp(-int((ss - 1)->staticEval + ss->staticEval) * SPSA::SCORE_SCALE,
                     SPSA::STATIC_EVAL_CLAMP_LO_Q8, SPSA::STATIC_EVAL_CLAMP_HI_Q8)
          + SPSA::STATIC_EVAL_OFFSET_Q8;
        mainHistory[~us][((ss - 1)->currentMove).raw()]
          << int(int64_t(evalDiffQ8) * SPSA::STATIC_EVAL_MAINHIST_MULT_Q8 / SPSA::SCORE_SCALE_SQ);
        if (!ttHit && type_of(pos.piece_on(prevSq)) != PAWN
            && ((ss - 1)->currentMove).type_of() != PROMOTION)
            sharedHistory.pawn_entry(pos)[pos.piece_on(prevSq)][prevSq]
              << int(int64_t(evalDiffQ8) * SPSA::STATIC_EVAL_PAWNHIST_MULT_Q8 / SPSA::SCORE_SCALE_SQ);
    }

    // Set up the improving flag, which is true if current static evaluation is
    // bigger than the previous static evaluation at our turn (if we were in
    // check at our previous move we go back until we weren't in check) and is
    // false otherwise. The improving flag is used in various pruning heuristics.
    // Similarly, opponentWorsening is true if our static evaluation is better
    // for us than at the last ply.

	// improving繝輔Λ繧ｰ繧定ｨｭ螳壹＠縺ｾ縺吶ゅ％繧後・縲∫樟蝨ｨ縺ｮ髱咏噪隧穂ｾ｡縺悟燕蝗槭・閾ｪ蛻・・謇狗分縺ｧ縺ｮ
    // 髱咏噪隧穂ｾ｡繧医ｊ螟ｧ縺阪＞蝣ｴ蜷医↓true縺ｨ縺ｪ繧翫∪縺呻ｼ亥燕蝗槭・謇九〒邇区焔繧貞女縺代※縺・◆蝣ｴ蜷医・
    // 邇区焔繧貞女縺代※縺・↑縺・ｱ髱｢縺ｾ縺ｧ驕｡縺｣縺ｦ隧穂ｾ｡縺励∪縺呻ｼ峨・
    // 縺昴ｌ莉･螟悶・蝣ｴ蜷医・false縺ｨ縺ｪ繧翫∪縺吶ゅ％縺ｮimproving繝輔Λ繧ｰ縺ｯ縲√＆縺ｾ縺悶∪縺ｪ譫晏・繧頑焔豕輔〒菴ｿ逕ｨ縺輔ｌ縺ｾ縺吶・
	// 蜷梧ｧ倥↓縲｛pponentWorsening 縺ｯ縲∫樟蝨ｨ縺ｮ髱咏噪隧穂ｾ｡縺・
	// 1謇句燕・亥燕縺ｮ謇狗分・峨・縺ｨ縺阪ｈ繧翫ｂ閾ｪ蛻・↓縺ｨ縺｣縺ｦ濶ｯ縺上↑縺｣縺ｦ縺・ｋ蝣ｴ蜷医↓ true 縺ｫ縺ｪ繧九・

	/*
		統 improving縺ｯ縲∬ｩ穂ｾ｡蛟､縺・謇句燕縺ｮ螻髱｢縺九ｉ荳翫′縺｣縺ｦ陦後▲縺ｦ縺・ｋ縺ｮ縺九・繝輔Λ繧ｰ
		    荳翫′縺｣縺ｦ陦後▲縺ｦ縺・ｋ縺ｪ繧画椌蛻医ｊ繧堤曝縺上☆繧九・

		    VALUE_NONE縺ｮ蝣ｴ蜷医・縲∫視謇九′縺九°縺｣縺ｦ縺・※evaluate()縺励※縺・↑縺・ｏ縺代□縺九ｉ縲・
		    譫晏・繧翫ｒ逕倥￥縺励※隱ｿ縺ｹ縺ｪ縺・→縺・￠縺ｪ縺・・縺ｧimproving謇ｱ縺・→縺吶ｋ縲・

		庁 VALUE_NONE == 32002縺ｪ縺ｮ縺ｧ縺薙ｌ繧医ｊ螟ｧ縺阪↑staticEval縺ｮ蛟､縺ｧ縺ゅｋ縺薙→縺ｯ縺ｪ縺・・
	*/

    improving = ss->staticEval > (ss - 2)->staticEval;

    // Threat eval 繧ｲ繝ｼ繝・ root 繝弱・繝峨・髱咏噪隧穂ｾ｡繧偵悟ｯｾ螻縺ｮ蠖｢蜍｢縲阪→縺励※騾驕ｿ縲・
    // 莉･髯阪％縺ｮ繧､繝・Ξ繝ｼ繧ｷ繝ｧ繝ｳ縺ｮ蜈ｨ繝弱・繝峨・ threatRootEval 繧貞盾辣ｧ縺励∬ｲ縺大ｱ髱｢縺ｮ縺ｨ縺阪□縺・threat 繧貞柑縺九○繧九・
    if (rootNode)
        threatRootEval = ss->staticEval;

	/*
		統 opponentWorsening縺ｯ縲∫嶌謇九・迥ｶ豕√′謔ｪ蛹悶＠縺ｦ縺・ｋ縺九・繝輔Λ繧ｰ縲・
	
		庁 ss->staticEval == - (ss-1)->staticEval 縺ｧ縺ゅｋ縺ｮ縺梧勸騾壹□縺後・
		    蟾ｦ霎ｺ縺ｮ縺ｻ縺・′螟ｧ縺阪＞(逶ｸ謇九・隧穂ｾ｡蛟､縺梧が蛹悶＠縺ｦ縺・ｋ)縺ｪ繧峨・縲・
		    逶ｸ謇九・隧穂ｾ｡蛟､縺梧が縺上↑縺｣縺ｦ縺・▲縺ｦ縺・ｋ縺薙→繧呈э蜻ｳ縺励※縺・ｋ縲・
	*/

    opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;

	// 統 1謇句燕縺ｮreduction縺ｫ蠢懊§縺滓ｮ九ｊdepth縺ｮ隱ｿ謨ｴ

    // Hindsight adjustment of reductions based on static evaluation difference.
	// 髱咏噪隧穂ｾ｡蛟､縺ｮ蟾ｮ縺ｫ蝓ｺ縺･縺・※縲∝炎貂幃㍼繧剃ｺ句ｾ檎噪縺ｫ隱ｿ謨ｴ縺吶ｋ縲・

    if (priorReduction >= 3 && !opponentWorsening)
        depth++;
    if (priorReduction >= 2 && depth >= 2 && ss->staticEval + (ss - 1)->staticEval > SPSA::HINDSIGHT_EVAL_SUM_TH)
        depth--;

	// -----------------------
    // Step 7. Razoring
    // -----------------------

    // If eval is really low, skip search entirely and return the qsearch value.
    // For PvNodes, we must have a guard against mates being returned.

	// 隧穂ｾ｡蛟､縺碁撼蟶ｸ縺ｫ菴弱＞蝣ｴ蜷医∵､懃ｴ｢繧貞ｮ悟・縺ｫ繧ｹ繧ｭ繝・・縺励※ qsearch 縺ｮ蛟､繧定ｿ斐＠縺ｾ縺吶・
    // PvNode 縺ｧ縺ｯ縲√メ繧ｧ繝・け繝｡繧､繝医′霑斐＆繧後ｋ縺ｮ繧帝亟縺舌◆繧√・繧ｬ繝ｼ繝峨′蠢・ｦ√〒縺吶・

    // 🚨RAZV: V >= RAZV_VMIN の局面では razoring を行わない (2026-09-28)。
    //   && の短絡評価なので RAZV_VMIN == 0 のときは P2Volm::value() を呼ばない = 完全に bit-exact。
    if (!PvNode && eval < alpha - SPSA::RAZORING_BASE - SPSA::RAZORING_DEPTH2 * depth * depth
        && (SPSA::RAZV_VMIN == 0 || YaneuraOu::P2Volm::value(pos) < SPSA::RAZV_VMIN) P2_PRUNE_NOT_BLOCKED)
#ifdef P2_PRUNE_STATS
    {
        const Value q_ = qsearch<NonPV>(pos, ss, alpha, beta);
        P2_PRUNE_NODE_VERIFY(0, (v_ > alpha) != (q_ > alpha))
        return q_;
    }
#else
        return qsearch<NonPV>(pos, ss, alpha, beta);
#endif

	// -----------------------
    // Step 8. Futility pruning: child node
    // Step 8. Futility譫晏・繧・: 蟄舌ヮ繝ｼ繝・
    // -----------------------

    // The depth condition is important for mate finding.
    // depth縺ｮ譚｡莉ｶ縺ｯ隧ｰ縺ｿ繧堤匱隕九☆繧九◆繧√↓驥崎ｦ√〒縺ゅｋ縲・

	/*
		統 縺薙・縺ゅ→縺ｮ谿九ｊ謗｢邏｢豺ｱ縺輔↓繧医▲縺ｦ縲∬ｩ穂ｾ｡蛟､縺悟､牙虚縺吶ｋ蟷・・futility_margin(depth)縺縺ｨ隕狗ｩ阪ｌ繧九・縺ｧ
			eval縺九ｉ縺薙ｌ繧貞ｼ輔＞縺ｦbeta繧医ｊ螟ｧ縺阪＞縺ｪ繧峨｜eta cut縺悟・譚･繧九・

			縺溘□縺励∝ｰ・｣九・邨ら乢縺ｧ縺ｯ隧穂ｾ｡蛟､縺ｮ螟牙虚縺ｮ蟷・・螟ｧ縺阪￥縺ｪ縺｣縺ｦ縺・￥縺ｮ縺ｧ縲・ｲ陦悟ｺｦ縺ｫ蠢懊§縺歿utility_margin縺悟ｿ・ｦ√→縺ｪ繧九・
			縺薙％縺ｧ縺ｯ騾ｲ陦悟ｺｦ縺ｨ縺励※gamePly()繧堤畑縺・ｋ縲ゅ％縺ｮ縺ｸ繧薙・縺ゅ→縺ｧ隱ｿ謨ｴ縺吶∋縺阪・

			Stockfish9縺ｾ縺ｧ縺ｯ縲’utility pruning繧偵〉oot node莉･螟悶↓驕ｩ逕ｨ縺励※縺・◆縺後・
			Stockfish10縺ｧnonPV縺ｫ縺ｮ縺ｿ縺ｮ驕ｩ逕ｨ縺ｫ螟画峩縺ｫ縺ｪ縺｣縺溘・
	*/

    {
        // futility margin
        // 庁 depth(谿九ｊ謗｢邏｢豺ｱ縺・縺ｫ蠢懊§縺歿utility margin縲・

		auto futility_margin = [&](Depth d) {
            Value futilityMult = SPSA::FUTILITY_MARGIN_BASE - SPSA::FUTILITY_MARGIN_TT_BONUS * !ss->ttHit;

            return futilityMult * d                                //
                 - (SPSA::FUTILITY_IMPROVING_SCALE * improving + SPSA::FUTILITY_OPP_WORSENING_SCALE * opponentWorsening) * futilityMult / 1024
#if defined(P2_SRCH_BUNDLE_A)
                 + BundleA::vdiv(bundleDivs.futilityCorr, std::abs(correctionValue),
                                 SPSA::FUTILITY_CORRECTION_DIV, BundleA::V_L2_FUTCORR);
#else
                 + std::abs(correctionValue) / SPSA::FUTILITY_CORRECTION_DIV;
#endif
        };

        // ★SF port #7(d) futility の連続化 (SF search.cpp:997-1008, commits c66acdac99/7bd32a51f5/7da31a1f1d):
        //   深さ上限 15 → SF_FUTILITY_V2_DEPTH_MAX (既定 19、"important for mate finding")、
        //   futilityMult = min(45 + 4*depth, 85) - 20*!ttHit (現行は定数 48 - 26*!ttHit)、
        //   improving/opponentWorsening 係数 2789/335、|corr|/198435、return (661*beta + 363*eval)/1024。
        //   eval 単位の定数は SF_EVAL_SCALE (既定 1024 = SF 値そのまま) で換算 (根拠は spsa_params.h)。
        //   既定 OFF なら分岐 1 つで従来式へ (bit-exact)。
        if (SPSA::SF_FUTILITY_V2)
        {
            if (!ss->ttPv && depth < SPSA::SF_FUTILITY_V2_DEPTH_MAX && eval >= beta
                && (!ttData.move || ttCapture) && !is_loss(beta) && !is_win(eval) P2_KHGATE_PRUNE_OK)
            {
                Value futilityMult = std::min(45 + int(depth) * 4, 85);
                futilityMult -= 20 * !ss->ttHit;
                futilityMult = futilityMult * SPSA::SF_EVAL_SCALE / 1024;

                Value futilityMargin = futilityMult * depth
                                     - (2789 * improving + 335 * opponentWorsening) * futilityMult / 1024
#if defined(P2_SRCH_BUNDLE_A)
                                     + BundleA::vdiv(bundleDivs.sfFutilityV2Corr, std::abs(correctionValue),
                                                     SPSA::SF_FUTILITY_V2_CORR_DIV, BundleA::V_L2_SFFUTCORR);
#else
                                     + std::abs(correctionValue) / SPSA::SF_FUTILITY_V2_CORR_DIV;
#endif

                if (eval - futilityMargin >= beta)
                    return (661 * beta + 363 * eval) / 1024;
            }
        }
        else if (!ss->ttPv && depth < 15 && eval >= beta && eval - futility_margin(depth) >= beta
            && (!ttData.move || ttCapture) && !is_loss(beta) && !is_win(eval) P2_PRUNE_NOT_BLOCKED P2_KHGATE_PRUNE_OK)
        {
            P2_PRUNE_NODE_VERIFY(1, v_ < beta)
            return (2 * beta + eval) / 3;
        }
    }

	// -----------------------
    // Step 9. Null move search with verification search
    // Step 9. 讀懆ｨｼ謗｢邏｢繧剃ｼｴ縺・ull move謗｢邏｢
    // -----------------------

    //  槙 eval縺恵eta繧定ｶ・∴縺ｦ縺・ｋ縺ｮ縺ｧ1謇九ヱ繧ｹ縺励※繧Ｃeta縺ｯ雜・∴縺昴≧縲ゅ□縺九ｉnull move繧定ｩｦ縺・
    if (cutNode && ss->staticEval >= beta - SPSA::NMP_MARGIN_DEPTH_MULT * depth - SPSA::NMP_MARGIN_IMPROVING_BONUS * improving + SPSA::NMP_MARGIN_OFFSET && !excludedMove
#if STOCKFISH
        && pos.non_pawn_material(us)
    // 庁 逶､荳翫↓pawn莉･螟悶・鬧偵′縺ゅｋ 竕・pawn縺縺代・邨ら乢縺ｧ縺ｯ縺ｪ縺・・
    // ､・蟆・｣九〒繧ゅ％繧後↓逶ｸ蠖薙☆繧区擅莉ｶ縺悟ｿ・ｦ√°繧ゅ・
        && ss->ply >= nmpMinPly
        // 蜷後§謇狗分蛛ｴ縺ｫ騾｣邯壹＠縺ｦnull move繧帝←逕ｨ縺励↑縺・
#endif
#if defined(EVAL_P2NNUE)
        // 決 V9.30/rust貅匁侠: NMP verification search 繧貞ｾｩ豢ｻ縺輔○繧九◆繧√・
        //    騾｣邯嗜ull move髦ｲ豁｢縺ｮ繧ｬ繝ｼ繝・ss->ply >= nmpMinPly)繧定ｿｽ蜉縺吶ｋ縲・
        && ss->ply >= nmpMinPly
#endif
        && !is_loss(beta) P2_PRUNE_NOT_BLOCKED P2_KHGATE_PRUNE_OK
    )
    {
        ASSERT_LV3((ss - 1)->currentMove != Move::null());

        // Null move dynamic reduction based on depth
        // (谿九ｊ謗｢邏｢)豺ｱ縺輔→隧穂ｾ｡蛟､縺ｫ蝓ｺ縺･縺熟ull move縺ｮ蜍慕噪縺ｪreduction

        // p2shogi 2026-11-01: SF 61c6871 の移植。 既定 (PARITY=0) で従来と bit-exact。
        //   odd thread だけ R を 1 減らす -> null move の部分木が深くなり検証が慎重になる。
#if defined(P2_SRCH_BUNDLE_A)
        Depth R = SPSA::NMP_REDUCTION_BASE
                + BundleA::vdiv(bundleDivs.nmpDepth, depth, SPSA::NMP_REDUCTION_DEPTH_DIV, BundleA::V_L2_NMP)
#else
        Depth R = SPSA::NMP_REDUCTION_BASE + depth / SPSA::NMP_REDUCTION_DEPTH_DIV
#endif
                - (SPSA::NMP_THREAD_PARITY ? int(threadIdx & 1) : 0);

        ss->currentMove                   = Move::null();
        ss->continuationHistory           = &contHistRoot[0][0][NO_PIECE][0];
        ss->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][0];

        // 庁  null move縺ｪ縺ｮ縺ｧ縲∫視謇九・縺九°縺｣縺ｦ縺・↑縺上※鬧貞叙繧翫〒繧ゅ↑縺・・
        //     繧医▲縺ｦ縲…ontinuationHistory[0(邇区焔縺九°縺｣縺ｦ縺ｪ縺・][0(鬧貞叙繧翫〒縺ｯ縺ｪ縺・][NO_PIECE][SQ_ZERO]
        //
        // 塔 邇区焔縺後°縺九▲縺ｦ縺・ｋ螻髱｢縺ｧ縺ｯ 竍代・譁ｹ縺ｫ縺ゅｋ goto moves_loop; 縺ｫ繧医▲縺ｦ縺昴▲縺｡縺ｫ陦後▲縺ｦ繧九・縺ｧ縲・
        //     縺薙％縺ｧ縺ｯ迴ｾ螻髱｢縺ｧ謇狗分蛛ｴ縺ｫ邇区焔縺後°縺九▲縺ｦ縺・↑縺・= 逶ｴ蜑阪・謖・＠謇・髱樊焔逡ｪ蛛ｴ)縺ｯ邇区焔縺ｧ縺ｯ縺ｪ縺・縺薙→縺後ｏ縺九▲縺ｦ縺・ｋ縲・
        //     do_null_move()縺ｯ縲√％縺ｮ譚｡莉ｶ繧呈ｺ縺溘☆蠢・ｦ√′縺ゅｋ縲・

        do_null_move(pos, st);

#if defined(P2_SRCH_BUNDLE_A)
        Value nullValue = -search_nonpv(pos, ss + 1, -beta, -beta + 1, depth - R, false);
#else
        Value nullValue = -search<NonPV>(pos, ss + 1, -beta, -beta + 1, depth - R, false);
#endif

        // Threat Continuation History (Phase B) 縺ｮ閼・ｨ∵治蜿・
        // THREAT_HISTORY_SCALE>0 縺九▽ Null Move 縺・cutoff 縺励↑縺九▲縺・騾壼ｸｸ moves loop 縺ｸ騾ｲ繧)繝弱・繝峨〒縺ｮ縺ｿ縲・
        // null 螻髱｢ TT 繧・undo 蜑阪↓ read-only probe 縺励∫嶌謇区怏蜉帶焔 threatMove 繧呈治蜿悶＠縺ｦ譁・ц繝上ャ繧ｷ繝･蛹悶・
        // probe() 縺ｯ read-only(tt.cpp:399: hit譎・read() 縺ｮ縺ｿ縲“eneration/age/replacement 髱槫､画峩)縲５TWriter 譛ｪ菴ｿ逕ｨ縲・
        // (ss+1)->currentMove 縺ｯ菴ｿ繧上↑縺・りｿｽ蜉謗｢邏｢/逕滓・/NNUE 縺ｪ縺励Ｔcale=0 縺ｧ縺ｯ probe 縺帙★=bit-exact縲・
        // 笘・PS譛驕ｩ蛹・ depth/root-eval 繧ｲ繝ｼ繝医ｒ gather 縺ｫ蜑榊偵＠縲よ治蜿也ｵ先棡(nodeThreatCtx)縺ｯ MovePicker(驕ｩ逕ｨ)縺ｨ
        //   譖ｴ譁ｰ(蟄ｦ鄙・縺ｮ2邂・園縺ｧ菴ｿ繧上ｌ繧九・縺ｧ縲√◎縺ｮ縲後←縺｡繧峨°縺悟ｿ・ｦ√→縺吶ｋ縲肴擅莉ｶ = 驕ｩ逕ｨ繧ｲ繝ｼ繝医→蟄ｦ鄙偵ご繝ｼ繝医・
        //   繧・ｋ縺・・(min depth / max eval)繧呈ｺ縺溘☆縺ｨ縺阪□縺第治蜿悶☆繧九よｺ縺溘＆縺ｪ縺・→縺阪・謗｡蜿也ｵ先棡縺御ｸ｡閠・→繧・
        //   菴ｿ繧上ｌ縺ｪ縺・・縺ｧ probe 繧堤怐縺・※繧よ嫌蜍穂ｸ榊､峨・
        // 笘・ｳｨ諢・ 蟄ｦ鄙偵ご繝ｼ繝・THREAT_LEARN_*)繧帝←逕ｨ繧ｲ繝ｼ繝医ｈ繧雁ｺ・￡繧九→縲√％縺ｮ謗｡蜿匁擅莉ｶ繧ゅ◎縺ｮ蛻・□縺醍ｷｩ縺上↑繧翫・
        //   蠕捺擂 probe 繧堤怐縺代※縺・◆繝弱・繝・豬・＞/蜆ｪ蜍｢)縺ｧ繧・null 螻髱｢ TT probe 縺瑚ｵｰ繧・= probe 蝗樊焚蠅・竊・NPS 菴惹ｸ九・
        //   譌｢螳壹〒縺ｯ荳｡繧ｲ繝ｼ繝医′蜷悟､縺ｪ縺ｮ縺ｧ謗｡蜿匁擅莉ｶ繧ょｾ捺擂縺ｨ蜷御ｸ = bit-exact縲・
        if (SPSA::THREAT_HISTORY_SCALE > 0 && nullValue < beta
            && depth >= std::min(SPSA::THREAT_MIN_APPLY_DEPTH, SPSA::THREAT_LEARN_MIN_DEPTH)
            && threatRootEval < std::max(SPSA::THREAT_MAX_EVAL, SPSA::THREAT_LEARN_MAX_EVAL))
        {
#ifdef THREAT_STATS
            g_threatStats.nmNoCutoff.fetch_add(1, std::memory_order_relaxed);
#endif
            auto [tHit, tData, tWriter] = tt.probe(pos.key(), pos);
            (void) tWriter;  // 譖ｸ縺九↑縺・TT髱槫､画峩)
            if (tHit && tData.move && (tData.bound == BOUND_LOWER || tData.bound == BOUND_EXACT)
                && tData.depth >= SPSA::THREAT_MIN_DEPTH && !is_decisive(tData.value)
                && pos.pseudo_legal_s<true>(tData.move))
            {
                Move        tm = tData.move;
                const bool  gc = pos.gives_check(tm);
                ThreatClass tc = classify_threat(pos, tm, gc);
                // THREAT_CLASS_MASK 縺ｧ謗｡逕ｨ繧ｯ繝ｩ繧ｹ繧堤ｵ槭ｋ(邊励＞ Capture 遲峨・繝弱う繧ｺ貅舌ｒ髯､螟門庄閭ｽ)縲・
                if ((SPSA::THREAT_CLASS_MASK >> int(tc)) & 1)
                {
                    // 譁・ц side 縺ｯ response 蛛ｴ(=迴ｾ繝弱・繝画焔逡ｪ us)縲Ｕhreat 鬧・蜈亥合縺ｯ null 螻髱｢(逶ｸ謇区焔逡ｪ)縺ｮ move 縺九ｉ縲・
                    nodeThreatCtx = threat_history_context(us, tc, pos.moved_piece(tm), tm.to_sq());
                    threatValid   = true;
#ifdef THREAT_STATS
                    g_threatStats.adopt.fetch_add(1, std::memory_order_relaxed);
                    g_threatStats.cls[int(tc)].fetch_add(1, std::memory_order_relaxed);
#endif
                }
            }
        }

        undo_null_move(pos);

#if defined(EVAL_P2NNUE)
        // 🌈 NmpWinCutoff (NAGISA V3.1 準拠、既定 false = bit-exact)
        //   本家 9.40/9.60 は「証明されていない詰みスコアは返さない」ため、null move が
        //   詰みスコアで beta を超えた場合は cutoff せずこのノードを通常探索する。
        //   NAGISA V3.1 は `if (nullValue >= beta) return is_win(nullValue) ? beta : nullValue;`
        //   と書いており、詰みスコアでも beta へ丸めて必ず cutoff する = 探索木が細くなる。
        //   NmpVerification=false と併用すると NMP が NAGISA と完全に同一の挙動になる。
        if (Eval::p2_nmp_win_cutoff && nullValue >= beta && is_win(nullValue))
            return beta;
#endif

        if (nullValue >= beta && !is_win(nullValue))
#if STOCKFISH || defined(EVAL_P2NNUE)
        // Do not return unproven mate or TB scores
        // 險ｼ譏弱＆繧後※縺・↑縺・ate score繧УB score縺ｯreturn縺ｧ霑斐＆縺ｪ縺・・
        // 決 EVAL_P2NNUE 縺ｧ縺ｯ V9.30/rust貅匁侠縺ｧ verification search 繧呈怏蜉ｹ蛹悶☆繧・
        //    (V9.40縺ｧ蟆・｣狗畑縺ｫ蜑企勁縺輔ｌ縺滓､懆ｨｼ謗｢邏｢繧・rollback)縲・
        {
#if defined(EVAL_P2NNUE)
            // 決 USI繧ｪ繝励す繝ｧ繝ｳ NmpVerification=false 縺ｪ繧・V9.40逶ｸ蠖・讀懆ｨｼ謗｢邏｢縺ｪ縺・縲・
            if (!Eval::p2_nmp_verification)
            {
                P2_PRUNE_NODE_VERIFY(2, v_ < beta)
                return nullValue;
            }
#endif
            // 1謇九ヱ繧ｹ縺励※繧Ｃeta繧剃ｸ雁屓繧翫◎縺・〒縺ゅｋ縺薙→縺後ｏ縺九▲縺溘・縺ｧ
            // 縺薙ｌ繧偵ｂ縺・ｰ代＠縺｡繧・ｓ縺ｨ讀懆ｨｼ縺励↑縺翫☆縲・

            if (nmpMinPly || depth < 16)
            {
                P2_PRUNE_NODE_VERIFY(2, v_ < beta)
                return nullValue;
            }

            ASSERT_LV3(!nmpMinPly);  // Recursive verification is not allowed
                                     // 蜀榊ｸｰ逧・↑讀懆ｨｼ縺ｯ隱阪ａ縺ｦ縺・↑縺・・

            // Do verification search at high depths, with null move pruning disabled
            // until ply exceeds nmpMinPly.
            //
            // 庁 null move譫晏・繧翫ｒ辟｡蜉ｹ蛹悶＠縺ｦ縲｝ly縺系mpMinPly繧定ｶ・∴繧九∪縺ｧ
            //     鬮倥＞depth縺ｧ讀懆ｨｼ縺ｮ縺溘ａ縺ｮ謗｢邏｢繧定｡後≧縲・

            nmpMinPly = ss->ply + 3 * (depth - R) / 4;

            // 統 nullMove縺帙★縺ｫ(迴ｾ蝨ｨ縺ｮnode縺ｨ蜷後§謇狗分縺ｧ)蜷後§豺ｱ縺輔〒謗｢邏｢縺励↑縺翫＠縺ｦ譛ｬ蠖薙↓beta繧定ｶ・∴繧九°讀懆ｨｼ縺吶ｋ縲・
            //     cutNode縺ｫ縺励↑縺・・

#if defined(P2_SRCH_BUNDLE_A)
            Value v = search_nonpv(pos, ss, beta - 1, beta, depth - R, false);
#else
            Value v = search<NonPV>(pos, ss, beta - 1, beta, depth - R, false);
#endif

            nmpMinPly = 0;

            if (v >= beta)
            {
                P2_PRUNE_NODE_VERIFY(2, v_ < beta)
                return nullValue;
            }
        }
#else
            // null move pruning縺ｮ讀懆ｨｼ謗｢邏｢縺ｯ縲√ヱ繧ｹ (null move) 縺励◆譁ｹ縺梧怏蛻ｩ縺ｫ縺ｪ繧句ｱ髱｢縺ｧ縺ｮ隱､縺｣縺滓椌蛻医ｊ髦ｲ豁｢縺ｮ縺溘ａ縺ｫ蟄伜惠縺吶ｋ縺後・
            // 蟆・｣九〒縺ｯ縺昴・繧医≧縺ｪ縺薙→縺ｯ繝√ぉ繧ｹ繧医ｊ縺ｯ繧九°縺ｫ蟆代↑縺・◆繧∽ｸ崎ｦ√・
            return nullValue;
#endif
    }

	// 縺薙％縺ｧimproving險育ｮ励＠縺ｪ縺翫☆縲・

    improving |= ss->staticEval >= beta;

	// -----------------------
    // Step 10. Internal iterative reductions
    // Step 10. 蜀・Κ蜿榊ｾｩ繝ｪ繝繧ｯ繧ｷ繝ｧ繝ｳ
    // -----------------------

	// At sufficient depth, reduce depth for PV/Cut nodes without a TTMove.
	// (*Scaler) Making IIR more aggressive scales poorly.

    // 蜊∝・縺ｪ謗｢邏｢豺ｱ縺輔′縺ゅｋ蝣ｴ蜷医∫ｽｮ謠幄｡ｨ・・TMove・峨↓謇九′縺ｪ縺ПV繝弱・繝峨ｄCut繝弱・繝峨↓縺､縺・※縺ｯ謗｢邏｢豺ｱ縺輔ｒ蜑頑ｸ帙☆繧九・
    //・・Scaler・迂IR 繧偵ｈ繧翫い繧ｰ繝ｬ繝・す繝悶↓縺吶ｋ縺ｨ縲√せ繧ｱ繝ｼ繝ｪ繝ｳ繧ｰ蜉ｹ邇・′謔ｪ蛹悶☆繧九・

    if (!ss->followPV && !allNode && depth >= SPSA::IIR_MIN_DEPTH && !ttData.move
        && priorReduction <= SPSA::IIR_PRIOR_REDUCTION_MAX)
        depth--;

#if OLD_CODE
    // 決 莉･蜑阪・繧ｳ繝ｼ繝峨・縺ｻ縺・′蠑ｷ縺・庄閭ｽ諤ｧ縺後≠繧九・

	if (    PvNode
		&& !ttData.move)
		depth -= 3;

	// Use qsearch if depth <= 0
	// (depth繧池eduction縺励◆邨先棡縲・繧ゅ＠depth <= 0縺ｪ繧衛search繧堤畑縺・ｋ

	if (depth <= 0)
		return qsearch<PV>(pos, ss, alpha, beta);

	// For cutNodes, if depth is high enough, decrease depth by 2 if there is no ttMove,
	// or by 1 if there is a ttMove with an upper bound.

	// 繧ｫ繝・ヨ繝弱・繝峨・蝣ｴ蜷医∵ｷｱ縺輔′蜊∝・縺ｫ縺ゅｋ蝣ｴ蜷医・縲》tMove縺後↑縺・ｴ蜷医↓豺ｱ縺輔ｒ2貂帙ｉ縺励・
	// ttMove縺御ｸ企剞蛟､繧呈戟縺､蝣ｴ蜷医・豺ｱ縺輔ｒ1貂帙ｉ縺励∪縺吶・

	if (cutNode && depth >= 7 && (!ttData.move || ttData.bound == BOUND_UPPER))
		depth -= 1 + !ttData.move;
#endif
        
	// -----------------------
    // Step 11. ProbCut
    // -----------------------

    // If we have a good enough capture (or queen promotion) and a reduced search
    // returns a value much above beta, we can (almost) safely prune the previous move.

	// 蜊∝・縺ｫ濶ｯ縺・ｧ貞叙繧奇ｼ医∪縺溘・繧ｯ繧､繝ｼ繝ｳ譏・ｼ・峨′縺ゅｊ縲・
    // (谿九ｊ謗｢邏｢豺ｱ縺輔ｒ)蜑頑ｸ帙＆繧後◆謗｢邏｢縺ｧbeta繧貞､ｧ蟷・↓荳雁屓繧句､縺瑚ｿ斐＆繧後ｋ蝣ｴ蜷医・
    // 逶ｴ蜑阪・謇九ｒ・医⊇縺ｼ・牙ｮ牙・縺ｫ譫晏・繧翫〒縺阪∪縺吶・

    // probCut縺ｫ菴ｿ縺・eta蛟､縲・
    probCutBeta = beta + SPSA::PROBCUT_MARGIN - SPSA::PROBCUT_IMPROVING_MARGIN * improving;

	if (depth >= 3
        && !is_decisive(beta)
        // If value from transposition table is lower than probCutBeta, don't attempt
        // probCut there
        // 鄂ｮ謠幄｡ｨ縺九ｉ蠕励◆蛟､縺・probCutBeta 繧医ｊ菴弱＞蝣ｴ蜷医・縲√◎縺薙〒 probCut 繧定ｩｦ縺ｿ縺ｪ縺・
        && !(is_valid(ttData.value) && ttData.value < probCutBeta) P2_PRUNE_NOT_BLOCKED)
    {
        ASSERT_LV3(probCutBeta < VALUE_INFINITE && probCutBeta > beta);

#if STOCKFISH
        MovePicker mp(pos, ttData.move, probCutBeta - ss->staticEval, &captureHistory);
#else
        MovePicker mp(pos, ttData.move, probCutBeta - ss->staticEval, &captureHistory,
                      search_options.generate_all_legal_moves);
#endif

        Depth      probCutDepth = depth - 4;

		// 庁 隧ｦ陦悟屓謨ｰ縺ｯ2蝗・cutNode縺ｪ繧・蝗・縺ｾ縺ｧ縺ｨ縺吶ｋ縲・繧医＆縺偵↑謖・＠謇九ｒ3縺､隧ｦ縺励※鬧・岼縺ｪ繧蛾ｧ・岼縺ｨ縺・≧謇ｱ縺・
        //     cf. Do move-count pruning in probcut : https://github.com/official-stockfish/Stockfish/commit/b87308692a434d6725da72bbbb38a38d3cac1d5f

        while ((move = mp.next_move()) != Move::none())
        {
            ASSERT_LV3(move.is_ok());
            ASSERT_LV5(pos.pseudo_legal_s<true>(move) && pos.legal_promote(move));

            if (move == excludedMove || !pos.legal(move))
                continue;

            //assert(pos.capture_stage(move));
            // 笞 move縺ｨ縺励※豁ｩ縺ｮ謌舌ｊ繧りｿ斐▲縺ｦ縺上ｋ縺後√％繧後′capture_stage()縺ｨ荳閾ｴ縺吶ｋ縺ｨ縺ｯ髯舌ｉ縺ｪ縺・・
            //     MovePicker縺ｯprob cut縺ｮ譎ゅ↓縲・
            //    (GenerateAllLegalMoves繧ｪ繝励す繝ｧ繝ｳ縺後が繝ｳ縺ｧ縺ゅ▲縺ｦ繧・豁ｩ縺ｮ謌舌ｉ縺壹・霑斐＠縺ｦ縺薙↑縺・％縺ｨ繧剃ｿ晁ｨｼ縺吶∋縺阪・

#if STOCKFISH
            do_move(pos, move, st, ss);
#else
            do_move_(pos, move, st, ss);
#endif

            // Perform a preliminary qsearch to verify that the move holds
            // 縺薙・謖・＠謇九′繧医＆縺偵〒縺ゅｋ縺薙→繧堤｢ｺ隱阪☆繧九◆繧√・莠亥ｙ逧・↑qsearch

            value = -qsearch<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1);

            // If the qsearch held, perform the regular search
            // qsearch 縺檎ｶｭ謖√＆繧後◆蝣ｴ蜷医・壼ｸｸ縺ｮ謗｢邏｢繧貞ｮ溯｡後☆繧・

            if (value >= probCutBeta && probCutDepth > 0)
#if defined(P2_SRCH_BUNDLE_A)
                value = -search_nonpv(pos, ss + 1, -probCutBeta, -probCutBeta + 1, probCutDepth,
                                      !cutNode);
#else
                value = -search<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1, probCutDepth,
                                       !cutNode);
#endif

            undo_move(pos, move);

            if (value >= probCutBeta)
            {
                // Save ProbCut data into transposition table
                // ProbCut縺ｮdata繧堤ｽｮ謠幄｡ｨ縺ｫ菫晏ｭ倥☆繧九・

				ttWriter.write(posKey, value_to_tt(value, ss->ply), ss->ttPv, BOUND_LOWER,
                               probCutDepth + 1, move, unadjustedStaticEval, tt.generation());

                if (!is_decisive(value))
                {
                    P2_PRUNE_NODE_VERIFY(3, v_ < beta)
                    return value - (probCutBeta - beta);
                }
            }
        } // end of while
    }

    // Uncertainty-linked LMR (隨ｬ1繝代ャ繝・: 隧穂ｾ｡諤･螟牙ｺｦ繧堤ｮ怜・縺励Ο繝ｼ繧ｫ繝ｫ騾驕ｿ縲・
    // 縺薙％縺ｯ髱枴nCheck縺ｮ蜷域ｵ∫せ(inCheck縺ｯ goto moves_loop 縺ｧ荳九∈霍ｳ縺ｶ)縺九▽ SE(Step15)繧医ｊ蜑阪・
    // 縺薙％縺ｧ遒ｺ螳壼､繧帝驕ｿ縺吶ｋ縺溘ａ縲ヾE縺悟酔荳ply縺ｮss->staticEval繧剃ｸ頑嶌縺阪＠縺ｦ繧ょｽｱ髻ｿ繧貞女縺代↑縺・・
    // 霑ｽ蜉eval/謗｢邏｢/逕滓・縺ｪ縺励よ里險育ｮ励・ staticEval(謇狗分隕也せ)縺ｮ縺ｿ菴ｿ逕ｨ縲・
    // 笘・PS譛驕ｩ蛹・ 髱櫁ｨ域ｸｬ繝薙Ν繝峨〒縺ｯ UNCERT_LMR_THRESHOLD=0(譌｢螳唹FF)縺ｪ繧臥ｮ怜・閾ｪ菴薙ｒ逵√￥縲・
    //   nodeEvalVolatility/validEvalVolatility 縺ｯ THRESHOLD>0 縺ｮ遏ｭ邨｡譚｡莉ｶ荳九〒縺励°隱ｭ縺ｾ繧後↑縺・◆繧・bit-exact縲・
    //   (UNCERT_LMR_MEASURE 繝薙Ν繝峨・髢ｾ蛟､0縺ｧ繧ょ・蟶・ｨ域ｸｬ縺ｫ菴ｿ縺・◆繧∝ｸｸ譎らｮ怜・繧堤ｶｭ謖・
    if (
#ifndef UNCERT_LMR_MEASURE
        SPSA::UNCERT_LMR_THRESHOLD > 0 &&
#endif
        !ss->inCheck && !(ss - 1)->inCheck && ((ss - 1)->currentMove).is_ok()
        && is_valid(ss->staticEval) && is_valid((ss - 1)->staticEval)
        && !is_decisive(ss->staticEval) && !is_decisive((ss - 1)->staticEval))
    {
        // staticEval 縺ｯ謇狗分隕也せ縲る壼ｸｸ ss->staticEval 竕・-(ss-1)->staticEval 縺ｪ縺ｮ縺ｧ縲・
        // 蜥後・邨ｶ蟇ｾ蛟､縺悟､ｧ縺阪＞ = 隕ｪ螻髱｢縺九ｉ隧穂ｾ｡縺梧･螟・= 鬮倅ｸ咲｢ｺ螳滓ｧ縲・
        nodeEvalVolatility  = std::min(std::abs(int(ss->staticEval) + int((ss - 1)->staticEval)),
                                       UNCERT_LMR_MAX_DELTA);
        validEvalVolatility = true;
#ifdef UNCERT_LMR_MEASURE
        g_uncertMeasure.validVolNodes.fetch_add(1, std::memory_order_relaxed);
#endif
    }

moves_loop:  // When in check, search starts here
			 // 邇区焔縺後°縺九▲縺ｦ縺・ｋ螻髱｢縺ｧ縺ｯ縲∵爾邏｢縺ｯ縺薙％縺九ｉ蟋九∪繧九・

	// -----------------------
    // Step 12. A small Probcut idea
    // Step 12. 蟆上＆縺ｪProbcut縺ｮ繧｢繧､繝・い
    // -----------------------

    probCutBeta = beta + SPSA::SMALL_PROBCUT_MARGIN;
    if ((ttData.bound & BOUND_LOWER) && ttData.depth >= depth - 4 && ttData.value >= probCutBeta
        && !is_decisive(beta) && is_valid(ttData.value) && !is_decisive(ttData.value) P2_PRUNE_NOT_BLOCKED)
    {
        P2_PRUNE_NODE_VERIFY(4, v_ < beta)
        return probCutBeta;
    }

	// -----------------------
    // 噫 moves loop縺ｫ蜈･繧句燕縺ｮ貅門ｙ
    // -----------------------

	// continuationHistory[0]  = Counter Move History    : 縺ゅｋ謖・＠謇九′謖・＆繧後◆縺ｨ縺阪・蠢懈焔
    // continuationHistory[1]  = Follow up Move History  : 2謇句燕縺ｮ閾ｪ蛻・・謖・＠謇九・邯咏ｶ壽焔
    // continuationHistory[3]  = Follow up Move History2 : 4謇句燕縺九ｉ縺ｮ邯咏ｶ壽焔

    const PieceToHistory* contHist[] = {
      (ss - 1)->continuationHistory, (ss - 2)->continuationHistory, (ss - 3)->continuationHistory,
      (ss - 4)->continuationHistory, (ss - 5)->continuationHistory, (ss - 6)->continuationHistory};


    if (P2Stat::enabled()) P2Stat::mv_loop.fetch_add(1, std::memory_order_relaxed);
    MovePicker mp(pos, ttData.move, depth, &mainHistory, &lowPlyHistory, &captureHistory, contHist,
                  &sharedHistory, ss->ply,
                  SPSA::DROP_CONT_HIST_SCALE > 0 ? &dropContHistory : nullptr,
                  (prevSq != SQ_NONE ? int(type_of(pos.piece_on(prevSq))) : -1),
                  (SPSA::THREAT_HISTORY_SCALE > 0 && threatValid
                     && depth >= SPSA::THREAT_MIN_APPLY_DEPTH
                     && threatRootEval < SPSA::THREAT_MAX_EVAL)
                    ? &threatHistory
                    : nullptr,
                  nodeThreatCtx
#if !STOCKFISH
                  ,
                  search_options.generate_all_legal_moves
#endif
    );

    value = bestValue;

	// 隱ｿ縺ｹ縺滓欠縺玲焔縺ｮ謨ｰ(蜷域ｳ墓焔縺ｫ髯舌ｋ)
    int moveCount = 0;

    // -----------------------
    // M1: 遅延 (ABDADA-lite / busy 表)   2026-11-02
    // -----------------------
    //   実測: 3 スレッドが各々「1 スレッドで読む木の 93-95%」を重複して読んでいた
    //     (depth 15/18/20 で同一 / TTD 0.97-1.00x)。 原因は多様性不足ではなく**分業の不在**。
    //   機構: 2 手目以降で、 子局面が他 thread に読まれていればその手を後回しにして兄弟手へ。
    //     後回しの手は **末尾で元の moveCount のまま必ず読む** ので、
    //     単スレッドの探索意味論は不変で、 **正しさは busy 表に一切依存しない**
    //     (偽陽性 = 余計に後回し / 偽陰性 = 従来どおり重複)。
    //   設計: runs/prereg/DESIGN-20261102-SMP-DEFER.md §2.1
    //   🚨不変条件:
    //     I-1 後回しは必ず読まれる (beta cut で抜けた場合を除く = 従来も読まない)
    //     I-2 後回しの moveCount は**最初に数えた値**。 ループ後は **総数**に戻す
    //         (忘れると `if (!moveCount)` の詰み判定が壊れる / 直後の ASSERT_LV5 で落ちる)
    //     I-3 enter / leave は RAII で対にする (全 return 経路で leave)
    //     I-4 excludedMove (SE の再帰) では enter しない (二重 enter で外側の印が消える)
    //     I-5 既定 (SMP_DEFER_MIN_DEPTH=0) / Threads=1 / MultiPV>1 では 1 命令も実行しない
    // M1 の後回し + M4 の委譲で共有する。 slot < 0 = 委譲していない (M1 のみ)。
    struct Deferred { Move m; int mc; int slot; };
    Deferred   deferred[8];
    int        nDeferred = 0, iDeferred = 0;
    int        moveCountTotal = 0;
    bool       phase2 = false;
    // smpBusy は iterative_deepening が条件を満たすときだけ非 nullptr にする (I-5)
    // M4 委譲の可否。 root と SE の再帰 (excludedMove) は除く。
    //   🚨SE の再帰では除外手の扱いが helper に伝わらないので提示してはいけない。
    //   🚨手順の再生の費用が見合う深さに限る (深い節は部分木が小さい)。
    const bool delegOK = deleg != nullptr && !rootNode && !excludedMove
                      && depth >= SPSA::YBWC_MIN_DEPTH && ss->ply >= 1
                      && ss->ply <= DelegSlot::PATH_CAP;
    const bool deferOK = smpBusy != nullptr && depth >= SPSA::SMP_DEFER_MIN_DEPTH
                      && !excludedMove;                                  // I-4
    SmpBusyScope busyScope(deferOK ? smpBusy : nullptr, posKey);         // I-3

    // 1 周目は MovePicker、 尽きたら 2 周目で後回しを元の moveCount のまま返す
    auto next_move_defer = [&]() -> Move {
        if (!phase2)
        {
            Move m = mp.next_move();
            if (m != Move::none())
                return m;
            phase2 = true;
        }
        if (iDeferred < nDeferred)
        {
            const Deferred& d = deferred[iDeferred++];
            ++smpDeferRead;                                              // D2
            moveCount     = d.mc;                                        // I-2
            ss->moveCount = d.mc;
            return d.m;
        }
        return Move::none();
    };

	// -----------------------
    // Step 13. Loop through all pseudo-legal moves until no moves remain
    //			or a beta cutoff occurs.
    // Step 13. 謫ｬ莨ｼ蜷域ｳ墓焔繧偵☆縺ｹ縺ｦ繝ｫ繝ｼ繝励＠縲∵焔縺後↑縺上↑繧九°
    //          繧ゅ＠縺上・beta繧ｫ繝・ヨ繧ｪ繝輔′逋ｺ逕溘☆繧九∪縺ｧ郢ｰ繧願ｿ斐＠縺ｾ縺吶・
    // -----------------------

	// 庁 MovePicker縺瑚ｿ斐☆謖・＠謇九・pseudo-legal縺ｧ縺ゅｋ縺薙→縺ｯ菫晁ｨｼ縺輔ｌ縺ｦ縺・ｋ縺後・
	//     do_move()縺ｾ縺ｧ縺ｫ縺ｯlegal縺九←縺・°縺ｮ蛻､螳壹′蠢・ｦ√・

    while ((move = next_move_defer()) != Move::none())
    {
        ASSERT_LV3(move.is_ok());
        ASSERT_LV5(pos.pseudo_legal_s<true>(move) && pos.legal_promote(move));

        if (move == excludedMove)
            continue;

        // Check for legality
        // 謖・＠謇九・蜷域ｳ墓ｧ縺ｮ繝√ぉ繝・け

        /*
		   統 root node縺ｪ繧峨〉ootMoves縺ｫ縺ｪ縺代ｌ縺ｰlegal縺ｧ縺ｯ縺ｪ縺・・縺ｧ縺薙・繝√ぉ繝・け縺ｯ荳崎ｦ√□縺後・
			   root node縺ｯ蜈ｨ菴薙°繧芽ｦ九ｋ縺ｨ讌ｵ繧上★縺九↑縺ｮ縺ｧ縺昴・繝√ぉ繝・け繧堤ｫｯ謚倥ｋ縺ｻ縺・′濶ｯ縺・ｈ縺・□縲・

			   髱槫粋豕墓焔縺ｯ縺ｻ縺ｨ繧薙←蜷ｫ縺ｾ繧後※縺・↑縺・°繧峨∽ｻ･蜑阪・縺薙・蛻､螳壹・do_move()縺ｮ逶ｴ蜑阪∪縺ｧ
			   驕・ｻｶ縺輔○縺溘⊇縺・′蠕励□縺｣縺溘′縲‥o_move()縺吶ｋ縺ｾ縺ｧ縺ｮ譫晏・繧翫′蠅励∴縺ｦ縺阪◆縺ｮ縺ｧ縲・
			   縺薙％縺ｧ繧・▲縺溘⊇縺・′濶ｯ縺・ｈ縺・□縲・
		*/

        if (!pos.legal(move))
            continue;

        // At root obey the "searchmoves" option and skip moves not listed in Root
        // Move List. In MultiPV mode we also skip PV moves that have been already
        // searched and those of lower "TB rank" if we are in a TB root position.

        // 繝ｫ繝ｼ繝医〒 "searchmoves" 繧ｪ繝励す繝ｧ繝ｳ縺ｫ蠕薙＞縲ヽoot Move List縺ｫ繝ｪ繧ｹ繝医＆繧後※縺・↑縺・焔繧偵せ繧ｭ繝・・縺励∪縺吶・
        // MultiPV繝｢繝ｼ繝峨〒縺ｯ縲∵里縺ｫ讀懃ｴ｢縺輔ｌ縺蘖V縺ｮ謇九ｄ縲ゝB繝ｫ繝ｼ繝井ｽ咲ｽｮ縺ｫ縺・ｋ蝣ｴ蜷医・TB繝ｩ繝ｳ繧ｯ縺御ｽ弱＞謇九ｂ
        // 繧ｹ繧ｭ繝・・縺励∪縺吶・

        // 庁 root node縺ｧ縺ｯ縲〉ootMoves()縺ｮ髮・粋縺ｫ蜷ｫ縺ｾ繧後※縺・↑縺・欠縺玲焔縺ｯ謗｢邏｢繧偵せ繧ｭ繝・・縺吶ｋ縲・

        if (rootNode
#if STOCKFISH
            && !std::count(rootMoves.begin() + pvLast,
        // 統 蟆・｣九〒縺ｯ縺薙・蜃ｦ逅・ｸ崎ｦ√↑縺ｮ縺ｧ繧・・縺・ｉ邇九〒縺ｯpvLast縺ｯ菴ｿ繧上↑縺・・
#else
            && !std::count(rootMoves.begin() + pvIdx,
#endif
                           rootMoves.end(), move))
            continue;

        // do_move()縺励◆謖・＠謇九・謨ｰ縺ｮ繧､繝ｳ繧ｯ繝ｪ繝｡繝ｳ繝・
        if (!phase2)
        {
            ss->moveCount  = ++moveCount;
            moveCountTotal = moveCount;

            // 2 手目以降で、 子局面が他 thread に展開されていれば後回し (**数えてから**: I-2)。
            // 1 手目 (TT 手) は絶対に後回しにしない
            // (cut node は 1 手目で切れる / PV node は alpha を確定させる)。
            if (deferOK && moveCount > 1 && nDeferred < 8 && move != Move::win())
            {
                ++smpBusyProbe;                                          // D2
                if (smpBusy->is_busy(pos.key_after(move)))
                {
                    deferred[nDeferred++] = {move, moveCount, -1};
                    ++smpDeferCnt;                                       // D2
                    continue;
                }
            }
        }

        // -----------------------
        // M3 根の YBWC: 長兄は master だけが読み、 兄弟手は取り合う   2026-11-02
        // -----------------------
        //   🚨既定では rootSplit == nullptr なのでこの分岐は 1 命令も実行されない。
        //   🚨M1 (遅延) とは排他なので phase2 との干渉は起きない (配線で排他にしてある)。
        if (rootNode && rootSplit && ybwcGen)
        {
            const int idx = moveCount - 1;
            if (idx == 0)
            {
                // 長兄。 master だけが読む (I-4)。 helper は読まずに次へ。
                //   helper の rootMoves は報告に使わないので rm は触らない。
                //   🚨search() の中には mainThread 変数が無いので is_mainthread() を使う。
                if (!is_mainthread())
                    continue;
            }
            else if (!rootSplit->try_take(ybwcGen, idx))
                continue;                       // 他 thread が取った手
            else
            {
                ybwcIdx       = idx;
                ybwcMine[idx] = true;
                if (!is_mainthread())
                    rootSplit->cntTaken.fetch_add(1, std::memory_order_relaxed);
                // 🎯共有 alpha を使う。 これが YBWC の肝 (投機の税を払わない)。
                //   他 thread が既に上げた alpha をそのまま使えるので、 兄弟手は
                //   null window の安い scout になる。
                alpha = std::max(alpha, rootSplit->alpha());
            }
        }

// ､・Stockfish譛ｬ螳ｶ縺ｮ縺薙・隱ｭ縺ｿ遲九・蜃ｺ蜉帙∫ｴｰ縺九☆縺弱ｋ縺ｮ縺ｧ譎る俣繧偵Ο繧ｹ縺吶ｋ縲ゅ＠縺ｪ縺・⊇縺・′縺・＞縺ｨ諤昴≧縲・
#if STOCKFISH

		if (rootNode && is_mainthread() && nodes > 10000000)
        {
            main_manager()->updates.onIter(
              {depth, UCIEngine::move(move, pos.is_chess960()), moveCount + pvIdx});
        }
#endif

        // 庁 谺｡縺ｮnode縺ｮpv繧偵け繝ｪ繧｢縺励※縺翫￥縲・

        if (PvNode)
            (ss + 1)->pv = nullptr;

        // -----------------------
        //      extension
        //      謗｢邏｢縺ｮ蟒ｶ髟ｷ
        // -----------------------

        // 莉雁屓蟒ｶ髟ｷ縺吶ｋ谿九ｊ謗｢邏｢豺ｱ縺・
        extension = 0;

        // 莉雁屓謐慕佐縺吶ｋ鬧・
        capture    = pos.capture_stage(move);

        // 莉雁屓遘ｻ蜍輔＆縺帙ｋ鬧・遘ｻ蜍募ｾ後・鬧・
        movedPiece = pos.moved_piece(move);

        // 莉雁屓縺ｮ謖・＠謇九〒邇区焔縺ｫ縺ｪ繧九°縺ｩ縺・°
        givesCheck = pos.gives_check(move);

        // Calculate new depth for this move
        // 莉雁屓縺ｮ謖・＠謇九↓髢｢縺励※譁ｰ縺励＞depth(谿九ｊ謗｢邏｢豺ｱ縺・繧定ｨ育ｮ励☆繧九・
        newDepth = depth - 1;

        int delta = beta - alpha;

        // 笞 reduction()縺ｧ縺ｯ縲‥epth繧・024蛟阪＠縺溷､縺瑚ｿ斐▲縺ｦ縺阪※縺・ｋ縲・
#if defined(P2_SRCH_BUNDLE_A)
        // 束 A (M1): NonPV では delta == 1 なので、 rootDelta での idiv を窓ごと 1 回に寄せた版を使う。
        int r = PvNode ? reduction(improving, depth, moveCount, delta)
                       : reduction_nonpv(improving, depth, moveCount);
#if defined(P2_SRCH_BUNDLE_A_VERIFY)
        if (!PvNode)
        {
            BundleA::check(BundleA::V_M1_DELTA1, delta, 1);
            BundleA::check(BundleA::V_M1_REDUCTION, r, reduction(improving, depth, moveCount, delta));
        }
#endif
#else
        int r = reduction(improving, depth, moveCount, delta);
#endif

        // Increase reduction for ttPv nodes (*Scaler)
        // Larger values scale well

        // ttPv 繝弱・繝峨↓蟇ｾ縺吶ｋ貂帛ｰ鷹㍼繧貞｢励ｄ縺呻ｼ・Scaler・・
		// 螟ｧ縺阪↑蛟､縺ｮ縺ｻ縺・′繧ｹ繧ｱ繝ｼ繝ｫ縺励ｄ縺吶＞縲・

		// 統
        //   遏ｭ縺・戟縺｡譎る俣蛻ｶ髯舌〒縺ｯ縲√ｈ繧雁ｰ上＆縺・､縲√≠繧九＞縺ｯ雋縺ｮ蛟､縺ｮ縺ｻ縺・′譛帙∪縺励＞
        //   髟ｷ縺・戟縺｡譎る俣蛻ｶ髯舌〒縺ｯ縲√ｈ繧雁､ｧ縺阪＞蛟､縺ｮ縺ｻ縺・′譛帙∪縺励＞

        if (ss->ttPv)
            r += SPSA::LMR_TTPV_INCREASE;

        // -----------------------
        // Step 14. Pruning at shallow depths
        // Step 14. (谿九ｊ謗｢邏｢豺ｱ縺輔′)豬・＞豺ｱ縺輔〒縺ｮ譫晏・繧・
        // -----------------------

        // Depth conditions are important for mate finding.
        // (谿九ｊ謗｢邏｢)豺ｱ縺輔・譚｡莉ｶ縺ｯ隧ｰ縺ｿ繧定ｦ九▽縺代ｋ荳翫〒驥崎ｦ√〒縺ゅｋ

        // 投 縲占ｨ域ｸｬ雉・侭 7.縲・豬・＞豺ｱ縺輔〒縺ｮ譫晏・繧翫ｒ陦後↑縺・→縺阪↓邇区焔縺後°縺九▲縺ｦ縺・↑縺・％縺ｨ繧呈擅莉ｶ縺ｫ蜈･繧後ｋ/蜈･繧後↑縺・

        if (!rootNode
            //&& pos.non_pawn_material(us)
            && !is_loss(bestValue))
        {
            // Skip quiet moves if movecount exceeds our threshold
            // movecount縺碁明蛟､繧定ｶ・∴縺溷ｴ蜷医〈uiet縺ｮ謇九ｒ繧ｹ繧ｭ繝・・縺励∪縺・

            if (moveCount >= (3 + depth * depth) / (2 - improving))
                mp.skip_quiet_moves();

            // Reduced depth of the next LMR search
            // 谺｡縺ｮLMR謗｢邏｢縺ｫ縺翫￠繧区ｸ帙ｉ縺輔◆縺ゅ→縺ｮ豺ｱ縺・

            // r縺ｯ1024蛟阪＆繧後※縺・ｋ縺ｮ縺ｧ豕ｨ諢上・
            int lmrDepth = newDepth - r / 1024;

            if (capture || givesCheck)
            {
                Piece capturedPiece = pos.piece_on(move.to_sq());
                int   captHist =
                  captureHistory[movedPiece][move.to_sq()][type_of(capturedPiece)];

                // Futility pruning for captures
                // 鬧偵ｒ蜿悶ｋ謖・＠謇九↓蟇ｾ縺吶ｋfutility譫晏・繧・

                if (!givesCheck && lmrDepth < 7)
                {
                    // ､・Stockfish縺ｮPieceValue[]縺ｯ縲・
                    //     繧・・縺・ｉ邇九〒縺ｯCapturePieceValue[] + CapturePieceValue[capturedPiece]
                    //       = CapturePieceValuePlusPromote()
                    //     縺ｮ縺ｻ縺・′繧医ｊ豁｣遒ｺ縺ｪ隧穂ｾ｡縺ｧ縺ｯ縺ｪ縺・°・・

                    Value futilityValue = ss->staticEval + Eval::p2_root_capfut_base
                                        + SPSA::CAPTURE_FUTILITY_LMRDEPTH_MULT * lmrDepth
                                        + PieceValue[capturedPiece]
                                        + SPSA::CAPTURE_FUTILITY_CAPTHIST_MULT * captHist / 1024;


                    if (futilityValue <= alpha)
                    {
                        P2_PRUNE_MOVE_VERIFY(5)
                        continue;
                    }
                }

                // SEE based pruning for captures and checks
                // Avoid pruning sacrifices of our last piece for stalemate
                // 鬧貞叙繧翫ｄ邇区焔縺ｫ蟇ｾ縺吶ｋSEE・磯撕逧・ｺ､謠幄ｩ穂ｾ｡・峨↓蝓ｺ縺･縺乗椌蛻医ｊ
				// 繧ｹ繝・う繝ｫ繝｡繧､繝茨ｼ亥ｼ輔″蛻・￠・峨ｒ迢吶≧縺溘ａ縺ｫ縲∵怙蠕後・鬧偵ｒ迥迚ｲ縺ｫ縺吶ｋ謇九・譫晏・繧翫・驕ｿ縺代ｋ

                int margin = std::max(SPSA::CAPTURE_SEE_PRUNING_DEPTH_MULT * depth + captHist * SPSA::CAPTURE_SEE_PRUNING_HISTORY_MULT / 1024, 0);
#if STOCKFISH
				if ((alpha >= VALUE_DRAW || pos.non_pawn_material(us) != PieceValue[movedPiece])
#else
				// ､・pos.non_pawn_material(us) != PieceValue[movedPiece]縺ｫ逶ｸ蠖薙☆繧区擅莉ｶ縺ｯ菴輔□繧阪≧・・
				//     豁ｩ莉･螟悶・鬧貞牡縺稽ovedPiece縺ｮ鬧貞牡縺ｨ縺・≧縺ｮ縺ｯ縲・ｧ偵′逶､荳翫↓縺溘￥縺輔ｓ谿九▲縺ｦ縺・ｋ諢溘§縺繧阪≧縺九・

                if ((alpha >= VALUE_DRAW)
#endif
                    && !pos.see_ge(move, -margin))
                {
                    P2_PRUNE_MOVE_VERIFY(6)
                    continue;
                }

            }
            else if (!ss->followPV || !PvNode)
            {
                int history = (*contHist[0])[movedPiece][move.to_sq()]
                            + (*contHist[1])[movedPiece][move.to_sq()]
                            + sharedHistory.pawn_entry(pos)[movedPiece][move.to_sq()];

                // Continuation history based pruning
                // Continuation history縺ｫ蝓ｺ縺･縺・◆譫晏・繧・history縺ｮ蛟､縺梧が縺・ｂ縺ｮ縺ｫ髢｢縺励※縺ｯskip)

                if (history < SPSA::CONTHIST_PRUNING_DEPTH_MULT * depth)
                {
                    P2_PRUNE_MOVE_VERIFY(7)
                    continue;
                }

                history += SPSA::CONTHIST_PRUNING_MAINHIST_MULT * mainHistory[us][move.raw()] / 32;

				// (*Scaler): Generally, lower divisors scales well
				// 荳闊ｬ縺ｫ縲∝牡繧区焚・・ivisor・峨′蟆上＆縺・⊇縺ｩ繧ｹ繧ｱ繝ｼ繝ｫ縺励ｄ縺吶＞縲・

                // ★SF port #7(a) (SF search.cpp:55 lmrDivisor[16] / :1207-1219, commit 95f3680080):
                //   除数を深さ依存の配列 lmrDivisor[clamp(depth,1,16)-1] にする (SF は定数 2995 を置換、
                //   我々の対応定数は 3712 なので既定は配列 × 3712/2995 = SF_LMR_DEPTH_DIV_SCALE 1269)。
                //   配列は setoption 時にスケール済み (SPSA::SF_LMR_DIVISOR) なので毎ノードの乗算なし。
                //   OFF 時は従来の定数除算 (分岐 1 つ追加のみ)。
#if defined(P2_SRCH_BUNDLE_A)
                lmrDepth += SPSA::SF_LMR_DEPTH_DIV
                          ? history / SPSA::SF_LMR_DIVISOR[std::clamp(int(depth), 1, 16) - 1]
                          : BundleA::vdiv(bundleDivs.conthistLmrDepth, history,
                                          SPSA::CONTHIST_PRUNING_LMRDEPTH_DIV, BundleA::V_L2_LMRDEPTH);
#else
                lmrDepth += history
                          / (SPSA::SF_LMR_DEPTH_DIV
                               ? SPSA::SF_LMR_DIVISOR[std::clamp(int(depth), 1, 16) - 1]
                               : SPSA::CONTHIST_PRUNING_LMRDEPTH_DIV);
#endif

                Value futilityValue = ss->staticEval + 42 + 151 * !bestMove
                                    + SPSA::FUTILITY_PARENT_LMRDEPTH_MULT * lmrDepth
                                    + SPSA::FUTILITY_PARENT_ALPHA_BONUS * (ss->staticEval > alpha);

                // Futility pruning: parent node
                // (*Scaler): Generally, more frequent futility pruning
				// scales well

                // Futility譫晏・繧・ 隕ｪ繝弱・繝・
                // (*Scaler): 荳闊ｬ逧・↓縲√ｈ繧企ｻ郢√↑辟｡鬧・椌蛻医ｊ縺ｯ繧医￥繧ｹ繧ｱ繝ｼ繝ｫ縺吶ｋ

                // 統 隕ｪnode縺ｮ譎らせ縺ｧ蟄刃ode繧貞ｱ暮幕縺吶ｋ蜑阪↓futility縺ｮ蟇ｾ雎｡縺ｨ縺ｪ繧翫◎縺・↑繧画椌蛻医ｊ縺励※縺励∪縺・・
                // ､・繝代Λ繝｡繝ｼ繧ｿ繝ｼ隱ｿ謨ｴ縺ｮ菫よ焚繧定ｪｿ謨ｴ縺励◆縺ｻ縺・′濶ｯ縺・・縺九ｂ遏･繧後↑縺・′縲・
                // 縲  縺薙％縲√◎繧薙↑縺ｫ螟ｧ縺阪↑Elo繧呈戟縺｣縺ｦ縺・↑縺・・縺ｧ縲∬ｪｿ謨ｴ縺励※繧やｦ縲・

                if (!ss->inCheck && lmrDepth < 13 && futilityValue <= alpha)
                {
                    if (bestValue <= futilityValue && !is_decisive(bestValue) && !is_win(futilityValue))
                        bestValue = futilityValue;
                    P2_PRUNE_MOVE_VERIFY(8)
                    continue;
                }

                /*
				   笞 莉･荳九・LMR縺ｾ繧上ｊ縲∵｣句鴨縺ｫ讌ｵ繧√※驥榊､ｧ縺ｪ蠖ｱ髻ｿ縺後≠繧九・縺ｧ譫晏・繧翫ｒ蜈･繧後ｋ縺九←縺・°繧貞性繧√※諷朱㍾縺ｫ隱ｿ謨ｴ縺吶∋縺阪・
				       蟆・｣九〒縺ｯsee縺瑚ｲ縺ｮ謖・＠謇九ｂ縺昴・縺ゅ→隧ｰ繧繧医≧縺ｪ蝣ｴ蜷医′縺ゅｋ縺九ｉ縲√≠縺ｾ繧顔┌遒阪↓繧ょ・譚･縺ｪ縺・ｈ縺・□縺娯ｦ縲・
				*/

                // 投縲占ｨ域ｸｬ雉・侭 20.縲全EE縺瑚ｲ縺ｮ謖・＠謇九ｒ譫晏・繧翫☆繧・縺励↑縺・

                lmrDepth = std::max(lmrDepth, 0);

                // Prune moves with negative SEE
                // 雋縺ｮSEE繧呈戟縺､謖・＠謇九ｒ譫晏・繧翫☆繧・
                // 庁 lmrDepth縺ｮ2荵励↓豈比ｾ九☆繧九・縺ｧ縺薙・繝代Λ繝｡繝ｼ繧ｿ繝ｼ縺ｮ蠖ｱ髻ｿ縺ｯ縺吶＃縺丞､ｧ縺阪＞縲・

                if (!pos.see_ge(move, SPSA::SEE_PRUNING_QUIET_MULT * lmrDepth * lmrDepth))
                {
                    P2_PRUNE_MOVE_VERIFY(9)
                    continue;
                }
            }
        }

		// -----------------------
        // Step 15. Extensions
        // Step 15. (謗｢邏｢縺ｮ)蟒ｶ髟ｷ
        // -----------------------

        // Singular extension search. If all moves but one
        // fail low on a search of (alpha-s, beta-s), and just one fails high on
        // (alpha, beta), then that move is singular and should be extended. To
        // verify this we do a reduced search on the position excluding the ttMove
        // and if the result is lower than ttValue minus a margin, then we will
        // extend the ttMove. Recursive singular search is avoided.

		// 繧ｷ繝ｳ繧ｮ繝･繝ｩ繝ｼ蟒ｶ髟ｷ謗｢邏｢縲ゅｂ縺・(alpha-s, beta-s) 縺ｮ謗｢邏｢縺ｧ 1 謇九ｒ髯､縺上☆縺ｹ縺ｦ縺ｮ謇九′ fail low 縺ｨ縺ｪ繧翫・
        // (alpha, beta) 縺ｮ謗｢邏｢縺ｧ縺溘□ 1 謇九□縺代′ fail high 縺ｨ縺ｪ縺｣縺溷ｴ蜷医・
        // 縺昴・謇九・繧ｷ繝ｳ繧ｮ繝･繝ｩ繝ｼ・育音逡ｰ・峨→蛻､譁ｭ縺励∝ｻｶ髟ｷ縺吶∋縺阪〒縺吶・
        // 縺薙ｌ繧呈､懆ｨｼ縺吶ｋ縺溘ａ縲》tMove 繧帝勁螟悶＠縺溷ｱ髱｢縺ｧ邵ｮ蟆乗爾邏｢繧定｡後＞縲・
        // 縺昴・邨先棡縺・ttValue 縺九ｉ繝槭・繧ｸ繝ｳ繧貞ｼ輔＞縺溷､繧医ｊ菴弱￠繧後・縲》tMove 繧貞ｻｶ髟ｷ縺励∪縺吶・
        // 蜀榊ｸｰ逧・↑繧ｷ繝ｳ繧ｮ繝･繝ｩ繝ｼ謗｢邏｢縺ｯ蝗樣∩縺輔ｌ縺ｾ縺吶・

        // (*Scaler) Generally, higher singularBeta (i.e closer to ttValue)
        // and lower extension margins scale well.

        // ・・Scaler・我ｸ闊ｬ逧・↓縲√ｈ繧企ｫ倥＞ singularBeta・医☆縺ｪ繧上■ ttValue 縺ｫ霑代＞蛟､・峨→縲・
        // 繧医ｊ菴弱＞蟒ｶ髟ｷ繝槭・繧ｸ繝ｳ縺ｯ繧ｹ繧ｱ繝ｼ繝ｪ繝ｳ繧ｰ縺ｫ驕ｩ縺励※縺・∪縺吶・

		/*
			塔 Stockfish縺ｮ螳溯｣・□縺ｨmargin = 2 * depth縺縺後・
				蟆・｣九□縺ｨ1謇倶ｻ･螟悶・縺吶∋縺ｦ縺昴ｌ縺舌ｉ縺・が縺・％縺ｨ縺ｯ螟壹・≠繧翫・
				縺ｻ縺ｨ繧薙←縺ｮ謖・＠謇九′singular縺ｨ蛻､螳壹＆繧後※縺励∪縺・・
				縺薙ｌ縺ｧ縺ｯ蜉ｹ譫懊′縺ｪ縺・・縺ｧ縲・蜑ｲ縺舌ｉ縺・・謖・＠謇九′singular縺ｨ縺ｪ繧九＄繧峨＞縺ｮ菫よ焚縺ｫ隱ｿ謨ｴ縺吶ｋ縲・
		
			庁 singular蟒ｶ髟ｷ縺ｧ蠑ｷ縺上↑繧九・縺ｯ縲√≠繧杵ode縺ｧ1謇九□縺代′迚ｹ蛻･縺ｫ濶ｯ縺・ｴ蜷医・
				逶ｸ謇九・繝励Ξ繧､繝､繝ｼ繧ゅ◎縺ｮnode縺ｧ縺ｯ縺昴・謖・＠謇九ｒ驕ｸ謚槭☆繧句庄閭ｽ諤ｧ縺碁ｫ倥￥縲・
				縺昴ｌ繧・∴縲∫嶌謇九・PV繧ゅ◎縺薙〒縺ゅｋ蜿ｯ閭ｽ諤ｧ縺碁ｫ倥＞縺九ｉ縲√◎縺薙ｒ逶ｸ謇九ｈ繧翫ｏ縺壹°縺ｫ縺ｧ繧・
				隱ｭ繧薙〒縺・※隧ｰ縺ｿ繧貞屓驕ｿ縺ｪ縺ｩ縺ｧ縺阪ｋ縺ｪ繧峨√◎縺ｮ逶ｸ謇九↓蟇ｾ縺吶ｋ蜍晉紫縺ｯ荳翫′繧九→縺・≧逅・ｱ医・

				縺・ｏ縺ｰ縲・.5謇句ｻｶ髟ｷ縺瑚・蟾ｱ蟇ｾ謌ｦ縺ｧ(縺ｮ縺ｿ)蠑ｷ縺上↑繧九・縺ｮ諡｡蠑ｵ縲・
				縺昴≧閠・∴繧九→繝吶せ繝医↑謖・＠謇九・繧ｹ繧ｳ繧｢縺ｨ2逡ｪ逶ｮ縺ｫ繝吶せ繝医↑謖・＠謇九・繧ｹ繧ｳ繧｢縺ｨ縺ｮ蟾ｮ縺ｫ
				蠢懊§縺ｦ1謇句ｻｶ髟ｷ縺吶ｋ縺ｮ縺梧ｭ｣縺励＞縺ｮ縺縺後・逡ｪ逶ｮ縺ｫ繝吶せ繝医↑謖・＠謇九・繧ｹ繧ｳ繧｢繧・
				蟆上＆縺ｪ繧ｳ繧ｹ繝医〒豎ゅａ繧九％縺ｨ縺ｯ蜃ｺ譚･縺ｪ縺・・縺ｧ窶ｦ縲・

			統 if縺ｮ譚｡莉ｶ蠑上↓ !excludedMove 縺後≠繧九・縺ｯ縲∝・蟶ｰ逧・↑singular蟒ｶ髟ｷ繧帝勁螟悶☆繧九◆繧√・
		*/

		// singular蟒ｶ髟ｷ繧偵☆繧杵ode縺ｧ縺ゅｋ縺九・
		if (P2Stat::enabled() && !rootNode && move == ttData.move && !excludedMove) {
			P2Stat::se_g1.fetch_add(1, std::memory_order_relaxed);
			if (depth >= 6 + ss->ttPv) {
				P2Stat::se_g2.fetch_add(1, std::memory_order_relaxed);
				if (is_valid(ttData.value) && !is_decisive(ttData.value)) {
					P2Stat::se_g3.fetch_add(1, std::memory_order_relaxed);
					if (ttData.bound & BOUND_LOWER) {
						P2Stat::se_g4.fetch_add(1, std::memory_order_relaxed);
						if (ttData.depth >= depth - 3) {
							P2Stat::se_g5.fetch_add(1, std::memory_order_relaxed);
							if (!is_shuffling(move, ss, pos))
								P2Stat::se_g6.fetch_add(1, std::memory_order_relaxed);
						}
					}
				}
			}
		}
        if (!rootNode && move == ttData.move && !excludedMove && depth >= 6 + ss->ttPv
            && is_valid(ttData.value) && !is_decisive(ttData.value) && (ttData.bound & BOUND_LOWER)
            && ttData.depth >= depth - 3 && !is_shuffling(move, ss, pos))
        {
            /*
				庁 縺薙・node縺ｫ縺､縺・※縺ゅｋ遞句ｺｦ隱ｿ縺ｹ縺溘％縺ｨ縺檎ｽｮ謠幄｡ｨ縺ｫ繧医▲縺ｦ險ｼ譏弱＆繧後※縺・ｋ縲・ttMove == move縺ｪ縺ｮ縺ｧttMove != Move::none())
				    (縺昴≧縺ｧ縺ｪ縺・→singular縺ｮ謖・＠謇倶ｻ･螟悶↓莉悶・譛画悍縺ｪ謖・＠謇九′縺ｪ縺・°縺ｩ縺・°繧定ｪｿ縺ｹ繧九◆繧√↓
					null window search縺吶ｋ縺ｨ縺阪↓螟ｧ縺阪↑繧ｳ繧ｹ繝医ｒ莨ｴ縺・°縺ｭ縺ｪ縺・°繧峨・
			*/

            //  桃 縺薙・margin蛟､縺ｯ隧穂ｾ｡髢｢謨ｰ縺ｮ諤ｧ雉ｪ縺ｫ蜷医ｏ縺帙※隱ｿ謨ｴ縺輔ｌ繧九∋縺阪・

            Value singularBeta  = ttData.value
                                - (SPSA::SE_SINGULAR_BETA_BASE
                                   + SPSA::SE_SINGULAR_BETA_TTPV * (ss->ttPv && !PvNode))
                                    * depth / SPSA::SE_SINGULAR_BETA_DIV;
            Depth singularDepth = newDepth / 2;

            // 庁 move(ttMove)縺ｮ謖・＠謇九ｒ莉･荳九・search()縺ｧ縺ｮ謗｢邏｢縺九ｉ髯､螟悶・

            ss->excludedMove = move;

            // 統 螻髱｢縺ｯdo_move()縺ｧ騾ｲ繧√★縺ｫ縺薙・node縺九ｉ豬・＞謗｢邏｢豺ｱ縺輔〒謗｢邏｢縺励↑縺翫☆縲・
            //     豬・＞depth縺ｧnull window縺ｪ縺ｮ縺ｧ縲√☆縺舌↓謗｢邏｢縺ｯ邨ゅｏ繧九・縺壹・

#if defined(P2_SRCH_BUNDLE_A)
            value = search_nonpv(pos, ss, singularBeta - 1, singularBeta, singularDepth, cutNode);
#else
            value = search<NonPV>(pos, ss, singularBeta - 1, singularBeta, singularDepth, cutNode);
#endif
            ss->excludedMove = Move::none();

            // 庁 鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇倶ｻ･螟悶′縺吶∋縺ｦfail low縺励※縺・ｋ縺ｪ繧鋭ingular蟒ｶ髟ｷ遒ｺ螳壹・
            //    (蟒ｶ髟ｷ縺輔ｌ邯壹￠繧九→縺ｾ縺壹＞縺ｮ縺ｧ菴輔ｉ縺九・閠・・縺ｯ蠢・ｦ・

            if (P2Stat::enabled()) P2Stat::se_try.fetch_add(1, std::memory_order_relaxed);
            if (value < singularBeta)
            {
				int corrValAdj   = std::abs(correctionValue) / SPSA::SE_CORRVAL_DIV;
                int doubleMargin = SPSA::SE_DOUBLE_MARGIN_BASE
                                 + SPSA::SE_DOUBLE_MARGIN_PVNODE * PvNode
                                 - SPSA::SE_DOUBLE_MARGIN_NOTTCAPTURE * !ttCapture - corrValAdj
                                 - SPSA::SE_DOUBLE_MARGIN_TTMOVEHIST * ttMoveHistory / 116517
                                 - (ss->ply > rootDepth) * SPSA::SE_DOUBLE_MARGIN_PLY;
                int tripleMargin = SPSA::SE_TRIPLE_MARGIN_BASE
                                 + SPSA::SE_TRIPLE_MARGIN_PVNODE * PvNode
                                 - SPSA::SE_TRIPLE_MARGIN_NOTTCAPTURE * !ttCapture
                                 + SPSA::SE_TRIPLE_MARGIN_TTPV * ss->ttPv - corrValAdj
                                 - (ss->ply > rootDepth) * SPSA::SE_TRIPLE_MARGIN_PLY;

                // 統 2驥榊ｻｶ髟ｷ繧貞宛髯舌＠縺ｦ謗｢邏｢縺ｮ邨・粋縺帷・逋ｺ繧貞屓驕ｿ縺吶ｋ蠢・ｦ√′縺ゅｋ縲・

                extension =
                    1 + (value < singularBeta - doubleMargin) + (value < singularBeta - tripleMargin);

                depth++;
            }

            // Multi-cut pruning
            // Our ttMove is assumed to fail high based on the bound of the TT entry,
            // and if after excluding the ttMove with a reduced search we fail high
            // over the original beta, we assume this expected cut-node is not
            // singular (multiple moves fail high), and we can prune the whole
            // subtree by returning a softbound.

            // 繝槭Ν繝√き繝・ヨ譫晏・繧・
            // 遘√◆縺｡縺ｮttMove縺ｯfail high縺吶ｋ縺ｨ諠ｳ螳壹＆繧後※縺翫ｊ縲・
            // 莉翫》tMove縺ｪ縺励・(縺薙・螻髱｢縺ｧttMove縺ｮ謖・＠謇九ｒ蛟呵｣懈焔縺九ｉ髯､螟悶＠縺・縲・
            // reduced search(謗｢邏｢豺ｱ縺輔ｒ貂帙ｉ縺励◆謗｢邏｢)縺ｧ繧Ｇail high縺励∪縺励◆縲・
            // 縺励◆縺後▲縺ｦ縲√％縺ｮ莠域Φ縺輔ｌ繧九き繝・ヨ繝弱・繝峨・singular(1縺､縺縺大ｑ蜃ｺ縺励◆謖・＠謇・縺ｧ縺ｯ縺ｪ縺・→諠ｳ螳壹＠縲・
            // 隍・焚縺ｮ謇九′fail high縺吶ｋ縺ｨ閠・∴縲《oftbound繧定ｿ斐☆縺薙→縺ｧ蜈ｨ繧ｵ繝悶ヤ繝ｪ繝ｼ繧呈椌蛻医ｊ縺吶ｋ縺薙→縺後〒縺阪∪縺吶・

            /*
				涛 險ｳ豕ｨ
            
				 cut-node  : ﾎｱﾎｲ謗｢邏｢縺ｫ縺翫＞縺ｦ譌ｩ譛溘↓譫晏・繧翫〒縺阪ｋnode縺ｮ縺薙→縲・
							 縺､縺ｾ繧翫《earch縺ｮ蠑墓焚縺ｧ貂｡縺輔ｌ縺歟eta繧剃ｸ雁屓繧九％縺ｨ縺後ｏ縺九▲縺溘・縺ｧreturn縺ｧ縺阪ｋ(縺薙ｌ繧鍛eta cut縺ｨ蜻ｼ縺ｶ)
							 縺ｧ縺阪ｋ繧医≧縺ｪnode縺ｮ縺薙→縲・
            
				 softbound : lowerbound(荳狗阜)繧еpperbound(荳顔阜)縺ｮ繧医≧縺ｫ逵溘・蛟､縺後◎縺ｮ蛟､繧医ｊ螟ｧ縺阪＞(蟆上＆縺・
							 縺薙→縺後ｏ縺九▲縺ｦ縺・ｋ繧医≧縺ｪ蛟､縺ｮ縺薙→縲・

			*/

            else if (value >= beta && !is_decisive(value))
            {
                ttMoveHistory << std::max(-424 - 107 * depth, -3375);

                // ★SF port #6 (SF search.cpp:1285-1296, commit 218c74ec4d):
                //   multicut で return する前に、「ttMove 抜きの縮小探索でも beta を超えた」= 静的評価が
                //   低すぎたという情報で correction history を更新する。
                //   bonus = clamp((value - staticEval) * singularDepth * MULT/1024, ±CORRECTION_HISTORY_LIMIT/4)。
                //   更新関数は通常の update_correction_history (pawn/minor/nonpawn/continuation + hand) をそのまま使う
                //   (SF と同種の更新。通常更新のように CORR_HIST_BONUS_SCALE は掛けない = SF 原文どおり)。
                //   multicut は稀な経路なので NPS への影響なし。既定 OFF。
                if (SPSA::SF_MULTICUT_CORR && !ss->inCheck && value > ss->staticEval)
                {
                    const int bonus = std::clamp(int(value - ss->staticEval) * singularDepth
                                                   * SPSA::SF_MULTICUT_CORR_MULT / 1024,
                                                 -CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
                    update_correction_history(pos, ss, *this, bonus);
                }

                return value;
            }

            // Negative extensions
            // If other moves failed high over (ttValue - margin) without the
            // ttMove on a reduced search, but we cannot do multi-cut because
            // (ttValue - margin) is lower than the original beta, we do not know
            // if the ttMove is singular or can do a multi-cut, so we reduce the
            // ttMove in favor of other moves based on some conditions:

            // 雋縺ｮ蟒ｶ髟ｷ
            // 繧ゅ＠ttMove繧剃ｽｿ逕ｨ縺帙★縺ｫ(ttValue - margin)莉･荳翫〒莉悶・謇九′reduced search
            // (邁｡逡･蛹悶＠縺滓爾邏｢)縺ｧ鬮倥＞繧ｹ繧ｳ繧｢繧貞・縺励◆縺後・ttValue - margin)縺悟・縺ｮbeta繧医ｊ繧・
            // 菴弱＞縺溘ａ縺ｫ繝槭Ν繝√き繝・ヨ繧定｡後∴縺ｪ縺・ｴ蜷医・
            // ttMove縺茎ingular縺九・繝ｫ繝√き繝・ヨ縺悟庄閭ｽ縺九・繧上°繧峨↑縺・・縺ｧ縲・
            // 縺・￥縺､縺九・譚｡莉ｶ縺ｫ蝓ｺ縺･縺・※莉悶・謇九ｒ蜆ｪ蜈医＠縺ｦttMove繧呈ｸ帙ｉ縺励∪縺呻ｼ・


            // If the ttMove is assumed to fail high over current beta
            // ttMove 縺檎樟蝨ｨ縺ｮ beta 繧定ｶ・∴縺ｦ fail high 縺吶ｋ縺ｨ諠ｳ螳壹＆繧後ｋ蝣ｴ蜷・

            else if (ttData.value >= beta)
                extension = -3;

            // If we are on a cutNode but the ttMove is not assumed to fail high
            // over current beta
            // 迴ｾ蝨ｨ縺ｮ繝弱・繝峨′繧ｫ繝・ヨ繝弱・繝峨〒縺ゅｋ縺後》tMove縺檎樟蝨ｨ縺ｮbeta繧定ｶ・∴縺ｦ
            // fail high縺吶ｋ縺ｨ諤昴ｏ繧後↑縺・ｴ蜷・

            else if (cutNode)
                extension = -2;

            /*
			  笞  邇区焔蟒ｶ髟ｷ縺ｫ髢｢縺励※縲ヾtockfish縺ｮ繧ｳ繝ｼ繝峨√％縺薙↓謖√▲縺ｦ縺上ｋ譎ゅ↓縺ｯ豌励ｒ縺､縺代ｋ縺薙→・・
			       蟆・｣九〒縺ｯ邇区焔縺ｯ繧上ｊ縺ｨ邯壹￥縺ｮ縺ｧ縺昴・縺ｾ縺ｾ謖√▲縺ｦ縺上ｋ縺ｨ繧・ｊ縺吶℃縺ｮ蜿ｯ閭ｽ諤ｧ縺碁ｫ倥＞縲・

			  涛 Stockfish縺ｧ蜑企勁縺輔ｌ縺溘′縲∫視謇句ｻｶ髟ｷ閾ｪ菴薙・菴輔ｉ縺九≠縺｣縺滓婿縺瑚憶縺・庄閭ｽ諤ｧ縺ｯ縺ゅｋ縺ｮ縺ｧ譚｡莉ｶ繧定ｪｿ謨ｴ縺励※縺ｯ縺ｩ縺・°縲・
					Remove check extension : https://github.com/official-stockfish/Stockfish/commit/96837bc4396d205536cdaabfc17e4885a48b0588
			*/
        }

		// -----------------------
        // Step 16. Make the move
        // Step 16. 謖・＠謇九〒騾ｲ繧√ｋ
        // -----------------------

		// 髯仙ｮ・check extension (SPSA::CHECK_EXT=1 縺ｧ譛牙柑, 譌｢螳・=baseline荳榊､・縲・
		// SEE螳牙・縺ｪ邇区焔繧・1蟒ｶ髟ｷ縺玲姶陦途efute縺ｮ豌ｴ蟷ｳ邱壹ｒ蟒ｶ縺ｰ縺吶ょｰ・｣九・邇区焔縺檎ｶ壹￥縺溘ａ:
		// singular蟒ｶ髟ｷ縺ｨ髱枹tack(extension==0) + 邨瑚ｷｯply荳企剞 + SEE>=0 縺ｧ辷・匱繧呈椛蛻ｶ縲・
		// (see_ge/givesCheck 縺ｯ do_move 蜑・迴ｾ螻髱｢縺ｧ隧穂ｾ｡)
		if (SPSA::CHECK_EXT && extension == 0 && givesCheck && !rootNode
		    && ss->ply < SPSA::CHECK_EXT_PLY_LIMIT_MULT * rootDepth
		    && pos.see_ge(move, Value(0)))
			extension = 1;

		// 謖・＠謇九〒1謇矩ｲ繧√ｋ
        // -----------------------
        // M4 木の内部での仕事の委譲   2026-11-02
        // -----------------------
        //   ここは **newDepth と窓が確定している唯一の場所** (do_move の直前)。
        //   提示できたらこの手を後回しにして次の手へ進む。 末尾の 2 周目で master が
        //   自分で読むが、 helper が置換表を温めていればほぼ無料で済む。
        //   🚨提示する深さは `newDepth + extension` (この加算は do_move の後にあるので
        //     ここで自分で足す)。 LMR の削減は後段なので helper は master の最初の読みより
        //     **深く**読む。 深い entry は浅い probe を満たすので当たりはする。
        //   🚨その代わり深く読みすぎて無駄になりうるので **序盤の手だけ**に限る
        //     (moveCount が大きい手は削減が大きく、 全深さで読むと割に合わない)。
        //   🚨正しさは helper に依存しない (master は必ず全手を自分で読む)。
        if (delegOK && moveCount > 1 && moveCount <= 8 && nDeferred < 8
            && move != Move::win() && curPathLen == ss->ply)
        {
            // 🚨手順は curPath から取る (ss - ply は配列の外を読む)。
            //   curPathLen == ss->ply が成り立たないとき (qsearch を跨いだ等) は提示しない。
            //   null move を含む手順は再生できないので弾く。
            const int p  = ss->ply;
            bool      ok = p <= DelegSlot::PATH_CAP;
            for (int j = 0; ok && j < p; ++j)
            {
                delegPath[j] = curPath[j];
                if (delegPath[j] == Move::none())
                    ok = false;
            }
            if (ok)
            {
                const int sl = deleg->offer(delegPath, p, move, newDepth + extension, alpha,
                                            beta, cutNode, PvNode, rootDelta);
                if (sl >= 0)
                {
                    deferred[nDeferred++] = {move, moveCount, sl};
                    continue;
                }
            }
        }

        do_move(pos, move, st, givesCheck, ss);

        // Add extension to new depth
        // 豎ゅ∪縺｣縺溷ｻｶ髟ｷ縺吶ｋ謇区焚繧呈眠縺励＞depth縺ｫ蜉邂・

		if (P2Stat::enabled()) {
			if (extension == 1) P2Stat::ext_p1.fetch_add(1, std::memory_order_relaxed);
			else if (extension == 2) P2Stat::ext_p2.fetch_add(1, std::memory_order_relaxed);
			else if (extension >= 3) P2Stat::ext_p3.fetch_add(1, std::memory_order_relaxed);
			else if (extension == -2) P2Stat::ext_m2.fetch_add(1, std::memory_order_relaxed);
			else if (extension <= -3) P2Stat::ext_m3.fetch_add(1, std::memory_order_relaxed);
		}
		newDepth += extension;

		uint64_t nodeCount = rootNode ? uint64_t(nodes) : 0;

        // Decrease reduction for PvNodes (*Scaler)
        // Pv Nodes縺ｫ蟇ｾ縺励※reduction繧呈ｸ帙ｉ縺・*Scaler)

		if (ss->ttPv)
            r -= SPSA::LMR_TTPV_BASE + PvNode * SPSA::LMR_TTPV_PVNODE
               + (ttData.value > alpha) * SPSA::LMR_TTPV_TTVALUE_GT_ALPHA
               + (ttData.depth >= depth)
                   * (SPSA::LMR_TTPV_TTDEPTH + cutNode * SPSA::LMR_TTPV_TTDEPTH_CUTNODE);

        // These reduction adjustments have no proven non-linear scaling
        // 縺薙ｌ繧峨・貂帛ｰ鷹㍼隱ｿ謨ｴ縺ｫ縺ｯ縲・撼邱壼ｽ｢繧ｹ繧ｱ繝ｼ繝ｪ繝ｳ繧ｰ縺ｮ譛牙柑諤ｧ縺瑚ｨｼ譏弱＆繧後※縺・∪縺帙ｓ

        r += SPSA::LMR_BASE_OFFSET;  // Base reduction offset (rust tune_params.rs:1494 = 843, C++譌ｧ=691)
                   // 莉悶・隱ｿ謨ｴ繧定｣懈ｭ｣縺吶ｋ縺溘ａ縺ｮ蝓ｺ貅悶Μ繝繧ｯ繧ｷ繝ｧ繝ｳ繧ｪ繝輔そ繝・ヨ

        r -= moveCount * SPSA::LMR_MOVECOUNT_MULT;
#if defined(P2_SRCH_BUNDLE_A)
        r -= BundleA::vdiv(bundleDivs.lmrCorr, std::abs(correctionValue), SPSA::LMR_CORRECTION_DIV,
                           BundleA::V_L2_LMRCORR);
#else
        r -= std::abs(correctionValue) / SPSA::LMR_CORRECTION_DIV;
#endif

        // Increase reduction for cut nodes
        // 繧ｫ繝・ヨ繝弱・繝峨・reduction繧貞｢励ｄ縺・

		/*
			庁 cut node縺ｫ縺翫＞縺ｦhistory縺ｮ蛟､縺梧が縺・欠縺玲焔縺ｫ蟇ｾ縺励※縺ｯreduction驥上ｒ蠅励ｄ縺吶・
			    PVnode縺ｧ縺ｯIID譎ゅ〒繧ＤutNode == true縺ｧ縺ｯ蜻ｼ縺ｰ縺ｪ縺・％縺ｨ縺ｫ縺励◆縺ｮ縺ｧ縲・
			        if (cutNode)縺ｨ縺・≧譚｡莉ｶ蠑上・證鈴ｻ吶↓ && !PvNode 繧貞性繧縲・

			投 縲占ｨ域ｸｬ雉・侭 18.縲祖ut node縺ｮ縺ｨ縺阪↓reduction繧貞｢励ｄ縺吶°縺ｩ縺・°縲・
		*/

        if (cutNode)
            r += SPSA::LMR_CUTNODE_ADD + SPSA::LMR_CUTNODE_NO_TTMOVE_ADD * !ttData.move;

        // Increase reduction if ttMove is a capture
        // ttMove 縺梧黒迯ｲ縺吶ｋ謖・＠謇九↑繧峨〉eduction繧貞｢励ｄ縺・

        if (ttCapture)
            r += SPSA::LMR_TTCAPTURE_ADD;

        // Increase reduction if next ply has a lot of fail high
        // 谺｡縺ｮ謇九〒fail high縺悟､壹＞蝣ｴ蜷医〉eduction繧貞｢励ｄ縺・

        if ((ss + 1)->cutoffCnt > 1)
        {
            r += SPSA::LMR_CUTOFFCNT_BASE + SPSA::LMR_CUTOFFCNT_GT2 * ((ss + 1)->cutoffCnt > 2)
               + SPSA::LMR_CUTOFFCNT_ALLNODE * allNode;

            // ★SF port #7(b) (SF search.cpp:1343-1346, commit 2be0b2cdb4 → 924d29d3ca):
            //   SF は ttMove の減算を cutoffCnt>1 の else-if にしている (両方は効かない)。
            //   OFF (既定) では従来どおり cutoffCnt>1 のときも ttMove の減算を併用 (bit-exact)。
            if (!SPSA::SF_LMR_TTMOVE_ELSEIF && move == ttData.move)
                r -= SPSA::LMR_TTMOVE_REDUCTION;
        }

        // For first picked move (ttMove) reduce reduction
        // 譛蛻昴↓驕ｸ縺ｰ繧後◆謖・＠謇具ｼ・tMove・峨〒縺ｯreduction繧呈ｸ帙ｉ縺・

        else if (move == ttData.move)  // ★SF port #7(b): 上の cutoffCnt>1 ブロックの else (OFF 時は上のブロック内でも減算するので従来と同値)
            r -= SPSA::LMR_TTMOVE_REDUCTION;

        if (capture)
        {
            ss->statScore = SPSA::LMR_CAPTURE_STATSCORE_PIECEVALUE_MULT
                              * int(PieceValue[pos.captured_piece()]) / 128
                          + captureHistory[movedPiece][move.to_sq()][type_of(pos.captured_piece())];
            // Decrease/increase reduction for moves with a good/bad history
            r -= ss->statScore * SPSA::LMR_STATSCORE_MULT / 4096;
        }
        else
        {
            // 投縲占ｨ域ｸｬ雉・侭 11.縲壮tatScore縺ｮ險育ｮ励〒contHist[3]繧りｪｿ縺ｹ繧九°縺ｩ縺・°縲・
            // 笘・PS譛驕ｩ蛹・ History Confidence LMR 逕ｨ縺ｮ 3 謌仙・繧剃ｸ蠎ｦ縺縺題ｪｭ縺ｿ縲《tatScore 縺ｨ蜈ｱ譛・蜀崎ｪｭ霎ｼ縺ｿ謗帝勁)縲・
            const int mscA = 2 * mainHistory[us][move.raw()];
            const int mscB = (*contHist[0])[movedPiece][move.to_sq()];
            const int mscC = (*contHist[1])[movedPiece][move.to_sq()];
            ss->statScore  = mscA + mscB + mscC;

            r -= ss->statScore * SPSA::LMR_STATSCORE_MULT / 4096;

            // History Confidence LMR (HIST_CONFIDENCE_SCALE=0 縺ｮ譌｢螳壹↑繧牙ｮ悟・縺ｫ辟｡蜉ｹ=bit-exact)縲・
            // 3 謌仙・縺悟酔隨ｦ蜿ｷ=縲御ｸ雋ｫ縺励◆螻･豁ｴ=鬮倅ｿ｡鬆ｼ縲阪↑縺ｨ縺阪□縺・reduction 隱ｿ謨ｴ繧剃ｸ谿ｵ蠑ｷ繧√ｋ縲・
            // NOTE 2026-10-17: the default is 150 (spsa_params.h:251), NOT 0.
            // The stale comment above hid this from the 09-28 audit. NAGISA/YO
            // have no such term; this adds +32% on top of LMR_STATSCORE_MULT=462.
            if (SPSA::HIST_CONFIDENCE_SCALE && (mscA > 0) == (mscB > 0) && (mscB > 0) == (mscC > 0))
                r -= ss->statScore * SPSA::HIST_CONFIDENCE_SCALE / 4096;
        }

        // Uncertainty-linked LMR (隨ｬ1繝代ャ繝・ 髮｢謨｣繝ｻ譛螟ｧ1ply)縲・
        // 鬮倅ｸ咲｢ｺ螳滓ｧ(隧穂ｾ｡諤･螟・螻髱｢縺ｮ荳驛ｨ quiet 縺縺・LMR 繧貞宍蟇・↓1ply蠑ｱ繧√ｋ縲・
        // UNCERT_LMR_THRESHOLD=0 縺ｧ螳悟・OFF(bit-exact)縲よｯ比ｾ句宛蠕｡縺ｧ縺ｪ縺城明蛟､蛻ｶ蠕｡縲・
        // History Confidence 縺ｨ縺ｯ蛻･陬懈ｭ｣縺ｧ縲∽ｻ雁屓縺ｮA/B縺ｧ縺ｯ HIST_CONFIDENCE_SCALE=0(蜊倡峡讀懆ｨｼ)縲・
#ifdef UNCERT_LMR_MEASURE
        bool uncertReduced = false;
        {
            // 險域ｸｬ縺ｯ髢ｾ蛟､縺ｫ萓晏ｭ倥○縺壹´MR驕ｩ逕ｨ蜿ｯ閭ｽ(髢ｾ蛟､莉･螟悶・蜈ｨ譚｡莉ｶ謌千ｫ・quiet 縺ｮ volatility 蛻・ｸ・ｒ隕ｳ貂ｬ縺吶ｋ縲・
            // r 縺ｯ貂帷ｮ怜燕(蜈・・reduction)縺ｧ蛻､螳壹☆繧九・
            const bool uncertCandidate =
                   validEvalVolatility && !PvNode && !capture && move.type_of() != PROMOTION
                && move != ttData.move && !givesCheck && depth >= 5 && moveCount > 3
                && r >= 2 * 1024 && ss->excludedMove == Move::none();
            if (uncertCandidate)
            {
                g_uncertMeasure.candidate(nodeEvalVolatility);
                if (SPSA::UNCERT_LMR_THRESHOLD > 0
                    && nodeEvalVolatility >= SPSA::UNCERT_LMR_THRESHOLD)
                {
                    r -= 1024;
                    uncertReduced = true;
                    g_uncertMeasure.fired.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
#else
        // 髱櫁ｨ域ｸｬ繝薙Ν繝・ UNCERT_LMR_THRESHOLD=0 縺ｪ繧臥ｬｬ1譚｡莉ｶ縺ｧ遏ｭ邨｡縺励√・繝・ヨ繝代せ縺ｫ霑ｽ蜉蜃ｦ逅・ｒ谿九＆縺ｪ縺・・
        if (SPSA::UNCERT_LMR_THRESHOLD > 0 && validEvalVolatility && !PvNode && !capture
            && move.type_of() != PROMOTION && move != ttData.move && !givesCheck && depth >= 5
            && moveCount > 3 && r >= 2 * 1024 && ss->excludedMove == Move::none()
            && nodeEvalVolatility >= SPSA::UNCERT_LMR_THRESHOLD)
            r -= 1024;
#endif

        // ★SF port #7(c) (SF search.cpp:1362-1363, commit 5f7348f03f "Reduce LMR less aggressively in loose alpha windows"):
        //   quiet 手で alpha が非決着値なら r += 3 * clamp(alpha - eval, -64, 96)。
        //   alpha が静的評価より高い (この手が alpha を超える見込みが薄い) ほど reduction を増やし、
        //   逆に eval が alpha を大きく上回るときは reduction を最大 192 (≈0.19ply) 減らす。
        //   定数は SF_EVAL_SCALE で換算済み (SPSA::SF_LMR_ALPHA_LO/HI/MULT_Q10、S=1024 なら 3*clamp と厳密一致)。
        //   OFF (既定) なら分岐 1 つのみ。
        if (SPSA::SF_LMR_ALPHA_TERM && !capture && !is_decisive(alpha))
            r += std::clamp(int(alpha - eval), SPSA::SF_LMR_ALPHA_LO, SPSA::SF_LMR_ALPHA_HI)
               * SPSA::SF_LMR_ALPHA_MULT_Q10 / 1024;

        // Scale up reductions for expected ALL nodes
        // 笘・ust(tune_params.rs:1502 coeff=0)縺ｧ縺ｯ辟｡蜉ｹ縲ＢllNode縺ｧ縺ｮ驕主臆蜑頑ｸ帙・縲∝━蜍｢譎ゅ↓
        // 逶ｸ謇九・PV螟匁姶陦途efute謇・=allNode縺ｧ莠域Φ螟悶↓fail high縺吶ｋ1謇・繧呈ｰｴ蟷ｳ邱壼､悶∈謚ｼ縺怜・縺吶・
        // suisho11 gap蛻・ｧ｣縺ｧ縲悟━蜍｢竊呈姶陦途efute縺ｮPV螟也紫50%縲阪・荳ｻ蝗蛟呵｣・竊・rust謨ｴ蜷医〒辟｡蜉ｹ蛹・譌｢螳・)縲・
        // 2026-10-16 訂正: 上の「rust 整合で無効化(既定0)」は**古い**。
        // 現行の spsa_params.h は **209** で **有効**。 exe の usi 応答も default 209。
        if (allNode && SPSA::LMR_ALLNODE_COEFF)
#if defined(P2_SRCH_BUNDLE_A)
            r += BundleA::allnode_div(r * SPSA::LMR_ALLNODE_COEFF, depth);
#else
            r += r * SPSA::LMR_ALLNODE_COEFF / (256 * depth + 260);
#endif

#if defined(P2_KHGATE)
        // KHGATE SELEXT: 荒れた node (浮き駒 / 自玉近傍の敵の利きが閾値以上) は読みを深く (r を SELEXT_LMR ply 減らす)。
        //   Step 17 (LMR) と Step 18 (full depth の r 判定) の両方に効く。 SELEXT_LMR=0 なら no-op。
        if (khGate.rough)
            r -= SPSA::SELEXT_LMR * 1024;
#endif

		// -----------------------
        // Step 17. Late moves reduction / extension (LMR)
        // Step 17. 驕・＞謖・＠謇九・蜑頑ｸ幢ｼ丞ｻｶ髟ｷ・・MR)
        // -----------------------

		/*
			涛 reduction(蜑頑ｸ・縺ｨ縺ｯ縲‥epth繧呈ｸ帙ｉ縺励◆謗｢邏｢縺ｮ縺薙→縲・
		        late move(驕・＞謖・＠謇・縺ｨ縺ｯ縲√％縺ｮnode縺ｧMovePicker縺ｧ逕滓・縺輔ｌ縺溘・縺後√≠縺ｨ縺ｮ縺ｻ縺・・謖・＠謇九・縺薙→縲・
		        Late Move Reduction縺ｯ縲・≦縺・欠縺玲焔縺ｮ蜑頑ｸ帙ｒ諢丞袖縺励※縺・※縲´MR縺ｨ逡･縺輔ｌ繧九・
		*/

        if (depth >= 2 && moveCount > 1)
        {
            /*
			  庁 depth繧呈ｸ帙ｉ縺励※謗｢邏｢縺輔○縺ｦ縲√◎縺ｮ謖・＠謇九′fail high縺励◆繧牙・縺ｮdepth縺ｧ蜀榊ｺｦ謗｢邏｢縺吶ｋ縲・
			      moveCount縺悟､ｧ縺阪＞繧ゅ・縺ｪ縺ｩ縺ｯ謗｢邏｢豺ｱ縺輔ｒ貂帙ｉ縺励※縺悶▲縺上ｊ隱ｿ縺ｹ繧九・
			      alpha蛟､繧呈峩譁ｰ縺励◎縺・↑繧・fail high縺瑚ｵｷ縺阪◆繧・縲’ull depth縺ｧ謗｢邏｢縺励↑縺翫☆縲・
			*/

            // In general we want to cap the LMR depth search at newDepth, but when
            // reduction is negative, we allow this move a limited search extension
            // beyond the first move depth.
            // To prevent problems when the max value is less than the min value,
            // std::clamp has been replaced by a more robust implementation.

			// 荳闊ｬ逧・↓縺ｯ LMR 謗｢邏｢縺ｮ豺ｱ縺輔ｒ newDepth 縺ｧ荳企剞縺ｫ縺励◆縺・′縲・
            // reduction 縺瑚ｲ縺ｮ蝣ｴ蜷医√％縺ｮ謖・＠謇九↓縺ｯ譛蛻昴・謖・＠謇九・豺ｱ縺輔ｒ雜・∴縺ｦ
            // 髯仙ｮ夂噪縺ｪ謗｢邏｢蟒ｶ髟ｷ繧定ｨｱ縺吶・
            // max 蛟､縺・min 蛟､繧医ｊ蟆上＆縺上↑繧句撫鬘後ｒ髦ｲ縺舌◆繧√・
            // std::clamp 縺ｯ繧医ｊ蝣・欧縺ｪ螳溯｣・↓鄂ｮ縺肴鋤縺医ｉ繧後◆縲・
            /*
                涛 C++縺ｮ莉墓ｧ倅ｸ翫《td::clamp(x, min, max)縺ｯ縲［in > max縺ｮ譎ゅ↓譛ｪ螳夂ｾｩ蜍穂ｽ懊〒縺ゅｋ縺九ｉ縲・
			        clamp()繧堤畑縺・ｋ縺ｮ縺ｧ縺ｯ縺ｪ縺上［ax()縺ｨmin()繧堤ｵ・∩蜷医ｏ縺帙※譖ｸ縺九↑縺・→縺・￠縺ｪ縺・・
			*/

			Depth d = std::max(1, std::min(newDepth - r / 1024, newDepth + 2)) + PvNode;

            ss->reduction = newDepth - d;
#if defined(P2_SRCH_BUNDLE_A)
            value         = -search_nonpv(pos, ss + 1, -(alpha + 1), -alpha, d, true);
#else
            value         = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);
#endif
            ss->reduction = 0;
#ifdef P2_LMR_STATS
            // 条件フラグ (1 手が複数の条件に同時に数えられる。 none は 1-6 のどれにも当たらない手)
            bool lmrc[P2LmrStat::NC] = {};
            lmrc[1] = cutNode && ttData.move;
            lmrc[2] = cutNode && !ttData.move;
            lmrc[3] = ttCapture;
            lmrc[4] = (ss + 1)->cutoffCnt == 2;
            lmrc[5] = (ss + 1)->cutoffCnt > 2;
            lmrc[6] = allNode && (ss + 1)->cutoffCnt > 1;
            lmrc[0] = !(lmrc[1] || lmrc[2] || lmrc[3] || lmrc[4] || lmrc[5] || lmrc[6]);
            const bool lmrReduced = d < newDepth;
            const bool lmrFh      = value > alpha;
            Depth      lmrFullDepth = newDepth;
            bool       lmrRs = false;
            // 見落としの標本: 本当に削った手で、 浅い探索が fail low したものを 1/N で newDepth から読み直す
            int lmrVerify = -1;
            if (lmrReduced && !lmrFh && P2LmrStat::verify_rate() > 0
                && P2LmrStat::sample.fetch_add(1, std::memory_order_relaxed) % P2LmrStat::verify_rate() == 0)
            {
#if defined(P2_SRCH_BUNDLE_A)
                Value v2 = -search_nonpv(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);
#else
                Value v2 = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);
#endif
                lmrVerify = v2 > alpha;
            }
#endif

            // Do a full-depth search when reduced LMR search fails high
            // 豺ｱ縺輔ｒ貂帙ｉ縺励◆ LMR 謗｢邏｢縺掲ail high繧貞・縺励◆蝣ｴ蜷医・縲’ull depth(蜈・・謗｢邏｢豺ｱ縺・縺ｧ謗｢邏｢繧定｡後≧

            // (*Scaler) Shallower searches here don't scale well
            // ・・Scaler・峨％縺薙〒豬・＞謗｢邏｢繧貞｢励ｄ縺励※繧・髟ｷ縺・戟縺｡譎る俣縺ｧ縺ｯ)縺・∪縺上せ繧ｱ繝ｼ繝ｫ縺励↑縺・

            if (value > alpha)
            {
#ifdef UNCERT_LMR_MEASURE
                if (uncertReduced)
                    g_uncertMeasure.firedFailHigh.fetch_add(1, std::memory_order_relaxed);
#endif
                // Adjust full-depth search based on LMR results - if the result was
                // good enough search deeper, if it was bad enough search shallower.
                // LMR縺ｮ邨先棡縺ｫ蝓ｺ縺･縺・※螳悟・縺ｪ謗｢邏｢豺ｱ縺輔ｒ隱ｿ謨ｴ縺励∪縺・-
                // 邨先棡縺悟香蛻・↓濶ｯ縺代ｌ縺ｰ豺ｱ縺乗爾邏｢縺励∝香蛻・↓謔ｪ縺代ｌ縺ｰ豬・￥謗｢邏｢縺励∪縺吶・

                const bool doDeeperSearch    = d < newDepth && value > bestValue + SPSA::LMR_RESEARCH_DEEPER_THRESHOLD;
                const bool doShallowerSearch = value < bestValue + SPSA::LMR_RESEARCH_SHALLOWER_THRESHOLD;

                newDepth += doDeeperSearch - doShallowerSearch;

                if (newDepth > d)
                {
#ifdef UNCERT_LMR_MEASURE
                    if (uncertReduced)
                        g_uncertMeasure.firedReSearch.fetch_add(1, std::memory_order_relaxed);
#endif
#if defined(P2_SRCH_BUNDLE_A)
                    value = -search_nonpv(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);
#else
                    value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);
#endif
#ifdef P2_LMR_STATS
                    lmrRs = true;
#endif
                }

                // Post LMR continuation history updates
                // LMR蠕後・continuation history縺ｮ譖ｴ譁ｰ

                update_continuation_histories(ss, movedPiece, move.to_sq(),
                                              SPSA::POST_LMR_CONTHIST_BONUS);
            }
#ifdef P2_LMR_STATS
            if (lmrReduced)
                for (int i = 0; i < P2LmrStat::NC; ++i)
                {
                    if (!lmrc[i]) continue;
                    P2LmrStat::n[i].fetch_add(1, std::memory_order_relaxed);
                    P2LmrStat::red[i].fetch_add(u64(lmrFullDepth - d), std::memory_order_relaxed);
                    P2LmrStat::rsum[i].fetch_add(u64(std::max(0, r)), std::memory_order_relaxed);
                    if (lmrFh) P2LmrStat::fh[i].fetch_add(1, std::memory_order_relaxed);
                    if (lmrRs) P2LmrStat::rs[i].fetch_add(1, std::memory_order_relaxed);
                    if (lmrFh && value > alpha) P2LmrStat::cf[i].fetch_add(1, std::memory_order_relaxed);
                    if (lmrVerify >= 0)
                    {
                        P2LmrStat::vn[i].fetch_add(1, std::memory_order_relaxed);
                        if (lmrVerify) P2LmrStat::vm[i].fetch_add(1, std::memory_order_relaxed);
                    }
                }
#endif
        }

		// -----------------------
        // Step 18. Full-depth search when LMR is skipped
        // Step 18. LMR縺後せ繧ｭ繝・・縺輔ｌ縺溷ｴ蜷医・螳悟・縺ｪ謗｢邏｢
        // -----------------------

        else if (!PvNode || moveCount > 1)
        {
            // Increase reduction if ttMove is not present
            // ttMove縺悟ｭ伜惠縺励↑縺・ｴ蜷医∝炎貂帙ｒ蠅励ｄ縺励∪縺吶・

            if (!ttData.move)
                r += SPSA::LMR_STEP18_NO_TTMOVE_ADD;  // rust(alpha_beta.rs full_depth_no_tt_add=1118, C++譌ｧ=1057)

            // Note that if expected reduction is high, we reduce search depth here
            // 譛溷ｾ・＆繧後ｋ蜑頑ｸ帙′螟ｧ縺阪＞蝣ｴ蜷医√％縺薙〒謗｢邏｢豺ｱ縺輔ｒ1貂帙ｉ縺吶％縺ｨ縺ｫ豕ｨ諢上＠縺ｦ縺上□縺輔＞縲・

#if defined(P2_SRCH_BUNDLE_A)
			value = -search_nonpv(pos, ss + 1, -(alpha + 1), -alpha,
                                   newDepth - (r > SPSA::LMR_STEP18_THRESHOLD1) - (r > SPSA::LMR_STEP18_THRESHOLD2 && newDepth > 2), !cutNode);
#else
			value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha,
                                   newDepth - (r > SPSA::LMR_STEP18_THRESHOLD1) - (r > SPSA::LMR_STEP18_THRESHOLD2 && newDepth > 2), !cutNode);
#endif
		}

        // For PV nodes only, do a full PV search on the first move or after a fail high,
        // otherwise let the parent node fail low with value <= alpha and try another move.
        // PV繝弱・繝峨・蝣ｴ蜷医・縺ｿ縲∵怙蛻昴・謇九ｄfail high縺ｮ蠕後↓螳悟・縺ｪPV謗｢邏｢繧定｡後＞縺ｾ縺吶・
        // 縺昴ｌ莉･螟悶・蝣ｴ蜷医∬ｦｪ繝弱・繝峨′ value <= alpha 縺ｧfail low縺励∝挨縺ｮ謇九ｒ隧ｦ縺ｿ縺ｾ縺吶・

		/*
		   涛 PV node縺ｫ縺翫＞縺ｦ縺ｯ縲’ull depth search縺掲ail high縺励◆縺ｪ繧臼V node縺ｨ縺励※search縺励↑縺翫☆縲・
               縺溘□縺励」alue >= beta縺ｪ繧峨∵ｭ｣遒ｺ縺ｪ蛟､繧呈ｱゅａ繧九％縺ｨ縺ｫ縺ｯ縺ゅ∪繧頑э蜻ｳ縺後↑縺・・縺ｧ縲・
		       縺薙ｌ縺ｯ縺帙★縺ｫbeta cut縺励※縺励∪縺・・
		*/

        if (PvNode && (moveCount == 1 || value > alpha))
        {
            // 谺｡縺ｮnode縺ｮPV繝昴う繝ｳ繧ｿ繝ｼ縺ｯ縺薙・node縺ｮpv繝舌ャ繝輔ぃ繧呈欠縺吶ｈ縺・↓縺励※縺翫￥縲・
            (ss + 1)->pv = &pv;
            (ss + 1)->pv->clear();

            // Extend move from transposition table if we are about to dive into qsearch.
            // decisive score handling improves mate finding and retrograde analysis.
			// 縺薙ｌ縺九ｉ髱呎ｭ｢謗｢邏｢・・search・峨↓蜈･繧阪≧縺ｨ縺励※縺・ｋ蝣ｴ蜷医∫ｽｮ謠幄｡ｨ縺九ｉ謖・＠謇九ｒ諡｡蠑ｵ縺吶ｋ縲・
			// 豎ｺ螳夂噪繧ｹ繧ｳ繧｢・・ecisive score・峨・蜃ｦ逅・・縲∬ｩｰ縺ｿ縺ｮ逋ｺ隕九ｄ蠕碁隗｣譫舌ｒ謾ｹ蝟・☆繧九・

			if (move == ttData.move
                && ((is_valid(ttData.value) && is_decisive(ttData.value) && ttData.depth > 0)
                    || ttData.depth > 1))
                newDepth = std::max(newDepth, 1);

			// 統 full depth縺ｧ謗｢邏｢縺吶ｋ縺ｨ縺阪・cutNode縺ｫ縺励※縺ｯ縺・￠縺ｪ縺・・
            value = -search<PV>(pos, ss + 1, -beta, -alpha, newDepth, false);
        }

		// -----------------------
        // Step 19. Undo move
        // Step 19. 1謇区綾縺・
        // -----------------------

		undo_move(pos, move);

		ASSERT_LV3(-VALUE_INFINITE < value && value < VALUE_INFINITE);

		// -----------------------
        // Step 20. Check for a new best move
        // -----------------------

        // Finished searching the move. If a stop occurred, the return value of
        // the search cannot be trusted, and we return immediately without updating
        // best move, principal variation nor transposition table.
		// 謖・＠謇九・謗｢邏｢縺檎ｵゆｺ・＠縺ｾ縺励◆縲ゅｂ縺怜●豁｢縺檎匱逕溘＠縺溷ｴ蜷医∵爾邏｢縺ｮ霑斐ｊ蛟､縺ｯ菫｡鬆ｼ縺ｧ縺阪↑縺・◆繧√・
        // 譛蝟・焔縲∽ｸｻ隕∝､牙喧縲√ヨ繝ｩ繝ｳ繧ｹ繝昴ず繧ｷ繝ｧ繝ｳ繝・・繝悶Ν繧呈峩譁ｰ縺帙★縺ｫ逶ｴ縺｡縺ｫ謌ｻ繧翫∪縺吶・

        if (threads.stop.load(std::memory_order_relaxed))
            return VALUE_ZERO;

		// -----------------------
        //  root node逕ｨ縺ｮ迚ｹ蛻･縺ｪ蜃ｦ逅・
        // -----------------------

        if (rootNode)
        {
            RootMove& rm = *std::find(rootMoves.begin(), rootMoves.end(), move);

			// 🚨2026-09-26: ここにあった BookAware の `value += rm.bookBias` は**削除**した。
			//   探索値に下駄を足すと bestValue/alpha 経由で α-β の窓が汚染され、
			//   返るのが真の値ではなく境界値になるため、 設計の保証「損は高々 B」が壊れていた。
			//   実測: B=0 対 B=60 で 48 手中 42 手が不変条件違反、 最大 349cp の手を指した。
			//   正しい選択は探索後の 2 パス方式 (start_searching() 内、 設計 §19)。

			/*
				涛
					effort           : 縺薙・RootMove縺ｮ縺溘ａ縺ｫ謗｢邏｢縺励◆node謨ｰ
					averageScore     : root縺ｮ蟷ｳ蝮・せ繧ｳ繧｢縲・
					meanSquaredScore : root縺ｮ莠御ｹ怜ｹｳ蝮・せ繧ｳ繧｢縲・

					縺薙ｌ繧峨・縲∥spiration search縺ｮ縺ｨ縺阪↓繧ゅ≧荳蝗枴teration縺悟屓繧九°縺ｮ蛻､螳壹↓逕ｨ縺・ｋ縲・
			*/

            // root縺ｧ縺薙・RootMoves縺ｮ謖・＠謇九↓蟇ｾ縺励※謗｢邏｢縺励◆node謨ｰ繧貞刈邂励・
			rm.effort += nodes - nodeCount;

            // M3 根の YBWC: **長兄を読み終えたら helper を解放する** (不変条件 I-4)。
            //   🚨2026-11-02 の最初の実装でここを書き忘れ、 helper が永久に待って
            //     taken=0 / wait_ms が探索時間のほぼ全部になった。 関門② で検出。
            if (rootSplit && ybwcGen && is_mainthread() && moveCount == 1)
                rootSplit->eldest_finished(ybwcGen, value);

			rm.averageScore =
              rm.averageScore != -VALUE_INFINITE ? (value + rm.averageScore) / 2 : value;

            rm.meanSquaredScore = rm.meanSquaredScore != -VALUE_INFINITE * VALUE_INFINITE
                                  ? (value * std::abs(value) + rm.meanSquaredScore) / 2
                                  : value * std::abs(value);

            // PV move or new best move?
            // PV縺ｮ謖・＠謇九°縲∵眠縺励＞best move縺具ｼ・

            if (moveCount == 1 || value > alpha)
            {
                // 庁 root node縺ｫ縺翫＞縺ｦPV縺ｮ謖・＠謇九∪縺溘・縲∃ｱ蛟､繧呈峩譁ｰ縺励◆蝣ｴ蜷医√せ繧ｳ繧｢繧偵そ繝・ヨ縺励※縺翫￥縲・
                //    (iteration縺ｮ邨ゅｏ繧翫〒sort縺吶ｋ縺ｮ縺ｧ縺昴・縺ｨ縺阪↓謖・＠謇九′蜈･繧梧崛繧上ｋ縲・

                rm.score = rm.uciScore = value;
                rm.selDepth            = selDepth;
                rm.scoreLowerbound = rm.scoreUpperbound = false;

                if (value >= beta)
                {
                    rm.scoreLowerbound = true;
                    rm.uciScore        = beta;
                }
                else if (value <= alpha)
                {
                    rm.scoreUpperbound = true;
                    rm.uciScore        = alpha;
                }

				// 統 PV縺ｯ螟牙喧縺吶ｋ縺ｯ縺壹↑縺ｮ縺ｧ縺・▲縺溘ｓ繝ｪ繧ｻ繝・ヨ縺励※縺・ｋ縲・
                rm.pv.resize(1);

				// 統 1謇矩ｲ繧√◆縺ｮ縺縺九ｉ縲∽ｽ輔ｉ縺輝V繧呈戟縺｣縺ｦ縺・ｋ縺ｯ縺壹・
                ASSERT_LV3((ss + 1)->pv);

				// 統 Root縺ｧPV縺悟､峨ｏ繧九・縺ｯ遞縺ｪ縺ｮ縺ｧ縺薙％縺後■繧・▲縺ｨ縺舌ｉ縺・㍾縺上※繧ょ撫鬘後↑縺・・
                //     譁ｰ縺励￥螟峨ｏ縺｣縺滓欠縺玲焔縺ｮ蠕檎ｶ壹・pv繧坦ootMoves::pv縺ｫ繧ｳ繝斐・縺励※縺上ｋ縲・

                for (Move pvMove : *(ss + 1)->pv)
                    rm.pv.push_back(pvMove);

                // We record how often the best move has been changed in each iteration.
                // This information is used for time management. In MultiPV mode,
                // we must take care to only do this for the first PV line.

				// 蜷・う繝・Ξ繝ｼ繧ｷ繝ｧ繝ｳ縺ｧ譛蝟・焔縺後←縺ｮ縺上ｉ縺・・鬆ｻ蠎ｦ縺ｧ螟牙喧縺励◆縺九ｒ險倬鹸縺励∪縺吶・
                // 縺薙・諠・ｱ縺ｯ譎る俣邂｡逅・↓菴ｿ逕ｨ縺輔ｌ縺ｾ縺吶・ultiPV 繝｢繝ｼ繝峨〒縺ｯ縲・
                // 譛蛻昴・ PV 繝ｩ繧､繝ｳ縺ｫ蟇ｾ縺励※縺ｮ縺ｿ縺薙ｌ繧定｡後≧繧医≧豕ｨ諢上☆繧句ｿ・ｦ√′縺ゅｊ縺ｾ縺吶・

				// 笞 !thisThread->pvIdx 縺ｨ縺・≧譚｡莉ｶ繧貞・繧後※縺翫°縺ｪ縺・→MultiPV縺ｧ
                //     time management縺後♀縺九＠縺上↑繧九・

                if (moveCount > 1 && !pvIdx)
                    ++bestMoveChanges;

                // M3 根の YBWC: 兄弟手の結果を共有表へ (PV なので PV も載せる)
                if (rootSplit && ybwcGen && ybwcIdx >= 0)
                {
                    rootSplit->publish_result(ybwcGen, ybwcIdx, value, rm.pv.begin(),
                                              int(rm.pv.size()));
                    if (!is_mainthread())        // 🚨master の分を混ぜると発火が読めない
                        rootSplit->cntDone.fetch_add(1, std::memory_order_relaxed);
                    ybwcIdx = -1;
                }
            }
            else
                // All other moves but the PV, are set to the lowest value: this
                // is not a problem when sorting because the sort is stable and the
                // move position in the list is preserved - just the PV is pushed up.

				// PV莉･螟悶・縺吶∋縺ｦ縺ｮ謇九・譛菴主､縺ｫ險ｭ螳壹＆繧後∪縺吶・
                // 縺薙ｌ縺ｯ繧ｽ繝ｼ繝域凾縺ｫ蝠城｡後→縺ｪ繧峨↑縺・〒縺吶・
                // 縺ｪ縺懊↑繧峨√た繝ｼ繝医・螳牙ｮ壹＠縺ｦ縺翫ｊ縲√Μ繧ｹ繝亥・縺ｮ謇九・菴咲ｽｮ縺ｯ菫晄戟縺輔ｌ縺ｦ縺・ｋ縺九ｉ縺ｧ縺・
                // - PV縺縺代′荳翫↓謚ｼ縺嶺ｸ翫￡繧峨ｌ縺ｾ縺吶・

            {
                rm.score = -VALUE_INFINITE;

                // M3 根の YBWC: PV でない手も値だけは共有する (回収側で規則を当てる)
                if (rootSplit && ybwcGen && ybwcIdx >= 0)
                {
                    rootSplit->publish_result(ybwcGen, ybwcIdx, value, nullptr, 0);
                    if (!is_mainthread())
                        rootSplit->cntDone.fetch_add(1, std::memory_order_relaxed);
                    ybwcIdx = -1;
                }
            }

				/*
					涛 root node縺ｫ縺翫＞縺ｦﾎｱ蛟､繧呈峩譁ｰ縺励↑縺九▲縺溘・縺ｧ縺ゅｌ縺ｰ縲√％縺ｮ謖・＠謇九・繧ｹ繧ｳ繧｢繧・
						-VALUE_INFINITE縺ｫ縺励※縺翫￥縲・
						縺薙≧縺励※縺翫°縺ｪ縺代ｌ縺ｰ縲《table_sort() 縺励※縺・ｋ縺ｫ繧ゅ°縺九ｏ繧峨★縲・
						蜑榊屓縺ｮ蜿榊ｾｩ豺ｱ蛹悶・縺ｨ縺阪・蛟､縺ｨ縺ｮ 螟ｧ蟆乗ｯ碑ｼ・＠縺ｦ縺励∪縺・欠縺玲焔縺ｮ鬆・分縺・
						蜈･繧梧崛繧上▲縺ｦ縺励∪縺・％縺ｨ縺ｫ繧医ｋ繧ｪ繝ｼ繝繝ｪ繝ｳ繧ｰ諤ｧ閭ｽ縺ｮ菴惹ｸ九′縺ゅｊ縺・ｋ縲・
				*/

        }

		// -----------------------
        //  alpha蛟､縺ｮ譖ｴ譁ｰ蜃ｦ逅・
        // -----------------------

        // In case we have an alternative move equal in eval to the current bestmove,
        // promote it to bestmove by pretending it just exceeds alpha (but not beta).

		// 隧穂ｾ｡蛟､縺檎樟蝨ｨ縺ｮ譛蝟・焔縺ｨ遲峨＠縺・ｻ｣譖ｿ謇九′縺ゅｋ蝣ｴ蜷医・
        // 縺昴・謇九ｒ譛蝟・焔縺ｫ譏・ｼ縺輔○縺ｾ縺吶ゅ％縺ｮ髫帙∃ｱ・医い繝ｫ繝輔ぃ・峨ｒ蟆代＠縺縺題ｶ・∴繧九′縲・
        // ﾎｲ・医・繝ｼ繧ｿ・峨・雜・∴縺ｪ縺・ｈ縺・↓隕九○縺九￠縺ｾ縺吶・

        int inc = (value == bestValue && ss->ply + 2 >= rootDepth && (int(nodes) & 14) == 0
                   && !is_win(std::abs(value) + 1));

        if (value + inc > bestValue)
        {
            bestValue = value;

            if (value + inc > alpha)
            {
                bestMove = move;

                if (PvNode && !rootNode)  // Update pv even in fail-high case
					                      // fail-high縺ｮ縺ｨ縺阪↓繧１V繧置pdate縺吶ｋ縲・
                    ss->pv->update(move, (ss + 1)->pv);

                if (value >= beta)
                {
					// (*Scaler) Infrequent and small updates scale well
					// ・・Scaler・峨∪繧後〒縲∝ｰ上＆縺ｪ譖ｴ譁ｰ縺ｯ繧ｹ繧ｱ繝ｼ繝ｫ縺吶ｋ縲・

                    ss->cutoffCnt += (extension < 2) || PvNode;
                    ASSERT_LV3(value >= beta);  // Fail high

					/*
						涛 value >= beta 縺ｪ繧・fail high(beta cut)
							縺ｾ縺溘］on PV縺ｧ縺ゅｋ縺ｪ繧画爾邏｢遯薙・蟷・′0縺ｪ縺ｮ縺ｧalpha繧呈峩譁ｰ縺励◆譎らせ縺ｧ縲・
							value >= beta縺瑚ｨ縺医※縲｜eta cut縺ｧ縺ゅｋ縲・
					*/
                    break;
                }

                // Reduce other moves if we have found at least one score improvement
                // 蟆代↑縺上→繧・縺､縺ｮ繧ｹ繧ｳ繧｢縺ｮ謾ｹ蝟・′隕九ｉ繧後◆蝣ｴ蜷医∽ｻ悶・謇・縺ｮ謗｢邏｢豺ｱ縺・繧貞炎貂帙＠縺ｾ縺吶・

                if (depth > 2 && depth < 14 && !is_decisive(value))
                    depth -= 2;

				ASSERT_LV3(depth > 0);
                alpha = value;  // Update alpha! Always alpha < beta
                                // alpha蛟､繧呈峩譁ｰ! 縺､縺ｭ縺ｫ alpha < beta

				/*
					町 縺薙・縺ｨ縺咲嶌謇九°繧峨・隧ｰ縺ｿ縺後≠繧九°縺ｩ縺・°繧・
					    隱ｿ縺ｹ繧九↑縺ｩ縺励◆縺ｻ縺・′濶ｯ縺・↑繧峨％縺薙↓譖ｸ縺上∋縺励・
				*/
            }
        }

        // If the move is worse than some previously searched move,
        // remember it, to update its stats later.

		// 縺昴・謇九′莉･蜑阪↓謗｢邏｢縺輔ｌ縺滉ｻ悶・謇九ｈ繧翫ｂ謔ｪ縺・ｴ蜷医・
        // 蠕後〒縺昴・邨ｱ險医ｒ譖ｴ譁ｰ縺吶ｋ縺溘ａ縺ｫ險俶・縺励※縺翫″縺ｾ縺吶・

        if (move != bestMove && moveCount <= SEARCHEDLIST_CAPACITY)
        {
            if (capture)
                // 謗｢邏｢縺励◆縲・ｧ偵ｒ謐慕佐縺吶ｋ謖・＠謇・
                capturesSearched.push_back(move);
			else
                // 謗｢邏｢縺励◆縲・ｧ偵ｒ謐慕佐縺励↑縺・欠縺玲焔
                quietsSearched.push_back(move);
        }
    } // end of while

    // I-2: 後回しを読んだ分 moveCount が小さく残っているので **総数**に戻す。
    //   これを忘れると直後の ASSERT_LV5 と `if (!moveCount)` の詰み判定が壊れる。
    //   遅延していないときも moveCountTotal == moveCount なので bit-exact。
    // M4: 自分が提示した枠を解放する (不変条件 I-3)。
    //   🚨helper がまだ読んでいても待たない。 結果は置換表にしか置かないので、
    //     解放してしまっても害は無い (helper の finish は CAS で落ちるだけ)。
    if (deleg)
        for (int i = 0; i < nDeferred; ++i)
            if (deferred[i].slot >= 0)
                deleg->release(deferred[i].slot);

    moveCount = moveCountTotal;

	// -----------------------
    // Step 21. Check for mate and stalemate
    // -----------------------

    // All legal moves have been searched and if there are no legal moves, it
    // must be a mate or a stalemate. If we are in a singular extension search then
    // return a fail low score.

	// 縺吶∋縺ｦ縺ｮ蜷域ｳ墓焔縺梧爾邏｢縺輔ｌ縺ｦ縺翫ｊ縲∝粋豕墓焔縺悟ｭ伜惠縺励↑縺・ｴ蜷医・
	// 縺昴ｌ縺ｯ隧ｰ縺ｿ縺九せ繝・う繝ｫ繝｡繧､繝医〒縺ゅｋ縲・
	// 繧ｷ繝ｳ繧ｮ繝･繝ｩ繝ｼ蟒ｶ髟ｷ謗｢邏｢荳ｭ縺ｧ縺ゅｌ縺ｰ縲’ail low 繧ｹ繧ｳ繧｢繧定ｿ斐☆縲・

	// 統 蟆・｣九〒縺ｯ繧ｹ繝・う繝ｫ繝｡繧､繝医・蟄伜惠縺励↑縺・・縺ｧ縲∝粋豕墓焔縺後↑縺代ｌ縺ｰ雋縺代・

	// 縺薙・Stockfish縺ｮassert縲∝粋豕墓焔繧堤函謌舌＠縺ｦ縺・ｋ縺ｮ縺ｧ驥阪☆縺弱ｋ縲り憶縺上↑縺・・
    ASSERT_LV5(moveCount || !ss->inCheck || excludedMove || !MoveList<LEGAL_ALL>(pos).size());

    // Adjust best value for fail high cases
    // fail high縺ｮ蝣ｴ蜷医↓譛濶ｯ蛟､繧定ｪｿ謨ｴ縺吶ｋ

    if (bestValue >= beta && !is_decisive(bestValue) && !is_decisive(alpha))
        bestValue = (bestValue * depth + beta) / (depth + 1);

	// 笞 Stockfish縺ｧ縺ｯ縲√％縺薙・繧ｳ繝ｼ繝峨・莉･荳九・繧医≧縺ｫ縺ｪ縺｣縺ｦ縺・ｋ縺後√％繧後・縲・
    //     閾ｪ邇峨↓邇区焔縺後°縺九▲縺ｦ縺翫ｉ縺壽欠縺玲焔縺後↑縺・ｴ蜷医・縲《talemate縺ｧ蠑輔″蛻・￠縺縺九ｉ縲・

#if STOCKFISH
    if (!moveCount)
        bestValue = excludedMove ? alpha : ss->inCheck ? mated_in(ss->ply) : VALUE_DRAW;
#else
    // 笞 竍薙％縺凪∮縲≫・Stockfish縺ｮ繧ｳ繝ｼ繝俄・繧偵◎縺ｮ縺ｾ縺ｾ繧ｳ繝斐・縺励※縺薙↑縺・ｈ縺・↓豕ｨ諢擾ｼ・

	// ､・(蟆・｣九〒縺ｯ)蜷域ｳ墓焔縺後↑縺・== 隧ｰ縺ｾ縺輔ｌ縺ｦ縺・ｋ 縺ｪ縺ｮ縺ｧ縲〉oot縺ｮ螻髱｢縺九ｉ縺ｮ謇区焚縺ｧ隧ｰ縺ｾ縺輔ｌ縺溘→縺・≧隧穂ｾ｡蛟､繧定ｿ斐☆縲・
    //     縺溘□縺励《ingular extension荳ｭ縺ｮ縺ｨ縺阪・縲》tMove縺ｮ謖・＠謇九′髯､螟悶＆繧後※縺・ｋ縺ｮ縺ｧ蜊倥↓alpha繧定ｿ斐☆縺ｹ縺阪・
    if (!moveCount)
        bestValue = excludedMove ? alpha : mated_in(ss->ply);
#endif

	// If there is a move that produces search value greater than alpha,
    // we update the stats of searched moves.
    // alpha繧医ｊ繧ょ､ｧ縺阪↑謗｢邏｢蛟､繧堤函縺ｿ蜃ｺ縺呎焔縺後≠繧句ｴ蜷医∵爾邏｢縺輔ｌ縺滓焔縺ｮ邨ｱ險医ｒ譖ｴ譁ｰ縺励∪縺・

	else if (bestMove)
    {
        // 庁 quiet縺ｪ(鬧偵ｒ謐慕佐縺励↑縺・best move縺ｪ縺ｮ縺ｧkiller縺ｨhistory縺ｨcountermoves繧呈峩譁ｰ縺吶ｋ縲・

        update_all_stats(pos, ss, *this, bestMove, prevSq, quietsSearched, capturesSearched, depth,
                         ttData.move, nodeThreatCtx, threatValid, bestValue >= beta, PvNode);
        if (!PvNode)
            ttMoveHistory << (bestMove == ttData.move ? SPSA::TTMOVE_HISTORY_BONUS
                                                      : SPSA::TTMOVE_HISTORY_MALUS);
    }

    // Bonus for prior quiet countermove that caused the fail low
    // fail low繧貞ｼ輔″襍ｷ縺薙＠縺・謇句燕縺ｮquiet countermove縺ｫ蟇ｾ縺吶ｋ繝懊・繝翫せ

	/*
		涛 bestMove縺後↑縺・== fail low縺励※縺・ｋ繧ｱ繝ｼ繧ｹ縺ｪ縺ｮ縺ｧ縲・
			fail low繧貞ｼ輔″襍ｷ縺薙＠縺溷燕node縺ｧ縺ｮcounter move縺ｫ蟇ｾ縺励※繝懊・繝翫せ繧貞刈轤ｹ縺吶ｋ縲・

		投 縲占ｨ域ｸｬ雉・侭 15.縲壮earch()縺ｧfail low縺励※縺・ｋ縺ｨ縺阪↓history縺ｮupdate繧定｡後↑縺・擅莉ｶ
	*/
    else if (!priorCapture && prevSq != SQ_NONE)
    {
        int bonusScale = SPSA::FAILLOW_BONUS_SCALE_BASE;
#if defined(P2_SRCH_BUNDLE_A)
        bonusScale -= BundleA::vdiv(bundleDivs.faillowStatScore, (ss - 1)->statScore,
                                    SPSA::FAILLOW_BONUS_STATSCORE_DIV, BundleA::V_L2_FAILLOW);
#else
        bonusScale -= (ss - 1)->statScore / SPSA::FAILLOW_BONUS_STATSCORE_DIV;
#endif
        bonusScale += std::min(SPSA::FAILLOW_BONUS_DEPTH_MULT * depth, SPSA::FAILLOW_BONUS_DEPTH_MAX);
        bonusScale += SPSA::FAILLOW_BONUS_MOVECOUNT
                    * ((ss - 1)->moveCount > SPSA::FAILLOW_BONUS_MOVECOUNT_TH);
        bonusScale += SPSA::FAILLOW_BONUS_EVAL
                    * (!ss->inCheck && bestValue <= ss->staticEval - SPSA::FAILLOW_BONUS_EVAL_MARGIN);
        bonusScale += SPSA::FAILLOW_BONUS_PREVEVAL
                    * (!(ss - 1)->inCheck
                       && bestValue <= -(ss - 1)->staticEval - SPSA::FAILLOW_BONUS_PREVEVAL_MARGIN);

        bonusScale = std::max(bonusScale, 0);

        const int scaledBonus = std::min(SPSA::FAILLOW_SCALED_DEPTH_MULT * depth
                                           - SPSA::FAILLOW_SCALED_OFFSET,
                                         SPSA::FAILLOW_SCALED_MAX) * bonusScale;

        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq,
                                      scaledBonus * SPSA::FAILLOW_CONTHIST_MULT / 16384);

        mainHistory[~us][((ss - 1)->currentMove).raw()] << scaledBonus * SPSA::FAILLOW_MAINHIST_MULT / 32768;

		// TODO : 縺薙ｌ縺ｧ蜷医▲縺ｦ繧九°・溘≠縺ｨ縺ｧ讀懆ｨｼ縺吶ｋ縲・
        if (type_of(pos.piece_on(prevSq)) != PAWN && ((ss - 1)->currentMove).type_of() != PROMOTION)
            sharedHistory.pawn_entry(pos)[pos.piece_on(prevSq)][prevSq] << scaledBonus * SPSA::FAILLOW_PAWNHIST_MULT / 8192;
    }

    // Bonus for prior capture countermove that caused the fail low
    // 蜑阪・fail low繧貞ｼ輔″襍ｷ縺薙＠縺歡apture countermove縺ｫ蟇ｾ縺吶ｋ繝懊・繝翫せ

    else if (priorCapture && prevSq != SQ_NONE)
    {
        Piece capturedPiece = pos.captured_piece();
        assert(capturedPiece != NO_PIECE);
        captureHistory[pos.piece_on(prevSq)][prevSq][type_of(capturedPiece)] << SPSA::FAILLOW_CAPTURE_BONUS;
    }

    // 笞 蟆・｣九〒縺ｯtable probe繧剃ｽｿ縺｣縺ｦ縺・↑縺・・縺ｧ縲［axValue縺ｯ菴ｿ繧上↑縺・・
    //     繧・∴縺ｫStockfish縺ｮ縺薙％縺ｮ繧ｳ繝ｼ繝峨・荳崎ｦ√・maxValue縺ｧcap縺吶ｋ蠢・ｦ√′縺ｪ縺・
#if STOCKFISH
    if (PvNode)
        bestValue = std::min(bestValue, maxValue);
#endif

    // If no good move is found and the previous position was ttPv, then the previous
    // opponent move is probably good and the new position is added to the search tree.

	// 繧ゅ＠濶ｯ縺・欠縺玲焔縺瑚ｦ九▽縺九ｉ縺・bestValue縺径lpha繧呈峩譁ｰ縺帙★)縲∝燕縺ｮ螻髱｢縺ｯttPv繧帝∈繧薙〒縺・◆蝣ｴ蜷医・縲・
    // 蜑阪・逶ｸ謇九・謇九′縺翫◎繧峨￥濶ｯ縺・焔縺ｧ縺ゅｊ縲∵眠縺励＞螻髱｢縺梧爾邏｢譛ｨ縺ｫ霑ｽ蜉縺輔ｌ繧九・
    // (ttPv繧稚rue縺ｫ螟画峩縺励※TTEntry縺ｫ菫晏ｭ倥☆繧・

    if (bestValue <= alpha)
        ss->ttPv = ss->ttPv || (ss - 1)->ttPv;

	// -----------------------
    //  鄂ｮ謠幄｡ｨ縺ｫ菫晏ｭ倥☆繧・
    // -----------------------

    // Write gathered information in transposition table. Note that the
    // static evaluation is saved as it was before correction history.

	// 蜿朱寔縺励◆諠・ｱ繧偵ヨ繝ｩ繝ｳ繧ｹ繝昴ず繧ｷ繝ｧ繝ｳ繝・・繝悶Ν縺ｫ譖ｸ縺崎ｾｼ縺ｿ縺ｾ縺吶・
    // 髱咏噪隧穂ｾ｡縺ｯ縲∽ｿｮ豁｣螻･豁ｴ縺碁←逕ｨ縺輔ｌ繧句燕縺ｮ迥ｶ諷九〒菫晏ｭ倥＆繧後ｋ縺薙→縺ｫ豕ｨ諢上＠縺ｦ縺上□縺輔＞縲・

	/*
		涛 beta繧定ｶ・∴縺ｦ縺・ｋ縺ｨ縺・≧縺薙→縺ｯbeta cut縺輔ｌ繧九ｏ縺代〒谿九ｊ縺ｮ謖・＠謇九ｒ隱ｿ縺ｹ縺ｦ縺・↑縺・°繧・
			逵溘・蛟､縺ｯ縺ｾ縺螟ｧ縺阪＞縺ｨ閠・∴繧峨ｌ繧九・

			縺吶↑繧上■縲√％縺ｮ縺ｨ縺榊､縺ｯ荳狗阜縺ｨ閠・∴繧峨ｌ繧九°繧峨。OUND_LOWER縲・
			縺輔ｂ縺ｪ縺上・縲・PvNode縺ｪ繧・譫晏・繧翫・縺励※縺・↑縺・・縺ｧ縲・
			縺薙ｌ縺梧ｭ｣遒ｺ縺ｪ蛟､縺ｧ縺ゅｋ縺ｯ縺壹□縺九ｉ縲。OUND_EXACT繧定ｿ斐☆縲・

			縺ｾ縺溘￣vNode縺ｧ縺ｪ縺・↑繧峨∵椌蛻医ｊ繧偵＠縺ｦ縺・ｋ縺ｮ縺ｧ縲√％繧後・豁｣遒ｺ縺ｪ蛟､縺ｧ縺ｯ縺ｪ縺・°繧峨・
			BOUND_UPPER縺ｨ縺・≧謇ｱ縺・↓縺吶ｋ縲・

			縺溘□縺励∵欠縺玲焔縺後↑縺・ｴ蜷医・縲∬ｩｰ縺ｾ縺輔ｌ縺ｦ縺・ｋ繧ｹ繧ｳ繧｢縺ｪ縺ｮ縺ｧ縲√％繧後ｈ繧顔洒縺・髟ｷ縺・焔鬆・・
			隧ｰ縺ｿ縺後≠繧九°繧ら衍繧後↑縺・°繧峨√☆縺ｪ繧上■縲√せ繧ｳ繧｢縺ｯ螟牙虚縺吶ｋ縺九ｂ遏･繧後↑縺・・縺ｧ縲・
			BOUND_UPPER縺ｨ縺・≧謇ｱ縺・ｒ縺吶ｋ縲・
	*/

    if (!excludedMove && !(rootNode && pvIdx))
        ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
                       bestValue >= beta    ? BOUND_LOWER
                       : PvNode && bestMove ? BOUND_EXACT
                                            : BOUND_UPPER,
                       moveCount != 0 ? depth : std::min(MAX_PLY - 1, depth + 6), bestMove,
                       unadjustedStaticEval, tt.generation());

    // Adjust correction history if the best move is not a capture
    // and the error direction matches whether we are above/below bounds.
	// 譛蝟・焔縺後く繝｣繝励メ繝｣・磯ｧ貞叙繧奇ｼ峨〒縺ｪ縺・ｴ蜷医・
	// 縺九▽隱､蟾ｮ縺ｮ譁ｹ蜷代′荳企剞・丈ｸ矩剞繧定ｶ・∴縺ｦ縺・ｋ縺九←縺・°縺ｨ荳閾ｴ縺吶ｋ蝣ｴ蜷医↓縺ｯ縲・
	// 陬懈ｭ｣螻･豁ｴ(correction history)繧定ｪｿ謨ｴ縺吶ｋ縲・

	if (!ss->inCheck && !(bestMove && pos.capture(bestMove))
        && (bestValue > ss->staticEval) == bool(bestMove))
    {
        auto bonus =
          std::clamp(int(bestValue - ss->staticEval) * depth
                       * (bestMove ? SPSA::CORR_HIST_BONUS_BESTMOVE
                                   : SPSA::CORR_HIST_BONUS_NOBESTMOVE) / 128,
                     -CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);

        update_correction_history(pos, ss, *this, SPSA::CORR_HIST_BONUS_SCALE * bonus / 1024);
    }

	// 痩 qsearch()蜀・・譛ｫ蟆ｾ縺ｫ縺ゅｋassert縺ｮ譁・・隱ｬ譏弱ｒ隱ｭ繧縺薙→縲・
	ASSERT_LV3(-VALUE_INFINITE < bestValue && bestValue < VALUE_INFINITE);

    return bestValue;
}


// Quiescence search function, which is called by the main search function with
// depth zero, or recursively with further decreasing depth. With depth <= 0, we
// "should" be using static eval only, but tactical moves may confuse the static eval.
// To fight this horizon effect, we implement this qsearch of tactical moves.
// See https://www.chessprogramming.org/Horizon_Effect
// and https://www.chessprogramming.org/Quiescence_Search

// 髱呎ｭ｢髢｢謨ｰ縲ゅ％繧後・繝｡繧､繝ｳ縺ｮ謗｢邏｢髢｢謨ｰ縺九ｉ豺ｱ縺・縺ｧ蜻ｼ縺ｳ蜃ｺ縺輔ｌ繧九°縲√＆繧峨↓豺ｱ縺輔ｒ貂帙ｉ縺励↑縺後ｉ蜀榊ｸｰ逧・↓蜻ｼ縺ｳ蜃ｺ縺輔ｌ繧九・
// depth <= 0 縺ｮ蝣ｴ蜷医∵悽譚･縺ｯ髱咏噪隧穂ｾ｡縺縺代ｒ菴ｿ縺・後∋縺阪阪□縺後∵姶陦鍋噪縺ｪ謇九′髱咏噪隧穂ｾ｡繧呈ヱ繧上○繧九％縺ｨ縺後≠繧九・
// 縺薙・蝨ｰ蟷ｳ邱壼柑譫懊↓蟇ｾ謚励☆繧九◆繧√∵姶陦鍋噪縺ｪ謇九□縺代ｒ蟇ｾ雎｡縺ｨ縺励◆縺薙・髱呎ｭ｢謗｢邏｢繧貞ｮ溯｣・＠縺ｦ縺・ｋ縲・
// 隧ｳ邏ｰ縺ｯ https://www.chessprogramming.org/Horizon_Effect
// 縺翫ｈ縺ｳ https://www.chessprogramming.org/Quiescence_Search 繧貞盾辣ｧ縲・

// 2026-09-11: qsearch 中かどうかを評価側に伝える。 P2QSearchThreatScale で
//   qsearch 中だけ threat の寄与を弱められるようにするため (既定 256 = 等倍 = 従来通り)。
//   Threads=1 運用なのでグローバルで足りる。 多スレッド化する場合は thread_local にすること。
// ★NPS + 潜在バグ修正: qsearch の出入りで毎回書かれるフラグ。
//   thread_local でないと Threads>=2 で 2 コアが同じキャッシュラインを奪い合い、
//   さらに P2QSearchThreatScale を既定 256 から変えたとき**別スレッドの qsearch 状態が漏れる**。
//   既定 256 では読み手 (p2nnue.cpp:4402/4429) の結果が値に依らないので bit-exact。
namespace Eval { namespace P2NNUE { thread_local bool g_in_qsearch = false; } }

template<NodeType nodeType, bool ReadTT>
Value Search::YaneuraOuWorker::qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth qsDepth) {
    if (P2Stat::enabled()) P2Stat::qs_nodes.fetch_add(1, std::memory_order_relaxed);
    // qsearch 中フラグ (ネストするので前の値を保存して戻す)
#if defined(P2_SRCH_BUNDLE_A)
    // 束 A (L5): P2QSearchThreatScale == 256 (既定) のときは、 読み手 (p2nnue.cpp) がすべて
    //   `g_in_qsearch ? g_qs_threat_scale : 256` の形なので結果が値に依らない。 そのとき
    //   (tlsNeedQsFlag = false) は出入りの読み書きを省く。 ガードは必ず元に戻すので qsearch の外では
    //   常に false で、 省いた go のあとで 256 以外に変えても状態は従来と同じになる。
    struct QsFlagGuard {
        bool       prev = false;
        const bool on;
        explicit QsFlagGuard(bool on_) :
            on(on_) {
            if (on)
            {
                prev                                    = ::YaneuraOu::Eval::P2NNUE::g_in_qsearch;
                ::YaneuraOu::Eval::P2NNUE::g_in_qsearch = true;
            }
        }
        ~QsFlagGuard() {
            if (on)
                ::YaneuraOu::Eval::P2NNUE::g_in_qsearch = prev;
        }
    } qs_flag_guard(tlsNeedQsFlag);
#else
    struct QsFlagGuard {
        bool prev;
        QsFlagGuard() : prev(::YaneuraOu::Eval::P2NNUE::g_in_qsearch) { ::YaneuraOu::Eval::P2NNUE::g_in_qsearch = true; }
        ~QsFlagGuard() { ::YaneuraOu::Eval::P2NNUE::g_in_qsearch = prev; }
    } qs_flag_guard;
#endif

    /*
		涛 繝√ぉ繧ｹ縺ｨ逡ｰ縺ｪ繧雁ｰ・｣九〒縺ｯ縲∵焔鬧偵′縺ゅｋ縺溘ａ縲∫視謇九ｒ辟｡譚｡莉ｶ縺ｧ蟒ｶ髟ｷ縺吶ｋ縺ｨ縺九↑繧翫・髟ｷ謇区焚縲∫視謇九′邯壹￥縺薙→縺後≠繧九・
			謇矩ｧ偵′隍・焚縺ゅｋ縺ｨ縲√◎縺ｮ邨・∩蜷医ｏ縺帙ｒ縺吶∋縺ｦ蟒ｶ髟ｷ縺励※縺励∪縺・％縺ｨ縺ｫ縺ｪ繧翫∫ｵ・∩蜷医ｏ縺帷・逋ｺ繧貞ｮｹ譏薙↓襍ｷ縺薙☆縲・
	
			縺薙・轤ｹ縺ｯ縲ヾtockfish繧貞盾閠・↓縺吶ｋ譎ゅ↓縲∝ｿ・★閠・・縺励↑縺代ｌ縺ｰ縺ｪ繧峨↑縺・・
	
			縺薙％縺ｧ縺ｯ縲∽ｻ･荳九・蟇ｾ遲悶ｒ縺吶ｋ縲・
			1. qsearch(髱呎ｭ｢謗｢邏｢)縺ｧ縺ｯcaptures(鬧偵ｒ蜿悶ｋ謖・＠謇・縺ｨchecks(邇区焔縺ｮ謖・＠謇・縺ｮ縺ｿ繧樽ovePicker縺ｧ逕滓・
			2. 邇区焔縺ｮ謖・＠謇九・縲‥epth縺轡EPTH_QS_CHECKS(== 0)縺ｮ譎ゅ□縺醍函謌舌・
			3. captures縺ｮ謖・＠謇九・縲‥epth縺轡EPTH_QS_RECAPTURES(== -5)莉･荳九↑繧峨∫峩蜑阪↓鬧偵′遘ｻ蜍輔＠縺溷合縺ｫ遘ｻ蜍輔☆繧議apture縺ｮ謇九□縺代ｒ逕滓・縲・蜿悶ｊ霑斐☆謇・
			4. capture縺ｧ繧よｭｩ謳堺ｻ･荳翫・謳阪ｒ縺吶ｋ謖・＠謇九・蟒ｶ髟ｷ縺励↑縺・・
			5. 騾｣邯夂視謇九・蜊・律謇九・讀懷・縺吶ｋ
			縺薙ｌ繧峨↓繧医▲縺ｦ縲∫視謇九Λ繝・す繝･繧・｣邯夂視謇九〒霑ｽ縺・屓縺励※蜊・律謇・螳滄圀縺ｯ蜿榊援雋縺・縺ｫ閾ｳ繧区焔鬆・ｒ謗帝勁縺励※縺・ｋ縲・
	
			縺溘□縺励∫ｽｮ謠幄｡ｨ縺ｮ謖・＠謇九↓髢｢縺励※縺ｯdepth < DEPTH_QS_CHECKS 縺ｧ繧ら視謇九・謖・＠謇九′莠､縺倥ｋ縺ｮ縺ｧ縲・
			鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇九・縺ｿ縺ｧ蠕ｪ迺ｰ縺吶ｋ繧医≧縺ｪ蝣ｴ蜷医∵爾邏｢縺檎ｵゅｏ繧峨↑縺上↑繧九・
	
			縺昴％縺ｧ縲・
			6. depth < -16 縺ｪ繧峨∫ｽｮ謠幄｡ｨ縺ｮ謖・＠謇九ｒ辟｡隕悶☆繧・
			縺ｮ繧医≧縺ｪ蟇ｾ遲悶′蠢・ｦ√□縺ｨ諤昴≧縲・
			 竊偵蠑輔″蛻・￠謇ｱ縺・☆繧九％縺ｨ縺ｫ縺励◆縲・
	*/

    // -----------------------
    //     螟画焚螳｣險
    // -----------------------

    // PV node縺ｧ縺ゅｋ縺九・
    // 統 縺薙％縺軍oot node縺ｧ縺ゅｋ縺薙→縺ｯ縺ｪ縺・・縺ｧ縲√◎縺ｮ繧ｱ繝ｼ繧ｹ縺ｯ閠・∴縺ｪ縺上※濶ｯ縺・・

    static_assert(nodeType != Root);
    constexpr bool PvNode = nodeType == PV;

#if defined(EVAL_P2NNUE)
    // lazy-head: 髱呎ｭ｢謗｢邏｢縺ｮ髱霸V繝弱・繝峨〒縺ｯ head 繧堤怐逡･(rust 蜷梧婿驥・縲・
#if defined(P2_SRCH_BUNDLE_A)
    // 束 A (L5): 評価関数がこの TLS を読まない設定 (tlsNeed* = false) なら書き込みを省く。
    //   判定と根拠は p2nnue.cpp の p2_srch_tls_needs()。 go の開始時に決め、 探索中は不変。
    if (tlsNeedLazyHead)
        Eval::p2_lazy_head = !PvNode;
    if (tlsNeedWin)
    {
        Eval::p2_win_a = (int) alpha;
        Eval::p2_win_b = (int) beta;
    }
#if defined(P2_SRCH_BUNDLE_A_VERIFY)
    bundle_a_verify_tls();
#endif
#else
    Eval::p2_lazy_head = !PvNode;
    Eval::p2_win_a = (int)alpha; Eval::p2_win_b = (int)beta;
#endif
#endif

    ASSERT_LV3(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    ASSERT_LV3(PvNode || (alpha == beta - 1));

    // ､・Stockfish縺ｧ縺ｯ縺薙％縺ｧ荳願ｨ倥・繧医≧縺ｫ蜊・律謇九↓遯∝・縺ｧ縺阪ｋ縺九・繝√ぉ繝・け縺後≠繧九ｈ縺・□縺・
    //     蟆・｣九〒縺薙ｌ繧偵ｄ縺｣縺ｦ繧ょｼｷ縺上↑繧峨↑縺・・縺ｧ蟆主・縺励↑縺・・
#if STOCKFISH
    // Check if we have an upcoming move that draws by repetition
    // 蜿榊ｾｩ縺ｫ繧医ｋ蠑輔″蛻・￠縺ｨ縺ｪ繧句庄閭ｽ諤ｧ縺ｮ縺ゅｋ谺｡縺ｮ謇九′縺ゅｋ縺九ｒ遒ｺ隱阪☆繧・

    // 庁 縺薙・繧ｳ繝ｼ繝峨・蜴溽炊縺ｨ縺励※縺ｯ縲∵ｬ｡縺ｮ荳謇九〒蜊・律謇句ｱ髱｢縺ｫ謖√■霎ｼ繧√ｋ縺ｪ繧峨・
    //     蟆代↑縺上→繧ゅ％縺ｮ螻髱｢縺ｯ蠑輔″蛻・￠縺ｧ縺ゅｋ縺九ｉ縲・
    //     beta縺悟ｼ輔″蛻・￠縺ｮ繧ｹ繧ｳ繧｢繧医ｊ菴弱＞縺ｪ繧叡eta cut縺ｧ縺阪ｋ縺ｨ縺・≧繧ゅ・縲・

    if (alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply))
    {
        alpha = value_draw(nodes);
        if (alpha >= beta)
            return alpha;
    }
#endif

    // PV豎ゅａ繧狗畑縺ｮbuffer
    // 庁 縺薙ｌnonPV縺ｧ縺ｯ菴ｿ繧上↑縺・・縺ｧ縲∝盾辣ｧ縺励※縺翫ｉ縺壼炎髯､縺輔ｌ繧九・
    PVMoves pv;

    // make_move()縺ｮ縺ｨ縺阪↓蠢・ｦ・
    StateInfo st;

    // 縺薙・螻髱｢縺ｮhash key
	Key posKey;

    // move				: MovePicker縺九ｉ繧ゅｉ縺｣縺溽樟蝨ｨ縺ｮ謖・＠謇・
    // bestMove			: 縺薙・螻髱｢縺ｧ縺ｮ繝吶せ繝医↑謖・＠謇・
    Move move, bestMove;

    // bestValue		: best move縺ｫ蟇ｾ縺吶ｋ謗｢邏｢繧ｹ繧ｳ繧｢(alpha縺ｨ縺ｯ逡ｰ縺ｪ繧・
    // value			: 迴ｾ蝨ｨ縺ｮmove縺ｫ蟇ｾ縺吶ｋ謗｢邏｢繧ｹ繧ｳ繧｢
    // futilityBase		: futility pruning縺ｮ蝓ｺ貅悶→縺ｪ繧句､
    Value bestValue, value, futilityBase;

    // pvHit			: 鄂ｮ謠幄｡ｨ縺九ｉ蜿悶ｊ蜃ｺ縺励◆謖・＠謇九′縲￣V node縺ｧsave縺輔ｌ縺溘ｂ縺ｮ縺ｧ縺ゅ▲縺溘・
    // givesCheck		: MovePicker縺九ｉ蜿悶ｊ蜃ｺ縺励◆謖・＠謇九〒邇区焔縺ｫ縺ｪ繧九°
    // capture          : 鬧偵ｒ謐慕佐縺吶ｋ謖・＠謇九°
    bool pvHit, givesCheck, capture;

    // 縺薙・node縺ｧ菴墓焔逶ｮ縺ｮ謖・＠謇九〒縺ゅｋ縺・
    int moveCount;

    // -----------------------
    // Step 1. Initialize node
    // Step 1. 繝弱・繝峨・蛻晄悄蛹・
    // -----------------------

    if (PvNode)
    {
        (ss + 1)->pv = &pv;
        ss->pv->clear();
    }

    bestMove                    = Move::none();
    ss->inCheck                 = pos.checkers();
    moveCount                   = 0;

#if defined(USE_CLASSIC_EVAL) && defined(USE_LAZY_EVALUATE)
    bool evaluated = false;
    auto evaluate  = [&](Position& pos) {
        evaluated = true;
        return this->evaluate(pos);
    };
    auto do_move = [&](Position& pos, Move move, StateInfo& st, bool givesCheck, Stack* ss) {
        if (!evaluated)
        {
            evaluated = true;
            Eval::evaluate_with_no_return(pos);
        }
        this->do_move(pos, move, st, givesCheck, ss);
    };
#endif

#if STOCKFISH
#else
    // 繧・・縺・ｉ邇区爾邏｢縺ｧ霑ｽ蜉縺励◆諤晁・お繝ｳ繧ｸ繝ｳ繧ｪ繝励す繝ｧ繝ｳ
    auto& search_options = main_manager()->search_options;
#endif

    // Used to send selDepth info to GUI (selDepth counts from 1, ply from 0)
    // selDepth諠・ｱ繧竪UI縺ｫ騾∽ｿ｡縺吶ｋ縺溘ａ縺ｫ菴ｿ逕ｨ縺励∪縺呻ｼ・elDepth縺ｯ1縺九ｉ繧ｫ繧ｦ繝ｳ繝医＠縲｝ly縺ｯ0縺九ｉ繧ｫ繧ｦ繝ｳ繝医＠縺ｾ縺呻ｼ峨・

    if (PvNode && selDepth < ss->ply + 1)
        selDepth = ss->ply + 1;

    // -----------------------
    // Step 2. Check for an immediate draw or maximum ply reached
    // Step 2. 蜊ｳ蠎ｧ縺ｫ蠑輔″蛻・￠縺ｫ縺ｪ繧九°縲∵怙螟ｧ縺ｮply(謇区焚)縺ｫ驕斐＠縺ｦ縺・↑縺・°繧堤｢ｺ隱阪＠縺ｾ縺吶・
    // -----------------------

    // 蜊・律謇九メ繧ｧ繝・け縺ｯ縲｀ovePicker縺ｧcaptures(鬧偵ｒ蜿悶ｋ謖・＠謇九＠縺狗函謌舌＠縺ｪ縺・縺ｪ繧峨・
    // 蜊・律謇九メ繧ｧ繝・け縺励↑縺・婿縺悟ｼｷ縺・ｈ縺・□縲・
    // 縺溘□縺励｀ovePicker縺ｧ縲ゝT縺ｮ謖・＠謇九↓蟇ｾ縺励※繧Ｄaptures縺ｧ縺ゅｋ縺ｨ縺・≧蛻ｶ髯舌ｒ縺九￠縺ｪ縺・→
    // TT縺ｮ謖・＠謇九□縺代〒辟｡髯舌Ν繝ｼ繝・MAX_PLY縺ｾ縺ｧ蜀榊ｸｰ逧・↓謗｢邏｢縺碁ｲ繧)縺ｫ縺ｪ繧雁ｼｱ縺上↑繧九・縺ｧ縲∵ｳｨ諢上′蠢・ｦ√・

#if STOCKFISH
    if (pos.is_draw(ss->ply) || ss->ply >= MAX_PLY)
        return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos) : VALUE_DRAW;

#else

    // 笞 Stockfish縺ｯis_draw()縺ｧ蜊・律謇句愛螳壹ｒ縺励※縺・ｋ縺後・
    //     繧・・縺・ｉ邇九〒縺ｯ蜉｣遲牙ｱ髱｢縺ｮ蛻､螳壹′縺ゅｋ縺ｮ縺ｧ is_repetition()縺ｧ蛻､螳壹＠縺ｪ縺上※縺ｯ縺ｪ繧峨↑縺・・

    // 決 繧・・縺・ｉ邇狗峡閾ｪ謾ｹ濶ｯ 決

    // 迴ｾ螻髱｢縺ｮ謇狗分蛛ｴ縺ｮColor
    Color us = pos.side_to_move();

    auto draw_type = pos.is_repetition(ss->ply, true);
    if (draw_type != REPETITION_NONE)
    /*
			涛 縺ｪ縺忻alue_from_tt()縺悟ｿ・ｦ√↑縺ｮ縺具ｼ・

			draw_value()縺ｧ縺ｯ縲∝━遲牙ｱ髱｢縲∝乾遲牙ｱ髱｢縺ｮ譎ゅ↓VALUE_MATE , -VALUE_MATE縺瑚ｿ斐▲縺ｦ縺上ｋ縲・
			縺薙ｌ縺ｯ1謇玖ｩｰ繧√・繧ｹ繧ｳ繧｢縺ｨ蜷後§諢丞袖縺ｧ縺ゅｋ縲・

			縺薙ｌ縺ｯ迴ｾ蝨ｨ縺ｮ謇区焚縺ｫ貅悶§縺溘せ繧ｳ繧｢縺ｫ螟画鋤縺吶ｋ蠢・ｦ√′縺ゅｋ縲・
			縺吶↑繧上■縲［ate_in(VALUE_MATE, ss->ply), mated_in(VALUE_MATE, ss->ply)縺ｨ蜷後§繧ｹ繧ｳ繧｢縺ｫ
			縺ｪ繧九ｈ縺・↓螟画鋤縺励↑縺代ｌ縺ｰ縺ｪ繧峨↑縺・・
			
			縺昴・縺溘ａ縲」alue_from_tt()縺悟ｿ・ｦ√↑縺ｮ縺ｧ縺ゅｋ縲・
	*/
    {
        if (draw_type == REPETITION_DRAW)
            // 騾壼ｸｸ縺ｮ蜊・律謇九・譎ゅ・繧・ｉ縺弱ｒ謖√◆縺帙ｋ縲・
            // 庁 蠑輔″蛻・￠縺ｮ繧ｹ繧ｳ繧｢v縺ｯ abs(vﾂｱ1) <= VALUE_MAX_EVAL縺ｧ縺ゅｋ縺薙→縺御ｿ晁ｨｼ縺輔ｌ縺ｦ縺・ｋ縺ｮ縺ｧ縲・
            //     value_from_tt()縺ｧ縺ｮ螟画鋤縺ｯ荳崎ｦ√・
            return draw_value(draw_type, pos.side_to_move()) + value_draw(nodes);
		else
	        return value_from_tt(draw_value(draw_type, us), ss->ply);
    }

// TODO : 縺ゅ→縺ｧ讀懆ｨ弱☆繧九・
#if 0
	// 16謇倶ｻ･蜀・・蠕ｪ迺ｰ縺ｫ縺ｪ縺｣縺ｦ縺ｪ縺・・縺ｫqsearch縺ｧ16謇九ｂ蟒ｶ髟ｷ縺励※縺・ｋ蝣ｴ蜷医・
    // 鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇九□縺代〒髟ｷ縺・ｾｪ迺ｰ縺ｫ縺ｪ縺｣縺ｦ縺・ｋ蜿ｯ閭ｽ諤ｧ縺碁ｫ倥￥縲・
    // 縺薙ｌ縺ｯ蠑輔″蛻・￠謇ｱ縺・↓縺励※縺励∪縺・・
    if (depth <= -16)
        return draw_value(REPETITION_DRAW, us);
#endif

    // 譛螟ｧ謇区焚縺ｮ蛻ｰ驕・
    if (ss->ply >= MAX_PLY || pos.game_ply() > search_options.max_moves_to_draw)
        return draw_value(REPETITION_DRAW, us) + value_draw(nodes);

    ASSERT_LV3(0 <= ss->ply && ss->ply < MAX_PLY);

#endif

    // -----------------------
    // Step 3. Transposition table lookup
    // Step 3. 鄂ｮ謠幄｡ｨ縺ｮlookup
    // -----------------------

    posKey                         = pos.key();
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey, pos);

#if !STOCKFISH
    if constexpr (!ReadTT)
        ttHit = false;
#endif

    // Need further processing of the saved data
    // 菫晏ｭ倥＆繧後◆繝・・繧ｿ縺ｮ縺輔ｉ縺ｪ繧句・逅・′蠢・ｦ√〒縺・

    ss->ttHit   = ttHit;
    ttData.move = ttHit ? ttData.move : Move::none();

    ttData.value =
#if STOCKFISH
      ttHit ? value_from_tt(ttData.value, ss->ply , pos.rule50_count()) : VALUE_NONE;
#else
      ttHit ? value_from_tt(ttData.value, ss->ply ) : VALUE_NONE;
#endif      
    pvHit = ttHit && ttData.is_pv;

    // 東 繧・・縺・ｉ邇九〒縺ｯ鄂ｮ謠幄｡ｨ縺ｫ蜈亥ｾ碁俣驕輔∴縺ｦ譖ｸ縺榊・縺吶ヰ繧ｰ繧堤函縺倥≧繧九・縺ｧ縲√％縺ｮassert霑ｽ蜉縺吶ｋ縲・
    ASSERT_LV3(pos.legal_promote(ttData.move));

    // At non-PV nodes we check for an early TT cutoff
    // non-PV node縺ｫ縺翫＞縺ｦ縲∫ｽｮ謠幄｡ｨ縺ｫ繧医ｋ譌ｩ譛滓椌蛻医ｊ繧偵メ繧ｧ繝・け縺励∪縺・

    /*
		涛 nonPV縺ｧ縺ｯ鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇九〒譫晏・繧翫☆繧・
		    PV縺ｧ縺ｯ鄂ｮ謠幄｡ｨ縺ｮ謖・＠謇九〒縺ｯ譫晏・繧翫＠縺ｪ縺・蜑榊屓evaluate縺励◆蛟､縺ｯ菴ｿ縺医ｋ)
	*/

    if (!PvNode && ttData.depth >= DEPTH_QS
        && is_valid(ttData.value)  // Can happen when !ttHit or when access race in probe()
        // 鄂ｮ謠幄｡ｨ縺九ｉ蜿悶ｊ蜃ｺ縺励◆縺ｨ縺阪↓莉悶せ繝ｬ繝・ラ縺悟､繧呈ｽｰ縺励※縺・ｋ蜿ｯ閭ｽ諤ｧ縺後≠繧翫≧繧・
        && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER)))
        /*
				庁 竊代％縺薙・縲≫・縺薙・諢丞袖縲・
				&& (ttData.value >= beta ? (ttData.bound & BOUND_LOWER)
										 : (ttData.bound & BOUND_UPPER)))
			*/
        return ttData.value;
    /*
			涛 ttData.value縺御ｸ狗阜(逵溘・隧穂ｾ｡蛟､縺ｯ縺薙ｌ繧医ｊ螟ｧ縺阪＞)繧ゅ＠縺上・繧ｸ繝｣繧ｹ繝医↑蛟､縺ｧ縲・
			    縺九▽ttData.value >= beta雜・∴縺ｪ繧叡eta cut縺ｧ縺阪ｋ縲・

				ttData.value縺御ｸ顔阜(逵溘・隧穂ｾ｡蛟､縺ｯ縺薙ｌ繧医ｊ蟆上＆縺・縺縺後・
				tte->depth()縺ｮ縺ｻ縺・′depth繧医ｊ豺ｱ縺代ｌ縺ｰ
				繧医ｊ豺ｱ縺・爾邏｢邨先棡縺ｨ縺励※蜊∝・菫｡鬆ｼ縺ｧ縺阪ｋ縺ｮ縺ｧ縲・
				縺薙・蛟､繧剃ｿ｡鬆ｼ縺励※縲√％縺ｮ蛟､縺ｧreturn縺励※濶ｯ縺・・
		*/

    // -----------------------
    // Step 4. Static evaluation of the position
    // Step 4. 縺薙・螻髱｢縺ｮ髱呎ｭ｢隧穂ｾ｡
    // -----------------------

    Value unadjustedStaticEval = VALUE_NONE;
    if (ss->inCheck)
    {
        /*
			涛	bestValue縺ｯalpha縺ｨ縺ｯ驕輔≧縲・
				邇区焔縺後°縺九▲縺ｦ縺・ｋ縺ｨ縺阪・-VALUE_INFINITE繧貞・譛溷､縺ｨ縺励※縲・
				縺吶∋縺ｦ縺ｮ謖・＠謇九ｒ逕滓・縺励※縺薙ｌ繧剃ｸ雁屓繧九ｂ縺ｮ繧呈爾縺吶・縺ｧ
				alpha縺ｨ縺ｯ蛹ｺ蛻･縺励↑縺代ｌ縺ｰ縺ｪ繧峨↑縺・・
		*/
        bestValue = futilityBase = -VALUE_INFINITE;
	}
    else
    {
        const auto correctionValue = correction_value(*this, pos, ss);
#if defined(P2_KHGATE)
        // KHGATE PESS: qsearch の stand pat にも同じ悲観補正を入れる (葉の値に効かせるため)。 PESS が全て 0 なら数えない。
        //   SELEXT は qsearch には無い (LMR / futility / NMP が無い) ので数えない。
        const int khPess = kh_pess_on() ? kh_gate_eval(pos, true, false).pess : 0;
#endif

		if (ss->ttHit)
        {
            // Never assume anything about values stored in TT
            // TT・育ｽｮ謠幄｡ｨ・峨↓菫晏ｭ倥＆繧後※縺・ｋ蛟､縺ｫ縺､縺・※縺ｯ縲∵ｱｺ縺励※菴輔ｂ莉ｮ螳壹＠縺ｪ縺・％縺ｨ縲・

			// 統 鄂ｮ謠幄｡ｨ縺ｫ隧穂ｾ｡蛟､縺梧ｼ邏阪＆繧後※縺・ｋ縺ｨ縺ｯ髯舌ｉ縺ｪ縺・・縺ｧ縺昴・蝣ｴ蜷医・隧穂ｾ｡髢｢謨ｰ縺ｮ蜻ｼ縺ｳ蜃ｺ縺励′蠢・ｦ√・
            //     bestValue縺ｮ蛻晄悄蛟､縺ｨ縺励※縺薙・螻髱｢縺ｮevaluate()縺ｮ蛟､繧剃ｽｿ縺・ゅ％繧後ｒ荳雁屓繧区欠縺玲焔縺後≠繧九・縺壹↑縺ｮ縺縺・.縲・

            unadjustedStaticEval = ttData.eval;
            if (!is_valid(unadjustedStaticEval))
                unadjustedStaticEval = evaluate(pos);
#if defined(USE_CLASSIC_EVAL)
			else if (PvNode) {
				// 決 繧・・縺・ｉ邇狗峡閾ｪ
				unadjustedStaticEval = evaluate(pos);
#ifdef P2_PRUNE_STATS   // 🚨P2PruneStat はこの中にある (本番ビルドでは消える)
			// 🎯計測用: この ply の bucket を覚える (跨ぎ判定の親側になる)。
			//   🚨evaluate した直後なので StateInfo に計算済み = 読んでも bit-exact を崩さない。
			P2PruneStat::ply_bucket[ss->ply] = Eval::P2NNUE::progress_bucket_if_computed(pos);
#endif
				// 竍ｨ NNUE縺縺ｨ縺薙ｌ蜈･繧後◆縺ｻ縺・′蠑ｷ縺・庄閭ｽ諤ｧ縺娯ｦ縲・
			}
#endif

			if (!PvNode && is_valid(ttData.eval) && unadjustedStaticEval == ttData.eval) P2SMP_CHECK_R(pos, ttData.eval);
			ss->staticEval = bestValue =
				to_corrected_static_eval(P2_OPT_ADD(P2_KHGATE_QPESS(unadjustedStaticEval)), correctionValue);

			// ttValue can be used as a better position evaluation
            // ttValue縺ｯ縲√ｈ繧願憶縺・ｱ髱｢隧穂ｾ｡縺ｨ縺励※菴ｿ逕ｨ縺ｧ縺阪ｋ

			/*
				涛 鄂ｮ謠幄｡ｨ縺ｫ譬ｼ邏阪＆繧後※縺・◆繧ｹ繧ｳ繧｢縺ｯ縲√％縺ｮ螻髱｢縺ｧ莉雁屓謗｢邏｢縺吶ｋ繧ゅ・縺ｨ蜷檎ｭ峨°蟆代＠縺縺大乾繧九＄繧峨＞縺ｮ
				    邊ｾ蠎ｦ縺ｧ謗｢邏｢縺輔ｌ縺溘ｂ縺ｮ縺ｧ縺ゅｋ縺ｪ繧峨√◎繧後ｒbestValue縺ｮ蛻晄悄蛟､縺ｨ縺励※菴ｿ縺・・
			
				    縺溘□縺励［ate value縺ｯ螟画峩縺励↑縺・婿縺瑚憶縺・・縺ｧ縲・is_decisive(ttData.value) 縺ｯ縲・
				    縺昴・縺溘ａ縺ｮ譚｡莉ｶ縲・
			*/

            if (is_valid(ttData.value) && !is_decisive(ttData.value)
                && (ttData.bound & (ttData.value > bestValue ? BOUND_LOWER : BOUND_UPPER)))
                bestValue = ttData.value;
        }
        else
        {
            // -----------------------
            //  決 荳謇玖ｩｰ繧∝愛螳・
            // -----------------------

            // 鄂ｮ謠幄｡ｨ縺ｫhit縺励◆蝣ｴ蜷医・縲√☆縺ｧ縺ｫ隧ｰ縺ｿ繧定ｪｿ縺ｹ縺溘・縺壹↑縺ｮ縺ｧ
            // 鄂ｮ謠幄｡ｨ縺ｫhit縺励↑縺九▲縺溘→縺阪↓縺ｮ縺ｿ隱ｿ縺ｹ繧九・

			ASSERT_LV3(!ss->inCheck && !ss->ttHit);
            // 決 qsearch縺ｮ1謇玖ｩｰ繧∝愛螳壹・rust(alpha_beta.rs縺ｮ譌｢蟄碗search mate_1ply邨瑚ｷｯ)縺ｫ繧ゅ≠繧九・縺ｧ譛牙柑縺ｮ縺ｾ縺ｾ縲・
            if (true)
            {
                // 笆 蛯呵・
                //
                // 竍ｨ 縺薙・qsearch()縺ｧ縺ｮ1謇玖ｩｰ繧√メ繧ｧ繝・け縺ｯ縲√＞縺ｾ縺ｮ縺ｨ縺薙ｍ縲∝・繧後◆縺ｻ縺・′濶ｯ縺・ｈ縺・□縲・
                //    play_time = b1000 ,  1631 - 55 - 1314(55.38% R37.54) [2016/08/19]
                //    play_time = b6000 ,  538 - 23 - 439(55.07% R35.33) [2016/08/19]

                // 1謇玖ｩｰ繧√↑縺ｮ縺ｧ縺薙・谺｡縺ｮnode縺ｧ(謖・＠謇九′縺ｪ縺上↑縺｣縺ｦ)隧ｰ繧縺ｨ縺・≧隗｣驥・
                move = Mate::mate_1ply(pos);
                if (move != Move::none())
                {
                    bestValue = mate_in(ss->ply + 1);

                    if (true)
                    {
                        // ､・縺薙・node縺ｫ蜀崎ｨｪ蝠上☆繧九％縺ｨ縺ｯ縺ｾ縺壹↑縺・□繧阪≧縺九ｉ縲・
						//     鄂ｮ謠幄｡ｨ縺ｫ菫晏ｭ倥☆繧倶ｾ｡蛟､縺ｯ縺ｪ縺・庄閭ｽ諤ｧ縺後≠繧九′縲・
						// 庁 [2025/10/6] SPSA縺励◆邨先棡縲√％縺ｮ蜃ｦ逅・・蜈･繧後◆縺ｻ縺・′濶ｯ縺・→縺・≧縺薙→縺後ｏ縺九▲縺溘・

						ttWriter.write(posKey, bestValue, ss->ttPv, BOUND_EXACT, DEPTH_QS, move,
                                       unadjustedStaticEval, tt.generation());

                    }

                    return bestValue;
                }
            }

			// 東 縺薙％縺九ｉStockfish縺ｮ蜈・・繧ｳ繝ｼ繝・東

            unadjustedStaticEval = evaluate(pos);

            ss->staticEval = bestValue =
				to_corrected_static_eval(P2_OPT_ADD(P2_KHGATE_QPESS(unadjustedStaticEval)), correctionValue);

#if 0       // 莉･蜑阪・繧ｳ繝ｼ繝・
            unadjustedStaticEval =
              (ss - 1)->currentMove != Move::null()
                ? evaluate(pos)
                : -(ss - 1)->staticEval;
#endif

        }

        // Stand pat. Return immediately if static value is at least beta
        // Stand pat縲る撕逧・ｩ穂ｾ｡蛟､縺悟ｰ代↑縺上→繧ゅ・繝ｼ繧ｿ蛟､縺ｫ驕斐＠縺ｦ縺・ｋ蝣ｴ蜷医・逶ｴ縺｡縺ｫ霑斐＠縺ｾ縺・

		/*
			涛 迴ｾ蝨ｨ縺ｮbestValue縺ｯ縲√％縺ｮ螻髱｢縺ｧ菴輔ｂ謖・＆縺ｪ縺・→縺阪・繧ｹ繧ｳ繧｢縲・
			    recapture縺吶ｋ縺ｨ謳阪ｒ縺吶ｋ螟牙喧繧ゅ≠繧九・縺ｧ縺薙・繧ｹ繧ｳ繧｢繧貞渕貅悶↓閠・∴繧九・

				邇区焔縺後°縺九▲縺ｦ縺・↑縺・こ繝ｼ繧ｹ縺ｫ縺翫＞縺ｦ縺ｯ縲√％縺ｮ譎らせ縺ｧ縺ｮ髱咏噪縺ｪeval縺ｮ蛟､縺・
				beta繧剃ｸ雁屓繧翫◎縺・↑繧峨％縺ｮ譎らせ縺ｧ蟶ｰ繧九・
		*/

        if (bestValue >= beta)
        {
            if (!is_decisive(bestValue))
                // bestValue繧貞ｰ代＠beta縺ｮ縺ｻ縺・↓蟇・○繧九・
                bestValue = (bestValue + beta) / 2;

            if (!ss->ttHit)
                ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), false, BOUND_LOWER,
                               DEPTH_UNSEARCHED, Move::none(), unadjustedStaticEval,
                               tt.generation());
            return bestValue;
        }

		/*
			涛 邇区焔縺後°縺九▲縺ｦ縺・↑縺上※PvNode縺ｧ縺九▽縲｜estValue縺径lpha繧医ｊ
				螟ｧ縺阪＞縺ｪ繧峨◎繧後ｒalpha縺ｮ蛻晄悄蛟､縺ｫ菴ｿ縺・・
				邇区焔縺後°縺九▲縺ｦ縺・ｋ縺ｪ繧牙・驛ｨ縺ｮ謖・＠謇九ｒ隱ｿ縺ｹ縺溘⊇縺・′縺・＞縲・
		*/ 
		if (bestValue > alpha)
            alpha = bestValue;

        // 庁 futility縺ｮ蝓ｺ貅悶→縺ｪ繧句､繧鍛estValue縺ｫmargin蛟､繧貞刈邂励＠縺溘ｂ縺ｮ縺ｨ縺励※縲・
        //     縺薙ｌ繧剃ｸ句屓繧九ｈ縺・〒縺ゅｌ縺ｰ譫晏・繧翫☆繧九・

        futilityBase = ss->staticEval + SPSA::QSEARCH_FUTILITY_BASE_MARGIN;

    }

	// -----------------------
    //     1謇九★縺､隱ｿ縺ｹ繧・
    // -----------------------

    const PieceToHistory* contHist[] = {(ss - 1)->continuationHistory};

	// 涛 蜿悶ｊ蜷医＞縺ｮ謖・＠謇九□縺醍函謌舌☆繧・
    //     search縺九ｉ蜻ｼ縺ｳ蜃ｺ縺輔ｌ縺溷ｴ蜷医∫峩蜑阪・謖・＠謇九′Move::null()縺ｧ縺ゅｋ縺薙→縺後≠繧翫≧繧九・
	//     縺薙・蝣ｴ蜷医ヾQ_NONE繧定ｨｭ螳壹☆繧九・

    Square prevSq = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;

    // Initialize a MovePicker object for the current position, and prepare to search
    // the moves. We presently use two stages of move generator in quiescence search:
    // captures, or evasions only when in check.

	// 迴ｾ蝨ｨ縺ｮ螻髱｢縺ｫ蟇ｾ縺励※ MovePicker 繧ｪ繝悶ず繧ｧ繧ｯ繝医ｒ蛻晄悄蛹悶＠縲∵欠縺玲焔縺ｮ謗｢邏｢繧呈ｺ門ｙ縺励∪縺吶・
    // 迴ｾ蝨ｨ縲・撕豁｢謗｢邏｢縺ｧ縺ｯ2谿ｵ髫弱・謖・＠謇狗函謌舌ｒ菴ｿ逕ｨ縺励※縺・∪縺呻ｼ・
    // captures(鬧偵ｒ蜿悶ｋ謖・＠謇・縲√∪縺溘・evasions(邇区焔縺ｮ蝗樣∩)縺ｮ縺ｿ縺ｧ縺吶・

    MovePicker mp(pos, ttData.move, qsDepth, &mainHistory, &lowPlyHistory, &captureHistory,
                  contHist, &sharedHistory, ss->ply,
                  nullptr, -1,  // qsearch 縺ｧ縺ｯ drop quiet 縺悟ｸ悟ｰ代↑縺溘ａ Drop Continuation History 縺ｯ荳堺ｽｿ逕ｨ
                  nullptr, 0    // qsearch 縺ｯ Threat History 蟇ｾ雎｡螟・繧ｹ繝壹ャ繧ｯ貅匁侠)
#if !STOCKFISH
                  ,
                  search_options.generate_all_legal_moves
#endif
    );

	// -----------------------
    // Step 5. Loop through all pseudo-legal moves until no moves remain or a beta cutoff occurs.
    // Step 5. 逍台ｼｼ蜷域ｳ墓焔繧偵☆縺ｹ縺ｦ繝ｫ繝ｼ繝怜・逅・＠縲∵焔縺梧ｮ九ｉ縺ｪ縺上↑繧九°縲√・繝ｼ繧ｿ繧ｫ繝・ヨ繧ｪ繝輔′逋ｺ逕溘☆繧九∪縺ｧ邯壹￠縺ｾ縺吶・
    // -----------------------

    while ((move = mp.next_move()) != Move::none())
    {
        //assert(move.is_ok());

		// ､・MovePicker縺ｧ逕滓・縺輔ｌ縺滓欠縺玲焔縺ｯpseudo_legal縺ｧ縺ゅｋ縺ｯ縺壹・
        ASSERT_LV3(pos.pseudo_legal_s<true>(move) && pos.legal_promote(move));

		/*
			蜷域ｳ墓焔縺九←縺・°縺ｮ繝√ぉ繝・け
		
			涛 謖・＠謇九・蜷域ｳ墓ｧ縺ｮ蛻､螳壹・逶ｴ蜑阪∪縺ｧ驕・ｻｶ縺輔○縺溘⊇縺・′蠕励□縺ｨ諤昴ｏ繧後※縺・◆縺ｮ縺縺・
			   (縺薙ｌ縺碁撼蜷域ｳ墓焔縺ｧ縺ゅｋ蜿ｯ閭ｽ諤ｧ縺ｯ縺九↑繧贋ｽ弱＞縺ｮ縺ｧ莉悶・蛻､螳壹↓繧医ｊskip縺輔ｌ縺溘⊇縺・′蠕・
			   Stockfish14縺九ｉ縲・撕豁｢謗｢邏｢縺ｧ繧ゅ∵掠縺・ｮｵ髫弱〒legal()繧貞他縺ｳ蜃ｺ縺吶ｈ縺・↓縺ｪ縺｣縺溘・
		*/

        if (!pos.legal(move))
            continue;

		//  螻髱｢繧帝ｲ繧√ｋ蜑阪・譫晏・繧・

        givesCheck = pos.gives_check(move);
        capture    = pos.capture_stage(move);

        moveCount++;

		// -----------------------
        // Step 6. Pruning
        // Step 6. 譫晏・繧・
        // -----------------------

		/*
			涛 move縺檎視謇九↓縺ｪ繧峨↑縺・欠縺玲焔縺ｧ縺ゅｊ縲・謇句燕縺ｧ逶ｸ謇九′遘ｻ蜍輔＠縺滄ｧ偵ｒ
				蜿悶ｊ霑斐☆謖・＠謇九〒繧ゅ↑縺上∽ｻ雁屓謐慕佐縺輔ｌ繧九〒縺ゅｍ縺・ｧ偵↓繧医ｋ隧穂ｾ｡蛟､縺ｮ荳頑・蛻・ｒ
			 蜉邂励＠縺ｦ繧Ｂlpha蛟､繧定ｶ・∴縺昴≧縺ｫ縺ｪ縺・↑繧峨％縺ｮ謖・＠謇九・譫晏・繧翫＠縺ｦ縺励∪縺・・
		*/

		if (!is_loss(bestValue))
        {
            // Futility pruning and moveCount pruning
            // futility譫晏・繧翫→move count縺ｫ蝓ｺ縺･縺乗椌蛻医ｊ

            if (!givesCheck && move.to_sq() != prevSq && !is_loss(futilityBase)
#if STOCKFISH            
				// TODO : 縺薙％縺ｮ譛驕ｩ蛹悶｛ptimizer縺ｫ莉ｻ縺帙ｋ縺具ｼ・
                && move.type_of() != PROMOTION
				// 統 縺薙・譛蠕後・譚｡莉ｶ縲∝・繧後◆縺ｻ縺・′縺・＞縺ｮ縺具ｼ・
				// 投 蜈･繧後↑縺・婿縺瑚憶縺輔￡縲・V7.74taya-t50 VS V7.74taya-t51)
#endif                
				)
            {
				// 庁 MoveCount縺ｫ蝓ｺ縺･縺乗椌蛻医ｊ
				// move.to_sq() != prevSq 縺ｮ縺ｨ縺阪↑縺ｮ縺ｧ縲√▽縺ｾ繧翫・縲∫峩蜑阪↓蜍輔°縺輔ｌ縺滄ｧ偵ｒ蜿悶ｊ霑斐☆莉･螟悶・謖・＠謇九・
				// 縺薙ｌ繧定ｪｭ縺ｿ蟋九ａ繧九→邨・∩蜷医ｏ縺帷・逋ｺ縺励※縺励∪縺・・縺ｧ3謇狗岼莉･髯阪・隱ｿ縺ｹ縺ｪ縺・・譫晏・繧翫☆繧・

				if (moveCount > SPSA::QSEARCH_MOVE_COUNT_PRUNING)
                    continue;

				/*
                    ､・move縺梧・繧翫・謖・＠謇九↑繧峨√◎縺ｮ謌舌ｋ縺薙→縺ｫ繧医ｋ萓｡蛟､荳頑・蛻・ｂ
					    縺薙％縺ｫ荵励○縺溘⊇縺・′豁｣縺励＞隕狗ｩ阪ｊ縺ｫ縺ｪ繧九・縺壹・
					投 縲占ｨ域ｸｬ雉・侭 14.縲・futility pruning縺ｮ縺ｨ縺阪↓promote繧定・・縺吶ｋ縺九←縺・°縲・
				*/

                Value futilityValue = futilityBase +
                                    PieceValue[pos.piece_on(move.to_sq())];
                                // 笞縲縺薙ｌ縲∝刈邂励＠縺溽ｵ先棡縲《16縺ｫ蜿弱∪繧峨↑縺・庄閭ｽ諤ｧ縺後≠繧九′縲・
                                //      險育ｮ励・s32縺ｧ陦後▲縺ｦ縲√◎縺ｮ縺ゅ→縲√％縺ｮ蛟､繧堤畑縺・↑縺・°繧峨そ繝ｼ繝輔・

                // If static eval + value of piece we are going to capture is
                // much lower than alpha, we can prune this move.

				// 髱咏噪隧穂ｾ｡蛟､縺ｨ繧ｭ繝｣繝励メ繝｣縺吶ｋ鬧偵・萓｡蛟､繧貞粋繧上○縺溘ｂ縺ｮ縺径lpha繧医ｊ螟ｧ蟷・↓菴弱＞蝣ｴ蜷医・
                // 縺薙・謇九ｒ譫晏・繧翫☆繧九％縺ｨ縺後〒縺阪∪縺吶・

				// 庁 縺薙％縲√ｏ繧翫→譽句鴨縺ｫ蠖ｱ髻ｿ縺吶ｋ縲ゆｸ区焔縺ｪ縺薙→縺吶ｋ縺ｨR30縺舌ｉ縺・､峨ｏ繧九・

				if (futilityValue <= alpha)
                {
                    bestValue = std::max(bestValue, futilityValue);
                    continue;
                }

                // If static exchange evaluation is low enough
                // we can prune this move.

				// 髱咏噪莠､謠幄ｩ穂ｾ｡縺悟香蛻・↓菴弱＞蝣ｴ蜷医・
                // 縺薙・謇九ｒ譫晏・繧翫〒縺阪∪縺吶・

				// 庁 futilityBase縺ｯ縺薙・螻髱｢縺ｮeval縺ｫmargin蛟､繧貞刈邂励＠縺ｦ縺・ｋ縺ｮ縺縺後・
                //     縺昴ｌ縺径lpha繧定ｶ・∴縺ｪ縺・・縺ｯ縲∵が縺・焔縺繧阪≧縺九ｉ譫晏・繧翫＠縺ｦ縺励∪縺・・

				if (!pos.see_ge(move, alpha - futilityBase))
                {
                    bestValue = std::max(bestValue, std::min(alpha, futilityBase));
                    continue;
                }
            }

            // Skip non-captures
			// 鬧貞叙繧翫〒縺ｯ縺ｪ縺・欠縺玲焔縺ｮskip

			// 2026-10-17: SPSA::QS_CHECKS==1 のときは、 qsearch の**入口の 1 層**に限り
			//   「駒を取らない王手」も読む (MovePicker の QCHECK 段から来る)。
			//   既定 0 なら従来どおり駒取り以外を全部捨てる = bit-exact。
			if (!capture && !(SPSA::QS_CHECKS && qsDepth >= DEPTH_QS && givesCheck))
                continue;

            // Do not search moves with bad enough SEE values
            // SEE縺悟香蛻・が縺・欠縺玲焔縺ｯ謗｢邏｢縺励↑縺・・

			/*
				､斐辟｡鬧・↑邇区焔繝ｩ繝・す繝･縺ｿ縺溘＞縺ｪ縺ｮ繧呈椛蛻ｶ縺ｧ縺阪ｋ・・
					 縺薙ｌ-90縺縺ｨPawnValue == 90縺ｪ縺ｮ縺ｧ豁ｩ謳阪・險ｱ縺励※縺励∪縺・・

					 豁ｩ謳阪☆繧区欠縺玲焔縺ｯ蟒ｶ髟ｷ縺励↑縺・ｈ縺・↓縺吶ｋ縺ｻ縺・′縺・＞縺具ｼ・
					 capture縺ｮ譎ゅ・豁ｩ謳阪・縲∵ｭｩ縺ｧ蜿悶ｋ縲∝酔隗偵∝酔隗偵∩縺溘＞縺ｪ螻髱｢縺ｪ縺ｮ縺ｧ縺昴％縺ｫ縺ｯ縺ゅ∪繧頑э蜻ｳ縺ｪ縺輔￡縲・
			*/

            if (!pos.see_ge(move, SPSA::QSEARCH_SEE_PRUNING))
                continue;
        }

		// -----------------------
        // Step 7. Make and search the move
        // Step 7. 謖・＠謇九〒騾ｲ繧∵爾邏｢縺吶ｋ
        // -----------------------

		// 統 1謇句虚縺九＠縺ｦ縲∝・蟶ｰ逧・↓qsearch()繧貞他縺ｶ

        do_move(pos, move, st, givesCheck, ss);

        value = -qsearch<nodeType, ReadTT>(pos, ss + 1, -beta, -alpha, qsDepth - 1);
        undo_move(pos, move);

		ASSERT_LV3(-VALUE_INFINITE < value && value < VALUE_INFINITE);

        // -----------------------
		// Step 8. Check for a new best move
        // Step 8. 譁ｰ縺励＞best move繧偵メ繧ｧ繝・け縺吶ｋ
        // -----------------------

		// bestValue(竕誕lpha蛟､)繧呈峩譁ｰ縺吶ｋ縺ｮ縺・

        if (value > bestValue)
        {
            bestValue = value;

            if (value > alpha)
            {
                bestMove = move;

                if (PvNode)  // Update pv even in fail-high case
							 // fail-high縺ｮ蝣ｴ蜷医ｂPV縺ｯ譖ｴ譁ｰ縺吶ｋ縲・
                    ss->pv->update(move, (ss + 1)->pv);

                if (value < beta)  // Update alpha here!
                                   // alpha蛟､縺ｮ譖ｴ譁ｰ縺ｯ縺薙・繧ｿ繧､繝溘Φ繧ｰ縺ｧ濶ｯ縺・・
				                   // 庁 縺ｪ縺懊↑繧峨√％縺ｮ繧ｿ繧､繝溘Φ繧ｰ莉･螟悶□縺ｨ譫晏・繧翫＆繧後ｋ縺九ｉ縲・else莉･荳九ｒ隱ｭ繧縺薙→)
                    alpha = value;
                else
                    break;  // Fail high
            }
        }
    }

	// -----------------------
    // Step 9. Check for mate
    // Step 9. 隧ｰ縺ｿ縺ｮ遒ｺ隱・
    // -----------------------

	// All legal moves have been searched. A special case: if we are
    // in check and no legal moves were found, it is checkmate.

	// 縺吶∋縺ｦ縺ｮ蜷域ｳ墓焔繧呈爾邏｢縺励∪縺励◆縲ら音蛻･縺ｪ繧ｱ繝ｼ繧ｹ: 繧ゅ＠迴ｾ蝨ｨ邇区焔繧貞女縺代※縺・※縲・
    // 縺九▽蜷域ｳ墓焔縺瑚ｦ九▽縺九ｉ縺ｪ縺九▲縺溷ｴ蜷医√◎繧後・繝√ぉ繝・け繝｡繧､繝茨ｼ郁ｩｰ縺ｿ・峨〒縺吶・

	/*
		涛 邇区焔縺後°縺九▲縺ｦ縺・ｋ迥ｶ豕√〒縺ｯ縺吶∋縺ｦ縺ｮ謖・＠謇九ｒ隱ｿ縺ｹ縺溘→縺・≧縺薙→縺縺九ｉ縲√％繧後・隧ｰ縺ｿ縺ｧ縺ゅｋ縲・
		    縺ｩ縺・○謖・＠謇九′縺ｪ縺・→縺・≧縺薙→縺縺九ｉ縲∵ｬ｡縺ｫ縺薙・node縺ｫ險ｪ蝠上＠縺ｦ繧ゅ∵欠縺玲焔逕滓・蠕後↓隧ｰ縺ｿ縺ｧ縺ゅｋ縺薙→縺ｯ
		    繧上°繧九ｏ縺代□縺励√◎繧ゅ◎繧ゅ％縺ｮnode縺瑚ｩｰ縺ｿ縺縺ｨ繧上°繧九→縺薙・node縺ｫ蜀崎ｨｪ蝠上☆繧狗｢ｺ邇・・讌ｵ繧√※菴弱￥縲・
		    鄂ｮ謠幄｡ｨ縺ｫ菫晏ｭ倥＠縺ｦ繧らｽｮ謠幄｡ｨ繧呈ｱ壹☆縺縺代〒縺ｻ縺ｨ繧薙←蠕励ｒ縺励↑縺・・繝ｬ繧｢繧ｱ繝ｼ繧ｹ縺ｪ縺ｮ縺ｧ縺ｻ縺ｨ繧薙←謳阪ｂ縺励↑縺・′)
		 
		    窶ｻ縲險域ｸｬ縺励◆縺ｨ縺薙ｍ縲∫ｽｮ謠幄｡ｨ縺ｫ菫晏ｭ倥＠縺溘⊇縺・′繧上★縺九↓蠑ｷ縺九▲縺溘′縲∵怏諢丞ｷｮ縺ｧ縺ｯ縺ｪ縺輔￡縺縺励・
		    Stockfish10縺ｮ繧ｳ繝ｼ繝峨′菫晏ｭ倥＠縺ｪ縺・さ繝ｼ繝峨↓縺ｪ縺｣縺ｦ縺・ｋ縺ｮ縺ｧ菫晏ｭ倥＠縺ｪ縺・％縺ｨ縺ｫ縺吶ｋ縲・

		投 縲占ｨ域ｸｬ雉・侭 26.縲・qsearch縺ｧ隧ｰ縺ｿ縺ｮ縺ｨ縺阪↓鄂ｮ謠幄｡ｨ縺ｫ菫晏ｭ倥☆繧・縺励↑縺・・

		統  繝√ぉ繧ｹ縺ｧ縺ｯ邇区焔縺輔ｌ縺ｦ縺・※縲∝粋豕墓焔縺後↑縺・凾縺ｫ隧ｰ縺ｿ縺縺後∝ｰ・｣九〒縺ｯ縲∝粋豕墓焔縺後↑縺代ｌ縺ｰ隧ｰ縺ｿ縺ｪ縺ｮ縺ｧ ss->inCheck縺ｮ譚｡莉ｶ縺ｯ荳崎ｦ√°縺ｨ諤昴▲縺溘ｉ縲・
		     qsearch()縺ｧ邇区焔縺輔ｌ縺ｦ縺・↑縺・凾縺ｯ縲…aptures(鬧偵ｒ謐慕佐縺吶ｋ謖・＠謇・縺ｨchecks(邇区焔縺ｮ謖・＠謇・縺ｮ謖・＠謇九＠縺狗函謌舌＠縺ｦ縺・↑縺・°繧峨・
		     moveCount==0縺縺九ｉ隧ｰ縺ｿ縺ｨ縺ｯ髯舌ｉ縺ｪ縺・・
	
			 邇区焔縺輔ｌ縺ｦ縺・ｋ螻髱｢縺ｪ繧峨‘vasion(邇区焔蝗樣∩謇・繧堤函謌舌☆繧九°繧峨［oveCount==0縺ｪ繧芽ｩｰ縺ｿ縺ｨ遒ｺ螳壹☆繧九・
			 縺励°縺礼ｽｮ謠幄｡ｨ縺ｫhit縺励◆譎ゅ↓縺ｯbestValue縺ｮ蛻晄悄蛟､縺ｯ -VALUE_INFINITE縺ｧ縺ｯ縺ｪ縺・・縺ｧ縲√◎縺・・∴繧九→
			 縺薙％縺ｯ(Stockfish縺ｮ繧ｳ繝ｼ繝峨・繧医≧縺ｫ)bestValue == -VALUE_INFINITE縺ｨ縺吶ｋ縺ｮ縺ｧ縺ｯ縺ｪ縺塾oveCount == 0縺ｨ縺励◆縺ｻ縺・′濶ｯ縺・ｈ縺・↓諤昴≧縺ｮ縺縺娯ｦ縲・
			 竊偵鄂ｮ謠幄｡ｨ縺ｫhit縺励◆縺ｮ縺ｫ譫晏・繧翫′縺ｪ縺輔ｌ縺ｦ縺・↑縺・凾轤ｹ縺ｧ譛牙柑謇九′縺ゅｋ繧上￠縺ｧ隧ｰ縺ｿ縺ｧ縺ｯ縺ｪ縺・％縺ｨ縺ｯ險縺医ｋ縺ｮ縺銀ｦ縲・
			 cf. https://yaneuraou.yaneu.com/2022/04/22/yaneuraous-qsearch-is-buggy/

		､・
			 if (ss->inCheck && bestValue == -VALUE_INFINITE)
			竊全tockfish縺ｮ繧ｳ繝ｼ繝峨や・縺薙≧螟画峩縺励◆縺ｻ縺・′濶ｯ縺・ｈ縺・↓諤昴≧縺瑚ｨ域ｸｬ縺励※縺ｿ繧九→螟ｧ蟾ｮ縺ｪ縺・・
			 Stockfish繧・2蟷ｴ蜑阪・竊代〒縺ｯ縺ｪ縺鞘・縺薙・譖ｸ縺肴婿縺縺｣縺溘ｈ縺・□縲ＮoveCount縺碁勁蜴ｻ縺輔ｌ縺滓凾縺ｫ螟画峩縺輔ｌ縺ｦ縺励∪縺｣縺溘ｈ縺・□縲・
			 cf. https://github.com/official-stockfish/Stockfish/commit/452f0d16966e0ec48385442362c94a810feaacd9
			 moveCount縺悟・蠎ｦ蟆主・縺輔ｌ縺溘°繧峨↓縺ｯ縲ヾtockfish繧ゅ％縺薙・縲≫・縺ｮ譖ｸ縺肴婿縺ｫ謌ｻ縺励◆縺ｻ縺・′濶ｯ縺・→諤昴≧縲・
	*/


#if STOCKFISH        
	if (ss->inCheck && bestValue == -VALUE_INFINITE)
    {
        assert(!MoveList<LEGAL>(pos).size());
#else
	if (ss->inCheck && moveCount == 0)
    {
		// 庁 蜷域ｳ墓焔縺ｯ蟄伜惠縺励↑縺・・縺壹□縺九ｉ謖・＠謇狗函謌舌＠縺ｦ繧ゅ☆縺舌↓邨ゅｏ繧九□繧阪≧縺九ｉ
        //     縺薙・assert縺ｯ縺昴ｓ縺ｪ縺ｫ驕・￥縺ｯ縺ｪ縺・・
        ASSERT_LV5(!MoveList<LEGAL_ALL>(pos).size());
#endif

		return mated_in(ss->ply);  // Plies to mate from the root
                                   // root縺九ｉ隧ｰ縺ｿ縺ｾ縺ｧ縺ｮ謇区焚縲・
    }

    if (!is_decisive(bestValue) && bestValue > beta)
        bestValue = (bestValue + beta) / 2;

	// 庁 逶､髱｢縺ｫking縺ｨpawn縺励°谿九▲縺ｦ縺ｪ縺・→縺阪↓迚ｹ蛹悶＠縺殱talemate蛻､螳壹・
	//     蟆・｣九〒縺ｯ逕ｨ縺・↑縺・・
#if STOCKFISH
    Color us = pos.side_to_move();
    if (!ss->inCheck && !moveCount && !pos.non_pawn_material(us)
        && type_of(pos.captured_piece()) >= ROOK)
    {
        if (!((us == WHITE ? shift<NORTH>(pos.pieces(us, PAWN))
                           : shift<SOUTH>(pos.pieces(us, PAWN)))
              & ~pos.pieces()))  // no pawn pushes available
        {
            pos.state()->checkersBB = Rank1BB;  // search for legal king-moves only
            if (!MoveList<LEGAL>(pos).size())   // stalemate
                bestValue = VALUE_DRAW;
            pos.state()->checkersBB = 0;
        }
    }
#endif

    // Save gathered info in transposition table. The static evaluation
    // is saved as it was before adjustment by correction history.

	// 蜿朱寔縺励◆諠・ｱ繧偵ヨ繝ｩ繝ｳ繧ｹ繝昴ず繧ｷ繝ｧ繝ｳ繝・・繝悶Ν縺ｫ菫晏ｭ倥＠縺ｾ縺吶・
    // 髱咏噪隧穂ｾ｡縺ｯ縲∽ｿｮ豁｣螻･豁ｴ縺ｫ繧医ｋ隱ｿ謨ｴ縺瑚｡後ｏ繧後ｋ蜑阪・迥ｶ諷九〒菫晏ｭ倥＆繧後∪縺吶・

	// 統 隧ｰ縺ｿ縺ｧ縺ｯ縺ｪ縺九▲縺溘・縺ｧ縺薙・諠・ｱ繧呈嶌縺榊・縺吶・
    //   縲qsearch()縺ｮ邨先棡縺ｯ菫｡逕ｨ縺ｪ繧峨↑縺・・縺ｧBOUND_EXACT縺ｧ譖ｸ縺榊・縺吶％縺ｨ縺ｯ縺ｪ縺・・

    ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), pvHit,
                   bestValue >= beta ? BOUND_LOWER : BOUND_UPPER, DEPTH_QS, bestMove,
                   unadjustedStaticEval, tt.generation());

	/*
		涛 鄂ｮ謠幄｡ｨ縺ｫ縺ｯ abs(value) < VALUE_INFINITE縺ｮ蛟､縺励°譖ｸ縺崎ｾｼ縺ｾ縺ｪ縺・＠縲√％縺ｮ髢｢謨ｰ繧ゅ％縺ｮ遽・峇縺ｮ蛟､縺励°霑斐＆縺ｪ縺・・
		   縺励°縺礼ｽｮ謠幄｡ｨ縺瑚｡晉ｪ√＠縺溷ｴ蜷医・縺昴≧縺ｧ縺ｯ縺ｪ縺・・謇玖ｩｰ繧√・螻髱｢縺ｧ縲∫ｽｮ謠幄｡ｨ陦晉ｪ√↓繧医ｊ1謇玖ｩｰ繧√・繧ｹ繧ｳ繧｢縺・
		   霑斐▲縺ｦ縺阪◆蝣ｴ蜷医′縺昴ｌ縺ｧ縺ゅｋ縲・
	
		    ASSERT_LV3(abs(bestValue) <= mate_in(ss->ply));
		    竍ｨ 縺薙・node縺ｯroot縺九ｉss->ply謇矩ｲ繧√◆螻髱｢縺ｪ縺ｮ縺ｧ縺薙％縺ｧss->ply繧医ｊ遏ｭ縺・ｩｰ縺ｿ縺後≠繧九・縺ｯ縺翫°縺励＞縺後・
		    縺薙・髢｢謨ｰ縺ｯ縺昴ｓ縺ｪ蛟､繧定ｿ斐＠縺ｦ縺励∪縺・ゅ＠縺九＠縺薙ｌ縺ｯ騾壼ｸｸ謗｢邏｢縺ｪ繧峨・谺｡縺ｮnode縺ｧ縺ｮ
		    mate distance pruning縺ｧ陬懈ｭ｣縺輔ｌ繧九・縺ｧ蝠城｡後↑縺・・
		    縺ｾ縺溘〃ALUE_INFINITE縺ｯint16_t縺ｮ譛螟ｧ蛟､繧医ｊMAX_PLY莉･荳雁ｰ上＆縺・↑縺ｮ縺ｧ繧ｪ繝ｼ繝舌・繝輔Ο繝ｼ縺ｮ蠢・・縺ｯ縺ｪ縺・・
	
		  繧医▲縺ｦsearch(),qsearch()縺ｮassert縺ｯ谺｡縺ｮ繧医≧縺ｫ譖ｸ縺上∋縺阪〒縺ゅｋ縲・
	*/

	ASSERT_LV3(-VALUE_INFINITE < bestValue && bestValue < VALUE_INFINITE);

    return bestValue;
}

// LMR縺ｮreduction縺ｮ蛟､繧定ｨ育ｮ励☆繧九・
int Search::YaneuraOuWorker::reduction(bool i, Depth d, int mn, int delta) const {
    int reductionScale = reductions[d] * reductions[mn];
    // rust(rshogi alpha_beta.rs:207-217)貅匁侠: delta菫よ焚757 / 髱枴mproving蛟咲紫218 / base 1200縲・
    // C++譌ｧ蛟､(585/206/1133)縺ｯ蜿､縺ТF蛟､縲Ｓust縺ｯbase縺・67螟ｧ縺阪￥蜈ｨ繝弱・繝峨〒蜑頑ｸ帛､壹・
    return reductionScale - delta * SPSA::LMR_REDUCTION_DELTA_MULT / rootDelta
         + !i * reductionScale * SPSA::LMR_REDUCTION_NOIMPROVING / 512 + SPSA::LMR_REDUCTION_BASE;
}
#if defined(P2_SRCH_BUNDLE_A)
// 束 A (M1): NonPV 専用の reduction()。
//   NonPV では窓幅 delta = beta - alpha が常に 1:
//     入口で ASSERT_LV3(PvNode || alpha == beta - 1)。 NonPV で alpha/beta を動かすのは
//     mate distance pruning だけで、 動けば alpha >= beta となって return する。 手のループで
//     alpha を上げる箇所 (alpha = value) は value >= beta の break より後なので NonPV では来ない
//     (alpha = max(alpha, ...) の 2 箇所は PvNode / rootNode 限定)。 VERIFY ビルドで毎回照合する。
//   よって delta * MULT / rootDelta == MULT / rootDelta (同じ整数の同じ除算なので切り捨ても同一)。
//   これを rootDelta を代入するたびに nonPvDeltaTerm として 1 回だけ割っておく。
int Search::YaneuraOuWorker::reduction_nonpv(bool i, Depth d, int mn) const {
    int reductionScale = reductions[d] * reductions[mn];
    return reductionScale - nonPvDeltaTerm
         + !i * reductionScale * SPSA::LMR_REDUCTION_NOIMPROVING / 512 + SPSA::LMR_REDUCTION_BASE;
}

#if !defined(EVAL_P2NNUE)
#error "P2_SRCH_BUNDLE_A は EVAL_P2NNUE 専用 (L5 が p2nnue.cpp の設定を読む)"
#endif

// 束 A: go の開始時 (start_searching。 YBWC / 委譲の helper を含む全 Worker が通る) に、
//   探索中は不変の値を作り直す。 パラメータ・option は setoption でしか変わらないので、
//   go の前に変えた値は必ずここで反映される。
void Search::YaneuraOuWorker::bundle_a_begin() {
    bundleDivs.refresh();
    Eval::p2_srch_tls_needs(tlsNeedLazyHead, tlsNeedWin, tlsNeedQsFlag);
}

#if defined(P2_SRCH_BUNDLE_A_VERIFY)
void Search::YaneuraOuWorker::bundle_a_verify_tls() const {
    bool h = true, w = true, q = true;
    Eval::p2_srch_tls_needs(h, w, q);
    BundleA::check(BundleA::V_L5_FLAGS,
                   (h && !tlsNeedLazyHead) * 4 + (w && !tlsNeedWin) * 2 + (q && !tlsNeedQsFlag), 0);
}
#endif
#endif

// 統 繧・・縺・ｉ邇九〒縺ｯ縲∽ｸ玖ｨ倥・elapsed(), elapsed_time()縺ｯ逕ｨ縺・↑縺・・
//     繧・・縺・ｉ邇九・Timer class繧呈戟縺｣縺ｦ縺・ｋ縺ｮ縺ｧ縲√◎縺｡繧峨ｒ逕ｨ縺・ｋ縲・
#if STOCKFISH
// elapsed() returns the time elapsed since the search started. If the
// 'nodestime' option is enabled, it will return the count of nodes searched
// instead. This function is called to check whether the search should be
// stopped based on predefined thresholds like time limits or nodes searched.

// elapsed() 縺ｯ謗｢邏｢髢句ｧ九°繧臥ｵ碁℃縺励◆譎る俣繧定ｿ斐☆縲ゅ◆縺縺励・nodestime' 繧ｪ繝励す繝ｧ繝ｳ縺梧怏蜉ｹ縺ｪ蝣ｴ蜷医・
// 謗｢邏｢縺励◆繝弱・繝画焚繧剃ｻ｣繧上ｊ縺ｫ霑斐☆縲ゅ％縺ｮ髢｢謨ｰ縺ｯ縲∵凾髢灘宛髯舌ｄ謗｢邏｢繝弱・繝画焚縺ｪ縺ｩ縺ｮ
// 莠句燕縺ｫ螳壹ａ縺滄明蛟､縺ｫ蝓ｺ縺･縺・※謗｢邏｢繧貞●豁｢縺吶∋縺阪°繧堤｢ｺ隱阪☆繧九◆繧√↓蜻ｼ縺ｳ蜃ｺ縺輔ｌ繧九・

// elapsed_time() returns the actual time elapsed since the start of the search.
// This function is intended for use only when printing PV outputs, and not used
// for making decisions within the search algorithm itself.

// elapsed_time() 縺ｯ謗｢邏｢髢句ｧ九°繧牙ｮ滄圀縺ｫ邨碁℃縺励◆譎る俣繧定ｿ斐☆縲・
// 縺薙・髢｢謨ｰ縺ｯPV蜃ｺ蜉帙・陦ｨ遉ｺ譎ゅ・縺ｿ菴ｿ逕ｨ縺輔ｌ縲∵爾邏｢繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝蜀・・諢乗晄ｱｺ螳壹↓縺ｯ菴ｿ逕ｨ縺輔ｌ縺ｪ縺・・

TimePoint Search::Worker::elapsed() const {
    return main_manager()->tm.elapsed([this]() { return threads.nodes_searched(); });
}

TimePoint Search::Worker::elapsed_time() const { return main_manager()->tm.elapsed_time(); }
#endif

Value Search::YaneuraOuWorker::evaluate(const Position& pos) {

#if defined(EVAL_SFNN)
	// 譛譁ｰ縺ｮStockfish縺ｮ繧ｳ繝ｼ繝・

    return Eval::evaluate(networks[numaAccessToken], pos, accumulatorStack, refreshTable,
                          optimism[pos.side_to_move()]);

#else
#if defined(P2_SMP_EVALCHECK)
	const Value v = Eval::evaluate(pos);
	P2SmpCheck::check_e(pos, v, threadIdx);
	return v;
#else
	return Eval::evaluate(pos);
#endif
#endif
}

namespace {

// Adjusts a mate or TB score from "plies to mate from the root" to
// "plies to mate from the current position". Standard scores are unchanged.
// The function is called before storing a value in the transposition table.

// 繝｡繧､繝医せ繧ｳ繧｢縺ｾ縺溘・TB・医ユ繝ｼ繝悶Ν繝吶・繧ｹ・峨せ繧ｳ繧｢繧・
// 縲後Ν繝ｼ繝医°繧峨Γ繧､繝医∪縺ｧ縺ｮ謇区焚縲阪°繧峨檎樟蝨ｨ縺ｮ螻髱｢縺九ｉ繝｡繧､繝医∪縺ｧ縺ｮ謇区焚縲阪↓隱ｿ謨ｴ縺吶ｋ縲・
// 騾壼ｸｸ縺ｮ繧ｹ繧ｳ繧｢縺ｯ螟画峩縺励↑縺・・
// 縺薙・髢｢謨ｰ縺ｯ縲√ヨ繝ｩ繝ｳ繧ｹ繝昴ず繧ｷ繝ｧ繝ｳ繝・・繝悶Ν縺ｫ蛟､繧剃ｿ晏ｭ倥☆繧句燕縺ｫ蜻ｼ縺ｳ蜃ｺ縺輔ｌ繧九・

/*
	涛 隧ｰ縺ｿ縺ｮ繧ｹ繧ｳ繧｢縺ｯ鄂ｮ謠幄｡ｨ荳翫・縲√％縺ｮnode縺九ｉ縺ゅ→菴墓焔縺ｧ隧ｰ繧縺九→縺・≧繧ｹ繧ｳ繧｢繧呈ｼ邏阪☆繧九・
	    縺励°縺励《earch()縺ｮ霑斐＠蛟､縺ｯ縲〉oot縺九ｉ縺ゅ→菴墓焔縺ｧ隧ｰ繧縺九→縺・≧繧ｹ繧ｳ繧｢繧剃ｽｿ縺｣縺ｦ縺・ｋ縲・
	   (縺薙≧縺励※縺翫°縺ｪ縺・→縲‥o_move(),undo_move()縺吶ｋ縺斐→縺ｫ隧ｰ縺ｿ縺ｮ繧ｹ繧ｳ繧｢繧偵う繝ｳ繧ｯ繝ｪ繝｡繝ｳ繝医＠縺溘ｊ繝・け繝ｪ繝｡繝ｳ繝医＠縺溘ｊ
	    縺励↑縺・→縺・￠縺ｪ縺上↑縺｣縺ｦ縺ｨ縺ｦ繧る擇蛟偵￥縺輔＞縺九ｉ縺ｧ縺ゅｋ縲・

		縺ｪ縺ｮ縺ｧ鄂ｮ謠幄｡ｨ縺ｫ譬ｼ邏阪☆繧句燕縺ｫ縲√％縺ｮ螟画鋤繧偵＠縺ｪ縺代ｌ縺ｰ縺ｪ繧峨↑縺・・
	    隧ｰ縺ｿ縺ｫ縺ｾ縺､繧上ｋ繧ｹ繧ｳ繧｢縺ｧ縺ｪ縺・↑繧蛾未菫ゅ′縺ｪ縺・・縺ｧ菴輔・螟画鋤繧り｡後ｏ縺ｪ縺・・

		ply : root node 縺九ｉ縺ｮ謇区焚縲・
*/

Value value_to_tt(Value v, int ply) { return is_win(v) ? v + ply : is_loss(v) ? v - ply : v; }
// ､・縺薙ｌ雜ｳ縺励◆邨先棡縺径bs(x) < VALUE_INFINITE縺ｧ縺ゅｋ縺薙→繧堤｢ｺ隱阪☆縺ｹ縺阪□縺ｨ諤昴≧縲・

// Inverse of value_to_tt(): it adjusts a mate or TB score from the transposition
// table (which refers to the plies to mate/be mated from current position) to
// "plies to mate/be mated (TB win/loss) from the root". However, to avoid
// potentially false mate or TB scores related to the 50 moves rule and the
// graph history interaction, we return the highest non-TB score instead.

// value_to_tt() 縺ｮ騾・・蜃ｦ逅・ｼ壹ヨ繝ｩ繝ｳ繧ｹ繝昴ず繧ｷ繝ｧ繝ｳ繝・・繝悶Ν縺九ｉ蜿門ｾ励＠縺・
// 繝｡繧､繝医せ繧ｳ繧｢縺ｾ縺溘・TB・医ユ繝ｼ繝悶Ν繝吶・繧ｹ・峨せ繧ｳ繧｢・育樟蝨ｨ縺ｮ螻髱｢縺九ｉ繝｡繧､繝・謨怜圏縺ｾ縺ｧ縺ｮ謇区焚繧堤､ｺ縺呻ｼ峨ｒ
// 縲後Ν繝ｼ繝医°繧峨Γ繧､繝・謨怜圏・・B蜍晏茜/謨怜圏・峨∪縺ｧ縺ｮ謇区焚縲阪↓隱ｿ謨ｴ縺吶ｋ縲・
// 縺溘□縺励・0謇九Ν繝ｼ繝ｫ繧・げ繝ｩ繝募ｱ･豁ｴ縺ｨ縺ｮ逶ｸ莠剃ｽ懃畑縺ｫ髢｢騾｣縺吶ｋ隱､縺｣縺溘Γ繧､繝・TB繧ｹ繧ｳ繧｢繧帝∩縺代ｋ縺溘ａ縲・
// 莉｣繧上ｊ縺ｫ譛螟ｧ縺ｮ髱杁B繧ｹ繧ｳ繧｢繧定ｿ斐☆縲・

// 涛 value_to_tt()縺ｮ騾・未謨ｰ
//     ply : root node 縺九ｉ縺ｮ謇区焚縲・

Value value_from_tt(Value v, int ply
#if STOCKFISH
    , int r50c
#endif
) {

    if (!is_valid(v))
        return VALUE_NONE;

    // handle TB win or better
    if (is_win(v))
    {
		// 東 蟆・｣九〒縺ｯTablebase髢｢菫ゅ↑縺・・縺ｧ繧ｳ繝｡繝ｳ繝医い繧ｦ繝・
#if STOCKFISH
        // Downgrade a potentially false mate score
        if (v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 100 /* - r50c */)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;

        // Downgrade a potentially false TB score.
        if (VALUE_TB - v > 100 /* - r50c */)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;
#endif

        return v - ply;
    }

    // handle TB loss or worse
    if (is_loss(v))
    {
		// 東 蟆・｣九〒縺ｯTablebase髢｢菫ゅ↑縺・・縺ｧ繧ｳ繝｡繝ｳ繝医い繧ｦ繝・

#if STOCKFISH
        // Downgrade a potentially false mate score.
        if (v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 100 /* - r50c */)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;

        // Downgrade a potentially false TB score.
        if (VALUE_TB + v > 100 /* - r50c */)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
#endif
        return v + ply;
    }

    return v;
}

// -----------------------
//     Stats縺ｮupdate
// -----------------------

// Updates stats at the end of search() when a bestMove is found
// update_all_stats()縺ｯ縲｜estmove縺瑚ｦ九▽縺九▲縺溘→縺阪↓縺昴・node縺ｮ謗｢邏｢縺ｮ邨らｫｯ縺ｧ蜻ｼ縺ｳ蜃ｺ縺輔ｌ繧九・

/*
	邨ｱ險域ュ蝣ｱ荳蠑上ｒ譖ｴ譁ｰ縺吶ｋ縲・
	prevSq           : 逶ｴ蜑阪・謖・＠謇九・鬧偵・遘ｻ蜍募・縲ら峩蜑阪・謖・＠謇九′Move::none()縺ｮ譎ゅ・SQ_NONE
	quietsSearched   : 縺薙・node縺ｧ逕滓・縺励◆quiet縺ｪ謖・＠謇九∬憶縺・・
	capturesSearched : 縺薙・node縺ｧ逕滓・縺励◆capture縺ｮ謖・＠謇九∬憶縺・・
*/

void update_all_stats(const Position&          pos,
                      Stack*                   ss,
                      Search::YaneuraOuWorker& workerThread,
                      Move                     bestMove,
                      Square                   prevSq,
                      SearchedList&            quietsSearched,
                      SearchedList&            capturesSearched,
                      Depth                    depth,
                      Move                     ttMove,
                      uint32_t                 threatCtx,
                      bool                     threatValid,
                      bool                     betaCutoff,
                      bool                     pvNode) {


    CapturePieceToHistory& captureHistory = workerThread.captureHistory;
    Piece                  movedPiece     = pos.moved_piece(bestMove);
    PieceType              capturedPiece;

    int bonus =
      std::min(SPSA::STAT_BONUS_DEPTH_MULT * depth + SPSA::STAT_BONUS_OFFSET, SPSA::STAT_BONUS_MAX)
      + SPSA::STAT_BONUS_TT_BONUS * (bestMove == ttMove)
#if defined(P2_SRCH_BUNDLE_A)
      + (SPSA::STATBONUS_PREV_DIV
           ? BundleA::vdiv(workerThread.bundleDivs.statBonusPrev, (ss - 1)->statScore,
                           SPSA::STATBONUS_PREV_DIV, BundleA::V_L2_STATPREV)
           : 0);
#else
      + (SPSA::STATBONUS_PREV_DIV ? (ss - 1)->statScore / SPSA::STATBONUS_PREV_DIV : 0);
#endif
    // H2 (SF 86f1df71 / rshogi 94f178a): non-PV then scale bonus by searched move count.
    // default NUM=0 keeps bit-exact behaviour.
    if (!pvNode && SPSA::UPDATE_ALL_SEARCHED_COUNT_NUM)
        bonus += int(int64_t(bonus)
                     * int64_t(quietsSearched.size() + capturesSearched.size())
                     * SPSA::UPDATE_ALL_SEARCHED_COUNT_NUM / (256 * 1024));
    // normalised H2: divide by the real branching factor instead of the fixed 256.
    // default 0 keeps bit-exact. one movegen per update_all_stats call when enabled.
    if (!pvNode && SPSA::UPDATE_ALL_NORM_NUM)
    {
        const int nlegal = int(MoveList<LEGAL_ALL>(pos).size());
        if (nlegal > 0)
            bonus += int(int64_t(bonus)
                         * int64_t(quietsSearched.size() + capturesSearched.size())
                         * SPSA::UPDATE_ALL_NORM_NUM / (int64_t(nlegal) * 1024));
    }
    int malus = std::min(SPSA::STAT_MALUS_DEPTH_MULT * depth + SPSA::STAT_MALUS_OFFSET, SPSA::STAT_MALUS_MAX);

    /*
		涛 Stockfish 14縺ｧ縺ｯcapture_or_promotion()縺九ｉcapture()縺ｫ螟画峩縺輔ｌ縺溘・2022/3/23]
			Stockfish 16縺ｧ縺ｯ縲…apture()縺九ｉcapture_stage()縺ｫ螟画峩縺輔ｌ縺溘・2023/10/15]
	*/

    if (!pos.capture_stage(bestMove))
    {
        update_quiet_histories(pos, ss, workerThread, bestMove, bonus * 806 / 1024);

        int actualMalus = malus * 1113 / 1024;
        // Decrease stats for all non-best quiet moves
        // 譛蝟・〒縺ｪ縺бuiet縺ｮ謖・＠謇九☆縺ｹ縺ｦ縺ｮ邨ｱ險医ｒ貂帛ｰ代＆縺帙ｋ

        for (Move move : quietsSearched)
        {
            actualMalus = actualMalus * SPSA::QUIET_MALUS_DECAY_NUM / 1024;
            update_quiet_histories(pos, ss, workerThread, move, -actualMalus);
        }

        // Threat Continuation History (Phase B) 縺ｮ譖ｴ譁ｰ: threatMove 縺後≠繧・beta cutoff 縺励◆ quiet best move 繧・
        // 豁｣譖ｴ譁ｰ縲∝酔繝弱・繝峨・謗｢邏｢貂医∩螟ｱ謨・quiet 繧定ｲ譖ｴ譁ｰ縲Ｃonus/malus 縺ｯ譌｢蟄倥・蜊雁・縲よ・繧・response 縺ｯ蟇ｾ雎｡螟悶・
        // depth/eval 繧ｲ繝ｼ繝・ 蟄ｦ鄙貞ｰら畑繧ｲ繝ｼ繝・THREAT_LEARN_*)繧剃ｽｿ縺・る←逕ｨ(蜉轤ｹ)繧ｲ繝ｼ繝医°繧牙・髮｢縺励※縺ゅｊ縲・
        // 蟄ｦ鄙貞・縺縺代ｒ蠎・￡繧後・縲御ｽｿ縺・ｧ九ａ繧狗椪髢薙↓繝・・繝悶Ν縺檎ｩｺ縲阪ｒ驕ｿ縺代※繝・・繝悶Ν繧貞・縺ｫ閧ｲ縺ｦ繧峨ｌ繧九・
        // 譌｢螳壼､縺ｯ驕ｩ逕ｨ繧ｲ繝ｼ繝医→蜷悟､縺ｪ縺ｮ縺ｧ縲∵里螳壹〒縺ｯ蠕捺擂縺ｨ螳悟・縺ｫ蜷御ｸ謖吝虚(bit-exact)縲・
        if (SPSA::THREAT_HISTORY_SCALE > 0 && threatValid && betaCutoff
            && depth >= SPSA::THREAT_LEARN_MIN_DEPTH
            && workerThread.threatRootEval < SPSA::THREAT_LEARN_MAX_EVAL)
        {
            if (bestMove.type_of() != PROMOTION)
            {
                uint32_t bi = threat_history_index(threatCtx, movedPiece, bestMove.to_sq());
                workerThread.threatHistory[bi] << (bonus / 2);
#ifdef THREAT_STATS
                {
                    uint32_t fk = threat_history_fullkey(threatCtx, movedPiece, bestMove.to_sq());
                    g_threatStats.updRefs.fetch_add(1, std::memory_order_relaxed);
                    if (g_threatTag[bi] && g_threatTag[bi] != fk)
                        g_threatStats.updCollisions.fetch_add(1, std::memory_order_relaxed);
                    g_threatTag[bi] = fk;
                }
                g_threatStats.posUpd.fetch_add(1, std::memory_order_relaxed);
#endif
            }
            const int tMalus = malus / 2;
            for (Move move : quietsSearched)
            {
                if (move == bestMove || move.type_of() == PROMOTION)
                    continue;
                uint32_t mi = threat_history_index(threatCtx, pos.moved_piece(move), move.to_sq());
                workerThread.threatHistory[mi] << -tMalus;
#ifdef THREAT_STATS
                {
                    uint32_t fk = threat_history_fullkey(threatCtx, pos.moved_piece(move), move.to_sq());
                    g_threatStats.updRefs.fetch_add(1, std::memory_order_relaxed);
                    if (g_threatTag[mi] && g_threatTag[mi] != fk)
                        g_threatStats.updCollisions.fetch_add(1, std::memory_order_relaxed);
                    g_threatTag[mi] = fk;
                }
                g_threatStats.negUpd.fetch_add(1, std::memory_order_relaxed);
#endif
            }
        }
    }
    else
    {
        // Increase stats for the best move in case it was a capture move
        // 譛蝟・焔縺梧黒迯ｲ縺吶ｋ謖・＠謇九□縺｣縺溷ｴ蜷医√◎縺ｮ邨ｱ險医ｒ蠅怜刈縺輔○繧・

        capturedPiece = type_of(pos.piece_on(bestMove.to_sq()));
        captureHistory[movedPiece][bestMove.to_sq()][capturedPiece] << bonus * 1286 / 1024;
    }

    // Extra penalty for a quiet early move that was not a TT move in
    // previous ply when it gets refuted.

    // quiet縺ｪ蛻晄悄縺ｮ謇九′縲∝燕縺ｮ謇九〒繝医Λ繝ｳ繧ｹ繝昴ず繧ｷ繝ｧ繝ｳ繝・・繝悶Ν・・T・峨・謇九〒縺ｯ縺ｪ縺上・
    // 縺九▽蜿崎ｨｼ縺輔ｌ縺溷ｴ蜷医↓霑ｽ蜉縺ｮ繝壹リ繝ｫ繝・ぅ繧剃ｸ弱∴縺ｾ縺吶・

    // 統 (ss-1)->ttHit : 荳縺､蜑阪・node縺ｧ鄂ｮ謠幄｡ｨ縺ｫhit縺励◆縺・
    // 庁 Move::null()縺ｮ蝣ｴ蜷医ヾtockfish縺ｧ縺ｯ65(遘ｻ蜍募ｾ後・蜊・′SQ_NONE縺ｧ縺ゅｋ縺薙→繧剃ｿ晁ｨｼ縺励※縺・ｋ縲ゅｄ縺ｭ縺・ｉ邇九ｂ縺昴≧螟画峩縺励◆縲・

    if (prevSq != SQ_NONE && ((ss - 1)->moveCount == 1 + (ss - 1)->ttHit) && !pos.captured_piece())
        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -malus * 616 / 1024);

    // Decrease stats for all non-best capture moves
    // 譛蝟・・謐慕佐縺吶ｋ謖・＠謇倶ｻ･螟悶・縺吶∋縺ｦ縺ｮ謇九・邨ｱ險医ｒ貂帛ｰ代＆縺帙∪縺・

    for (Move move : capturesSearched)
    {
        /*
		   ､・縺薙％縲［oved_piece_before() 縺ｧ縲∵黒迯ｲ蜑阪・鬧偵・萓｡蛟､縺ｧ閠・∴縺溘⊇縺・′縺・＞縺具ｼ・

		       竊・MovePicker縺ｧcaptureHistory繧堤畑縺・ｋ譎ゅ↓
			      moved_piece_after縺ｮ譁ｹ縺ｧ陦ｨ蠑輔″縺励※繧九・縺ｧ縲√◎繧後↓蛟｣縺・ｿ・ｦ√′縺ゅｋ縲・
		*/

        movedPiece    = pos.moved_piece(move);
        capturedPiece = type_of(pos.piece_on(move.to_sq()));
        captureHistory[movedPiece][move.to_sq()][capturedPiece] << -malus * 1559 / 1024;
    }
}


// Updates the continuation histories for the move pairs formed by
// the current move and the moves played in previous plies.

// 迴ｾ蝨ｨ縺ｮ謖・＠謇九→驕主悉縺ｮply縺ｧ謖・＆繧後◆謇九↓繧医▲縺ｦ蠖｢謌舌＆繧後ｋ謖・＠謇九・繧｢縺ｫ縺､縺・※縲・
// continuation history繧呈峩譁ｰ縺吶ｋ縲・

/*
	涛 1謇句燕縺ｫ蟇ｾ縺吶ｋ迴ｾ蝨ｨ縺ｮ謖・＠謇・竕・counterMove  (蠢懈焔)
		2謇句燕縺ｫ蟇ｾ縺吶ｋ迴ｾ蝨ｨ縺ｮ謖・＠謇・竕・followupMove (邯咏ｶ壽焔)
		4謇句燕縺ｫ蟇ｾ縺吶ｋ迴ｾ蝨ｨ縺ｮ謖・＠謇・竕・followupMove (邯咏ｶ壽焔)
		竍ｨ縲Stockfish 10縺ｧ6謇句燕繧りｦ九ｋ繧医≧縺ｫ縺ｪ縺｣縺溘・
		竍ｨ縲Stockfish 16縺ｧ3謇句燕繧りｦ九ｋ繧医≧縺ｫ縺ｪ縺｣縺溘・
*/
void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus) {
    // 驥阪∩縺ｯ SPSA 繝代Λ繝｡繝ｼ繧ｿ縺九ｉ(default = C++蛟､ {1157,648,288,576,140,441})縲・
    const std::array<ConthistBonus, 6> conthist_bonuses = {
      {{1, SPSA::CONT_HISTORY_WEIGHT_1}, {2, SPSA::CONT_HISTORY_WEIGHT_2},
       {3, SPSA::CONT_HISTORY_WEIGHT_3}, {4, SPSA::CONT_HISTORY_WEIGHT_4},
       {5, SPSA::CONT_HISTORY_WEIGHT_5}, {6, SPSA::CONT_HISTORY_WEIGHT_6}}};

    // ★SF port #5 CMHC: Counter-Move History Continuity (SF search.cpp:2073-2091, commit 8b6d8def30、
    //   overflow 修正 47be34c を反映)。
    //   「ここまでに見た継続履歴のうち既に正の値を持つものの数」positiveCount に応じた乗数
    //   {94,103,110,106,119,126,121} を掛ける (履歴が一貫して正なら更新を強め、そうでなければ弱める)。
    //   SF 現行式は entry << bonus*weight*mult/65536 だが、SF は 47be34c で weight を半分にしている。
    //   我々の weight は旧スケール (1024 基準) のままなので等価式は bonus*weight*mult/131072。
    //   bonus は fail-low 経路で数万になり得るため、47be34c と同じ overflow を避けて int64 で計算する。
    //   OFF (既定) なら分岐 1 つで従来ループへ (bit-exact)。ON 時の追加コストは entry ごとの比較 1 回と表引き 1 回。
    if (SPSA::SF_CMHC)
    {
        int positiveCount = 0;

        for (const auto [i, weight] : conthist_bonuses)
        {
            // Only update the first 2 continuation histories if we are in check
            if (ss->inCheck && i > 2)
                break;

            if (((ss - i)->currentMove).is_ok())
            {
                auto& historyEntry = (*(ss - i)->continuationHistory)[pc][to];
                if (historyEntry > 0)
                    positiveCount++;

                const int multiplier = SPSA::SF_CMHC_MULTIPLIERS[positiveCount];
                historyEntry << int(int64_t(bonus) * weight * multiplier / 131072)
                                  + SPSA::CONT_HISTORY_NEAR_PLY_OFFSET * (i < 2);
            }
        }
        return;
    }

    for (const auto [i, weight] : conthist_bonuses)
    {
        // Only update the first 2 continuation histories if we are in check
        // 邇区焔縺後°縺九▲縺ｦ縺・ｋ蝣ｴ蜷医・縺ｿ縲∵怙蛻昴・2縺､縺ｮcontinuation histories繧呈峩譁ｰ縺吶ｋ

        if (ss->inCheck && i > 2)
            break;
        if (((ss - i)->currentMove).is_ok())
            (*(ss - i)->continuationHistory)[pc][to] << (bonus * weight / 1024) + SPSA::CONT_HISTORY_NEAR_PLY_OFFSET * (i < 2);
    }
}

// Updates move sorting heuristics
// 謇九・繧ｽ繝ｼ繝医・繝偵Η繝ｼ繝ｪ繧ｹ繝・ぅ繝・け繧呈峩譁ｰ縺励∪縺・

// 庁 譁ｰ縺励＞best move縺瑚ｦ九▽縺九▲縺溘→縺阪↓謖・＠謇九・荳ｦ縺ｹ譖ｿ縺・euristics繧呈峩譁ｰ縺吶ｋ縲・
//     蜈ｷ菴鍋噪縺ｫ縺ｯ鬧偵ｒ蜿悶ｉ縺ｪ縺・欠縺玲焔縺ｮstat tables遲峨ｒ譖ｴ譁ｰ縺吶ｋ縲・

// 統  move      : 縺薙ｌ縺瑚憶縺九▲縺滓欠縺玲焔

void update_quiet_histories(
  const Position& pos, Stack* ss, Search::YaneuraOuWorker& workerThread, Move move, int bonus) {

    Color us = pos.side_to_move();
    workerThread.mainHistory[us][move.raw()] << bonus;  // Untuned to prevent duplicate effort
														// 驥崎､・＠縺溷・逅・ｒ髦ｲ縺舌◆繧√↓繝√Η繝ｼ繝九Φ繧ｰ縺輔ｌ縺ｦ縺・↑縺・

    if (ss->ply < LOW_PLY_HISTORY_SIZE)
        workerThread.lowPlyHistory[ss->ply][move.raw()] << bonus * SPSA::LOW_PLY_HISTORY_MULTIPLIER / 1024;

    update_continuation_histories(ss, pos.moved_piece(move), move.to_sq(), bonus * SPSA::CONT_HISTORY_MULTIPLIER / 1024);

	workerThread.sharedHistory.pawn_entry(pos)[pos.moved_piece(move)][move.to_sq()]
      << bonus * (bonus > 0 ? SPSA::PAWN_HISTORY_POS_MULTIPLIER : SPSA::PAWN_HISTORY_NEG_MULTIPLIER) / 1024;

    // 鬧呈遠縺｡ quiet 謇句ｰら畑 Drop Continuation History 縺ｮ譖ｴ譁ｰ(譌｢蟄・cont 縺ｮ蜊雁・縺ｮ蠑ｷ縺・縲・
    // DROP_CONT_HIST_SCALE==0(譌｢螳・縺ｪ繧牙ｮ悟・縺ｫ菴輔ｂ縺励↑縺・= bit-exact縲・
    if (SPSA::DROP_CONT_HIST_SCALE > 0 && move.is_drop())
    {
        Square psq = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;
        if (psq != SQ_NONE)
            workerThread.dropContHistory[drop_cont_index(
              us, type_of(pos.moved_piece(move)), type_of(pos.piece_on(psq)),
              move.to_sq(), pos.square<KING>(~us))]
              << bonus * SPSA::CONT_HISTORY_MULTIPLIER / 2048;
    }
}

} // namespace

// 東 Skill class縲√ｄ縺ｭ縺・ｉ邇九〒縺ｯ繧ｵ繝昴・繝医＠縺ｪ縺・・
#if STOCKFISH
// When playing with strength handicap, choose the best move among a set of
// RootMoves using a statistical rule dependent on 'level'. Idea by Heinz van Saanen.

// 謇句刈貂帙′譛牙柑縺ｧ縺ゅｋ縺ｪ繧峨｜est move繧・level'縺ｫ萓晏ｭ倥☆繧狗ｵｱ險医Ν繝ｼ繝ｫ縺ｫ蝓ｺ縺･縺蹴ootMoves縺ｮ髮・粋縺九ｉ驕ｸ縺ｶ縲・
// Heinz van Saanen縺ｮ繧｢繧､繝・い縲・
Move Skill::pick_best(const RootMoves& rootMoves, size_t multiPV) {
    static PRNG rng(now());  // PRNG sequence should be non-deterministic
							 // 荵ｱ謨ｰ繧ｸ繧ｧ繝阪Ξ繝ｼ繧ｿ繝ｼ縺ｯ髱樊ｱｺ螳夂噪縺ｧ縺ゅｋ縺ｹ縺阪・

    // RootMoves are already sorted by score in descending order
	// RootMoves縺ｯ縺吶〒縺ｫscore縺ｧ髯埼・↓繧ｽ繝ｼ繝医＆繧後※縺・ｋ縲・

	Value  topScore = rootMoves[0].score;
    int    delta    = std::min(topScore - rootMoves[multiPV - 1].score, int(PawnValue));
    int    maxScore = -VALUE_INFINITE;
    double weakness = 120 - 2 * level;

    // Choose best move. For each move score we add two terms, both dependent on
    // weakness. One is deterministic and bigger for weaker levels, and one is
    // random. Then we choose the move with the resulting highest score.

	// best move繧帝∈縺ｶ縲ゅ◎繧後◇繧後・謖・＠謇九↓蟇ｾ縺励※蠑ｱ縺輔↓萓晏ｭ倥☆繧・縺､縺ｮterm(逕ｨ隱・繧定ｿｽ蜉縺吶ｋ縲・
	// 1縺､縺ｯ縲∵ｱｺ螳夂噪縺ｧ縲∝ｼｱ縺・Ξ繝吶Ν縺ｧ縺ｯ螟ｧ縺阪￥縺ｪ繧九ｂ縺ｮ縺ｧ縲・縺､縺ｯ繝ｩ繝ｳ繝繝縺ｧ縺ゅｋ縲・
	// 谺｡縺ｫ蠕礼せ縺後ｂ縺｣縺ｨ繧るｫ倥＞謖・＠謇九ｒ驕ｸ謚槭☆繧九・

	for (size_t i = 0; i < multiPV; ++i)
    {
        // This is our magic formula
		// 縺薙ｌ縺碁ｭ疲ｳ輔・蜈ｬ蠑・

		int push = int(weakness * int(topScore - rootMoves[i].score)
                       + delta * (rng.rand<unsigned>() % int(weakness)))
                 / 128;

        if (rootMoves[i].score + push >= maxScore)
        {
            maxScore = rootMoves[i].score + push;
            best     = rootMoves[i].pv[0];
        }
    }

    return best;
}
#endif


// Used to print debug info and, more importantly, to detect
// when we are out of available time and thus stop the search.

// 繝・ヰ繝・げ諠・ｱ縺ｮ蜃ｺ蜉帙√◎縺励※繧医ｊ驥崎ｦ√↑縺ｮ縺ｯ縲・
// 蛻ｩ逕ｨ蜿ｯ閭ｽ縺ｪ譎る俣繧剃ｽｿ縺・・縺｣縺溘％縺ｨ繧呈､懷・縺励∵爾邏｢繧貞●豁｢縺吶ｋ縺溘ａ縺ｫ菴ｿ繧上ｌ繧九・

void SearchManager::check_time(Search::YaneuraOuWorker& worker) {
    if (--callsCnt > 0)
        return;

    // When using nodes, ensure checking rate is not lower than 0.1% of nodes
    // 繝弱・繝画焚繧貞渕貅悶↓縺吶ｋ蝣ｴ蜷医√メ繧ｧ繝・け鬆ｻ蠎ｦ縺後ヮ繝ｼ繝画焚縺ｮ0.1%譛ｪ貅縺ｫ縺ｪ繧峨↑縺・ｈ縺・↓縺吶ｋ

    callsCnt = worker.limits.nodes ? std::min(512, int(worker.limits.nodes / 1024)) : 512;

    static TimePoint lastInfoTime = now();

    TimePoint elapsed = tm.elapsed([&worker]() { return worker.threads.nodes_searched(); });
    TimePoint tick    = worker.limits.startTime + elapsed;

    if (tick - lastInfoTime >= 1000)
    {
        lastInfoTime = tick;
        dbg_print();
    }

    // We should not stop pondering until told so by the GUI
    // GUI縺九ｉ謖・､ｺ縺後≠繧九∪縺ｧ縲√・繝ｳ繝繝ｪ繝ｳ繧ｰ繧貞●豁｢縺吶∋縺阪〒縺ｯ縺ｪ縺・
    // 庁 ponder繝輔Λ繧ｰ縺檎ｫ九▲縺ｦ縺・◆繧峨・go ponder"縺ｮ譛荳ｭ縺ｪ縺ｮ縺ｧ
    //     "stop"縺・ponderhit"縺梧擂繧九∪縺ｧ縺ｯ蛛懈ｭ｢縺吶ｋ蠢・ｦ√′縺ｪ縺・・

    if (ponder)
        return;

	if (
    // Later we rely on the fact that we can at least use the mainthread previous
    // root-search score and PV in a multithreaded environment to prove mated-in scores.

    // 蠕後〒縲∝ｰ代↑縺上→繧ゅΓ繧､繝ｳ繧ｹ繝ｬ繝・ラ縺ｮ逶ｴ蜑阪・
    // 繝ｫ繝ｼ繝域爾邏｢縺ｮ繧ｹ繧ｳ繧｢縺ｨPV繧偵・繝ｫ繝√せ繝ｬ繝・ラ迺ｰ蠅・〒蛻ｩ逕ｨ縺励※縲・
    // 隧ｰ縺ｿ縺ｮ繧ｹ繧ｳ繧｢繧定ｨｼ譏弱〒縺阪ｋ縺ｨ縺・≧莠句ｮ溘↓萓晏ｭ倥＠縺ｦ縺・ｋ縲・

    /*
				涛 time management繧定｡後≧譎ゅ↓縲・

					1. 莉雁屓菴ｿ縺医ｋ譛螟ｧ譎る俣(tm.maximum())繧定ｶ・℃
					2. stopOnPonderhit縺荊rue
					3. movetime繧定ｶ・℃
					4. nodes縺梧欠螳壹＆繧後※縺・※縺昴ｌ繧定ｶ・℃
					5. 決 邨ゆｺ・凾蛻ｻ(tm.search_end)縺卦imeManagement縺ｫ繧医▲縺ｦ險ｭ螳壹＆繧後※縺・※縺昴ｌ繧定ｶ・℃

					縺励※縺・ｋ縺ｪ繧画爾邏｢繧貞叉蠎ｧ縺ｫ蛛懈ｭ｢縺輔○繧九・

				繧・・縺・ｉ邇九・蝣ｴ蜷医・,4,5 縺ｯ蜊ｳ蠎ｧ縺ｫ蛛懈ｭ｢縺輔○縺ｦ縺・＞縺ｨ諤昴≧縲・
				1. 縺ｯ縲｝onderhit縺励※縺・ｋ蝣ｴ蜷医・go"縺九ｉtm.maximum()繧定ｶ・℃縺励※縺・ｋ縺ｪ繧峨・諤晁・ｵゆｺ・＆縺帙※濶ｯ縺・′縲・
				   縺昴・縺ｨ縺阪∫ｧ偵℃繧翫℃繧翫∪縺ｧ諤晁・＠縺溘⊇縺・′蠕励・縺､縺ｾ繧翫∝叉蠎ｧ縺ｫ蛛懈ｭ｢縺輔○繧九・縺ｯ蟆代＠謳阪・
				2.縺ｮ繧ｱ繝ｼ繧ｹ繧ゅ∫ｧ偵℃繧翫℃繧翫∪縺ｧ縺ｯ諤晁・＠縺溘⊇縺・′蠕励・

				､・ponderhit縺励◆譎ょ綾縺九ｉ險育ｮ励＠縺ｦ縲∫ｧ貞腰菴阪〒蛻・ｊ荳翫￡縺滓凾蛻ｻ縺ｾ縺ｧ諤晁・＆縺帙◆縺・・
			*/

#if STOCKFISH
      worker.completedDepth >= 1
      && ((worker.limits.use_time_management() && (elapsed > tm.maximum() || stopOnPonderhit)) || (worker.limits.movetime && elapsed >= worker.limits.movetime)
          || (worker.limits.nodes && worker.threads.nodes_searched() >= worker.limits.nodes)))
        worker.threads.stop = worker.threads.abortedSearch = true;
#else
      worker.completedDepth >= 1)
    {
        if (
          // 3.
          (worker.limits.movetime && elapsed >= worker.limits.movetime)
          // 4.
          || (worker.limits.nodes && worker.threads.nodes_searched() >= worker.limits.nodes)
          // 5.
          || (tm.search_end && tm.search_end <= elapsed))

            worker.threads.stop = worker.threads.abortedSearch = true;

        else if (!tm.search_end && worker.limits.use_time_management() &&
				 // 1.
				 (elapsed > tm.maximum()
				 // 2
				  || stopOnPonderhit))
            tm.set_search_end(elapsed);
    }


#endif
}

// 東 Tablebase髢｢菫ゅ・蜃ｦ逅・ょｰ・｣九〒縺ｯ逕ｨ縺・↑縺・・縺ｧ繧ｳ繝｡繝ｳ繝医い繧ｦ繝医・
#if STOCKFISH
// Used to correct and extend PVs for moves that have a TB (but not a mate) score.
// Keeps the search based PV for as long as it is verified to maintain the game
// outcome, truncates afterwards. Finally, extends to mate the PV, providing a
// possible continuation (but not a proven mating line).
void syzygy_extend_pv(const OptionsMap&         options,
                      const Search::LimitsType& limits,
                      Position&                 pos,
                      RootMove&                 rootMove,
                      Value&                    v) {

    auto t_start      = std::chrono::steady_clock::now();
    int  moveOverhead = int(options["Move Overhead"]);
    bool rule50       = bool(options["Syzygy50MoveRule"]);

    // Do not use more than moveOverhead / 2 time, if time management is active
    auto time_abort = [&t_start, &moveOverhead, &limits]() -> bool {
        auto t_end = std::chrono::steady_clock::now();
        return limits.use_time_management()
            && 2 * std::chrono::duration<double, std::milli>(t_end - t_start).count()
                 > moveOverhead;
    };

    std::list<StateInfo> sts;

    // Step 0, do the rootMove, no correction allowed, as needed for MultiPV in TB.
    auto& stRoot = sts.emplace_back();
    pos.do_move(rootMove.pv[0], stRoot);
    int ply = 1;

    // Step 1, walk the PV to the last position in TB with correct decisive score
    while (size_t(ply) < rootMove.pv.size())
    {
        Move& pvMove = rootMove.pv[ply];

        RootMoves legalMoves;
        for (const auto& m : MoveList<LEGAL>(pos))
            legalMoves.emplace_back(m);

        TB::Config config = TB::rank_root_moves(options, pos, legalMoves, false, time_abort);
        RootMove&  rm     = *std::find(legalMoves.begin(), legalMoves.end(), pvMove);

        if (legalMoves[0].tbRank != rm.tbRank)
            break;

        ply++;

        auto& st = sts.emplace_back();
        pos.do_move(pvMove, st);

        // Do not allow for repetitions or drawing moves along the PV in TB regime
        if (config.rootInTB && ((rule50 && pos.is_draw(ply)) || pos.is_repetition(ply)))
        {
            pos.undo_move(pvMove);
            ply--;
            break;
        }

        // Full PV shown will thus be validated and end in TB.
        // If we cannot validate the full PV in time, we do not show it.
        if (config.rootInTB && time_abort())
            break;
    }

    // Resize the PV to the correct part
    rootMove.pv.resize(ply);

    // Step 2, now extend the PV to mate, as if the user explored syzygy-tables.info
    // using top ranked moves (minimal DTZ), which gives optimal mates only for simple
    // endgames e.g. KRvK.
    while (!(rule50 && pos.is_draw(0)))
    {
        if (time_abort())
            break;

        RootMoves legalMoves;
        for (const auto& m : MoveList<LEGAL>(pos))
        {
            auto&     rm = legalMoves.emplace_back(m);
            StateInfo tmpSI;
            pos.do_move(m, tmpSI);
            // Give a score of each move to break DTZ ties restricting opponent mobility,
            // but not giving the opponent a capture.
            for (const auto& mOpp : MoveList<LEGAL>(pos))
                rm.tbRank -= pos.capture(mOpp) ? 100 : 1;
            pos.undo_move(m);
        }

        // Mate found
        if (legalMoves.size() == 0)
            break;

        // Sort moves according to their above assigned rank.
        // This will break ties for moves with equal DTZ in rank_root_moves.
        std::stable_sort(
          legalMoves.begin(), legalMoves.end(),
          [](const Search::RootMove& a, const Search::RootMove& b) { return a.tbRank > b.tbRank; });

        // The winning side tries to minimize DTZ, the losing side maximizes it
        TB::Config config = TB::rank_root_moves(options, pos, legalMoves, true, time_abort);

        // If DTZ is not available we might not find a mate, so we bail out
        if (!config.rootInTB || config.cardinality > 0)
            break;

        ply++;

        Move& pvMove = legalMoves[0].pv[0];
        rootMove.pv.push_back(pvMove);
        auto& st = sts.emplace_back();
        pos.do_move(pvMove, st);
    }

    // Finding a draw in this function is an exceptional case, that cannot happen when rule50 is false or
    // during engine game play, since we have a winning score, and play correctly
    // with TB support. However, it can be that a position is draw due to the 50 move
    // rule if it has been been reached on the board with a non-optimal 50 move counter
    // (e.g. 8/8/6k1/3B4/3K4/4N3/8/8 w - - 54 106 ) which TB with dtz counter rounding
    // cannot always correctly rank. See also
    // https://github.com/official-stockfish/Stockfish/issues/5175#issuecomment-2058893495
    // We adjust the score to match the found PV. Note that a TB loss score can be
    // displayed if the engine did not find a drawing move yet, but eventually search
    // will figure it out (e.g. 1kq5/q2r4/5K2/8/8/8/8/7Q w - - 96 1 )
    if (pos.is_draw(0))
        v = VALUE_DRAW;

    // Undo the PV moves
    for (size_t i = rootMove.pv.size(); i > 0; --i)
        pos.undo_move(rootMove.pv[i - 1]);

    // Inform if we couldn't get a full extension in time
    if (time_abort())
        sync_cout
          << "info string Syzygy based PV extension requires more time, increase Move Overhead as needed."
          << sync_endl;
}
#endif


void SearchManager::pv(Search::YaneuraOuWorker&  worker,
                       const ThreadPool&         threads,
                       const TranspositionTable& tt,
                       Depth                     depth,
                       bool                      useCompleted) {

    const auto nodes     = threads.nodes_searched();
    auto&      rootMoves = worker.rootMoves;
    auto&      pos       = worker.rootPos;
    size_t     pvIdx     = worker.pvIdx;
    size_t     multiPV   = std::min(size_t(worker.options["MultiPV"]), rootMoves.size());
#if STOCKFISH
    uint64_t tbHits = threads.tb_hits() + (worker.tbConfig.rootInTB ? rootMoves.size() : 0);
#endif

    for (size_t i = 0; i < multiPV; ++i)
    {
        bool updated = rootMoves[i].score != -VALUE_INFINITE;

        if (depth == 1 && !updated && i > 0)
            continue;

        Depth d = updated ? depth : std::max(1, depth - 1);
        // 2026-10-17: 中断 iteration の途中値を吐かない。
        Value v = (useCompleted && SPSA::FINAL_PV_COMPLETED
                   && rootMoves[i].previousScore != -VALUE_INFINITE)
                    ? rootMoves[i].previousScore
                    : (updated ? rootMoves[i].uciScore : rootMoves[i].previousScore);

        if (v == -VALUE_INFINITE)
            v = VALUE_ZERO;

#if STOCKFISH
        bool tb = worker.tbConfig.rootInTB && std::abs(v) <= VALUE_TB;
        v       = tb ? rootMoves[i].tbScore : v;
#endif

#if STOCKFISH
        bool isExact = i != pvIdx || tb || !updated;  // tablebase- and previous-scores are exact

        // Potentially correct and extend the PV, and in exceptional cases v
        // 蠢・ｦ√↓蠢懊§縺ｦPV繧剃ｿｮ豁｣繝ｻ蟒ｶ髟ｷ縺励∽ｾ句､也噪縺ｪ蝣ｴ蜷医↓縺ｯv繧ょ・逅・☆繧・

        if (is_decisive(v) && std::abs(v) < VALUE_MATE_IN_MAX_PLY
            && ((!rootMoves[i].scoreLowerbound && !rootMoves[i].scoreUpperbound) || isExact))
            syzygy_extend_pv(worker.options, worker.limits, pos, rootMoves[i], v);
#else
        bool isExact =
          i != pvIdx /* || tb */ || !updated;  // tablebase- and previous-scores are exact
                                               // tablebase縺ｮ繧ｹ繧ｳ繧｢縺翫ｈ縺ｳ莉･蜑阪・繧ｹ繧ｳ繧｢縺ｯ豁｣遒ｺ縺ｧ縺ゅｋ

#endif

        std::string pv;
#if STOCKFISH
        for (Move m : rootMoves[i].pv)
            pv += UCIEngine::move(m, pos.is_chess960()) + " ";
#else
		// 決 繧・・縺・ｉ邇九〒縺ｯ縲…onsideration_mode縺ｮ縺ｨ縺阪・鄂ｮ謠幄｡ｨ縺九ｉPV繧偵°縺埼寔繧√ｋ縲・
        if (worker.main_manager()->search_options.consideration_mode)
        {
            Move      moves[MAX_PLY + 1];
            StateInfo si[MAX_PLY];
            int       ply = 0;

            while (ply < MAX_PLY)
            {
                // 蜊・律謇九・縺昴％縺ｧ邨ゆｺ・ゅ◆縺縺怜・謇九・PV繧貞・蜉帙・
                // 蜊・律謇九′繝吶せ繝医・縺ｨ縺阪∫ｽｮ謠幄｡ｨ繧呈峩譁ｰ縺励※縺・↑縺・・縺ｧ
                // 鄂ｮ謠幄｡ｨ荳翫・Move::none()縺後・繧ｹ繝医・謖・＠謇九↓縺ｪ縺｣縺ｦ縺・ｋ蜿ｯ閭ｽ諤ｧ縺後≠繧九・縺ｧ譌ｩ繧√↓讀懷・縺吶ｋ縲・
                auto rep = pos.is_repetition(ply);
                if (rep != REPETITION_NONE && ply >= 1)
                {
                    // 蜊・律謇九〒PV繧呈遠縺｡蛻・ｋ縺ｨ縺阪・縺昴・譌ｨ繧定｡ｨ遉ｺ
                    pv += to_usi_string(rep) + ' ';
                    break;
                }

                Move m;

                // 縺ｾ縺壹〉ootMoves.pv繧定ｾｿ繧後ｋ縺ｨ縺薙ｍ縺ｾ縺ｧ霎ｿ繧九・
                // rootMoves[i].pv[0]縺ｯ螳｣險蜍昴■縺ｮ謖・＠謇・Move::win())縺ｮ蜿ｯ閭ｽ諤ｧ縺後≠繧九・縺ｧ豕ｨ諢上・
                if (ply < int(rootMoves[i].pv.size()))
                    m = rootMoves[i].pv[ply];
                else
                {
                    // 谺｡縺ｮ謇九ｒ鄂ｮ謠幄｡ｨ縺九ｉ諡ｾ縺・・
                    auto [ttHit, ttData, ttWriter] = tt.probe(pos.key(), pos);
                    // 鄂ｮ謠幄｡ｨ縺ｫ縺ｪ縺九▲縺・
                    if (!ttHit)
                        break;

                    m = ttData.move;

                    // leaf node縺ｯ繧上ｊ縺ｨ鬮倥＞遒ｺ邇・〒Move::none()
                    if (m == Move::none())
                        break;

                    // 鄂ｮ謠幄｡ｨ縺ｫ縺ｯpseudo_legal縺ｧ縺ｯ縺ｪ縺・欠縺玲焔縺悟性縺ｾ繧後ｋ縺ｮ縺ｧ縺昴ｌ繧貞ｼｾ縺上・
                    if (!(pos.pseudo_legal_s<true>(m) && pos.legal(m)))
                        break;
                }
                // leaf node譛ｫ蟆ｾ縺ｫMove::resign()縺後≠繧九％縺ｨ縺ｯ縺ｪ縺・′縲・
                // 隧ｰ縺ｿ螻髱｢縺ｧ蜻ｼ縺ｳ蜃ｺ縺輔ｌ繧九→1謇句・縺稽ove resign縺ｪ縺ｮ縺ｧ縲√％繧後〒do_move()縺吶ｋ縺ｮ縺ｯ
                // 髱槫粋豕輔□縺九ｉ縲‥o_move()縺帙★縺ｫ繝ｫ繝ｼ繝励ｒ謚懊￠繧九・
                if (!m.is_ok())
                {
                    pv += USIEngine::move(m) + ' ';
                    break;
                }

                moves[ply] = m;
                pv += USIEngine::move(m) + ' ';

                pos.do_move(m, si[ply]);
                ++ply;
            }
            while (ply > 0)
                pos.undo_move(moves[--ply]);
		}
        else
            for (Move m : rootMoves[i].pv)
                pv += USIEngine::move(m) + ' ';
#endif

        // Remove last whitespace
        // 譛蠕後・遨ｺ逋ｽ繧貞炎髯､縺吶ｋ

        if (!pv.empty())
            pv.pop_back();

#if STOCKFISH
        auto wdl = worker.options["UCI_ShowWDL"] ? UCIEngine::wdl(v, pos) : "";
#endif
        auto bound = rootMoves[i].scoreLowerbound
                     ? "lowerbound"
                     : (rootMoves[i].scoreUpperbound ? "upperbound" : "");

        InfoFull info;

        info.depth    = d;
        info.selDepth = rootMoves[i].selDepth;
        info.multiPV  = i + 1;
#if STOCKFISH
        info.score = {v, pos};  // 統 Stockfish縺ｧ縺ｯValue,Position&縺九ｉScore蝙九↓螟画鋤縺吶ｋ縲・
        info.wdl   = wdl;
#else
        info.score = v;
#endif

        if (!isExact)
            info.bound = bound;

        TimePoint time = std::max(TimePoint(1), tm.elapsed_time());
        info.timeMs    = time;
        info.nodes     = nodes;
        info.nps       = nodes * 1000 / time;
#if STOCKFISH
        info.tbHits = tbHits;
#endif
        info.pv       = pv;
        info.hashfull = tt.hashfull();

        updates.onUpdateFull(info);
    }
}

// Called in case we have no ponder move before exiting the search,
// for instance, in case we stop the search during a fail high at root.
// We try hard to have a ponder move to return to the GUI,
// otherwise in case of 'ponder on' we have nothing to think about.

// 謗｢邏｢繧堤ｵゆｺ・☆繧句燕縺ｫponder move縺後↑縺・ｴ蜷医↓蜻ｼ縺ｳ蜃ｺ縺輔ｌ縺ｾ縺吶・
// 萓九∴縺ｰ縲〉oot縺ｧfail high縺檎匱逕溘＠縺ｦ謗｢邏｢繧剃ｸｭ譁ｭ縺励◆蝣ｴ蜷医↑縺ｩ縺ｧ縺吶・
// GUI縺ｫ霑斐☆ponder move繧偵〒縺阪ｋ髯舌ｊ貅門ｙ縺励ｈ縺・→縺励∪縺吶′縲・
// 縺昴≧縺ｧ縺ｪ縺・ｴ蜷医√継onder on縲阪・髫帙↓閠・∴繧九∋縺阪ｂ縺ｮ縺御ｽ輔ｂ縺ｪ縺上↑繧翫∪縺吶・

bool RootMove::extract_ponder_from_tt(const TranspositionTable& tt,
                                              Position&                 pos,
                                              Move                      ponder_candidate) {
    StateInfo st;

    ASSERT_LV3(pv.size() == 1);

    // 庁 Stockfish縺ｧ縺ｯ if (pv[0] == Move::none()) 縺ｨ縺ｪ縺｣縺ｦ縺・ｋ縺後・
    //     隧ｰ縺ｿ縺ｮ螻髱｢縺・ponderhit"縺ｧ霑斐▲縺ｦ縺上ｋ縺薙→縺後≠繧九・縺ｧ縲・
    //     縺薙％縺ｧ縺ｮpv[0] == Move::resign()縺ｧ縺ゅｋ縺薙→縺後≠繧翫≧繧九・
    //     縺縺九ｉ縲√ｄ縺ｭ縺・ｉ邇九〒縺ｯ縲√％縺薙・縲（s_ok()縺ｧ蛻､螳壹☆繧九・

    if (!pv[0].is_ok())
        return false;

    pos.do_move(pv[0], st);

    auto [ttHit, ttData, ttWriter] = tt.probe(pos.key(), pos);
    if (ttHit)
    {
        Move m = ttData.move;
        //if (MoveList<LEGAL>(pos).contains(ttData.move))
        // 竍ｨ Stockfish縺ｮ縺薙・繧ｳ繝ｼ繝峨｝seudo_legal縺ｨlegal縺ｧ蜊∝・縺ｪ縺ｮ縺ｧ縺ｯ縺ｪ縺・°・・
        if (pos.pseudo_legal_s<true>(m) && pos.legal(m))
            pv.push_back(m);
    }
    // 鄂ｮ謠幄｡ｨ縺ｫ繧ゅ↑縺九▲縺溘・縺ｧ莉･蜑阪・iteration譎ゅ・pv[1]繧偵⊇縺倥￥繧願ｿ斐☆縲・
    // 決 繧・・縺・ｉ邇狗峡閾ｪ謾ｹ濶ｯ
    else if (ponder_candidate)
    {
        Move m = ponder_candidate;
        if (pos.pseudo_legal_s<true>(m) && pos.legal(m))
            pv.push_back(m);
    }

    pos.undo_move(pv[0]);
    return pv.size() > 1;
}

}  // namespace YaneuraOu

using namespace YaneuraOu;
namespace {

// 閾ｪ菴懊・繧ｨ繝ｳ繧ｸ繝ｳ縺ｮentry point
void engine_main() {

	// 縺薙％縺ｧ菴懊▲縺溘お繝ｳ繧ｸ繝ｳ
    auto engine = std::make_unique<YaneuraOuEngine>();

    // USI繧ｳ繝槭Φ繝峨・蠢懃ｭ秘Κ
    auto usi = std::make_unique<USIEngine>();
    usi->set_engine(*engine);  // 繧ｨ繝ｳ繧ｸ繝ｳ螳溯｣・ｒ蟾ｮ縺玲崛縺医ｋ縲・
    usi->enqueue_startup_commands(CommandLine::g);

    // USI繧ｳ繝槭Φ繝峨・蠢懃ｭ斐・縺溘ａ縺ｮ繝ｫ繝ｼ繝・
    usi->loop();
}

// 縺薙・entry point繧堤匳骭ｲ縺励※縺翫￥縲・
static EngineFuncRegister r(engine_main, "YaneuraOuEngine", 0);
}

#endif // YANEURAOU_ENGINE


