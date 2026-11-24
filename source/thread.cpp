#include <algorithm> // std::count
#include <cmath>     // std::abs
#include <map>
#include <memory>
#include <unordered_map>

#include "history.h"
#include "thread.h"
#include "spsa_params.h"   // SPSA::NALLOC_* (2026-09-28)
#include "p2volm.h"        // P2Volm::value / bin (2026-09-28)
#include "usi.h"
#include "book/bookaware.h"   // 🎯BookAware (定跡反映型探索、 2026-09-25)
#include "tt.h"
#include "movegen.h"

namespace YaneuraOu {

// Constructor launches the thread and waits until it goes to sleep
// in idle_loop(). Note that 'searching' and 'exit' should be already set.

// コンストラクタはスレッドを起動し、スレッドが idle_loop() 内で
// スリープ状態に入るまで待機します。
// 'searching' および 'exit' は、すでに設定されている必要がある点に注意してください。

Thread::Thread(Search::SharedState& sharedState,
               //std::unique_ptr<Search::ISearchManager> sm,
               Search::WorkerFactory          worker_factory,
               size_t                         n,               // threadIdx
               size_t                         numaN,           // numaThreadIdx
               size_t                         totalNumaCount,  // numaTotal
               OptionalThreadToNumaNodeBinder binder) :
    idx(n),
    idxInNuma(numaN),
    totalNuma(totalNumaCount),
    //nthreads(sharedState.options["Threads"]),
    stdThread(&Thread::idle_loop, this) {

#if !defined(__EMSCRIPTEN__)

#if STOCKFISH
	run_custom_job([this, &binder, &sharedState /*, &sm*/, n]() {
#else
	run_custom_job([this, &binder, &sharedState , n , worker_factory]() {
#endif
		// Use the binder to [maybe] bind the threads to a NUMA node before doing
        // the Worker allocation. Ideally we would also allocate the SearchManager
        // here, but that's minor.

        // スレッドを Worker 割り当ての前に NUMA ノードに（必要なら）バインドするために binder を使う。
        // 理想的にはここで SearchManager も割り当てたいが、それは些細なことだ。

        this->numaAccessToken = binder();
#if STOCKFISH
        this->worker = make_unique_large_page<Search::Worker>(
          sharedState, std::move(sm), n, idxInNuma, totalNuma, this->numaAccessToken);
#else
		// 🌈 やねうら王では、ここでworker_factoryを使ってWorker派生classを生成する。
		this->worker = std::move(worker_factory(sharedState,
												{n, idxInNuma, totalNuma, this->numaAccessToken}));
#endif
    });

    // スレッドはsearching == trueで開始するので、このままworkerのほう待機状態にさせておく
    wait_for_search_finished();

#else
    // yaneuraou.wasm
    // wait_for_search_finished すると、ブラウザのメインスレッドをブロックしデッドロックが発生するため、コメントアウト。
    //
    // 新しいスレッドが cv を設定するのを待ってから、ブラウザに処理をパスしたいが、
    // 新しいスレッド用のworkerを作成するためには、いったんブラウザに処理をパスする必要がある。
    //
    // https://bugzilla.mozilla.org/show_bug.cgi?id=1049079
    //
    // threadStarted という変数を設けて全てのスレッドが開始するまでリトライするようにする
    //
    // 参考：https://github.com/lichess-org/stockfish.wasm/blob/a022fa1405458d1bc1ba22fe813bace961859102/src/thread.cpp#L38
#endif
}

// Destructor wakes up the thread in idle_loop() and waits
// for its termination. Thread should be already waiting.

// デストラクタは idle_loop() 内でスレッドを起こし、
// 終了するのを待ちます。
// スレッドはすでに待機状態にある必要があります。

Thread::~Thread() {

    // 探索中にスレッドオブジェクトが解体されることはない。
    ASSERT_LV3(!searching);

    // 探索は終わっているのでexitフラグをセットしてstart_searching()を呼べば終了するはず。
    exit = true;
    start_searching();
    stdThread.join();
}

// Wakes up the thread that will start the search
// 探索を開始するスレッドを起こします

void Thread::start_searching() {
    assert(worker != nullptr);
    run_custom_job([this]() { worker->start_searching(); });
}

// Clears the histories for the thread worker (usually before a new game)
// スレッドワーカーの履歴をクリアします（通常は新しい対局の前に実行されます）

void Thread::clear_worker() {
    assert(worker != nullptr);
    run_custom_job([this]() { worker->clear(); });
}

// Blocks on the condition variable until the thread has finished searching
// 条件変数でブロックし、スレッドが探索を完了するのを待ちます

void Thread::wait_for_search_finished() {

    std::unique_lock<std::mutex> lk(mutex);
    cv.wait(lk, [&] { return !searching; });
}

// Launching a function in the thread
// スレッド内で関数を実行します

void Thread::run_custom_job(std::function<void()> f) {
    {
        std::unique_lock<std::mutex> lk(mutex);
        cv.wait(lk, [&] { return !searching; });
        jobFunc   = std::move(f);
        searching = true;
    }
    cv.notify_one();
}

void Thread::ensure_network_replicated() { worker->ensure_network_replicated(); }

// Thread gets parked here, blocked on the condition variable
// when the thread has no work to do.

// スレッドに処理すべき仕事がないとき、ここで待機状態（パーク）になり、
// 条件変数でブロックされます。

void Thread::idle_loop() {
    while (true)
    {
        std::unique_lock<std::mutex> lk(mutex);
        searching = false;
        cv.notify_one();  // Wake up anyone waiting for search finished
                          // 探索の完了を待っているすべてのスレッドを起こします

        cv.wait(lk, [&] { return searching; });

        if (exit)
            return;

        std::function<void()> job = std::move(jobFunc);
        jobFunc                   = nullptr;

        lk.unlock();

        if (job)
            job();
    }
}


//Search::SearchManager* ThreadPool::main_manager() { return main_thread()->worker->main_manager(); }

uint64_t ThreadPool::nodes_searched() const { return accumulate(&Search::Worker::nodes); }
//uint64_t ThreadPool::tb_hits() const { return accumulate(&Search::Worker::tbHits); }

static size_t next_power_of_two(uint64_t count) { return count > 1 ? (2ULL << msb(count - 1)) : 1; }

// Creates/destroys threads to match the requested number.
// Created and launched threads will immediately go to sleep in idle_loop.
// Upon resizing, threads are recreated to allow for binding if necessary.

// 要求されたスレッド数に合わせてスレッドを作成・破棄します。
// 作成され起動されたスレッドは、すぐに idle_loop 内でスリープ状態に入ります。
// リサイズ時には、必要に応じてスレッドのバインディングを可能にするために再作成されます。

void ThreadPool::set(const NumaConfig&   numaConfig,
                     Search::SharedState sharedState,
                     const Search::UpdateContext& updateContext,
                     // 🤔 やねうら王ではさらに抽象化する。
                     size_t                       requested_threads,
                     const Search::WorkerFactory& worker_factory
) {

    /*  📓
		このあと、スレッドをいったん全部解体しているのは、確保するスレッド数がいま確保しているスレッド数と
		変わらないとしても、NumaPolicyに変更があると、割り当て方法が変わるからである。

		NumaPolicyとoptions["Threads"]に変更がなければ、再確保せずに済むのだが、
		worker_factoryが一致しない場合作り直す必要があり、その判定が難しいので毎回再確保することにする。
	*/

    // いま生成済みのスレッドは全部解体してしまう。
    if (threads.size() > 0)  // destroy any existing thread(s)
    {
        main_thread()->wait_for_search_finished();

        // 📝 これは、vector::clear()を呼び出してスレッドを解体している。
        threads.clear();

        boundThreadToNumaNode.clear();
    }

#if STOCKFISH
    const size_t requested = sharedState.options["Threads"];
#else
    const size_t requested = requested_threads;
    // 🤔 やねうら王では、ここ、"Threads"の値を反映させたくない。(DL系などで、ここに柔軟性が必要)
#endif

    if (requested > 0)  // create new thread(s)
    {
        // Binding threads may be problematic when there's multiple NUMA nodes and
        // multiple Stockfish instances running. In particular, if each instance
        // runs a single thread then they would all be mapped to the first NUMA node.
        // This is undesirable, and so the default behaviour (i.e. when the user does not
        // change the NumaConfig UCI setting) is to not bind the threads to processors
        // unless we know for sure that we span NUMA nodes and replication is required.

        // スレッドのバインディングは、複数のNUMAノードや複数のStockfishインスタンスが
        // 実行されている場合に問題を引き起こす可能性があります。
        // 特に、各インスタンスが1スレッドだけで動作する場合、それらはすべて
        // 最初のNUMAノードに割り当てられてしまうことになります。
        // これは望ましくないため、デフォルトの動作（つまり、ユーザーが
        // NumaConfig UCI設定を変更していない場合）は、
        // NUMAノードをまたいでおりレプリケーションが必要であることが
        // 確実に分かっている場合を除き、スレッドをプロセッサにバインドしないようになっています。

        // NumaPolicy
        //   none     ... バインドしない(1PCで複数エンジンを動かすときはこちらにすべき。)
        //   system   ... システムから利用可能なNUMA情報を取得。
        //   auto     ... systemとnoneを自動選択。
        //   hardware ... Windows10など古いシステムでスレッドを使い切らない時用。
        // 💡 詳しくは、やねうら王Wikiの「思考エンジンオプション」の説明を参考にすること。

        // options["NumaPolicy"]と要求されたスレッド数から考慮して、スレッドのbindが必要であるかを判定する。

        const std::string numaPolicy(sharedState.options["NumaPolicy"]);
        const bool        doBindThreads = [&]() {
            if (numaPolicy == "none")
                return false;

            if (numaPolicy == "auto")
                return numaConfig.suggests_binding_threads(requested);

            // numaPolicy == "system", or explicitly set by the user
            // numaPolicy が "system" であるか、またはユーザーによって明示的に設定された場合

            return true;
        }();

        std::map<NumaIndex, size_t> counts;

        boundThreadToNumaNode = doBindThreads
                                ? numaConfig.distribute_threads_among_numa_nodes(requested)
                                : std::vector<NumaIndex>{};

        if (boundThreadToNumaNode.empty())
            counts[0] = requested;  // Pretend all threads are part of numa node 0
        else
        {
            for (size_t i = 0; i < boundThreadToNumaNode.size(); ++i)
                counts[boundThreadToNumaNode[i]]++;
        }

        sharedState.sharedHistories.clear();
        for (auto pair : counts)
        {
            NumaIndex numaIndex = pair.first;
            uint64_t  count     = pair.second;
            auto      f         = [&]() {
                sharedState.sharedHistories.try_emplace(numaIndex, next_power_of_two(count));
            };
            if (doBindThreads)
                numaConfig.execute_on_numa_node(numaIndex, f);
            else
                f();
        }

        auto threadsPerNode = counts;
        counts.clear();

        while (threads.size() < requested)
        {
            const size_t    threadId      = threads.size();
            const NumaIndex numaId        = doBindThreads ? boundThreadToNumaNode[threadId] : 0;
            auto create_thread = [&]() {
#if STOCKFISH
                auto manager = threadId == 0
                               ? std::unique_ptr<Search::ISearchManager>(
                                   std::make_unique<Search::SearchManager>(updateContext))
                               : std::make_unique<Search::NullSearchManager>();
                // 💡 Stockfishのこの実装は、main threadのときだけSearchManagerを渡して、main thread以外のときは
                //     SearchManagerを使わせない(NullSearchManagerを渡す)という意味。しかし、結局探索部からmain threadでしか
                //     SearchManagerを呼び出さないので、このような設計にする必要はないと思う。
                //     WorkerからSearchManagerにアクセスできればそれだけでいいので、やねうら王では上の設計は採用しない。
#endif

                // When not binding threads we want to force all access to happen
                // from the same NUMA node, because in case of NUMA replicated memory
                // accesses we don't want to trash cache in case the threads get scheduled
                // on the same NUMA node.

                // スレッドをバインドしない場合、すべてのアクセスが同じNUMAノードから
                // 行われるように強制したいと考えています。
                // なぜなら、NUMAのレプリケートメモリにアクセスする場合、
                // スレッドが同じNUMAノードにスケジューリングされるときに
                // キャッシュが破棄されるのを防ぎたいからです。

                auto binder = doBindThreads ? OptionalThreadToNumaNodeBinder(numaConfig, numaId)
                                            : OptionalThreadToNumaNodeBinder(numaId);
#if STOCKFISH
                threads.emplace_back(std::make_unique<Thread>(sharedState, std::move(manager),
                                                              threadId, counts[numaId]++,
                                                              threadsPerNode[numaId], binder));
#else
                threads.emplace_back(std::make_unique<Thread>(sharedState, worker_factory, threadId,
                                                              counts[numaId]++,
                                                              threadsPerNode[numaId], binder));
#endif
            };

            // Ensure the worker thread inherits the intended NUMA affinity at creation.
            if (doBindThreads)
                numaConfig.execute_on_numa_node(numaId, create_thread);
            else
                create_thread();
        }

        // 生成したスレッドに対してThread::clear_worker()を呼び出す。
        // 🤔 std::make_unique<Thread>()でもWorker::clear()が呼び出されるので
        //     起動時には二重にclearしてしまうが、仕方がないか…。
        clear();

        // 🤔 これ、ThreadPool::clear()のなかでやっているので不要なのでは…。
        main_thread()->wait_for_search_finished();
    }
}


// Sets threadPool data to initial values
// threadPool のデータを初期値に設定する

// 📝 このmethodは、resize_threads()に対して呼び出される。
//     resize_threads()は、"isready"コマンドに対して呼び出されるので、
//     つまりは、このmethodは対局ごとに対局開始時に必ず呼び出される。

void ThreadPool::clear() {
	if (threads.size() == 0)
		return;

	for (auto&& th : threads)
		th->clear_worker();

	for (auto&& th : threads)
		th->wait_for_search_finished();

	// 🤔 これはEngine派生class側で行うべき。
	//     ここにあったコードは、YaneuraOuEngine::clear()に移動させた。
#if STOCKFISH
	// These two affect the time taken on the first move of a game:
	main_manager()->bestPreviousAverageScore = VALUE_INFINITE;
	main_manager()->previousTimeReduction = 0.85;

	main_manager()->callsCnt = 0;
	main_manager()->bestPreviousScore = VALUE_INFINITE;
	main_manager()->originalTimeAdjust = -1;
	main_manager()->tm.clear();
#endif
}

void ThreadPool::run_on_thread(size_t threadId, std::function<void()> f) {
	assert(threads.size() > threadId);
	threads[threadId]->run_custom_job(std::move(f));
}

void ThreadPool::wait_on_thread(size_t threadId) {
	assert(threads.size() > threadId);
	threads[threadId]->wait_for_search_finished();
}

size_t ThreadPool::num_threads() const { return threads.size(); }


// Wakes up main thread waiting in idle_loop() and returns immediately.
// Main thread will wake up other threads and start the search.

// idle_loop() で待機しているメインスレッドを起こし、すぐにリターンする。
// メインスレッドは他のスレッドを起こして探索を開始する。

// 2026-09-28 NALLOC: 同じ総ノード数の中で、 揺れる局面 (V が高い局面) に多く配る。
//   V = P2Volm::value(pos) = 両玉の 8 近傍のうち**駒が乗っていない升**への敵の利き数の和。
//   ΔR² +3.475 で |静的評価 - 20万ノード探索| を当てる (runs/prereg/DESIGN-20260928-VOLM-FAMILY.md)。
//
//   🚨総和は「銀行」で保証する。 平均で正規化すると局ごとに総和がずれ、
//     さらに「高 V 局面へ誘導するほどノードを稼げる」フィードバックが開いて判定が壊れる。
//   🚨残高は**要求量**で記帳する (実消費ではない)。 同ノード対局の 6.6% の手は詰みで早期終了し
//     1 局の実消費は N x 手数 の中央 93.8%。 余りを回収すると NALLOC 側だけ約 6% ノードが増え
//     +6 Elo ≈ +0.9pt の偽陽性が出る (期待効果と同じ大きさ)。
//   既定は全 bin 1024 = 一様 = 現行と完全に bit-exact (下の early return で V も計算しない)。
namespace NAlloc {

// 1 局のあいだ持ち越す残高 (main thread からしか触らない。 usinewgame でリセット)
inline int64_t bank = 0;

// 🚨枯渇の計測: 「重みは 1024 より大きいのに、 残高が足りず希望額を配れなかった」手の数。
//   序盤は貯金が無いので高 V 局面が来ても増やせない。 これを測らないと
//   「効いているつもりで効いていない」が起きる (fable 指定の関門 2)。
inline int64_t n_go     = 0;   // この局で配った手数
inline int64_t n_want   = 0;   // うち「増やしたかった」手数 (w > 1024)
inline int64_t n_short  = 0;   // うち「残高不足で希望額に届かなかった」手数
inline int64_t n_floor  = 0;   // うち WMIN の下限で切り上げた手数
inline int64_t sum_req  = 0;   // 要求量の合計 (総ノードの検証用)
inline int64_t sum_base = 0;   // 一様なら使ったはずの合計 (= N x 手数)

void reset() { bank = 0; n_go = n_want = n_short = n_floor = sum_req = sum_base = 0; }

// 1 局分の要約。 枯渇率 = short / want。 高いと treatment が薄い。
void dump() {
    if (n_go == 0)
        return;
    sync_cout << "info string P2NALLOC go=" << n_go << " want=" << n_want << " short=" << n_short
              << " floor=" << n_floor << " sum_req=" << sum_req << " sum_base=" << sum_base
              << " bank=" << bank << sync_endl;
}

// limits.nodes を V に応じて書き換える。 戻り値は実際に割り当てた量 (0 = 何もしなかった)
inline int64_t apply(const Position& pos, Search::LimitsType& limits) {
    const int w[5] = {SPSA::NALLOC_W0, SPSA::NALLOC_W1, SPSA::NALLOC_W2,
                      SPSA::NALLOC_W3, SPSA::NALLOC_W4};
    // 全部 1024 なら何もしない (V も計算しない = bit-exact)
    if (w[0] == 1024 && w[1] == 1024 && w[2] == 1024 && w[3] == 1024 && w[4] == 1024)
        return 0;
    if (limits.nodes <= 0)
        return 0;                       // 同ノードモードでないときは触らない

    const int64_t N = (int64_t) limits.nodes;
    int b;
    if (SPSA::NALLOC_PLACEBO)
    {
        // プラセボ腕: V ではなく局面のハッシュで同じ重みを配る。
        //   「不均一そのもの」と「V を狙ったこと」を分離するための対照。
        const uint64_t h = pos.key();
        const int      r = (int) ((h >> 32) % 100);
        // 🚨2026-09-28 修正: **実戦の探索中**の V 分布に合わせる。
        //   局面標本では p(V>=5) = 10% だったが、 実戦の実測は **6.1%** (対局 48 局の nodes 分布より)。
        //   標本の分布でプラセボを振ると、 プラセボだけ高 V 側に多く配られて
        //   NALLOC より総ノードを使ってしまい**対照として不公平**になる。
        b = r < 40 ? 0 : r < 73 ? 1 : r < 94 ? 2 : r < 98 ? 3 : 4;   // 40/33/21/4/2 ≈ 実戦の分布
    }
    else
        b = P2Volm::bin(P2Volm::value(pos));

    int64_t want = N * (int64_t) w[b] / 1024;                        // 希望額

    // 🚨2026-09-28 修正: 固定の重みでは総ノードが揃わない。
    //   V>=5 の出現率は局面集合ごとに **1.4% - 10%** と変わるので、 どの値で正規化しても
    //   次の集合でずれる。 銀行は借金禁止なので、 ずれは常に NALLOC 側の損になる
    //   (実測: 対局 48 局で -4.4%、 関門 1 の 6 局で -4.9%)。
    //   → **使い残しを低 V 側に還元する**。 残高が貯まっているときは低 V 局面にも上乗せし、
    //     ただし一様の N は超えない (高 V に配る分を食わないため)。
    //   これで出現率に依存せず総和が自動で揃う。
    //   🚨還元を無条件にすると**傾斜が消える** (低 V が毎回 N に戻るので貯金ができず、
    //     シミュレーションで傾斜 2.2 倍 → 1.07 倍)。 → **残高の上限を超えた分だけ**還元する。
    //     上限 = 高 V に 3 回配れる分。 これで総ノードが ±0.02% で揃い、 傾斜も 2.07-2.22 倍残る
    //     (V>=5 の出現率 1.4% / 6.1% / 10% のどれでも成立)。
    const int64_t cap = N * (int64_t)(SPSA::NALLOC_W3 - 1024) / 1024 * 3;
    if (want < N && bank > cap)
        want += std::min(bank - cap, N - want);

    int64_t    req       = std::min(want, N + bank);                 // 借金禁止
    const bool shortfall = (req < want);
    const int64_t floored   = N * (int64_t) SPSA::NALLOC_WMIN / 1024;
    const bool    hitFloor  = (req < floored);
    req = std::max(req, floored);                                    // 下限
    bank += N - req;                                                 // 🚨要求量で記帳
    limits.nodes = (uint64_t) req;

    ++n_go;
    if (w[b] > 1024) ++n_want;
    if (shortfall)   ++n_short;
    if (hitFloor)    ++n_floor;
    sum_req  += req;
    sum_base += N;
    return req;
}

}  // namespace NAlloc

void ThreadPool::start_thinking(const OptionsMap&  options,
                                Position&          pos,
                                StateListPtr&      states,
                                Search::LimitsType limits) {

    main_thread()->wait_for_search_finished();

    // 📝 increaseDepthはmain_managerに移動させた。
    //     ここのある初期化のうち、stopとabortedSearch以外は、Worker派生classで処理すべき。
    // 🌈 SearchManager::pre_start_searching()に移動させた。
	//     これは、Workerの派生classのpre_start_searching()から呼び出される。
#if STOCKFISH
    main_manager()->stopOnPonderhit = stop = abortedSearch = false;
    main_manager()->ponder                                 = limits.ponderMode;
    increaseDepth                                          = true;
#else
    stop = abortedSearch = false;
#endif
    // 🎯理論離脱 v2: 新しい go のたびに「外から止められた」印を下ろす。
    external_stop = false;

    Search::RootMoves rootMoves;
#if STOCKFISH
    const auto legalmoves = MoveList<LEGAL_ALL>(pos);

    for (const auto& usiMove : limits.searchmoves)
    {
        auto move = USIEngine::to_move(pos, usiMove);

        if (std::find(legalmoves.begin(), legalmoves.end(), move) != legalmoves.end())
            rootMoves.emplace_back(move);
    }

    // limits.searchmovesが指定されていないとき、rootMovesがemptyになる。
    // このとき、すべての合法手でスタートする必要がある。
    if (rootMoves.empty())
        for (const auto& m : legalmoves)
            rootMoves.emplace_back(m);

#else

    // 🌈  GenerateAllLegalMoves反映させないと..
    bool generate_all_legal_moves =
      options.count("GenerateAllLegalMoves") && options["GenerateAllLegalMoves"];

    // ⚠ MoveList<LEGAL_ALL>とMoveList<LEGAL>は異なる型なので1つの変数に代入できない。
    //     std::variantを使うよりは次のように書いたほうがすっきりする。

    auto setup_rootMoves = [&](auto& moveList) {
        for (const auto& usiMove : limits.searchmoves)
        {
            auto move = USIEngine::to_move(pos, usiMove);

            if (std::find(moveList.begin(), moveList.end(), move) != moveList.end())
                rootMoves.emplace_back(move);
        }

        if (rootMoves.empty())
            for (const auto& m : moveList)
                rootMoves.emplace_back(m);
    };

    if (generate_all_legal_moves) {
        auto legalmoves = MoveList<LEGAL_ALL>(pos);
        setup_rootMoves(legalmoves);
    } else {
        auto legalmoves = MoveList<LEGAL>(pos);
        setup_rootMoves(legalmoves);
    }

#endif

	// 🎯BookAware (定跡反映型探索、 2026-09-25): root の各手に下駄を付ける。
	//   ここで 1 回だけ計算する。 この直後に rootMoves が全 worker へコピーされるので、
	//   worker 側は int を読むだけで済む (探索中の分岐も定跡参照も増えない = NPS 税ゼロ)。
	//   ybb の fstream seek は main thread からのこの 1 回だけなので thread safe。
	//   BookAwareBias=0 のときは annotate が即座に return するので bit-exact。
	//   設計 runs/prereg/DESIGN-20260925-BOOKAWARE.md §5.1 / §5.2
#if defined(USE_BOOK_AWARE)
	// 🚨ゲート (設計 §15): 相手の消費時間から「相手が定跡を辿っているか」を更新する。
	//   root が定跡に載っていることは相手の準備の証拠にならない (我々自身が定跡なしで
	//   96% の局を 60 手目まで定跡内で指す) ので、 証拠が無ければ avoid は効かせない。
	//   limits と root 手番が揃うのはここだけ。
	BookAware::manager.on_go(limits, pos.side_to_move(), pos.game_ply());
#endif

	// 🚨NALLOC はここで limits.nodes を書き換える (worker へコピーされる前)。
	//   停止判定は main thread が worker.limits.nodes を読むので、 コピー前なら Threads>1 でも一貫する。
	NAlloc::apply(pos, limits);

    //Tablebases::Config tbConfig = Tablebases::rank_root_moves(options, pos, rootMoves);
    // ⇨  Tablebasesは将棋では用いないのでコメントアウト

    // After ownership transfer 'states' becomes empty, so if we stop the search
    // and call 'go' again without setting a new position states.get() == nullptr.

    // 所有権の移動後、'states' は空になるため、検索を中断して
    // 新しい局面を設定せずに再度 'go' を呼び出すと、states.get() == nullptr となる。

    assert(states.get() || setupStates.get());

    if (states.get())
        setupStates = std::move(states);  // Ownership transfer, states is now empty

    // We use Position::set() to set root position across threads. But there are
    // some StateInfo fields (previous, pliesFromNull, capturedPiece) that cannot
    // be deduced from a fen string, so set() clears them and they are set from
    // setupStates->back() later. The rootState is per thread, earlier states are
    // shared since they are read-only.

    // 複数のスレッドでルート局面を設定するために Position::set() を使用します。
    // しかし、StateInfo の一部のフィールド（previous、pliesFromNull、capturedPiece）は
    // FEN 文字列からは推測できないため、set() はそれらをクリアし、後で setupStates->back() から設定されます。
    // rootState はスレッドごとに個別ですが、それ以前の状態は読み取り専用のため共有されます。

    for (auto&& th : threads)
    {
        th->run_custom_job([&]() {
#if STOCKFISH
            th->worker->limits = limits;
            th->worker->nodes = th->worker->tbHits = th->worker->bestMoveChanges = 0;
            th->worker->nmpMinPly                                                = 0;
            th->worker->rootDepth = th->worker->completedDepth = 0;
            th->worker->rootMoves                              = rootMoves;
            th->worker->rootPos.set(pos.fen(), pos.is_chess960(), &th->worker->rootState);
            th->worker->rootState = setupStates->back();
            th->worker->tbConfig  = tbConfig;
#else

            th->worker->limits = limits;
            th->worker->nodes  = 0;
#endif

			// 📝 tbHits、tbConfigは将棋では使わない。

#if STOCKFISH
            th->worker->nmpMinPly = 0;
			th->worker->bestMoveChanges = 0;
            th->worker->rootDepth = th->worker->completedDepth = 0;

            // 🤔 やねうら王では、Worker派生classのpre_start_searching()で行うようにする。
            //     やねうら王では、void Search::YaneuraOuWorker::pre_start_searching()で行っている。
#endif

            th->worker->rootMoves = rootMoves;
            th->worker->rootPos.set(pos.sfen(), &th->worker->rootState);
            th->worker->rootState = setupStates->back();

#if !STOCKFISH
			// ⚠ どうせなら、↑でworker->rootPos.set()が終わってから呼び出したい。
			//     (rootPosを使って入玉判定などを行いたいため)
            th->worker->pre_start_searching();
#endif
		});
    }

    for (auto&& th : threads)
        th->wait_for_search_finished();

    main_thread()->start_searching();
}

// ⚠ このメソッドは、やねうら王の標準探索エンジンでしか使わないので、
//     YaneuraOuEngine側に移動させた。
//Thread* ThreadPool::get_best_thread() const


// Start non-main threads.
// Will be invoked by main thread after it has started searching.

// メインスレッド以外のスレッドを開始する。
// メインスレッドが探索を開始した後に呼び出される。

void ThreadPool::start_searching() {

	for (auto&& th : threads)
		if (th != threads.front())
			th->start_searching();
}


// Wait for non-main threads

// メインスレッド以外のスレッドを待機する

void ThreadPool::wait_for_search_finished() const {

	for (auto&& th : threads)
		if (th != threads.front())
			th->wait_for_search_finished();
}

std::vector<size_t> ThreadPool::get_bound_thread_count_by_numa_node() const {
	std::vector<size_t> counts;

	if (!boundThreadToNumaNode.empty())
	{
		NumaIndex highestNumaNode = 0;
		for (NumaIndex n : boundThreadToNumaNode)
			if (n > highestNumaNode)
				highestNumaNode = n;

		counts.resize(highestNumaNode + 1, 0);

		for (NumaIndex n : boundThreadToNumaNode)
			counts[n] += 1;
	}

	return counts;
}

void ThreadPool::ensure_network_replicated() {
	for (auto&& th : threads)
		th->ensure_network_replicated();
}

} // namespace YaneuraOu
