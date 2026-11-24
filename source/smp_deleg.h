#ifndef SMP_DELEG_H_INCLUDED
#define SMP_DELEG_H_INCLUDED

// ============================================================================
//  M4 木の内部での仕事の委譲 (YBWC の実装形)              2026-11-02
// ============================================================================
//
// ■ 何をするか
//
//   深い節で 2 手目以降の手を、 **master の窓と深さのまま** 空いている thread に
//   提示する。 helper は root からの手順を再生してその手を読み、 結果は**置換表**に
//   残る。 master は末尾 (M1 の後回しの 2 周目) で同じ手を普通に読むが、 そのときは
//   置換表に当たるのでほぼ無料で済む。
//
// ■ なぜこの形にしたか (素朴な YBWC との違い)
//
//   素朴な YBWC は節の状態 (alpha / bestValue / bestMove / moveCount / MovePicker) を
//   thread 間で共有し、 helper の返り値を master の局所変数に流し込む。 それには
//   **1,016 行の手ループ**を並行化し、 中断を再帰的に伝播させる必要がある。
//   我々の手ループは fork で最も改造が集中している場所で、 `SE の再帰 search_node が
//   同一 ply の stack を上書きする` という既知の罠もある。
//
//   そこで **値を渡さず、 置換表だけを介する**形にした。 得られる性質:
//
//     ✅**正しさが helper に一切依存しない**。 master は必ず全手を自分で読む。
//       helper が遅れても、 落ちても、 間違った手を読んでも、 master の探索結果は
//       「helper が居なかったとき」と同じになる (遅いだけ)。
//     ✅手ループへの手術が **2 箇所**で済む (提示と、 使い終わった枠の解放)。
//     ✅M1 の後回し機構 (後回しの手は末尾で元の moveCount のまま必ず読む) に乗るので、
//       履歴更新・bestValue 更新・root の採点といった周辺処理が**全部そのまま動く**。
//
//   払う代償:
//     🚨helper が間に合わなければ master が自分で読む = 仕事が重複する。
//       どれだけ間に合ったかは `cntHit` (置換表に当たった委譲) で数える。
//     🚨helper が読む窓は提示した時点の alpha。 master が後から alpha を上げても
//       helper は古い窓で読み続ける (= 少し余分に読む)。 正しさには影響しない。
//
// ■ 不変条件
//
//   I-1 枠の内容は `state` を OFFERED にする**前に**全部書く。 helper は state を
//       acquire で読んでから内容を読む。
//   I-2 helper は claim したら内容を**局所変数へ写してから**探索する。 写し終えれば
//       master が枠を解放してよい。
//   I-3 枠は **master が解放する** (提示した本人)。 helper は TAKEN のまま探索し、
//       終わったら DONE にする。 master は手ループを抜けるときに自分の枠を全部回収する。
//   I-4 state は**世代でタグ付け**する。 枠が再利用されたときに古い helper が
//       書き込んで状態機械を壊すのを防ぐ。
//   I-5 既定 (YBWC_MIN_DEPTH=0) では worker の `deleg` が nullptr = 1 命令も実行しない。
// ============================================================================

#include <atomic>
#include <cstdint>

#include "types.h"

namespace YaneuraOu {

/// 1 件の委譲。 提示 -> 取得 -> 完了 -> 解放 の 4 状態を回す。
struct DelegSlot {
    enum : std::uint32_t { FREE = 0, OFFERED = 1, TAKEN = 2, DONE = 3 };

    /// 委譲する節の深さの上限 (root からの手数)。 深い節は部分木が小さく、
    /// 手順の再生の費用が見合わないので切る。 **出典はここだけ** (spsa_params には置かない)。
    static constexpr int PATH_CAP = 48;

    /// `世代 * 4 + 状態`。 世代は提示ごとに 1 ずつ増える。
    alignas(64) std::atomic<std::uint32_t> state{0};

    // --- 提示内容 (state を OFFERED にする前に書く / I-1)
    int  pathLen = 0;             // root からこの節までの手数
    Move path[PATH_CAP];          // root からこの節までの手順
    Move move    = Move::none();  // 読んでほしい手
    int  depth   = 0;             // search に渡す深さ (master の newDepth)
    int  alpha   = 0;             // 提示時点の窓
    int  beta    = 0;
    bool cutNode   = false;
    bool pvNode    = false;
    // 🚨master の root aspiration 窓の幅。 `reduction()` が **これで割る**
    //   (yaneuraou-search.cpp:7224)。 専従 helper は自分の aspiration ループを
    //   通らないので 0 のままになり、 **ゼロ除算で即死する** (2026-11-02 に踏んだ)。
    int  rootDelta = 1;

    static constexpr std::uint32_t tag(std::uint32_t g, std::uint32_t st) { return g * 4 + st; }
    static constexpr std::uint32_t gen_of(std::uint32_t s) { return s / 4; }
    static constexpr std::uint32_t st_of(std::uint32_t s) { return s % 4; }
};

/// 委譲の置き場。 枠は少数で足りる (thread 数 x 数本)。
class DelegPool {
   public:
    static constexpr int SLOTS = 16;

    /// 空き枠に提示する。 成功したら枠番号、 失敗したら -1。
    ///   @param ss_path  root からこの節までの手順 (長さ n)
    int offer(const Move* ss_path, int n, Move m, int depth, Value a, Value b, bool cutNode,
              bool pvNode, int rootDelta) {
        if (n > DelegSlot::PATH_CAP)
            return -1;
        // 🚨枠が埋まっているときに 16 枠を acquire でなめると、 master の
        //   ホットパスで毎回 cache line を往復させてしまう。 1 つの relaxed な
        //   読みで先に弾く (2026-11-02 に full > offer になって時間が 1.7 倍に膨れた)。
        if (used_.load(std::memory_order_relaxed) >= SLOTS)
        {
            nFull.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }
        for (int i = 0; i < SLOTS; ++i)
        {
            DelegSlot&          s   = slot_[i];
            const std::uint32_t cur = s.state.load(std::memory_order_acquire);
            if (DelegSlot::st_of(cur) != DelegSlot::FREE)
                continue;
            std::uint32_t want = cur;
            // まず自分のものにする (状態を TAKEN 相当にせず、 世代だけ進めて OFFERED へ)
            const std::uint32_t g = DelegSlot::gen_of(cur) + 1;
            if (!s.state.compare_exchange_strong(want, DelegSlot::tag(g, DelegSlot::TAKEN),
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_relaxed))
                continue;   // 他の master に取られた
            // ここから先はこの枠を独占しているので、 内容を自由に書ける (I-1)
            s.pathLen = n;
            for (int j = 0; j < n; ++j)
                s.path[j] = ss_path[j];
            s.move    = m;
            s.depth   = depth;
            s.alpha   = int(a);
            s.beta    = int(b);
            s.cutNode   = cutNode;
            s.pvNode    = pvNode;
            s.rootDelta = rootDelta > 0 ? rootDelta : 1;
            s.state.store(DelegSlot::tag(g, DelegSlot::OFFERED), std::memory_order_release);
            used_.fetch_add(1, std::memory_order_relaxed);
            avail_.fetch_add(1, std::memory_order_relaxed);
            nOffer.fetch_add(1, std::memory_order_relaxed);
            return i;
        }
        nFull.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    /// helper: 提示されている枠を 1 つ取る。 内容を out に写して枠番号を返す。
    ///   取れなければ -1。 写し終えた時点で master が解放してよい (I-2)。
    int claim(DelegSlot& out, std::uint32_t& genOut) {
        // 🚨helper は仕事が無いあいだこれを延々と回す。 16 枠を acquire でなめると
        //   master と同じ cache line を叩き続けて探索そのものを遅くする。
        //   **1 つの relaxed な読みで空振りを弾く**のが肝 (実測 idle 150 万回/局面)。
        if (avail_.load(std::memory_order_relaxed) <= 0)
            return -1;
        for (int i = 0; i < SLOTS; ++i)
        {
            DelegSlot&          s   = slot_[i];
            const std::uint32_t cur = s.state.load(std::memory_order_acquire);
            if (DelegSlot::st_of(cur) != DelegSlot::OFFERED)
                continue;
            const std::uint32_t g    = DelegSlot::gen_of(cur);
            std::uint32_t       want = cur;
            if (!s.state.compare_exchange_strong(want, DelegSlot::tag(g, DelegSlot::TAKEN),
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_relaxed))
                continue;
            out.pathLen = s.pathLen;
            for (int j = 0; j < out.pathLen; ++j)
                out.path[j] = s.path[j];
            out.move    = s.move;
            out.depth   = s.depth;
            out.alpha   = s.alpha;
            out.beta    = s.beta;
            out.cutNode   = s.cutNode;
            out.pvNode    = s.pvNode;
            out.rootDelta = s.rootDelta;
            genOut      = g;
            avail_.fetch_sub(1, std::memory_order_relaxed);
            nClaim.fetch_add(1, std::memory_order_relaxed);
            return i;
        }
        return -1;
    }

    /// helper: 読み終えた。
    void finish(int i, std::uint32_t g) {
        std::uint32_t want = DelegSlot::tag(g, DelegSlot::TAKEN);
        if (slot_[i].state.compare_exchange_strong(want, DelegSlot::tag(g, DelegSlot::DONE),
                                                   std::memory_order_release,
                                                   std::memory_order_relaxed))
            nFinish.fetch_add(1, std::memory_order_relaxed);
    }

    /// master: 自分が提示した枠を解放する (I-3)。 helper がまだ読んでいる最中なら
    ///   待たずに **DONE を待たずに解放する**。 helper の finish は CAS で落ちるだけで
    ///   害が無い (結果は置換表にしか置かないため)。
    void release(int i) {
        if (i < 0)
            return;
        const std::uint32_t cur = slot_[i].state.load(std::memory_order_acquire);
        if (DelegSlot::st_of(cur) == DelegSlot::FREE)
            return;                       // 二重解放を数え間違えない
        // まだ誰も取っていない提示を取り下げるときは avail_ も戻す
        if (DelegSlot::st_of(cur) == DelegSlot::OFFERED)
            avail_.fetch_sub(1, std::memory_order_relaxed);
        slot_[i].state.store(DelegSlot::tag(DelegSlot::gen_of(cur), DelegSlot::FREE),
                             std::memory_order_release);
        used_.fetch_sub(1, std::memory_order_relaxed);
    }

    void clear() {
        for (int i = 0; i < SLOTS; ++i)
            slot_[i].state.store(0, std::memory_order_relaxed);
        used_.store(0, std::memory_order_relaxed);
        avail_.store(0, std::memory_order_relaxed);
        clear_counters();
    }

    // --- 計測 (★abdada_firing_rate_is_structural: 発火を直接数える)
    std::atomic<std::uint64_t> nOffer{0};    // 提示した件数
    std::atomic<std::uint64_t> nFull{0};     // 枠が無くて提示できなかった件数
    std::atomic<std::uint64_t> nClaim{0};    // helper が取った件数
    std::atomic<std::uint64_t> nFinish{0};   // helper が読み終えた件数
    std::atomic<std::uint64_t> nIdle{0};     // helper が仕事を探して空振りした回数

    void clear_counters() {
        nOffer  = 0;
        nFull   = 0;
        nClaim  = 0;
        nFinish = 0;
        nIdle   = 0;
    }

   private:
    // 🚨空振りを 1 つの relaxed な読みで弾くための数。 厳密でなくてよい
    //   (多少ずれても offer/claim は state の CAS で正しく決まる)。
    alignas(64) std::atomic<int> used_{0};    // FREE でない枠の数
    alignas(64) std::atomic<int> avail_{0};   // OFFERED の枠の数 (= 取れる仕事)
    DelegSlot slot_[SLOTS];
};

}  // namespace YaneuraOu

#endif  // SMP_DELEG_H_INCLUDED
