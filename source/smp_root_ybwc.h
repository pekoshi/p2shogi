#ifndef SMP_ROOT_YBWC_H_INCLUDED
#define SMP_ROOT_YBWC_H_INCLUDED

// ============================================================================
//  根の YBWC (Young Brothers Wait Concept) の共有状態          2026-11-02
// ============================================================================
//
// ■ なぜ作るか
//
//   Lazy SMP は各 thread が**同じ木を独立に**読む。 そのため alpha が確定する前に
//   兄弟手を読む「投機」になり、 実測で 1 thread の木の **92-95%** を重複して読んでいた
//   (★smp_problem_is_duplication)。 thread を増やしても depth に届く時間は縮まない
//   (1/2/3/4/6 thread で 1.00 / 1.05 / **0.98** / 0.95 / 0.86 倍 / ★thread_curve)。
//
//   投機が高い理由は実測した: **1 手目を外して残りの手だけ読むと、 全手を読むのと
//   同じかそれ以上かかる** (3 局面で 1.49 / 0.76 / 2.35 倍)。 兄弟手が安いのは
//   「1 手目が先に alpha を上げたから」であって、 手が少ないからではない。
//
//   YBWC はこの投機を払わない代わりに「長兄 (1 手目) を読み終えるまで兄弟を配らない」。
//   その待ち時間が Amdahl の直列部分になる。 集中率 (1 手目が使う木の割合) の実測は
//   **0.84** なので、 **根だけ**の分業の天井は P=3 で **1.12 倍**、 P=16 でも 1.18 倍。
//   🚨つまりこの段階だけでは大きくは勝てない。 木の内部でも分けて初めて伸びる。
//      本実装は内部分割の土台 (世代・取得・共有 alpha・結果の回収) を兼ねる。
//
// ■ 設計の要点 (ここが壊れると静かに嘘の探索になる)
//
//   I-1 **helper は master の深さに従う**。 Lazy SMP では各 thread が独立に rootDepth を
//       進めるので、 そのままでは世代が噛み合わない。 helper は master が publish した
//       世代 (rootDepth, pvIdx, alpha, beta) をそのまま使う。
//   I-2 **共有表は添字で引く**。 ところが各 thread は自分の rootMoves を自分の
//       スコアで並べ替えるので順序が食い違う。 -> master が世代ごとに手順 moves_ を
//       publish し、 helper は自分の rootMoves を**その順に揃えてから**読む。
//   I-3 **publish の順序**: 状態を全部書いた**最後に** gen を release で書く。
//       helper は gen を acquire で読んでから他を読む。 逆にすると半端な状態を読む。
//   I-4 **長兄は master だけが読む**。 helper は eldest_done(g) を待ち、 読まずに次へ。
//   I-5 **取得した手は必ず読み終えて publish する**。 中断 (threads.stop) でも
//       state_ を DONE か FREE のどちらかに必ず落とす。 放棄した手は master が
//       自分で読み直す (= 手が 1 つも読まれないことが無い)。
//   I-6 **既定 (SPSA_ROOT_YBWC=0) では 1 命令も実行しない**。 worker の rootSplit が
//       nullptr になり、 全分岐が落ちる。
//
// ■ 既定で bit-exact であること
//
//   SPSA_ROOT_YBWC の既定は 0。 iterative_deepening の先頭で Threads>1 かつ
//   MultiPV==1 かつ searchmoves 無しのときだけポインタを張る。 それ以外は従来の
//   Lazy SMP がそのまま動く。
// ============================================================================

#include <atomic>
#include <cstdint>

#include "types.h"

namespace YaneuraOu {

class RootSplit {
   public:
    // PV は報告用なので長さを切る。 32 手あれば info pv の実用上十分。
    static constexpr int PV_CAP = 32;

    // --- state_ の意味。 🚨**世代でタグ付けする** (`gen * 4 + st`)。
    //   古い世代の helper が新しい世代の表へ書き込んで結果を混ぜる事故を、
    //   CAS の期待値に世代を含めることで検出して捨てる。
    enum : int { FREE = 0, TAKEN = 1, DONE = 2 };
    static constexpr int tag(int g, int st) { return g * 4 + st; }

    // ------------------------------------------------------------------
    //  master 側
    // ------------------------------------------------------------------

    /// 新しい世代を publish する。 **他を全部書いた最後に gen を書く** (I-3)。
    /// @param g   世代 (master が 1 ずつ増やす。 0 は無効を表すので 1 から)
    /// @param n   根の手数
    /// @param ms  master の順に並べた根の手 (長さ n)
    /// @param a      この世代の aspiration 窓の alpha
    /// @param b      この世代の aspiration 窓の beta
    /// @param depth  master が search<Root> に渡す実効深さ (adjustedDepth)
    /// @param pvIdx  MultiPV の添字 (本実装は 1 本のみなので常に 0)
    void publish(int g, int n, const Move* ms, Value a, Value b, int depth, int pvIdx) {
        nMoves_ = n;
        depth_  = depth;
        pvIdx_  = pvIdx;
        for (int i = 0; i < n; ++i)
        {
            moves_[i] = ms[i];
            state_[i].store(tag(g, FREE), std::memory_order_relaxed);
            score_[i].store(int(-VALUE_INFINITE), std::memory_order_relaxed);
            pvLen_[i].store(0, std::memory_order_relaxed);
        }
        alpha_.store(int(a), std::memory_order_relaxed);
        beta_.store(int(b), std::memory_order_relaxed);
        next_.store(1, std::memory_order_relaxed);   // 添字 0 は長兄 = master 専用
        eldestDone_.store(0, std::memory_order_relaxed);
        gen_.store(g, std::memory_order_release);    // 🚨最後に書く (I-3)
    }

    /// 長兄を読み終えたことを知らせる。 共有 alpha も同時に上げる。
    void eldest_finished(int g, Value v) {
        raise_alpha(v);
        eldestDone_.store(g, std::memory_order_release);
    }

    /// この世代を閉じる (helper が新しい手を取らなくなる)。
    void close(int g) {
        if (gen_.load(std::memory_order_acquire) == g)
            next_.store(nMoves_, std::memory_order_release);
    }

    /// 完全に無効化する (探索の終わり / clear)。
    void reset() {
        gen_.store(0, std::memory_order_release);
        eldestDone_.store(0, std::memory_order_relaxed);
    }

    // ------------------------------------------------------------------
    //  両者
    // ------------------------------------------------------------------

    int  gen() const { return gen_.load(std::memory_order_acquire); }
    int  n_moves() const { return nMoves_; }
    Move move_at(int i) const { return moves_[i]; }
    bool eldest_done(int g) const { return eldestDone_.load(std::memory_order_acquire) == g; }

    Value alpha() const { return Value(alpha_.load(std::memory_order_relaxed)); }
    Value beta() const { return Value(beta_.load(std::memory_order_relaxed)); }
    int   depth() const { return depth_; }
    int   pv_idx() const { return pvIdx_; }

    /// 取られた手が全部 DONE になったか。 中断で放棄 (FREE に戻った) ものは
    /// master が自分で読み直すので、 ここでは待たない (I-5)。
    bool all_taken_done(int g) const {
        for (int i = 1; i < nMoves_; ++i)
            if (state_[i].load(std::memory_order_acquire) == tag(g, TAKEN))
                return false;
        return true;
    }

    /// 共有 alpha を上げる (下げない)。 CAS で競合を解く。
    void raise_alpha(Value v) {
        int cur = alpha_.load(std::memory_order_relaxed);
        while (int(v) > cur
               && !alpha_.compare_exchange_weak(cur, int(v), std::memory_order_relaxed,
                                                std::memory_order_relaxed))
        {}
    }

    /// 添字 i を自分が取れるか。 既に誰かが取っていれば false。
    /// 世代 g が合っていなければ必ず false (古い世代の横取りを防ぐ)。
    bool try_take(int g, int i) {
        int want = tag(g, FREE);
        return i >= 0 && i < nMoves_
            && state_[i].compare_exchange_strong(want, tag(g, TAKEN),
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_relaxed);
    }

    // 🚨helper も master と同じ手ループ (search<Root>) を通るので、 どちらも
    //   自分の添字に対して try_take() を撃つ。 先に CAS した側が取る。
    //   fetch_add で配る方式は使わない (使わないものは書かない)。

    /// 読み終えた結果を書く。
    /// 読み終えた結果を書く。 世代が合っていなければ**何も書かない**。
    ///   🚨書く順序: 値 -> PV -> 状態。 読む側は状態を acquire で見てから値を読む。
    void publish_result(int g, int i, Value v, const Move* pv, int len) {
        int want = tag(g, TAKEN);
        if (state_[i].load(std::memory_order_acquire) != want)
            return;                                  // 古い世代 / 自分の持ち物でない
        const int k = len < PV_CAP ? len : PV_CAP;
        for (int j = 0; j < k; ++j)
            pvBuf_[i][j] = pv[j];
        pvLen_[i].store(k, std::memory_order_relaxed);
        score_[i].store(int(v), std::memory_order_relaxed);
        state_[i].compare_exchange_strong(want, tag(g, DONE), std::memory_order_release,
                                          std::memory_order_relaxed);
        raise_alpha(v);
    }

    /// 取ったのに読めなかった手を返す (中断時 / I-5)。
    void give_back(int g, int i) {
        int want = tag(g, TAKEN);
        state_[i].compare_exchange_strong(want, tag(g, FREE), std::memory_order_release,
                                          std::memory_order_relaxed);
    }

    bool is_done(int g, int i) const {
        return state_[i].load(std::memory_order_acquire) == tag(g, DONE);
    }
    Value score_of(int i) const { return Value(score_[i].load(std::memory_order_relaxed)); }
    int   pv_len(int i) const { return pvLen_[i].load(std::memory_order_relaxed); }
    Move  pv_at(int i, int j) const { return pvBuf_[i][j]; }

    // --- 計測 (★abdada_firing_rate_is_structural: 配線と発火を直接数える)
    //   🚨共有の atomic だが **根でしか触らない** ので 1 iteration に数回しか
    //     加算されない = cache line の往復は無視できる。
    std::atomic<std::uint64_t> cntTaken{0};     // helper が取った手の数
    std::atomic<std::uint64_t> cntDone{0};      // 読み終えた数
    std::atomic<std::uint64_t> cntGiveBack{0};  // 放棄した数
    std::atomic<std::uint64_t> cntWaitUs{0};    // helper が長兄を待った時間 (us)
    std::atomic<std::uint64_t> cntImport{0};    // master が回収した数

    void clear_counters() {
        cntTaken    = 0;
        cntDone     = 0;
        cntGiveBack = 0;
        cntWaitUs   = 0;
        cntImport   = 0;
    }

   private:
    alignas(64) std::atomic<int> gen_{0};
    alignas(64) std::atomic<int> next_{0};
    alignas(64) std::atomic<int> eldestDone_{0};
    alignas(64) std::atomic<int> alpha_{0};
    std::atomic<int>             beta_{0};
    int                          nMoves_ = 0;
    int                          depth_  = 0;
    int                          pvIdx_  = 0;
    Move                         moves_[MAX_MOVES];
    std::atomic<int>             state_[MAX_MOVES];
    std::atomic<int>             score_[MAX_MOVES];
    std::atomic<int>             pvLen_[MAX_MOVES];
    Move                         pvBuf_[MAX_MOVES][PV_CAP];
};

}  // namespace YaneuraOu

#endif  // SMP_ROOT_YBWC_H_INCLUDED
