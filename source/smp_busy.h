#ifndef SMP_BUSY_H_INCLUDED
#define SMP_BUSY_H_INCLUDED

// **「今どの thread かが展開中の局面」の集合** (2026-11-02 / fable 設計 §5.1)。
//
// なぜ必要か:
//   実測で 3 スレッドが各々「1 スレッドで読む木の 93-95%」を重複して読んでいた
//   (depth 15/18/20 で同じ / TTD は 0.97-1.00x = 並列化がまったく効いていない)。
//   原因は多様性不足ではなく**分業の不在**。 この表を見て「他 thread が今読んでいる
//   子局面」への手を後回しにすることで、 ALL node の子を thread 間で分け合う。
//
// 🚨🚨**正しさには一切関与しない。**
//   偽陽性 (古い印が残っている) = 余計に後回しにするだけ -> 末尾で必ず読むので結果は同じ。
//   偽陰性 (印が無い)           = 従来どおり重複するだけ。
//   だから relaxed atomic で十分で、 way の奪い合いも許す (奪われた側の leave は空振り)。
//   後回しにした手は**必ず同じ節点の末尾で元の moveCount のまま読む** (不変条件 I-1/I-2)。
//
// 容量: 16384 set x 4 way x 4 B = **256 KB**。 tag は key の上位 32 bit に 1 を立てたもの
//   (0 を「空き」に使うため)。 set は key の下位 bit。

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "types.h"
// Key64 / Key128 の定義。 config.h の `#define Key Key64` を解決するために必要。
#include "extra/key128.h"

namespace YaneuraOu {

class SmpBusyTable {
   public:
    static constexpr std::size_t SETS = std::size_t(1) << 14;
    static constexpr int         WAYS = 4;

    // 🚨tt.cpp:227 と同じ流儀で 64bit を取り出す。 HASH_KEY_BITS > 64 のビルド
    //   (USE_POLICY_BOOK) でも壊れないようにしておく。
    static inline std::uint64_t key64(const Key k) {
#if HASH_KEY_BITS <= 64
        return std::uint64_t(k);
#else
        return k.extract64<1>();
#endif
    }
    static inline std::uint32_t tag(const Key k) {
        // 0 を「空き」に使うので 0 だけは避ける。
        // 🚨設計書は `| 1u` と書いていたが、 それだとタグが常に奇数になり
        //   **タグ空間が半分 (2^31) に落ちて別名衝突が倍増する**。 単体テストで
        //   「同一 set の 5 keys が 4 way に全部入って見える」= タグ別名が実測できた。
        //   偽陽性は正しさには無害 (余計に後回しにするだけ) だが、 無駄に増やす理由が無い。
        //   -> 0 のときだけ 1 に寄せる形にして 2^32-1 通りを使う。
        const std::uint32_t t = std::uint32_t(key64(k) >> 32);
        return t ? t : 1u;
    }
    static inline std::size_t set_of(const Key k) {
        return std::size_t(key64(k)) & (SETS - 1);
    }

    bool is_busy(const Key k) const {
        const std::size_t  s = set_of(k);
        const std::uint32_t t = tag(k);
        for (int w = 0; w < WAYS; ++w)
            if (slot[s][w].load(std::memory_order_relaxed) == t)
                return true;
        return false;
    }

    void enter(const Key k) {
        const std::size_t  s = set_of(k);
        const std::uint32_t t = tag(k);
        for (int w = 0; w < WAYS; ++w)
            if (slot[s][w].load(std::memory_order_relaxed) == 0) {
                slot[s][w].store(t, std::memory_order_relaxed);
                return;
            }
        // 空きが無ければ 1 つ奪う。 奪われた側の leave は tag が一致しないので空振りする
        // (= 無害。 その thread は「自分の印が消えた」状態で探索を続けるだけ)。
        slot[s][(t >> 2) & (WAYS - 1)].store(t, std::memory_order_relaxed);
    }

    void leave(const Key k) {
        const std::size_t  s = set_of(k);
        const std::uint32_t t = tag(k);
        for (int w = 0; w < WAYS; ++w)
            if (slot[s][w].load(std::memory_order_relaxed) == t) {
                slot[s][w].store(0, std::memory_order_relaxed);
                return;
            }
    }

    void clear() {
        for (std::size_t s = 0; s < SETS; ++s)
            for (int w = 0; w < WAYS; ++w)
                slot[s][w].store(0, std::memory_order_relaxed);
    }

   private:
    alignas(64) std::atomic<std::uint32_t> slot[SETS][WAYS];
};

// 不変条件 I-3: enter / leave は対で、 **どの return 経路でも** leave される。
struct SmpBusyScope {
    SmpBusyScope(SmpBusyTable* t, const Key k) : tbl(t), key(k) {
        if (tbl)
            tbl->enter(key);
    }
    ~SmpBusyScope() {
        if (tbl)
            tbl->leave(key);
    }
    SmpBusyScope(const SmpBusyScope&)            = delete;
    SmpBusyScope& operator=(const SmpBusyScope&) = delete;

    SmpBusyTable* tbl;
    Key           key;
};

}  // namespace YaneuraOu

#endif  // SMP_BUSY_H_INCLUDED
