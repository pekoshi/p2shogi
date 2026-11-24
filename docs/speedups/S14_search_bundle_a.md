# S14 SRCH_BUNDLE_A: 探索部の結果を変えない高速化の束

- 状態: 採用 (LZ 段 1 に同梱) / bit 一致: 一致 / requires: M00
- 人間向け: `../SPEEDUPS_STORY.md` 7 章
- 本家への提案の可否: `../UPSTREAM_PROPOSALS.md` P5

## 1. 要約

やねうら王 (Stockfish 系) の探索部で、 実行時の整数除算 (idiv)・本番で常に偽の分岐・不要な thread_local の書き込み・関数呼び出しの振り分けを削る 5 項目。 単独で 101.53% [100.86, 102.18]、 LZ の IDXV2・ROWPF と束ねて PGO 同士 +4.19% [+3.43, +5.01]。

## 2. 前提

我々は探索の係数を USI option (実行時の変数) にしている。 本家の配布ビルドは係数が `constexpr` なので、 L2 のような除算はコンパイラが既に乗算に変えている。 **項目ごとに、 移植先で同じ形の費用が実在するかを先に確かめること**。

| 項目 | 一般の言葉 | 移植先で効く条件 |
|---|---|---|
| M1 | NonPV の LMR の reduction の `delta * A / rootDelta` で、 delta==1 のとき商は窓ごとに一定なので、 go (または窓の更新) ごとに 1 回だけ割って持つ | reduction に `delta / rootDelta` の項がある (🚨本家 HEAD 11-16 の `reduction(i, d, mn)` ではこの項が消えた) |
| L1 | MovePicker の静かな手の採点で、 本番の設定では常に偽になる機能の分岐を外した速い経路 (`noinline` の別関数) を用意し、 機能が全部 off のときだけそれを使う | 採点に実行時の機能スイッチがある |
| L2 | 実行時の係数による符号付き除算 `n / d` を、 magic number の乗算 1 回とシフトに置き換える (Granlund-Montgomery / libdivide の round-up 法)。 係数は setoption でしか変わらないので、 go の開始時に作り直す | 除数が実行時の変数 |
| L5 | node の入口で毎回書いている thread_local の変数を、 その値が読まれない設定のときは書かない | 読まれない書き込みがある |
| L6 | `search<NonPV>(…, depth, …)` を depth <= 0 で呼ぶ箇所を、 呼び出し側で `qsearch` に振り分ける (search の入口の `if (depth <= 0) return qsearch(…)` を通らない) | 呼び出し側で depth が分かっている |

| 本実装 | 場所 |
|---|---|
| 束 | `<engine>/source/srch_bundle_a.h` (`SDiv32`、 `SearchDivs`)、 `yaneuraou-search.cpp`、 `yaneuraou-search.h`、 `movepick.h`、 `evaluate.h` |
| define | `-DP2_SRCH_BUNDLE_A`、 検算 `-DP2_SRCH_BUNDLE_A_VERIFY` |

### L2 の式と証明 (そのまま使える)

```
正の除数 d (1 <= d <= 2^31-1) で、 x (0 <= x <= 2^31) について floor(x / d) を求める:
  l = ceil(log2 d)、 k = 31 + l、 M = ceil(2^k / d)   (M <= 2^32、 x*M <= 2^63 で uint64 に収まる)
  floor(x * M / 2^k) == floor(x / d)
証明: e = M*d − 2^k とおくと 0 <= e <= d−1 < 2^l。 x = q*d + r (0 <= r < d) と書くと
  x*M / 2^k = q + (r + x*e/2^k) / d。 x*e < 2^31 * 2^l = 2^k より x*e/2^k < 1、 よって r + x*e/2^k < d、 床は q。
負の n は |n| を割って符号を戻す (= C++ の 0 方向切り捨て)。 |INT_MIN| = 2^31 も範囲内。
```

## 3. 不変条件

- 各項目で、 新しい式の値が旧式と全入力で一致する (M1: 商、 L1: 採点の値、 L2: 商、 L5: 読まれる値、 L6: 返す値)。
- L2: 除数 >= 1 (option の範囲の下限で保証する。 0 を「項を使わない」の意味にしている係数は、 使う側が 0 を見て割らないので、 magic は 1 で埋める)。
- L2: 係数は探索中に変わらない。 変わるのは setoption だけなので、 go の開始時に作り直せば必ず反映される。

## 4. 変更の最小単位

項目ごとに独立。 1 項目ずつ入れて、 そのたびに検算と関門を通す。

1. 検算の枠を先に作る: `VERIFY` のときは各箇所で旧式と新式を両方計算し、 不一致なら stderr に出して abort。 終了時に項目ごとの照合回数を出す。
2. L2 → M1 → L5 → L6 → L1 の順 (小さい順)。
3. 全部入れたら検算 (5 局面 x 20 万) と関門 (M00)。

## 5. bit 一致の検証

検算 14 項目 (束の各箇所) で total=14,779,636、 mismatch=0 (11-16、 本番 option)。 発火回数の目安 (回数であって得ではない): L1_quiet 約 712 万、 M1_delta1 / M1_reduction 約 143 万、 L5_flags 約 112 万、 L2_lmrdepth 約 111 万。 関門 49 局面完全一致 (11-12)、 段 1 として G-a / G-b / G-c 全一致 (INFREP の on / off とも)。

## 6. NPS の測り方と期待値

我々: 単独 101.53% [100.86, 102.18] (11-12)。 段 1 (+ IDXV2 + ROWPF、 PGO 同士 64 周) で 104.19% [103.43, 105.01]。 移植先での期待値は、 移植先に残っている項目の数で決まる。 係数が定数の本家では L2 は 0、 HEAD では M1 も対象外。

## 7. はまりどころ

- L2 の magic の作り直しを忘れると、 setoption で変えた係数が反映されない。 go の開始時 (各 worker の start_searching) に作る。
- L1 の「本番で常に偽」は、 設定を変えると偽でなくなる。 速い経路はスイッチが全部 off のときだけ選ぶ (起動時に 1 回判定)。
- L6 は PvNode の扱いを変えないこと (PV と NonPV で qsearch のテンプレートが違う)。

## 8. 根拠

実験記録 (設計書・事前登録・判定ログ) は非公開です。
