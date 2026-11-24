# S23 / S24 Finny 表・多段の追いつき・lazy evaluate (土台から持っているもの)

- 状態: 採用 (我々の on/off の実測は無い) / bit 一致: 一致 / requires: M00
- この仕様は「我々の実装の形」を記録するもので、 期待値は示せません。 やねうら王の本家にも同じ手法が opt-in で入っています (`../UPSTREAM_PROPOSALS.md` 付録)。

## 1. 要約

| 手法 | 中身 | 我々の場所 | 本家 (HEAD 11-16) |
|---|---|---|---|
| Finny 表 | 視点 x 玉の区分 (45) x 鏡像 の 180 個の accumulator を持ち、 玉が動いた視点は「前回その区分で作った acc + 駒番号ごとの差分」で作る。 refresh で Finny の acc と局面の acc に同じループで 2 回 store | `p2nnue.cpp` `refresh_perspective` | `USE_FINNY_TABLES` (opt-in、 08-05) |
| 多段の追いつき | 計算済みの祖先まで最大 128 手遡り、 途中の手の引く行・足す行を全部集めて「祖先 + Σ足す − Σ引く」を 1 回の列ループで作る。 玉が動いたら打ち切り、 行が 96 本を超えたら Finny に落とす | `p2nnue.cpp` `ensure_perspective` | `USE_LAZY_EVALUATE=N` (opt-in、 11-12)。 本家は駒番号ごとに最初と最後の位置だけを残す (我々の F02 がこの形) |
| lazy evaluate | 探索で評価を必要になるまで遅らせる | `-DUSE_LAZY_EVALUATE` (本家由来) | 本家の機能 |
| null move で acc を写さない | null move の子は差分 0 で親から継承 | `position.cpp` | 同等 |

## 2. 前提

FT の差分更新を持つ NNUE。 玉の位置で特徴が全部変わる入力 (HalfKA 系)。

## 3. 不変条件

- Finny: 区分ごとの acc は「その区分で最後に作った局面」の値で、 差分は駒番号ごとの BonaPiece の違いから作る。 結果は全計算と同じ (i16 の wrapping 加算)。
- 多段の追いつき: 途中で玉が動いた視点は遡りを止める (区分が変わるため)。

## 4. 実装の目安

我々の数字はありません。 実装するなら M00 の手順で、 Finny と多段の追いつきを別々に on/off して測ってください。 🚨本家の `USE_LAZY_EVALUATE=N` は値付きで、 我々の `USE_LAZY_EVALUATE` (値なし = 1) と同じ名前で意味が違います。 本家の新しい版に追従するときは注意。

## 5. 根拠

実験記録 (設計書・事前登録・判定ログ) は非公開です。 関連: `../UPSTREAM_PROPOSALS.md` 付録 A (本家の SFNN 実装との手法の比較表)。
