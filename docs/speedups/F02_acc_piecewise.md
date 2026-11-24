# F02 多段の追いつきを駒単位に畳む (`P2_ACC_PIECEWISE`)

- 状態: **採用** (final2。 検算 不一致 0 / 関門 完全一致 / NPS は F01〜F03 の合計で 101.74%、 区間なし) / bit 一致: 一致 / requires: S23 (多段の追いつき)、 M00
- 人間向け: `../SPEEDUPS_STORY.md` 6 章
- 出どころ: やねうら王本家の `update_from_computed_ancestor` (`nnue_feature_transformer.h`) と同じ考え方

## 1. 要約

FT の accumulator を計算済みの祖先から何手分も差分で作るとき、 手ごとの「引く行・足す行」を全部足し引きしていたのを、 駒番号ごとに「祖先での特徴 (最初の old)」と「今の特徴 (最後の new)」だけに畳む。 途中で 2 回以上動いた駒も引く 1 本・足す 1 本で済み、 元の位置に戻った駒は 0 本になる。

## 2. 前提

- 差分の記録 (DirtyPiece) が「駒番号・変化前の特徴・変化後の特徴」を持つ。
- 同じ駒番号の連続する変化では、 次の変化前 = 前の変化後 (old(k+1) == new(k))。
- 追いつきの区間の中で自玉が動かない (動いたら打ち切る、 は S23 の既存の規則)。
- 行は i16 で wrapping 加算 (i8 行 + 補正の構成では使えない: `P2_FT_I8` と排他)。

| 一般の言葉 | 本実装 |
|---|---|
| 畳む関数 | `fold_pieces(chain, i_hi, i_lo, persp, pob, pnb)` (`p2nnue.cpp`、 `ensure_perspective` の直前) |
| define | `-DP2_ACC_PIECEWISE`、 検算 `-DP2_ACC_PIECEWISE_VERIFY` |

## 3. 不変条件

- Σ (new − old) は new(最後) − old(最初) に等しい (行の多重集合としての和が同じ)。 よって accumulator と PSQT は従来と同じ値。
- 駒番号が範囲外なら畳まずに従来の経路 (または Finny) に落とす。

## 4. 変更の最小単位

```
slot[駒番号] = −1 で初期化、 np = 0
for 手 in 古い順:
    for 変化 in その手の dirty:
        pn = 駒番号
        if slot[pn] < 0: slot[pn] = np; old[np] = 変化前; new[np] = 変化後; ++np
        else:            new[slot[pn]] = 変化後
残すのは old != new のものだけ → 引く行の列 = old、 足す行の列 = new
```

1. 上の関数を足し、 追いつき (d >= 2) のときだけ使う (d = 1 は従来どおり)。
2. 検算: 差分で作った本体の acc / PSQT を「bias + 今の局面の全駒の行」と全要素比較。 畳む前と後の行の本数も数える (効く局面の割合が分かる)。 OPN の accumulator も同じ。
3. 関門 (M00)。

## 5. bit 一致の検証

検算 `PIECEWISE_VERIFY checked=4237159 multi=35720 rows_before=354714 rows_after=282774 opn_checked=2956491 opn_multi=0 mismatch=0`。 関門 (G-a / G-b / G-c x INFREP off/on) はすべて完全一致。

## 6. NPS の測り方と期待値

M00。 効くのは追いつきが 2 手以上の評価だけで、 その中でも同じ駒が 2 回動く場合 (または元に戻る場合)。 行の本数の減り方 (検算の `rows_before` / `rows_after`) から先に期待値を見積もる。 我々の検算では 2 手以上の追いつき 35,720 回で行が 354,714 → 282,774 本 (−20%)。 結果: 3 本 (F01〜F03) を一緒に入れた exe で 28 周で打ち切り、 101.74% (区間なし、 3 本の合計。 単独の値はない)。

## 7. はまりどころ

- old(k+1) == new(k) が成り立たない記録の仕方 (同じ手の中で同じ駒番号が 2 回出るなど) なら畳めない。 記録の仕方を先に確かめる。
- 自玉が動いた視点は区分が変わるので畳む前に打ち切る (既存の規則に従う)。

## 8. 根拠

実験記録 (設計書・事前登録・判定ログ) は非公開です。 関連: 本家 `source/eval/nnue/nnue_feature_transformer.h` の `update_from_computed_ancestor`。
