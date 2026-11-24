# F01 transform の出力を 32 B の store 1 回で書く (`P2_TR_STORE32`)

- 状態: **採用** (final2。 検算 不一致 0 / 関門 完全一致 / NPS は F01〜F03 の合計で 101.74%、 区間なし) / bit 一致: 一致 / requires: M00
- 人間向け: `../SPEEDUPS_STORY.md` 6 章
- 出どころ: やねうら王本家の `nnue_feature_transformer.h` の transform の形 (分岐点 `717da87` に既にある)

## 1. 要約

NNUE の transform (`out[j] = (clamp(a[j], 0, 254) * clamp(a[j+H], 0, 254)) >> 9`、 H = FT の半分) を AVX2 で書くとき、 16 列ごとに packus の結果の lane を取り出して 8 B の store を 2 回していたのを、 32 列ごとに packus → `permute4x64` → 32 B の store 1 回にする。 あわせて片側 (hi 側) の `max(0)` を省く。

## 2. 前提

- transform が「両側を clamp して mulhi (またはシフト付き乗算) → packus で u8」の形。
- FT の半分 H が 32 の倍数。

| 一般の言葉 | 本実装 |
|---|---|
| transform | `transform_pair` / `transform_pair_sum` (`p2nnue.cpp`)、 OPN の `opn::transform`・`apply_rows_tr2_t` (`p2nnue_opn.h`)、 R2 の融合核 |
| define | `-DP2_TR_STORE32`、 検算 `-DP2_TR_STORE32_VERIFY` |

## 3. 不変条件

- 出力の各バイトが従来と同じ。
- hi 側の `max(0)` を省けるのは、 lo 側を [0, 254] に clamp した後の「lo x hi」で hi が負なら積が 0 以下になり、 packus (符号付き → u8 の飽和) が 0 にするから。 lo 側の `max(0)` は省けない。

## 4. 変更の最小単位

1. 32 列 (lo 側 i16 x 32 と hi 側 i16 x 32) を読み、 2 組の 16 列について clamp・乗算・シフトして i16 x 16 を 2 本作る。
2. `packus_epi16` で 2 本を u8 x 32 に詰める。 AVX2 の packus は 128 bit lane ごとに詰めるので、 `permute4x64_epi64(x, 0b11011000)` で並びを戻してから 32 B を store。 (本家は重みを読み込み時に並べ替えておき、 permute も省いている)
3. hi 側の `max(0)` を外す。
4. 検算: SIMD の出力を従来のスカラーの式と全要素比較 (`TR_STORE32_VERIFY`)。 本体と OPN の両方。
5. 関門 (M00)。

## 5. bit 一致の検証

検算 `TR_STORE32_VERIFY main=1677755 opn=1710201 mismatch=0` (本体 / OPN の transform、 融合核を含む)。 関門 (G-a / G-b / G-c x INFREP off/on) はすべて完全一致。

## 6. NPS の測り方と期待値

M00。 的の大きさ: transform は評価 1 回 392 cycle (評価の 4.3%、 11-10)。 期待値 +0.5〜1% (推測)。 結果: 3 本 (F01〜F03) を一緒に入れた exe で 28 周で打ち切り、 101.74% (区間なし、 3 本の合計。 単独の値はない)。

## 7. はまりどころ

- packus の lane 内の詰め方を忘れると、 列が 8 個ずつ入れ替わった出力になる (検算で必ず捕まえる)。
- 我々の 9 月の束 A345 に「transform の store 半減」があったが、 いつの間にか本体の transform から消えていた (経緯不明)。 改造を足すときは、 最終ビルドの前処理の結果にそれが在ることを確かめる。

## 8. 根拠

実験記録 (設計書・事前登録・判定ログ) は非公開です。 関連: 本家 `source/eval/nnue/nnue_feature_transformer.h` の transform。
