# S05 コンパイラとフラグ: `-march=corei7-avx` をやめる (S06 clang 23 + 静的リンクを含む)

- 状態: 採用 / bit 一致: 一致 (`-ffp-contract=off` を付けること) / requires: M00
- 人間向け: `../SPEEDUPS_STORY.md` 3.1 節

## 1. 要約

やねうら王の Makefile の AVX2 ターゲットの `-march=corei7-avx` (Sandy Bridge 向けの調整) をやめて `-march=native -ffp-contract=off` にした。 corei7-avx の調整では clang が 256 bit の unaligned load / store を 128 bit x 2 (`vinsertf128` / `vextractf128` など) に割る。 NPS +約 10% (504,975 → 中央 約 554k)、 bench のノード数は一致。

## 2. 前提

- AVX2 で動く SIMD のコード (NNUE の推論) があり、 unaligned の 256 bit の load / store を含む。
- `-march=native` は配布には向かない (ビルドした CPU に依存する)。 配布するなら `-march=haswell` (または `x86-64-v3`) など AVX2 世代の名前を明示するか、 `-mtune` だけを替える。

| 一般の言葉 | 本実装 |
|---|---|
| フラグの場所 | `<engine>/source/Makefile` の AVX2 の分岐 (本家 HEAD では `source/Makefile:615` 付近) |
| コンパイラ | llvm-mingw 20260908 (clang 23)、 ucrt、 x86_64。 libc++ を `-static` でリンク |

## 3. 不変条件

- 浮動小数の結果が変わらないこと。 `-march` を上げると FMA が使えるようになり、 コンパイラが a*b+c を FMA に縮約すると丸めが変わる。 `-ffp-contract=off` で縮約を止める。 我々の探索・評価の整数部分は影響を受けないが、 進行度の計算などに float がある。

## 4. 変更の最小単位

1. AVX2 の分岐の `-march=corei7-avx` を `-march=native` (配布なら `-march=haswell`) に替え、 `-ffp-contract=off` を足す。
2. 測る前の当否の確認: 熱い関数 (FT の差分更新、 transform、 fc0) を逆アセンブルし、 `vinsertf128` / `vextractf128` / `vinserti128` / `vextracti128` で 256 bit を割っている箇所が減ったかを数える。
3. bench (または関門) でノード数の一致を確かめる。
4. (S06) コンパイラを新しくし、 C++ の実行時ライブラリを静的リンクする (配布先の dll の取り違えを防ぐ)。 版を替えたら関門を通す。

## 5. bit 一致の検証

`bench` のノード数の一致 (我々: `bench 256 1 17` で 5,357,284 完全一致) と関門 (M00)。 S06 は 89/89 局面一致。

## 6. NPS の測り方と期待値

我々: +約 10% (07-31、 区間なし、 PGO 併用)。 本家のコードは FT の load が整列 (`vec_load`) である箇所が多いので、 効きは我々より小さい見込み (0〜+10%、 推測)。 unaligned の load が残る fc0・後段で効く。 S06 単独の NPS は測っていない (採用の決定だけが記録にある)。

## 7. はまりどころ

- `-march=native` の exe は、 ビルドした CPU より古い CPU で落ちる (不正命令)。
- 本家は gcc でもビルドする。 gcc でも corei7-avx の調整で分割の既定が入るので同じ傾向のはずだが、 我々は clang でしか測っていない。

## 8. 根拠

実験記録 (設計書・事前登録・判定ログ) は非公開です。 関連: `../UPSTREAM_PROPOSALS.md` P1。
