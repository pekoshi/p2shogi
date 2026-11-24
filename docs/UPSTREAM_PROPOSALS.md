# やねうら王への高速化の提案候補

- 方針: 本家への提案は**エンジンの高速化だけ** (学習器・道具・探索の挙動を変える変更・SPSA の値は提案しない)。 PR は任意で、 出すとすれば公開の後です。
- 照合した本家: `yaneurao/YaneuraOu` の HEAD `ad45ae58` (2026-11-16)。 我々の分岐点は `717da87` (2026-06-03)。 本家のソースは**読んだだけ**で、 **本家のビルドでは 1 件も測っていません**。 期待値はすべて推測です。
- 我々の測定は i7-7700 (AVX2 のみ、 VNNI なし) です。 本家の主な測定は AVX-512 VNNI なので、 AVX2 の話は本家の主戦場ではない点に注意してください。
- 各候補の実装仕様は `speedups/` にあります (本家の NNUE に合わせて書き直す前提)。

## 0. 先に結論

| 番号 | 案 | 我々の根拠 (実測) | 本家 HEAD の現状 | 期待値 (推測) | 大きさ |
|---|---|---|---|---|---|
| **P1** | AVX2 ターゲットの `-march=corei7-avx` を `-march=haswell` (または `-mtune` だけ) に | +約 10% (clang、 bench nodes 一致) | `source/Makefile:615` に残っている | 0〜+10% | 1 行 |
| P2 | AVX2 (VNNI なし) では fc0 を疎でなく密に回せるようにする | 疎は 98.84% [98.24, 99.48] (64 周) = 密が +1.17% 速い。 非零 chunk 率 約 70% | AVX2 でも疎 (`find_nnz_explicit`) | −1〜+1.5% (net の非零率しだい) | 小 |
| P3 | AVX2 の fc0 で maddubs の結果を i16 のまま 8 chunk 足してから madd (G8)。 溢れないことを読み込み時に重みから確かめた head だけ | 幅 2048 で −244〜−393 cycle / 評価。 幅 1024 では差なし | 無い | 幅 1024 で 0〜+0.5%、 2304 で +1〜2% | 中 |
| P4 | 差分更新 (と Finny の refresh) の最後の列ループで transform まで書く融合 | 幅 2048 で −430 cycle / 評価、 refresh 側 −65〜−120 | AVX-512 の `NNUE_SFNN_FUSED_FT` は transform と fc0 の融合で別物。 AVX2 は融合なし | 幅 1024 で +1〜2%、 2304 で +2〜3% | 中〜大 |
| P5 | 本数固定の差分核を 3/3・4/4 まで広げ、 既定で有効に | 本体の acc −61〜−253 cycle / 評価 (対照は我々の汎用ループ) | 1/1・2/2 だけ、 opt-in (`NNUE_FT_SPECIALIZED`) | 0〜+1% (多段の追いつきを使うとき) | 小 |
| P6 | fc0 の重みの先読み | 密な fc0 で fc0 + 後段 −15〜17% (rdtsc) | 無い | 0〜+2% | 小 |
| P7 | PGO の採取の点検 (net が読めたかをログで / 採取局面を中盤・終盤に / 熱い経路を変えたら採り直す) | 存在しない net で採取して −7.0%、 直して +7.0% / 採り直し +3.08% | `tools/build/pgo.py` は `readyok` と `Nodes/second` だけを待つ。 採取は bench の既定 4 局面 depth 14 | 失敗時の −7% を防ぐ。 局面の寄せ方で 0〜+2% | 小 (スクリプトと文書) |
| P8 | Win64 で熱い関数のフレームが 4 KB を超えていないかの確認 (`__chkstk`) | 4.7 KB → 3.3 KB を含む束で +4.81% (単独は不明) | 未確認。 幅 2304 では transform の出力 + 後段の buffer が評価関数のフレームに乗る | 不明 (超えていなければ 0) | 確認だけなら `-Wframe-larger-than=4096` |
| P9 | SRCH_BUNDLE_A の残り (下の 3 章) | 我々の束で 101.53% | 項目の大半は本家に当たらない | 0〜+0.5% | 小 |

**最初の 1 本は P1** を勧めます (1 行で、 測る前に逆アセンブルで当否が分かる)。

## 1. 各案の説明

### P1 `-march=corei7-avx` をやめる

`-march=corei7-avx` は Sandy Bridge 向けの調整で、 clang は 256 bit の unaligned の load / store を 128 bit x 2 に割ります (LLVM の `FeatureSlowUAMem32`。 gcc も同様の split の既定が入る)。 AVX2 を持つ CPU は Haswell 以降なので、 AVX2 ターゲットで Sandy Bridge の調整をする理由はありません。

- 変更: `-DUSE_AVX2 -DUSE_BMI2 -mbmi -mbmi2 -mavx2 -march=corei7-avx` → `… -march=haswell` (FMA が入るので、 浮動小数の結果を変えないなら `-ffp-contract=off` も)。
- 測る前の確認: 熱い関数 (FT の差分更新、 transform、 fc0) の逆アセンブルで `vinsertf128` / `vextractf128` / `vinserti128` / `vextracti128` を数える。
- 本家の FT は整列 load (`vec_load`) が多いので、 我々の +10% より小さいはずです。 unaligned の load が残る fc0・後段で効く見込み。
- 仕様: `speedups/S05_compiler_flags.md`

### P2 / P3 fc0 (AVX2)

本家は AVX2 でも Stockfish 式の疎な fc0 (非零の 4 byte chunk だけを積和) を使います。 我々の net では非零 chunk の率が約 70% と高く、 密に回すほうが速かった (疎 98.84%)。 重みの並び (`get_weight_index`) は我々の密な版と同じなので、 `find_nnz` を飛ばして全 chunk を回す経路を足すだけです。 net の非零率を測ってから選ぶのが筋です (疎な net なら疎が勝つ)。 P3 (G8) は密が前提で、 幅の広い net (2304) で効く見込みです。 仕様: `speedups/S20_opn_r2.md`。

### P4 融合

差分更新の最後の列ループで accumulator を書いた直後のレジスタから transform を計算して書く。 「その評価で差分を取る視点・Finny で作る視点・計算済みの視点」の 3 通りを分ける必要があります。 🚨**我々の本体 (幅 1024) にもまだ当てていません** (OPN の幅 2048 で測った値)。 先に我々の本体で測ってから出すのが筋です。 仕様: `speedups/S19_opn_f2.md` (F2-6)、 `S20_opn_r2.md` (R2-3)。

### P5 本数固定の核

多段の追いつき (本家 `USE_LAZY_EVALUATE=N`、 N >= 2) を使うと、 引く行・足す行が 3/3、 4/4 になる評価が出ます。 本家の核は 1/1・2/2 だけです。 仕様: `speedups/S20_opn_r2.md` (R2-2)。

### P6 fc0 の重みの先読み

LayerStack が 9 / 16 あると、 bucket ごとの fc0 の重み (16 KB) が L1 から落ちやすい。 疎な fc0 なら「次の非零 chunk の列」を先読みする形になります。 我々の本体 fc0 にも final2 の F03 として入れました (3 本合わせて NPS 101.74%、 28 周で打ち切り・区間なし。 F03 単独の値はありません)。 仕様: `speedups/F03_fc0_pf.md`。

### P7 PGO の点検

本家の `pgo.py` は採取の成否を net の読み込みで確かめていません。 net のパスを誤ると評価関数が profile に入らず、 PGO 版が遅くなります (我々は −7%)。 仕様: `speedups/S04_pgo_procedure.md`。

### P8 `__chkstk`

Win64 (mingw / MSVC) では 4 KB を超えるフレームの関数の入口で `__chkstk` が呼ばれます。 熱い関数で起きていれば消すだけで効きます。 仕様: `speedups/S08_fixed_cost_cleanup.md`。

## 2. 本家に既にある / 提案にならないもの

| 我々のもの | 本家 | 判断 |
|---|---|---|
| fc0 の重みの 64 B 整列 (S07) | 各層が最初から `alignas(64)` | 我々の不具合を直しただけ |
| Finny 表・多段の追いつき・本数固定の核 (1/1・2/2) | ある (`USE_FINNY_TABLES` 08-05、 `USE_LAZY_EVALUATE=N` 11-12、 `NNUE_FT_SPECIALIZED` 10-25)。 どれも opt-in | 本家のほうが先か同時期。 我々は本家から持ってきたものではない |
| 利きライブラリを外す (S01) | 本家の既定がもともと利きライブラリなし | 我々は threat のために有効にしていたのを戻しただけ |
| lazy evaluate | 本家の機能 | — |
| null move で accumulator を写さない | 同等 (11-13 以降は accumulator を worker の stack に) | — |
| threat・KH・OPN・LZ・L1Q128 の核 | 本家に threat 入力が無い | 資料として紹介するだけ (`speedups/S11_threat_lz.md` ほか)。 本家に threat 入力が入るときの参考 |

逆に、 **本家にあって我々の本体に無かったもの** (公開版の final2 で一部を取り込んだ): transform で FT の重みを読み込み時に並べ替えて 32 B を 1 回で書く形 (我々の F01 は permute を残した版)、 多段の追いつきで駒番号ごとに最初と最後の位置だけを残す処理 (F02)、 計算済みの子から親を逆向きに作る処理 (`backward_update_incremental`。 我々の S25 BACKMAT は threat 側の同じ発想)、 Finny で差分のほうが重いときに bias から作り直す / 多重集合で相殺 (`USE_FINNY_RESET` / `NNUE_FINNY_CANCEL`)。

## 3. SRCH_BUNDLE_A の残り項目の照合

我々の束 (`speedups/S14_search_bundle_a.md`) の 5 項目を本家 HEAD と照合しました。

| 項目 | 中身 | 本家 HEAD に当たるか |
|---|---|---|
| M1 | NonPV の reduction の `delta * A / rootDelta` を窓ごとに 1 回だけ割る | **当たらない**。 HEAD の `reduction(i, d, mn)` から delta の項が消えた (`717da87` には在る) |
| L1 | MovePicker の静かな手の採点で、 本番で常に偽の機能を外した速い経路 | **当たらない**。 我々が足した機能 (打つ手専用の contHist など) の分岐 |
| L2 | 実行時の係数による除算を magic number の乗算に | **当たらない** (配布ビルドは係数が `constexpr` で、 コンパイラが既に乗算にしている)。 `ENABLE_SEARCH_TUNE` のビルドでだけ意味がある |
| L5 | node の入口の thread_local の書き込みを省く | **当たらない** (我々の計測用の変数) |
| L6 | `search<NonPV>` を depth <= 0 で呼ぶ箇所を、 呼び出し側で qsearch に振り分ける | **当たりうる**。 HEAD の search の入口に `if (depth <= 0) return qsearch<…>(…)` がある (`yaneuraou-search.cpp:2484`)。 呼び出し側の振り分けで関数呼び出しと入口の処理を省けるかは要測定 |

→ 束 A は本家にはほぼ当たりません (P9 の期待値は小さい)。

## 4. 本家の 11-16 前後の外部 PR 7 本との重なり

| PR | 中身 | 我々の候補との重なり |
|---|---|---|
| #336 | AVX-512 で FT のレジスタ数を 24 に統一 | なし (AVX-512) |
| #337 | Bitboard の extract64 / insert64 を `p[n]` の読み書きに | なし |
| #349 | isready の keep-alive スレッドを条件変数で待機 | なし |
| #350 | Win64 で探索・SEE・1 手詰め・next_move を SysV 呼び出し規約に (`__attribute__((sysv_abi))`) | **近い領域**: Win64 の呼び出しの費用。 P8 (`__chkstk`) とは別の機構だが、 Win64 の熱い関数を見直す点で同じ方向。 重複はない |
| #353 | NNUE 読み込み用の一時領域を large page を使わずに確保 | なし (我々は large page を使わない) |
| #354 | Finny キャッシュを large page に確保 | なし (同上) |
| #357 | Stochastic Ponder の ponderhit 時刻を思考開始時刻に | なし |

→ 7 本とも我々の候補 (P1〜P9) と直接は重なりません。 P1 (`-march`) は 7 本のどれも触っていません。

## 5. 本家の最新版での検証手順 (パッチごと)

1. 本家の HEAD を clone し、 標準の NNUE エディションを README の手順でビルドできることを確かめる。 評価関数は本家が案内する配布物を各自で入手 (再配布しない)。
2. 基準: パッチ前の exe で `bench` のノード数、 `go nodes` の全一致の器 (`speedups/M00_measurement.md`: Threads=1、 局面ごとに `isready`、 先に base 対 base) の出力、 NPS (CPU 時間の ABBA、 64 周以上、 PGO の有無を固定)。
3. パッチを当てる (我々のコードの移植でなく、 本家のコードに対して書き直す)。
4. bit 一致: `bench` のノード数の一致、 `go nodes` で手・nodes・depth・score・pv の全一致 (200 局面 x 20 万 + 24 局面 x 200 万)。
5. NPS: 本家の標準 net・既定のビルド設定で、 **nopgo と PGO の両方**。 区間が 0 をまたぐものは出さない。 目安は PGO 版で +0.5% 以上かつ区間の下限が正。
6. gcc と clang の両方でビルドが通ること (可能なら MSVC)。 SIMD に触るパッチは SSE4.2・AVX-512・ARM の枝もコンパイルだけは通す。
7. PR の本文: 環境 / 本家の commit / 評価関数 / bench のノード数の一致 / `go nodes` の全一致の局面数とノード数 / NPS の値・区間・周回数。 本家の流儀 (`.clang-format`、 Makefile と vcxproj の両方、 日本語のコメント) に合わせ、 我々の略号・日付・実験名を持ち込まない。

## 付録 A. 本家の SFNN 実装と p2nnue の手法の比較

p2nnue は本家の `eval/nnue/` を使わない独自実装 (rshogi の Rust 版から移したもの) で、 本家の SFNN 実装とは別に育ちました。 形 (HalfKA_hm 系・FT 1024・fc0 1024 → 16 (15 + skip)・sqr/lin 30 → 64 → 1) はほぼ同じです。

| 手法 | 我々 (公開版) | 本家 HEAD (11-16) |
|---|---|---|
| Finny | あり (視点 x 玉の区分 45 x 鏡像、 駒番号ごとの差分) | あり (`USE_FINNY_TABLES`、 opt-in。 鍵は玉の升 81、 差分は特徴の索引リストの比較) |
| Finny の refresh で 2 か所へ同時に書く | あり | あり (`update_accumulator_tiled_to_two`) |
| 差分更新のタイル | 4 レジスタ (64 列) | 16 レジスタ (256 列、 既定) |
| 多段の追いつき | あり (最大 128 手。 公開版は final2 の F02 で駒単位に畳む) | あり (`USE_LAZY_EVALUATE=N`、 opt-in、 駒単位) |
| 本数固定の核 | 1〜4 本 (`r2_rows_fx<N>`) | 1/1・2/2 (`NNUE_FT_SPECIALIZED`、 opt-in) |
| transform | `clamp(0..254)` の対の積 >> 9。 公開版は F01 で 32 B x 1 の store (以前は 16 列ごと 8 B x 2) | 同じ式。 重みを読み込み時に並べ替えて 32 B を 1 回で書き、 片側の max を省く |
| fc0 (AVX2) | 密 (重みを [入力 4 byte][出力][4] に並べ、 入力 4 byte を broadcast → maddubs → madd) | 疎 (`find_nnz_explicit`) |
| 重みの整列 | 64 B (09-29 に直した) | 最初から `alignas(64)` |
| PGO | clang の instr PGO (自前の採取) | あり (`tools/build/pgo.py`、 11-09 で整理) |
| `-march` (AVX2) | `-march=native -ffp-contract=off` | `-march=corei7-avx` |
| FT を i8 で持つ | 試して負け (AVX2、 96.06%) | `NNUE_FT_SPARSE8` (AVX-512 かつ幅 2304 専用) |
