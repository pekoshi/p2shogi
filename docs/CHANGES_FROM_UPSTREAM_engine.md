# エンジンの改造点 (やねうら王 `717da87` から)

- 土台: やねうら王 `717da871e7a620702b8b9433bd8f9f181710435a` (V9.40、 2026-06-03)。 公開リポジトリ (https://github.com/pekoshi/p2shogi 、 master) はこの commit から GitHub の fork として履歴をつないでいます。 `git diff 717da87` で全差分が見えます。
- ライセンス: GPLv3 (やねうら王から継承)。
- この文書は**機能ごと**に、 目的・実装場所・実測値をまとめます (根拠にした実験記録は非公開)。 パスの書き方は `OVERVIEW.md` 0 章。
- 高速化は件数が多いので一覧だけを 13 章に置き、 中身は `SPEEDUPS_STORY.md` と `speedups/` にあります。
- 11-17 以降に足したもの (v9) は 15 章にまとめました。

## 0. 差分の規模と取り方

### 0.1 規模 (公開前の作業ツリー、 2026-11-17 時点)

| 区分 | 量 | 中身 |
|---|---|---|
| 本家のファイルへの変更 | 27 ファイル、 +6,391 / −1,671 行 | 大半は `yaneuraou-search.cpp` (+約 6,000 行。 探索係数の option 化、 INFREP、 束 A、 SMP の実験機構、 定跡の抜け方) |
| 新規 (評価関数) | `source/eval/p2nnue/` (`p2nnue.cpp` 約 10,500 行、 `p2nnue_lz.h` 857、 `p2nnue_opn.h` 971、 `p2nnue_kh.h` 454、 `p2nnue_threat.h` 377、 `p2nnue_reader.h` 143 ほか) | P2NNUE 一式 |
| 新規 (探索) | `spsa_params.h` (約 600 行)、 `srch_bundle_a.h` (177)、 `smp_deleg.h` ほか | 係数の option、 高速化の束、 並列の実験 |

ファイル別の増減は `diff/engine_diffstat_vs_717da87.txt`。 🚨 これは 11-17 時点の数字です。 その後の v9 の追加 (15 章) で `p2nnue.cpp` は約 11,400 行になりました。 公開版でも実験・診断の define はソースに残っています (ビルドの define に入れなければ無効)。

### 0.2 自分で差分を取る手順 (読むだけ)

```bash
git clone <engine の公開 URL> && cd <engine>
git diff --stat 717da87 -- source          # ファイル別の増減
git diff 717da87 -- source/position.cpp     # 1 ファイルずつ
git diff 717da87 --diff-filter=A --name-only # 新規ファイル
```

改造のほぼ全部は `#if defined(P2_xxx)` で囲まれています。 各節の「元の define」の列は、 ソースを読むときの検索語として使ってください。 公開版の exe に入る define の組は 0.3 の表のとおりです。

### 0.3 公開版のビルド

| 項目 | 値 |
|---|---|
| コンパイラ | clang 23 (llvm-mingw 20260908、 ucrt、 x86_64)、 libc++ を静的リンク |
| フラグ | `-march=native -ffp-contract=off` (本家 AVX2 ターゲットの `-march=corei7-avx` をやめた。 13 章 S05) |
| PGO | clang の instr PGO。 配布する net 一式と公開版の option で、 実戦の 60 局面 x 100 万ノードを採取 (13 章 S04、 15 章) |
| 公開版 (v9) の DEFINES | `P2_ARCH_1024 P2_L1_MAIN=15 P2_THREAT_PROFILE=0 P2_MATE1PLY_NOEFFECT P2_THREAT_OUT=64 USE_LAZY_EVALUATE P2_THREAT_LZ P2_OPN P2_OPN_F2 P2_KH P2_OPN_R2 P2_LZ_IDXV2=1 P2_LZ_ROWPF=1 P2_SRCH_BUNDLE_A P2_LZ_D1FAST P2_TR_STORE32 P2_ACC_PIECEWISE P2_FC0_PF P2_FT_KS=3 P2_FT_OUT16=5 P2_FT_SPARSE16 P2_FC1_PREC P2_BUCKET_OFFSET P2_OPTIMISM` (エディションは `YANEURAOU_ENGINE_P2NNUE`。 `P2_OPTIMISM` の option は既定 0 で無効) |
| 配布 exe の sha256 | `p2shogi_v9.exe`: `e290955a5488a72d8447daaa1f77525bce9597ae23bdff0a8b36709c4d38f40b` |

`LONG_EFFECT_LIBRARY` (本家の長い利きライブラリ) は**使いません** (13 章 S01)。

---

## 1. 評価関数の全体 (P2NNUE エディション)

| 項目 | 内容 |
|---|---|
| 目的 | 本家の NNUE (`eval/nnue/`) では表せない入力 (threat・KH)・2 本の net の切り替え (OPN)・独自の量子化を持つ評価関数 |
| 実装場所 | `source/eval/p2nnue/p2nnue.cpp` (読み込み・差分更新・推論・USI option)、 `p2nnue_reader.h` (ファイル形式)。 エディションは Makefile の P2NNUE 用の分岐 |
| 由来 | 2026-05〜06 に rshogi (Rust) で書いた NNUE 推論を C++ に移植。 本家の SFNN 実装とは独立に育った (本家との手法の比較は `UPSTREAM_PROPOSALS.md` 付録) |
| ファイル | `EvalDir` (既定 `eval`) の固定名: `nn.bin` (本体) / `kh.bin` + `kh.bin.json` / `opn.bin` / `progress.bin`。 相対名は「exe のフォルダ / EvalDir / 名前」(本家の `nn.bin` と同じ作法)。 **どれかが読めなければ `isready` で Error を出して止まる** |
| 形式 | 学習器の `--output-format rshogi` が出す形式 (の書き直し)。 先頭の arch 文字列で層の形を判定し、 知らない形は読まずに止まる (例: L1Q128 や FTI32K8 の net を古い exe に渡すと止まる) |
| 既定値の変更 (11-16) | 既定値を「対局で使っていた値」に揃えた。 option で渡していた値を既定に焼いただけで、 渡す値が同じなら探索は変わらない (関門で確認) |

## 2. 入力と FT: HalfKA_hm・幅 1024・PSQT

| 項目 | 内容 |
|---|---|
| 目的 | NNUE の主入力。 玉の位置 (左右の鏡像で 45 区分) x 全駒の位置と持駒 = 73,305 行 |
| 構造 | FT 1024 列 x 2 視点 (i16)、 transform は `clamp(0..254)` の対の積 (前半 512 列 x 後半 512 列) → u8 1024 本。 PSQT (bucket ごとの線形項) を出力に足す |
| 実装場所 | `p2nnue.cpp` の `ensure_perspective` (多段の差分更新)、 `refresh_perspective` (Finny 表)、 `transform_pair` / `transform_pair_sum` |
| 本家との関係 | 形は本家の SFNN (`SFNN_halfkahm2_1024_15_64_*`) とほぼ同じ。 違うのは LayerStack の選び方と、 threat・KH・OPN が上に載ること |
| 元の define | `P2_ARCH_1024`、 `P2_L1_MAIN=15` |

## 3. threat 特徴 (216,720 → 64)

| 項目 | 内容 |
|---|---|
| 目的 | 「どの駒がどの駒に利いているか」(攻め駒・標的・相対位置の対。 玉は攻め駒にも標的にも入れない) を直接入力にする。 駒の当たり・ひも・浮き駒を評価が見られるようにする |
| 構造 | 216,720 次元の疎入力 → i8 の行 (1 行 64 B) → threat acc 64 列 x 2 視点 (i16)。 64 列のうち前半 32 を FT の [0,32)、 後半 32 を [512,544) に足す (対の積の両側に対称に入れるため)。 学習時は bucket を 4 群 (0 / 1-2 / 3-4 / 5-7) に分け、 群ごとに別の表を持つ (群化 net) |
| 実装場所 | `p2nnue_threat.h` (列挙の表)、 `p2nnue.cpp` の `ensure_threat_both` / `add_threat_dual` (全計算)、 `p2nnue_lz.h` (多段 lazy 差分 = LZ) |
| 経緯 | 7 月に全幅 (1024 列) で入れ、 「full threat 搭載」を公表。 税 (NPS の低下) が大きく、 9 月に出力幅を 64 に絞った (K の掃引で 64 が最適点。 税 −37.5% → −17%)。 税の本体は分岐予測ミスと特定 (11-05)、 LZ で一部を回収 (11-06〜) |
| 効果 | threat を剥いだ net 比で NPS は +22〜30% になる (= 税)。 それでも外さないのは公表した方針 |
| 試して採らなかったもの | 次元削減 (最軽量案でも −145 Elo)、 低ランク化、 FT への焼き込み、 Direct15 head、 部分 threat、 threat → L1/L2 直結 (TL1/TL2) ほか。 一覧は `SPEEDUPS_STORY.md` 10 章 |
| 元の define | `P2_THREAT_PROFILE=0`、 `P2_THREAT_OUT=64`、 `P2_THREAT_LZ` |

## 4. LayerStack と進行度 bucket

| 項目 | 内容 |
|---|---|
| 目的 | 序盤・中盤・終盤で fc0 以降の重みを切り替える |
| 構造 | 進行度 8 bucket (`progress8kpabs` = 玉と駒の絶対位置 (KP-Absolute) の重みの和から作る進行度を 8 区間に分ける)。 進行度の係数は NAGISA v3.1 の `progress.bin` (GPLv3)。 閾値も NAGISA と同じ |
| 実装場所 | `p2nnue.cpp` の `get_layer_stack_index`、 `ensure_progress` |
| 層 | fc0 1024 → 16 (15 本 + skip 1)、 sqr_crelu と crelu の連結 30 → fc1 64 → fc2 1 |
| FV_SCALE | 評価値の目盛りを進行度で変える動的設定: 序盤 26 / 中盤 28 / 終盤 28 (`FV_SCALE_OPENING` / `FV_SCALE` / `FV_SCALE_ENDGAME`)。 🚨FV_SCALE だけを変えても序盤局面では OPENING 側が効くので no-op になることがある |
| 試して採らなかったもの | 進行度 x 持駒の 16 bucket (P8H2)、 regime 切り替え、 4 層目 (L4)、 幅 1536、 narrow 1024-8-64 |

## 5. KH (浮き駒の追加入力)

| 項目 | 内容 |
|---|---|
| 目的 | どの駒にも守られていない駒 (浮き駒) を直接の入力にして、 当たりの危険を評価に入れる |
| 構造 | 浮き駒の 78 次元 (h 部) → 64 列。 FT 行 [0,32) と [512,544) に足す (threat と同じ位置)。 重みは `kh.bin` (f32、 列優先) と説明の `kh.bin.json` |
| 実装場所 | `p2nnue_kh.h`、 `p2nnue_kh_fast.h`、 `p2nnue.cpp` の評価入口 (threat acc + KH 行を一時 buffer で作る。 StateInfo の acc は書かないので LZ と独立) |
| 学習 | 本体の FT・threat・fc1・fc2・PSQT を凍結し、 KH と fc0 だけを 4 SB 学習 (`REPRODUCE.md` 4 章) |
| 効果 | 篩の KH on − off: bucket 0〜2 **+0.0055 ±0.0022** / 3〜7 **+0.0048 ±0.0009**。 NPS の税 −0.34% [−0.93, +0.26] (64 周)。 見積もり (−4〜−9%) より大幅に小さかった。 採用は 11-14 のユーザー決定 |
| v9 | 配布の `kh.bin` は KH の重みを 1.25 倍したもの (11-17 の決定。 `kh.bin.json` の `kh_scale_a`) |
| 元の define | `P2_KH`、 option `KHFile` |

## 6. 量子化と L1Q128

v9 ではさらに FT の刻みと fc1 以降の丸めを細かくしました (15.1)。

| 項目 | 内容 |
|---|---|
| 目的 | fc0 の本体 15 行の量子化の誤差を減らす |
| 構造 | 学習器の量子化は FT QA=127 (積は 254 まで)、 層の重み QB=64。 L1Q128 は fc0 の本体 15 行 x 8 bucket だけを、 f32 の重み (`raw.bin`) から 1/128 刻み (i8、 飽和させる)・bias 1/16256 刻みで書き直す。 arch の 3 番目の token が `AffineTransformL1Q128[16<-2048]` になり、 エンジンは shift を 6 → 7 に変える |
| 実装場所 | エンジン: `p2nnue.cpp` (定数 shift 30 か所 → 変数 shift)。 net の書き出し: python `mk_l1q128.py` (学習は不要) |
| 効果 | 篩 (L1Q128 − 1/64 刻み): bucket 2-7 **+0.0027 ±0.0009**、 3-7 +0.0029 ±0.0009、 0-1 +0.0070 ±0.0036 (bucket 3 だけ −0.0021 ±0.0022)。 NPS は 99.75% [99.20, 100.32] (64 周、 PGO 同士) = 非劣化。 エンジンの整数推論と python の期待値が 130,965 局面で一致 |

## 7. OPN (序盤専用の副 net)

| 項目 | 内容 |
|---|---|
| 目的 | 序盤 (bucket 0〜2) の判断を DL 系エンジンの正確さに寄せる。 NPS・深さの低下は許容し、 質を最優先 |
| 構造 | HalfKA_hm K16 (玉区分 16 = 自玉 7〜9 段 x 5 筋 + その他。 26,064 次元 + factoriser 1,629) → FT 2048 x 2 → fc0 16 → 32 → 1。 threat なし・PSQT なし。 8 bucket の LayerStack を持つが学習は bucket 0〜2 だけ |
| 切り替え | 進行度から重み λ (0〜`OPN_LambdaMax`、 256 = OPN だけ) を作り、 OPN (bucket ごとに `OPN_Scale*` で目盛りを合わせた値) と本体を λ で混ぜる。 `OPN_ProgWidth=0` のとき λ は 256 か 0 だけ (混ぜない)。 λ=256 の局面では本体も threat も計算しない (`p2nnue_opn.h` の lambda の計算) |
| 継ぎ目と目盛りの既定値 | v9 の既定: `OPN_ProgHi=-1946` / `OPN_ProgWidth=0` / `OPN_LambdaMax=256` / `OPN_Scale0=1196`。 OPN が受け持つのは進行度 bucket 0 だけなので、 `OPN_Scale1` / `OPN_Scale2` は実際には効かない。 11-16 の確認対局の設定は -1299 / 0 / 256 / 1196・1204・1377 だった (Scale は OPN の篩の 1/SD 比から作った値。 L1Q128 で取り直した値 [1005, 1121, 1293] は、 幅 0 では本体と混ざらないので採らなかった)。 配布直前の sweep はしていない |
| 実装場所 | `p2nnue_opn.h`、 `p2nnue.cpp` の `opn_eval_only` ほか |
| 効果 | 篩 bucket 0-1: OPN 0.3016〜0.3019 対 本体 0.2555 (+0.046)。 対局での単独の効果は 50 局の仮対局では誤差の内 |
| 速さ | 幅 2048 は素の実装で本体の約 80% の NPS だった。 F2・R2 の 2 段の高速化と継ぎ目の幅 0 で、 1 手 1 秒の到達ノードは対 v4 110.0% [108.0, 112.0] まで戻した (13 章 S19〜S21) |
| 試して採らなかったもの | OPN を bucket 2 へ延長 (b2 で 0.369 で頭打ち、 b2 だけ追加学習すると b0 を −0.030 壊す)、 幅 512 / 1024、 bucket 0〜1 を受け持つ設定 (対局で負けた)、 FT も動かす学習のし直し (対局で勝てなかった) |
| 元の define | `P2_OPN`、 `P2_OPN_F2`、 `P2_OPN_R2` |

## 8. 探索の係数 (SPSA)

| 項目 | 内容 |
|---|---|
| 目的 | 我々の評価関数の目盛りと速さに合わせて探索の係数を合わせる |
| 実装場所 | `source/spsa_params.h` (係数の宣言と範囲)、 `yaneuraou-search.cpp` (本家の定数を `SPSA::XXX` に置換)。 USI option `SPSA_*` 約 180 個 + 時間管理 `SPSA_TM_*` |
| 値の出どころ | 多くは NAGISA v3.1 の SPSA の値の移植、 その上に我々の SPSA。 RFP・razoring・qsearch の margin は 9 月に 2 倍、 NMP の margin は半分 (対局設定)。 v9 では自己対局の SPSA (SPSA24 の 29 周目) の値を既定値に焼き込んだ (15 章) |
| 本家との違い (要点) | 我々は遅い静かな手を本家より強く縮め (LMR の手数の項 13 対 本家 65)、 静かな ttMove の 3 手延長をほぼしない (tripleMargin の !ttCapture 項 0 対 本家 218) |
| `HIST_CONFIDENCE_SCALE` (既定 150) | LMR の statScore の 3 成分が同じ符号のとき、 その効きを強める (150 で 1.32 倍)。 🚨 長く toml に書かずに暗黙で効いていた |
| 試して採らなかったもの | Stockfish の探索項の移植 (`SF_*`、 全部 off)、 王手延長、 qsearch の静かな王手、 打つ手専用の contHist、 打つ手の刈り免除 (DROPWIDE、 −4〜−6) ほか |
| 地平線問題・評価の楽観への腕 | OPTSRCH (14 腕)、 HORIZON (root の fail-low で窓を広げる FLW3 / FLW8、 係数を本家の値へ寄せる C1〜C4、 bucket 2 限定の腕) を篩にかけたが、 どれも採用に至らなかった (`FUTURE_WORK.md` 2 章)。 v9 の後の LMR の係数 2490 も不採用 (15 章) |

## 9. INFREP (`P2InferiorRepHistory`、 既定 true)

| 項目 | 内容 |
|---|---|
| 目的 | 優等・劣等局面 (盤と手番が同じで、 持駒が一方的に多い / 少ない) の判定を、 root より前 (対局の履歴) の局面との比較にも広げる。 NAGISA v4 と同じ挙動の推定 |
| 規則 | 千日手 (引き分け・連続王手) は従来どおり root 以降の繰り返しのときだけ返す。 劣等は ply >= 2 のときだけ履歴をまたいで返す (root の子で「相手が劣等」を返すと root がその手を無条件に選ぶ問題を避ける)。 履歴またぎは探索本体の呼び出しだけで、 詰み探索は旧条件のまま |
| 実装場所 | `position.cpp` の `is_repetition(ply, infrep_history)`、 呼び出し側 `yaneuraou-search.cpp` |
| 効果 | 関門 (bit 一致の器で off は従来と一致)。 NPS 99.56% (30 周、 判定前に打ち切り。 採用はユーザー判断)。 強さへの効果は確認対局の中に含まれる (単独の判定なし) |

## 10. 時間管理

| 項目 | 内容 |
|---|---|
| 本家からの変更 | 本家の式の係数を `SPSA_TM_*` に出した (値は本家と同じ)。 `SPSA_TM_MAX_BANK_PCT` (1 手の最大を「加算 + 持ち時間 x pct」に抑える。 既定 45)、 `SPSA_TM_RESERVE_INC_X10`、 `SPSA_TM_LOW_BANK_MS` (残り時間が少ないときの床の修正。 既定 0 = 従来と同じ) |
| 定跡を抜けた直後の時間配分の修正 (`BookExitTimeFix`、 既定 true) | 定跡・宣言勝ちなどで探索を飛ばした手のあと、 本家は `bestPreviousScore` に既定値 (−∞) を残すので、 定跡を抜けた最初の 1 手で fallingEval が下限に張り付き、 時間が約 4 割減る。 修正は「前回評価なし = VALUE_INFINITE にする」。 false で従来と一致。 定跡なしの探索では bit 一致を確認 |
| 罠 | 長手数で先手が時間切れになる床の問題、 `NetworkDelay2` が短い秒読みを食う問題 (`OVERVIEW.md` 2 章) |

## 11. 定跡まわりの拡張 (コードは公開・定跡ファイルは非公開)

| option | 中身 |
|---|---|
| `BookFile2` / `BookFile3` | 1 本目で引けなかったときに引く 2 本目・3 本目の定跡。 空なら何もしない |
| `BookExitFile` / `BookExitMargin` / `BookExitMaxPly` | 定跡を抜ける局面の指定と、 抜けるかどうかの margin・手数 |
| `BookRepetitionGuard` | 定跡の中で千日手に向かう手を避ける (既定 false) |

実装は `source/book/book.cpp`、 `yaneuraou-search.cpp`。 定跡ファイルとその元データは公開しません。

## 12. 並列 (Lazy SMP) の実験機構

既定はすべて off で、 Threads=1 でも Threads>1 でも既定値なら本家の Lazy SMP と同じ動作です。 診断と機構の説明は `SPEEDUPS_STORY.md` 9 章、 仕様は `speedups/P06_m4_tt_delegation.md`。

| option | 中身 | 状態 |
|---|---|---|
| `SPSA_ASPIRATION_THREAD_STEP` | aspiration の thread 別ずらしを強める | 採らず |
| `SPSA_SHARED_CONT_HIST` | continuation history を thread 間で共有 | 採らず |
| `SPSA_SMP_DEFER_MIN_DEPTH` | ABDADA 風に他 thread が読んでいる子を後回し | 閉 (発火率 0.025%) |
| `SPSA_ROOT_YBWC` | 根の 2 手目以降を helper に配る | 閉 |
| `SPSA_YBWC_MIN_DEPTH` (M4) | 内部節で 2 手目以降を helper に委ね、 結果は置換表だけで受け取る | 採らず (既定 off。 時間切れ停止時の不具合を直した版でも対局で改善なし) |
| `SPSA_NMP_THREAD_PARITY` | null move の R を thread の偶奇でずらす | 閉 |

## 13. 高速化 (一覧)

中身は `SPEEDUPS_STORY.md`、 実装仕様は `speedups/<id>.md`、 機械可読の一覧は `speedups/index.json`。

| id | 名前 | 結果を変えるか | 実測 (判定行) | 状態 |
|---|---|---|---|---|
| S01 | 長い利きライブラリを外す (LE-off) | 不変 | 非 PGO 108.88% [106.70, 111.18] / PGO で pf4 込み +3.91% | 採用 |
| S02 | 1 手詰めを利きを使わない版に | 不変 (LE-off と組) | — | 採用 |
| S03 | threat の出力幅を 64 に | net が変わる | 税 −37.5% (K=1024) → −17% (K=64) | 採用 |
| S04 | clang instr PGO の手順 | 不変 | 採取 net の誤りで −7.0% → 直して +7.0% / 採り直しで +3.08% | 採用 |
| S05 | `-march=native -ffp-contract=off` | 不変 | +約 10% | 採用 |
| S06 | clang 23 + 静的リンク | 不変 | (単独の測定なし) | 採用 |
| S07 | fc0 / fc1 の重みを 64 B 整列 | 不変 | +1.93% [+0.32, +3.76] | 採用 |
| S08 | 固定費の束 (P2 / A345 / A1〜A3) | 不変 | A1〜A3 で +4.81% (壁時計) | 採用 |
| S11 | LZ (threat の多段 lazy 差分) | 不変 | PGO 128 周 102.25% [101.69, 102.87] | 採用 |
| S12 | LZ IDXV2 | 不変 | 非 PGO 102.27% [101.63, 102.93] | 採用 (段 1) |
| S13 | LZ ROWPF | 不変 | 非 PGO 101.84% [101.09, 102.57] | 採用 (段 1) |
| S14 | SRCH_BUNDLE_A (探索側の束) | 不変 | 101.53% [100.86, 102.18] | 採用 (段 1) |
| S15 | 段 1 = S12+S13+S14 の PGO 束 | 不変 | **104.19% [103.43, 105.01]** (PGO 同士 64 周) | 採用 |
| S16 | LZ D1FAST | 不変 | NPS は測らず採用 (関門・検算は合格) | 採用 |
| S17 | LZ STEPCAP=96 | 不変 | 3 周で約 99.6% | 不採用 |
| S18 | KH の推論核 | (KH 自体は値を変える) | 税 −0.34% [−0.93, +0.26] | 採用 |
| S19 | OPN F2 | 不変 | 置き換えで +17% / +9% | 採用 |
| S20 | OPN R2 | 不変 | +2.0% [0.2, 3.9] (PGO 同士) | 採用 |
| S21 | OPN の継ぎ目の幅 0 | 値が変わる | 到達ノード 対 v4 110.0% [108.0, 112.0] | 採用 (v9 の継ぎ目は -1946) |
| S22 | L1Q128 の核 | net が変わる | 99.75% [99.20, 100.32] = 非劣化 | 採用 |
| S23 / S24 | Finny と多段の差分 / lazy evaluate | 不変 | 我々の on/off の測定なし | 採用 (土台から) |
| F01〜F03 | final2: TR_STORE32 / ACC_PIECEWISE / FC0_PF | 不変 | 3 本合わせて 101.74% (28 周で打ち切り、 区間なし)。 検算・関門は合格 | 採用 |
| S25 | LZ BACKMAT | 不変 | 14 周で 100.19% (打ち切り、 線 +1.0% に届かず) | 不採用 (公開版のソースに無い) |

## 14. ビルドと検査の器

| 器 | 中身 | 場所 |
|---|---|---|
| bit 一致の関門 | 同じ局面で `go nodes` を回し、 手・nodes・depth・score・pv の全一致を見る。 Threads=1、 局面ごとに `isready`、 先に base 対 base。 G-a 49 局面 x 20 万 / G-b 150 x 20 万 / G-c 24 x 200 万 | `<tools>/tools/python/` の `gate.py` |
| 二重経路の検算 | 新旧の経路を同じ exe で両方計算して不一致を数える (`*_VERIFY` define。 診断ビルド専用) | ソースの `*_VERIFY` の箇所 |
| NPS の A/B | CPU 時間・ABBA・64 周 x 4 局面 x 100 万ノード | `nps_ab.py` |
| PGO の採取 | 配布の net 一式と本番 option で採取 | `pgo_collect_*.py` |
| net の照合 | エンジンの `eval` と python の整数推論の全局面一致 | `engcheck.py` (L1Q128)、 `fti32_verify.py` / `fc1p_verify.py` / `fc1p_verify_opn.py` (v9 の形式) |

## 15. 11-17 以降の変更 (v9)

公開版 p2shogi v9 で足したものです。 強さの数字と試して採らなかったものは `OVERVIEW.md` 8 章、 仕組みの解説は `TECHNICAL_NOTES.md`。

### 15.1 評価関数の量子化の精度 (FTI32 系・FC1PREC)

| 項目 | 内容 |
|---|---|
| 目的 | 整数推論と f32 の差 (同じ親の子どうしの評価の順位の入れ替わり) を減らす。 python の模擬では、 差 10〜30cp の帯で最善手が入れ替わる率が 0.205 (従来) → 0.057 (FT 側) → 0.019 (後段も) |
| FT 側 | `-DP2_FT_KS=3`: FT の重み・bias を 254 x 8 刻みの i16 で持つ。 `-DP2_FT_OUT16=5`: transform の出力を i16 (0..126x32) にし、 fc0 を i16 入力の積和に。 `-DP2_FT_SPARSE16`: その fc0 を非零の 2 列単位で回す疎な版 (密な版と bit 一致)。 累積器は i16 のまま (探索中の和は i16 に 2.7 倍の余裕。 `-DP2_FT_I32` で i32 にもできるが公開版では使わない)。 threat 表と KH は従来の目盛りの別の累積器で持ち、 transform で 8 倍して足す |
| 後段 (FC1PREC) | `-DP2_FC1_PREC`: fc0 の活性 (SqrClippedReLU / ClippedReLU) を 0..508 の i16 に、 fc1 を i16 入力の madd (重みは 1/128 刻み)、 fc1 の活性も 0..508、 fc2 は i16 の内積 (重みは 1/64 刻みのまま)。 本体と OPN の両方 |
| net の形式 | arch の FT 幅の前に `I32K8:` (版 0x7AF32F21)、 fc1 の token を `AffineTransformF1P128[` (版 0x7AF32F22)。 古い exe は読み込みで止まる。 新しい exe に古い net を渡すと読み込み時に変換して動く (値は古い exe と同じにはならない) |
| net の作り方 | 学習し直さず、 学習器が書いた f32 の重み (`weights.bin`) から python で書き直す (`fti32_write.py` → `fc1prec_write.py`。 `TRAINING_FROM_SCRATCH.md` 6 章) |
| 照合 | python の整数模擬とエンジンの静的評価が、 本体 1,000 局面・OPN 1,000 局面で完全一致。 探索中 780 万回の評価で i16 の範囲を超えた値 0 件 |
| NPS | FT 側 (`KS=3 OUT16=5 SPARSE16`) は −6.53% [−7.21, −5.90] (32 周)。 FC1PREC は 8 周で打ち切り、 幾何平均 100.40% (影響なしと判断) |
| 実装場所 | `eval/p2nnue/p2nnue.cpp` の `P2_FT_KS` / `P2_FT_OUT16` / `P2_FT_SPARSE16` / `P2_FC1_PREC` の箇所 (`affine16_t`、 `affine16_sparse_t` ほか)、 `eval/p2nnue/p2nnue_opn.h` |

### 15.2 bucket 補正と先後別の表

| 項目 | 内容 |
|---|---|
| 目的 | 精度を上げた net は、 古い量子化にあった「手番側を悲観的に見る片寄り」が消えて対局で弱くなった。 その片寄りを bucket ごとの定数で足し戻す |
| 中身 | 本体の評価 (OPN と混ぜる前) に、 LayerStack の bucket ごとに定数 [cp、 手番側の視点] を足す。 USI option `P2_BUCKET_OFFSET_B0..B7` (後手番の探索で使う表) と `P2_BUCKET_OFFSET_SENTE_B0..B7` (先手番の探索で使う表)。 表は探索の開始時に root の手番で 1 回だけ選び、 評価の熱い経路では分岐しない |
| 切り替え | `P2_SENTE_TABLE_SWITCH_CP` (v9 の既定 500): 同じ対局の直前の手の root の評価値が表示 cp で +500 以上なら、 後手番でも先手の表を使う。 読みの途中では切り替えない |
| v9 の値 | 後手の表 = SPSA24 の 29 周目の値 (−39 / −14 / −27 / +19 / +4 / −12 / +10 / +29)、 先手の表 = 古い量子化の片寄りを写した値 (−18 / −3 / −32 / −5 / +16 / −15 / −4 / +15)。 bucket 0 は OPN が先に値を返すので、 B0 の値は実際には効かない |
| 注意 | 評価が「局面と root の手番」の関数になる。 局の境目で `isready` を送らない器では、 別の手番で作った評価を置換表から再利用しうる (`usinewgame` はこのエンジンでは何もしない) |
| 元の define | `P2_BUCKET_OFFSET` |
| 実装場所 | `eval/p2nnue/p2nnue.cpp` の `P2_BUCKET_OFFSET` の箇所 (`select_bucket_off`、 `p2_search_begin`) |

### 15.3 既定値の焼き込みの仕組み

| 項目 | 内容 |
|---|---|
| 仕組み | `OptionsMap::set_release_default(name, value)` (`usioption.cpp` / `usioption.h`) を足した。 USI の既定値 (`option ... default`) と現在値の両方をその値にする。 名前が無ければ `info string Error!` を出す |
| 呼ぶ場所 | `eval/p2nnue/p2nnue.cpp` の P2NNUE の `add_options` の最後。 公開版の値 48 個 (調整した 47 個の option と `SPSA_LMR_TABLE_COEFF_OPENING=2763`) を表にして順に呼ぶ |
| 確認 | ファイルのパスの option だけを渡して起動した探索が、 以前の 206 個の option を渡した設定と 40 局面 x 100 万ノードで同一 |

### 15.4 その他

| 項目 | 内容 |
|---|---|
| 千日手の評価値 | `engine/yaneuraou-engine/yaneuraou-search.cpp` の `DrawValueBlack` / `DrawValueWhite` の既定を -2 / -2 に戻した。 以前の -53 / +41 は電竜戦の規定 (引き分けを先手 0.4 / 後手 0.6 と数える) に合わせた値で、 floodgate では後手が千日手を受け入れやすすぎた |
| エンジン名 | `engine.cpp` で、 `EVAL_P2NNUE` のとき `id name p2shogi v9` を出す (評価関数名や CPU の接尾辞は付けない)。 作者名は `engine.h` の `get_engine_author()` で `pekoshi` |
| Threads | 既定 4 (`engine.cpp`) |
| FV_SCALE | `FV_SCALE_PER_NODE=true` で bucket ごと (`FV_SCALE_B0..B7` = 26 / 26 / 27 / 28 / 28 / 28 / 28 / 28) |
| KH | 配布の `kh.bin` は 1.25 倍の重み (5 章) |
| PGO | 公開版の option で、 実戦の 60 局面 x 100 万ノードで採取 (手数の帯 24-59: 15 局面 / 60-99: 20 / 100-139: 15 / 140 以上: 10)。 CSPGO (IR の文脈つき PGO) はリンクで `__llvm_profile_filename` が未定義になり使わなかった |
| 試して採らなかったもの | LMR の係数 2490 (浅く広く読む。 56 局で 38.4%、 悪手の試験でも変化なし)、 threat 入力の倍率 k = 0.90 / 1.05 / 1.10 (対局で負け)、 手の並べ順の policy head (入れなかった)、 OPN を bucket 0〜1 に (対局で負け) |
