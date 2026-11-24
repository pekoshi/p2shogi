# 評価関数を一から作る手順 (p2shogi v9 の eval/ 一式)

何も無いところから、 p2shogi で動く `eval/` フォルダ (`nn.bin` / `opn.bin` / `kh.bin` + `kh.bin.json` / `progress.bin`) を作るまでの手順です。

## 0. 先に断っておくこと

- **我々の net は一から一気に作ったものではありません**。 本体の系譜は 7 月から約 100 SB を継ぎ足したもので、 その途中の checkpoint (`_base_ctl90` → `newd-98` → `top3q-102`) から先だけが記録に残っています。 最初の数十 SB の引数とデータは公開できる形で残っていません。
- そこでこの文書は 2 本立てにしました。
  - **A. 一から作る最短の道** (1〜7 章): 公開データだけで、 4 本の部品を全部作る。 本体を 0 から学習する段は、 我々が記録に残した引数から `--resume` を外したもので、 **この形で我々が通しで回した記録はありません (記録なし)**。 期待値も出せません。
  - **B. 我々が実際に使った手順** (8 章): `top3q-102` から先。 コマンドは実際に走らせたものの写しで、 詳細は `REPRODUCE.md`。
- コマンドの引数は、 我々の記録 (各 run の `experiment.json` と起動スクリプト) に残っているものだけを書きました。 記録に無いところは「記録なし」と書いています。 推測で補った引数はありません。
- 学習の段は bit 単位では再現しません (seed と GPU の演算順)。 書き直しの段 (6 章) は入力が同じなら bit 単位で同じものが出ます。

## 1. 必要なもの

| 項目 | 我々の環境 | 備考 |
|---|---|---|
| GPU | NVIDIA GeForce GTX 1050 Ti (VRAM 4 GB) | 4 GB で全段が回った。 本体は threat の群化で GPU メモリを `--threat-group-mem-budget-mib 1200` に抑えた |
| CPU | Intel Core i7-7700 (AVX2) | エンジンの実行には AVX2 が要る。 学習と CPU の重い作業 (対局・篩) を並行させると GPU が 0% になったので、 学習器は低い優先度、 データのスレッド 3 本にした |
| OS | Windows 10 + Git Bash | Linux でのビルドは確かめていない |
| Rust / CUDA | 版は記録なし | 学習器は既定 feature `cuda` でビルドする |
| C++ | llvm-mingw 20260908 (clang 23、 ucrt、 x86_64) | エンジンのビルド |
| python | python 3 + numpy (照合には cshogi も) | 書き直し・篩・照合の器 |
| ディスク | 教師 数十〜百 GB 超、 checkpoint 1 個 約 1.7 GB | OPN の教師だけで 67.9 GB |
| 学習の速さ | 1 SB (= 4096 x 24415 = 100,003,840 局面) が 本体 約 52 分、 OPN 約 55 分 (約 30,000 局面 / 秒) | |

## 2. リポジトリとビルド

### 2.1 置き方

```
<作業フォルダ>/
  bullet-shogi/   https://github.com/pekoshi/bullet-shogi の p2shogi ブランチ (学習器、 nodchip/bullet-shogi の fork)
  rshogi/         https://github.com/pekoshi/rshogi の p2shogi ブランチ (道具、 ラム将棋 SH11235/rshogi の fork)
  p2shogi/        https://github.com/pekoshi/p2shogi の master (エンジン、 やねうら王の fork)
```

学習器の `crates/bullet_lib/Cargo.toml` は `rshogi-core` を相対パス `../../../rshogi/crates/rshogi-core` で参照します。 **道具のリポジトリは学習器と同じ親フォルダに `rshogi` という名前で置いてください**。

### 2.2 学習器

```bash
cd bullet-shogi
cargo build --release -p bullet_lib --example shogi_layerstack      # 学習器本体
cargo build --release -p bullet_lib --example psv_bucket_filter     # 進行度 bucket で PSV を抜き出す
```

- 本体・KH・OPN・QAT は全部この 1 本の `shogi_layerstack` の引数で切り替えます。 我々は時点の違う 3 つのビルドを使い分けていたので、 **公開版の 1 本で全段を回した記録はまだありません**。 回し始めたら、 起動直後のログ (4〜5 章の「確かめる行」) が期待と合うかを最初に見てください。
- 存在しない引数を渡すと即落ちます (例: `--end-superbatch` は無い。 終わりは `--superbatches`)。

### 2.3 エンジン

公開版の exe と同じ define でビルドします (`<engine>/README.md`)。

```bash
cd p2shogi/source
make -j4 normal COMPILER="clang++ -static" YANEURAOU_EDITION=YANEURAOU_ENGINE_P2NNUE \
  EXTRA_CPPFLAGS="-DP2_ARCH_1024 -DP2_L1_MAIN=15 -DP2_THREAT_PROFILE=0 -DP2_MATE1PLY_NOEFFECT -DP2_THREAT_OUT=64 \
  -DUSE_LAZY_EVALUATE -DP2_THREAT_LZ -DP2_OPN -DP2_OPN_F2 -DP2_KH -DP2_OPN_R2 -DP2_LZ_IDXV2=1 \
  -DP2_LZ_ROWPF=1 -DP2_SRCH_BUNDLE_A -DP2_LZ_D1FAST -DP2_TR_STORE32 -DP2_ACC_PIECEWISE -DP2_FC0_PF \
  -DP2_FT_KS=3 -DP2_FT_OUT16=5 -DP2_FT_SPARSE16 -DP2_FC1_PREC -DP2_BUCKET_OFFSET -DP2_OPTIMISM"
```

- `-DP2_FT_KS=3 -DP2_FT_OUT16=5 -DP2_FT_SPARSE16 -DP2_FC1_PREC` の exe は、 6 章の書き直しをした net を読みます。 学習器が書いたままの net (刻み 254) を渡しても読み込み時に変換して動きますが、 値は書き直した net と同じにはなりません。 途中の確認には、 この 4 つを外した exe (= 11-16 の形式) も 1 本作っておくと便利です。
- PGO (任意、 NPS が数 % 上がる): `-fprofile-instr-generate` を付けた exe で採取 → `llvm-profdata merge` → `-fprofile-instr-use=<profdata>` で再ビルド。 我々は公開版の option で、 実戦の 60 局面 (手数の帯 24-59: 15 / 60-99: 20 / 100-139: 15 / 140 以上: 10) を 100 万ノードずつ読ませて採取しました。 🚨採取のときに net が実際に読めているかをログで確かめてください (存在しない net で採取して NPS が −7.0% になったことがあります)。 採取の局面集は記録なし (自分の対局から同じ帯で選んでください)。

## 3. データ

| 用途 | データ | 入手元 | 備考 |
|---|---|---|---|
| 進行度の係数 | `progress.bin` | NAGISA v3.1 の配布物 (我々の Releases の `eval/` にも同梱) | GPLv3。 sha256 `e7ed0eef88868335f9a46c58a121dccb5ad82a5eb1c8ee12de90365ab351e37d`。 LayerStack の bucket と OPN の切り替えに使うので、 **学習の前に要る** |
| 教師 (公開) | DL 水匠のラベルを付けた蒸留データ 001〜018 (ponkotsu 系) | Hugging Face `ngs436/dlsuisho-ponkotsu-distilled` (`penguinkumimanu/ponkotsuWCSC36` の 001-018 と sha256 一致、 と記録) | 40 B / 局面の PSV。 利用条件は配布元に従う。 我々は OPN の教師に使った |
| 本体の教師 (我々の系譜) | `novel_ext3` / `dlsuisho_preext3_residual_v1` / `fgblend_top3q` / `fgb_tr` | 自作 (DL 系ラベルのデータの加工と、 floodgate 上位陣の棋譜からの局面の注入) | **配布しない**。 作り方の一部は記録なし |

一から作るなら、 本体の教師も公開の ponkotsu 系データを使うことになります。 その場合の結果 (篩・強さ) は記録なしです。 我々の経験では、 v4 と同じ出所のデータでも FT の質に差が出ており (`FUTURE_WORK.md` 1 章)、 データより学習器の設定・構造の差を疑っています。

### 3.1 score の換算

我々は ponko の値を、 我々の目盛りに合わせる換算表に通してから使いました (`calib_lut.py`: ponko の値 p ごとに E[σ(d/600) | p] を求め、 600·logit で score に戻す)。 表の作り方 (何と対応づけたか) の詳細は記録なしです。 一から作るなら、 まず換算なしで始め、 必要なら `--loss-scale` で教師の勝率変換の目盛りを変えるのが簡単です。

### 3.2 序盤帯の抜き出し (OPN 用)

```bash
psv_bucket_filter.exe --in "<ponko_002.psv>,…,<ponko_018.psv>" \
  --out <ponko_early>/ponko_b02_all.psv --bucket-min 0 --bucket-max 2 \
  --bucket-mode progress8kpabs --progress-coeff <progress.bin> --threads 8
```

我々の結果: 読んだ 4,974,820,631 / 通した 1,697,493,603 (34.1%)、 2,375 秒。 学習器の bucket の計算との照合 77,731,582 件で不一致 0。

## 4. 本体 (LayerStack-1024-16-64 + threat 64)

### 4.1 共通の引数 (記録にあるもの)

我々の本体の全段に共通する引数です。

```
--batch-size 4096 --batches-per-superbatch 24415 --save-rate 1
--lr 0.000006 --warmup-batches 500 --lr-gamma 1.0 --lr-step 1
--wdl 0.15 --scale 600 --loss-power 2.6
--l0 1024 --l1 16 --l2 64 --psqt
--threat --threat-out 64 --threat-bucket-groups 0,1,1,2,2,3,3,3 --threat-group-mem-budget-mib 1200
--bucket-mode progress8kpabs --progress-coeff <progress.bin>
--output-format rshogi --optimizer ranger21 --weight-decay 0.005 --ranger21-lookahead-mergetime 20 --ranger21-total-steps 0
--data-format psv --interleave-file-batches 1
```

- 構造: HalfKA_hm (73,305) → FT 1024 x 2 → fc0 16 (15 + skip) → sqr / lin 30 → fc1 64 → fc2 1、 PSQT あり、 LayerStack 8 bucket (進行度)、 QA 127 / QB 64、 fv_scale 14。
- threat: 216,720 次元 → 64 列、 bucket を 4 群 (0 / 1-2 / 3-4 / 5-7) に分けて群ごとに別の表。
- `--ranger21-total-steps 0` は段をまたいで `--resume` するときに必須です (付けないと内部の予定が尽きて lr が 0 になり、 学習が静かに何もしない)。
- lr の減衰は SB の絶対番号で決まるので、 我々は `--lr-gamma 1.0` のまま、 段ごとに起動し直して lr を引数で変えました。

### 4.2 A: 0 から学習する (記録なし)

我々は本体をこの形で 0 から回していません。 4.1 の引数から `--resume` を付けずに起動すれば、 学習器は重みを初期化して始めます。 次の点は記録が無く、 決めるのは追試者です。

| 決めること | 我々の記録 |
|---|---|
| lr | 系譜の後半 (SB91 以降) は 6e-6。 初期の lr は記録なし。 6e-6 は継ぎ足し用の小さな値で、 0 からの学習には小さすぎるはず (推測) |
| SB 数 | 我々の系譜は約 100 SB (7〜10 月)。 0 からの必要数は記録なし |
| threat の扱い | 系譜の途中 (SB91〜102) は `--threat-freeze` に `--threat-ft-lr-mult 72 --threat-grad-mask 600:1.0,1000:0.5,inf:0.25 --threat-profile full` を足していた。 KH 段は `--threat-freeze-hard`。 0 から threat を学習する段の引数は記録なし |
| データ | 3 章。 本体の我々の教師は配布しない |

🚨 我々の経験 (実験済み): FT を大きく動かすと弱くなり、 一から学習した net は手持ちの資源では継ぎ足した net に追いつけませんでした。 4 GB の GPU で 0 から作るなら、 強さは我々の配布 net より下がる前提で考えてください。

起動の型 (引数は 4.1 を全部付ける。 `<LR>` と `<N>` は追試者が決める):

```bash
shogi_layerstack.exe --data <教師.psv> <4.1 の共通の引数> --lr <LR> \
  --start-superbatch 0 --superbatches <N> --output <out> --net-id main
```

起動直後に確かめる行: `Features: HalfKA_hm (73305 dimensions)`、 `Architecture: LayerStack L0=1024, L1=16 (15 effective + 1 skip), L2=64`、 `Threat: enabled (profile=full, 216720 dimensions, out_width=64)`、 threat 群化 (群数 4、 bucket→群 [0,1,1,2,2,3,3,3])、 `WDL lambda: Constant (0.15)`。

出力の checkpoint `<out>/main-<N>/` には `quantised.bin` (エンジンが読む形式)、 `raw.bin`、 `optimiser_state/weights.bin` (f32) が入ります。 6 章の書き直しには `raw.bin` と `weights.bin` が要るので消さないでください。

### 4.3 KH の学習

本体の FT・threat・fc1・fc2・PSQT を凍結し、 KH (浮き駒 78 次元 → 64 列) と fc0 だけを学習します。 我々の M2 段の写しです (所要 12,552 秒 = 3.5 時間、 4 SB)。

```bash
shogi_layerstack.exe --data <教師.psv> --data-format psv --interleave-file-batches 1 \
  --resume <4.2 の checkpoint> --start-superbatch 0 --superbatches 4 --batches-per-superbatch 24415 --save-rate 1 \
  --batch-size 4096 --lr 0.000006 --warmup-batches 500 --lr-gamma 1.0 --lr-step 1 \
  --wdl 0.15 --scale 600 --loss-power 2.6 --l0 1024 --l1 16 --l2 64 --psqt \
  --threat --threat-out 64 --threat-freeze-hard --bucket-mode progress8kpabs --progress-coeff <progress.bin> \
  --output-format rshogi --optimizer ranger21 --weight-decay 0.005 --ranger21-lookahead-mergetime 20 --ranger21-total-steps 0 \
  --threat-bucket-groups 0,1,1,2,2,3,3,3 --threat-group-mem-budget-mib 1200 \
  --kh --kh-feats h --kh-freeze-rest --kh-unfreeze-fc0 --kh-lr-mult 64 --kh-dump <out>/kh_final.bin \
  --output <out> --net-id a
```

- 起動直後に確かめる行: `KH ブロック: 部分 h / 出力幅 64 …`、 `threat_ft HARD FREEZE`、 `KH FREEZE-REST: […]`、 `Resuming from checkpoint: …`。
- 出力: `<out>/a-4/` (本体の net。 fc0 が KH 込みで学習し直されている) と `kh_final.bin` + `kh_final.bin.json` (dims 142 = k 64 + h 78。 KH の f32 重み)。
- loss だけでは KH の効きは読めません。 篩の KH on − off で見ます (我々: bucket 0〜2 +0.0055 ±0.0022 / 3〜7 +0.0048 ±0.0009)。
- 配布の `kh.bin` は `kh_final.bin` の重みを 1.25 倍したものです (`kh_scale.py 1.25`)。 1.25 倍は 11-17 の決定で、 倍率の選び方の詳しい記録は公開していません。 一から作る場合は 1.0 (そのまま) でも動きます。

## 5. OPN (序盤専用 net、 幅 2048)

OPN は本体の FT を写して出発します。 **一から学習した幅 2048 は同じ 2 SB で篩 0.2050、 写しから始めると 0.2756** でした。 本体 (4 章) を先に作ってください。

### 5.1 初期重み

```bash
py -3 f1_transplant2048.py build      # 入出力のパスはスクリプトの中 (公開版で引数化が要る)
py -3 f1_transplant2048.py verify
```

本体の f32 の FT (l0w [73,305 x 1024] / l0b) を、 幅 2048・K16 入力 (27,693 行 = 26,064 + factoriser 1,629) の半分の列に写します: 列 j < 512 → j、 j + 512 → j + 1024。 行は K16 の 7〜9 段の 15 区分をそのまま、 他の 30 区分は f32 の平均。 写した列の factoriser 行は 0。 残り 1,024 列は N(0, 0.25)、 bias 0 (seed 20261115)。 後段は学習器の初期化に任せます。

### 5.2 学習 (16 SB、 我々の実績 16.4 時間)

共通の引数:

```bash
shogi_layerstack.exe --data <ponko_b02_all.psv> --data-format psv --bucket-filter 0-2 \
  --batch-size 4096 --batches-per-superbatch 24415 --save-rate 1 --log-rate 256 --threads 3 \
  --warmup-batches 500 --lr-gamma 1.0 --lr-step 1 --wdl 0 --scale 600 --loss-power 2.6 \
  --l0 2048 --l1 16 --l2 32 --factorise --bucket-mode progress8kpabs --progress-coeff <progress.bin> \
  --opn-king16 --loss-scale 300 \
  --output-format rshogi --optimizer ranger21 --weight-decay 0.005 --ranger21-lookahead-mergetime 20 --ranger21-total-steps 0 \
  --superbatches <END> --start-superbatch <START> --lr <LR> <入力> [--qat] --output <out>/<NID> --net-id <NID>
```

| 段 | START→END | LR | 入力 | --qat | 我々の所要 |
|---|---|---|---|---|---:|
| O1 | 1→2 | 0.001 | `--resume-weights-only <5.1 の出力>/optimiser_state` | | 9,888 s |
| O2 | 3→6 | 0.001 | `--resume <O1 の SB2>` | | 12,899 s |
| O3 | 7→10 | 0.001 | `--resume <O2 の SB6>` (篩の最高の SB を採る。 我々は SB9) | | 12,905 s |
| O4 | 10→10 | 0.0005 | `--resume <O3 の採った SB>` | | 3,330 s |
| O5 | 11→11 | 0.00025 | `--resume <O4>` | | 3,266 s |
| O6 | 12→12 | 0.000125 | `--resume <O5>` | | 3,329 s |
| O7 | 13→13 | 0.0000625 | `--resume <O6>` | | 3,259 s |
| O8 | 14→15 | 0.0001 | `--resume <O7>` | ○ | 6,662 s |
| O9 | 16→16 | 0.0001 | `--resume <O8 の SB15>` | ○ | 3,349 s |

- 起動直後に確かめる行 (O1): `Features: HalfKA_hmK16 (27693 dimensions)`、 `Architecture: LayerStack L0=2048, L1=16 (15 effective + 1 skip), L2=32`、 `PSQT shortcut: disabled`、 `Threat: disabled`、 `Loss scale: target = sigmoid(score / 300), pred = sigmoid(out * 2)`、 `Loading graph weights only; optimiser state is new: …`。
- `[warn] --bucket-filter を --hard-freeze-shared なしで使っている` は OPN では意図どおりです (OPN は序盤でしか使わない)。
- 我々の値 (篩 b01 = bucket 0-1): SB2 0.2758 / SB9 0.2993 / SB13 0.3010 / SB16 (QAT、 最終) 0.3019。 QAT の効果は +0.002 前後で 2SD を超えていません。
- 試して採らなかったもの: OPN を bucket 2 まで受け持たせる (b2 で頭打ち)、 bucket 0〜1 を受け持たせる (対局で負け)、 FT も動かして学習し直す (対局で勝てない)。 v9 の既定では OPN は bucket 0 だけを受け持ちます。

## 6. エンジンの形式への書き出し (学習なし)

学習器が書いた `quantised.bin` は、 そのままでも 11-16 の形式の exe なら読めます。 v9 の exe と同じ精度にするには、 f32 の重みから 3 段 (OPN は 2 段) の書き直しをします。 どれも CPU で数秒〜数分です。

### 6.1 本体 → nn.bin

```bash
# (1) L1Q128: fc0 の本体 15 行 x 8 bucket を 1/128 刻みに
py -3 mk_l1q128.py <a-4>/quantised.bin <a-4>/raw.bin -                          # 自己検算と表示だけ
py -3 mk_l1q128.py <a-4>/quantised.bin <a-4>/raw.bin <a-4>/quantised_l1q128.bin
# (2) FT を 8 倍細かい刻みに
py -3 fti32_write.py <a-4>/quantised_l1q128.bin <a-4>/optimiser_state/weights.bin 8 <a-4>/quantised_l1q128_fti32k8.bin
# (3) fc1 の刻み 1/128 と後段の bias の目盛り
py -3 fc1prec_write.py <a-4>/quantised_l1q128_fti32k8.bin <a-4>/optimiser_state/weights.bin <eval>/nn.bin
```

| 段 | 確かめること |
|---|---|
| (1) | 自己検算 OK (現行の式での作り直しが元と全バイト一致)。 飽和の本数 (我々 13 本 / 122,880 本)。 桁違いに増えたら (目安 100 本超) 止めて調べる |
| (2) | i16 に入らない値で止まらないこと (我々は FT の重みの最大 \|q\| 3,691)。 `fti32_write.py --check <元 quantised.bin> <weights.bin>` で weights.bin の取り違えを先に確かめられる |
| (3) | fc1 の重みで切られる数 0 (我々 max\|q\| 88)、 元の値との照合で不一致 0。 大きさが +6 byte |

### 6.2 OPN → opn.bin

```bash
py -3 fti32_write.py --opn <O9>/quantised.bin <O9>/optimiser_state/weights.bin 8 <O9>/quantised_fti32k8.bin
py -3 fc1prec_write.py --opn <O9>/quantised_fti32k8.bin <O9>/optimiser_state/weights.bin <eval>/opn.bin
```

OPN の fc0 は L1Q128 にしません (1/64 刻みのまま)。 fc1 で切られる数 0 (我々 max|q| 124) を確かめます。

### 6.3 KH → kh.bin

```bash
py -3 kh_scale.py 1.25     # kh_final.bin -> 1.25 倍 (入出力のパスはスクリプトの中)
```

出力の `.bin` と `.bin.json` を `<eval>/kh.bin` / `<eval>/kh.bin.json` にします。 倍率を掛けないなら `kh_final.bin` と `kh_final.bin.json` をそのまま改名します。

### 6.4 eval/ を組み立てる

```
eval/
  nn.bin          6.1 の出力
  opn.bin         6.2 の出力
  kh.bin          6.3 の出力
  kh.bin.json     6.3 の出力 (形の検査に使う)
  progress.bin    3 章 (NAGISA v3.1)
```

固定名なので、 exe と同じ場所に `eval/` を置けば option は要りません。 別の場所なら `EvalDir` で指定します。

## 7. 確かめる

| 順 | 何を | どうやって | 合格 |
|---|---|---|---|
| 1 | 読み込めるか | `usi` → `isready` | `readyok`。 形式の違う net や欠けたファイルは `Error` を出して止まる (黙って別の評価で走らない) |
| 2 | 整数推論が正しいか (bit 一致) | `fc1p_verify.py` (本体)、 `fc1p_verify_opn.py` (OPN)。 書き直しの前段の net なら `engcheck.py` / `fti32_verify.py` | python の整数模擬とエンジンの静的評価が全局面で一致 (我々: 本体・OPN とも 1,000 / 1,000 局面)。 1 件でも違えば先に進まない |
| 3 | 探索が壊れていないか | `gate.py` で同じ局面を `go nodes` で読ませ、 2 つの exe の手・nodes・depth・score・pv を比べる (局面ごとに `isready`) | 値を変えない変更なら全一致。 net を替えたなら「止まらない・詰みを正しく返す」の確認だけ |
| 4 | 静的評価の質 | 篩 (`screen_future.py`、 `b01_screen.py`、 `pair_screen.py`、 `screen128.py`) | 2 本の net の差 (A − B) で見る。 SB ごとの揺れは 2σ 0.011 と大きいので 1 SB の上下で判断しない |
| 5 | 強さ | 対局 (固定ノード、 Threads 1、 局面集を先頭から順に x 先後) | 数百局でも 2SE が ±5〜13pt。 局面集を 2 つ以上使う (`TECHNICAL_NOTES.md` 4 章) |

🚨 篩の局面集と正解値は、 相手エンジン (NAGISA v4) の探索値を含むので同梱しません。 自分の対局から局面を取り、 正解値を自分で作り直してください (`REPRODUCE.md` 7 章)。

🚨 `usinewgame` はこのエンジンでは何もしません。 局面集を 1 プロセスで続けて解く器では、 局面ごとに `isready` を送ってください (置換表・history が持ち越され、 bucket 補正の表の選択も root の手番で変わるため)。

## 8. B: 我々が実際に使った手順 (top3q-102 から)

我々の配布 net は次の道で作りました。 コマンドの全文と期待値は `REPRODUCE.md` にあります。

```
_base_ctl90 (SB90、 それ以前は非公開の記録のみ)
  └ SB91-98: novel_ext3 + dlsuisho_preext3_residual、 lr 6e-6 → newd-98 (09-27)
      └ SB99-102: novel_ext3 + fgblend_top3q、 lr 6e-6、 threat-freeze → top3q-102 (10-12、 14,080 秒)
          ├ M2: KH の学習 4 SB (fgb_tr) → khhr/a-4 + kh_final.bin (11-12、 12,552 秒)
          │   ├ mk_l1q128.py → fti32_write.py (k=8) → fc1prec_write.py → nn.bin
          │   └ kh_scale.py 1.25 → kh.bin
          └ O0: FT の写し → O1〜O9 (16 SB、 58,887 秒) → opnF2Qx-16
              └ fti32_write.py --opn → fc1prec_write.py --opn → opn.bin
```

- 4.2 の「0 から」の段の代わりに、 我々は `top3q-102` まで継ぎ足した checkpoint を使っています。 `top3q-102` の f32 重み (`raw.bin`、 `optimiser_state/weights.bin`、 `quantised.bin`) は Releases の [top3q-102](https://github.com/pekoshi/p2shogi/releases/tag/top3q-102) で配布しています。 optimizer の momentum などの状態は含まないので、 `--resume` ではなく `--resume-weights-only <展開先>/top3q-102/optimiser_state` で読み込んでください (我々の実行は状態つきの `--resume` だったので、 学習の軌跡は完全には一致しません)。
- 本体の教師 (`novel_ext3`、 `fgblend_top3q`、 `fgb_tr`) は配布しません。 公開データで追うと値は変わります。
- 我々の出力の sha256 (入力が我々のものなら書き直しの段は bit 単位で一致するはず): `nn.bin` = `a83bf75edf0db2bea08ede4b92ac721b383e589caeffdfcf49652a43f067af79`、 `opn.bin` = `e8ecccd6aa2d64f4347a1222fcc265ceac814565a3c1fcad9f2a028a627db732`、 `kh.bin` = `4186aba33a1fe47849761fca5c369aa6bd63a4511e3fb7a6bee125e59a2c803f`。

## 9. 所要時間のまとめ (我々の機械)

| 段 | 時間 |
|---|---:|
| 3.2 序盤帯の抜き出し | 2,375 秒 (CPU) |
| 4.2 本体を 0 から | 記録なし (1 SB 約 52 分。 我々の系譜は約 100 SB) |
| 4.3 KH 4 SB | 3.5 時間 |
| 5.2 OPN 16 SB | 16.4 時間 |
| 6 書き直し | 数秒〜数分 (CPU) |
| 7 篩 (1 net) | 約 6 分 (CPU) |
| 7 照合 | 14 秒〜数分 (CPU) |

## 10. 必要な python スクリプト

この手順で使う python の器は、 https://github.com/pekoshi/rshogi の `p2shogi` ブランチの `tools/python/` に置きます。 器どうしが import し合うので、 同じフォルダに置いて使ってください。 一部の器は入出力のパスがスクリプトの中に書いてあり、 使う前に書き換えが要ります。

| 段 | スクリプト | 部品 (import されるもの) |
|---|---|---|
| 3.1 score の換算 | `calib_lut.py` | |
| (任意) 既見局面の除去 | `ponko_novel.py` | |
| 4.3 / 6.3 KH の倍率 | `kh_scale.py` | |
| 5.1 OPN の初期重み | `f1_transplant2048.py` | `f1_transplant.py`、 `opn_static2.py`、 `opn_static.py`、 `bullet_inference_v19.py` |
| 6.1 L1Q128 | `mk_l1q128.py` | |
| 6.1 / 6.2 FT の刻み | `fti32_write.py` | `k256_merge.py` |
| 6.1 / 6.2 fc1 以降の丸め | `fc1prec_write.py` | `k256_merge.py` |
| OPN の目盛り | `calib_l1q128.py` | |
| 7-2 照合 | `engcheck.py`、 `fti32_verify.py`、 `fc1p_verify.py`、 `fc1p_verify_opn.py` | `k256_merge.py`、 `bullet_inference_v19.py`、 `opn_static2.py`、 `eng.py`、 `v4c.py`、 `p0.py` (+ `make_sets.py`)、 `m2_games_static.py` (cshogi も要る) |
| 7-3 探索の関門 | `gate.py` | `eng.py`、 `v4c.py` |
| 7-4 篩 | `screen_future.py`、 `b01_screen.py`、 `pair_screen.py`、 `screen128.py`、 `screen_kh_onoff.py` | `p0.py` (+ `make_sets.py`)、 `m2_games_static.py`、 `opn_static2.py` |
| (エンジン) NPS の A/B | `nps_ab.py`、 `nps_cputime_ab.py`、 `nps_ab_prod.py` | `eng.py`、 `v4c.py` |
| (エンジン) PGO の採取 | `pgo_collect_l1q128.py` (11-16 の形)、 `collect.py` (v9 の 60 局面 x 100 万ノード) | `v4c.py` |
