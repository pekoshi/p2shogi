# p2shogi の全体像

p2shogi は、 やねうら王 (YaneuraOu) を土台にした将棋エンジンと、 その評価関数を作るための学習器・道具の一式です。 2026-11 に p2shogi v9 として公開し、 これをもって開発はいったん終了します (理由は 6 章)。

この文書は入口です。 細部は次の文書にあります。

| 文書 | 中身 | 読み手 |
|---|---|---|
| `OVERVIEW.md` (この文書) | 構成・経緯・到達点・制約・使い方の注意 | 全員 |
| `TECHNICAL_NOTES.md` | p2shogi がどう動くか・なぜそうしたかの技術解説 (評価関数・探索・測り方・教訓) | 全員 |
| `TRAINING_FROM_SCRATCH.md` | 何も無いところから `eval/` 一式 (nn.bin / opn.bin / kh.bin / progress.bin) を作る手順 | 評価関数を自分で学習したい人 |
| `MECHANISMS_GUIDE.md` | 独自の機構ごとの、 中身・ソースの場所・切り替え方・測り方・教訓 | 我々の機構を土台に新しいものを作る人 |
| `CHANGES_FROM_UPSTREAM_engine.md` | エンジン (やねうら王 `717da87` からの改造点) | エンジンを読む人 |
| `CHANGES_FROM_UPSTREAM_trainer.md` | 学習器 (nodchip/bullet-shogi `df431b3` からの改造点) | 学習を追試する人 |
| `CHANGES_FROM_UPSTREAM_tools.md` | 道具 (SH11235/rshogi `afd941a` からの改造点と python の器) | 計測・対局を再現する人 |
| `SPEEDUPS_STORY.md` | 高速化の読み物 (人間向け) | 全員 |
| `speedups/*.md`、 `speedups/index.json` | 高速化の実装仕様 (AI 向け。 1 件 1 ファイルで自己完結) | 他のエンジンに移植する人・コーディングエージェント |
| `REPRODUCE.md` | 我々の学習の追試の手順書 (配布した net の出発点から) | GPU を持つ追試者 |
| `FUTURE_WORK.md` | 今後の課題 (引き継ぎ) | 続きをやる人 |
| `UPSTREAM_PROPOSALS.md` | やねうら王への高速化の提案候補 | やねうら王の開発者 |

## 0. 文書の書き方の約束

- パスは公開リポジトリの中の相対表記で書く。
  - `<engine>/` = エンジンのリポジトリ (https://github.com/pekoshi/p2shogi 、 master)。 ソースは `<engine>/source/`。
  - `<trainer>/` = 学習器のリポジトリ (https://github.com/pekoshi/bullet-shogi の `p2shogi` ブランチ)。
  - `<tools>/` = 道具のリポジトリ (https://github.com/pekoshi/rshogi の `p2shogi` ブランチ)。 python の器は `<tools>/tools/python/`。
- 実験記録 (設計書・事前登録・判定ログ・作業メモ) は非公開です。 数字はその記録の**判定行**から写しています。 途中経過の数字は載せません。 後で訂正された数字は訂正後を載せます。
- 区間は特記がなければ 95% 区間です。 NPS の比は「CPU 時間で測った ABBA の交互実行、 N 周」の値です (測り方は `SPEEDUPS_STORY.md` 2 章)。

## 1. 構成

### 1.1 3 つのリポジトリ

```
yaneurao/YaneuraOu 717da87 (V9.40) ──fork──> pekoshi/p2shogi (master)              C++ / GPLv3  探索 + 評価関数 P2NNUE
nodchip/bullet-shogi df431b3 ──────fork──> pekoshi/bullet-shogi (branch p2shogi)  Rust / MIT    評価関数の学習器 (CUDA)
SH11235/rshogi (ラム将棋) afd941a ──fork──> pekoshi/rshogi (branch p2shogi)        Rust / GPLv3  rshogi-core・対局器・SPSA・PSV 変換 + python の器
```

- エンジンは本家の探索をほぼそのまま使い、 評価関数 (P2NNUE) を独自に足しています。 P2NNUE は本家の `eval/nnue/` を使わない独立した実装で、 当初ラム将棋の fork (Rust) で書いたものを C++ に移しました。
- 学習器は道具のリポジトリの `rshogi-core` を参照してビルドします (2 つを同じ親フォルダに置く。 `TRAINING_FROM_SCRATCH.md` 2 章)。
- 配布物 (GitHub Releases): exe と評価関数一式 (`eval/` の `nn.bin` / `kh.bin` + `kh.bin.json` / `opn.bin` / `progress.bin`)。

### 1.2 評価関数

本体 (中盤・終盤) と OPN (序盤専用) の 2 本の net を、 進行度で切り替えます。

```
                ┌──────────── 本体 nn.bin ──────────────────────────────────────┐
HalfKA_hm 73,305 ─> FT 1024 x 2 視点 (i16、 刻み 254x8) ─┐
threat 216,720  ─> threat acc 64 列 ────────────────────┼─> 足す (64 列を FT の [0,32) と [512,544) に)
KH (浮き駒 78 次元, kh.bin) ─> 64 列 ───────────────────┘   ※ KH は評価のときに一時 buffer で足す
                   PSQT ───────────────────────────────────────────────┐
  transform (clamp・対の積、 i16 出力) ─> fc0 1024 -> 16 (15 + skip 1、 重みは 1/128 刻み)
  ─> sqr / lin 30 (0..508) ─> fc1 64 (重み 1/128 刻み) ─> fc2 1 ─(+ PSQT)─> 評価値
  fc0 以降は LayerStack: 進行度 8 bucket (progress8kpabs、 係数は progress.bin) ごとに別の重み
  評価値には bucket ごとの補正 (P2_BUCKET_OFFSET_*、 先手番の探索では別の表) を足す
                └──────────────────────────────────────────────────────────────┘
                ┌──────────── OPN opn.bin (序盤専用) ──────────────────────────┐
HalfKA_hm K16 27,693 (玉区分 16 + factoriser) ─> FT 2048 x 2 (threat なし・PSQT なし)
  ─> fc0 2048 -> 16 ─> 32 ─> 1 (8 bucket の LayerStack、 学習は bucket 0〜2)
                └──────────────────────────────────────────────────────────────┘
切り替え: 進行度が OPN_ProgHi (既定 -1946) より序盤側なら OPN、 それより後は本体だけ (OPN_ProgWidth 0 = 混ぜない)。
         v9 の既定では OPN が受け持つのは進行度 bucket 0 だけ。
```

| 部品 | 中身 | 何のためか | 詳細 |
|---|---|---|---|
| HalfKA_hm | 玉の位置 (鏡像で 45 区分) x 全駒の位置・持駒 | 本体の主入力 (NNUE の標準形) | engine 2 章 |
| threat 216,720 → 64 | 「どの駒がどの駒に利いているか」の対 (玉を除く)。 出力は 64 列だけ | 駒の当たり・ひもを直接見る | engine 3 章、 `SPEEDUPS_STORY.md` 1 章 |
| PSQT | 駒の位置の線形項 (bucket ごと) | 素朴な駒得の近道 | engine 2 章 |
| KH | 浮き駒 (どの駒にも守られていない駒) の 78 次元 → 64 列 | 当たりの危険を補う | engine 5 章 |
| LayerStack (進行度 8) | 進行度で fc0 以降を切り替える。 進行度の係数は NAGISA v3.1 の `progress.bin` (GPLv3) | 序盤と終盤で別の判断 | engine 4 章 |
| 量子化 (L1Q128 / FTI32 系 / FC1PREC) | fc0 の重みを 1/128 刻みに、 FT を 8 倍細かい刻みに、 fc1 と後段の活性を細かく | 量子化の誤差 (手の入れ替わり) を減らす | engine 6 章・15 章、 `TECHNICAL_NOTES.md` 2.6 |
| bucket 補正 | LayerStack の bucket ごとに評価値へ足す定数。 先手番の探索と後手番の探索で別の表 | 量子化の精度を上げたときに消えた「手番側を悲観的に見る片寄り」を戻す | engine 15 章、 `TECHNICAL_NOTES.md` 2.7 |
| OPN 2048 | 序盤専用の幅 2048 の net。 本体の FT を半分の列に写して出発し、 DL 系のラベル付きデータで学習 | 序盤の判断の正確さ | engine 7 章 |

### 1.3 探索

- やねうら王の探索 (Stockfish 系) をそのまま使い、 係数を USI option (`SPSA_*` 約 190 個) に出して SPSA と移植で調整しました。 v9 では自己対局の SPSA (SPSA24 の 29 周目) の値を option の既定値に焼き込んであるので、 探索の option を渡す必要はありません。
- 独自の変更: INFREP (優等・劣等局面の判定を対局の履歴まで広げる、 `P2InferiorRepHistory`)、 `HIST_CONFIDENCE_SCALE` (history の 3 成分が同じ向きのとき LMR を強める)、 定跡を抜けた直後の時間配分の修正、 探索側の結果を変えない高速化の束 (SRCH_BUNDLE_A) など。 → engine 8〜11 章
- 高速化: threat の多段 lazy 差分更新 (LZ) と、 その仕上げ (IDXV2・ROWPF・D1FAST)、 利きライブラリを外す、 PGO、 `-march` の修正など。 → `SPEEDUPS_STORY.md`
- 並列: やねうら王の Lazy SMP のまま (配布の既定は Threads 4)。 診断の結果と独自の委譲機構 M4 (既定 off) は `SPEEDUPS_STORY.md` 9 章。

## 2. 使い方の注意 (配布物)

- `EvalDir` (既定 `eval`) に次の 5 つを置きます。 どれかが欠けると `isready` で Error を出して止まります (黙って弱い評価で動くのを防ぐため)。

  | ファイル | 中身 |
  |---|---|
  | `nn.bin` | 本体 (FTI32K8 + L1Q128 + F1P128 形式。 古い exe は読めずに止まる) |
  | `kh.bin` と `kh.bin.json` | KH の重みとその説明 |
  | `opn.bin` | OPN (FTI32K8 + F1P128 形式) |
  | `progress.bin` | 進行度の係数。 **NAGISA v3.1 由来で GPLv3** (NAGISA のリリースページと、 作者の 2026-11-06 の投稿「v3 は nn.bin まで含めて GPLv3 に変更」による。 原本と md5 一致) |

- USI では `id name p2shogi v9` / `id author pekoshi` と名乗ります。
- 既定値: Threads 4、 定跡なし。 探索係数・bucket 補正・FV_SCALE (bucket ごと 26 / 26 / 27 / 28 / 28 / 28 / 28 / 28) は公開版の値を既定に焼き込み済み。 千日手の評価値 `DrawValueBlack` / `DrawValueWhite` は -2 / -2 (8 章)。 `NetworkDelay2` / `MinimumThinkingTime` / `RoundUpToFullSecond` はやねうら王の標準値 (1120 / 2000 / true) のままです。
- 🚨 **秒読みが短い (例: 1 手 1 秒) ときは `NetworkDelay2` を小さく (例: 120) してください。** そうしないとほとんど考えずに指します。 実測では byoyomi 1000 ms・`NetworkDelay2=2000` で 1 手 105 ms しか考えず (2.8 万ノード / depth 12)、 `NetworkDelay2=120` なら 23.8 万ノード / depth 18 (8.37 倍) でした。
- 動作には AVX2 対応の CPU が必要です。
- 定跡まわりの option (`BookExitFile` など) のコードは公開しますが、 定跡ファイルとそれを作った元データは公開しません。 起動時に `can't read file : ./book/standard_book.db` と出ることがありますが、 既定では定跡を使わないので動作には影響しません。
- 棋風のボーナス option (`FuribishaBonus` など 20 個) は既定 0 で挙動に影響しません。 棋風を変える実験用に残しています。

## 3. 開発の経緯 (要約)

| 時期 | 出来事 |
|---|---|
| 2026-04〜06 | ラム将棋 (rshogi、 Rust) の fork で探索と NNUE を試作。 探索側の NPS 改善 (+7.94%) や Rust 版 NNUE の実験 |
| 2026-06〜07 | エンジンをやねうら王 (`717da87`) に移し、 P2NNUE を C++ で実装。 threat 特徴 (駒の当たりの対) を入れ、 「full threat 搭載」を公表 |
| 2026-07〜09 | データの拡大 (DL 系ラベルの novel データ)、 threat の出力幅を 64 に、 PGO・`-march` の修正。 NAGISA v3.1 を当面の比較相手に。 floodgate で運用 |
| 2026-10 | 中盤 2,000 局の判定器、 データ軸の探索 (9 本とも外れ)、 定跡の整備。 `top3q-102` (以後の FT の土台) |
| 2026-11 前半 | 比較相手を NAGISA v4 に。 threat の税が分岐予測ミスだと特定、 LZ (多段 lazy 差分) で回収。 KH (浮き駒) を追加 |
| 2026-11 中旬 | 序盤専用の OPN (幅 2048) を学習・高速化、 L1Q128、 INFREP、 LZ の仕上げ。 電竜戦 (11-15 終了) の後、 公開の準備 |
| 2026-11 下旬 | FT と後段の量子化の精度を上げ (FTI32 系・FC1PREC)、 bucket 補正と先後別の表、 SPSA24 の値を既定に焼き込み。 p2shogi v9 として公開 (8 章) |

開発のやり方: 実装と計測の多くを AI (Claude) と分担しました。 施策ごとに測る前に判定線を書き (事前登録)、 結果を判定記録に残しました。 失敗の記録 (誤った数字・器の欠陥) の要約は `SPEEDUPS_STORY.md` 10 章と `TECHNICAL_NOTES.md` 5 章にあります。

## 4. 到達点

### 4.1 対局の成績

| 対局 | 条件 | 結果 |
|---|---|---|
| floodgate (`p2shogi_v9_nobook_nnue_i7700_4T`) | Core i7-7700 の 4 スレッド、 定跡なし | 2026-11-23 に 28 局でレート **4248** (115 名中 32 位)。 56 局で 33 勝 13 敗 10 分。 レート 4300 以上の相手には 1 勝 10 敗 3 分と分が悪く、 実力は**レート 4250 前後**と見ています |
| v9 対 NAGISA v4 (同ノード) | 30 万ノード、 Threads 1、 j100 局面集の 100 局面 x 先後 = 200 局 | **38.0%** (局面 1〜50: 48.0% / 局面 51〜100: 28.0%) |
| v9 対 NAGISA v4 (時間制) | 1 手 1 秒、 Threads 1、 100 局 | 約 **39.5%** |
| (参考) 11-10 の版 対 NAGISA v4 | 1 手 1 秒 (byoyomi 1000)、 Threads 1、 Hash 512 MB、 定跡なし、 ND/ND2 120、 ply24 互角局面 423 x 先後 = 845 局 | 40.0% [36.8, 43.1] (対応ペア) = -70 Elo |

- NAGISA v4 (やねうら王系、 幅 2304・threat なし) とはまだ互角に届いていません。 探索はやねうら王の同じコードから出ていて、 違いは主に評価関数と各自の SPSA です。
- 同じ局面集でも前半と後半で 20pt 違います。 局面集の偏りが大きいので、 1 つの局面集だけで「改善した」と読まないでください (`TECHNICAL_NOTES.md` 5 章)。
- 同じ時間で読めるノード数は相手より多い (対 v4 で 1.12 倍・+1.0 ply、 11-06 の測定) のに勝ち越せていないので、 残る差は**評価の質**だと読んでいます (`FUTURE_WORK.md` 1 章)。

### 4.2 評価関数の篩 (静的評価の質の物差し)

篩 = 対 v4 の対局から取った 130,965 局面で、 静的評価が「10 手先の両エンジン 1 秒探索の平均」をどれだけ当てるか (手番別の順位相関の平均)。 値が大きいほど良い。 11-16 時点の net の値です。

| net | bucket 0-1 (序盤) | bucket 2 | bucket 2-7 |
|---|---:|---:|---:|
| 我々の本体 (L1、 KH あり) | 0.2583 | 0.3937 | 0.7307 (L1Q128 は +0.0027) |
| 我々の OPN | 0.3016 | — | — |
| NAGISA v4 | 0.3127 | 0.4125 | 0.7257 |

序盤 (OPN が受け持つ) と bucket 2 で相手に届いていません。 bucket 3〜7 は並んでいます。 篩は参考程度の物差しで、 bucket 補正のような定数の足し算で対局の結果が大きく動くこともあったので、 篩の差だけで勝敗を予想しないでください。

### 4.3 速さ

- 公開版の NPS の主な積み上げ: 利きライブラリを外す + PGO の採り直し (+7.96%)、 LZ (+4.95%、 のち PGO 128 周で +2.25%)、 LZ 段 1 (IDXV2 + ROWPF + SRCH_BUNDLE_A、 PGO 同士 +4.19% [+3.43, +5.01])、 final2 の 3 本 (+1.74%、 28 周で打ち切り)。 詳細は `SPEEDUPS_STORY.md` の表。
- v9 では評価の精度のために FT の刻みを細かくし、 NPS を −6.53% [−7.21, −5.90] 払っています (8 章)。
- threat の税 (threat を剥いだ net 比): 約 22〜30%。 その本体は分岐予測ミスでした。

## 5. 開発環境と、 それが結果に与えた制約

| 資源 | 中身 | 影響 |
|---|---|---|
| CPU | Intel Core i7-7700 (Kaby Lake、 4C/8T、 AVX2 のみ、 VNNI・AVX-512 なし、 L2 256 KB、 L3 8 MB) | 計測・対局・学習の前処理が 1 台で競合する。 対局は同時 2 本まで、 確認対局は Threads 1。 fc0 の高速化の結論 (疎より密) は AVX2 の話 |
| GPU | NVIDIA GeForce GTX 1050 Ti (VRAM 4 GB)、 ローカル 1 枚 | 学習 1 SB (1 億局面) に約 1 時間。 net の幅・データ量・試行回数がここで決まった。 DL で教師を付け直すことは非現実的。 クラウド GPU は使わない方針 |
| OS | Windows 10、 Git Bash、 llvm-mingw (clang 23) | ビルドは clang の instr PGO。 large page は汎用性がないので使わない |

## 6. 公開と、 開発の終了

- 公開: 2026-11、 p2shogi v9。 コード 3 本 (エンジン・学習器・道具)、 exe と評価関数一式、 この文書群。
- 公開で開発はいったん終了します。 理由は開発 PC の限界 (上の表) で、 これ以上強くする手が手元の資源では難しいためです。 続きの候補は `FUTURE_WORK.md` に引き継ぎとして書きました。
- やねうら王への高速化の提案 (PR) は任意です。 候補は `UPSTREAM_PROPOSALS.md`。

## 7. ライセンスと第三者の資産

| 物 | ライセンス・扱い |
|---|---|
| エンジン | GPLv3 (やねうら王から継承) |
| 学習器 | MIT (bullet から継承) |
| 道具 | GPLv3 (ラム将棋から継承。 一部 crate は MIT) |
| `progress.bin` | NAGISA v3.1 由来、 GPLv3 (上の 2 章) |
| 我々の net (`nn.bin` / `kh.bin` / `opn.bin`) | 公開する。 学習データの出所は `REPRODUCE.md` 2 章 |
| NAGISA v4 の評価関数・定跡 | 同梱しない |
| 我々の定跡ファイルと元データ | 公開しない |
| 実験記録 (設計書・判定ログ・対局の生ログ・作業メモ) | 公開しない |

## 8. 11-17 以降の変更 (v9)

11-17 にこの文書群の初版を書いた後、 公開版 p2shogi v9 までに次のことをしました。 エンジン側の詳細は `CHANGES_FROM_UPSTREAM_engine.md` 15 章、 仕組みの解説は `TECHNICAL_NOTES.md`。

### 8.1 公開版の中身

| 項目 | v9 の値 |
|---|---|
| exe | `p2shogi_v9.exe` (USI の `id name p2shogi v9` / `id author pekoshi`)。 Threads 既定 4、 定跡は既定で使わない |
| 評価関数の精度 | FT を 8 倍細かい刻み (`-DP2_FT_KS=3`) で持ち、 transform の出力を i16 (`-DP2_FT_OUT16=5`) にして fc0 を i16 入力の疎な積和 (`-DP2_FT_SPARSE16`) で回す。 fc1 の重みを 1/128 刻み、 fc0・fc1 の活性を 2 ビット細かく (`-DP2_FC1_PREC`)。 net は学習し直さず、 同じ f32 の重みから書き直した |
| 探索の係数 | 自己対局の SPSA (SPSA24) の 29 周目の値。 調整した 47 個の option と `SPSA_LMR_TABLE_COEFF_OPENING=2763` を既定値に焼き込んだ |
| bucket 補正 | LayerStack の bucket ごとに評価値へ足す定数 (`P2_BUCKET_OFFSET_B0..B7`)。 我々が先手番の探索では別の表 (`P2_BUCKET_OFFSET_SENTE_B0..B7`) を使う。 `P2_SENTE_TABLE_SWITCH_CP=500`: 直前の手の root の評価値が表示 cp で +500 以上なら、 後手番でも先手の表を使う |
| 既定値の焼き込みの確認 | ファイルのパスの option だけを渡して起動した探索が、 以前の 206 個の option を渡した設定と、 40 局面 x 100 万ノードで同一 |
| 千日手の評価値 | `DrawValueBlack` / `DrawValueWhite` の既定を -2 / -2 に。 以前の -53 / +41 は電竜戦の規定 (引き分けを先手 0.4 / 後手 0.6 と数える) に合わせた値で、 floodgate では後手が千日手を受け入れやすすぎた |
| OPN | 進行度 bucket 0 だけを受け持つ (`OPN_ProgHi=-1946`)。 配布の OPN は 11-16 と同じ `opnF2Qx-16` (上の精度の形式に書き直したもの) |
| PGO | 公開版の option で、 実戦の 60 局面 x 100 万ノードで採取 (手数の帯 24-59: 15 局面 / 60-99: 20 / 100-139: 15 / 140 以上: 10)。 CSPGO (IR の文脈つき PGO) も試したが、 リンクで `__llvm_profile_filename` が未定義になり使わなかった |
| final2 | transform の 32 B store・駒単位の追いつき・本体 fc0 の先読みの 3 本を採用 (NPS 101.74%、 28 周で打ち切り)。 BACKMAT は不採用 |

### 8.2 強さ

4.1 の表のとおりです。 floodgate ではレート 4248 (11-23、 28 局) で、 実力は 4250 前後と見ています。 NAGISA v4 との同ノード比較 (30 万ノード、 Threads 1、 200 局) は 38.0%、 1 手 1 秒 Threads 1 は約 39.5% (100 局) で、 互角には届いていません。

### 8.3 試して採らなかったもの

| 試したこと | 結果 |
|---|---|
| LMR の係数 2490 (浅く広く読み、 v4 と同じくらいの深さにする) | 56 局で 38.4%。 悪手の試験でも変化なし |
| threat 入力の倍率 k = 0.90 / 1.05 / 1.10 | 対局で負けた |
| 手の並べ順のための policy head | p2shogi には入れなかった |
| OPN を bucket 0〜1 に広げる | 対局で負けた |
| OPN を FT も動かして学習し直す | 対局で勝てなかった |
| 地平線問題・評価の楽観を探索で補う腕 (fail-low で窓を広げる、 LMR・SE・futility・ProbCut の係数を本家寄りに、 bucket 2 限定の腕など) | どれも採用に至らなかった |
| M4 (置換表を介した仕事の委譲、 `SPSA_YBWC_MIN_DEPTH`) | 既定 off のまま (`SPEEDUPS_STORY.md` 9 章) |
