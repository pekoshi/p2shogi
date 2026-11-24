# 道具の改造点 (SH11235/rshogi `afd941a` から) と python の器

- 土台: `SH11235/rshogi` の `afd941ab` (2026-04-30、 「csa-client: UsiEngineDriver trait で外部 USI 以外の engine 実装を受け入れ可能にする (#575)」)。 手元の写しに git が無かったため、 各ファイルの git blob hash を本家の候補 commit の tree と突き合わせて、 一致するファイルが最も多い commit を基点とした。 公開リポジトリはこの commit から GitHub の fork として履歴をつなぎます。 🚨基点は推定です。 近い commit との差は小さいはずですが、 一意とは言い切れません。
- ライセンス: GPLv3 (各 crate の表記は fork が `GPL-3.0-only`、 本家の 6 月以降は `GPL-3.0-or-later`。 `rshogi-csa` は MIT)。
- 本家への提案はしません (方針)。 この文書は公開物を読む人のための対応表です。

## 0. 使い方の位置づけ

道具のリポジトリは 2 つの役割を持ちます。

1. **対局と計測の器** (Rust): 対局器 `tournament`、 SPSA、 CSA の対局クライアント、 PSV の変換。 確認対局・判定はすべてここの `tournament` で行いました。
2. **python の器** (`tools/python/`): ビルド・PGO・bit 一致の関門・NPS の A/B・篩・net の後処理・対局の集計。

初期 (2026-04〜06) はこのリポジトリで Rust 版のエンジン (`rshogi-usi` + `p2-nnue` / `p2-search`) も育てましたが、 公開する C++ エンジンとは別物です。

## 1. crate

| crate | 本家 | このフォーク | 公開版 |
|---|---|---|---|
| `rshogi-core` | ある | 実験用の feature (NNUE の追加入力など) を追加。 学習器が依存する | 残す |
| `rshogi-csa` / `-csa-client` / `-csa-server*` | ある | `csa_client` を floodgate 運用に使用 | 残す。 🚨floodgate の接続設定 (パスワード入り) は公開しない |
| `rshogi-usi` | ある | Rust 版エンジン (初期の実験) | 残す |
| `tools` | ある | bin を 30 本超追加 (下の表) | 使った bin を残し、 どこからも使われない bin 20 本を消す |
| `p2-nnue` / `p2-search` / `p2-midgame-policy` | 無い | Rust 版エンジンの評価・探索・policy | 他の crate から使われているので残す (依存の書き換えは公開物の正しさに関わるため) |
| `bucket-profiler` / `nnue-b` / `nnue-convert` / `nnue-trainer-rs` / `p2-progress-predictor` / `p2-transformer*` / `hcpe3_aggregator` | 無い | 5〜6 月の実験 | 消す |

## 2. `tournament` (対局器) の変更

本家の `tournament` (1,478 行) を 2,894 行に拡張しました。

| 変更 | 中身 | なぜ |
|---|---|---|
| `--startpos-sequential` | 局面集を先頭から順に、 1 局面 x 先後を 1 回ずつ指す | 既定の動作は run ごとに局面を復元抽出し直すため、 被覆が 60〜68% になり腕どうしで局面が揃わなかった |
| `control.json` | 走行中に目標局数などを変える | 🚨`target_games` は「1 方向あたり」の局数。 途中で値を書き換えても反映されないことがある |
| `position ... moves ...` を渡す | 開始局面から手順付きで送る | 手順を渡さないと千日手を検出できなかった |
| 時間・固定ノードの枠の拡張 | byoyomi・固定ノード・同時対局数まわり | 時間制と固定ノードの両方の判定に使うため |

注意: 対局 jsonl の `game_id` は**ファイルごとの連番**です。 複数ファイルをまとめるときは (ファイル, game_id) を鍵にしてください。

## 3. 追加した bin (主なもの)

| bin | 用途 |
|---|---|
| `csa_to_psv` / `psv_to_sfen` / `sfen_to_psv` / `psv_to_jsonl` / `filter_hand_psv` / `jsonl_to_psv_dakai` | 学習データの変換 |
| `build_washiun_train` / `build_dataset` / `rescore_psv` / `shuffle_psv` | 学習データの作成・再ラベル (`rescore_psv`、 `shuffle_psv` は本家にあるものを拡張) |
| `spsa` | 探索係数の SPSA (本家にある bin を拡張) |
| `export_p2nnue` / `p2nnue_usi` / `train_p2nnue` | Rust 版 P2NNUE (初期) |
| `train_progress_kpabs` | 進行度の係数の学習 (最終 net は NAGISA v3.1 の `progress.bin` を使うので不使用) |

## 4. python の器 (`tools/python/`)

開発中は実験ごとのフォルダに散らばっていた器を、 用途別に集めて `tools/python/` に置きました (公開先は https://github.com/pekoshi/rshogi の `p2shogi` ブランチ)。 学習の手順に要る器の一覧は `TRAINING_FROM_SCRATCH.md` の「必要な python スクリプト」にあります。 絶対パスは `paths.toml` (見本 `paths.example.toml`) に寄せ、 環境変数 → `paths.toml` → 引数の順で解決します。 🚨置き換えた器は「同じ入力に同じ出力」を小さい入力で確かめたものだけを入れ、 確かめていないものは README に「未検証」と書きます。

| 区分 | 器 | 中身 |
|---|---|---|
| net の後処理 | `mk_l1q128.py`、 `fti32_write.py`、 `fc1prec_write.py` (+ 読み出しの部品 `k256_merge.py`)、 `kh_scale.py`、 `f1_transplant2048.py` (+ `f1_transplant.py`)、 `calib_l1q128.py` | L1Q128・FT 8 倍刻み・fc1 精度版の書き出し、 KH の倍率、 OPN の FT の写し、 OPN の目盛り |
| 照合 | `engcheck.py`、 `fti32_verify.py`、 `fc1p_verify.py`、 `fc1p_verify_opn.py` | エンジンの静的評価と python の整数推論の全局面一致 |
| bit 一致・NPS | `gate.py`、 `nps_ab.py`、 `nps_cputime_ab.py`、 `nps_ab_prod.py` (+ 部品 `eng.py` `v4c.py`) | 関門 (`go nodes` 全一致)、 CPU 時間の ABBA |
| 篩 | `screen_future.py`、 `b01_screen.py`、 `pair_screen.py`、 `opn_static2.py`、 `screen128.py`、 `screen_kh_onoff.py` (+ 部品 `p0.py` `make_sets.py` `m2_games_static.py` `opn_static.py` `bullet_inference_v19.py`) | 静的評価の質 |
| 学習データ | `ponko_novel.py`、 `calib_lut.py` | 既見局面の除去、 score の換算 |

対局の運用・棋譜の解析に使った器 (`sprt.py`、 `run_league.sh`、 floodgate の棋譜の取得・集計など) は公開に含めていません。 閉じた実験だけに使った器も入れていません。

## 5. 使うときの罠 (記録から)

- USI option の区切り文字が器ごとに違う (`exe_agree_posset.py` はカンマ、 `run_league.sh` は空白)。
- `go` の直後に `quit` をパイプで送ると探索が即止まる。 `bestmove` を読んでから `quit`。
- `usinewgame` はこのエンジンでは何もしない。 置換表と history を消すには `isready`。
- `go nodes N` は Threads=1 でも完全には決定的でない (同じ局面 8 回で depth 15〜17)。 局面ごとに `isready` で揃える。
- 判定の完了待ちをプロセス数で書くと、 別の対局が始まった瞬間に永久に終わらない / 自分のコマンド行を数える。
