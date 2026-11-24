# 学習器の改造点 (nodchip/bullet-shogi `df431b3` から)

- 土台: `nodchip/bullet-shogi` の `df431b3b753b710a62c5a8dd1fc6be11b6ce7ff3` (2026-05-07、 「resume 空文字を未指定として扱う」)。 その上流は `jw1912/bullet`。 公開リポジトリは https://github.com/pekoshi/bullet-shogi の `p2shogi` ブランチで、 この commit からの GitHub の fork です。
- ライセンス: MIT (bullet から継承。 著作権表示は保持)。
- 学習の追試の手順は `REPRODUCE.md`。 この文書は「何を足したか」をまとめます。
- やねうら王の公式の学習器 BulletOu (`yaneurao/BulletOu`、 SH11235/bullet-shogi の fork) とは**別の枝**です。 我々の差分は BulletOu にそのまま当たりません。 BulletOu への提案はしません (方針)。 下の表の「BulletOu にあるか」の列は、 読む人が機能を対応づけるための参考です。

## 0. 差分の規模と取り方

| 区分 | 量 |
|---|---|
| 土台のファイルへの変更 | 73 ファイル、 +20,513 / −1,883 行 (うち `examples/shogi_layerstack.rs` が +14,370 行) |
| 新規 | 入力型 (`shogi_kh.rs`、 `shogi_halfka_hm_threat_profile11split.rs` ほか)、 最適化器 `ranger21.rs`、 検算の examples ほか |

ファイル別の増減は `diff/trainer_diffstat_vs_df431b3.txt`。 🚨11-17 時点の数字です。 公開版にも閉じた実験の入力型・引数・examples が残っています (引数で使わなければ無効)。

```bash
git diff --stat df431b3          # 公開リポジトリで
git diff df431b3 -- examples/shogi_layerstack.rs
```

🚨`crates/bullet_lib/Cargo.toml` は道具のリポジトリの `rshogi-core` を相対パス `../../../rshogi/crates/rshogi-core` で参照します。 ビルドするときは、 道具のリポジトリ (https://github.com/pekoshi/rshogi の `p2shogi` ブランチ) を学習器と同じ親フォルダに `rshogi` という名前で置いてください (`TRAINING_FROM_SCRATCH.md` 2 章)。 この依存があるので、 学習器のバイナリは GPLv3 の `rshogi-core` を含みます。

## 1. 土台が既に持っていたもの / このフォークで足したもの

| 機能 | 土台 (nodchip 版) | このフォーク | BulletOu にあるか (参考) |
|---|---|---|---|
| 将棋の入力 HalfKA / HalfKA_hm、 LayerStack、 進行度 bucket (`--bucket-mode`、 `--progress-coeff`)、 PSQT | ある | 使う | ある |
| threat の入力型 (`shogi_halfka_hm_threat`) | ある (初版) | 本番の profile (除外規則・群化・64 出力) に書き直し | ある (上流由来)。 我々の profile との差は未確認 |
| factoriser | ある | OPN の K16 で使う | ある |
| `--loss-power` (損失の冪) | 無い | 追加 | ある (`--loss-pow-exp`) |
| `--loss-scale` (教師の勝率変換の目盛りだけを変える) | 無い | 追加 | `--scale` と WRM の定数で近いことができる可能性 (未照合) |
| `--bucket-filter` / `--hard-freeze-shared` / `psv_bucket_filter` | 無い | 追加 | 無い (dataloader に除外の分岐が無い) |
| threat の凍結・群化 (`--threat-freeze(-hard)` / `--threat-bucket-groups` / `--threat-out` / `--threat-ft-lr-mult` / `--threat-grad-mask`) | 無い | 追加 | 未確認 |
| KH (`--kh` 一式) | 無い | 追加 | 無い |
| OPN の K16 入力 (`--opn-king16`) | 無い | 追加 | king bucket はある (k3k3 など。 区分の定義が違う) |
| QAT (`--qat`) | 無い | 追加 | ある (tatara 由来、 dense QAT + STE) |
| Ranger21 (`--optimizer ranger21`) と resume の修正 | 無い | 追加 | — |
| 重みだけの resume (`--resume-weights-only`) | 無い | 追加 | 未確認 |
| `--output-format rshogi` (エンジンが読む形式) | 無い | 追加 | — (形式が別) |
| record shuffle (256 MB チャンク内) | 無い | 追加 | 窓 shuffle が opt-in である |

## 2. 入力

| 入力 | 中身 | 場所 | 使う net |
|---|---|---|---|
| HalfKA_hm | 73,305 次元 (玉 45 区分 x 1,629) | `crates/bullet_lib/src/game/inputs/shogi_halfka.rs` | 本体 |
| threat (profile 0 = full) | 216,720 次元。 攻め駒・標的・相対位置の対。 玉は含めない。 除外規則は `shogi_threat_exclusion.rs` | 本番の型は `shogi_halfka_hm_threat_profile11split.rs` (`inputs.rs` の `#[path]` で `shogi_halfka_hm_threat` として読む)。 🚨同名の使われないファイル `shogi_halfka_hm_threat.rs` がある (読まれるのは profile11split 側) | 本体 |
| KH | 浮き駒 78 次元 (h 部) + 玉近傍 (k 部 64、 最終 net では使っていない) | `shogi_kh.rs`、 `examples/kh_feat_dump.rs` | 本体 (KH 段) |
| HalfKA_hm K16 | 玉区分 16 (自玉 7〜9 段 x 5 筋 + その他) x 1,629 = 26,064 + factoriser 1,629 | `inputs.rs` (`--opn-king16`)、 検算 `examples/opn_k16_check.rs` | OPN |

## 3. 損失

| 引数 | 中身 | 最終 net での値 |
|---|---|---|
| `--scale` | 出力の目盛り (評価値 = 出力 x scale) | 600 |
| `--loss-power` | 損失 = \|予測 − 教師\|^p | 2.6 |
| `--loss-scale` | 教師の勝率変換だけ `sigmoid(score / loss_scale)` にする (出力の目盛りは `--scale` のまま)。 互角付近に勾配を配る | OPN 300 (本体は使わない) |
| `--wdl` | 勝敗の混ぜ率 | 本体 0.15 / OPN 0 |

記録: 本体の `--wdl 0` は篩で 5 SB すべて負。 FT を `--loss-scale 300` で動かすと、 データの系統によって bucket 3-7 を壊す (`FUTURE_WORK.md` 5 章)。

## 4. 部分学習と凍結

| 引数 | 中身 | 使い方 |
|---|---|---|
| `--bucket-filter a-b` | 学習時に進行度 bucket a〜b の局面だけを通す | OPN (0-2)。 🚨`--hard-freeze-shared` なしで使うと FT・PSQT・factoriser は全 bucket 共有なので範囲外の bucket の評価も変わる (学習器が警告を出す)。 OPN では範囲外を使わないので意図どおり |
| `--hard-freeze-shared` | 全 bucket 共有の層を凍結 (勾配ノードを作らない) | 本体を帯ごとに直すとき |
| `--threat-freeze` / `--threat-freeze-hard` | threat の表を凍結 (hard は勾配ノード・optimiser 状態を作らない) | 本体の M1・M2 段 |
| `--threat-bucket-groups 0,1,1,2,2,3,3,3` | bucket → 群の対応。 群ごとに別の threat の表 | 本体 |
| `--threat-group-mem-budget-mib` | 群化の GPU メモリの上限 | 4 GB の GPU で 1200 |
| `--kh --kh-feats h --kh-freeze-rest --kh-unfreeze-fc0 --kh-lr-mult 64 --kh-dump <file>` | KH を足し、 KH と fc0 以外を凍結 (lr_scale 1e-12、 decay 0、 lookahead 無効)、 KH の lr を 64 倍、 KH の f32 重みを書き出す | 本体の KH 段 |
| `--resume-weights-only` | 重みだけを読み、 optimiser 状態は新規 | OPN の最初の段 (FT を写した初期値から) |

FT を別の net から写して出発する手順 (幅の違う net への写しを含む) は python (`f1_transplant2048.py`) で行います (`REPRODUCE.md` 6.1)。 一から学習した幅 2048 は同じ 2 SB で篩 0.2050、 写しから始めると 0.2756 でした。

## 5. 量子化・QAT・保存形式

- 保存: `--output-format rshogi` (エンジンの P2NNUE が読む形式)。 FT は QA=127 (積は 254 まで)、 層の重み QB=64、 bias 8128。 公開版 (v9) の net は、 学習後に python で 3 段の書き直しをしています: L1Q128 (fc0 の刻み 1/128)、 FTI32 系 (FT の刻みを 8 倍細かく)、 FC1PREC (fc1 の刻み 1/128 と後段の丸め)。 どれも学習器を使わず、 f32 の重みから作り直します (`REPRODUCE.md` 5 章)。
- `--qat`: 順伝播で FT の積 (Q=127 切り捨て)、 L1・L2・L3 の重み (Q=64 四捨五入、 l1w + l1fw を足してから [−2, 127/64] で clip)、 bias (Q=8128)、 活性 (Q=127 切り捨て) を量子化し、 逆伝播は STE。 master 重みは f32。 OPN の最後の 3 SB で使った (効果は +0.002 前後で 2SD を超えず。 害は無いので採用)。

## 6. データ

| 機能 | 中身 |
|---|---|
| PSV の読み込み | 40 B / 局面の PSV。 複数ファイルを `--interleave-file-batches 1` で 1 batch ずつ交互に |
| record shuffle | 256 MB のチャンク内で Fisher-Yates。 seed は `--file-shuffle-seed` (既定 0)。 🚨ファイルの並びの偏り (先頭に固まった局面) は消えない |
| `examples/psv_bucket_filter.rs` | 学習器と同じ進行度の計算で PSV から bucket 範囲を抜き出す (学習器の計算と 7,773 万件で不一致 0 を確認) |

## 7. 最適化器

- Ranger21 を追加 (`crates/acyclib/src/trainer/optimiser/ranger21.rs`)。 最終 net の全段で使用 (weight decay 0.005、 lookahead mergetime 20)。
- 🚨 `--resume` で Ranger21 を続けると、 内部の予定 (total steps) が尽きて lr が 0 になり、 学習が静かに何もしない不具合があった。 修正済みで、 段をまたぐときは `--ranger21-total-steps 0` を付ける。
- lr の減衰は SB の絶対番号で決まるので、 最終 net では `--lr-gamma 1.0` のまま 1 SB ずつ起動し直して lr を引数で下げた。

## 8. 検算の examples

| example | 中身 |
|---|---|
| `bucketed_affine_check` | bucket 別の affine の順伝播を CPU で照合 |
| `opn_k16_check` | K16 入力の index を照合 |
| `kh_feat_dump` | KH の特徴を局面ごとに書き出し、 エンジン側と照合 |
| `psv_bucket_filter` | 6 章 |

## 9. 試して採らなかったもの

追加入力: AUX / AUXCHEAP / MOB / KZT / ActivityRisk / KingRing / PairShape / DualCross / Pin / HandThreat / Direct15 / policy 系。 構造: residual-wide16、 threat → L1 / L2 直結、 L4 (4 層目)、 幅 1536、 hand dense。 学習法: 注入率・データ配合の 9 本、 SWA、 FT を動かす追加学習。 学習時のログの冒頭に出る `disabled` の行の一覧が、 そのまま閉じた機能の一覧です。

## 10. 学習に使った exe とソースの対応 (注意)

最終 net を作った exe は、 未 commit の作業ツリーから 3 つの時点でビルドしたものです (KH 段 = 11-12、 OPN = 11-14、 OPN の QAT = 11-15)。 各時点のソース全体は残っていません。 公開版の学習器は引数で全機能を切り替える 1 本で、 全段を回せるはずですが、 公開版のビルドで学習を回した記録はまだありません。 回すときは、 学習の冒頭のログが `REPRODUCE.md` の期待と一致することを最初に確かめてください。
