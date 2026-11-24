# 独自の機構の手引き (新しい機構を作る人へ)

p2shogi の独自の機構を、 土台にして新しいものを作る人のための手引きです。 機構ごとに「何をするか / ソースのどこにあるか / どう切り替えるか / どう測ったか / 何がうまくいかなかったか」をまとめました。

- パス: エンジンは `<engine>/source/` からの相対 (https://github.com/pekoshi/p2shogi 、 master)。 学習器は `<trainer>/` (https://github.com/pekoshi/bullet-shogi の `p2shogi` ブランチ)。 python の器は `<tools>/tools/python/` (https://github.com/pekoshi/rshogi の `p2shogi` ブランチ)。
- 切り替え: エンジンの機構の多くは `#if defined(P2_xxx)` のビルド時 define と、 USI option の 2 段です。 define を外すとそのコードはビルドに入りません。 option は既定値なら従来と bit 一致になるように作ってあります。 公開版の define の組は `<engine>/README.md` と `CHANGES_FROM_UPSTREAM_engine.md` 0.3。
- 高速化の個々の仕様は `speedups/*.md` (1 件 1 ファイル、 AI にも渡せる形) と `SPEEDUPS_STORY.md` にあります。 ここでは機構の中身と、 その上に何かを足すときの注意を中心に書きます。
- 仕組みの「なぜ」は `TECHNICAL_NOTES.md`。

## 0. 新しい機構を足すときの型 (我々のやり方)

1. **define で囲む**。 既定のビルドに入れない状態で、 既定値 = 従来と bit 一致を先に確かめる。
2. **新旧 2 経路の検算**を診断ビルドに置く (`*_VERIFY` define)。 同じ exe の中で古い経路と新しい経路を両方計算し、 不一致を数える。 例: `P2_LZ_VERIFY` (`eval/p2nnue/p2nnue_lz.h`)、 `P2_ACC_PIECEWISE_VERIFY` / `P2_TR_STORE32_VERIFY` (`eval/p2nnue/p2nnue.cpp`)。
3. **探索の関門** (`gate.py`): 同じ局面を `go nodes` で読ませ、 手・nodes・depth・score・pv の全一致を見る。 Threads 1、 局面ごとに `isready`、 先に base 対 base。 我々の形は 49 / 150 局面 x 20 万ノード + 24 局面 x 200 万ノード (崩れが 4% の局面にしか出ない不具合があったので広めに取る)。
4. **NPS の A/B**: CPU 時間で、 ABBA の交互実行、 64〜128 周、 PGO 同士 (`speedups/M00_measurement.md`)。
5. **評価を変える機構**は、 篩 (静的評価の質) で見当を付け、 対局で決める。 篩は参考程度 (`TECHNICAL_NOTES.md` 4 章)。
6. **判定の線を、 測る前に書いておく**。 途中経過で判断しない。

## 1. p2nnue の骨組み

| 項目 | 内容 |
|---|---|
| 何をするか | やねうら王の `eval/nnue/` を使わない独自の NNUE。 HalfKA_hm → FT 1024 x 2 → fc0 16 (15 + skip) → 30 → 64 → 1、 PSQT、 LayerStack 8 bucket |
| ソース | `eval/p2nnue/p2nnue.cpp` (読み込み・差分更新・推論・USI option)、 `eval/p2nnue/p2nnue_reader.h` (ファイル形式)。 主な関数: `ensure_perspective` (多段の差分更新)、 `refresh_perspective` (Finny 表)、 `transform_pair` / `transform_pair_sum`、 `evaluate_internal`。 エディションは `YANEURAOU_EDITION=YANEURAOU_ENGINE_P2NNUE` |
| 学習器 | `<trainer>/examples/shogi_layerstack.rs`、 入力は `crates/bullet_lib/src/game/inputs/shogi_halfka.rs`。 `--output-format rshogi` がエンジンの読む形式 |
| 切り替え | `-DP2_ARCH_1024 -DP2_L1_MAIN=15` (形)。 ファイルは `EvalDir` + `EvalFile` (既定 `eval` / `nn.bin`) |
| 形式の関門 | net の先頭の arch 文字列で層の形を判定し、 知らない形は読まずに `isready` で止まる。 形式を足すときは arch の token を変え、 古い exe が読めずに止まるようにする (黙って別の値で動かさない) |
| 測り方 | `engcheck.py` 系: エンジンの静的評価 (`eval` コマンド) と python の整数推論を全局面で比べる |
| 教訓 | 重みを `std::vector` で確保して 64 B 境界に乗っておらず、 評価ごとに 256 本の split load を出していた (直して +1.93%)。 `speedups/S07_fc0_alignment.md` |

## 2. threat 入力と LZ (多段 lazy 差分)

| 項目 | 内容 |
|---|---|
| 何をするか | 「どの駒がどの駒に利いているか」の対 (攻め駒・標的・相対位置、 玉を除く) 216,720 次元を、 64 列の threat acc にして FT の [0,32) と [512,544) に足す。 bucket を 4 群 (0 / 1-2 / 3-4 / 5-7) に分けて群ごとに別の表 |
| ソース | `eval/p2nnue/p2nnue_threat.h` (列挙の表)、 `eval/p2nnue/p2nnue.cpp` の `ensure_threat_both` / `add_threat_dual` (全計算)、 `eval/p2nnue/p2nnue_lz.h` (LZ) |
| 学習器 | `crates/bullet_lib/src/game/inputs/shogi_halfka_hm_threat_profile11split.rs` (`inputs.rs` の `#[path]` で `shogi_halfka_hm_threat` として読む)、 除外規則 `shogi_threat_exclusion.rs`。 引数 `--threat --threat-out 64 --threat-bucket-groups 0,1,1,2,2,3,3,3`、 凍結 `--threat-freeze(-hard)` |
| 切り替え | `-DP2_THREAT_PROFILE=0 -DP2_THREAT_OUT=64 -DP2_THREAT_LZ`、 LZ の仕上げ `-DP2_LZ_IDXV2=1 -DP2_LZ_ROWPF=1 -DP2_LZ_D1FAST`。 利きライブラリは使わない (`-DP2_MATE1PLY_NOEFFECT`、 `LONG_EFFECT_LIBRARY` なし) |
| LZ の考え方 | do_move では何もせず、 評価のときに「計算済みの祖先」まで最大 8 手さかのぼり、 途中の手で変わった対だけを足し引きする。 さかのぼれないときは全計算し、 途中の node にも結果を書いて (物質化) 兄弟が使えるようにする |
| 測り方 | 検算 `P2_LZ_VERIFY` (差分の結果と全計算を比べる)、 関門 3 本、 NPS は PGO 同士 128 周で 102.25% [101.69, 102.87]。 税 (threat を剥いだ net 比) は約 22〜30% |
| 教訓 | 税の本体は分岐予測ミスだった (表のキャッシュ常駐は +0.96pt だけ)。 Stockfish 式に do_move で毎手差分を作る方式は −22.7% (評価されない node の差分も作るため)。 物質化が valid な祖先を別の群で上書きして崩れた例がある (群をまたぐときは書かない)。 do_move で FT 行を先読みする pf4 は LZ の前は得、 後は有害 (98.77%) になった。 BACKMAT (全計算した子から親を逆向きに作る) は線に届かず不採用。 詳細は `speedups/S11_threat_lz.md`、 `S12`、 `S13`、 `S16`、 `S25`、 `REJECTED.md` |
| 足すなら | 列の幅 K を増やすと篩は上がるが accumulator の費用が増える (`speedups/S03_threat_out_width.md`)。 全幅 (1024 列) を LZ 並みの速さで回せるかは未踏 (`FUTURE_WORK.md` 3 章) |

## 3. KH (浮き駒の追加入力)

| 項目 | 内容 |
|---|---|
| 何をするか | どの駒にも守られていない駒 (浮き駒) の 78 次元 (h 部) を 64 列にして、 threat と同じ FT の位置に足す。 評価のときに threat acc + KH 行を一時 buffer で作るので、 StateInfo の acc は書かない (LZ と独立) |
| ソース | `eval/p2nnue/p2nnue_kh.h` (読み込みと特徴)、 `eval/p2nnue/p2nnue_kh_fast.h` (推論核)、 `eval/p2nnue/p2nnue.cpp` の評価入口 |
| 学習器 | `crates/bullet_lib/src/game/inputs/shogi_kh.rs`、 検算 `examples/kh_feat_dump.rs`。 引数 `--kh --kh-feats h --kh-freeze-rest --kh-unfreeze-fc0 --kh-lr-mult 64 --kh-dump <file>` |
| 切り替え | `-DP2_KH`、 option `KHFile` (既定 `kh.bin`、 `kh.bin.json` で形を検査)。 ファイル形式は f32、 列優先 (`values[feature * out + row]`) |
| 測り方 | 篩の KH on − off (`screen_kh_onoff.py`): bucket 0〜2 +0.0055 ±0.0022 / 3〜7 +0.0048 ±0.0009。 NPS の税 −0.34% [−0.93, +0.26] (64 周) |
| 教訓 | 税の見積もり (−4〜−9%) は大きく外れた (実測 ±0)。 loss には効きが出ない。 別の物差し (互角帯の篩) では差が出なかった = 物差しで結論が変わる。 配布の重みは 1.25 倍 (`kh_scale.py`) |
| 足すなら | 本体の FT を凍結して「新しい入力 + fc0」だけを学習する形 (`--kh-freeze-rest --kh-unfreeze-fc0`) は、 FT を動かさずに入力を足す型として使える (FT を動かすと弱くなった、 5 章) |

## 4. OPN (序盤専用 net) と継ぎ目

| 項目 | 内容 |
|---|---|
| 何をするか | 序盤を幅 2048 の別の net で評価する。 入力は HalfKA_hm K16 (玉区分 16 = 自玉 7〜9 段 x 5 筋 + その他、 26,064 次元 + factoriser 1,629)、 threat なし・PSQT なし |
| ソース | `eval/p2nnue/p2nnue_opn.h`、 `eval/p2nnue/p2nnue.cpp` の `opn_eval_only` ほか |
| 学習器 | `--opn-king16 --factorise --l0 2048 --l2 32 --bucket-filter 0-2 --loss-scale 300`、 検算 `examples/opn_k16_check.rs`。 初期重みは本体の FT を写す (`f1_transplant2048.py`) |
| 切り替え | `-DP2_OPN -DP2_OPN_F2 -DP2_OPN_R2`。 option `OPNFile` (既定 `opn.bin`)、 継ぎ目 `OPN_ProgHi` (v9 既定 -1946)、 `OPN_ProgWidth` (0 = 混ぜない)、 `OPN_LambdaMax` (256 = OPN だけ)、 目盛り `OPN_Scale0..2`、 片寄り `OPN_Offset` |
| 継ぎ目の仕組み | 進行度から重み λ (0〜256) を作り、 OPN と本体を λ で混ぜる。 幅 0 なら λ は 256 か 0 だけで、 λ=256 の局面では本体も threat も計算しない (速い)。 v9 では OPN が受け持つのは進行度 bucket 0 だけ |
| 測り方 | 篩 bucket 0-1 (`b01_screen.py` / `pair_screen.py`): OPN 0.3016〜0.3019 対 本体 0.2555。 速さは 1 手 1 秒の到達ノード (幅 0 で対 v4 110.0% [108.0, 112.0]) |
| 教訓 | 一から学習した幅 2048 は写しから始めた版に大きく負けた (2 SB で 0.2050 対 0.2756)。 bucket 2 まで受け持たせると頭打ち、 bucket 2 だけ追加学習すると bucket 0 を壊す。 bucket 0〜1 を受け持たせる設定と、 FT も動かす学習のし直しは対局で負けた。 素の実装は本体の約 80% の NPS で、 F2・R2 の 2 段で戻した (`speedups/S19_opn_f2.md`、 `S20`、 `S21`) |
| 落とし穴 | OPN が先に値を返す局面 (bucket 0) では本体側の補正 (bucket 補正の B0 など) が効かない。 SPSA で B0 が約 20cp 動いたのは信号の無い鍵の動き = 雑音の目安 (6 章) |

## 5. 量子化 (L1Q128・FTI32 系・FC1PREC)

| 項目 | 内容 |
|---|---|
| 何をするか | 整数推論と f32 の差を減らす。 L1Q128 = fc0 の本体 15 行を 1/128 刻みに。 FTI32 系 = FT を 8 倍細かい刻み (`-DP2_FT_KS=3`)、 transform の出力を i16 にして fc0 を i16 入力に (`-DP2_FT_OUT16=5`、 疎な版 `-DP2_FT_SPARSE16`)。 FC1PREC = fc0・fc1 の活性を 0..508、 fc1 の重みを 1/128 刻み (`-DP2_FC1_PREC`) |
| ソース | `eval/p2nnue/p2nnue.cpp` の `P2_FT_KS` / `P2_FT_OUT16` / `P2_FT_SPARSE16` / `P2_FT_I32` / `P2_FC1_PREC` の箇所 (`ft_add_thr`、 `affine16_t`、 `affine16_sparse_t`)、 `eval/p2nnue/p2nnue_opn.h` |
| net の作り方 | 学習し直さない。 学習器の f32 の重みから python で書き直す: `mk_l1q128.py` → `fti32_write.py` → `fc1prec_write.py` (`TRAINING_FROM_SCRATCH.md` 6 章)。 学習時の QAT (`--qat`) は別の話で、 OPN の最後の 3 SB にだけ使った |
| 形式の関門 | arch に `AffineTransformL1Q128[`、 `I32K8:` (版 0x7AF32F21)、 `AffineTransformF1P128[` (版 0x7AF32F22) を入れる。 古い exe は読めずに止まり、 新しい exe は古い net を読み込み時に変換する |
| 測り方 | python の模擬で、 差 10〜30cp の帯で同じ親の子の最善手が入れ替わる率 (従来 0.205 → FT 側 0.057 → 後段も 0.019)。 エンジンとの完全一致 (本体・OPN 1,000 局面)、 探索中に i16 の範囲を超えた値 0 件。 NPS は FT 側 −6.53% [−7.21, −5.90]、 FC1PREC は影響なし |
| 教訓 | **誤差を減らした net はそのままでは対局で弱くなった** (20 局で 27.5%、 線 40% に届かず)。 古い量子化の「手番側を悲観的に見る片寄り」が楽観のブレーキになっていた、 と読んで 6 章の bucket 補正で足し戻した。 i16 の累積器が溢れないかは探索中の実測 (和の範囲) で確かめてから i32 をやめた。 FT を i8 にする案は AVX2 で負けた (96.06%) |

## 6. LayerStack の bucket 補正 (先後別の表と切り替え)

| 項目 | 内容 |
|---|---|
| 何をするか | 本体の評価 (OPN と混ぜる前) に、 LayerStack の bucket ごとの定数 [cp、 手番側の視点] を足す。 我々が先手番の探索と後手番の探索で別の表を使い、 優勢なら後手でも先手の表に切り替える |
| ソース | `eval/p2nnue/p2nnue.cpp` の `P2_BUCKET_OFFSET` の箇所 (`select_bucket_off`、 `p2_search_begin`)。 表は探索の開始時に 1 回だけ選び、 評価の熱い経路は選んだ表を読むだけ |
| 切り替え | `-DP2_BUCKET_OFFSET`。 option `P2_BUCKET_OFFSET_B0..B7` (後手番の表)、 `P2_BUCKET_OFFSET_SENTE_B0..B7` (先手番の表。 未設定の要素は後手の表の値)、 `P2_SENTE_TABLE_SWITCH_CP` (v9 既定 500 = 直前の手の root の評価値が表示 cp +500 以上なら後手でも先手の表)、 `P2_SENTE_TABLE_SWITCH_LOG` (判定を `info string` で出す)。 全部 0 / 未設定なら従来と bit 一致 |
| v9 の値 | 後手の表 = SPSA24 の 29 周目 (−39 / −14 / −27 / +19 / +4 / −12 / +10 / +29)、 先手の表 = 古い量子化の片寄りを写した値 (−18 / −3 / −32 / −5 / +16 / −15 / −4 / +15) |
| 測り方 | 対 v4 の同ノード対局 (40 局ずつ) で腕を並べた。 片寄りを打ち消す向き (0 倍) は崩れ、 古い片寄りを足し戻した腕が最良だったが、 差は 2SE の内 |
| 落とし穴 | 評価が「局面と root の手番」の関数になる。 置換表は手番で分けていないので、 局の境目で `isready` を送らない器では別の手番で作った評価を再利用しうる (`usinewgame` はこのエンジンで何もしない)。 bucket 0 は OPN が先に返すので B0 は効かない。 その効かない B0 が SPSA で約 20cp 動いた = SPSA の値の 20cp 程度の動きは雑音でも出る |
| 足すなら | 手番・優劣・進行度などで評価に定数を足す機構は、 篩ではほとんど見えず対局でだけ効く。 篩の差で採否を決めない |

## 7. SPSA で調整した探索 option と既定値の焼き込み

| 項目 | 内容 |
|---|---|
| 何をするか | やねうら王の探索の定数を `SPSA::XXX` の変数に置き換え、 USI option (`SPSA_*` 約 190 個、 時間管理 `SPSA_TM_*`) に出す。 SPSA で調整した値を、 公開版では既定値に焼き込む |
| ソース | `spsa_params.h` (宣言と範囲)、 `engine/yaneuraou-engine/yaneuraou-search.cpp` (置き換え)、 `usioption.cpp` / `usioption.h` の `OptionsMap::set_release_default` (USI の既定値と現在値の両方を書き換える)、 呼ぶ場所は `eval/p2nnue/p2nnue.cpp` の P2NNUE の `add_options` の最後 (48 個の表) |
| 調整の器 | 道具のリポジトリの `spsa` bin (ラム将棋の bin を拡張) |
| v9 の値 | 自己対局の SPSA (SPSA24) の 29 周目。 調整した 47 個の option と `SPSA_LMR_TABLE_COEFF_OPENING=2763` |
| 確認 | ファイルのパスの option だけで起動した探索が、 以前の 206 個の option を渡した設定と 40 局面 x 100 万ノードで同一 |
| 教訓 | SPSA の器に θ+ / θ− を二重に入れ替える不具合があり (11-19 に修正)、 それ以前の SPSA の結果は見直しの対象になった。 信号の無い鍵 (B0) の動きを雑音の物差しにする。 option で渡す値と既定値が食い違っていた鍵がある (`SPSA_LMR_TABLE_COEFF_OPENING` は表示 2762 / 実際 2763) ので、 焼き込みの後は「option なしで同じ探索になるか」を関門にする |
| 足すなら | 新しい探索項は `spsa_params.h` に変数を足し、 既定値 = 本家の値で bit 一致を確かめてから option で動かす |

## 8. 探索の独自の変更

| 機構 | 何をするか | ソース | 切り替え | 測り方・教訓 |
|---|---|---|---|---|
| INFREP | 優等・劣等局面の判定を root より前 (対局の履歴) にも広げる。 千日手は従来どおり root 以降だけ。 劣等は ply >= 2 のときだけ履歴をまたぐ (root の子で「相手が劣等」を返すと root がその手を無条件に選ぶため) | `position.cpp` の `is_repetition(ply, infrep_history)`、 呼び出しは `yaneuraou-search.cpp` | option `P2InferiorRepHistory` (既定 true) | off で従来と bit 一致を関門で確認。 NPS 99.56% (30 周、 打ち切り)。 単独の強さの判定はない |
| HIST_CONFIDENCE_SCALE | LMR の statScore の 3 成分が同じ符号のとき効きを強める (150 で 1.32 倍) | `yaneuraou-search.cpp`、 `spsa_params.h` | option `HIST_CONFIDENCE_SCALE` (既定 150、 0 で無効) | 長く設定ファイルに書かずに暗黙で効いていた。 設定の監査で見つけた。 0 にしても probe で改善しなかった |
| BookExitTimeFix | 定跡などで探索を飛ばした手のあと、 `bestPreviousScore` を「前回評価なし」にする (本家は −∞ を残し、 定跡を抜けた最初の手の時間が約 4 割減る) | `yaneuraou-search.cpp` | option `BookExitTimeFix` (既定 true) | 定跡なしの探索で bit 一致。 定跡ありの対局での効果は測っていない |
| SRCH_BUNDLE_A | 探索側の結果を変えない高速化の束 (idiv の定数化、 quiet 採点の速い経路など) | `srch_bundle_a.h`、 `yaneuraou-search.cpp` | `-DP2_SRCH_BUNDLE_A` | 検算 `BUNDLEA_VERIFY` 14,779,636 回で不一致 0、 101.53%。 `speedups/S14_search_bundle_a.md` |
| 時間管理 | 本家の式の係数を `SPSA_TM_*` に。 1 手の上限 (`SPSA_TM_MAX_BANK_PCT`)、 残りが少ないときの床 (`SPSA_TM_LOW_BANK_MS`) | `yaneuraou-search.cpp`、 `timeman` まわり | option | 長手数で先手が時間切れになる床の問題があった。 `NetworkDelay2` が短い秒読みを食う (`OVERVIEW.md` 2 章) |
| 千日手の評価値 | `DrawValueBlack` / `DrawValueWhite` | `yaneuraou-search.cpp` | option (v9 既定 -2 / -2) | 電竜戦の規定 (引き分け 0.4 / 0.6) に合わせた -53 / +41 は、 floodgate では後手が千日手を受け入れやすすぎた。 大会の規定ごとに値を変える |
| 楽観の非対称 | 静的評価に自分と相手で別の定数を足す (Stockfish の optimism に近い) | `yaneuraou-search.cpp` / `.h` の `P2_OPTIMISM` | `-DP2_OPTIMISM`、 option `P2_OPT_US_CP` / `P2_OPT_THEM_CP` / `P2_OPT_DYN_*` (既定 0 = 無効) | 公開版にはビルドで入っているが既定 0 で使っていない |

試して採らなかった探索の腕 (地平線問題・評価の楽観を補う OPTSRCH / HORIZON、 LMR の係数 2490 など) は `FUTURE_WORK.md` 2 章。

## 9. 並列 (Lazy SMP) の実験機構

| option | 何をするか | 結果 |
|---|---|---|
| `SPSA_ASPIRATION_THREAD_STEP` | aspiration の窓を thread ごとにずらす | 採らず |
| `SPSA_SHARED_CONT_HIST` | continuation history を thread 間で共有 (構築時に確保。 遅延確保は data race) | 採らず (メモリは 161 → 54 MB) |
| `SPSA_SMP_DEFER_MIN_DEPTH` | ABDADA 風に、 他 thread が読んでいる子を後回し | 閉 (発火率 0.025%) |
| `SPSA_ROOT_YBWC` | 根の 2 手目以降を helper に配る | 閉 |
| `SPSA_YBWC_MIN_DEPTH` (M4) | 内部節で 2 手目以降を helper に委ね、 結果は置換表だけで受け取る (master は全手を自分で読むので、 helper が遅れても結果は同じ) | 既定 off。 時間切れ停止時に helper の枠を返さない不具合があり、 直した版でも対局で改善なし |
| `SPSA_NMP_THREAD_PARITY` | null move の R を thread の偶奇でずらす | 閉 |

- ソース: `smp_deleg.h` (M4)、 `smp_root_ybwc.h`、 `smp_busy.h`、 `yaneuraou-search.cpp`。 既定値なら Threads 1 でも Threads > 1 でも本家の Lazy SMP と同じ動作。
- 測り方: time-to-depth (同じ深さに届くまでの時間) と、 Threads を変えた自己対局。 🚨比べるときは相手のスレッド数も揃える (片方だけ変えて「SMP が原因」と誤って結論しかけた)。
- 教訓: 「thread を増やしても木の 92〜95% を重複して読む」という見立ては 1 本あたりのノード比から出したもので、 局面のキーで数えた重複率は 21〜25% だった。 前提を測り直してから機構を選ぶ。 詳細は `SPEEDUPS_STORY.md` 9 章、 `speedups/P06_m4_tt_delegation.md`。

## 10. 速さの仕事 (一覧へのリンク)

| 区分 | 仕様 |
|---|---|
| 測り方 (全仕様の前提) | `speedups/M00_measurement.md` |
| ビルド (`-march`、 PGO、 clang) | `speedups/S04_pgo_procedure.md`、 `S05_compiler_flags.md` |
| threat (出力幅、 利きライブラリを外す、 LZ と仕上げ) | `S01`、 `S03`、 `S11`〜`S16`、 `S25` |
| forward・配置・固定費 | `S07`、 `S08`、 `F01`〜`F03` |
| 2 本目の net (KH・OPN・L1Q128) | `S18`〜`S22` |
| 本家由来の土台 (Finny、 多段の追いつき、 lazy evaluate) | `S23_finny_chain_inherited.md` |
| 並列 | `P06_m4_tt_delegation.md` |
| 効かなかったもの | `REJECTED.md` |

## 11. 必要な python スクリプト

機構の検証に使う器は、 https://github.com/pekoshi/rshogi の `p2shogi` ブランチの `tools/python/` に置きます (器どうしが import し合うので同じフォルダに)。

| 用途 | スクリプト |
|---|---|
| net の書き出し | `mk_l1q128.py`、 `fti32_write.py`、 `fc1prec_write.py`、 `k256_merge.py` (部品)、 `kh_scale.py`、 `f1_transplant2048.py` (+ `f1_transplant.py`)、 `calib_l1q128.py` |
| 整数推論の照合 | `engcheck.py`、 `fti32_verify.py`、 `fc1p_verify.py`、 `fc1p_verify_opn.py` (部品 `bullet_inference_v19.py`、 `opn_static2.py`、 `opn_static.py`、 `eng.py`、 `v4c.py`、 `p0.py` (+ `make_sets.py`)、 `m2_games_static.py`) |
| 探索の関門 | `gate.py` |
| NPS の A/B | `nps_ab.py`、 `nps_cputime_ab.py`、 `nps_ab_prod.py` |
| PGO の採取 | `pgo_collect_l1q128.py`、 `collect.py` (v9) |
| 篩 | `screen_future.py`、 `b01_screen.py`、 `pair_screen.py`、 `screen128.py`、 `screen_kh_onoff.py` |
| 学習データの前処理 | `ponko_novel.py`、 `calib_lut.py` |
