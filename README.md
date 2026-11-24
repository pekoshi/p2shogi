# p2shogi v9

やねうら王をベースにした将棋の思考エンジン (USI) です。評価関数は独自の NNUE (p2nnue) を使います。

USI では `id name p2shogi v9` / `id author pekoshi` と名乗ります。

## 特徴

- 評価関数 p2nnue: HalfKA_hm + 利き (threat) 入力 + 玉周りの補助入力 (KH)、LayerStack 8 本、L1 1024。
- 序盤専用の評価関数 OPN (進行度 bucket 0 だけを担当)。
- 探索パラメータは SPSA で調整し、エンジンの既定値に焼き込み済みです。対局時に探索の option を指定する必要はありません。
- 千日手の評価値 (`DrawValueBlack` / `DrawValueWhite`) の既定は -2 / -2 (中立) です。

## 棋力の目安

- floodgate (`p2shogi_v9_nobook_nnue_i7700_4T`、定跡なし、Core i7-7700 の 4 スレッド): 2026-11-23 時点でレート 4248。
- やねうら王系の強豪エンジンとの同ノード比較では、まだ互角に届いていません。

## 使い方

[Releases](../../releases) の zip を展開すると、次の構成になります。

```
p2shogi_v9/
  p2shogi_v9.exe
  eval/
    nn.bin        本体の評価関数
    opn.bin       序盤用の評価関数
    kh.bin        玉周りの補助入力の表
    progress.bin  進行度 (LayerStack の bucket 選択に使う)
```

将棋所や ShogiGUI などに `p2shogi_v9.exe` を登録してください。`eval` フォルダが exe と同じ場所にあれば、追加の設定は不要です。

- `Threads` の既定は 4、定跡は既定で使いません。
- 動作には AVX2 対応の CPU が必要です。

## ビルド

Windows + llvm-mingw (clang 23) で確認しています。

```
cd source
make -j4 normal COMPILER="clang++ -static" YANEURAOU_EDITION=YANEURAOU_ENGINE_P2NNUE \
  EXTRA_CPPFLAGS="(下の define)"
```

公開版の exe は次の define で作り、100 万ノードの実戦局面 60 個で PGO をかけています。

```
-DP2_ARCH_1024 -DP2_L1_MAIN=15 -DP2_THREAT_PROFILE=0 -DP2_MATE1PLY_NOEFFECT -DP2_THREAT_OUT=64
-DUSE_LAZY_EVALUATE -DP2_THREAT_LZ -DP2_OPN -DP2_OPN_F2 -DP2_KH -DP2_OPN_R2 -DP2_LZ_IDXV2=1
-DP2_LZ_ROWPF=1 -DP2_SRCH_BUNDLE_A -DP2_LZ_D1FAST -DP2_TR_STORE32 -DP2_ACC_PIECEWISE -DP2_FC0_PF
-DP2_FT_KS=3 -DP2_FT_OUT16=5 -DP2_FT_SPARSE16 -DP2_FC1_PREC -DP2_BUCKET_OFFSET -DP2_OPTIMISM
```

## 関連リポジトリ

- 学習器: [pekoshi/bullet-shogi](https://github.com/pekoshi/bullet-shogi) の `p2shogi` ブランチ (nodchip 氏の bullet-shogi の fork)
- 学習器が使う `rshogi-core`: [pekoshi/rshogi](https://github.com/pekoshi/rshogi) (ラム将棋 SH11235/rshogi の fork)

## ライセンス

GPLv3 です (`LICENSE`)。やねうら王 (GPLv3) の派生物です。

## 謝辞

- やねうら王 (yaneurao 氏) と、その元になった Stockfish
- 進行度ファイル `progress.bin` は NAGISA v3.1 に同梱のものです
- 学習データ: DL 水匠の評価で付け直した公開データ (dlsuisho-ponkotsu-distilled)
- 学習器: bullet-shogi (nodchip 氏)
- ラム将棋 (SH11235 氏)

元のやねうら王の README は `README_YaneuraOu.md` にあります。
