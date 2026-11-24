# P06 M4: 置換表を介した仕事の委譲 (Lazy SMP の上の内部節の YBWC)

- 状態: **不採用** (既定 off のまま公開。 Threads 4 の正式判定はしていない) / 既定 off (`SPSA_YBWC_MIN_DEPTH=0`) / bit 一致: 既定値と Threads=1 では一致 / requires: M00
- 人間向け: `../SPEEDUPS_STORY.md` 9 章

## 1. 要約

Lazy SMP では各 thread がほぼ同じ木を時間差で読み、 thread を増やしても time-to-depth が縮まない (3 thread で 0.98 倍)。 M4 は、 深い節で master が 2 手目以降を helper に「提示」し、 helper はその手を読んで結果を**置換表に書くだけ**にする。 master は全手を自分で読むので、 helper の手が済んでいれば置換表に当たってほぼ無料になる。 動作確認 (判定ではない、 8 局面 depth 16) で 1 thread 比 time 0.975 倍 (Lazy SMP) → 0.710 倍 (min_depth 8)。

## 2. 前提

- Lazy SMP (thread が置換表を共有して独立に探索) のエンジン。
- master の手のループに「後回し」の仕組み (後回しの手を末尾で、 元の moveCount のまま必ず読む) がある、 または足せる。 我々は ABDADA 風の後回し (M1、 既定 off) の機構を流用した。
- helper が root からの手順を再生して、 提示された節の局面まで降りられる。

| 一般の言葉 | 本実装 |
|---|---|
| 委譲の枠 | `source/smp_deleg.h` (16 枠) |
| 提示の下限の深さ | `SPSA_YBWC_MIN_DEPTH` (0 = 無効) |
| 段の切り分け | `P2_YBWC_STAGE` (1 = 再生だけ、 2 = 再生 + 探索) |
| 統計 | stderr の `DELEG_INFO` (offer / claim / finish / full / idle) |

## 3. 不変条件

- **正しさが helper に依存しない**: master は必ず全手を自分で読む。 helper が遅れても、 落ちても、 読み違えても、 結果は helper が居ないときと同じ (遅いだけ)。 helper は値を返さず、 置換表に書くだけ。
- Threads=1 では何もしない (`deleg = (MIN_DEPTH > 0 && threads > 1) ? … : nullptr`)。
- M1 (ABDADA 風の後回し) と排他。

## 4. 変更の最小単位

1. **提示**: 深さ >= min_depth の節で、 moveCount が 2〜8 の手について、 newDepth と窓が確定した do_move の直前に (root からの手順・窓 (提示時点の alpha) ・深さ) を枠に置き、 その手を後回しにして次の手へ進む。 枠が埋まっていれば提示しない。
2. **helper**: 空いた helper は枠を取り (claim)、 root から手順を再生して節に降り、 null window の NonPV 探索でその手を読む。 返り値は捨てる (結果は置換表に残る)。 終わったら finish。
3. **master**: 後回しの手を末尾で普通に読む (置換表に当たればほぼ無料)。
4. 関門 G1: Threads=1 で md=0 と md=8 が nodes・bestmove・score で全一致し、 統計が 1 行も出ない。 関門 G2: Threads=4 / md=8 で claim > 0 かつ finish > 0 (配線切れでないこと)。

## 5. bit 一致の検証

既定値 (md=0) では Lazy SMP と同じ。 Threads=1 では md に依らず一致 (G1)。 Threads>1 の探索はもともと非決定的なので、 一致ではなく time-to-depth で評価する。

## 6. 測り方と期待値

```
局面: 中盤局面集から無作為に 300 (先頭から取らない)
go depth 18、 2 反復、 run ごとに isready
腕: T1 (Threads 1) / L4 (Threads 4, md 0) / M4_6, M4_8, M4_10 (Threads 4) / Threads 3 の補助腕
物差し: 局面ごとに log(time) を反復で平均し、 M4 腕 − L4 の対応ペアの差の平均と 95% 区間
採用候補: time の区間の上端 < 1.00 倍。 3 本から選ぶので Bonferroni (z=2.394) でも上端 < 1.00
```

動作確認 (n=8、 depth 16、 判定ではない): 1 thread 比 nodes / time = Lazy SMP 3.230 / 0.975 倍、 md8 1.453 / **0.710 倍**、 md12 1.511 / 1.042、 md16 1.058 / 1.060。 Threads 4 の正式判定はしていません。 11-21 の点検で、 時間切れで探索を止めたときに helper に渡した枠を返さず、 以後の手で実質 Threads 1 になる不具合が見つかりました。 直した版でも 1 手 1 秒の対局で改善が見えなかったので、 既定 off のまま公開しています。

## 7. はまりどころ (実装中に踏んだ 4 件 + 速度の 1 件)

1. 完了通知の書き忘れで helper が永久に待つ (待ち時間が探索時間のほぼ全部になる)。
2. 探索 stack (`ss`) から手順を取ろうとすると、 `ss = stack + 7` なので `ss - ply` は ply <= 7 でしか有効でなく、 配列の外を読む。 手順は root からの手の列として別に渡す。
3. helper は aspiration のループを通らないので `rootDelta` が 0 で、 `reduction()` の除算で整数ゼロ除算 (0xC0000094)。 helper 側で設定する。
4. `search<PV>` は `ss->pv` を触るのでアクセス違反 (0xC0000005)。 helper は null window の NonPV だけで読む。
5. 最初の実装は Lazy SMP より 1.7 倍遅かった。 helper が空の枠を acquire で舐め回し (1 局面 150 万回)、 master が書く cache line を叩き続けていた。 空振りの判定を 1 つの relaxed な読みで弾くようにして逆転した。

終了コードが原因域を教えた (0xC0000094 = ゼロ除算、 0xC0000005 = アクセス違反)。 並行機構は段を切り分けるスイッチ (再生だけ / 再生 + 探索) を残しておく。

## 8. 根拠

実験記録 (設計書・事前登録・判定ログ) は非公開です。
