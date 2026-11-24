# S04 clang の instr PGO の手順と採取の点検

- 状態: 採用 / bit 一致: 一致 (PGO は最適化の配置だけを変える。 確認は必須) / requires: M00
- 人間向け: `../SPEEDUPS_STORY.md` 3.2 節

## 1. 要約

clang の計装 PGO (`-fprofile-instr-generate` → 採取 → `llvm-profdata merge` → `-fprofile-instr-use`) で +7〜9%。 ただし採取の条件を誤ると逆に −7% になる。 採取は**配布する net 一式と本番の option** で行い、 熱い経路を変えたら採り直す。

## 2. 前提

- clang (我々は llvm-mingw 20260908、 clang 23)。 gcc の `-fprofile-generate` でも考え方は同じだが、 やねうら王の Makefile の pgo ターゲットは gcc 用 (`-lgcov`) で clang には使えなかった (07-14 時点)。
- 評価関数のファイル (net) を実行時に読むエンジン。

## 3. 不変条件

- PGO の有無で探索の結果は変わらない (浮動小数の扱いを変えるフラグを足さない)。 確認として PGO 版と非 PGO 版で関門 (M00) を通す。

## 4. 手順

```bash
# 1. 計装ビルド (本番と同じ DEFINES・フラグ + 計装)
make clean            # 🚨 clean が本当に全オブジェクトを消すか確かめる (下の 7)
make ... EXTRA_CPPFLAGS="-fprofile-instr-generate" LDFLAGS+="-fprofile-instr-generate"
# 2. 採取: 本番の net 一式・本番の option で、 中盤・終盤を含む局面を go nodes で回す
LLVM_PROFILE_FILE=prof_%p.profraw ./engine_instr.exe < collect_commands.txt
#    採取のログに「net を読んだ」行 (我々は "P2NNUE loaded: <path>"、 "KH: loaded"、 "P2OPN loaded:") があることを確かめる
# 3. 統合
llvm-profdata merge -o default.profdata prof_*.profraw
# 4. 使用ビルド
make clean && make ... EXTRA_CPPFLAGS="-fprofile-instr-use=default.profdata"
```

採取の局面: 我々は本番の option (OPN・KH を含む) で、 中盤の局面を中心に数十局面 x 数十万ノード。 2 本の net を切り替える構成では、 **両方の net が使われる局面**を入れる (序盤と中盤以降)。 器は `<tools>/tools/python/build/pgo_collect_*.py`。

## 5. bit 一致の検証

PGO 版と、 同じソースの非 PGO 版で関門 (M00)。 我々の記録: bench のノード数一致 (07-14)、 89 局面一致 (09-30)。

## 6. NPS の測り方と期待値

M00。 PGO の効果は「非 PGO 対 PGO」で測る。 我々の実測: +7〜9% (07-14、 12 対の交互、 bench nodes 一致)。 採取の net を実在のものに直して −7.0% → +7.0% (09-14、 8 周すべてで上回る)。 熱い経路を変えた後の採り直しだけで +3.08% [+0.65, +5.48] (20 周、 09-30)。 `make clean` の消し残しを直した採り直しで +11.0% (09-01)。

高速化の改造を PGO 下で評価するときは、 **A も B もそれぞれのソースで採り直した PGO** で比べる。

## 7. はまりどころ

- **採取で net が読めていない**と、 評価関数の熱い経路が profile に入らず PGO 版が遅くなる (−7%)。 存在しないパスを渡しても、 エンジンが黙って既定の動作で走ることがある。 採取のログを必ず見る。
- **`make clean` が一部のオブジェクトを消し残す**と、 計装ビルドでもその TU が計装されず profile に入らない (我々は評価関数の TU で起きた)。 clean の後に .o が残っていないかを確かめる。
- **古い profile を使い回さない**。 ホットパスを変えたら採り直す。
- PGO は非 PGO の改造の得を半分前後先取りする。 非 PGO の得をそのまま期待しない。

## 8. 根拠

実験記録 (設計書・事前登録・判定ログ) は非公開です。 関連: `../UPSTREAM_PROPOSALS.md` P7。
