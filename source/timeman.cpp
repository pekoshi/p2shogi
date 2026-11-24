#include "config.h"

#if defined(USE_TIME_MANAGEMENT)

#include "misc.h"
#include "search.h"
#include "thread.h"
#include "usi.h"
#include "timeman.h"

namespace YaneuraOu {

namespace {

// これぐらい自分が指すと終局すると考えて計画を練る。
// 近年、将棋ソフトは終局までの平均手数が伸びているので160に設定しておく。
const int MoveHorizon = 160;

// 思考時間のrtimeが指定されたときに用いる乱数
PRNG prng;

} // namespace

// 起動時に呼び出す。
// このclassが使用するengine optionを追加する。
void TimeManagement::add_options(OptionsMap& options) {

		// 指し手がGUIに届くまでの時間。
#if defined(YANEURAOU_ENGINE_DEEP)
    // GPUからの結果を待っている時間も込みなのでdefault値を少し上げておく。
    int time_margin = 400;
#else
    int time_margin = 120;
#endif

    // ネットワークの平均遅延時間[ms]
    // この時間だけ早めに指せばだいたい間に合う。
    // 切れ負けの瞬間は、NetworkDelayのほうなので大丈夫。
    options.add("NetworkDelay", Option(time_margin, 0, 10000));

    // ネットワークの最大遅延時間[ms]
    // 切れ負けの瞬間だけはこの時間だけ早めに指す。
    // 1.2秒ほど早く指さないとfloodgateで切れ負けしかねない。
    options.add("NetworkDelay2", Option(time_margin + 1000, 0, 10000));

    // 最小思考時間[ms]
    options.add("MinimumThinkingTime", Option(2000, 1, 100000));

    // 切れ負けのときの思考時間を調整する。序盤重視率。百分率になっている。
    // 例えば200を指定すると本来の最適時間の200%(2倍)思考するようになる。
    // 対人のときに短めに設定して強制的に早指しにすることが出来る。
    options.add("SlowMover", Option(100, 1, 1000));

    // p2shogi: Fischer(増分あり・秒読みなし)における maximumTime の係数(max_ratio)を
    // 百分率で与える。default 150 = 1.5(現状の floodgate 切れ負け対策値)を維持する。
    // 値を上げると1手あたりの上限思考時間が伸び、中盤に時間を厚く投下できるが、
    // 主時間を食い潰す per-move spike が増えて切れ負けリスクが上がる点に注意。
    options.add("MaxRatioX100", Option(500, 10, 2000));
    // p2shogi 2026-09-14: Fischer の銀行保護。 ★既定値で従来と bit-exact (ガードで経路ごとスキップ)。
    //   floodgate 24 局で 7 敗すべてが「銀行を床まで枯渇させた局」と一致したため導入。
    options.add("SPSA_TM_MAX_BANK_PCT",    Option(45, 10, 100));  // 11-16 既定を対局時の値へ (旧 100)。 // maximum <= inc + 銀行*pct/100。 100 で無効
    options.add("SPSA_TM_RESERVE_INC_X10", Option(20,  0,  80));  // 11-16 既定を対局時の値へ (旧 0)。 // 計画上の温存 = inc*X/10。 0 で無効
    // p2shogi 2026-11-01: 残り時間が少ないときの 1 手の上限を**加算の内側**に落とす。
    //   0 で無効 = 従来と bit-exact (ガードで経路ごとスキップ)。
    //   長手数の将棋 (入玉戦) で先手が時間切れ負けするのを防ぐ。 詳細は init_() の該当ブロック。
    options.add("SPSA_TM_LOW_BANK_MS",     Option(0,   0,  600000));

	// 持ち時間、各秒のギリギリまで使うか。
    options.add("RoundUpToFullSecond", true);

}

void TimeManagement::init(const Search::LimitsType& limits,
                          Color               us,
                          int                 ply,
                          const OptionsMap&   options
#if STOCKFISH
                          ,
                          double& originalTimeAdjust
// 💡 やねうら王では使わないことにする。
#else
                          ,
                          int max_moves_to_draw
#endif

) {

    // 📝 探索開始時刻をコピーしておく。
    //     以降、elapsed_time()は、ここからの経過時間を返す。
    startTime = limits.startTime;

	// TODO あとで
#if 0
    // reinit()が呼び出された時のために呼び出し条件を保存しておく。
    lastcall_Limits = &limits;
    lastcall_Us     = us;
    lastcall_Ply    = ply;
    lastcall_Opt    = const_cast<OptionsMap*>(&options);
#endif

    init_(limits, us, ply, options, max_moves_to_draw);
}

// 今回の思考時間を計算して、optimum(),maximum()が値をきちんと返せるようにする。
// これは探索の開始時に呼び出されて、今回の指し手のための思考時間を計算する。
// limitsで指定された条件に基いてうまく計算する。
// ply : ここまでの手数。平手の初期局面なら1。(0ではない)
void TimeManagement::init_(const Search::LimitsType& limits,
                           Color               us,
                           int                 ply,
                           const OptionsMap&   options, int max_moves_to_draw) {

#if STOCKFISH
	TimePoint npmsec = TimePoint(options["nodestime"]);
	// nodes as timeモード。やねうら王では用いない。

    // If we have no time, we don't need to fully initialize TM.
    // startTime is used by movetime and useNodesTime is used in elapsed calls.
    startTime    = limits.startTime;

	useNodesTime = npmsec != 0;

    if (limits.time[us] == 0)
        return;

	TimePoint npmsec = Options["nodestime"];

	// npmsecがUSI optionで指定されていれば、時間の代わりに、ここで指定されたnode数をベースに思考を行なう。
	// nodes per millisecondの意味。
	// nodes as timeモードで対局しなければならないなら、時間をノード数に変換して、
	// 持ち時間制御の計算式では、この結果の値を用いる。
	if (npmsec)
	{
		// ゲーム開始時に1回だけ
		if (!availableNodes)
			availableNodes = npmsec * limits.time[us];
		
		// ミリ秒をnode数に変換する
		limits.time[us] = TimePoint(availableNodes);
		for (auto c : COLOR)
		{
			limits.inc[c] *= npmsec;
			limits.byoyomi[c] *= npmsec;
		}
		limits.rtime *= npmsec;
		limits.npmsec = npmsec;

		// NetworkDelay , MinimumThinkingTimeなどもすべてnpmsecを掛け算しないといけないな…。
		// 1000で繰り上げる必要もあるしなー。これtime managementと極めて相性が悪いのでは。
	}
#endif

	// ネットワークのDelayを考慮して少し減らすべき。
	// かつ、minimumとmaximumは端数をなくすべき
    network_delay = (TimePoint) options["NetworkDelay"];

	// 秒未満を切り上げるのか
	round_up_to_fullsecond = options["RoundUpToFullSecond"];

	// 探索開始時刻と終了予定時刻。このタイミングで初期化しておく。
	// 終了時刻は0ならば未確定という意味である。
    startTime = ponderhitTime = limits.startTime;
    search_end                = 0;

	// 今回の最大残り時間(これを超えてはならない)
	// byoyomiは残り時間に加算。incは「この指し手の後に加算される時間」ゆえ現在の残り時間には足さない。
	// (旧実装は+incしていたが、remain_estimateが別途inc*MTG(将来分)を加算=現在手incの二重計上。
	//  YO v9.60(98455708)準拠で除去。Fischer TCでremain_timeがinc1回分過大になる退行を修正。)
    remain_time =
      limits.time[us] + limits.byoyomi[us] - (TimePoint) options["NetworkDelay2"];

	// remain_timeを0にすると時間切れのあと自爆するのでとりあえず100はあることにしておく。
	// round_up_to_fullsecond == falseのときは秒未満での戦いなので、1にしておく。
	remain_time = std::max(remain_time, (TimePoint) (round_up_to_fullsecond ? 100 : 1) );

	// 最小思考時間
    minimum_thinking_time = (TimePoint) options["MinimumThinkingTime"];

	// 序盤重視率
	// 　これはこんなパラメーターとして手で調整するべきではなく、探索パラメーターの一種として
	//   別の方法で調整すべき。ただ、対人でソフトに早指ししたいときには意味があるような…。
	int slowMover = (int)options["SlowMover"];

	if (limits.rtime)
	{
		// これが指定されているときは最小思考時間をランダム化する。
		// 連続自己対戦時に同じ進行になるのを回避するためのもの。
		// 終盤で大きく持ち時間を変えると、勝率が5割に寄ってしまうのでそれには注意。

		auto r = limits.rtime;
#if 1
		// 指し手が進むごとに減衰していく曲線にする。
		if (ply)
			r += (int)prng.rand((int)std::min(r * 0.5f, r * 10.0f / (ply)));
#endif

		remain_time = minimumTime = optimumTime = maximumTime = r;
		return;
	}

	// 時間固定モード
	// "go movetime 100"のようにして思考をさせた場合。
	if (limits.movetime)
	{
		remain_time = minimumTime = optimumTime = maximumTime = limits.movetime;
		return;
	}

	// 切れ負けであるか？
	bool time_forfeit = limits.inc[us] == 0 && limits.byoyomi[us] == 0;

	// 1. 切れ負けルールの時は、MoveHorizonを + 40して考える。
	// 2. ゲーム開始直後～40手目ぐらいまでは定跡で進むし、そこまで進まなかったとしても勝負どころはそこではないので
	// 　ゲーム開始直後～40手目付近のMoveHorizonは少し大きめに考える必要がある。逆に40手目以降、MoveHorizonは40ぐらい減らして考えていいと思う。
	// 3. 切れ負けでないなら、100手時点で残り60手ぐらいのつもりで指していいと思う。(これくらいしないと勝負どころすぎてからの持ち時間が余ってしまう..)
	// (現在の大会のフィッシャールールは15分+inctime5秒とか5分+inctime10秒そんな感じなので、160手目ぐらいで持ち時間使い切って問題ない)
	int move_horizon;
	if (time_forfeit)
		move_horizon = MoveHorizon + 40 - std::min(ply , 40);
	else
		// + 20は調整項
		move_horizon = MoveHorizon + 20 - std::min(ply , 80);

	// 残りの自分の手番の回数
	// ⇨　plyは平手の初期局面が1。256手ルールとして、max_game_ply == 256だから、256手目の局面においてply == 256
	// 　その1手前の局面においてply == 255。ply == 255 or 256のときにMTGが1にならないといけない。だから2足しておくのが正解。
    const int MTG = std::min(max_moves_to_draw - ply + 2, move_horizon) / 2;

	if (MTG <= 0)
	{
		// 本来、終局までの最大手数が指定されているわけだから、この条件で呼び出されるはずはないのだが…。
		sync_cout << "info string Error! : MaxMovesToDraw is too small." << sync_endl;
		// 事故防止のために何か設定はしておく。
        minimumTime = optimumTime = maximumTime = 500;
        return;
	}
	if (MTG == 1)
	{
		// この手番で終了なので使いきれば良い。
		minimumTime = optimumTime = maximumTime = remain_time;
		return;
	}

	// minimumとoptimumな時間を適当に計算する。

	{
		// 最小思考時間(これが1000より短く設定されることはないはず..)
		// round_up_to_fullsecond == falseのときは秒未満での戦いなので、1にしておく。
		minimumTime = std::max(minimum_thinking_time - network_delay,
							   (TimePoint)(round_up_to_fullsecond ? 1000 : 1));

		// 最適思考時間と、最大思考時間には、まずは上限値を設定しておく。
		optimumTime = maximumTime = remain_time;

		// optimumTime = min ( minimumTime + α     , remain_time)
		// maximumTime = min ( minimumTime + α * 5 , remain_time)
		// みたいな感じで考える

		// 残り手数において残り時間はあとどれくらいあるのか。
		// p2shogi 2026-09-14: Fischer のとき、 計画上だけ温存分を差し引く。
		//   ハード上限の remain_time には触れないので、 既定 (X=0) では plan_time == limits.time[us]。
		const bool p2_fischer = limits.inc[us] > 0 && limits.byoyomi[us] == 0;
		TimePoint p2_reserve = 0;
		if (p2_fischer)
		{
			const int rx = (int)options["SPSA_TM_RESERVE_INC_X10"];
			if (rx) p2_reserve = limits.inc[us] * rx / 10;
		}
		const TimePoint p2_plan_time = std::max(limits.time[us] - p2_reserve, TimePoint(0));

		TimePoint remain_estimate = p2_plan_time
			+ limits.inc[us] * MTG
			// 秒読み時間も残り手数に付随しているものとみなす。
			+ limits.byoyomi[us] * MTG;

		// 秒単位切り捨てルールでは、各手で1秒未満を使う前提で予約しておく。
		if (round_up_to_fullsecond)
			remain_estimate -= (MTG + 1) * 1000;
		remain_estimate = std::max(remain_estimate, TimePoint(0));

		// -- optimumTime
		TimePoint t1 = minimumTime + remain_estimate / MTG;

		// -- maximumTime

		// 5.0fでもうまくマネージメントできる。
		// 📝 上手くタイムマネージメントできない探索部なら3.0fぐらいにするのが無難。
		float max_ratio = 5.0f;

		// 切れ負けルールにおいては、5分を切っていたら、このratioを抑制する。
		if (time_forfeit)
		{
			// 3分     : ratio = 3.0
			// 2分     : ratio = 2.0
			// 1分以下 : ratio = 1.0固定
			max_ratio = std::min(max_ratio, std::max(float(limits.time[us]) / (60 * 1000), 1.0f));
		}
		// p2shogi: Fischer(増分あり・秒読みなし)は主時間を食い潰すと切れ負けする。
		// max_ratio=5.0 だと1手で60-69s使い、増分10sでは追いつかずクロックが枯渇する
		// (floodgate-300-10F で %TIME_UP 実発生 2026-07-05)。
		// per-move spikeを抑えてsustainableに保つため max_ratio を 1.5 に抑制していた。
		// MaxRatioX100 オプションで調整可能とし、default 150(=1.5)で従来挙動を維持する。
		else if (limits.inc[us] > 0 && limits.byoyomi[us] == 0)
		{
			max_ratio = (float)(int)options["MaxRatioX100"] / 100.0f;
		}
		TimePoint t2 = minimumTime + (int)(remain_estimate * max_ratio / MTG);

		// ただしmaximumは残り時間の30%以上は使わないものとする。
		// optimumが超える分にはいい。それは残り手数が少ないときとかなので構わない。
		t2 = std::min(t2, (TimePoint)(remain_estimate * 0.3));

		// slowMoverは100分率で与えられていて、optimumTimeの係数として作用するものとする。
		optimumTime = std::min(t1, optimumTime) * slowMover / 100;
		maximumTime = std::min(t2, maximumTime);

		// p2shogi 2026-09-14: 1 手で銀行 (持ち時間の実残高) の pct% 超は使わせない。
		//   従来の `0.3 * remain_estimate` は inc*MTG (将来の加算) を含むため銀行保護になっていない。
		//   Stockfish の `0.825 * timeLeft` 相当。 pct=100 (既定) ではこのブロックは何もしない。
		if (p2_fischer)
		{
			const int bank_pct = (int)options["SPSA_TM_MAX_BANK_PCT"];
			if (bank_pct < 100)
			{
				const TimePoint bank = std::max(p2_plan_time - (TimePoint)(int)options["NetworkDelay2"], TimePoint(0));
				maximumTime = std::min(maximumTime, limits.inc[us] + bank * bank_pct / 100);
			}
		}

		// Ponderが有効でStochastic_Ponderが無効の場合、
		// ponderhitすると時間が本来の予測より余っていくので思考時間を心持ち多めにとっておく。
        if (options["USI_Ponder"] && !options["Stochastic_Ponder"])
			optimumTime += optimumTime / 4;

	}

	// 秒読みモードでかつ、持ち時間がないなら、最小思考時間も最大思考時間もその時間にしたほうが得
    isFinalPush = false;
	if (limits.byoyomi[us])
	{
		// 持ち時間が少ないなら(秒読み時間の1.2倍未満なら)、思考時間を使いきったほうが得
		// これには持ち時間がゼロのケースも含まれる。
        if (limits.time[us] < (int) (limits.byoyomi[us] * 1.2))
        {
            minimumTime = optimumTime = maximumTime = limits.byoyomi[us] + limits.time[us];
			// "ponderhit"の時刻から数えてminimumTime分は使って欲しい。
            isFinalPush                             = true;
        }
	}

	// 残り時間 - network_delay2よりは短くしないと切れ負けになる可能性が出てくる。
	minimumTime = std::min(round_up(minimumTime), remain_time);
	optimumTime = std::min(         optimumTime , remain_time);
	maximumTime = std::min(round_up(maximumTime), remain_time);

	// p2shogi 2026-11-01: 残り時間が少ないときの「床」を加算の内側に落とす。
	//   ★既定 (SPSA_TM_LOW_BANK_MS=0) ではこのブロックを通らないので従来と bit-exact。
	//
	// 🚨なぜ必要か (実測 / 器 runs/rescore/tm_scales_with_clock.py / Threads=3 / 本番 option):
	//   電竜戦の先手は 3 分 + 1 手 2 秒。 自分の棋譜の ply200 超の局面 16 件で測ると
	//     残 60 秒  中央 3.88 s / 90% 10.88 s / **加算 2.0 秒を超えた手 69%**
	//     残 20 秒  中央 1.88 s / 90%  5.88 s / **同 44%**
	//   床は optimumTime の定数項 `minimumTime + (inc - 1000)` に Ponder 補正 x1.25 が
	//   掛かったもので (minimumTime の設定 / t1 の式 / USI_Ponder の分岐)、
	//   **残り時間に依存しない**。 長手数の将棋 (入玉戦) で純減が続き先手が切れ負けする。
	//
	// 🚨**round_up の後**に置くこと (上で optimum/maximum が X.88 に丸められた後に min を取る)。
	//   実際の停止時刻は set_search_end() が round_up(max(elapsed, minimum())) で決め、
	//   minimum() = round_up(MinimumThinkingTime - NetworkDelay) = 1880 が**床**になる。
	//   よって cap <= 1880 なら停止は 1.88 s、 1881..2880 なら 2.88 s と **1 秒刻みに量子化**される。
	//   cap = inc - network_delay - 1000 は「加算の 1 秒内側」を狙う値で、
	//   inc 2000 / NetworkDelay 120 -> cap 880 -> 停止 1.88 s (= minimum() の床)。
	//   🚨2026-11-01 訂正: 初版のコメントは「round_up の t < t0 分岐が +1000 する」と
	//     書いていたが誤り。 1.88 s を作っているのは minimum() の床である。
	//   計測ログ: runs/prereg/tm_dist_20261101.log / tm_threshold_20261101.log /
	//             verify_lowbank.log (関門 4 本)
	// 🚨minimumTime は触らない。 床が MinimumThinkingTime - NetworkDelay より下がらないので安全側。
	// 🚨p2_fischer は上のブロックでスコープが閉じているのでここで再導出する。
	{
		const TimePoint low_bank = (TimePoint)(int)options["SPSA_TM_LOW_BANK_MS"];
		const bool      p2_f     = limits.inc[us] > 0 && limits.byoyomi[us] == 0;
		if (low_bank && p2_f && limits.time[us] < low_bank)
		{
			const TimePoint cap = std::max(limits.inc[us] - network_delay - 1000, TimePoint(1));
			optimumTime = std::min(optimumTime, cap);
			maximumTime = std::min(maximumTime, cap);
		}
	}
}

// 1秒単位で繰り上げてdelayを引く。
// ただし、remain_timeよりは小さくなるように制限する。
TimePoint TimeManagement::round_up(TimePoint t0) {
	if (round_up_to_fullsecond)
	{
		// 📓 秒未満を切り上げる時

		// 1000で繰り上げる。Options["MinimalThinkingTime"]が最低値。
		auto t = std::max(((t0 + 999) / 1000) * 1000, minimum_thinking_time);

		// そこから、Options["NetworkDelay"]の値を引く
		t = t - network_delay;

		// これが元の値より小さいなら、もう1秒使わないともったいない。
		if (t < t0)
			t += 1000;

		// remain_timeを上回ってはならない。
		t = std::min(t, remain_time);
		return t;

	} else {
		// 📓 秒未満を切り上げない時

		// Options["MinimalThinkingTime"]が最低値。
		auto t = std::max(t0, minimum_thinking_time);

		// そこから、Options["NetworkDelay"]の値を引く
		t = t - network_delay;

		// remain_timeを上回ってはならない。
		t = std::min(t, remain_time);
		return t;
	}
};

// 探索を終了させることが確定しているが、秒単位で切り上げて、search_endにそれを設定したい時に呼び出す。
void TimeManagement::set_search_end(TimePoint e) {
    /*
		🤔 現在時刻から、ponderhitした時刻から計算して、秒単位で切り上げた時刻まで思考させたい。

		📓
		    1. 使用した時間の計測はponderhitした時刻からの経過時間としてなされるために、
			   ponderhitからの経過時間を秒単位で切り上げしたい。
		       そのためには、round_up(elapsed_from_ponderhit())のようにround upする必要がある。

			2. 一方、"go"した時刻から計算して、tm.minimum()の分は思考することを守らせたい。
			   しかし、ponderhitからの経過時間で切り上げはしたい。

			   しかし、秒読みで秒まで使い切りたいとき(isFinalPush)は、
			   "ponderhit"から数えてtm.minimum()の分は思考しないともったいない。
	*/

    // 1. ponderhitからの経過時間。(go ponder～ponderhitしていない場合は、単にgoからの経過時間)
    TimePoint t1 = e + startTime - ponderhitTime;

	// 2. "go"した時刻からminimum()を足して、ponderhitからの経過時間に変換したもの。
	//    ただし、finalpushであるなら、"ponderhit"した時刻から数える。
    TimePoint t2 = isFinalPush ? minimum() : minimum() + startTime - ponderhitTime;

	// t1,t2の大きいほうを秒単位で切り上げて、それをstartTimeからの経過時間に換算したもの。
    // 💡 search_endの値は、startTimeからの経過時間なので。
    search_end = round_up(std::max(t1, t2)) + ponderhitTime - startTime;
}

TimePoint TimeManagement::minimum() const { return minimumTime; }
TimePoint TimeManagement::optimum() const { return optimumTime; }
TimePoint TimeManagement::maximum() const { return maximumTime; }


} // namespace YaneuraOu

#endif
