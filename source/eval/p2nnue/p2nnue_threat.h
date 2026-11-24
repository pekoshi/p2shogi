#ifndef P2NNUE_THREAT_H_INCLUDED
#define P2NNUE_THREAT_H_INCLUDED

// p2shogi NNUE: Threat 入力(216,720 次元)の index 計算。
// 移植元(source of truth): bullet-shogi-nodchip-v2
//   crates/bullet_lib/src/game/inputs/shogi_halfka_hm_threat.rs (full profile)
//
// ★bit-exact 契約: YaneuraOu Square (int 0..80) == bullet Square raw (file*9+rank,
//   file 0=1筋 / rank 0=1段, Inv=80-sq, mirror_file=(8-file)*9+rank)。
//   ∴ 変換不要で bullet と同一 index を再現する。検証: 末尾 canonical vector。

#include "../../config.h"

#if defined(EVAL_P2NNUE)

#include <cstdint>
#include "../../types.h"

namespace YaneuraOu::Eval::P2NNUE::Threat {

// ThreatClass 数(King 除外、9 family)
constexpr int NUM_THREAT_CLASSES = 9;
// full profile 次元(除外なし)
constexpr int THREAT_DIM_FULL = 216720;
// attack pattern 総数: Black 9 + White 方向性 5
constexpr int NUM_ATTACK_PATTERNS = 14;
// pair_base テーブルサイズ 2*9*2*9
constexpr int NUM_PAIRS = 2 * NUM_THREAT_CLASSES * 2 * NUM_THREAT_CLASSES; // 324

// ThreatClass: 0=Pawn 1=Lance 2=Knight 3=Silver 4=GoldLike 5=Bishop 6=Rook 7=Horse 8=Dragon
// 各クラスの空盤面利き総数(全 81 マス合計、per color)。sum=6020 → THREAT_DIM=36*6020=216720。
constexpr int ATTACKS_PER_COLOR[NUM_THREAT_CLASSES] = {
	72, 324, 112, 328, 416, 816, 1296, 1104, 1552,
};
// 代表行圧縮で使用する駒種優先度。被攻撃駒の価値を最優先し、
// 同価値なら攻撃駒の価値、最後にfeature indexで安定してtie-breakする。
constexpr std::uint8_t REPRESENTATIVE_CLASS_VALUE[NUM_THREAT_CLASSES] = {
        1, 2, 2, 3, 4, 7, 8, 9, 10,
};
constexpr std::uint8_t representative_pair_priority(int ac, int dc) {
        return static_cast<std::uint8_t>(REPRESENTATIVE_CLASS_VALUE[dc] * 16
                                         + REPRESENTATIVE_CLASS_VALUE[ac]);
}


// ---- 除外 profile(ビルド時固定)----
// 契約元: bullet crates/bullet_lib/src/game/inputs/shogi_threat_exclusion.rs
//   id 0 = full            : 除外なし              → 216,720 次元
//   id 3 = no-pawn-target  : dc == 0(被攻撃側 Pawn)を attacker 不問で全除外 → 192,640 次元
// ビルド指定: -DP2_THREAT_PROFILE=3。未指定は従来どおり full。
// net 側は profile_id > 0 のとき arch に "ThreatProfile=<pid>," marker を持ち、
// threat 重み blob の直前に u32 LE の profile id prefix が付く(load 時に照合)。
#if defined(P2_THREAT_PROFILE)
constexpr int THREAT_PROFILE_ID = P2_THREAT_PROFILE;
#else
constexpr int THREAT_PROFILE_ID = 0;
#endif
static_assert(THREAT_PROFILE_ID == 0 || THREAT_PROFILE_ID == 3 || THREAT_PROFILE_ID == 11 || THREAT_PROFILE_ID == 12 || THREAT_PROFILE_ID == 13 || THREAT_PROFILE_ID == 14 || THREAT_PROFILE_ID == 15 || THREAT_PROFILE_ID == 16 || THREAT_PROFILE_ID == 17,
              "P2_THREAT_PROFILE: 0/3/11/12/13 legacy, 14 direct-v1, 15 direct-v2-all-piece, 16 full+hanging-split, 17 enemy-only-all");
// Profile 17 (2026-11-05, enemy-only-all): each perspective keeps only enemy-attacker -> own-target pairs
//   (as == 1 && ds == 0), all 9x9 classes (same-class pairs kept, unlike profile 11). 54,180 rows.
//   Enumeration = the one-perspective path (P2_THREAT_PARTIAL == 2 in p2nnue.cpp). Design/measurements:
//   runs/threat_tax/partial_20261104/ (bullet: ThreatProfile::EnemyOnlyAll, --threat-profile enemy-only-all).
// Profile 16 = full の pair 空間に「被攻撃駒にひもが無いか (h)」を足した hanging-split。
//   敵味方をまたぐ pair だけ h=1 用の別行を持つ (同色 pair は攻め駒自身がひもなので h=0 に縮退)。
//   設計: runs/prereg/DESIGN-20260924-THREAT-DEFENDED.md
constexpr bool THREAT_HANG_SPLIT = (THREAT_PROFILE_ID == 16);
#if defined(P2_THREAT_PROFILE) && P2_THREAT_PROFILE == 16 && !defined(LONG_EFFECT_LIBRARY)
#error "P2_THREAT_PROFILE=16 (hanging-split) requires LONG_EFFECT_LIBRARY for board_effect"
#endif
// profile 12はprofile 11と同じ特徴空間・重みレイアウトを使い、局面ごとに
// 各視点の優先度上位8行だけを評価する代表行圧縮。
constexpr bool THREAT_REPRESENTATIVE_ROWS = (THREAT_PROFILE_ID == 12);
constexpr int THREAT_REPRESENTATIVE_MAX_ROWS = 8;
// Profile 13 retains profile-11 features but routes them through a 16D scalar branch.
constexpr bool THREAT_FACTORIZED = (THREAT_PROFILE_ID == 13);
// Profile 14 is legacy P2THRD1. Profile 15 is the independent all-piece P2THRD2 path.
constexpr bool THREAT_DIRECT = (THREAT_PROFILE_ID == 14 || THREAT_PROFILE_ID == 15);

// ThreatDirect same-class pair scope for isolated NPS measurements.
// 1: minor (Pawn/Lance/Knight/Silver/GoldLike) only.
// 2: minor plus major (all 9 ThreatClass values; king excluded).
// 3: scope 2 plus king scanned as GoldLike for NPS only.
#ifndef P2_THREAT_DIRECT_SCOPE
#define P2_THREAT_DIRECT_SCOPE 2
#endif
static_assert(P2_THREAT_DIRECT_SCOPE >= 1 && P2_THREAT_DIRECT_SCOPE <= 3,
              "P2_THREAT_DIRECT_SCOPE must be 1 (minor), 2 (all non-king), or 3 (king as GoldLike)");
constexpr int THREAT_FACTOR_DIM = 16;

// 被攻撃側 class が Pawn(dc==0)のペアを除外する profile か。
// ★注意: ここでの Pawn は「歩(未成)」のみ。と金・成香等は GoldLike(dc==4)なので除外されない
//   (tc_from_pt を参照)。列挙側の bitboard mask も pos.pieces(PAWN)(=未成の歩のみ)を使うこと。
constexpr bool THREAT_EXCLUDE_PAWN_TARGET = (THREAT_PROFILE_ID == 3);

// bullet ThreatProfile::is_excluded と同一ロジック。
constexpr bool is_excluded_pair(int as, int ac, int ds, int dc) {
        // EnemyOnly(profile 11): retain only enemy attacker -> friend target
        // and discard same-class pairs. This is exactly bullet ThreatProfile::EnemyOnly.
        if (THREAT_PROFILE_ID == 11 || THREAT_PROFILE_ID == 12 || THREAT_PROFILE_ID == 13)
                return !(as == 1 && ds == 0) || ac == dc;
        // Profile 14 retains EnemyOnly pairs and only re-admits same-class
        // Bishop/Rook/Horse/Dragon pressure. King-zone attack/defense is
        // deliberately delegated to Dual PSQT.
        if (THREAT_PROFILE_ID == 14)
                return !(as == 1 && ds == 0);
        if (THREAT_PROFILE_ID == 17)
                return !(as == 1 && ds == 0);
        return THREAT_EXCLUDE_PAWN_TARGET ? (dc == 0) : false;
}

// profile 適用後の threat 次元(bullet build_pair_base の cumulative と同一)。
constexpr int threat_dim_of_profile() {
	int cumulative = 0;
	for (int as = 0; as < 2; ++as)
		for (int ac = 0; ac < NUM_THREAT_CLASSES; ++ac)
			for (int ds = 0; ds < 2; ++ds)
				for (int dc = 0; dc < NUM_THREAT_CLASSES; ++dc)
					if (!is_excluded_pair(as, ac, ds, dc)) cumulative += ATTACKS_PER_COLOR[ac];
	return cumulative;
}
// hanging-split 用: 敵味方をまたぐ pair だけを詰めた cumulative の総和。
constexpr bool is_cross_pair(int as, int ds) { return as != ds; }
constexpr int threat_hang_dim_of_profile() {
	if (!THREAT_HANG_SPLIT) return 0;
	int cumulative = 0;
	for (int as = 0; as < 2; ++as)
		for (int ac = 0; ac < NUM_THREAT_CLASSES; ++ac)
			for (int ds = 0; ds < 2; ++ds)
				for (int dc = 0; dc < NUM_THREAT_CLASSES; ++dc)
					if (!is_excluded_pair(as, ac, ds, dc) && is_cross_pair(as, ds))
						cumulative += ATTACKS_PER_COLOR[ac];
	return cumulative;
}
constexpr int THREAT_BASE_DIM = threat_dim_of_profile();
constexpr int THREAT_HANG_DIM = threat_hang_dim_of_profile();
constexpr int THREAT_DIM_PROFILE = THREAT_BASE_DIM + THREAT_HANG_DIM;
static_assert(THREAT_PROFILE_ID != 0 || THREAT_DIM_PROFILE == 216720, "full profile dim");
static_assert(THREAT_PROFILE_ID != 3 || THREAT_DIM_PROFILE == 192640, "no-pawn-target profile dim");
static_assert(THREAT_PROFILE_ID != 11 || THREAT_DIM_PROFILE == 48160, "enemy-only profile dim");
static_assert(THREAT_PROFILE_ID != 12 || THREAT_DIM_PROFILE == 48160, "enemy-only-rep8 profile dim");
static_assert(THREAT_PROFILE_ID != 13 || THREAT_DIM_PROFILE == 48160, "enemy-only-factor16 profile dim");
static_assert(THREAT_PROFILE_ID != 14 || THREAT_DIM_PROFILE == 54180, "enemy-only-major-same-direct profile dim");
static_assert(THREAT_PROFILE_ID != 17 || THREAT_DIM_PROFILE == 54180, "enemy-only-all profile dim");
static_assert(THREAT_PROFILE_ID != 16 || THREAT_BASE_DIM == 216720, "hang-split base dim");
static_assert(THREAT_PROFILE_ID != 16 || THREAT_HANG_DIM == 108360, "hang-split hang dim");
static_assert(THREAT_PROFILE_ID != 16 || THREAT_DIM_PROFILE == 325080, "hang-split total dim");

// YaneuraOu PieceType → ThreatClass。King/None は -1。
inline int tc_from_pt(PieceType pt) {
	switch (pt) {
	case PAWN:   return 0;
	case LANCE:  return 1;
	case KNIGHT: return 2;
	case SILVER: return 3;
	case GOLD:
	case PRO_PAWN:
	case PRO_LANCE:
	case PRO_KNIGHT:
	case PRO_SILVER: return 4; // GoldLike
	case BISHOP: return 5;
	case ROOK:   return 6;
	case HORSE:  return 7;
    case DRAGON: return 8;
    case KING:
        // Scope 3 maps king to GoldLike for NPS measurements only.
        return (THREAT_PROFILE_ID == 14 && P2_THREAT_DIRECT_SCOPE == 3) ? 4 : -1;
    default:     return -1; // NO_PIECE_TYPE
	}
}

// target(被攻撃駒)として emit してはいけない駒種か。
// King/NO_PIECE_TYPE は常に除外。no-pawn-target profile では歩(未成)も除外。
inline bool target_excluded(PieceType pt) {
	const int dc = tc_from_pt(pt);
	return dc < 0 || (THREAT_EXCLUDE_PAWN_TARGET && dc == 0);
}

// ---- square helpers(bullet raw == YaneuraOu Square int)----
inline int sq_file(int s) { return s / 9; }
inline int sq_rank(int s) { return s % 9; }
inline int sq_inverse(int s) { return 80 - s; }
inline int sq_new(int f, int r) { return f * 9 + r; }
inline int sq_mirror_file(int s) { return (8 - s / 9) * 9 + (s % 9); }

// 方向性駒(色別 attack pattern を持つ): Pawn/Lance/Knight/Silver/GoldLike
inline bool is_directional(int tc) { return tc <= 4; }

// attack_pattern_id: 方向性駒は White を 9..13 に、非方向性は色不問 0..8。
inline int attack_pattern_id(int tc, Color oriented) {
	return (oriented == WHITE && is_directional(tc)) ? (NUM_THREAT_CLASSES + tc) : tc;
}

// マス正規化: perspective(Black 素 / White inverse)+ HM mirror。
inline int normalize_sq(int s, Color persp, bool hm) {
	int sn = (persp == BLACK) ? s : sq_inverse(s);
	return hm ? sq_mirror_file(sn) : sn;
}

// 空盤面上の攻撃先を raw 昇順で targets[] に詰め、count を返す。
// targets は 36 要素以上必須(Horse/Dragon 中央は最大 20)。
// bullet attacks_empty_board と同一(色: Black forward = rank-1, White = rank+1)。
inline int attacks_empty_board(int tc, Color color, int from, std::uint8_t* targets) {
	int count = 0;
	int file = from / 9;
	int rank = from % 9;
	auto push = [&](int f, int r) {
		if (f >= 0 && f < 9 && r >= 0 && r < 9) targets[count++] = (std::uint8_t)(f * 9 + r);
	};
	const int fwd = (color == BLACK) ? -1 : 1; // 前方(段の増減)

	switch (tc) {
	case 0: // Pawn
		push(file, rank + fwd);
		break;
	case 1: // Lance(空盤面: 盤端まで)
		if (color == BLACK) { for (int r = rank - 1; r >= 0; --r) push(file, r); }
		else { for (int r = rank + 1; r < 9; ++r) push(file, r); }
		break;
	case 2: // Knight
		push(file - 1, rank + 2 * fwd);
		push(file + 1, rank + 2 * fwd);
		break;
	case 3: // Silver: 前3 + 斜め後2
		push(file - 1, rank + fwd); push(file, rank + fwd); push(file + 1, rank + fwd);
		push(file - 1, rank - fwd); push(file + 1, rank - fwd);
		break;
	case 4: // GoldLike: 前3 + 横2 + 後1
		push(file - 1, rank + fwd); push(file, rank + fwd); push(file + 1, rank + fwd);
		push(file - 1, rank);       push(file + 1, rank);
		push(file, rank - fwd);
		break;
	case 5: // Bishop: 斜め4方向 盤端まで
		for (int d = 0; d < 4; ++d) {
			int df = (d < 2) ? -1 : 1, dr = (d & 1) ? 1 : -1;
			int f = file + df, r = rank + dr;
			while (f >= 0 && f < 9 && r >= 0 && r < 9) { targets[count++] = (std::uint8_t)(f * 9 + r); f += df; r += dr; }
		}
		break;
	case 6: // Rook: 直4方向 盤端まで
		for (int d = 0; d < 4; ++d) {
			int df = (d == 0) ? -1 : (d == 1) ? 1 : 0;
			int dr = (d == 2) ? -1 : (d == 3) ? 1 : 0;
			int f = file + df, r = rank + dr;
			while (f >= 0 && f < 9 && r >= 0 && r < 9) { targets[count++] = (std::uint8_t)(f * 9 + r); f += df; r += dr; }
		}
		break;
	case 7: // Horse: 角(斜め盤端)+ 上下左右1
		for (int d = 0; d < 4; ++d) {
			int df = (d < 2) ? -1 : 1, dr = (d & 1) ? 1 : -1;
			int f = file + df, r = rank + dr;
			while (f >= 0 && f < 9 && r >= 0 && r < 9) { targets[count++] = (std::uint8_t)(f * 9 + r); f += df; r += dr; }
		}
		push(file - 1, rank); push(file + 1, rank); push(file, rank - 1); push(file, rank + 1);
		break;
	case 8: // Dragon: 飛(直線盤端)+ 斜め1
		for (int d = 0; d < 4; ++d) {
			int df = (d == 0) ? -1 : (d == 1) ? 1 : 0;
			int dr = (d == 2) ? -1 : (d == 3) ? 1 : 0;
			int f = file + df, r = rank + dr;
			while (f >= 0 && f < 9 && r >= 0 && r < 9) { targets[count++] = (std::uint8_t)(f * 9 + r); f += df; r += dr; }
		}
		push(file - 1, rank - 1); push(file - 1, rank + 1); push(file + 1, rank - 1); push(file + 1, rank + 1);
		break;
	default: break;
	}

	// raw 昇順に挿入ソート
	for (int i = 1; i < count; ++i) {
		std::uint8_t key = targets[i];
		int j = i;
		while (j > 0 && targets[j - 1] > key) { targets[j] = targets[j - 1]; --j; }
		targets[j] = key;
	}
	return count;
}

// ---- LUT 群(初回 build 後 O(1) lookup)----
// 除外された pair の sentinel(bullet の EXCLUDED_PAIR_BASE 相当)。
// ★列挙側は除外 pair を emit しないので、この値が index 計算に混ざることは無い。
constexpr int EXCLUDED_PAIR_BASE = -1;

struct Tables {
	int pair_base[NUM_PAIRS];                                // profile 適用後の cumulative(除外は -1)
	// profile 16 (hanging-split): [0] = pair_base と同一、 [1] = THREAT_BASE_DIM + hang cumulative。
	//   同色 pair は h=1 の行を持たない (攻め駒自身がひも = h=0 に縮退する) ので -1。
	//   HANG_SPLIT でないときは [1] も pair_base と同じ値にして、 h を渡しても壊れないようにする。
	int pair_base2[2][NUM_PAIRS];
	int threat_dim = 0;                                      // = THREAT_DIM_PROFILE
        std::uint8_t representative_priority[THREAT_DIM_FULL] = {};   // profile 12用、idx→pair優先度
	int from_offset[NUM_ATTACK_PATTERNS][81];                // pattern×from → cumulative
	std::uint8_t attack_order[NUM_ATTACK_PATTERNS][81][81];  // pattern×from×to → order(255=invalid)
	bool built = false;

	void build() {
		if (built) return;
		// pair_base(profile 適用: 除外 pair は sentinel、残りを詰めて base を振り直す)。
		// bullet build_pair_base と同一ロジック。
		int cumulative = 0;
		for (int as = 0; as < 2; ++as)
			for (int ac = 0; ac < NUM_THREAT_CLASSES; ++ac)
				for (int ds = 0; ds < 2; ++ds)
					for (int dc = 0; dc < NUM_THREAT_CLASSES; ++dc) {
						int idx = as * 162 + ac * 18 + ds * 9 + dc;
						if (is_excluded_pair(as, ac, ds, dc)) {
							pair_base[idx] = EXCLUDED_PAIR_BASE;
						} else {
							pair_base[idx] = cumulative;
							                                                        const std::uint8_t priority = representative_pair_priority(ac, dc);
                                                        for (int row = 0; row < ATTACKS_PER_COLOR[ac]; ++row)
                                                                representative_priority[cumulative + row] = priority;
cumulative += ATTACKS_PER_COLOR[ac];
						}
					}
		// hanging-split 用の 2 段目 (cross pair のみ、 同じ入れ子順で詰める)。
		{
			int hang_cumulative = 0;
			for (int as = 0; as < 2; ++as)
				for (int ac = 0; ac < NUM_THREAT_CLASSES; ++ac)
					for (int ds = 0; ds < 2; ++ds)
						for (int dc = 0; dc < NUM_THREAT_CLASSES; ++dc) {
							const int idx = as * 162 + ac * 18 + ds * 9 + dc;
							pair_base2[0][idx] = pair_base[idx];
							if (!THREAT_HANG_SPLIT) { pair_base2[1][idx] = pair_base[idx]; continue; }
							if (pair_base[idx] == EXCLUDED_PAIR_BASE || !is_cross_pair(as, ds)) {
								pair_base2[1][idx] = EXCLUDED_PAIR_BASE;
							} else {
								pair_base2[1][idx] = cumulative + hang_cumulative;
								hang_cumulative += ATTACKS_PER_COLOR[ac];
							}
						}
		}
		threat_dim = cumulative + THREAT_HANG_DIM; // = THREAT_DIM_PROFILE

		// from_offset + attack_order
		for (int p = 0; p < NUM_ATTACK_PATTERNS; ++p)
			for (int f = 0; f < 81; ++f) {
				from_offset[p][f] = 0;
				for (int t = 0; t < 81; ++t) attack_order[p][f][t] = 255;
			}
		std::uint8_t targets[36];
		for (int tc = 0; tc < NUM_THREAT_CLASSES; ++tc) {
			// Black pattern
			build_pattern(tc, BLACK, tc, targets);
			// White 方向性駒は別 pattern
			if (is_directional(tc)) build_pattern(tc, WHITE, NUM_THREAT_CLASSES + tc, targets);
		}
		built = true;
	}

	        inline std::uint8_t rep_priority(int feature_index) const {
                return representative_priority[feature_index];
        }
void build_pattern(int tc, Color color, int pattern, std::uint8_t* targets) {
		int cum = 0;
		for (int from = 0; from < 81; ++from) {
			from_offset[pattern][from] = cum;
			int cnt = attacks_empty_board(tc, color, from, targets);
			for (int order = 0; order < cnt; ++order) attack_order[pattern][from][targets[order]] = (std::uint8_t)order;
			cum += cnt;
		}
	}

	// base + from_offset + attack_order。★除外 pair(pair_base == -1)を渡してはならない
	//   (呼び出し側の列挙で target を先に弾く契約)。
	// h = 1 は「被攻撃駒にひもが無い」。 profile 16 以外では無視される (pair_base2[1] == pair_base)。
	inline int index(int as, int ac, Color oriented, int ds, int dc, int from_n, int to_n, int h = 0) const {
		int base = pair_base2[h][as * 162 + ac * 18 + ds * 9 + dc];
		int pattern = attack_pattern_id(ac, oriented);
		return base + from_offset[pattern][from_n] + (int)attack_order[pattern][from_n][to_n];
	}
};

} // namespace YaneuraOu::Eval::P2NNUE::Threat

#endif // EVAL_P2NNUE
#endif // P2NNUE_THREAT_H_INCLUDED
