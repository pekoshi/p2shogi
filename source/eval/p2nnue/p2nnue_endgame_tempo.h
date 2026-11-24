#ifndef P2NNUE_ENDGAME_TEMPO_H_INCLUDED
#define P2NNUE_ENDGAME_TEMPO_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_P2NNUE) && defined(P2_ENDGAME_TEMPO)

#if !defined(LONG_EFFECT_LIBRARY)
#error "P2_ENDGAME_TEMPO requires LONG_EFFECT_LIBRARY"
#endif

#include "../../bitboard.h"
#include "../../position.h"

namespace YaneuraOu::Eval::P2NNUE::EndgameTempo {

constexpr int CLASS_NB = 64;
constexpr int BUCKET_NB = 8;
constexpr int WEIGHT_NB = 2 * CLASS_NB * BUCKET_NB;
constexpr int OUTPUT_SCALE = 4;

inline int king_urgency(const Position& pos, Color side) {
	const Color enemy = ~side;
	const Square king = pos.square<KING>(side);
	Bitboard zone = kingEffect(king) | Bitboard(king);
	int pressure = 0;
	while (zone) pressure += pos.board_effect[enemy].effect(zone.pop());

	Bitboard escape = pos.pieces(side).andnot(kingEffect(king));
	int safe = 0;
	while (escape) {
		const Square sq = escape.pop();
		if (pos.board_effect[enemy].effect(sq) == 0) ++safe;
	}
	if (safe == 0) return 0;
	if (safe <= 2) return 1;
	return pressure < 2 ? 2 : 3;
}

inline int drop_pressure(const Position& pos, Color side) {
	const Color enemy = ~side;
	const Hand hand = pos.hand_of(enemy);
	const int raw = 2 * (hand_count(hand, ROOK) + hand_count(hand, BISHOP)
	                   + hand_count(hand, GOLD) + hand_count(hand, SILVER))
	              + hand_count(hand, LANCE) + hand_count(hand, KNIGHT) + hand_count(hand, PAWN);
	return raw == 0 ? 0 : raw <= 2 ? 1 : raw <= 4 ? 2 : 3;
}

inline int major_state(const Position& pos, Color side) {
	const Color enemy = ~side;
	Bitboard majors = pos.pieces(side, BISHOP, ROOK, HORSE, DRAGON);
	int state = 0;
	while (majors) {
		const Square sq = majors.pop();
		state = (std::max)(state, 1);
		if (pos.board_effect[enemy].effect(sq) != 0)
			state = (std::max)(state, pos.board_effect[side].effect(sq) != 0 ? 2 : 3);
	}
	return state;
}

inline int compute_class(const Position& pos, Color side) {
	const int value = (king_urgency(pos, side) * 4 + drop_pressure(pos, side)) * 4
	                + major_state(pos, side);
	ASSERT_LV3(0 <= value && value < CLASS_NB);
	return value;
}

} // namespace YaneuraOu::Eval::P2NNUE::EndgameTempo

#endif
#endif
