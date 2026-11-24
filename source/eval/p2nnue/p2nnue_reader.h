#ifndef P2NNUE_READER_H_INCLUDED
#define P2NNUE_READER_H_INCLUDED

// p2shogi NNUE 移植: バイナリ読み込みヘルパ + LEB128 デコーダ
// 移植元: crates/rshogi-core/src/nnue/leb128.rs
//
// p2shogi の重みファイルは COMPRESSED_LEB128(符号付きLEB128)で量子化重みを格納する。
// 本ヘッダはメモリ上に読み込んだバイト列(std::vector<u8>)を逐次デコードする。

#include "../../config.h"

#if defined(EVAL_P2NNUE)

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace YaneuraOu::Eval::P2NNUE {

// COMPRESSED_LEB128 マジック(17 byte)
inline constexpr char LEB128_MAGIC[] = "COMPRESSED_LEB128"; // 末尾NUL含めて18byte。比較は17byteで行う

// バイト列を保持し、現在位置から逐次読み出すリーダ。
// 失敗時は ok_=false を立て、以降の読み出しは安全側(0)を返す。呼び出し側は ok() を最終確認する。
class Reader {
public:
	Reader(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}
	explicit Reader(const std::vector<std::uint8_t>& v) : data_(v.data()), size_(v.size()) {}

	bool ok() const { return ok_; }
	std::size_t pos() const { return pos_; }
	std::size_t remaining() const { return pos_ <= size_ ? size_ - pos_ : 0; }
	bool eof() const { return pos_ >= size_; }

	// 残りバイト数が n 以上あるか
	bool has(std::size_t n) const { return pos_ + n <= size_; }

	// 生バイト n 個を dst へコピー
	bool read_bytes(void* dst, std::size_t n) {
		if (!ok_ || !has(n)) { ok_ = false; return false; }
		std::memcpy(dst, data_ + pos_, n);
		pos_ += n;
		return true;
	}

	std::uint8_t read_u8() {
		if (!ok_ || !has(1)) { ok_ = false; return 0; }
		return data_[pos_++];
	}

	std::uint32_t read_u32_le() {
		std::uint8_t b[4];
		if (!read_bytes(b, 4)) return 0;
		return std::uint32_t(b[0]) | (std::uint32_t(b[1]) << 8)
			| (std::uint32_t(b[2]) << 16) | (std::uint32_t(b[3]) << 24);
	}

	std::int16_t read_i16_le() {
		std::uint8_t b[2];
		if (!read_bytes(b, 2)) return 0;
		return std::int16_t(std::uint16_t(b[0]) | (std::uint16_t(b[1]) << 8));
	}

	std::int32_t read_i32_le() {
		return std::int32_t(read_u32_le());
	}

	// 固定長 ASCII 文字列を読む(arch 文字列等)
	std::string read_string(std::size_t n) {
		std::string s;
		if (!ok_ || !has(n)) { ok_ = false; return s; }
		s.assign(reinterpret_cast<const char*>(data_ + pos_), n);
		pos_ += n;
		return s;
	}

	// 8 byte の section magic を覗き見る(消費しない)。残量不足なら false。
	bool peek_magic8(std::uint8_t out[8]) const {
		if (pos_ + 8 > size_) return false;
		std::memcpy(out, data_ + pos_, 8);
		return true;
	}

	// 符号付き LEB128 を 1 値読む(leb128.rs read_signed_leb128 と同一ロジック)
	std::int64_t read_signed_leb128() {
		std::int64_t result = 0;
		int shift = 0;
		while (true) {
			if (!has(1)) { ok_ = false; return 0; }
			std::uint8_t b = data_[pos_++];
			result |= (std::int64_t(b & 0x7f) << shift);
			shift += 7;
			if ((b & 0x80) == 0) {
				// 符号拡張(最後のバイトの 0x40 ビットが符号)
				if (shift < 64 && (b & 0x40) != 0)
					result |= (~std::int64_t(0)) << shift;
				break;
			}
			if (shift >= 64) { ok_ = false; return 0; } // overflow
		}
		return result;
	}

	// COMPRESSED_LEB128 ブロックを読み、i16 配列へデコードする。
	// 形式: "COMPRESSED_LEB128"(17B) + compressed_size(u32 LE) + LEB128列
	// count を指定した場合はその個数だけ out へ詰める(過不足は ok_=false)。
	// count==0 のときはブロック内の全値をデコードして out へ返す。
	bool read_compressed_i16(std::vector<std::int16_t>& out, std::size_t count) {
		if (!ok_) return false;
		char magic[17];
		if (!read_bytes(magic, 17)) return false;
		if (std::memcmp(magic, LEB128_MAGIC, 17) != 0) { ok_ = false; return false; }

		std::uint32_t comp = read_u32_le();
		if (!ok_) return false;
		const std::size_t MAX_COMPRESSED = std::size_t(256) * 1024 * 1024;
		if (comp == 0 || comp > MAX_COMPRESSED || !has(comp)) { ok_ = false; return false; }

		const std::size_t end = pos_ + comp;
		if (count != 0) out.reserve(out.size() + count);
		while (pos_ < end) {
			std::int64_t v = read_signed_leb128();
			if (!ok_) return false;
			out.push_back(std::int16_t(v));
		}
		// ブロック境界は compressed_size で厳密に区切られる(pos_ は end と一致するはず)
		if (pos_ != end) { ok_ = false; return false; }
		if (count != 0 && out.size() != count) { ok_ = false; return false; }
		return true;
	}

private:
	const std::uint8_t* data_;
	std::size_t size_;
	std::size_t pos_ = 0;
	bool ok_ = true;
};

} // namespace YaneuraOu::Eval::P2NNUE

#endif // EVAL_P2NNUE
#endif // P2NNUE_READER_H_INCLUDED
