#!/usr/bin/env python3
import sentencepiece as spm

# ko_bpe.model 로드
sp_ko = spm.SentencePieceProcessor()
sp_ko.load('/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model')

# 원본 bpe.model 로드
sp_base = spm.SentencePieceProcessor()
sp_base.load('/mnt/sda1/models/IndexTTS-2/bpe.model')

print(f"ko_bpe.model vocab size: {sp_ko.vocab_size()}")
print(f"base bpe.model vocab size: {sp_base.vocab_size()}")

# 한국어 테스트
korean_text = "안녕하세요 테스트입니다"
print(f"\n한국어 테스트: '{korean_text}'")
print(f"  ko_bpe tokens: {sp_ko.encode_as_pieces(korean_text)}")
print(f"  base tokens: {sp_base.encode_as_pieces(korean_text)}")

# 영어 테스트
english_text = "Hello this is a test"
print(f"\n영어 테스트: '{english_text}'")
print(f"  ko_bpe tokens: {sp_ko.encode_as_pieces(english_text)}")
print(f"  base tokens: {sp_base.encode_as_pieces(english_text)}")

# 중국어 테스트
chinese_text = "你好世界"
print(f"\n중국어 테스트: '{chinese_text}'")
print(f"  ko_bpe tokens: {sp_ko.encode_as_pieces(chinese_text)}")
print(f"  base tokens: {sp_base.encode_as_pieces(chinese_text)}")
