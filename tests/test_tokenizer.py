import pytest
from tokenizer.bpe import ByteBPETokenizer, SpecialTokens


def test_tokenizer_roundtrip():
    text = "Hello, world! 123 + 456 = 579. How's it going? \n\t Test: def func(x): return x * 2"
    tok = ByteBPETokenizer.train(text, vocab_size=300, min_frequency=1, verbose=False)
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text


def test_special_tokens():
    text = "<system>You are AI.<user>Hi<assistant>Hello!"
    tok = ByteBPETokenizer.train("Sample corpus for training", vocab_size=280, min_frequency=1, verbose=False)
    encoded = tok.encode(text, allowed_special=True)
    assert tok.system_id in encoded
    assert tok.user_id in encoded
    assert tok.assistant_id in encoded

    decoded_with_special = tok.decode(encoded, skip_special_tokens=False)
    assert "<system>" in decoded_with_special
    assert "<user>" in decoded_with_special
    assert "<assistant>" in decoded_with_special


def test_legacy_format_loading(tmp_path):
    # Test loading character list legacy format
    legacy_file = tmp_path / "legacy.vocab.json"
    legacy_file.write_text('["<unk>", "\\n", " ", "a", "b", "c"]', encoding="utf-8")
    tok = ByteBPETokenizer.load(legacy_file)
    assert tok.vocab_size >= 6
