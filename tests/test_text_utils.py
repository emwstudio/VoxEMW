import pytest

from agent.text_utils import split_text_for_tts


def test_short_text_single_chunk():
    assert split_text_for_tts("味真足！") == ["味真足！"]


def test_empty_and_whitespace():
    assert split_text_for_tts("") == []
    assert split_text_for_tts("   \n  ") == []


def test_exact_boundary_stays_single_chunk():
    text = "a" * 200
    assert split_text_for_tts(text, max_chars=200) == [text]


def test_splits_on_punctuation():
    sentence = "今天吃了十斤焖子，味真足！减肥这一块，明天再说吧。"
    text = sentence * 10  # well over 200 chars
    chunks = split_text_for_tts(text, max_chars=60)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 60
    # punctuation-preference: every chunk except the last ends on punctuation
    for chunk in chunks[:-1]:
        assert chunk[-1] in "。！？；，、.!?;,"
    # no content lost
    assert "".join(chunks) == text.strip()


def test_no_punctuation_hard_split():
    text = "a" * 500
    chunks = split_text_for_tts(text, max_chars=200)
    assert chunks == ["a" * 200, "a" * 200, "a" * 100]


def test_long_clause_after_short_one():
    text = "好的。" + "b" * 250
    chunks = split_text_for_tts(text, max_chars=100)
    assert chunks == ["好的。", "b" * 100, "b" * 100, "b" * 50]


def test_mixed_ascii_punctuation():
    text = "hello world, this is great! " * 20
    chunks = split_text_for_tts(text, max_chars=50)
    assert all(len(c) <= 50 for c in chunks)
    assert "".join(chunks) == text.strip()


def test_invalid_max_chars():
    with pytest.raises(ValueError):
        split_text_for_tts("text", max_chars=0)
