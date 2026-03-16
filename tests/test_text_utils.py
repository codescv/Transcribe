from transcribe.text_utils import remove_overlap

def test_remove_overlap_english():
    assert remove_overlap("hello world", "world is great") == "is great"
    assert remove_overlap("hello world", "world is great") == "is great"
    assert remove_overlap("one two three", "three four five") == "four five"
    assert remove_overlap("one two three", "two three four") == "four"

def test_remove_overlap_chinese():
    # Character based CJK
    assert remove_overlap("今天天气不错", "天气不错挺好的") == "挺好的"
    assert remove_overlap("你好世界", "世界很大") == "很大"

def test_no_overlap():
    assert remove_overlap("hello", "world") == "world"
    assert remove_overlap("", "world") == "world"
    assert remove_overlap("hello", "") == ""

def test_min_overlap_threshold():
    # "the" is small, shouldn't match if it's the only overlap for non-space-separated?
    # Actually word-based takes priority if spaces are present.
    assert remove_overlap("I went to the", "the store") == "store"
    # Testing character based with short overlap
    assert remove_overlap("abc", "cde") == "cde" # Overlap is 1, too small if not CJK
    assert remove_overlap("你好", "好哇") == "好哇" # Overlap 1 CJK character, too small (best-effort)
    # Testing overlapping length >= 2 for CJK
    assert remove_overlap("你好吗", "好吗我很好") == "我很好"

def test_exact_match():
    assert remove_overlap("hello world", "hello world") == ""
    assert remove_overlap("你好", "你好") == ""
