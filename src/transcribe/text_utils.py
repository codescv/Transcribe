def remove_overlap(prev: str, current: str) -> str:
    """
    Remove overlapping text at the end of 'prev' and start of 'current'.
    Supports both space-separated (English) and dense (Chinese) text.
    """
    prev = prev.strip()
    current = current.strip()
    
    if not prev or not current:
        return current
        
    # 1. Try word-based matching (for English / space-separated)
    if " " in prev and " " in current:
        prev_words = prev.split()
        curr_words = current.split()
        max_words = min(len(prev_words), len(curr_words))
        
        for i in range(max_words, 0, -1):
            if prev_words[-i:] == curr_words[:i]:
                return " ".join(curr_words[i:])
                
    # 2. Try character-based matching (for Chinese or Mixed without spaces)
    # This is useful for CJK where spaces might be absent.
    max_len = min(len(prev), len(current))
    for i in range(max_len, 0, -1):
        if prev[-i:] == current[:i]:
            # Avoid matching single short words or single characters which might be coincidental
            # CJK characters usually have ord > 127
            is_cjk = all(ord(c) > 127 for c in current[:i])
            if i >= 4 or (i >= 2 and is_cjk):
                return current[i:].strip()
                
    return current
