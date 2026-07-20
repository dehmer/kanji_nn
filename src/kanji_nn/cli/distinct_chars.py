from . import char_key

def distinct_chars(rows):
    keys = [char_key(row) for row in rows]
    return set(keys)
