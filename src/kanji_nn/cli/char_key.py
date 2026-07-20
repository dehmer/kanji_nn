
def char_key(row):
    dataset = row['dataset']
    literal = row["literal"]
    code_point = f"U+{ord(literal):04X}"
    return f"{dataset}:{code_point}"
