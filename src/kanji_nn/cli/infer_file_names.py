def infer_file_names(literals):
    return [f"U+{literal_to_hex(literal)}.npy" for literal in literals]
