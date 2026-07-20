from kanji_nn.data.character import Character

def load_strokes(keys):
    chars = dict()
    for key in keys:
        dataset, code_point = tuple(key.split(':'))
        filename = f"data/dataset/{dataset}/npy-raw/{code_point}.npy"
        character = Character.of_npy(dataset, filename)
        chars[key] = character.strokes()
    return chars
