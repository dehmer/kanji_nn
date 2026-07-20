from .char_key import char_key

def preload_cuts(strokes_dict, rows):
    for row in rows:
        stroke_idx = int(row["stroke_idx"])
        cuts = (int(row["head_cut"]), int(row["tail_cut"]))
        strokes = strokes_dict[char_key(row)]
        stroke = strokes[stroke_idx]
        strokes[stroke_idx] = stroke.clone(props={"cuts": cuts})

    return strokes_dict
