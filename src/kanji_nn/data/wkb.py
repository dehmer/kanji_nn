
def wkb(stroke, wkb_reader):
    _, strokes = wkb_reader[stroke.code_point]
    wkb = strokes[stroke.stroke_index]
    return stroke.clone(props={"wkb": wkb})
