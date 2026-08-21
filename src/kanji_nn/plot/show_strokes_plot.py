from kanji_nn.plot import strokes_plot

xy_fn = lambda s: s.xy

def show_strokes_plot(xy_fn=xy_fn, alpha=0.1):
    strokes = []
    def inner(stroke):
        nonlocal strokes
        strokes.append(stroke)

        if len(strokes) == stroke.stroke_count:
            xy = [xy_fn(stroke) for stroke in strokes]
            strokes_plot.show(xy, alpha=alpha)
            strokes = []
        return stroke

    return inner
