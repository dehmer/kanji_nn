from kanji_nn.plot import strokes_plot

xy_fun = lambda s: s.xy

def save_strokes_plot(dirname, xy_fun=xy_fun, alpha=0.1):
    strokes = []
    def inner(stroke):
        nonlocal strokes
        strokes.append(stroke)

        if len(strokes) == stroke.stroke_count:
            filename = f"data/dataset/{stroke.dataset}/{dirname}/{stroke.code_point}"
            xy = [xy_fun(stroke) for stroke in strokes]
            strokes_plot.save(filename, xy, alpha=alpha)
            strokes = []
        return stroke
    return inner
