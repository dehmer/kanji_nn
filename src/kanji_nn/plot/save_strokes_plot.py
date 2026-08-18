import os
from kanji_nn.plot import strokes_plot

xy_fn = lambda s: s.xy

def save_strokes_plot(filename_fn, xy_fn=xy_fn, alpha=0.1, title=None):
    strokes = []
    def inner(stroke):
        nonlocal strokes
        strokes.append(stroke)

        if len(strokes) == stroke.stroke_count:
            filename = filename_fn(stroke)
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.mkdir(dirname)

            xy = [xy_fn(stroke) for stroke in strokes]
            strokes_plot.save(filename, xy, alpha=alpha, title=title)
            strokes = []
        return stroke
    return inner
