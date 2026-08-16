import numpy as np
from kanji_nn.plot import strokes_plot


def plot_overlays():
    query, reference, struts = [], [], []

    def inner(stroke):
        nonlocal query, reference, struts

        query.append(stroke.xy)
        reference.append(stroke.props["path:xys"][:, :2])
        struts.append(stroke.props["struts"])

        if len(query) == stroke.stroke_count:

            styles = [
                {"color": "green"},
                {"color": "black"},
            ]

            struts = np.vstack(struts)
            strokes_plot.overlays([query, reference], styles=styles, struts=struts)
            query, reference, struts = [], [], []
        return stroke
    return inner
