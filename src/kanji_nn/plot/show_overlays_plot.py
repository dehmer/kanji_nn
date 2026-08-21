import numpy as np
from kanji_nn.plot import strokes_plot


def show_overlays_plot():
    query, reference, struts = [], [], []

    def inner(stroke):
        nonlocal query
        nonlocal reference
        nonlocal struts

        query.append(stroke.xy)
        reference.append(stroke.props["path:xys"][:, :2])

        if "struts" in stroke.props:
            struts.append(stroke.props["struts"])

        if len(query) == stroke.stroke_count:

            styles = [
                {"color": "green"},
                {"color": "black"},
            ]

            if len(struts):
                struts = np.vstack(struts)

            strokes_plot.overlays([query, reference], styles=styles, struts=struts)
            query, reference, struts = [], [], []
        return stroke
    return inner
