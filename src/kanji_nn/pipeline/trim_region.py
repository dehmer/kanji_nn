from kanji_nn.data import Stroke


def trim_region(stroke):
    region = stroke.props["cuts"]
    trimmed_raw = stroke.raw[region[0]:region[1], :]

    # implicitly create new stroke from trimmed raw:
    return Stroke(
        dataset=stroke.dataset,
        key=stroke.key,
        raw=trimmed_raw,
        sticky=stroke.sticky
    )
