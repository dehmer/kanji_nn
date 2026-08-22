from kanji_nn.data import Stroke


def trim_region(stroke):
    region = stroke.props["cuts"]
    xy = stroke.features["xy"]
    xy = xy[region[0]:region[1], :]

    # implicitly create new stroke from trimmed raw:
    return Stroke(
        dataset=stroke.dataset,
        key=stroke.key,
        sticky=stroke.sticky,
        features={"xy": xy}
    )
