

def trim_region(stroke):
    region = stroke.props["cuts"]
    xy = stroke.xy[region[0]:region[1], :]
    return stroke.clone(props={"trimmed:xy": xy})
