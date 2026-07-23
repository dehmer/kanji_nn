

def trim_region(stroke):
    region = stroke.props["cuts"]
    return stroke.trim(region)
