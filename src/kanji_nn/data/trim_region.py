

def trim_region(stroke):
    region = stroke.props["cuts"]
    head_cut, tail_cut = region

    header = f"[trim_region] {stroke.literal}/{stroke.stroke_index}"

    if head_cut >= tail_cut:
        print(f"{header} - trespassing {region}")
        return stroke
    elif (tail_cut - head_cut) / stroke.n_points < 0.4:
        print(f"{header} - overcut {region}")
        return stroke
    else:
        return stroke.trim(region)
