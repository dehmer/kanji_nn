from kanji_nn.bezier import classify_bezier


init_classes = {"empty": 0, "near-straight": 0, "s-bend": 0, "right-bend": 0, "left-bend": 0}


def segment_classification(stroke):
    path = stroke.sticky["path"]
    classes = stroke.props.get("path:classes", init_classes)
    for i, segment in enumerate(path):
        if i == 0: continue # Move
        classes[classify_bezier(segment)] += 1
        print(f"{stroke.literal}/{stroke.stroke_index} - ", classes)

    return stroke.clone(props={"path:classes": classes})
