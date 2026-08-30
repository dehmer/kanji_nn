import kanji_nn.bezier as bezier


def transform_splines(glyph):
    paths = glyph["kvg:paths"]
    splines_bbox = glyph["kvg:bbox"]
    skeleton_bbox = glyph["skeleton:bbox"]

    splines = bezier.paths_to_array(paths)
    skeleton_dx = skeleton_bbox[2] - skeleton_bbox[0]
    skeleton_dy = skeleton_bbox[3] - skeleton_bbox[1]
    splines_dx = splines_bbox[2] - splines_bbox[0]
    splines_dy = splines_bbox[3] - splines_bbox[1]
    sx = skeleton_dx / splines_dx
    sy = skeleton_dy / splines_dy

    m = bezier.m_translate(skeleton_bbox[0], skeleton_bbox[1]) @ bezier.m_scale(sx, sy) @ bezier.m_translate(-splines_bbox[0], -splines_bbox[1])
    splines = bezier.transform_splines(splines, m)
    return glyph | {"splines": splines}
