import pytest
from kanji_nn.etlcdb import border_touches

size = (128, 127) # width (x), height (y)

# Feature bounding boxes are
#   min_x/min_y: inclusive -> [  0,   0]
#   max_x/max_y: exclusive -> [128, 127]

@pytest.mark.parametrize(
    "feature, margin, expected",
    [
        pytest.param([  1,   1, 127, 126], 0, [  0,   0,   0,   0], id="[830d]"),
        pytest.param([  0,   1, 127, 126], 0, [127,   0,   0,   0], id="[0cbe]"),
        pytest.param([  1,   0, 127, 126], 0, [  0, 126,   0,   0], id="[48b3]"),
        pytest.param([  1,   1, 128, 126], 0, [  0,   0, 127,   0], id="[b2ff]"),
        pytest.param([  1,   1, 127, 127], 0, [  0,   0,   0, 126], id="[b6a2]"),
        pytest.param([  0,   0, 128, 127], 0, [128, 127, 128, 127], id="[40f3]"),

        pytest.param([  3,   3, 125, 124], 2, [  0,   0,   0,   0], id="[d9d0]"),
        pytest.param([  2,   3, 125, 124], 2, [125,   0,   0,   0], id="[1084]"),
        pytest.param([  3,   2, 125, 124], 2, [  0, 124,   0,   0], id="[4168]"),
        pytest.param([  3,   3, 126, 124], 2, [  0,   0, 125,   0], id="[b673]"),
        pytest.param([  3,   3, 125, 125], 2, [  0,   0,   0, 124], id="[9c3c]"),
        pytest.param([  0,   0, 128, 127], 2, [128, 127, 128, 127], id="[410d]"),
        pytest.param([  1,   1, 127, 126], 2, [127, 126, 127, 126], id="[b97e]"),
        pytest.param([  2,   2, 126, 125], 2, [126, 125, 126, 125], id="[ba1f]"),

        pytest.param([  0,   1,  10, 126], 0, [ 10,   0,   0,   0], id="[2cae]"),
        pytest.param([  1,   0, 127,  10], 0, [  0,  10,   0,   0], id="[4939]"),
        pytest.param([118,   1, 128, 126], 0, [  0,   0,  10,   0], id="[92cd]"),
        pytest.param([  1, 117, 127, 127], 0, [  0,   0,   0,  10], id="[6e02]"),
    ],
)
def test_border_touches(feature, margin, expected):
    actual = border_touches(size, feature, margin)
    assert actual == expected