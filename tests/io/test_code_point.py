import pytest
from kanji_nn.io.code_point import groups

@pytest.mark.parametrize(
    "literal, expected",
    [
        pytest.param("0", ["DIGIT"], id="digit [b699]"),
        pytest.param("A", ["ROMAJI"], id="romaji [a999]"),
        pytest.param("ア", ["KATAKANA"], id="katakana [4f09]"),
        pytest.param("ァ", ["KATAKANA", "SUTEGATA"], id="katakana/sutegata [89fa]"),
        pytest.param("バ", ["KATAKANA", "DAKUTEN"], id="katakana/dakuten [4ffb]"),
        pytest.param("パ", ["KATAKANA", "HANDAKUTEN"], id="katakana/handakuten [a995]"),
        pytest.param("あ", ["HIRAGANA"], id="hiragana [2f45]"),
        pytest.param("ぁ", ["HIRAGANA", "SUTEGATA"], id="hiragana/sutegata [43f9]"),
        pytest.param("ば", ["HIRAGANA", "DAKUTEN"], id="hiragana/dakuten [a3f5]"),
        pytest.param("ぱ", ["HIRAGANA", "HANDAKUTEN"], id="hiragana/handakuten [67ed]"),
        pytest.param("学", ["KANJI", "JOYO", "KANKEN_10", "KANKEN"], id="kanken-10 [4aaa]"),
        pytest.param("繁", ["JINMEIYO"], id="jinmeiyo [b021]"),
    ],
)
def test_groups(literal, expected):
    assert groups(literal) == expected