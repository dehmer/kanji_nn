

def flag_label_mismatch(glyph):
    """
    Flag glyphs where literal does not match the actual image.
    The corresponsing ids are collected during manual inspection/eyeballing.
    """
    ids = [
        "a98b65e7-f2ce-4c2c-b354-6d93658993f6",

        # unusable but still escaping rejection:
        "04c3a630-985c-40a8-9e29-72a26f11afcd",
        "5c507fd6-9c38-4982-b3b8-ffdc0f20807d",
        "7648c656-8493-4c1c-8eeb-1f06af510ff6",
        "0746678f-f0a6-41ba-a491-a967fa0288f5",
        "75189492-53e1-4b5d-9ada-c3e205ea2bf2",
        "a0fb2291-e3ee-48d1-9c83-381d12518647",
        "a66b08ee-38e0-4905-a4ce-452a0c2b6e11",
        "b547e2ea-f14d-472b-b942-e453828183aa",
        "b662f4a7-1947-4894-addc-38c9cbecf844",
        "c094df8e-c157-4cc0-84f0-a24a115fa5f2",
        "dc5beddf-59a6-4d0a-a7a4-66e4c074039d"
    ]

    if glyph["id"] in ids:
        return glyph | {"skip": True, "reason": "label mismatch"}
    else:
        return glyph
