import bitstring


def unpack(chunk, fields):
    fmt = ",".join(fields.values())
    keys = [k for k, v in fields.items() if "pad" not in v]
    s = bitstring.ConstBitStream(bytes=chunk)
    return dict(zip(keys, s.unpack(fmt)))
