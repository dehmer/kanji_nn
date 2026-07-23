import numpy as np
from functools import partial
from itertools import accumulate
from kanji_nn.plot import strokes_plot
from . import dtw_align, densify


def find_cut(s, patterns):
    """
    Matches stroke signature against patterns and calls
    extractor to get cut index.
    Note: Order of patterns matters.
    """
    cut = None
    for ps, e in patterns:
        if len(s) < len(ps): continue # not enough segments for predicates
        cut = e(s) if all(p(s[i]) for i, p in enumerate(ps)) else None
        if cut != None:
            cut = int(cut)
            break
    return cut


def vg_trace_align(stroke):

    # Add interpolated points to reference
    # to make up for sparse point distribution.
    # max_ds := target delta arc length:
    hs = stroke.features["raw:s"]
    max_ds = hs[-1] / stroke.n_points
    reference = stroke.props["wkb"]
    densified = densify(reference, max_ds)
    path, _ = dtw_align(stroke.xy, densified)

    A = 0 # path column index: (handwritten) stroke
    B = 1 # path column index: reference
    # True/1:  consecutive A-indices hit same B-index => dirty
    # False/0: strict monotonic advancement           => clean
    mask = np.diff(path[:, B]) == 0

    # Create "stroke signature" with respective run lengths
    # and cumulative run length sums for T/F groups:
    edges = np.r_[0, np.flatnonzero(mask[1:] != mask[:-1]) + 1, len(mask)]
    signature = list(zip(mask[edges[:-1]], np.diff(edges)))

    signature = [
        (tag, rl, path[cs, A]) for (tag, rl), cs in zip(
            signature,
            accumulate(length for _, length in signature)
        )
    ]

    tag = lambda entry: 'D' if entry[0] else 'C'
    readable = lambda s, sep: sep.join([f"{tag(g)}-{g[1]}:{g[2]}" for g in s])

    # Predicates:
    def c(): return lambda e: e[0] == False                 # clean, unbounded
    def d(): return lambda e: e[0] == True                  # dirty, unbounded
    def cgt(n): return lambda e: e[0] == False and e[1] > n # clean, greater than

    # print(readable(signature, " -> "))
    head_cut = find_cut(signature, [
        ([c(), d(), c()], lambda s: s[1][2] + 1),
        ([d(), c(), d()], lambda s: s[0][2] - 1),
        ([d(), c()], lambda s: s[0][2] + 1), # 校/0, 字/1
        ([c(), d()], lambda s: 0),           # 森/11, 林/2
    ])

    # print(readable(signature[::-1], " <- "))
    tail_cut = find_cut(signature[::-1], [
        ([d(), c(), d()], lambda s: s[1][2] + 2),
        # TODO: Needs a reliable lower bound: 16 is good for Kanken (10)
        ([cgt(16), d(), c()], lambda s: s[0][2] + 1), # EOS
        ([c(), d(), c()], lambda s: s[2][2] + 1),
        ([c(), d()], lambda s: stroke.n_points), # 校/0, 字/1
        ([d(), c()], lambda s: s[1][2] + 1),     # 森/11, 林/2
    ])

    head_cut = 0 if head_cut == None else head_cut
    tail_cut = stroke.n_points if tail_cut == None else tail_cut
    cuts = (head_cut, tail_cut)

    return stroke.clone(props={"cuts": cuts})
