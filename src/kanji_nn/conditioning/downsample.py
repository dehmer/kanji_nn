from .rdp_budget import rdp_to_budget_flat
from .point_count import point_count

def downsample(strokes, max_budget=256):
    # Downsample only if necessary:
    target_total = max_budget - len(strokes)
    count_total = point_count(strokes)
    if count_total > target_total:
        strokes, _, count_total = rdp_to_budget_flat(strokes, target_total=target_total)
        assert count_total <= target_total

    return strokes
