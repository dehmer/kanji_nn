import numpy as np

# 1. Feature Combination & Z-Score Standardization Engine
def compute_combined_signal(stroke, w1, w2, w3):
    """
    Transforms and standardizes features so they align to a positive mean shift.
    stroke_data: dict containing 'pressure', 'central_speed', and 'ds' arrays.
    """

    # Transform so all features increase during the dirty pre-pen-up zone
    f1 = 1.0 - stroke.features['P']
    f2 = stroke.features['c_speed']
    f3 = stroke.features["ds"]

    # Equalize scales via Z-score normalization
    f1_n = (f1 - np.mean(f1)) / (np.std(f1) + 1e-6)
    f2_n = (f2 - np.mean(f2)) / (np.std(f2) + 1e-6)
    f3_n = (f3 - np.mean(f3)) / (np.std(f3) + 1e-6)

    # Return weighted sum signal
    return (w1 * f1_n) + (w2 * f2_n) + (w3 * f3_n)
