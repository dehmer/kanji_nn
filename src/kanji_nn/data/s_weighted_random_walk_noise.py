from .paced_sigma import paced_sigma
from .random_walk_delta_noise import random_walk_delta_noise

def s_weighted_random_walk_noise(tensor, sigma_base=0.03, k=1.0, rho=0.85, rng=None):
    """

    sigma_base — the base noise amplitude in (Δx, Δy) units, before any
    pace weighting is applied. This is your master volume knob: it scales
    everything uniformly regardless of k. If everything looks too aggressive
     — including segments that should be nearly untouched — this is the
     first thing to pull down, since k only redistributes noise across
     fast/slow regions, it doesn't reduce the total amount in the system.


    k — controls which regions get more noise, and how sharply.
    Sign picks the direction (positive → more noise on large-pace/fast/straight
    segments; negative → more on small-pace/slow/curved segments),
    magnitude controls contrast (small |k| ≈ mild tilt toward one end;
    large |k| ≈ almost all the noise budget dumped onto one type of segment,
    the other left nearly clean). k=0 recovers plain uniform noise.
    Note k doesn't add noise, it reshuffles the existing sigma_base
    budget — so it's not really an independent "how much" knob, more a
    "where" knob.

    k > 0: more noise on large-pace (fast, straight) segments.
    k < 0: more noise on small-pace (slow, careful/curved) segments.
    k = 0: uniform, falls back to plain random_walk_delta_noise.


    rho — how strongly noise correlates from one point to the next.
    This is the one I'd suspect first for what you're seeing. Low
    rho → each step's noise mostly forgets the last, so consecutive points
    wobble independently around the true path (real hand tremor).
    High rho (0.85, what we used) → noise barely decays, so a random
    early "nudge in one direction" persists and compounds across many
    consecutive points. Once you integrate a long run of same-direction
    correlated noise via cumsum for reconstruction, that's not tremor
    anymore — it's a smooth, systematic bend. Enough of that in one
    region, and a gentle curve turns into an actual loop.
    """
    sigma = paced_sigma(tensor, sigma_base, k)
    return random_walk_delta_noise(tensor, sigma, rho=rho, rng=rng)
