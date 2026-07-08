from .paced_sigma import paced_sigma
from .random_walk_delta_noise import random_walk_delta_noise

def s_weighted_random_walk_noise(tensor, sigma_base=0.03, k=1.0, rho=0.85, rng=None):
    sigma = paced_sigma(tensor, sigma_base, k)
    return random_walk_delta_noise(tensor, sigma, rho=rho, rng=rng)
