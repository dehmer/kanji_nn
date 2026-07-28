from functools import reduce

def compose(*functions):
    """Composes functions right to left."""
    return lambda x: reduce(lambda acc, f: f(acc), reversed(functions), x)
