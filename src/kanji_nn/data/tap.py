def tap(fn):
    def inner(x):
        fn(x)
        return x
    return inner
