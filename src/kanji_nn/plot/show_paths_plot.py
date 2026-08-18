from .paths_plot import paths_plot

path_fn = lambda s: s.sticky["path"]

def show_paths_plot(path_fn=path_fn, show_obb=True, show_badges=True, show_quads=False):
    paths = []

    def inner(stroke):
        nonlocal paths
        paths.append(path_fn(stroke))

        if len(paths) == stroke.stroke_count:
            paths_plot(paths, show_obb=show_obb, show_badges=show_badges, show_quads=show_quads)
            paths = []

        return stroke

    return inner