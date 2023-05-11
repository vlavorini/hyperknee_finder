# based on https://stackoverflow.com/a/43551628/4120777
import numpy as np


def z_cross_product(a, b, c):
    return (a[0] - b[0]) * (b[1] - c[1]) - (a[1] - b[1]) * (b[0] - c[0])


def is_convex(vertices):
    # if len(vertices) < 4:
    #     return True
    signs = [z_cross_product(a, b, c) for a, b, c in zip(vertices[2:], vertices[1:], vertices)]
    return signs


def calc_signs(x, y):
    vertices = [(a, b) for a, b in zip(x, y)]
    signs = is_convex(vertices)
    return signs


def calc_pseudo_convexity(x, y, z):
    all_signs = []
    # for each slice in X
    for i, _ in enumerate(x):
        z_ = z[i]
        signs = calc_signs(y, z_)
        all_signs.extend(signs)

    # for each slice in Y
    for i, _ in enumerate(y):
        z_ = z[:, i]
        signs = calc_signs(x, z_)
        all_signs.extend(signs)
    return np.sign(np.mean(all_signs))
