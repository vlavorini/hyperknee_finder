# based on https://stackoverflow.com/a/43551628/4120777
import numpy as np
"""
This tool calculate the pseudo-convexity of a 3-d complex curve.
The difference from the usual convex definition (https://en.wikipedia.org/wiki/Convex_function) is that in our case,
we define as quasi-convex a function that can be non-convex locally, but that globally can be approximated
with a convex function.

"""


def z_cross_product(a, b, c):
    """
    The 'z' component of the cross product
    """
    return (a[0] - b[0]) * (b[1] - c[1]) - (a[1] - b[1]) * (b[0] - c[0])


def calc_dc(x: np.ndarray, y: np.ndarray):
    """
    This function calculate the "degree of convexity", here defined as the z component
    of the cross product of consecutive triplets of vertices
    """
    vertices = [(a, b) for a, b in zip(x, y)]
    d_c = [z_cross_product(a, b, c) for a, b, c in zip(vertices[2:], vertices[1:], vertices)]
    return d_c


def calc_pseudo_convexity(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    This function calculate the pseudo-convexity of all the slices in the 3-d curve, then returns the sign (+-1)
    of the average of the pseudo-convexities
    """
    all_dc = []
    # for each slice in X
    for i, _ in enumerate(x):
        z_ = z[i]
        dc = calc_dc(y, z_)
        all_dc.extend(dc)

    # for each slice in Y
    for i, _ in enumerate(y):
        z_ = z[:, i]
        dc = calc_dc(x, z_)
        all_dc.extend(dc)
    return np.sign(np.mean(all_dc))
