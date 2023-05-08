import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional
import sklearn.linear_model


class HyperKneeFinder:
    """
    hyperKnee point finder.
    TIt's about a tool for optimizing two inter-dependent parameters
    """
    xx = None
    yy = None
    independent_data = None
    dependent_data = None
    model = None
    translated_plane_data = None

    def __init__(self, data_x: Union[list, np.ndarray], data_y: Union[list, np.ndarray],
                 data_z: Union[list, np.ndarray],
                 name_x: Optional[str] = None, name_y: Optional[str] = None):
        if len(data_x) != len(data_y) or len(data_x) != len(data_z) or len(data_y) != len(data_z):
            raise ValueError("Input arrays must be of the same length.")
            
        self.X = data_x
        self.Y = data_y
        self.Z = data_z
        if name_x is not None and name_y is not None:
            self.name_x = name_x
            self.name_y = name_y

    def reshape_data(self):
        """
        Shape the data to be fed to the Linear Model
        """
        if self.independent_data is None or self.dependent_data is None:
            xx, yy = self.get_meshgrid()

            independent_data = np.array(
                [[xx[i, j], yy[i, j]] for i in range(len(self.X)) for j in range(len(self.Y))]).flatten().reshape(
                (len(self.X) * len(self.Y), 2))

            dependent_data = self.Z.flatten()
            self.independent_data = independent_data
            self.dependent_data = dependent_data

        return self.independent_data, self.dependent_data

    def get_fitted_plane(self):
        """Fit the Linear Model to find the plane which minimize the variance"""
        if self.model is None:
            independent_data, dependent_data = self.reshape_data()
            model = sklearn.linear_model.LinearRegression()
            model = model.fit(independent_data, dependent_data)
            self.model = model
        return self.model

    def translate_plane(self):
        """
        Translate the fitted plane to let it pass by the very first point, p0
        """
        if self.translated_plane_data is None:
            model = self.get_fitted_plane()

            v_n = np.array([model.coef_[0], model.coef_[1], -1])
            p0 = [self.X[0], self.Y[0], self.Z[0, 0]]
            ps_intercept = np.sum(v_n * p0)
            factor_x = model.coef_[0]
            factor_y = model.coef_[1]
            new_intercept = -ps_intercept

            self.translated_plane_data = factor_x, factor_y, new_intercept, v_n
        return self.translated_plane_data

    def cal_distance(self):
        """Calculate the distance from each data point to the translated plane"""
        factor_x, factor_y, new_intercept, v_n = self.translate_plane()
        independent_data, dependent_data = self.reshape_data()

        dist_1 = independent_data * v_n[:2]
        dist_2 = dependent_data * v_n[2]

        dist_comb = np.concatenate((dist_1, np.expand_dims(dist_2, axis=1)), axis=1)
        dist_tot = np.abs(np.sum(dist_comb, axis=1) + new_intercept)

        return dist_tot

    def max_dist_from_plane(self):
        """
        The maximum distance of the given data to the translated plane
        """
        dist_tot = self.cal_distance()

        knee_point_at = np.argmax(dist_tot)

        return knee_point_at

    def get_hypoerkee_point(self, printout=True):
        """
        The x, y coordinates of the hyper knee point
        """
        independent_data, dependent_data = self.reshape_data()

        knee_point_at = self.max_dist_from_plane()
        hk_x, hk_y = independent_data[knee_point_at]
        if printout:
            if self.name_x is not None:
                print(f"HyperKnee is at {self.name_x}={hk_x}, {self.name_y}={hk_y}")
            else:
                print(f"HyperKnee is at {hk_x}, {hk_y}")

        return hk_x, hk_y

    def get_meshgrid(self):
        if self.xx is None or self.yy is None:
            self.xx, self.yy = np.meshgrid(self.X, self.Y)
        return self.xx, self.yy

    def calculate_plane_points(self):
        """A set of points belonging to the translated plane, useful for plotting"""
        factor_x, factor_y, new_intercept, _ = self.translate_plane()

        xp = np.tile(np.linspace(min(self.X), max(self.X), 61), (61, 1))
        yp = np.tile(np.linspace(min(self.Y), max(self.Y), 61), (61, 1)).T

        zp = factor_x * xp + factor_y * yp + new_intercept

        return xp, yp, zp

    def visualise_hyperknee(self):
        knee_point_at = self.max_dist_from_plane()
        xx, yy = self.get_meshgrid()
        independent_data, dependent_data = self.reshape_data()
        xp, yp, zp = self.calculate_plane_points()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, self.Z, linewidth=1, antialiased=True, alpha=0.5)

        if self.name_x is not None:
            ax.set_xlabel(self.name_x)
            ax.set_ylabel(self.name_y)
        else:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.scatter(*(independent_data[knee_point_at]), dependent_data[knee_point_at], c='b', s=30,
                   label='knee point')
        ax.plot_surface(xp, yp, zp, alpha=0.5)
        plt.legend()
        plt.show()
