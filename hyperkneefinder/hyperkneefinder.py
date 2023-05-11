import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional
from sklearn.linear_model import LinearRegression
from .pseudo_convexity import calc_pseudo_convexity


class HyperKneeFinder:
    """
    hyperKnee point finder.
    TIt's about a tool for optimizing two inter-dependent parameters
    """
    __xx = None
    __yy = None
    __independent_data = None
    __dependent_data = None
    __model = None
    __translated_plane_data = None
    Z = None

    def __init__(self, data_x: Union[list, np.ndarray], data_y: Union[list, np.ndarray],
                 data_z: np.ndarray,
                 name_x: Optional[str] = None, name_y: Optional[str] = None,
                 clean_data: bool = True):
        self.X = data_x
        self.Y = data_y
        if name_x is not None and name_y is not None:
            self.name_x = name_x
            self.name_y = name_y
        if clean_data:
            self.__clean_data(data_z)
        else:
            self.Z = data_z.T

    def __clean_data(self, data_z: np.ndarray, threshold: float = 0.8):
        """
        Clean the data by ignoring the simil-plateau at the end of the matrix
        """
        max_distances = np.zeros_like(data_z)

        for (i, j), val in np.ndenumerate(data_z):
            if i == 0 and j == 0:
                # ignoring the very first point because of too few neighbors
                continue

            # for each point we will evaluate the points in the neighbors

            # must account for the slicing where i or j are equal to 0, otherwise such slices
            # will result in an empty array
            if i == 0:
                i_start_at = 0
            else:
                i_start_at = i - 1
            if j == 0:
                j_start_at = 0
            else:
                j_start_at = j - 1

            # the slice/matrix around the central point
            distances = np.abs(data_z[i_start_at:i + 2, j_start_at:j + 2] - data_z[i, j])

            # for each point, we set the maximum distance to its neighbors
            max_distances[i, j] = distances.max()

        thr = max_distances.max() * threshold
        to_delete = max_distances < thr

        # if all the values in a column are equal to the number of lines, i.e. all of them are True
        cols2delete = to_delete.sum(axis=0) == to_delete.shape[0]

        # will delete all the columns after the first where all the values are True
        last_good_column = np.argwhere(cols2delete)[0][0] - 1
        new_z = data_z[:, :last_good_column]
        self.Y = self.Y[:last_good_column]

        # same for rows
        rows2delete = to_delete.sum(axis=1) == to_delete.shape[1]

        # will delete all the columns after the first where all the values are True
        last_good_row = np.argwhere(rows2delete)[0][0] - 1
        self.Z = new_z[:last_good_row, :].T
        self.X = self.X[:last_good_row]

    def __get_meshgrid(self):
        self.__xx, self.__yy = np.meshgrid(self.X, self.Y)
        return self.__xx, self.__yy

    def __reshape_data(self):
        """
        Shape the data to be fed to the Linear Model
        """
        if self.__independent_data is None or self.__dependent_data is None:
            xx, yy = self.__get_meshgrid()

            independent_data = np.array(
                [[xx[i, j], yy[i, j]] for i in range(xx.shape[0]) for j in range(xx.shape[1])]).flatten().reshape(
                (len(self.X) * len(self.Y), 2))

            dependent_data = self.Z.flatten()
            self.__independent_data = independent_data
            self.__dependent_data = dependent_data

        return self.__independent_data, self.__dependent_data

    def __get_fitted_plane_model(self):
        """Fit the Linear Model to find the plane which minimize the variance"""
        if self.__model is None:
            independent_data, dependent_data = self.__reshape_data()
            model = LinearRegression()
            model = model.fit(independent_data, dependent_data)
            self.__model = model
        return self.__model

    def __translate_plane(self):
        """
        Translate the fitted plane to let it pass by the very first point, p0
        """
        if self.__translated_plane_data is None:
            model = self.__get_fitted_plane_model()

            # the vector normal to the fitted plane
            v_n = np.array([model.coef_[0], model.coef_[1], -1])

            # the pseudo-convexity of the curve
            p_conv = calc_pseudo_convexity(self.X, self.Y, self.Z.T)

            # the first point of the curve, shifted by a good amount
            p0_shifted = [self.X[0], self.Y[0], self.Z[0, 0] - p_conv * (
                        self.Z.max() - self.Z.min())]

            # the projection of the normal vector to the vector passing by the shifted first point of the curve
            ps_intercept = np.sum(
                v_n * p0_shifted)
            factor_x = model.coef_[0]
            factor_y = model.coef_[1]
            new_intercept = -ps_intercept

            self.__translated_plane_data = factor_x, factor_y, new_intercept, v_n
        return self.__translated_plane_data

    def __cal_distance(self):
        """Calculate the distance from each data point to the translated plane"""
        factor_x, factor_y, new_intercept, v_n = self.__translate_plane()
        independent_data, dependent_data = self.__reshape_data()

        dist_1 = independent_data * v_n[:2]
        dist_2 = dependent_data * v_n[2]

        dist_comb = np.concatenate((dist_1, np.expand_dims(dist_2, axis=1)), axis=1)
        dist_tot = np.abs(np.sum(dist_comb, axis=1) + new_intercept)

        return dist_tot

    def __max_dist_from_plane(self):
        """
        The maximum distance of the given data to the translated plane
        """
        dist_tot = self.__cal_distance()

        knee_point_at = np.argmax(dist_tot)

        return knee_point_at

    def get_hypoerkee_point(self, printout=True):
        """
        The x, y coordinates of the hyper knee point
        """
        independent_data, dependent_data = self.__reshape_data()

        knee_point_at = self.__max_dist_from_plane()
        hk_x, hk_y = independent_data[knee_point_at]
        if printout:
            if self.name_x is not None:
                print(f"HyperKnee is at {self.name_x}={hk_x}, {self.name_y}={hk_y}")
            else:
                print(f"HyperKnee is at {hk_x}, {hk_y}")

        return hk_x, hk_y

    def __calculate_plane_points(self):
        """A set of points belonging to the translated plane, useful for plotting"""
        factor_x, factor_y, new_intercept, _ = self.__translate_plane()

        xp = np.tile(np.linspace(min(self.X), max(self.X), 61), (61, 1))
        yp = np.tile(np.linspace(min(self.Y), max(self.Y), 61), (61, 1)).T

        zp = factor_x * xp + factor_y * yp + new_intercept

        return xp, yp, zp

    def visualise_hyperknee(self):
        knee_point_at = self.__max_dist_from_plane()
        xx, yy = self.__get_meshgrid()
        independent_data, dependent_data = self.__reshape_data()
        xp, yp, zp = self.__calculate_plane_points()

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
