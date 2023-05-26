import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional
from sklearn.linear_model import LinearRegression


def put_in_shape(x, y, z):
    xx, yy = np.meshgrid(x, y)

    independent_data = np.array(
        [[xx[i, j], yy[i, j]] for i in range(xx.shape[0]) for j in range(xx.shape[1])]).flatten().reshape(
        (len(x) * len(y), 2))

    dependent_data = z.flatten()
    return independent_data, dependent_data


class HyperKneeFinder:
    """
    hyperKnee point finder.
    TIt's about a tool for optimizing two inter-dependent parameters
    """
    __independent_data = None
    __dependent_data = None
    __independent_data_cut = None
    __dependent_data_cut = None
    __model = None
    __shifted_plane_data = None
    Z = None

    def __init__(self, data_x: Union[list, np.ndarray], data_y: Union[list, np.ndarray],
                 data_z: np.ndarray,
                 name_x: Optional[str] = None, name_y: Optional[str] = None,
                 clean_data: bool = True, clean_threshold: float = 0.75):
        self.X = data_x
        self.Y = data_y
        if name_x is not None and name_y is not None:
            self.name_x = name_x
            self.name_y = name_y
        if clean_data:
            self.__clean_data(data_z, threshold=clean_threshold)
        else:
            self.Z = data_z.T

    def __clean_data(self, data_z: np.ndarray, threshold: float):
        """
        Clean the data by ignoring the simil-plateau at the end of the matrix.

        The threshold represents the limit under which two points are considered belonging to the same simil-plateau.
        Note that only rows and columns where ALL the values falls behind that threshold will be deleted,
        i.e. two points in the same axis with similar values will be kept if the other points in the same row/column
        are not set as to be deleted.
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

    def __reshape_data(self, all_data: bool = True):
        """
        Shape the data as independent (X and Y) and dependent (Z) for convenience of the Linear model
        which will be used.

        if all_data is false, only the central part of the data is considered. This is
        used for finding the pseudo-convexity of the curve
        """
        if all_data:
            if self.__independent_data is None or self.__dependent_data is None:
                self.__independent_data, self.__dependent_data = put_in_shape(self.X, self.Y, self.Z)

            return self.__independent_data, self.__dependent_data
        else:
            if self.__independent_data_cut is None or self.__dependent_data_cut is None:
                # the central part of the data is the data between the 1/4 and the 2/4 of the total data,
                # in both axes
                start_x = int(len(self.X) / 4)
                stop_x = int(3 * len(self.X) / 4)
                slice_x = self.X[start_x:stop_x]

                start_y = int(len(self.Y) / 4)
                stop_y = int(3 * len(self.Y) / 4)
                slice_y = self.Y[start_y:stop_y]

                slice_z = self.Z.T[start_x:stop_x, start_y:stop_y]
                self.__independent_data_cut, self.__dependent_data_cut = put_in_shape(slice_x, slice_y, slice_z)

            return self.__independent_data_cut, self.__dependent_data_cut

    def get_fitted_plane_model(self):
        """Fit with a Linear Model to find the plane which minimize the variance"""
        if self.__model is None:
            independent_data, dependent_data = self.__reshape_data()
            model = LinearRegression()
            model = model.fit(independent_data, dependent_data)
            self.__model = model
        return self.__model

    def __shift_plane(self):
        """
        This function will shift the fitted plane of a decent amount in the direction of the convexity.
        """
        if self.__shifted_plane_data is None:
            model = self.get_fitted_plane_model()

            # the vector normal to the fitted plane
            v_n = np.array([model.coef_[0], model.coef_[1], -1])

            # the pseudo-convexity of the curve
            part_dists = self.__cal_distance(all_data=False)
            p_conv = -1 * np.sign(np.mean(part_dists))

            # the first point of the curve, shifted by a good amount
            p0_shifted = [self.X[0], self.Y[0], self.Z[0, 0] - p_conv * (
                        self.Z.max() - self.Z.min())]

            # the shifted plane has the same factor for x and y,
            # but the intercept is given by the projection of the normal vector
            # to the vector passing by the shifted first point of the curve
            new_intercept = - np.sum(
                v_n * p0_shifted)
            factor_x = model.coef_[0]
            factor_y = model.coef_[1]

            self.__shifted_plane_data = factor_x, factor_y, new_intercept, v_n
        return self.__shifted_plane_data

    def __get_plane_data(self,  what: str = 'shifted'):
        """
        Getting the data for the plane: a, b, intercept, normal vector
        """
        if what == 'shifted':
            return self.__shift_plane()
        elif what == 'fitted':
            model = self.get_fitted_plane_model()
            v_n = np.array([model.coef_[0], model.coef_[1], -1])
            return model.coef_[0], model.coef_[1], model.intercept_, v_n
        else:
            raise ValueError(f"unexpected value for 'what' parameter: {what}")

    def __cal_distance(self, all_data: bool = True):
        """Calculate the distance from each data point to the proper plane.

        if all_data is false, only the central part of the data is considered,
        and the plane to which calculate the distance will be the fitted one, not the shifted one.
        This is used for finding the pseudo-convexity of the curve
        """

        if all_data:
            independent_data, dependent_data = self.__reshape_data()
            factor_x, factor_y, intercept, v_n = self.__get_plane_data()
        else:
            independent_data, dependent_data = self.__reshape_data(all_data=all_data)
            factor_x, factor_y, intercept, v_n = self.__get_plane_data(what='fitted')

        dist_1 = independent_data * v_n[:2]
        dist_2 = dependent_data * v_n[2]

        dist_comb = np.concatenate((dist_1, np.expand_dims(dist_2, axis=1)), axis=1)
        dist_tot = np.sum(dist_comb, axis=1) + intercept

        if all_data:
            return np.abs(dist_tot)
        else:
            return dist_tot

    def __max_dist_from_plane(self):
        """
        The maximum distance of the given data to the shifted plane
        """
        dist_tot = self.__cal_distance(all_data=True)

        knee_point_at = np.argmax(dist_tot)

        return knee_point_at

    def get_hyperkee_point(self, printout: bool = False) -> (float, float):
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

    def __calculate_plane_points(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """A set of points belonging to the shifted plane, useful for plotting"""
        factor_x, factor_y, new_intercept, _ = self.__shift_plane()

        xp = np.tile(np.linspace(min(self.X), max(self.X), 61), (61, 1))
        yp = np.tile(np.linspace(min(self.Y), max(self.Y), 61), (61, 1)).T

        zp = factor_x * xp + factor_y * yp + new_intercept

        return xp, yp, zp

    def visualise_hyperknee(self) -> None:
        """
        Create a plot with:
        - the data, cleaned if required (clean_data=True in initialization)
        - the plane to which calculate the distances
        - the hyper-knee point
        """
        knee_point_at = self.__max_dist_from_plane()
        xx, yy = np.meshgrid(self.X, self.Y)
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
