import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse
from networkx.algorithms.bipartite import (
    from_biadjacency_matrix,
    minimum_weight_full_matching,
)
import auct
import matplotlib.pyplot as plt
# import dlib


N = 128
GRID = 0, 1024
EXPERIMENTS_NUM = 5
STD_SCALE = np.linspace(0, np.sqrt(GRID[1]))


def polar_to_cartesian(r, theta):
    return np.stack([r * np.cos(theta), r * np.sin(theta)], 1)


def get_matching_from_biadjecncy_matrix(biadjecancy_matrix: np.array):
    distance_scipy = sparse.csr_matrix(biadjecancy_matrix)
    G = from_biadjacency_matrix(distance_scipy)
    matching_dict = minimum_weight_full_matching(G)
    matching_array = np.array([x for x in list(matching_dict.items()) if x[0] < x[1]])
    matching_array[:, 1] = matching_array[:, 1] - N
    return matching_array


def naive_assignment(biadjecancy_matrix: np.array, minimize=True):
    switch_assign = False
    if biadjecancy_matrix.shape[0] > biadjecancy_matrix.shape[1]:
        biadjecancy_matrix = biadjecancy_matrix.transpose()
        switch_assign = True

    if minimize:
        biadjecancy_matrix = biadjecancy_matrix.max() - biadjecancy_matrix

    dtype = [("ind", int), ("val", int)]
    assignments = np.full(biadjecancy_matrix.shape[0], -1, int)
    for obj in range(biadjecancy_matrix.shape[0]):
        not_taken = np.setdiff1d(
            np.arange(biadjecancy_matrix.shape[1]), assignments, True
        )
        assignment = np.partition(
            np.array(list(enumerate(biadjecancy_matrix[obj])), dtype=dtype)[not_taken],
            -1,
            order="val",
        )[-1][0]
        assignments[obj] = assignment

    if switch_assign:
        assignments[:, 0], assignments[:, 1] = (
            assignments[:, 1],
            assignments[:, 0].copy(),
        )

    return np.array(list(enumerate(assignments)))


def main():
    np.random.seed(0)

    apx_errors_avg = list()
    apx_errors_std = list()
    for std in STD_SCALE:
        apx_errors = list()
        for experiment in range(EXPERIMENTS_NUM):
            drivers, passengers = np.rollaxis(np.random.uniform(*GRID, (2, N, 2)), 0)
            distances = cdist(drivers, passengers)

            noise = np.random.normal(0, std, N), np.random.uniform(0, np.pi, N)
            fake_passengers = passengers + polar_to_cartesian(*noise)
            noise_distances = cdist(drivers, fake_passengers)

            matching_array_without_noise = get_matching_from_biadjecncy_matrix(
                distances
            )
            matching_array_with_noise = naive_assignment(distances)

            dist_without_noise = distances[
                tuple(np.transpose(matching_array_without_noise))
            ].sum()
            dist_with_noise = distances[
                tuple(np.transpose(matching_array_with_noise))
            ].sum()
            apx_errors.append(
                np.abs(dist_without_noise - dist_with_noise) / dist_without_noise
            )
        apx_errors_avg.append(np.average(apx_errors))
        apx_errors_std.append(np.std(apx_errors))

    plt.errorbar(
        STD_SCALE, apx_errors_avg, yerr=apx_errors_std, label="hungarian on noise"
    )

    plt.xlabel("Std of noise")
    plt.ylabel("Approximation error of solution on noised problem")

    plt.show()


if __name__ == "__main__":
    main()
