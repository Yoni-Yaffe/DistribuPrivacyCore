import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse
from networkx.algorithms.bipartite import from_biadjacency_matrix, minimum_weight_full_matching
# import dlib


N = 5
GRID = 0, 1024


def polar_to_cartesian(r, theta):
    return np.stack([r * np.cos(theta), r * np.sin(theta)], 1)

def get_matching_from_biadjecncy_matrix(biadjecancy_matrix: np.array):
    distance_scipy = sparse.csr_matrix(biadjecancy_matrix)
    G = from_biadjacency_matrix(distance_scipy)
    matching_dict = minimum_weight_full_matching(G)
    matching_array = np.array([x for x in list(matching_dict.items()) if x[0] < x[1]])
    matching_array[:, 1] = matching_array[:, 1] - N
    return matching_array

def main():
    np.random.seed(0)
    drivers, passengers = np.rollaxis(np.random.uniform(*GRID, (2, N, 2)), 0)
    distances = cdist(drivers, passengers)

    noise = np.abs(np.random.normal(0, np.sqrt(GRID[1]), N)), np.random.uniform(0, 2 * np.pi, N)
    fake_passengers = passengers + polar_to_cartesian(*noise)
    noise_distances = cdist(drivers, fake_passengers)

    matching_array_without_noise = get_matching_from_biadjecncy_matrix(distances)
    matching_array_with_noise = get_matching_from_biadjecncy_matrix(noise_distances)
    dist_without_noise = distances[tuple(np.transpose(matching_array_without_noise))].sum()
    dist_with_noise = distances[tuple(np.transpose(matching_array_with_noise))].sum()
    print(dist_without_noise - dist_with_noise)
    x = 1


if __name__ == "__main__":
    main()
