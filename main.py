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
from tqdm import tqdm
from datetime import datetime

SAVE = True
N = 128  # Drivers
M = N * 2  # Passengers
GRID = 0, 1024
EXPERIMENTS_NUM = 50
STD_SCALE = np.linspace(0, GRID[1] / 10, num=10)


def unif(loc: float = 0.0, scale: float = 1.0, size=None) -> float:
    # std = sqrt([low - high] ** 2 / 12) -> |low - high| / 2 = std * sqrt(3)
    relative_boundary = scale * np.sqrt(3)
    return np.random.uniform(loc - relative_boundary, loc + relative_boundary, size)


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


def naive_assignment_random(biadjecancy_matrix: np.array, minimize=True):
    K = 3
    switch_assign = False
    if biadjecancy_matrix.shape[0] > biadjecancy_matrix.shape[1]:
        biadjecancy_matrix = biadjecancy_matrix.transpose()
        switch_assign = True

    assignments = np.full(biadjecancy_matrix.shape[0], -1, int)
    indexes = list(range(biadjecancy_matrix.shape[0]))
    np.random.shuffle(indexes)
    for obj in indexes:
        not_taken = np.setdiff1d(
            np.arange(biadjecancy_matrix.shape[1]), assignments, True
        )
        assignments_partition = not_taken[
            np.argpartition(
                biadjecancy_matrix[obj][not_taken], min(K - 1, len(not_taken) - 1)
            )[:K]
        ]
        weights = biadjecancy_matrix[obj][assignments_partition]
        weights = 1 / weights
        # weights = np.max(weights) * 1.5 - np.array(weights)
        assignment = np.random.choice(
            assignments_partition, 1, p=(np.exp(weights) / np.sum(np.exp(weights)))
        )[0]
        assignments[obj] = assignment

    if switch_assign:
        assignments[:, 0], assignments[:, 1] = (
            assignments[:, 1],
            assignments[:, 0].copy(),
        )

    return np.array(list(enumerate(assignments)))


def naive_las_vegas(biadjecancy_matrix: np.array, minimize=True, K=5, max_iter=500):
    switch_assign = False
    if biadjecancy_matrix.shape[0] > biadjecancy_matrix.shape[1]:
        biadjecancy_matrix = biadjecancy_matrix.transpose()
        switch_assign = True

    assignments = np.full(biadjecancy_matrix.shape[0], -1, int)
    indexes = list(range(biadjecancy_matrix.shape[0]))
    np.random.shuffle(indexes)
    not_taken = np.arange(biadjecancy_matrix.shape[1])
    not_assigned = np.arange(biadjecancy_matrix.shape[0])
    i = 0
    while i < max_iter and not_assigned.size > 0:
        curr_matrix = biadjecancy_matrix[not_assigned, :][:, not_taken]
        best_k_each_driver = np.argpartition(curr_matrix, min(K - 1, curr_matrix.shape[1] - 1), axis=1)[:, :K]
        assignment_ind = np.random.randint(0, min(K, curr_matrix.shape[1]), curr_matrix.shape[0])
        assignment = best_k_each_driver[np.arange(curr_matrix.shape[0]), assignment_ind]
        vals, counts = np.unique(assignment, return_counts=True)
        unique_vals = vals[counts == 1]
        # taken = not_taken[vals[counts == 1]]
        not_conflict = np.isin(assignment, unique_vals)
        took = not_assigned[not_conflict]
        taken = not_taken[assignment[not_conflict]]
        assignments[took] = taken
        not_taken = not_taken[assignment[~ not_conflict]]
        not_assigned = not_assigned[~ not_conflict]
        i += 1
    if i >= max_iter:
        raise RuntimeError("Failed to match")
    if switch_assign:
        assignments[:, 0], assignments[:, 1] = (
            assignments[:, 1],
            assignments[:, 0].copy(),
        )

    return np.array(list(enumerate(assignments)))


def visualize_matching(passenger_loc, drivers_loc, matching, title=None):
    plt.plot(
        passenger_loc[:, 0], passenger_loc[:, 1], "o", color="red", label="passengers"
    )
    plt.plot(
        drivers_loc[:, 0], drivers_loc[:, 1], "o", color="blue", label="passengers"
    )
    for p, d in matching:
        locs = np.vstack([passenger_loc[p], drivers_loc[d]])
        plt.plot(locs[:, 0], locs[:, 1], color="black")
    if title:
        plt.title(title)
    plt.legend()
    plt.show()


def experiment_M_N():
    np.random.seed(0)
    apx_errors_avg = list()
    apx_errors_std = list()
    apx_errors_avg_naive = list()
    apx_errors_avg_naive_random = list()
    apx_errors_avg_auction = list()
    apx_errors_std_naive = list()
    apx_errors_std_naive_random = list()
    apx_errors_std_auction = list()
    M_factors = np.linspace(1, 5, 50)
    for factor in tqdm(M_factors):
        M = int(np.round(N * factor))
        apx_errors = list()
        apx_errors_without_noise = list()
        apx_errors_naive = list()
        apx_errors_naive_random = list()
        apx_errors_auction = list()
        for experiment in range(EXPERIMENTS_NUM):
            drivers, passengers = np.random.uniform(*GRID, (N, 2)), np.random.uniform(
                *GRID, (M, 2)
            )
            distances = cdist(drivers, passengers)

            matching_array_hungarian = get_matching_from_biadjecncy_matrix(distances)
            matching_array_naive = naive_assignment(distances)
            # matching_array_naive_random = naive_assignment_random(distances)

            # matching_array_with_noise_auction = auct.auction_assignment(noise_distances)
            # visualize_matching(passengers, drivers, matching_array_without_noise, title='hungarian')
            # visualize_matching(passengers, drivers, matching_array_without_noise_naive, title='naive')
            # exit(1)

            dist_hungarian = distances[
                tuple(np.transpose(matching_array_hungarian))
            ].sum()

            dist_naive = distances[tuple(np.transpose(matching_array_naive))].sum()
            apx_errors_naive.append(
                np.abs(dist_hungarian - dist_naive) / dist_hungarian
            )

            # dist_naive_random = distances[
            #     tuple(np.transpose(matching_array_naive_random))
            # ].sum()
            # apx_errors_naive_random.append(
            #     np.abs(dist_hungarian - dist_naive_random) / dist_hungarian
            # )

        # apx_errors_avg.append(np.average(apx_errors))
        # apx_errors_std.append(np.std(apx_errors))

        apx_errors_avg_naive.append(np.average(apx_errors_naive))
        apx_errors_std_naive.append(np.std(apx_errors_naive))

        # apx_errors_avg_naive_random.append(np.average(apx_errors_naive_random))
        # apx_errors_std_naive_random.append(np.std(apx_errors_naive_random))

        # apx_errors_avg_auction.append(np.average(apx_errors_auction))
        # apx_errors_std_auction.append(np.std(apx_errors_auction))

    # plt.errorbar(
    #     STD_SCALE / GRID[1], apx_errors_avg, yerr=apx_errors_std, label="hungarian on noise"
    # )
    plt.errorbar(
        M_factors, apx_errors_avg_naive, yerr=apx_errors_std_naive, label="naive"
    )

    # plt.errorbar(
    #     M_factors, apx_errors_avg_naive_random, yerr=apx_errors_std_naive_random,
    #     label="naive random"
    # )
    # plt.errorbar(
    #     STD_SCALE, apx_errors_avg_auction, yerr=apx_errors_std_auction, label="auction"
    # )
    plt.xlabel("#Passengers/#Cars")
    plt.ylabel("Approximation error of solution compared to hungarian")
    plt.legend()
    plt.title(f"Approximation error of solution as a function of Passengers ")
    plt.grid()
    if SAVE:
        filename = (
            "plots/plot" + datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
        )
        plt.savefig(filename)
    plt.show()

def calculate_min_distance():
    M = 5
    min_dist = []
    N_range = np.arange(1500, 25000, 5000)
    for N in tqdm(N_range):
        curr_min_distance = []
        for i in range(EXPERIMENTS_NUM):
            drivers, passengers = np.random.uniform(*GRID, (N, 2)), np.random.uniform(
                *GRID, (M, 2)
            )
            distances = cdist(drivers, passengers)
            # print(distances.min(axis=0).shape)
            curr_min_distance.append(distances.min(axis=0).mean() / GRID[1])
        min_dist.append(np.mean(curr_min_distance))
    plt.plot(N_range, min_dist)
    plt.grid()
    plt.show()

def main():
    np.random.seed(0)
    apx_errors_avg = list()
    apx_errors_std = list()
    apx_errors_avg_naive = list()
    apx_errors_avg_naive_random = list()
    apx_errors_avg_auction = list()
    apx_errors_std_naive = list()
    apx_errors_std_naive_random = list()
    apx_errors_std_auction = list()
    for std in tqdm(STD_SCALE):
        apx_errors = list()
        apx_errors_without_noise = list()
        apx_errors_naive = list()
        apx_errors_naive_random = list()
        apx_errors_auction = list()
        for experiment in range(EXPERIMENTS_NUM):
            drivers, passengers = np.random.uniform(*GRID, (N, 2)), np.random.uniform(
                *GRID, (M, 2)
            )
            distances = cdist(drivers, passengers)

            noise = np.random.normal(0, std, M), np.random.uniform(0, np.pi, M)
            fake_passengers = passengers + polar_to_cartesian(*noise)
            noise_matrix = np.random.normal(0, std, (N, M))
            noise_distances = cdist(drivers, fake_passengers)
            # noise_distances = np.maximum(
            #     GRID[1] / 100, cdist(drivers, passengers) + noise_matrix
            # )

            matching_array_without_noise = get_matching_from_biadjecncy_matrix(
                distances
            )
            matching_array_without_noise_naive = naive_assignment(distances)

            # matching_array_with_noise_auction = auct.auction_assignment(noise_distances)
            matching_array_with_noise = get_matching_from_biadjecncy_matrix(
                noise_distances
            )
            matching_array_with_noise_naive = naive_assignment(noise_distances)
            # matching_array_with_noise_naive_random = naive_assignment_random(
            #     noise_distances
            # )
            matching_array_with_noise_naive_random = naive_las_vegas(noise_distances, K=3)
            # test = naive_las_vegas(distances)

            # visualize_matching(passengers, drivers, matching_array_without_noise, title='hungarian')
            # visualize_matching(passengers, drivers, matching_array_without_noise_naive, title='naive')
            # exit(1)

            dist_without_noise = distances[
                tuple(np.transpose(matching_array_without_noise))
            ].sum()
            apx_errors_without_noise.append(dist_without_noise)
            dist_with_noise = distances[
                tuple(np.transpose(matching_array_with_noise))
            ].sum()
            apx_errors.append(
                np.abs(dist_without_noise - dist_with_noise) / dist_without_noise
            )

            dist_with_noise_naive = distances[
                tuple(np.transpose(matching_array_with_noise_naive))
            ].sum()
            apx_errors_naive.append(
                np.abs(dist_without_noise - dist_with_noise_naive) / dist_without_noise
            )

            dist_with_noise_naive_random = distances[
                tuple(np.transpose(matching_array_with_noise_naive_random))
            ].sum()
            apx_errors_naive_random.append(
                np.abs(dist_without_noise - dist_with_noise_naive_random)
                / dist_without_noise
            )

            # dist_with_noise_auction = distances[
            #     tuple(np.transpose(matching_array_with_noise_auction))
            # ].sum()
            # apx_errors_auction.append(
            #     np.abs(dist_without_noise - dist_with_noise_auction) / dist_without_noise
            # )

        apx_errors_avg.append(np.average(apx_errors))
        apx_errors_std.append(np.std(apx_errors))

        apx_errors_avg_naive.append(np.average(apx_errors_naive))
        apx_errors_std_naive.append(np.std(apx_errors_naive))

        apx_errors_avg_naive_random.append(np.average(apx_errors_naive_random))
        apx_errors_std_naive_random.append(np.std(apx_errors_naive_random))

        # apx_errors_avg_auction.append(np.average(apx_errors_auction))
        # apx_errors_std_auction.append(np.std(apx_errors_auction))

    plt.errorbar(
        STD_SCALE / GRID[1],
        apx_errors_avg,
        yerr=apx_errors_std,
        label="hungarian on noise",
    )
    plt.errorbar(
        STD_SCALE / GRID[1],
        apx_errors_avg_naive,
        yerr=apx_errors_std_naive,
        label="naive on noise",
    )

    plt.errorbar(
        STD_SCALE / GRID[1],
        apx_errors_avg_naive_random,
        yerr=apx_errors_std_naive_random,
        label="naive random on noise",
    )
    # plt.errorbar(
    #     STD_SCALE, apx_errors_avg_auction, yerr=apx_errors_std_auction, label="auction"
    # )
    plt.xlabel("Std of noise divided by grid length")
    plt.ylabel("Approximation error of solution on noised problem")
    plt.legend()
    plt.title(
        f"Approximation error of solution on noised problem - Noised Passengers\n N={N}, M={M}"
    )
    plt.grid()
    if SAVE:
        filename = (
            "plots/plot" + datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
        )
        plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    main()
    # experiment_M_N()
