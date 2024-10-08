import numpy as np
from scipy.spatial.distance import cdist
import scipy
import auct
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from datetime import datetime
import time


SAVE = True
N = 256  # Drivers
M = int(N)    # Passengers
GRID = 0, 1024
EXPERIMENTS_NUM = 500
STD_SCALE = np.linspace(0, GRID[1] / 10, num=10)
RADIUS_SCALE = np.linspace(0, GRID[1] / 5 , num=10)
MAX_ITER = 500

def unif(loc: float = 0.0, scale: float = 1.0, size=None) -> float:
    # std = sqrt([low - high] ** 2 / 12) -> |low - high| / 2 = std * sqrt(3)
    relative_boundary = scale * np.sqrt(3)
    return np.random.uniform(loc - relative_boundary, loc + relative_boundary, size)


def polar_to_cartesian(r, theta):
    return np.stack([r * np.cos(theta), r * np.sin(theta)], 1)


def get_matching_from_biadjecncy_matrix(biadjecancy_matrix: np.array):
    matching = scipy.optimize.linear_sum_assignment(biadjecancy_matrix)
    matching_array = np.vstack(matching).T
    return matching_array


def plot_distance_density(n=1000, m=1000, factor=0.4):
    factors_list = [0, 0.2, 0.4]
    # factors_list = [0.1]
    for factor in factors_list:
        drivers, passengers = np.random.uniform(0, 1, (n, 2)), np.random.uniform(
            0, 1, (m , 2)
        )

        distances = cdist(drivers, passengers)
        noise = np.random.uniform(1-factor, 1+factor, (n, m))
        noised_distances = distances * noise
        # sns.kdeplot(data=distances.flatten())
        # sns.kdeplot(distances.flatten(), fill=True)
        sns.kdeplot(noised_distances.flatten(), fill=True, label=f'factor={factor}')

    plt.title('PDF of distance for different factors')
    plt.xlabel('distance')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()
    plt.savefig("distance_density_plot.png", dpi=300)
    plt.show()


def greedy_assignment(biadjecancy_matrix: np.array, minimize=True):
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


def greedy_assignment_random(biadjecancy_matrix: np.array, minimize=True):
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


def naive_las_vegas(biadjecancy_matrix: np.array, minimize=True, K=5, max_iter=MAX_ITER):
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
        raise RuntimeError("Las vegas algorithm did not converge")
    if switch_assign:
        assignments[:, 0], assignments[:, 1] = (
            assignments[:, 1],
            assignments[:, 0].copy(),
        )

    return np.array(list(enumerate(assignments)))

def naive_las_vegas_modified(biadjecancy_matrix: np.array,MAX_K=5, max_iter=MAX_ITER):
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
    K = 1
    num_idle = 0
    while i < max_iter and not_assigned.size > 0:
        prev_assigned = (assignments > - 1).sum()
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
        curr_assigned = (assignments > - 1).sum()
        if curr_assigned == prev_assigned:
            num_idle += 1
        if num_idle == 50:
            K = max(K + 1, MAX_K)
            num_idle = 0
        i += 1
    if i >= max_iter:
        raise RuntimeError("Las vegas algorithm did not converge")
    if switch_assign:
        assignments[:, 0], assignments[:, 1] = (
            assignments[:, 1],
            assignments[:, 0].copy(),
        )

    return np.array(list(enumerate(assignments)))

def visualize_matching_full(n=10):
    drivers, passengers = np.random.uniform(*GRID, (n, 2)), np.random.uniform(
        *GRID, (n, 2)
    )
    distances = cdist(drivers, passengers)
    matching = get_matching_from_biadjecncy_matrix(distances)

    visualize_matching(passengers, drivers, matching)

def visualize_matching(passenger_loc, drivers_loc, matching, title=None):
    plt.plot(
        passenger_loc[:, 0], passenger_loc[:, 1], "o", color="red", label="passengers"
    )
    plt.plot(
        drivers_loc[:, 0], drivers_loc[:, 1], "o", color="blue", label="drivers"
    )
    for p, d in matching:
        locs = np.vstack([passenger_loc[p], drivers_loc[d]])
        plt.plot(locs[:, 0], locs[:, 1], color="black")
    if title:
        plt.title(title)
    plt.legend()
    plt.grid()
    if SAVE:
        filename = (
            "plots/plot" + datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
        )
        plt.savefig(filename)
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
    M_factors = np.linspace(1, 2, 10)
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
            matching_array_naive = greedy_assignment(distances)
            matching_array_naive_random = naive_las_vegas(distances, K=3)

            dist_hungarian = distances[
                tuple(np.transpose(matching_array_hungarian))
            ].sum()

            dist_naive = distances[tuple(np.transpose(matching_array_naive))].sum()
            apx_errors_naive.append(
                np.abs(dist_hungarian - dist_naive) / dist_hungarian
            )

            dist_naive_random = distances[
                tuple(np.transpose(matching_array_naive_random))
            ].sum()
            apx_errors_naive_random.append(
                np.abs(dist_hungarian - dist_naive_random) / dist_hungarian
            )

        apx_errors_avg_naive.append(np.average(apx_errors_naive))
        apx_errors_std_naive.append(np.std(apx_errors_naive))

        apx_errors_avg_naive_random.append(np.average(apx_errors_naive_random))
        apx_errors_std_naive_random.append(np.std(apx_errors_naive_random))

    plt.errorbar(
        M_factors, apx_errors_avg_naive, yerr=apx_errors_std_naive, label="greedy"
    )

    plt.errorbar(
        M_factors, apx_errors_avg_naive_random, yerr=apx_errors_std_naive_random,
        label="las vegas"
    )
    plt.xlabel("#Passengers/#Drivers")
    plt.ylabel("Approximation error of solution\ncompared to hungarian")
    plt.title(f"Approximation error of Greedy solution\nas a function of #Passengers/#Drivers")
    plt.grid()
    plt.legend()
    if SAVE:
        filename = (
            "plots/plot" + datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
        )
        plt.savefig(filename)
    plt.show()





def calculate_min_distance():
    M = 5
    min_dist = []
    N_range = np.linspace(10, 2500, 100).astype(int)
    for N in tqdm(N_range):
        curr_min_distance = []
        for i in range(EXPERIMENTS_NUM):
            # print(GRID)
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
def average_distance_calculation():
    np.random.seed(0)
    grid_size = np.arange(10, 10000, 100)
    avg_dist = []
    for size in tqdm(grid_size):
        curr = 0

        for i in range(EXPERIMENTS_NUM):
            # print(size, )

            drivers = np.random.uniform(0, size, (N, 2))
            passengers = np.random.uniform(
                0, size, (M, 2)
            )
            distances = cdist(drivers, passengers)
            curr += distances.mean()
        avg_dist.append(curr / (EXPERIMENTS_NUM * size))
    plt.plot(grid_size, avg_dist)
    plt.show()


def entropy_graph_empiric():
    np.random.seed(1)

    apx_errors_avg_hungarian_ball = list()
    apx_errors_std_hungarian_ball = list()

    apx_errors_avg_hungarian_ring = list()
    apx_errors_std_hungarian_ring = list()


    apx_errors_avg_naive_ball = list()
    apx_errors_std_naive_ball = list()

    apx_errors_avg_naive_ring = list()
    apx_errors_std_naive_ring = list()

    apx_errors_avg_las_vegas_ring = list()
    apx_errors_std_las_vegas_ring = list()

    distance_rad_list = []
    for rad in tqdm(RADIUS_SCALE):
        apx_errors_naive_ball = list()
        apx_errors_naive_ring = list()
        apx_errors_las_vegas_ring = list()
        apx_errors_hungarian_ball = list()
        apx_errors_hungarian_ring = list()
        curr_distance_rad = []
        for experiment in range(EXPERIMENTS_NUM):
            drivers, passengers = np.random.uniform(*GRID, (N, 2)), np.random.uniform(
                *GRID, (M, 2)
            )
            distances = cdist(drivers, passengers)
            d_mean = distances.mean()
            rad_distances = rad ** 2 / (d_mean * 2)
            curr_distance_rad.append(rad_distances)
            noise = np.sqrt(np.random.uniform(0, rad ** 2, M)), np.random.uniform(0, 2 * np.pi, M)
            fake_passengers = passengers + polar_to_cartesian(*noise)

            noise_matrix = np.random.uniform(-rad_distances / 2, rad_distances / 2, (N, M))
            # relative_noise_matrix = np.random.uniform(1 - relative_factor / 2, 1 + relative_factor / 2, (N, M))

            noise_distances_ball = cdist(drivers, fake_passengers)
            noise_distances_ring = np.maximum(
                GRID[1] / 10000, distances + noise_matrix
            )
            # noise_distances_ring = distances * relative_noise_matrix

            matching_array_without_noise = get_matching_from_biadjecncy_matrix(
                distances
            )

            matching_array_with_noise = get_matching_from_biadjecncy_matrix(
                noise_distances_ball
            )
            matching_array_with_noise_ring = get_matching_from_biadjecncy_matrix(
                noise_distances_ring
            )

            matching_array_with_noise_naive_ball = greedy_assignment(noise_distances_ball)
            matching_array_with_noise_naive_ring = greedy_assignment(noise_distances_ring)


            matching_array_with_noise_las_vegas = naive_las_vegas(noise_distances_ring, K=3)


            dist_without_noise = distances[
                tuple(np.transpose(matching_array_without_noise))
            ].sum()

            dist_with_noise_hungarian = distances[
                tuple(np.transpose(matching_array_with_noise))
            ].sum()
            apx_errors_hungarian_ball.append(
                np.abs(dist_without_noise - dist_with_noise_hungarian) / dist_without_noise
            )

            dist_with_noise_hungarian_ring = distances[
                tuple(np.transpose(matching_array_with_noise_ring))
            ].sum()
            apx_errors_hungarian_ring.append(
                np.abs(dist_without_noise - dist_with_noise_hungarian_ring) / dist_without_noise
            )

            dist_with_noise_naive_ball = distances[
                tuple(np.transpose(matching_array_with_noise_naive_ball))
            ].sum()
            apx_errors_naive_ball.append(
                np.abs(dist_without_noise - dist_with_noise_naive_ball) / dist_without_noise
            )

            dist_with_noise_naive_ring = distances[
                tuple(np.transpose(matching_array_with_noise_naive_ring))
            ].sum()
            apx_errors_naive_ring.append(
                np.abs(dist_without_noise - dist_with_noise_naive_ring) / dist_without_noise
            )

            dist_with_noise_las_vegas_ring = distances[
                tuple(np.transpose(matching_array_with_noise_las_vegas))
            ].sum()
            apx_errors_las_vegas_ring.append(
                np.abs(dist_without_noise - dist_with_noise_las_vegas_ring) / dist_without_noise
            )


        apx_errors_avg_hungarian_ball.append(np.average(apx_errors_hungarian_ball))
        apx_errors_std_hungarian_ball.append(np.std(apx_errors_hungarian_ball))

        apx_errors_avg_hungarian_ring.append(np.average(apx_errors_hungarian_ring))
        apx_errors_std_hungarian_ring.append(np.std(apx_errors_hungarian_ring))


        apx_errors_avg_naive_ball.append(np.average(apx_errors_naive_ball))
        apx_errors_std_naive_ball.append(np.std(apx_errors_naive_ball))

        apx_errors_avg_naive_ring.append(np.average(apx_errors_naive_ring))
        apx_errors_std_naive_ring.append(np.std(apx_errors_naive_ring))

        apx_errors_avg_las_vegas_ring.append(np.average(apx_errors_las_vegas_ring))
        apx_errors_std_las_vegas_ring.append(np.std(apx_errors_las_vegas_ring))

        distance_rad_list.append(np.mean(curr_distance_rad))

    plot_data = [
        {'values': apx_errors_avg_hungarian_ball, 'errors': apx_errors_std_hungarian_ball, 'label': "hungarian ball"},
        # {'values': apx_errors_avg_hungarian_ring, 'errors': apx_errors_std_hungarian_ring, 'label': "hungarian ring"},
        # {'values': apx_errors_avg_naive_ball, 'errors': apx_errors_std_naive_ball, 'label': "greedy ball"},
        # {'values': apx_errors_avg_naive_ring, 'errors': apx_errors_std_naive_ring, 'label': "greedy ring"},
        {'values': apx_errors_avg_las_vegas_ring, 'errors': apx_errors_std_las_vegas_ring, 'label': "las vegas ring"}
    ]

    # Loop through each entry and plot the error bars
    for data in plot_data:
        plt.errorbar(
            RADIUS_SCALE / GRID[1],
            data['values'],
            yerr=data['errors'],
            label=data['label']
        )
    plt.ylabel("Approximation error compared to\noptimal solution without noise")
    plt.legend()
    plt.title(
        f"Approximation error as a function of entropy \nN={N}, M={M}"
    )

    # Here we add the second x-axis
    ax1 = plt.gca()  # Get the current axes instance
    ax1.set_xlim(RADIUS_SCALE[0] / GRID[1], RADIUS_SCALE[-1] / GRID[1])
    ax1.set_xticks(RADIUS_SCALE / GRID[1])
    ax1.set_xticklabels(["{:.2f}".format(t * 100) for t in RADIUS_SCALE / GRID[1]])
    ax1.set_xlabel('ball noise as a percentage of grid size')
    ax2 = ax1.twiny()  # Create a new set of axes (twin) that shares the same y-axis

    new_tick_locations = np.array(distance_rad_list) / GRID[1]
    ax2.set_xlim(new_tick_locations[0], new_tick_locations[-1])
    ax2.set_xticks(RADIUS_SCALE / GRID[1])
    ax2.set_xticklabels(["{:.2f}".format(t * 100) for t in new_tick_locations])
    ax2.set_xlabel('distance (ring) noise length as a percentage of grid size')

    plt.tight_layout()

    plt.grid()
    if SAVE:
        filename = (
                "plots/plot" + datetime.today().strftime("%Y_%m_%d__%H_%M_%S") + ".png"
        )
        plt.savefig(filename)
    plt.show()


def entropy_graph():
    np.random.seed(1)

    apx_errors_avg_hungarian_ball = list()
    apx_errors_std_hungarian_ball = list()

    apx_errors_avg_hungarian_ring = list()
    apx_errors_std_hungarian_ring = list()


    apx_errors_avg_naive_ball = list()
    apx_errors_std_naive_ball = list()

    apx_errors_avg_naive_ring = list()
    apx_errors_std_naive_ring = list()

    apx_errors_avg_las_vegas_ring = list()
    apx_errors_std_las_vegas_ring = list()

    distance_rad_list = []
    for rad in tqdm(RADIUS_SCALE):
        rad_distances = rad ** 2 / GRID[1]
        distance_rad_list.append(rad_distances)
        relative_factor = 2 * rad ** 2 / GRID[1] ** 2
        # rad_distances = rad / 2000
        apx_errors_naive_ball = list()
        apx_errors_naive_ring = list()
        apx_errors_las_vegas_ring = list()
        apx_errors_hungarian_ball = list()
        apx_errors_hungarian_ring = list()
        for experiment in range(EXPERIMENTS_NUM):
            drivers, passengers = np.random.uniform(*GRID, (N, 2)), np.random.uniform(
                *GRID, (M, 2)
            )
            distances = cdist(drivers, passengers)

            noise = np.sqrt(np.random.uniform(0, rad ** 2, M)), np.random.uniform(0, 2 * np.pi, M)
            fake_passengers = passengers + polar_to_cartesian(*noise)

            # noise_matrix = unif(0, std_distance, (N, M))
            noise_matrix = np.random.uniform(-rad_distances / 2, rad_distances / 2, (N, M))
            relative_noise_matrix = np.random.uniform(1 - relative_factor / 2, 1 + relative_factor / 2, (N, M))

            noise_distances_ball = cdist(drivers, fake_passengers)

            noise_distances_ring = distances * relative_noise_matrix

            matching_array_without_noise = get_matching_from_biadjecncy_matrix(
                distances
            )

            matching_array_with_noise = get_matching_from_biadjecncy_matrix(
                noise_distances_ball
            )
            matching_array_with_noise_ring = get_matching_from_biadjecncy_matrix(
                noise_distances_ring
            )
            matching_array_with_noise_naive_ball = greedy_assignment(noise_distances_ball)
            matching_array_with_noise_naive_ring = greedy_assignment(noise_distances_ring)


            matching_array_with_noise_las_vegas = naive_las_vegas(noise_distances_ring, K=3)


            dist_without_noise = distances[
                tuple(np.transpose(matching_array_without_noise))
            ].sum()

            dist_with_noise_hungarian = distances[
                tuple(np.transpose(matching_array_with_noise))
            ].sum()
            apx_errors_hungarian_ball.append(
                np.abs(dist_without_noise - dist_with_noise_hungarian) / dist_without_noise
            )

            dist_with_noise_hungarian_ring = distances[
                tuple(np.transpose(matching_array_with_noise_ring))
            ].sum()
            apx_errors_hungarian_ring.append(
                np.abs(dist_without_noise - dist_with_noise_hungarian_ring) / dist_without_noise
            )
            # apx_errors_without_noise.append(dist_without_noise)
            # dist_with_noise = distances[
            #     tuple(np.transpose(matching_array_with_noise))
            # ].sum()
            # apx_errors.append(
            #     np.abs(dist_without_noise - dist_with_noise) / dist_without_noise
            # )

            dist_with_noise_naive_ball = distances[
                tuple(np.transpose(matching_array_with_noise_naive_ball))
            ].sum()
            apx_errors_naive_ball.append(
                np.abs(dist_without_noise - dist_with_noise_naive_ball) / dist_without_noise
            )

            dist_with_noise_naive_ring = distances[
                tuple(np.transpose(matching_array_with_noise_naive_ring))
            ].sum()
            apx_errors_naive_ring.append(
                np.abs(dist_without_noise - dist_with_noise_naive_ring) / dist_without_noise
            )

            dist_with_noise_las_vegas_ring = distances[
                tuple(np.transpose(matching_array_with_noise_las_vegas))
            ].sum()
            apx_errors_las_vegas_ring.append(
                np.abs(dist_without_noise - dist_with_noise_las_vegas_ring) / dist_without_noise
            )
            # dist_with_noise_naive_random = distances[
            #     tuple(np.transpose(matching_array_with_noise_naive_random))
            # ].sum()
            # apx_errors_naive_random.append(
            #     np.abs(dist_without_noise - dist_with_noise_naive_random)
            #     / dist_without_noise
            # )

            # dist_with_noise_auction = distances[
            #     tuple(np.transpose(matching_array_with_noise_auction))
            # ].sum()
            # apx_errors_auction.append(
            #     np.abs(dist_without_noise - dist_with_noise_auction) / dist_without_noise
            # )

        # apx_errors_avg.append(np.average(apx_errors))
        # apx_errors_std.append(np.std(apx_errors))


        apx_errors_avg_hungarian_ball.append(np.average(apx_errors_hungarian_ball))
        apx_errors_std_hungarian_ball.append(np.std(apx_errors_hungarian_ball))

        apx_errors_avg_hungarian_ring.append(np.average(apx_errors_hungarian_ring))
        apx_errors_std_hungarian_ring.append(np.std(apx_errors_hungarian_ring))


        apx_errors_avg_naive_ball.append(np.average(apx_errors_naive_ball))
        apx_errors_std_naive_ball.append(np.std(apx_errors_naive_ball))

        apx_errors_avg_naive_ring.append(np.average(apx_errors_naive_ring))
        apx_errors_std_naive_ring.append(np.std(apx_errors_naive_ring))

        apx_errors_avg_las_vegas_ring.append(np.average(apx_errors_las_vegas_ring))
        apx_errors_std_las_vegas_ring.append(np.std(apx_errors_las_vegas_ring))


        # apx_errors_avg_naive_random.append(np.average(apx_errors_naive_random))
        # apx_errors_std_naive_random.append(np.std(apx_errors_naive_random))

        # apx_errors_avg_auction.append(np.average(apx_errors_auction))
        # apx_errors_std_auction.append(np.std(apx_errors_auction))

    plot_data = [
        {'values': apx_errors_avg_hungarian_ball, 'errors': apx_errors_std_hungarian_ball, 'label': "hungarian ball"},
        # {'values': apx_errors_avg_hungarian_ring, 'errors': apx_errors_std_hungarian_ring, 'label': "hungarian ring"},
        # {'values': apx_errors_avg_naive_ball, 'errors': apx_errors_std_naive_ball, 'label': "naive ball"},
        {'values': apx_errors_avg_naive_ring, 'errors': apx_errors_std_naive_ring, 'label': "naive ring"},
        # {'values': apx_errors_avg_las_vegas_ring, 'errors': apx_errors_std_las_vegas_ring, 'label': "las vegas ring"}
    ]

    # Loop through each entry and plot the error bars
    for data in plot_data:
        plt.errorbar(
            RADIUS_SCALE / GRID[1],
            data['values'],
            yerr=data['errors'],
            label=data['label']
        )
    plt.ylabel("Approximation error compared \n to optimal solution without noise")
    plt.legend()
    plt.title(
        f"Approximation error as a function of entropy \n Distance Relative Noise N={N}, M={M}"
    )

    # Here we add the second x-axis
    ax1 = plt.gca()  # Get the current axes instance
    ax1.set_xlim(RADIUS_SCALE[0] / GRID[1], RADIUS_SCALE[-1] / GRID[1])
    ax1.set_xticks(RADIUS_SCALE / GRID[1])
    ax1.set_xticklabels(["{:.2f}".format(t) for t in RADIUS_SCALE / GRID[1]])
    ax1.set_xlabel('noise ball radius')
    ax2 = ax1.twiny()  # Create a new set of axes (twin) that shares the same y-axis

    new_tick_locations = np.array(distance_rad_list) / GRID[1]
    ax2.set_xlim(new_tick_locations[0], new_tick_locations[-1])
    ax2.set_xticks(RADIUS_SCALE / GRID[1])
    ax2.set_xticklabels(["{:.2f}".format(t) for t in new_tick_locations])
    ax2.set_xlabel('distance noise length')

    plt.tight_layout()

    plt.grid()
    if SAVE:
        filename = (
                "plots/plot" + datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
        )
        plt.savefig(filename)
    plt.show()


def relative_noise_distances():
    np.random.seed(0)
    apx_errors_avg = list()
    apx_errors_std = list()
    apx_errors_avg_naive = list()
    apx_errors_avg_naive_random = list()
    apx_errors_avg_auction = list()
    apx_errors_std_naive = list()
    apx_errors_std_naive_random = list()
    apx_errors_std_auction = list()
    factors_list = np.linspace(0, 0.75, 10)
    for factor in tqdm(factors_list):
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

            # noise = unif(0, std, M), np.random.uniform(0, np.pi, M)
            # fake_passengers = passengers + polar_to_cartesian(*noise)

            # noise_matrix = np.random.normal(0, std, (N, M))
            noise_matrix = np.random.uniform(1 - factor, 1 + factor, (N, M))
            noise_distances = distances * noise_matrix
            # noise_distances = cdist(drivers, fake_passengers)
            # noise_distances = np.maximum(
            #     GRID[1] / 1000, cdist(drivers, passengers) + noise_matrix
            # )

            matching_array_without_noise = get_matching_from_biadjecncy_matrix(
                distances
            )
            # matching_array_without_noise_naive = naive_assignment(distances)

            # matching_array_with_noise_auction = auct.auction_assignment(noise_distances)
            matching_array_with_noise = get_matching_from_biadjecncy_matrix(
                noise_distances
            )
            matching_array_with_noise_naive = greedy_assignment(noise_distances)
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
        factors_list,
        apx_errors_avg,
        yerr=apx_errors_std,
        label="hungarian on noise",
    )
    plt.errorbar(
        factors_list,
        apx_errors_avg_naive,
        yerr=apx_errors_std_naive,
        label="naive on noise",
    )

    plt.errorbar(
        factors_list,
        apx_errors_avg_naive_random,
        yerr=apx_errors_std_naive_random,
        label="las vegas on noise",
    )
    # plt.errorbar(
    #     STD_SCALE, apx_errors_avg_auction, yerr=apx_errors_std_auction, label="auction"
    # )
    plt.xlabel("factor")
    plt.ylabel("Approximation error of solution on noised problem")
    plt.legend()
    plt.title(
        f"Approximation error of solution on noised problem - Relative Noised Distances\n N={N}, M={M}"
    )
    plt.grid()
    if SAVE:
        filename = (
            "plots/plot" + datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
        )
        plt.savefig(filename)
    plt.show()


def relative_noise_distances_vs_uniform():
    np.random.seed(0)
    apx_errors_avg_uniform = list()
    apx_errors_std_uniform = list()
    apx_errors_avg_naive_uniform = list()
    apx_errors_avg_naive_random_uniform = list()
    apx_errors_std_naive_uniform = list()
    apx_errors_std_naive_random_uniform = list()

    apx_errors_avg_relative = list()
    apx_errors_std_relative = list()
    apx_errors_avg_naive_relative = list()
    apx_errors_avg_naive_random_relative = list()
    apx_errors_std_naive_relative = list()
    apx_errors_std_naive_random_relative = list()

    relative_factor_list = []
    for rad in tqdm(RADIUS_SCALE):
        apx_errors_without_noise = list()


        apx_errors_uniform = list()
        apx_errors_naive_uniform = list()
        apx_errors_naive_random_uniform = list()

        apx_errors_relative = list()
        apx_errors_naive_relative = list()
        apx_errors_naive_random_relative = list()
        curr_factor_list = []
        for experiment in range(EXPERIMENTS_NUM):
            drivers, passengers = np.random.uniform(*GRID, (N, 2)), np.random.uniform(
                *GRID, (M, 2)
            )
            distances = cdist(drivers, passengers)
            distances_mean = distances.mean()
            distances_square_mean = (distances ** 2).mean()
            # relative_factor = 2 * rad ** 2 / GRID[1] ** 2 # 1 / alpha
            relative_factor = rad * distances_mean / distances_square_mean # 1 / alpha
            curr_factor_list.append(relative_factor)
            # noise = unif(0, std, M), np.random.uniform(0, np.pi, M)
            # fake_passengers = passengers + polar_to_cartesian(*noise)

            # noise_matrix = np.random.normal(0, std, (N, M))
            relative_noise_matrix = np.random.uniform(1 - relative_factor / 2, 1 + relative_factor / 2, (N, M))

            noise_distances_relative = distances * relative_noise_matrix

            # noise_distances = cdist(drivers, fake_passengers)
            noise_matrix_uniform = np.random.uniform(-rad / 2, rad / 2, (N, M))

            noise_distances_uniform = np.maximum(
                GRID[1] / 1000, distances + noise_matrix_uniform
            )

            matching_array_without_noise = get_matching_from_biadjecncy_matrix(
                distances
            )
            # matching_array_without_noise_naive = naive_assignment(distances)

            # matching_array_with_noise_auction = auct.auction_assignment(noise_distances)
            matching_array_with_uniform_noise = get_matching_from_biadjecncy_matrix(
                noise_distances_uniform
            )
            matching_array_with_relative_noise = get_matching_from_biadjecncy_matrix(
                noise_distances_relative
            )
            matching_array_with_uniform_noise_naive = greedy_assignment(noise_distances_uniform)
            matching_array_with_relative_noise_naive = greedy_assignment(noise_distances_relative)

            # matching_array_with_noise_naive_random = naive_assignment_random(
            #     noise_distances
            # )


            matching_array_with_uniform_noise_naive_random = naive_las_vegas(noise_distances_uniform, K=3)
            matching_array_with_relative_noise_naive_random = naive_las_vegas(noise_distances_relative, K=3)

            # test = naive_las_vegas(distances)

            # visualize_matching(passengers, drivers, matching_array_without_noise, title='hungarian')
            # visualize_matching(passengers, drivers, matching_array_without_noise_naive, title='naive')
            # exit(1)

            dist_without_noise = distances[
                tuple(np.transpose(matching_array_without_noise))
            ].sum()
            apx_errors_without_noise.append(dist_without_noise)



            dist_with_uniform_noise = distances[
                tuple(np.transpose(matching_array_with_uniform_noise))
            ].sum()
            apx_errors_uniform.append(
                np.abs(dist_without_noise - dist_with_uniform_noise) / dist_without_noise
            )

            dist_with_uniform_noise_naive = distances[
                tuple(np.transpose(matching_array_with_uniform_noise_naive))
            ].sum()
            apx_errors_naive_uniform.append(
                np.abs(dist_without_noise - dist_with_uniform_noise_naive) / dist_without_noise
            )

            dist_with_uniform_noise_naive_random = distances[
                tuple(np.transpose(matching_array_with_uniform_noise_naive_random))
            ].sum()
            apx_errors_naive_random_uniform.append(
                np.abs(dist_without_noise - dist_with_uniform_noise_naive_random)
                / dist_without_noise
            )


            dist_with_relative_noise = distances[
                tuple(np.transpose(matching_array_with_relative_noise))
            ].sum()
            apx_errors_relative.append(
                np.abs(dist_without_noise - dist_with_relative_noise) / dist_without_noise
            )

            dist_with_relative_noise_naive = distances[
                tuple(np.transpose(matching_array_with_relative_noise_naive))
            ].sum()
            apx_errors_naive_relative.append(
                np.abs(dist_without_noise - dist_with_relative_noise_naive) / dist_without_noise
            )

            dist_with_relative_noise_naive_random = distances[
                tuple(np.transpose(matching_array_with_relative_noise_naive_random))
            ].sum()
            apx_errors_naive_random_relative.append(
                np.abs(dist_without_noise - dist_with_relative_noise_naive_random)
                / dist_without_noise
            )


        apx_errors_avg_uniform.append(np.average(apx_errors_uniform))
        apx_errors_std_uniform.append(np.std(apx_errors_uniform))

        apx_errors_avg_naive_uniform.append(np.average(apx_errors_naive_uniform))
        apx_errors_std_naive_uniform.append(np.std(apx_errors_naive_uniform))

        apx_errors_avg_naive_random_uniform.append(np.average(apx_errors_naive_random_uniform))
        apx_errors_std_naive_random_uniform.append(np.std(apx_errors_naive_random_uniform))




        apx_errors_avg_relative.append(np.average(apx_errors_relative))
        apx_errors_std_relative.append(np.std(apx_errors_relative))

        apx_errors_avg_naive_relative.append(np.average(apx_errors_naive_relative))
        apx_errors_std_naive_relative.append(np.std(apx_errors_naive_relative))

        apx_errors_avg_naive_random_relative.append(np.average(apx_errors_naive_random_relative))
        apx_errors_std_naive_random_relative.append(np.std(apx_errors_naive_random_relative))

        # apx_errors_avg_auction.append(np.average(apx_errors_auction))
        # apx_errors_std_auction.append(np.std(apx_errors_auction))
        relative_factor_list.append(np.mean(curr_factor_list))
    plt.errorbar(
        RADIUS_SCALE/ GRID[1],
        apx_errors_avg_uniform,
        yerr=apx_errors_std_uniform,
        label="hungarian on  uniform noise",
    )
    # plt.errorbar(
    #     RADIUS_SCALE/ GRID[1],
    #     apx_errors_avg_naive_uniform,
    #     yerr=apx_errors_std_naive_uniform,
    #     label="naive on uniform noise",
    # )

    plt.errorbar(
        RADIUS_SCALE/ GRID[1],
        apx_errors_avg_naive_random_uniform,
        yerr=apx_errors_std_naive_random_uniform,
        label="las vegas on noise",
    )


    plt.errorbar(
        RADIUS_SCALE/ GRID[1],
        apx_errors_avg_relative,
        yerr=apx_errors_std_relative,
        label="hungarian on  relative noise",
    )
    # plt.errorbar(
    #     RADIUS_SCALE/ GRID[1],
    #     apx_errors_avg_naive_relative,
    #     yerr=apx_errors_std_naive_relative,
    #     label="naive on relative noise",
    # )

    plt.errorbar(
        RADIUS_SCALE / GRID[1],
        apx_errors_avg_naive_random_relative,
        yerr=apx_errors_std_naive_random_relative,
        label="las vegas on relative noise",
    )
    # plt.errorbar(
    #     STD_SCALE, apx_errors_avg_auction, yerr=apx_errors_std_auction, label="auction"
    # )
    plt.xlabel("radius of noise devided by GRID length")
    plt.ylabel("Approximation error")
    plt.legend()
    plt.title(
        f"Approximation errors - Relative and uniform  Noised Distances\n N={N}, M={M}"
    )
    plt.grid()

    # Here we add the second x-axis
    ax1 = plt.gca()  # Get the current axes instance
    ax2 = ax1.twiny()  # Create a new set of axes (twin) that shares the same y-axis

    new_tick_locations = relative_factor_list
    ax2.set_xlim(new_tick_locations[0], new_tick_locations[-1])
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(["{:.2f}".format(t) for t in new_tick_locations])
    ax2.set_xlabel('relative factor')

    plt.tight_layout()

    if SAVE:
        filename = (
            "plots/plot" + datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
        )
        plt.savefig(filename)
    plt.show()


def time_domain_expansion_hungari(T=500):
    rad = 0.2 * GRID[1]
    drivers_timers = np.random.randint(0, N // 10, N)
    passengers_timers = np.zeros(M)
    # drivers_timers = np.zeros(N)
    all_drivers, all_passengers = np.random.uniform(*GRID, (N, 2)), np.random.uniform(
        *GRID, (M, 2)
    )
    all_noise = np.sqrt(np.random.uniform(0, rad ** 2, M)), np.random.uniform(0, 2 * np.pi, M)
    num_passengers_picked = 0

    picked_per_iteration = np.zeros(T)
    for t in range(T):
        drivers_timers = np.maximum(0, drivers_timers - 1)
        open_drivers = np.where(drivers_timers == 0)[0]
        open_passengers = np.where(passengers_timers == 0)[0]
        if len(open_passengers) == 0 or len(open_drivers) == 0:
            print("Skipped iteration")
            picked_per_iteration[t] = 0
            continue
        drivers = all_drivers[open_drivers]
        passengers = all_passengers[open_passengers]
        distances = cdist(drivers, passengers)
        noise = all_noise[0][open_passengers], all_noise[1][open_passengers]
        fake_passengers = passengers + polar_to_cartesian(*noise)
        noise_distances_ball = cdist(drivers, fake_passengers)
        matching_array_with_noise = get_matching_from_biadjecncy_matrix(noise_distances_ball)
        num_passengers_picked += matching_array_with_noise.shape[0]
        picked_per_iteration[t] += matching_array_with_noise.shape[0]
        for d, p in matching_array_with_noise:
            distance = distances[d, p]
            driver_ind = open_drivers[d]
            passenger_ind = open_passengers[d]
            drivers_timers[driver_ind] += (distance * 100) // GRID[1]
            all_noise[0][passenger_ind] = np.sqrt(np.random.uniform(0, rad ** 2))
            all_noise[1][passenger_ind] = np.random.uniform(0, 2 * np.pi)
            all_drivers[driver_ind] = all_passengers[passenger_ind]
            all_passengers[passenger_ind] = np.random.uniform(GRID[0], GRID[1], 2)
    plt.plot(np.arange(T), picked_per_iteration)
    plt.grid(True)
    plt.xlabel('iteration')
    plt.ylabel('number of passengers picked')
    plt.title('Pickup in each iteration')
    plt.show()

    return num_passengers_picked, picked_per_iteration


def time_domain_expansion_greedy(T=500):
    rad = 0.2 * GRID[1]
    drivers_timers = np.random.randint(0, N // 10, N)
    passengers_timers = np.zeros(M)
    # drivers_timers = np.zeros(N)
    all_drivers, all_passengers = np.random.uniform(*GRID, (N, 2)), np.random.uniform(
        *GRID, (M, 2)
    )
    relative_factor = 2 * rad ** 2 / GRID[1] ** 2
    all_relative_noise_factors = np.random.uniform(1 - relative_factor / 2, 1 + relative_factor / 2, (N, M))

    num_passengers_picked = 0
    picked_per_iteration = np.zeros(T)
    for t in range(T):
        drivers_timers = np.maximum(0, drivers_timers - 1)
        open_drivers = np.where(drivers_timers == 0)[0]
        open_passengers = np.where(passengers_timers == 0)[0]
        if len(open_passengers) == 0 or len(open_drivers) == 0:
            print("Skipped iteration")
            picked_per_iteration[t] = 0
            continue
        drivers = all_drivers[open_drivers]
        passengers = all_passengers[open_passengers]
        distances = cdist(drivers, passengers)
        relative__noise_factors = all_relative_noise_factors[open_drivers][:, open_passengers]
        noise_distances_ring = distances * relative__noise_factors
        matching_array_with_noise = greedy_assignment(noise_distances_ring)
        # matching_array_with_noise = naive_las_vegas_modified(noise_distances_ring, MAX_K=3)
        num_passengers_picked += matching_array_with_noise.shape[0]
        picked_per_iteration[t] += matching_array_with_noise.shape[0]
        for d, p in matching_array_with_noise:
            distance = distances[d, p]
            driver_ind = open_drivers[d]
            passenger_ind = open_passengers[d]
            drivers_timers[driver_ind] += (distance * 100) // GRID[1]
            all_relative_noise_factors[driver_ind] = np.random.uniform(1 - relative_factor / 2, 1 + relative_factor / 2, N)
            all_relative_noise_factors[:, passenger_ind] = np.random.uniform(1 - relative_factor / 2, 1 + relative_factor / 2, M)
            all_drivers[driver_ind] = all_passengers[passenger_ind]
            all_passengers[passenger_ind] = np.random.uniform(GRID[0], GRID[1], 2)
    plt.plot(np.arange(T), picked_per_iteration)
    plt.grid(True)
    plt.xlabel('iteration')
    plt.ylabel('number of passengers picked')
    plt.show()
    plt.title('Pickup in each iteration')
    return num_passengers_picked, picked_per_iteration



def plot_time_domain_graphs(T=500):
    num_picked_hungari, picked_per_iteration_hungari = time_domain_expansion_hungari(T)
    num_picked_greedy, picked_per_iteration_greedy = time_domain_expansion_greedy(T)
    plt.plot(np.arange(T), picked_per_iteration_hungari, label='centralized')
    plt.plot(np.arange(T), picked_per_iteration_greedy, label='distributed')
    plt.grid(True)
    plt.xlabel('iteration')
    plt.ylabel('number of passengers picked')
    plt.legend()
    plt.title('#Pickup in each iteration')
    plt.show()



def experiment_las_vegas_and_greedy_errors():
    np.random.seed(0)
    apx_errors_avg = list()
    apx_errors_std = list()
    apx_errors_avg_naive = list()
    apx_errors_avg_naive_random = list()
    apx_errors_avg_auction = list()
    apx_errors_std_naive = list()
    apx_errors_std_naive_random = list()
    apx_errors_std_auction = list()
    N_list = np.arange(32, 256, 16)
    for n in tqdm(N_list):
        N = M = n
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
            matching_array_naive = greedy_assignment(distances)

            matching_array_naive_random = naive_las_vegas(distances, K=3)

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

            dist_naive_random = distances[
                tuple(np.transpose(matching_array_naive_random))
            ].sum()
            apx_errors_naive_random.append(
                np.abs(dist_hungarian - dist_naive_random) / dist_hungarian
            )

        # apx_errors_avg.append(np.average(apx_errors))
        # apx_errors_std.append(np.std(apx_errors))

        apx_errors_avg_naive.append(np.average(apx_errors_naive))
        apx_errors_std_naive.append(np.std(apx_errors_naive))

        apx_errors_avg_naive_random.append(np.average(apx_errors_naive_random))
        apx_errors_std_naive_random.append(np.std(apx_errors_naive_random))

        # apx_errors_avg_auction.append(np.average(apx_errors_auction))
        # apx_errors_std_auction.append(np.std(apx_errors_auction))

    # plt.errorbar(
    #     STD_SCALE / GRID[1], apx_errors_avg, yerr=apx_errors_std, label="hungarian on noise"
    # )
    plt.errorbar(
        N_list, apx_errors_avg_naive, yerr=apx_errors_std_naive, label="greedy"
    )

    plt.errorbar(
        N_list, apx_errors_avg_naive_random, yerr=apx_errors_std_naive_random,
        label="las vegas"
    )
    # plt.errorbar(
    #     STD_SCALE, apx_errors_avg_auction, yerr=apx_errors_std_auction, label="auction"
    # )
    plt.xlabel("#Drivers")
    plt.ylabel("Approximation error of solution\ncompared to hungarian")
    plt.title(f"Approximation error of Greedy and Las Vegas\nas a function of #Drivers when #Drivers=#Passengers")
    plt.grid()
    plt.legend()
    if SAVE:
        filename = (
            "plots/plot" + datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
        )
        plt.savefig(filename)
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

            noise = unif(0, std, M), np.random.uniform(0, np.pi, M)
            fake_passengers = passengers + polar_to_cartesian(*noise)

            noise_matrix = np.random.normal(0, std, (N, M))
            noise_matrix = np.random.uniform(1/2, 3/2, (N, M))
            noise_distances = distances * noise_matrix
            # noise_distances = cdist(drivers, fake_passengers)
            # noise_distances = np.maximum(
            #     GRID[1] / 1000, cdist(drivers, passengers) + noise_matrix
            # )


            matching_array_without_noise = get_matching_from_biadjecncy_matrix(
                distances
            )
            # matching_array_without_noise_naive = naive_assignment(distances)

            # matching_array_with_noise_auction = auct.auction_assignment(noise_distances)
            matching_array_with_noise = get_matching_from_biadjecncy_matrix(
                noise_distances
            )
            matching_array_with_noise_naive = greedy_assignment(noise_distances)
            # matching_array_with_noise_naive_random = naive_assignment_random(
            #     noise_distances
            # )


            # matching_array_with_noise_naive_random = naive_las_vegas(noise_distances, K=3)


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

            # dist_with_noise_naive_random = distances[
            #     tuple(np.transpose(matching_array_with_noise_naive_random))
            # ].sum()
            # apx_errors_naive_random.append(
            #     np.abs(dist_without_noise - dist_with_noise_naive_random)
            #     / dist_without_noise
            # )

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

        # apx_errors_avg_naive_random.append(np.average(apx_errors_naive_random))
        # apx_errors_std_naive_random.append(np.std(apx_errors_naive_random))

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

    # plt.errorbar(
    #     STD_SCALE / GRID[1],
    #     apx_errors_avg_naive_random,
    #     yerr=apx_errors_std_naive_random,
    #     label="naive random on noise",
    # )
    # plt.errorbar(
    #     STD_SCALE, apx_errors_avg_auction, yerr=apx_errors_std_auction, label="auction"
    # )
    plt.xlabel("Std of noise divided by grid length")
    plt.ylabel("Approximation error of solution on noised problem")
    plt.legend()
    plt.title(
        f"Approximation error of solution on noised problem - Relative Noised Distances\n N={N}, M={M}"
    )
    plt.grid()
    if SAVE:
        filename = (
            "plots/plot" + datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
        )
        plt.savefig(filename)
    plt.show()

def las_vegas_vs_greedy_noise():
    np.random.seed(1)

    apx_errors_avg_naive_ring = list()
    apx_errors_std_naive_ring = list()

    apx_errors_avg_las_vegas_ring = list()
    apx_errors_std_las_vegas_ring = list()
    for rad in tqdm(RADIUS_SCALE):
        apx_errors_naive_ring = list()
        apx_errors_las_vegas_ring = list()
        for experiment in range(EXPERIMENTS_NUM):
            drivers, passengers = np.random.uniform(*GRID, (N, 2)), np.random.uniform(
                *GRID, (M, 2)
            )
            distances = cdist(drivers, passengers)
            noise_matrix = np.random.uniform(-rad / 2, rad / 2, (N, M))
            noise_distances_ring = np.maximum(
                GRID[1] / 10000, distances + noise_matrix
            )

            matching_array_without_noise = get_matching_from_biadjecncy_matrix(
                distances
            )

            matching_array_with_noise_ring = get_matching_from_biadjecncy_matrix(
                noise_distances_ring
            )
            matching_array_with_noise_naive_ring = greedy_assignment(noise_distances_ring)

            matching_array_with_noise_las_vegas = naive_las_vegas(noise_distances_ring, K=3)

            dist_without_noise = distances[
                tuple(np.transpose(matching_array_without_noise))
            ].sum()


            dist_with_noise_naive_ring = distances[
                tuple(np.transpose(matching_array_with_noise_naive_ring))
            ].sum()
            apx_errors_naive_ring.append(
                np.abs(dist_without_noise - dist_with_noise_naive_ring) / dist_without_noise
            )

            dist_with_noise_las_vegas_ring = distances[
                tuple(np.transpose(matching_array_with_noise_las_vegas))
            ].sum()
            apx_errors_las_vegas_ring.append(
                np.abs(dist_without_noise - dist_with_noise_las_vegas_ring) / dist_without_noise
            )

        apx_errors_avg_naive_ring.append(np.average(apx_errors_naive_ring))
        apx_errors_std_naive_ring.append(np.std(apx_errors_naive_ring))

        apx_errors_avg_las_vegas_ring.append(np.average(apx_errors_las_vegas_ring))
        apx_errors_std_las_vegas_ring.append(np.std(apx_errors_las_vegas_ring))



    plot_data = [
        # {'values': apx_errors_avg_hungarian_ball, 'errors': apx_errors_std_hungarian_ball, 'label': "hungarian ball"},
        # {'values': apx_errors_avg_hungarian_ring, 'errors': apx_errors_std_hungarian_ring, 'label': "hungarian ring"},
        # {'values': apx_errors_avg_naive_ball, 'errors': apx_errors_std_naive_ball, 'label': "greedy ball"},
        {'values': apx_errors_avg_naive_ring, 'errors': apx_errors_std_naive_ring, 'label': "greedy"},
        {'values': apx_errors_avg_las_vegas_ring, 'errors': apx_errors_std_las_vegas_ring, 'label': "las vegas"}
    ]

    # Loop through each entry and plot the error bars
    for data in plot_data:
        plt.errorbar(
            RADIUS_SCALE / GRID[1],
            data['values'],
            yerr=data['errors'],
            label=data['label']
        )
    plt.ylabel("Approximation error compared to\noptimal solution without noise")
    plt.legend()
    plt.title(
        f"Approximation error as a function of noise \nN={N}, M={M}"
    )

    # Here we add the second x-axis
    ax1 = plt.gca()  # Get the current axes instance
    ax1.set_xlim(RADIUS_SCALE[0] / GRID[1], RADIUS_SCALE[-1] / GRID[1])
    ax1.set_xticks(RADIUS_SCALE / GRID[1])
    ax1.set_xticklabels(["{:.2f}".format(t * 100) for t in RADIUS_SCALE / GRID[1]])
    ax1.set_xlabel('distance noise as a percentage of grid size')

    plt.tight_layout()

    plt.grid()
    if SAVE:
        filename = (
                "plots/plot" + datetime.today().strftime("%Y_%m_%d__%H_%M_%S") + ".png"
        )
        plt.savefig(filename)
    plt.show()

def runtime_test_experiment(num_of_trials=10):
    N_values = np.arange(100, 1100, 100)

    for n in N_values:
        print(f"\nN = {n}\n")
        times_hungary = []
        times_greedy = []
        times_las_vegas = []
        for _ in range(num_of_trials):
            drivers, passengers = np.random.uniform(*GRID, (n, 2)), np.random.uniform(
                *GRID, (n, 2)
            )
            distances = cdist(drivers, passengers)

            start_time = time.time()
            get_matching_from_biadjecncy_matrix(distances)
            elapsed_time_hungary = time.time() - start_time

            start_time = time.time()
            greedy_assignment(distances)
            elapsed_time_greedy = time.time() - start_time

            start_time = time.time()
            naive_las_vegas(distances, K=3)
            elapsed_time_las_vegas = time.time() - start_time

            times_hungary.append(elapsed_time_hungary)
            times_greedy.append(elapsed_time_greedy)
            times_las_vegas.append(elapsed_time_las_vegas)
        # print(times_hungary)
        print(f"  Hungary: {np.mean(times_hungary)*1000:.2f} ms")
        print(f"  Greedy: {np.mean(times_greedy)*1000:.2f} ms")
        print(f"  Las Vegas: {np.mean(times_las_vegas)*1000:.2f} ms")





if __name__ == "__main__":
    # np.random.seed(2)
    # visualize_matching_full(n=7)
    # main()
    # relative_noise_distances()
    # plot_distance_density(100, 50000, )
    # relative_noise_distances_vs_uniform()
    # entropy_graph()
    # entropy_graph_empiric()
    # calculate_min_distance()
    # average_distance_calculation()
    # experiment_M_N()
    # plot_time_domain_graphs()
    # las_vegas_vs_greedy_noise()
    # experiment_las_vegas_and_greedy_errors()
    runtime_test_experiment(10)
