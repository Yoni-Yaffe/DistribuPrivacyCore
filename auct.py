import numpy as np


def auction_assignment(
    biadjecancy_matrix: np.ndarray,
    minimize=True,
    scaling=True,
    noised=False,
    eps=0.0,
    theta=0.15,
    std=1,
):
    switch_assign = False
    if biadjecancy_matrix.shape[0] > biadjecancy_matrix.shape[1]:
        biadjecancy_matrix = biadjecancy_matrix.transpose()
        switch_assign = True

    if minimize:
        biadjecancy_matrix = biadjecancy_matrix.max() - biadjecancy_matrix

    if scaling:
        if noised:
            assignments = auction_with_scaling_noised(biadjecancy_matrix, theta, std)
        else:
            assignments = auction_with_scaling(biadjecancy_matrix, theta)
    else:
        assignments = auction(biadjecancy_matrix, eps)

    if switch_assign:
        assignments[:, 0], assignments[:, 1] = (
            assignments[:, 1],
            assignments[:, 0].copy(),
        )

    return assignments


def auction(biadjecancy_matrix: np.ndarray, eps):
    unhappy_bidders = list(range(biadjecancy_matrix.shape[0]))
    prices = np.zeros(biadjecancy_matrix.shape[1])
    assignment = np.arange(biadjecancy_matrix.shape[0])

    while len(unhappy_bidders) > 0:
        bidder = unhappy_bidders.pop(0)
        rel_vals = (biadjecancy_matrix[bidder] - prices).flatten()

        cur_val_obj = assignment[bidder]
        cur_val = rel_vals[cur_val_obj]

        max_val_obj = np.argmax(rel_vals)
        max_val = rel_vals[max_val_obj]

        if max_val > cur_val + eps:
            outbidded = np.argwhere(assignment == max_val_obj)
            if outbidded.size:
                assignment[bidder], assignment[outbidded] = (
                    assignment[outbidded],
                    assignment[bidder],
                )
                unhappy_bidders.append(outbidded)
            else:
                assignment[bidder] = max_val_obj
            prices[max_val_obj] += max_val - np.partition(rel_vals, -2)[-2] + eps

    return np.array(list(enumerate(assignment)))


def auction_with_scaling(biadjecancy_matrix: np.ndarray, theta):
    prices = np.zeros(biadjecancy_matrix.shape[1])
    assignment = np.arange(biadjecancy_matrix.shape[0])
    eps = biadjecancy_matrix.max() / 2

    while eps > theta / biadjecancy_matrix.shape[0]:
        unhappy_bidders = list(range(biadjecancy_matrix.shape[0]))
        while len(unhappy_bidders) > 0:
            bidder = unhappy_bidders.pop(0)
            rel_vals = (biadjecancy_matrix[bidder] - prices).flatten()

            cur_val_obj = assignment[bidder]
            cur_val = rel_vals[cur_val_obj]

            max_val_obj = np.argmax(rel_vals)
            max_val = rel_vals[max_val_obj]

            if max_val > cur_val + eps:
                outbidded = np.argwhere(assignment == max_val_obj)
                if outbidded.size:
                    assignment[bidder], assignment[outbidded] = (
                        assignment[outbidded],
                        assignment[bidder],
                    )
                    unhappy_bidders.append(outbidded)
                else:
                    assignment[bidder] = max_val_obj
                prices[max_val_obj] += max_val - np.partition(rel_vals, -2)[-2] + eps
        eps *= theta

    return np.array(list(enumerate(assignment)))


def auction_with_scaling_noised(biadjecancy_matrix: np.ndarray, theta, std):
    prices = np.zeros(biadjecancy_matrix.shape[1])
    assignment = np.arange(biadjecancy_matrix.shape[0])
    eps = biadjecancy_matrix.max() / 2

    while eps > theta / biadjecancy_matrix.shape[0]:
        unhappy_bidders = list(range(biadjecancy_matrix.shape[0]))
        while len(unhappy_bidders) > 0:
            bidder = unhappy_bidders.pop(0)
            rel_vals = (biadjecancy_matrix[bidder] - prices).flatten()

            cur_val_obj = assignment[bidder]
            cur_val = rel_vals[cur_val_obj]

            max_val_obj = np.argmax(rel_vals)
            max_val = rel_vals[max_val_obj]

            if max_val > cur_val + eps:
                outbidded = np.argwhere(assignment == max_val_obj)
                if outbidded.size:
                    assignment[bidder], assignment[outbidded] = (
                        assignment[outbidded],
                        assignment[bidder],
                    )
                    unhappy_bidders.append(outbidded)
                else:
                    assignment[bidder] = max_val_obj
                prices[max_val_obj] += np.amax(
                    (
                        max_val
                        - np.partition(rel_vals, -2)[-2]
                        + eps
                        + np.random.normal(0, std),
                        eps,
                    )
                )
        eps *= theta

    return np.array(list(enumerate(assignment)))
