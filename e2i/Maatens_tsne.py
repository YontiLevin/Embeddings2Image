#  this  implementation of the tsne is based on the tsne code written by Laurens van der Maaten.
#  the code are more can by found in his website: http://lvdmaaten.github.io/tsne/

import numpy as np
from tqdm import tqdm


def h_beta(d=np.array([]), beta=1.0):
    """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

    # Compute P-row and corresponding perplexity
    p = np.exp(-d.copy() * beta)
    sum_p = sum(p) + 1

    h = np.log(sum_p) + beta * np.sum(d * p) / sum_p
    p /= sum_p
    return h, p


def x2p(x=np.array([]), tol=1e-5, perplexity=30.0):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

    # Initialize some variables
    print("Computing pairwise distances...")
    n, _ = x.shape
    sum_x = np.sum(np.square(x), 1)
    d = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    p = np.zeros((n, n))
    beta = np.ones((n, 1))
    log_u = np.log(perplexity)

    # Loop over all datapoints
    for i in tqdm(range(n)):

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        di = d[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        h, this_p = h_beta(di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        h_diff = h - log_u
        tries = 0
        while np.abs(h_diff) > tol and tries < 50:

            # If not, increase or decrease precision
            if h_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            h, this_p = h_beta(di, beta[i])
            h_diff = h - log_u
            tries += 1

        # Set the final row of P
        p[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = this_p

    # Return final P-matrix
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return p


def pca(x=np.array([]), no_dims=50):
    """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""
    print("Preprocessing the data using PCA...")
    n, _ = x.shape
    x -= np.tile(np.mean(x, 0), (n, 1))
    _, m = np.linalg.eig(np.dot(x.T, x))
    y = np.dot(x, m[:, 0:no_dims])
    return y


def TSNE(x=np.array([]), no_dims=2, initial_dims=50, perplexity=40.0):
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    x = pca(x, initial_dims).real
    n, _ = x.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    y = np.random.randn(n, no_dims)
    dy = np.zeros((n, no_dims))
    iy = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    p = x2p(x, 1e-5, perplexity)
    p += np.transpose(p)
    p /= np.sum(p)
    p *= 4									# early exaggeration
    p = np.maximum(p, 1e-12)
    c = j = 0
    # Run iterations
    for j in tqdm(range(max_iter)):

        # Compute pairwise affinities
        sum_y = np.sum(np.square(y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        num[range(n), range(n)] = 0
        q = num / np.sum(num)
        q = np.maximum(q, 1e-12)

        # Compute gradient
        pq = p - q
        for i in range(n):
            dy[i, :] = np.sum(np.tile(pq[:, i] * num[:, i], (no_dims, 1)).T * (y[i, :] - y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
        gains[gains < min_gain] = min_gain
        iy = momentum * iy - eta * (gains * dy)
        y += iy
        y -= np.tile(np.mean(y, 0), (n, 1))

        # Stop lying about P-values
        if iter == 100:
            p /= 4

        c = np.sum(p * np.log(p / q))
    print("After {} Iterations the error is {}".format(j + 1, c))

    return y
