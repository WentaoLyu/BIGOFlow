import torch


def gaussian_kernel(batch, epsilon):
    r = torch.sum(batch * batch, 1, keepdim=True)
    distance_matrix = torch.exp(-(r - 2 * batch @ batch.t() + r.t()) / (2 * epsilon**2))
    return distance_matrix


def compute_markov(batch, param, method: str = "Gaussian"):
    if method == "Gaussian":
        K = gaussian_kernel(batch, param)
    elif method == "Alpha-Decay":
        pass
    else:
        raise ValueError("Invalid method")
    q = 1 / torch.sum(K, dim=1)
    q = q @ q.t()
    K = K * q
    q = 1 / torch.sum(K, dim=1, keepdim=True)
    K = K * q
    q = 1 / q
    q = q.squeeze(1)
    sum_q = torch.sum(q)
    q = q / sum_q
    return K, q


def compute_distance(batch, param, alpha, K, method="Gaussian"):
    P, pi = compute_markov(batch, param, method)
    G = torch.zeros(size=(batch.shape[0], batch.shape[0]))
    for k in range(K):
        G += 2 ** (-(K - k) * alpha) * torch.norm(
            P[None, :, :] - P[:, None, :], p=1, dim=2
        )
        G += 2 ** (-(K + 1) / 2) * torch.abs(pi[:, None] - pi[None, :])
        P = P @ P
    return G
