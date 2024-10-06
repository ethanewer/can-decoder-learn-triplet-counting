import numpy as np

EOS = -1


def full(x, const):
    return np.full_like(x, const, dtype=int)


def indices(x):
    return np.arange(len(x), dtype=int)


def tok_map(x, func):
    return np.array([func(xi) for xi in x]).astype(int)


def seq_map(x, y, func):
    return np.array([func(xi, yi) for xi, yi in zip(x, y)]).astype(int)


def select(k, q, pred, causal=False):
    s = len(k)
    A = np.zeros((s, s), dtype=bool)
    for qi in range(s):
        for kj in range(qi + 1) if causal else range(s):  # k_index <= q_index if causal
            A[qi, kj] = pred(k[kj], q[qi])
    return A


def sel_width(A):
    return np.dot(A, np.ones(len(A))).astype(int)


def aggr_mean(A, v, default=0):
    out = np.dot(A, v)
    norm = sel_width(A)
    out = np.divide(
        out, norm, out=np.full_like(v, default, dtype=float), where=(norm != 0)
    )
    return out.astype(int)


def aggr_max(A, v, default=0):
    out = np.full_like(v, default)
    for i, row in enumerate(A):
        idxs = np.flatnonzero(row)
        if len(idxs) > 0:
            out[i] = np.max(v[idxs])  # max of selected elements in v
    return out.astype(int)


def aggr(A, v, default=0, reduction="mean"):
    if reduction == "mean":
        return aggr_mean(A, v, default)
    elif reduction == "max":
        return aggr_max(A, v, default)
    elif reduction == "min":
        return -aggr_max(A, -v, -default)


def kqv(k, q, v, pred, default=0, reduction="mean", causal=False):
    return aggr(select(k, q, pred, causal), v, default=default, reduction=reduction)


def equals(a, b):
    return a == b


def true(a, b):
    return True


def count_triplets_decoder_cot_rasp(x):
    idxs = indices(x)
    n = kqv(k=x, q=full(x, EOS), v=idxs, pred=equals, reduction="min", causal=True)
    n = tok_map(n, lambda a: a if a else -2)
    last_x = kqv(k=idxs, q=n - 1, v=x, pred=equals, reduction="mean")
    seq_len = kqv(k=x, q=x, v=idxs, pred=true, reduction="max", causal=True)

    i = seq_len - n
    j = seq_len - 2 * n
    xi = kqv(k=idxs, q=i, v=x, pred=equals, reduction="max", causal=True)
    xj = kqv(k=idxs, q=j, v=x, pred=equals, reduction="max", causal=True)

    y = (n - xi) % n + 1
    z = (last_x + xj) % n + 1

    y_mask_write = (n <= idxs) & (idxs < 2 * n)
    z_mask_write = (2 * n <= idxs) & (idxs < 3 * n)
    y_mask_read = (n < idxs) & (idxs <= 2 * n)
    z_mask_read = (2 * n < idxs) & (idxs <= 3 * n)

    z_count = sel_width(
        select(k=x * y_mask_read, q=z, pred=lambda a, b: a == b and a != 0, causal=True)
    )

    count = kqv(
        k=z_mask_read,
        q=z_mask_read,
        v=n * x * z_mask_read,
        pred=lambda a, b: a & b,
        reduction="mean",
        causal=True,
    )
    ans = count % n

    ans_mask_write = idxs == 3 * n
    eos_mask_write = idxs > 3 * n

    return (
        y * y_mask_write
        + z_count * z_mask_write
        + ans * ans_mask_write
        + EOS * eos_mask_write
    )
