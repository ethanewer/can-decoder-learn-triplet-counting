import random

import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore
from tqdm import trange  # type: ignore

from rasp import count_triplets_decoder_cot_rasp

EOS = -1
N = 16
BATCH_SIZE = 3 * N + 3
ERR_PROB = 1 - 0.5 ** (1 / (BATCH_SIZE - N - 1))

random.seed(0)


def generate_sequence() -> tuple[list[int], list[int]]:
    x = [random.randint(0, N - 1) for _ in range(N)]
    x.append(EOS)
    y = [-1] * (N + 1)
    good = True
    while len(x) < BATCH_SIZE:
        if good:
            if random.random() < ERR_PROB:
                good = False
            else:
                x.append(int(count_triplets_decoder_cot_rasp(np.array(x))[-1]))
                y.append(1)

        if not good:
            x.append(random.randint(0, N))
            y.append(0)

    return x, y


if __name__ == "__main__":
    xs = []
    ys = []
    for _ in trange(100000):
        x, y = generate_sequence()
        xs.append(x)
        ys.append(y)

    x_train, x_val, y_train, y_val = train_test_split(
        xs,
        ys,
        train_size=95000,
        random_state=0,
    )

    np.savez(
        "data/prm/train",
        x=np.array(x_train, dtype=np.uint8),
        y=np.array(y_train, dtype=np.uint8),
    )

    np.savez(
        "data/prm/val",
        x=np.array(x_val, dtype=np.uint8),
        y=np.array(y_val, dtype=np.uint8),
    )
