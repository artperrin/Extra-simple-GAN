import matplotlib
import numpy as np
import logging as lg
import torch.utils.data as dt
import matplotlib.pyplot as plt

import constants as cst
import modules.database as database
import modules.neuralnetwork as neuralnetwork

matplotlib.use("Qt5Agg")
plt.style.use("ggplot")


def get_polynomial_function(coefs: list):
    deg = len(coefs) - 1
    return lambda x: sum([coef * x ** (deg - i) for (i, coef) in enumerate(coefs)])


def sample_data(num: int, gen, scale: float = 100) -> float:
    X = scale * (np.random.random_sample((num,)) - 0.5)
    for x in X:
        yield (x, gen(x))


def plot_data(data: list) -> None:
    data = np.array(data)
    _, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], color="blue", alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def main():
    """FOR TESTING PURPOSES"""
    POLY = cst.POLY
    gen = get_polynomial_function(POLY)
    sampler = sample_data(1000, gen, scale=1)
    dataset = dt.DataLoader(database.Samples(sampler), batch_size=128)
    neuralnetwork.train_gan(
        dataset,
        neuralnetwork.Generator(2, 2),
        neuralnetwork.Discriminator(2, 2),
        epochs=100001,
        lr=cst.LEARNING_RATE,
        display_test=True,
    )


if __name__ == "__main__":
    lg.root.setLevel(lg.INFO)
    main()
