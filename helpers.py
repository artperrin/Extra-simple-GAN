import os
import numpy as np
import torch.utils.data as dt
import matplotlib.pyplot as plt

import constants as cst
import modules.database as database
import modules.neuralnetwork as neuralnetwork

import matplotlib
matplotlib.use("Qt5Agg") # to use interative backend, needs PyQt5 library

plt.style.use("ggplot")


def get_polynomial_function(coefs: list):
    """get_polynomial_function returns the polynomial function created by the given coefficients

    Parameters
    ----------
    coefs : list
        coefficients of the polynomial function (first = higher degree)

    Returns
    -------
    function
        polynomial function
    """
    deg = len(coefs) - 1
    return lambda x: sum([coef * x ** (deg - i) for (i, coef) in enumerate(coefs)])


def sample_data(num: int, gen, scale: float = 100) -> tuple:
    """sample_data creates a generator of 2D data

    Parameters
    ----------
    num : int
        length of the dataset to be generated
    gen : function
        function to generate y-axis data
    scale : float, optional
        scale factor, by default 100

    Yields
    -------
    Iterator[tuple]
        2D dataset
    """
    X = scale * (np.random.random_sample((num,)) - 0.5)
    for x in X:
        yield (x, gen(x))


def plot_data(data: list):
    """plot_data plots the dataset in a matplotlib object

    Parameters
    ----------
    data : list
        2D dataset to be plotted (list of tuples (x, y))

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        matplotlib object with the dataset plotted
    """
    data = np.array(data)
    _, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], color="blue", alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def main():
    """main creates the output directory, a dataset and trains the GAN to generate realistic fake data"""
    if not os.path.isdir("output"):
        os.mkdir("output")
    POLY = cst.POLY
    gen = get_polynomial_function(POLY)
    sampler = sample_data(10000, gen, scale=10)
    dataset = dt.DataLoader(database.Samples(sampler), batch_size=512)
    neuralnetwork.train_gan(
        dataset,
        neuralnetwork.Generator(2, 2, dropout=.25, hidden_sizes=[32, 32, 16]),
        neuralnetwork.Discriminator(2, 2, hidden_sizes=[16, 16]),
        epochs=cst.NUM_EPOCHS,
        lr=cst.LEARNING_RATE,
        display_test=True,
    )
