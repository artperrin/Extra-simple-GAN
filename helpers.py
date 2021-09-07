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


def sample_data(num: int, gen, scale: float = 100, dim: float = 2) -> tuple:
    """sample_data creates a generator of 2D data

    Parameters
    ----------
    num : int
        length of the dataset to be generated
    gen : function
        function to generate y-axis data
    scale : float, optional
        scale factor, by default 100
    dim : float, optional
        dimension of the data, by default 2

    Yields
    -------
    Iterator[tuple]
        dim-D dataset
    """
    X = scale * (np.random.random_sample((num,)) - 0.5)
    if dim == 3: # temporary solution, not a random 3D mesh
        X = np.linspace(-scale, scale, int(np.sqrt(num))+1)
        Y = np.linspace(-scale, scale, int(np.sqrt(num))+1)
        X, Y = np.meshgrid(X, Y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        X, Y, Z = np.reshape(X, (1, -1))[0], np.reshape(Y, (1, -1))[0], np.reshape(Z, (1, -1))[0]
        for i in range(len(Z)):
            yield (X[i], Y[i], Z[i])
    else:
        for x in X:
            yield (x, gen(x))


def plot_data(data: list):
    """plot_data plots the dataset in a matplotlib object

    Parameters
    ----------
    data : list
        2D or 3D dataset to be plotted (list of tuples (x, y) or (x, y, z))

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        matplotlib object with the dataset plotted
    """
    data = np.array(data)
    if len(data[0]) == 2: # 2D plot
        _, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], color="blue", alpha=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    if len(data[0]) == 3: # 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], color="blue", alpha=0.5)
    return ax


def main():
    """main creates the output directory, a dataset and trains the GAN to generate realistic fake data"""
    if not os.path.isdir("output"):
        os.mkdir("output")
    gen = get_polynomial_function(cst.POLY)
    sampler = sample_data(10000, gen, scale=10, dim=3)
    dataset = dt.DataLoader(database.Samples(sampler), batch_size=512)
    neuralnetwork.train_gan(
        dataset,
        neuralnetwork.Generator(4, 3, dropout=.25, hidden_sizes=[32, 32, 16]),
        neuralnetwork.Discriminator(3, 2, hidden_sizes=[16, 16]),
        epochs=cst.NUM_EPOCHS,
        lr=cst.LEARNING_RATE,
        display_test=True,
    )

if __name__ == "__main__":
    data = list(sample_data(1000, get_polynomial_function(cst.POLY), dim=3))
    ax = plot_data(data)
    plt.show()