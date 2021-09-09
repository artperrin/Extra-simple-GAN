import os
import numpy as np
import logging as lg
import torch.utils.data as dt
import matplotlib.pyplot as plt
from torchinfo import summary

import constants as cst
import modules.database as database
import modules.neuralnetwork as neuralnetwork

import matplotlib

matplotlib.use("Qt5Agg")  # to use interative backend, needs PyQt5 library

plt.style.use("ggplot")


def get_surface():
    """get_surface returns the surface

    Returns
    -------
    function
        surface function
    """
    return lambda vars_: sum([var_ ** 2 for var_ in vars_])


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
        dimension of the data (greater than 2), by default 2

    Yields
    -------
    Iterator[list]
        dim-D dataset
    """
    if dim < 2:
        lg.error(f"Wrong dimension! ({dim} < 2)")
        quit()
    num_mesh = int(np.ceil(np.exp(1 / (dim - 1) * np.log(num))))
    vars_ = [np.linspace(-scale, scale, num_mesh) for _ in range(dim - 1)]
    vars_ = np.meshgrid(*vars_)
    vars_.append(gen(vars_))
    reshaped = []
    for var in vars_:
        var = np.reshape(var, (1, -1))[0]
        reshaped.append(var)
    for i in range(num):
        yield [var[i] for var in reshaped]


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
    if len(data[0]) == 2:  # 2D plot
        _, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], color="blue", alpha=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    elif len(data[0]) == 3:  # 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], color="blue", alpha=0.5)
    else:
        lg.error(f"Can't plot this data! (dimension = {len(data[0])})")
        ax = None
    return ax


def main():
    """main creates the output directory, a dataset and trains the GAN to generate realistic fake data"""
    lg.info(
        f"Beginning of the script...\n----------------------\nLearning rate = {cst.LEARNING_RATE}\nBatch size    = {cst.BATCH_SIZE}\nEpochs        = {cst.NUM_EPOCHS}\n----------------------"
    )
    if not os.path.isdir("output"):
        os.mkdir("output")
        lg.info("'./output' directory did not exist, created it...")

    gen = get_surface()
    sampler = sample_data(10000, gen, scale=10, dim=3)

    lg.info("Loading data...")
    dataset = dt.DataLoader(database.Samples(sampler), batch_size=cst.BATCH_SIZE)

    dimension = list(dataset.dataset.__getitem__(0).shape)[0]

    generator = neuralnetwork.Generator(
        4, dimension, dropout=0.2, hidden_sizes=[32, 32, 16]
    )
    discriminator = neuralnetwork.Discriminator(dimension, 2, hidden_sizes=[16, 16])

    lg.info("Generator and Discriminator created:")
    print("### Generator ###")
    summary(generator, (cst.BATCH_SIZE, cst.LATENT_SIZE))
    print("### Discriminator ###")
    summary(discriminator, (cst.BATCH_SIZE, dimension))

    lg.info("Beginning training...")
    neuralnetwork.train_gan(
        dataset,
        generator,
        discriminator,
        epochs=cst.NUM_EPOCHS,
        lr=cst.LEARNING_RATE,
        write_graph=cst.WRITE_GRAPHS,
        n_test_graph=cst.N_TEST_GRAPH,
        display_test=cst.DISPLAY_TEST,
        live_plot=cst.LIVE_PLOT,
    )


### TESTS SCRIPT ###
if __name__ == "__main__":
    data = list(sample_data(100, get_surface(), dim=4))
    print(np.array(data).shape)
    ax = plot_data(data)
    plt.show()
