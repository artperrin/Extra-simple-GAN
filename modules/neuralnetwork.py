import torch
import logging as lg
from torch import nn
import matplotlib.pyplot as plt
from torch.nn.modules.loss import BCEWithLogitsLoss


class Generator(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list = [16, 16],
        relu_slope: float = 0.01,
    ):
        super(Generator, self).__init__()
        self.input_size = input_size  # save the input size

        dense_layers = []  # list of nn.Modules to be used
        for h in hidden_sizes:  # build the list given the different hidden layers sizes
            dense = nn.Linear(in_features=input_size, out_features=h)
            activation = nn.LeakyReLU(negative_slope=relu_slope)
            dense_layers.append(dense)
            dense_layers.append(activation)
            input_size = h  # update the input_size for the next dense layer
        # unpack the list in a Sequential object
        self.dense_stack = nn.Sequential(*dense_layers)
        # build the ouput layer
        self.output = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x):
        features = self.dense_stack(x)
        y = self.output(features)
        return y


class Discriminator(nn.Module):
    def __init__(
        self,
        input_size: int,
        feature_size: int,
        hidden_sizes: list = [16, 16],
        relu_slope: float = 0.01,
    ):
        super(Discriminator, self).__init__()
        dense_layers = []  # list of nn.Modules to be used
        for h in hidden_sizes:  # build the list given the different hidden layers sizes
            dense = nn.Linear(in_features=input_size, out_features=h)
            activation = nn.LeakyReLU(negative_slope=relu_slope)
            dense_layers.append(dense)
            dense_layers.append(activation)
            input_size = h  # update the input_size for the next dense layer
        # add the feature output layer
        dense_layers.append(
            nn.Linear(in_features=input_size, out_features=feature_size)
        )
        # unpack the list in a Sequential object
        self.dense_stack = nn.Sequential(*dense_layers)
        # build the output layer
        self.output = nn.Linear(in_features=feature_size, out_features=1)

    def forward(self, x):
        features = self.dense_stack(x)
        logit = self.output(features)
        return logit, features


def loss(target, output):
    loss_ = BCEWithLogitsLoss()(output, target)
    return loss_


def train_loop(dataloader, generator, discriminator, lr=1e-3):
    gen_input_size = generator.input_size
    gen_optimizer = torch.optim.RMSprop(params=generator.parameters(), lr=lr)
    dis_optimizer = torch.optim.RMSprop(params=discriminator.parameters(), lr=lr)
    discloss, genloss = [], []
    for T in dataloader:
        # generate data from noise
        noise = torch.randn([T.shape[0], gen_input_size])
        generated = generator(noise)
        # create labels
        labels_fake = torch.zeros([T.shape[0], 1])
        labels_real = torch.ones([T.shape[0], 1])
        # train the discriminator
        dis_optimizer.zero_grad()
        # (1) with real data
        preds_real, _ = discriminator(T.float())
        dloss_real = loss(labels_real, preds_real)
        dloss_real.backward()
        # (2) with fake data
        preds_fake, _ = discriminator(generated.detach())
        dloss_fake = loss(labels_fake, preds_fake)
        dloss_fake.backward()
        # update the loss and discriminator
        dloss = dloss_real + dloss_fake
        dis_optimizer.step()
        # train the generator
        gen_optimizer.zero_grad()
        preds, _ = discriminator(generated)
        gloss = loss(
            labels_real, preds
        )  # fooling the discriminator with 'real' labels to compute generator's loss
        gloss.backward()
        gen_optimizer.step()

    discloss = dloss.item()
    genloss = gloss.item()

    return discloss, genloss


def write_test_graph(generator, noise_bench, title=None, overlay=None):
    with torch.no_grad():
        preds = generator(noise_bench).numpy()
        plt.figure()
        plt.scatter(preds[:, 0], preds[:, 1], color="red", marker="+")
        if overlay:
            plt.scatter(overlay[0], overlay[1], color="blue", marker=".", alpha=0.5)
        if not title:
            plt.savefig(f"output.png")
        else:
            plt.savefig(title)


def train_gan(
    dataloader,
    generator,
    discriminator,
    lr=1e-3,
    epochs=100,
    write_graph: bool = False,
    n_test_graph: int = 200,
    display_test: bool = False,
):
    dloss, gloss = [], []
    gen_input_size = generator.input_size
    noise_bench = torch.randn([n_test_graph, gen_input_size])
    try:
        lg.info("Beginning training...")
        for t in range(epochs):
            d, g = train_loop(dataloader, generator, discriminator, lr=lr)
            print(
                f"Epoch {t} -- Disc loss: {d:>7f} | Gen loss: {g:>7f}",
                end="\r",
                flush=True,
            )
            dloss.append(d)
            gloss.append(g)
            if write_graph:
                if epochs <= 10 or t % (epochs // 10):
                    title = f"./output/epoch_{t}.png"
                    write_test_graph(generator, noise_bench, title)
            if g < 0.0001 or d < 0.0001 or abs(d - g) < 0.001:
                print("")
                lg.info(f"Training stopped after epoch {t+1}...")
                if write_graph:
                    write_test_graph(
                        generator, noise_bench, title=f"./output/epoch_{t}(last).png"
                    )
                break
        print("")
    except KeyboardInterrupt:
        print("")
        lg.warning(f"Training interrupted at epoch {t}!")

    lg.info("Plotting training info...")
    plt.figure()
    plt.plot(dloss, label="Discriminator")
    plt.plot(gloss, label="Generator")
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (BCE with logits)")
    plt.savefig("./output/training_graph.png")
    if display_test:
        lg.info("Generating a test set...")
        dataset = dataloader.dataset
        data = [dataset.__getitem__(i).view(1, -1) for i in range(len(dataset))]
        data = torch.cat(data, dim=0).numpy()
        write_test_graph(
            generator,
            noise_bench,
            title="./output/display_test_set.png",
            overlay=(data[:, 0], data[:, 1]),
        )
