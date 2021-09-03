import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.nn.modules.loss import BCEWithLogitsLoss

class Generator(nn.Module):
    def __init__(self, input_size:int, output_size:int, hidden_sizes:list=[16, 16], relu_slope:float=.01):
        super(Generator, self).__init__()
        self.input_size = input_size # save the input size

        dense_layers = [] # list of nn.Modules to be used
        for h in hidden_sizes: # build the list given the different hidden layers sizes
            dense = nn.Linear(in_features=input_size, out_features=h)
            activation = nn.LeakyReLU(negative_slope=relu_slope)
            dense_layers.append(dense)
            dense_layers.append(activation)
            input_size = h # update the input_size for the next dense layer
        # unpack the list in a Sequential object
        self.dense_stack = nn.Sequential(*dense_layers)
        # build the ouput layer
        self.output = nn.Linear(in_features=input_size, out_features=output_size)
    
    def forward(self, x):
        features = self.dense_stack(x)
        y = self.output(features)
        return y

class Discriminator(nn.Module):
    def __init__(self, input_size:int, feature_size:int, hidden_sizes:list=[16, 16], relu_slope:float=.01):
        super(Discriminator, self).__init__()
        dense_layers = [] # list of nn.Modules to be used
        for h in hidden_sizes: # build the list given the different hidden layers sizes
            dense = nn.Linear(in_features=input_size, out_features=h)
            activation = nn.LeakyReLU(negative_slope=relu_slope)
            dense_layers.append(dense)
            dense_layers.append(activation)
            input_size = h # update the input_size for the next dense layer
        # add the feature output layer
        dense_layers.append(nn.Linear(in_features=input_size, out_features=feature_size))
        # unpack the list in a Sequential object
        self.dense_stack = nn.Sequential(*dense_layers)
        # build the output layer
        self.output = nn.Linear(in_features=feature_size, out_features=1)
    
    def forward(self, x):
        features = self.dense_stack(x)
        logit = self.output(features)
        return logit, features

def gen_loss(target, output):
    loss = BCEWithLogitsLoss()(output, target)
    return loss

def disc_loss(real_target, real_output, fake_target, fake_output):
    loss_real = BCEWithLogitsLoss()(real_output, real_target)
    loss_fake = BCEWithLogitsLoss()(fake_output, fake_target)
    return loss_real + loss_fake

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
        preds_real, _ = discriminator(T.float())
        preds_fake, _ = discriminator(generated.float())
        dloss = disc_loss(labels_real, preds_real, labels_fake, preds_fake)
        dloss.backward()
        dis_optimizer.step()
        # train the generator
        gen_optimizer.zero_grad()
        noise = torch.randn([T.shape[0], gen_input_size])
        preds, _ = discriminator(generator(noise.float()))
        gloss = gen_loss(labels_real, preds) # fooling the discriminator with 'real' labels to compute generator's loss
        gloss.backward()
        gen_optimizer.step()

    discloss = dloss.item()
    genloss = gloss.item()
    
    return discloss, genloss

def write_test_graph(generator, noise_bench, title=None):
    with torch.no_grad():
        preds = generator(noise_bench).numpy()
        plt.figure()
        plt.scatter(preds[:, 0], preds[:, 1], color='red', marker='+')
        if not title:
            plt.savefig(f'output.png')
        else:
            plt.savefig(title)

def train_gan(dataloader, generator, discriminator, lr=1e-3, epochs=100, write_graph:bool=False, n_test_graph:int=200):
    dloss, gloss = [], []
    gen_input_size = generator.input_size
    noise_bench = torch.randn([n_test_graph, gen_input_size])
    for t in range(epochs):
        d, g = train_loop(dataloader, generator, discriminator, lr=lr)
        print(f'Epoch {t} -- Disc loss: {d:>7f} | Gen loss: {g:>7f}', end='\r', flush=True)
        dloss.append(d)
        gloss.append(g)
        if write_graph:
            if epochs<=10 or t % (epochs//5):
                title = f'epoch_{t}.png'
                write_test_graph(generator, noise_bench, title)
        if g<.0001 or abs(d-g)<0.001:
            write_test_graph(generator, noise_bench)
            break
    print('')