# Different tests on generating realistic data

The aimed distribution is y = -2x²+5x, with a training dataset of 10000 points 10 times scaled.

## Learning rate effects

First we don't modify the models :

| Model | Hidden layers | Dropout |
| :---: | :-----------: | :-----: |
| Generator | [32, 32, 16] | p=0.5 |
| Discriminator | [16, 16] | . |

The batch size is 512.

### Dropout randomness

Even if the generator's dropout is quite high, training with the same hyperparameters gives quite robust (and good) results.

With a learning rate = 5e-4 and 1000 epochs:

* first try:
![lr5e-4epochs1000](./study_assets/lr5e-4_epochs1000_1.PNG)
* second try:
![lr5e-4epochs1000](./study_assets/lr5e-4_epochs1000_2.PNG)
* third try:
![lr5e-4epochs1000](./study_assets/lr5e-4_epochs1000_3.PNG)

### Learning rates tests

We now that higher learning rates allows the GAN to approximate the aiming distribution faster, and it seems to do a great jobs below lr=1e-2.

In addition, after lr=1e-3, we see that the losses are quite unstable (even if the distribution is fairly approximated), especially the generator's loss.

* learning rate = 1e-4:
![lr1e-4epochs1000](./study_assets/lr1e-4_epochs1000_2.PNG)
* learning rate = 5e-4:
![lr5e-4epochs1000](./study_assets/lr5e-4_epochs1000_1.PNG)
* learning rate = 1e-3:
![lr1e-3epochs1000](./study_assets/lr1e-3_epochs1000_1.PNG)
* learning rate = 5e-3:
![lr5e-3epochs1000](./study_assets/lr5e-3_epochs1000_1.PNG)
* learning rate = 1e-2:
![lr1e-2epochs1000](./study_assets/lr1e-2_epochs1000_1.PNG)



