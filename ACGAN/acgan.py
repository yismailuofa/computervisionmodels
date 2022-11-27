import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt

# -----
# AC-GAN Build Blocks

class Generator(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.out_channels = out_channels

        # Architecture adapted from Erik Linder-Norén

        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.init_size = 32 // 4  # Initial size before upsampling
        self.l1 = nn.Linear(latent_dim, 128 * self.init_size ** 2)

        # Pass through series of batchnorm, upsample, leaky relu
        self.upscale = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, out_channels, 3, 1, 1),
            nn.Sigmoid(),
        )


        
    def forward(self, z, y):

        if torch.cuda.is_available():
            y = y.cuda()
            z = z.cuda()

        # Embed label and pass through a fc layer
        gen_input = torch.mul(self.label_emb(y), z)
        out = self.l1(gen_input)

        # Reshape for upscaling
        out = out.view(-1, self.latent_dim, self.init_size, self.init_size)

        return self.upscale(out)   

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels

        # Architecture adapted from Erik Linder-Norén

        dims = [16, 32, 64, 128]
        modules = []

        for d in dims:
            block = [
                nn.Conv2d(in_channels, d, 3, 2, 1),
                nn.LeakyReLU(0.2, True)
            ]

            # No batch norm on first layer
            if in_channels != self.in_channels:
                block.append(nn.BatchNorm2d(d, 0.8))
            
            in_channels = d

            modules.append(nn.Sequential(*block))

        self.conv = nn.Sequential(*modules)

        newSize = 32 // 2 ** 4
        
        # One layer to predict one to classify
        self.adv_fc = nn.Sequential(nn.Linear(128 * newSize ** 2, 1), nn.Sigmoid())
        self.aux_fc = nn.Sequential(nn.Linear(128 * newSize ** 2, 10), nn.Softmax(dim=1))


    def forward(self, x):

        out = self.conv(x)
        out = out.view(out.shape[0], -1)

        probs, label = self.adv_fc(out), self.aux_fc(out)

        return probs, label
        

# -----
# Hyperparameters
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# NOTE: Feel free to change the hyperparameters as long as you meet the marking requirement
batch_size = 256
workers = 0
latent_dim = 128
lr = 0.001
num_epochs = 150
validate_every = 1
print_every = 10

save_path = os.path.join(os.path.curdir, "visualize", "gan")
if not os.path.exists(os.path.join(os.path.curdir, "visualize", "gan")):
    os.makedirs(os.path.join(os.path.curdir, "visualize", "gan"))
ckpt_path = 'acgan.pt'

# -----
# Dataset
# NOTE: Data is only normalized to [0, 1]. THIS IS IMPORTANT!!!
tfms = transforms.Compose([
    transforms.ToTensor(), 
    ])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True,
    transform=tfms)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=workers)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True,
    transform=tfms)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size,
    shuffle=False, 
    num_workers=workers)


# -----
# Model
generator = Generator()

discriminator = Discriminator()

# Maybe also add normal init here
# Initialize weights
useCheckpoint = False
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

if useCheckpoint:
    checkpoint = torch.load('acgan.pt')
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])

else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)


# -----
# Losses

adv_loss = nn.BCELoss()
aux_loss = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    adv_loss = adv_loss.cuda()
    aux_loss = aux_loss.cuda()

# Optimizers for Discriminator and Generator, separate


optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))


# -----
# Train loop

def denormalize(x):
    """Denomalize a normalized image back to uint8.
    """


    # We move the channel back here so visualization can work
    return (x * 255).permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)

# For visualization part
# Generate 20 random sample for visualization
# Keep this outside the loop so we will generate near identical images with the same latent features per train epoch
random_z = torch.randn(20, latent_dim)
random_y = torch.randint(high=10, size=(20,),)


def train_step(x, y):
    """One train step for AC-GAN.
    You should return loss_g, loss_d, acc_d, a.k.a:
        - average train loss over batch for generator
        - average train loss over batch for discriminator
        - auxiliary train accuracy over batch
    """
    # Batch size
    N = x.shape[0]

    random_z_train = torch.randn(N, latent_dim)
    random_y_train = torch.randint(high=10, size=(N,),)

    realLabels = torch.full((N, 1), 1., dtype=torch.float32)
    fakeLabels = torch.full((N, 1), 0., dtype=torch.float32)

    if torch.cuda.is_available():
        realLabels = realLabels.cuda()
        fakeLabels = fakeLabels.cuda()
        random_z_train = random_z_train.cuda()
        random_y_train = random_y_train.cuda()            

    # GENERATOR
    optimizer_G.zero_grad()

    imgs = generator(random_z_train, random_y_train)

    probs, labels = discriminator(imgs)

    # Use real labels since we want to fool the discriminator
    loss_g = 0.5 * (adv_loss(probs, realLabels) + aux_loss(labels, random_y_train))

    loss_g.backward()
    optimizer_G.step()

    # DISCRIMINATOR 
    optimizer_D.zero_grad()

    # Train with all real
    probs_r, labels_r = discriminator(x)
    if torch.cuda.is_available():
        probs_r = probs_r.cuda()
        labels_r = labels_r.cuda()
    loss_d_real = 0.5*(adv_loss(probs_r, realLabels) + aux_loss(labels_r, y))
    # loss_d_real.backward()

    # Train with all fake
    probs_f, labels_f = discriminator(imgs.detach())
    if torch.cuda.is_available():
        probs_f = probs_f.cuda()
        labels_f = labels_f.cuda()    
    loss_d_fake = 0.5*(adv_loss(probs_f, fakeLabels) + aux_loss(labels_f, random_y_train))
    # loss_d_fake.backward()
    

    loss_d = 0.5*(loss_d_real + loss_d_fake)
    loss_d.backward()
    
    optimizer_D.step()    

    # Accuracy
    pred = torch.cat([labels_r, labels_f])
    gt = torch.cat([y, random_y_train])

    acc_d = torch.mean((torch.argmax(pred, 1) == gt).float())
    
    return loss_g.detach().cpu(), loss_d.detach().cpu(), acc_d.detach().cpu()

def test(
    test_loader,
    ):
    """Calculate accuracy over Cifar10 test set.
    """
    size = len(test_loader.dataset)
    corrects = 0

    discriminator.eval()
    with torch.no_grad():
        for inputs, gts in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                gts = gts.cuda()

            # Forward only
            _, outputs = discriminator(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == gts.data)

    acc = corrects / size
    print("Test Acc: {:.4f}".format(acc))
    return acc


g_losses = []
d_losses = []
best_acc_test = 0.0

for epoch in range(1, num_epochs + 1):
    generator.train()
    discriminator.train()

    avg_loss_g, avg_loss_d = 0.0, 0.0
    for i, (x, y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        # train step
        loss_g, loss_d, acc_d = train_step(x, y)
        avg_loss_g += loss_g * x.shape[0]
        avg_loss_d += loss_d * x.shape[0]

        # Print
        if i % print_every == 0:
            print("Epoch {}, Iter {}: LossD: {:.6f} LossG: {:.6f}, D_acc {:.6f}".format(epoch, i, loss_d, loss_g, acc_d))

    g_losses.append(avg_loss_g / len(train_dataset))
    d_losses.append(avg_loss_d / len(train_dataset))

    # Save
    if epoch % validate_every == 0:
        acc_test = test(test_loader)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            # Wrap things to a single dict to train multiple model weights
            state_dict = {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                }
            torch.save(state_dict, ckpt_path)
            print("Best model saved w/ Test Acc of {:.6f}.".format(best_acc_test))


        # Do some reconstruction
        generator.eval()
        with torch.no_grad():
            # Forward
            xg = generator(random_z, random_y)
            xg = denormalize(xg)

            # Plot 20 randomly generated images
            plt.figure(figsize=(10, 5))
            for p in range(20):
                plt.subplot(4, 5, p+1)
                plt.imshow(xg[p])
                plt.text(0, 0, "{}".format(classes[int(random_y[p].item())]), color='black',
                            backgroundcolor='white', fontsize=8)
                plt.axis('off')

            plt.savefig(os.path.join(os.path.join(save_path, "E{:d}.png".format(epoch))), dpi=300)
            plt.clf()
            plt.close('all')

        # Plot losses
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(g_losses, label="G")
        plt.plot(d_losses, label="D")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xlim([1, epoch])
        plt.legend()
        plt.savefig(os.path.join(os.path.join(save_path, "loss.png")), dpi=300)

print(best_acc_test)
