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
from typing import Optional

from ssim import SSIM

# -----
# VAE Build Blocks

class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        in_channels: int = 3,
        conditional: bool = False
        ):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        # For the label channel
        if conditional:
            in_channels += 1

        # Adapted architecture from A.K Subramanian's PyTorch VAE        
        self.encoder = nn.Sequential(
            # down sample 16 * 16
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # down sample 8 * 8
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # conv layer 6 * 6
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),  
            # conv layer 4 * 4
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # conv layer 2 * 2          
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),            
        )

        self.fc_mu = nn.Linear(512*4, latent_dim)
        self.fc_var = nn.Linear(512*4, latent_dim)

    def forward(self, x):

        # Pass through the encoder, and flatten result
        res = torch.flatten(self.encoder(x), 1)
        
        # Return our calculated mu and log_var
        return [self.fc_mu(res), self.fc_var(res)]

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 3,
        conditional: bool = False
        ):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        # Adapted architecture from A.K Subramanian's PyTorch VAE

        # Account for the possible extra input
        decoder_in_dims = latent_dim + 10 if conditional else latent_dim
        self.decoder_input = nn.Linear(decoder_in_dims, 512 * 4)

        # Build the opposite of the encoder for the U shape
        self.decoder = nn.Sequential(
            # conv layer 4 * 4       
            nn.Conv2d(512, 256, 3, 1, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # conv layer 6 * 6     
            nn.Conv2d(256, 128, 3, 1, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # conv layer 8 * 8
            nn.Conv2d(128, 64, 3, 1, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # up sample 16 * 16
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        # Up sample 32 * 32
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(32, 32, 3, 2 ,1 ,1),
                            nn.BatchNorm2d(32),
                            nn.LeakyReLU(),
                            nn.Conv2d(32, 3, 3, 1, 1),
                            nn.Sigmoid())
        
    def forward(self, z):

        # Pass the z through the head and reshape
        res = self.decoder_input(z).view(-1, 512, 2, 2)
        
        return self.final_layer(self.decoder(res))



# #####
# Wrapper for Variational Autoencoder
# #####

class VAE(nn.Module):
    def __init__(
        self, 
        latent_dim: int = 128,
        ):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encode = Encoder(latent_dim=latent_dim)
        self.decode = Decoder(latent_dim=latent_dim)

    def reparameterize(self, mu, log_var):
        """Reparameterization Tricks to sample latent vector z
        from distribution w/ mean and variance.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x, y):

        """Forward for CVAE.
        Returns:
            xg: reconstructed image from decoder.
            mu, log_var: mean and log(std) of z ~ N(mu, sigma^2)
            z: latent vector, z = mu + sigma * eps, acquired from reparameterization trick. 
        """

        # Get our params from the encoder and perform the trick
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)

        return [self.decode(z), mu, log_var, z]

    def generate(
        self,
        n_samples: int,
        ):

        """Randomly sample from the latent space and return
        the reconstructed samples.
        Returns:
            xg: reconstructed image
            None: a placeholder simply.
        """

        # randn gives us a gaussian
        z = torch.randn(n_samples,
                self.latent_dim)

        if torch.cuda.is_available():
            z = z.cuda()

        # move the channel to end for visualization
        return self.decode(z), None


# #####
# Wrapper for Conditional Variational Autoencoder
# #####

class CVAE(nn.Module):
    def __init__(
        self, 
        latent_dim: int = 128,
        num_classes: int = 10,
        img_size: int = 32,
        ):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size

        self.encode = Encoder(latent_dim=latent_dim, in_channels=3, conditional= True)
        self.decode = Decoder(latent_dim=latent_dim, conditional=True)

        # Adapted architecture from A.K Subramanian's PyTorch CVAE
        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(3, 3, kernel_size=1)


    def reparameterize(self, mu, log_var):
        """Reparameterization Tricks to sample latent vector z
        from distribution w/ mean and variance.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x, y):

        """Forward for CVAE.
        Returns:
            xg: reconstructed image from decoder.
            mu, log_var: mean and log(std) of z ~ N(mu, sigma^2)
            z: latent vector, z = mu + sigma * eps, acquired from reparameterization trick. 
        """

        # Convert label to a one hot for embedding
        y = nn.functional.one_hot(y, self.num_classes).float()

        embedded_class = self.embed_class(y).view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(x)

        # Pass this embedding to our encoder to get the params
        x_ = torch.cat([embedded_input, embedded_class], dim = 1)
        mu, log_var = self.encode(x_)

        # We can now generate the latent var and join it with the labels
        z = self.reparameterize(mu, log_var)

        z = torch.cat([z, y], dim=1)

        # now just decode the z to get your image

        return [self.decode(z), mu, log_var, z] 


    def generate(
        self,
        n_samples: int,
        y: Optional[torch.Tensor] = None,
        ):
        """Randomly sample from the latent space and return
        the reconstructed samples.
        NOTE: Randomly generate some classes here, if not y is provided.
        Returns:
            xg: reconstructed image
            y: classes for xg. 
        """
        # randn gives us a gaussian
        z = torch.randn(n_samples,
                self.latent_dim)

        # If we have labels, generate based on them, otherwise random
        if y:
            y_ = y
        else:
            y_ = nn.functional.one_hot(torch.randint(high=self.num_classes, size=(z.shape[0],)), self.num_classes).float()
        
        if torch.cuda.is_available():
            z = z.cuda()
            y_ = y_.cuda()

        z = torch.cat([z, y_], dim=1)

        # move the channel to end for visualization
        return self.decode(z), y


# #####
# Wrapper for KL Divergence
# #####

class KLDivLoss(nn.Module):
    def __init__(
        self,
        lambd: float = 1.0,
        ):
        super(KLDivLoss, self).__init__()
        self.lambd = lambd

    def forward(
        self, 
        mu, 
        log_var,
        ):
        loss = 0.5 * torch.sum(-log_var - 1 + mu ** 2 + log_var.exp(), dim=1)
        return self.lambd * torch.mean(loss)


# -----
# Hyperparameters
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
batch_size = 1
workers = 0
latent_dim = 128
lr = 0.0005
num_epochs = 10
validate_every = 1
print_every = 10

conditional = True     # Flag to use VAE or CVAE

if conditional:
    name = "cvae"
else:
    name = "vae"

# Set up save paths
if not os.path.exists(os.path.join(os.path.curdir, "visualize", name)):
    os.makedirs(os.path.join(os.path.curdir, "visualize", name))
save_path = os.path.join(os.path.curdir, "visualize", name)
ckpt_path = name + '.pt'


# We use an exponential KL annealing strategy which gradually increases to 1
kl_lams = [0.0, 0.00001, 0.0001, 0.0005, 0.001]

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

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True,
    transform=tfms,
    )

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=workers)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size,
    shuffle=False, 
    num_workers=workers)

subset = torch.utils.data.Subset(
    test_dataset, 
    [0, 380, 500, 728, 1000, 2300, 3400, 4300, 4800, 5000])

loader = torch.utils.data.DataLoader(
    subset, 
    batch_size=10)

# -----
# Model
if conditional:
    model = CVAE(latent_dim=latent_dim)
else:
    model = VAE(latent_dim=latent_dim)

# -----
# Losses
mseCrit = nn.MSELoss()
bceCrit = nn.BCELoss()
KLdivCrit = KLDivLoss(0)

def ssimCrit(input, target):
    return 1 - SSIM().forward(input, target)

best_total_loss = float("inf")

# Send to GPU
if torch.cuda.is_available():
    model = model.cuda()



optimizer = optim.Adam(model.parameters(), lr=lr)

# To further help with training
# NOTE: You can remove this if you find this unhelpful
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40, 50], 0.1)


# -----
# Train loop

# #####
# TODO: Complete train_step for VAE/CVAE
# #####

def train_step(x, y):
    """One train step for VAE/CVAE.
    You should return average total train loss(sum of reconstruction losses, _ divergence loss)
    and all individual average reconstruction loss (l2, bce, ssim) per sample.
    Args:
        x, y: one batch (images, labels) from Cifar10 train set.
    Returns:
        loss: total loss per batch.
        l2_loss: MSE loss for reconstruction.
        bce_loss: binary cross-entropy loss for reconstruction.
        ssim_loss: ssim loss for reconstruction.
        kldiv_loss: kl divergence loss.
    """
    
    xg, mu, log_var, _ = model(x, y)

    l2_loss = mseCrit(xg, x)
    bce_loss = bceCrit(xg, x)
    ssim_loss = ssimCrit(xg, x)
    kldiv_loss = KLdivCrit(mu, log_var)

    loss = l2_loss + ssim_loss + bce_loss + kldiv_loss
    
    return loss, l2_loss, bce_loss, ssim_loss, kldiv_loss

def denormalize(x):
    """Denomalize a normalized image back to uint8.
    Args:
        x: torch.Tensor, in [0, 1].
    Return:
        x_denormalized: denormalized image as numpy.uint8, in [0, 255].
    """
    # #####
    # TODO: Complete denormalization.
    # #####

    # We move the channel back here so visualization can work
    return (x * 255).permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)

# Loop HERE
l2_losses = []
bce_losses = []
ssim_losses = []
kld_losses = []
total_losses = []

total_losses_train = []

for epoch in range(1, num_epochs + 1):
    total_loss_train = total_loss_test = totalL2 = totalBCE = totalSSIM = totalKLD = avg_total_recon_loss_test = 0.0

    for i, (x, y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        # Train step
        model.train()
        optimizer.zero_grad()
        loss, l2_loss, bce_loss, ssim_loss, kldiv_loss = train_step(x, y)
        
        loss.backward()
        optimizer.step()

        total_loss_train += loss.cpu() * x.shape[0]

        totalL2 += l2_loss.cpu() * x.shape[0]
        totalBCE += bce_loss.cpu() * x.shape[0]
        totalSSIM += ssim_loss.cpu() * x.shape[0]
        totalKLD += kldiv_loss.cpu() * x.shape[0]
        
        # Print
        if i % print_every == 0:
            print("Epoch {}, Iter {}: Total Loss: {:.6f} MSE: {:.6f}, SSIM: {:.6f}, BCE: {:.6f}, KLDiv: {:.6f}".format(epoch, i, loss, l2_loss, ssim_loss, bce_loss, kldiv_loss))

    total_losses_train.append(total_loss_train / len(train_dataset))

    l2_losses.append(totalL2 / len(train_dataset))
    bce_losses.append(totalBCE / len(train_dataset))
    ssim_losses.append(totalSSIM / len(train_dataset))
    kld_losses.append(totalKLD / len(train_dataset))

    # Test loop
    if epoch % validate_every == 0:
        # Loop through test set
        model.eval()


        with torch.no_grad():
            for x, y in test_loader:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                xg, mu, log_var, _ = model(x, y)
                l2_loss = mseCrit(x, xg)
                bce_loss = bceCrit(x, xg)
                ssim_loss = ssimCrit(x, xg)
                kld_loss = KLdivCrit(mu, log_var)
                recon_loss = l2_loss + ssim_loss + bce_loss

                total_loss_test += x.shape[0]*(recon_loss + kld_loss).cpu()
                
                avg_total_recon_loss_test += x.shape[0] * recon_loss.cpu()

            avg_total_recon_loss_test /= len(test_dataset)

            total_losses.append(total_loss_test / len(test_dataset))

            # Plot losses
            if epoch > 1:
                plt.plot(l2_losses, label="L2 Reconstruction")
                plt.plot(bce_losses, label="BCE")
                plt.plot(ssim_losses, label="SSIM")
                plt.plot(kld_losses, label="KL Divergence")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.xlim([1, epoch])
                plt.legend()
                plt.savefig(os.path.join(os.path.join(save_path, "losses.png")), dpi=300)
                plt.clf()
                plt.close('all')

                plt.plot(total_losses, label="Total Loss Test")
                plt.plot(total_losses_train, label="Total Loss Train")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.xlim([1, epoch])
                plt.legend()
                plt.savefig(os.path.join(os.path.join(save_path, "total_loss.png")), dpi=300)
                plt.clf()
                plt.close('all')
            
            # Save best model
            if avg_total_recon_loss_test < best_total_loss:
                torch.save(model.state_dict(), ckpt_path)
                best_total_loss = avg_total_recon_loss_test
                print("Best model saved w/ Total Reconstruction Loss of {:.6f}.".format(best_total_loss))

        # Do some reconstruction
        model.eval()
        with torch.no_grad():
            x, y = next(iter(loader))
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            #y_onehot = F.one_hot(y, 10).float()
            xg, _, _, _ = model(x, y)

            # Visualize
            xg = denormalize(xg)
            x = denormalize(x)

            y = y.cpu().numpy()

            plt.figure(figsize=(10, 5))
            for p in range(10):
                plt.subplot(4, 5, p+1)
                plt.imshow(xg[p])
                plt.subplot(4, 5, p + 1 + 10)
                plt.imshow(x[p])
                plt.text(0, 0, "{}".format(classes[y[p].item()]), color='black',
                            backgroundcolor='white', fontsize=8)
                plt.axis('off')

            plt.savefig(os.path.join(os.path.join(save_path, "E{:d}.png".format(epoch))), dpi=300)
            plt.clf()
            plt.close('all')
            print("Figure saved at epoch {}.".format(epoch))

    # KL Annealing
    # Adjust scalar for KL Divergence loss
    
    # Scales the num of epochs linearly into buckets 
    kl_idx = int(min(epoch // (num_epochs / len(kl_lams)), len(kl_lams) - 1))
    KLdivCrit.lambd = kl_lams[kl_idx]
    
    print("Lambda:", KLdivCrit.lambd)

    # LR decay
    scheduler.step()
    
    print()

# Generate some random samples
if conditional:
    model = CVAE(latent_dim=latent_dim)
else:
    model = VAE(latent_dim=latent_dim)
if torch.cuda.is_available():
    model = model.cuda()
ckpt = torch.load(name+'.pt')
model.load_state_dict(ckpt)

# Generate 20 random images
xg, y = model.generate(20)
xg = denormalize(xg)
if y is not None:
    y = y.cpu().numpy()

plt.figure(figsize=(10, 5))
for p in range(20):
    plt.subplot(4, 5, p+1)
    if y is not None:
        plt.text(0, 0, "{}".format(classes[y[p].item()]), color='black',
                 backgroundcolor='white', fontsize=8)
    plt.imshow(xg[p])
    plt.axis('off')

plt.savefig(os.path.join(os.path.join(save_path, "random.png")), dpi=300)
plt.clf()
plt.close('all')

if conditional:
    min_val, max_val = 0.92, 1.0
else:
    min_val, max_val = 0.92, 1.0

print("Reconstruction loss:", best_total_loss)
