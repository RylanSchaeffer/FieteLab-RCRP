# copied from https://github.com/pytorch/examples/tree/master/mnist
# and from https://github.com/AntixK/PyTorch-VAE/blob/8700d245a9735640dda458db4cf40708caf2e77f/tests/test_vae.py#L11
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class VanillaVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int = 16,
                 hidden_dims: tuple = None) -> None:

        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self._hidden_dims = hidden_dims

        # Build Encoder
        for hidden_dim in self._hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=hidden_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 9, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 9, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[-1],
                               out_channels=1,
                               kernel_size=3,
                               stride=3,
                               padding=1,
                               output_padding=0),  # (64, 32, 95, 95)
            nn.Sigmoid())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)  # output shape: 64, 32, 32, 32
        result = self.final_layer(result)  # output shape: 64, 1, 94, 94
        # cut to 80x80 matrix, to match Omniglot
        result = result[:, :, 7:-7, 7:-7]  # output shape: 64, 1, 80, 80
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        forward_result = dict(
            input=input,
            recon_input=self.decode(z),
            mu=mu,
            log_var=log_var)
        return forward_result

    def loss_function(self,
                      input,
                      recon_input,
                      mu,
                      log_var,
                      kld_weight: float = 0.5) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """

        recons_loss = F.mse_loss(recon_input, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(),
                                               dim=1),
                              dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


def cnn_load(pretrained_lenet_path='utils/lenet_mnist_model.pth'):
    # Create the LeNet network
    lenet = LeNet()

    # Load or train LeNet
    if not os.path.isfile(pretrained_lenet_path):
        lenet = cnn_train_and_save(lenet=lenet,
                                   pretrained_lenet_path=pretrained_lenet_path)
    lenet.load_state_dict(torch.load(pretrained_lenet_path, map_location='cpu'))

    # remove last layer with trick: set last layer weights to identity,
    # and set last layer bias to zeros
    lenet.fc2.bias.data = torch.zeros(lenet.fc2.weight.data.shape[1])
    lenet.fc2.weight.data = torch.eye(lenet.fc2.weight.data.shape[1])
    return lenet


def cnn_train_and_save(lenet,
                       pretrained_lenet_path: str,
                       batch_size: int = 128,
                       test_batch_size: int = 1000,
                       epochs: int = 14,
                       lr: float = 1.0,
                       gamma: float = 0.7,
                       seed: int = 1,
                       log_interval: int = 10):
    torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    lenet.to(device)
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    optimizer = optim.Adadelta(lenet.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        cnn_train_step(model=lenet, device=device, train_loader=train_loader,
                       optimizer=optimizer, epoch=epoch, log_interval=log_interval)
        _, test_accuracy = cnn_test_step(model=lenet, device=device, test_loader=test_loader)
        if test_accuracy > 0.95:
            break
        scheduler.step()

    torch.save(lenet.state_dict(), pretrained_lenet_path)
    return lenet


def cnn_train_step(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # import matplotlib.pyplot as plt
        # plt.hist(data.numpy().flatten())
        # plt.show()

        # plt.imshow(data.numpy()[0, 0], cmap='gray')
        # plt.show()

        # MNIST is 28 x 28 between 0 and 1
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def cnn_test_step(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_accuracy))

    return test_loss, test_accuracy


def vae_load(omniglot_dataset,
             pretrained_vae_path='utils/vae_full_omniglot_model.pth'):
    # Create the LeNet network
    vae = VanillaVAE(in_channels=1)

    # Load or train VAE
    lr = 1e-4
    gamma = 0.9  # change next
    pretrained_vae_path = f'utils/vae_omniglot=12k_lr={str(lr)}_gamma={gamma}_big.pth'
    print(f'Pretrained VAE path: {pretrained_vae_path}')
    if not os.path.isfile(pretrained_vae_path):
        vae = vae_train_and_save(vae=vae,
                                 omniglot_dataset=omniglot_dataset,
                                 pretrained_vae_path=pretrained_vae_path,
                                 lr=lr,
                                 gamma=gamma)
    vae.load_state_dict(torch.load(pretrained_vae_path, map_location='cpu'))
    return vae


def vae_train_and_save(vae,
                       omniglot_dataset,
                       pretrained_vae_path: str,
                       batch_size: int = 128,
                       epochs: int = 100,
                       lr: float = 1e-2,
                       gamma: float = 0.9,
                       seed: int = 1,
                       log_interval: int = 1):

    omniglot_dataloader = torch.utils.data.DataLoader(
        dataset=omniglot_dataset,
        batch_size=batch_size,
        shuffle=False)

    torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    vae.to(device)

    optimizer = optim.Adadelta(vae.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        vae_train_step(model=vae, device=device, train_loader=omniglot_dataloader,
                       optimizer=optimizer, epoch=epoch, log_interval=log_interval)
        # test_loss = vae_test_step(model=vae, device=device, test_loader=omniglot_dataloader)
        # if test_loss < 0.01:
        #     break
        scheduler.step()
        torch.save(vae.state_dict(), pretrained_vae_path)
    return vae


def vae_train_step(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        forward_result = model(data)
        loss_result = model.loss_function(
            input=forward_result['input'],
            recon_input=forward_result['recon_input'],
            mu=forward_result['mu'],
            log_var=forward_result['log_var'])
        loss_result['loss'].backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_result['loss'].item()))


def vae_test_step(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            forward_result = model(data)
            loss_result = model.loss_function(
                input=forward_result['input'],
                recon_input=forward_result['recon_input'],
                mu=forward_result['mu'],
                log_var=forward_result['log_var'])
            test_loss += loss_result['loss']

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

    return test_loss


if __name__ == '__main__':
    cnn_load()
    # vae_load()
