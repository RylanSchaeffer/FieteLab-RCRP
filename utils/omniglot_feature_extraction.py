# copied from https://github.com/pytorch/examples/tree/master/mnist
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


def load_lenet(pretrained_lenet_path='utils/lenet_mnist_model.pth'):
    # TODO: automatically download pretrained model from
    # https://drive.google.com/drive/folders/1fn83DF14tWmit0RTKWRhPq5uVXt73e0h?usp=sharing

    # Initialize the network
    lenet = LeNet()

    # Load or train LeNet
    if not os.path.isfile(pretrained_lenet_path):
        lenet = train_and_save_lenet(lenet=lenet,
                                     pretrained_lenet_path=pretrained_lenet_path)
    lenet.load_state_dict(torch.load(pretrained_lenet_path, map_location='cpu'))

    # remove last layer with trick: set last layer weights to identity,
    # and set last layer bias to zeros
    lenet.fc2.weight.data = torch.eye(lenet.fc2.weight.data.shape[1])
    lenet.fc2.bias.data[:] = 0
    return lenet


def train_and_save_lenet(lenet,
                         pretrained_lenet_path: str,
                         batch_size: int = 64,
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
        train_step(model=lenet, device=device, train_loader=train_loader,
                   optimizer=optimizer, epoch=epoch, log_interval=log_interval)
        test_accuracy = test_step(model=lenet, device=device, test_loader=test_loader)
        if test_accuracy > 0.95:
            break
        scheduler.step()

    torch.save(lenet.state_dict(), pretrained_lenet_path)
    return lenet


def train_step(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
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


def test_step(model, device, test_loader):
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

    return test_accuracy


if __name__ == '__main__':
    load_lenet()
