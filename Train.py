from CoarseNet import CoarseNet
from torchvision.transforms import ToPILImage, ToTensor, RandomResizedCrop, RandomRotation, RandomHorizontalFlip
from torchvision import transforms
from utils.preprocess import *
import torch
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn

custom_transforms = transforms.Compose([
    RandomResizedCrop(size=224, scale=(0.8, 1.2)),
    RandomRotation(degrees=(-30, 30)),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    RandomNoise(p=0.5, mean=0, std=0.1)])

train_dataset = PlacesDataset(txt_path='data/filelist.txt',
                              img_dir='data/data.tar',
                              transform=custom_transforms)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=2)

criterion = nn.MSELoss(reduction='mean')

coarsenet = CoarseNet()
optimizer = optim.Adam(coarsenet.parameters(), lr=0.0001)


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight)  # TODO there is a "uniform" version too. Check out which is correct
        m.bias.data.fill_(0.0)


coarsenet.apply(init_weights)


def train_model(net, data_loader, optimizer, criterion, epochs=2):
    """
    Train model
    :param net: Parameters of defined neural network
    :param data_loader: A data loader object defined on train data set
    :param epochs: Number of epochs to train model
    :param optimizer: Optimizer to train network
    :param criterion: The loss function to minimize by optimizer
    :return: None
    """

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs
            X = data['X']
            y_d = data['y_descreen']
            y_e = data['y_edge']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(X)
            loss = criterion(outputs, y_d)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')


train_model(coarsenet, train_loader, optimizer, criterion)
