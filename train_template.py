from torchvision.transforms import ToPILImage, ToTensor, RandomResizedCrop, RandomRotation, RandomHorizontalFlip
from torchvision import transforms
from utils.preprocess import *
import torch
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

custom_transforms = transforms.Compose([
    RandomResizedCrop(size=224, scale=(0.8, 1.2)),
    RandomRotation(degrees=(-30, 30)),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    RandomNoise(p=0.5, mean=0, std=0.1)])


# https://discuss.pytorch.org/t/what-does-pil-images-of-range-0-1-mean-and-how-do-we-save-images-as-that-format/2103
train_dataset = PlacesDataset(txt_path='data/filelist.txt',
                              img_dir='data.tar',
                              transform=custom_transforms)

# test_dataset = PlacesDataset(txt_path='data_test/filelist.txt', img_dir='data_test.tar', transform=ToTensor())
# test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=2)

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(params, lr=0.0001)


def init_weights(m):
    torch.nn.init.kaiming_normal_(m.weight)
    m.bias.data.fill_(0.0)


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


def test_model_sample(net, data_loader):
    """
    Return array of PIL result images
    :param net: The NN network
    :param data_loader: Data loader containing test set
    :return: array of PIL images edited by net
    """
    array = []
    with torch.no_grad():
        for data in data_loader:
            X = data['X']
            output = net(X)
            output = ToPILImage()(output)
            array.append(output)
    return array


def test_model(net, data_loader):
    """
    Return loss on test set
    :param net: The trained NN network
    :param data_loader: Data loader containing test set
    :return: Loss value over test set in console
    """
    running_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            X = data['X']
            y_d = data['y_descreen']
            outputs = net(X)
            loss = criterion(outputs, y_d)
            running_loss += loss
    return running_loss


def show_sample(max_len):
    """
    Shows some samples and their sizes as tensor to see how preprocessing works
    :param max_len: number of samples to show
    :return: None -> print some data on console
    """
    for i in range(len(train_dataset)):
        sample = train_dataset[i]

        X = sample['X']
        y_d = sample['y_descreen']
        y_e = sample['y_edge']
        print(i)

        print(type(X), X.size())
        print(type(y_d), y_d.size())
        print(type(y_e), y_e.size())

        if i == max_len:
            break
