# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:40:16 2020

@author: CT_NT_CNY
"""

# from __future__ import print_function
# import argparse
# import torch
# import torch.utils.data
# from torch import nn, optim
# from torch.nn import functional as F
# from torchvision import datasets, transforms
# from torchvision.utils import save_image


# parser = argparse.ArgumentParser(description='VAE MNIST Example')
# parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()

# torch.manual_seed(args.seed)

# device = torch.device("cuda" if args.cuda else "cpu")

# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)


# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()

#         self.fc1 = nn.Linear(784, 400)
#         self.fc21 = nn.Linear(400, 20)
#         self.fc22 = nn.Linear(400, 20)
#         self.fc3 = nn.Linear(20, 400)
#         self.fc4 = nn.Linear(400, 784)

#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, 784))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar


# model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)


# # Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     return BCE + KLD


# def train(epoch):
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader):
#         data = data.to(device)
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data)
#         loss = loss_function(recon_batch, data, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.item() / len(data)))

#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#           epoch, train_loss / len(train_loader.dataset)))


# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n],
#                                       recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(),
#                          'results/reconstruction_' + str(epoch) + '.png', nrow=n)

#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))

# if __name__ == "__main__":
#     for epoch in range(1, args.epochs + 1):
#         train(epoch)
#         test(epoch)
#         with torch.no_grad():
#             sample = torch.randn(64, 20).to(device)
#             sample = model.decode(sample).cpu()
#             save_image(sample.view(64, 1, 28, 28),
#                        'results/sample_' + str(epoch) + '.png')

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import copy
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def make_batch(img_list, batch_size):
    i = 0
    train_dataset = []
    tmp = []
    for each in img_list:
        if i == batch_size:
            i = 0
            train_dataset.append(tmp)
            tmp = []
        img = cv2.imread(each)
        img = cv2.resize(img, (20, 100), interpolation=cv2.INTER_AREA)
        
        img = np.moveaxis(img, -1, 0)
        tmp.append(img.astype("float32")/255.0)
        i += 1
    # import ipdb; ipdb.set_trace()
    # Y_data = np.vstack(train_dataset)
    # Z_data = copy.deepcopy(Y_data)
    # Z_data = (Z_data - Z_data.mean()) / Z_data.std()
    return train_dataset

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                     transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)


train_imgs = glob.glob("C:/Users/CT_NT_CNY/Desktop/python test1/mel ae/image/image cut/*.png")
test_imgs = glob.glob("C:/Users/CT_NT_CNY/Desktop/python test1/mel ae/image/image cut/*.png")
train_loader = make_batch(train_imgs, 128)
test_loader = make_batch(test_imgs, 128)

train_loader = torch.FloatTensor(train_loader)
test_loader = torch.FloatTensor(test_loader)


batch_size = 128
learning_rate = 0.0002
num_epoch = 5

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder_1 = nn.Linear(6000,1000)
        self.encoder_2 = nn.Linear(1000,526)
        self.encoder_3 = nn.Linear(526,128)
        # self.latent = nn.Linear(128,128)
        self.decoder_1 = nn.Linear(128,526)
        self.decoder_2 = nn.Linear(526,1000)
        self.decoder_3 = nn.Linear(1000,6000) 
                
    def forward(self,x):
        x = x.view(batch_size,-1)
        encoded = self.encoder_1(x)
        encoded = self.encoder_2(encoded)
        encoded = self.encoder_3(encoded)
        # encoded = self.latent(encoded)
        decoded = self.decoder_1(encoded)
        decoded = self.decoder_2(decoded)
        out = self.decoder_3(decoded).view(batch_size,3,100,20)
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = Autoencoder().to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_arr =[]
for i in range(num_epoch):
    for j,image in enumerate(train_loader):
        x = image.to(device)
        
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output,x)
        loss.backward()
        optimizer.step()
        
    if j % 1000 == 0:
        print(loss)
        loss_arr.append(loss.cpu().data.numpy()[0])
        
out_img = torch.squeeze(output.cpu().data)
print(out_img.size())

img_tmp = []
out_tmp = []
for i in range(2):
    img = image[i].numpy()
    out = out_img[i].numpy()
    img_tmp.append(img)
    out_tmp.append(out)
for i in range(2):
    plt.subplot(1,2,1)
    img = np.moveaxis(img_tmp[i], 0, -1)
    
    # plt.imshow(torch.squeeze(img),cmap='gray')
    plt.imshow(img)
    plt.subplot(1,2,2)
    out_img = np.moveaxis(out_tmp[i], 0, -1)
    # plt.imshow(out_img[i].numpy(),cmap='gray')
    plt.imshow(out_img)


'''
with torch.no_grad():
  for i in range(1):
      for j,[image,label] in enumerate(test_loader):
          x = image.to(device)

          optimizer.zero_grad()
          output = model.forward(x)

      if j % 1000 == 0:
          print(loss)

out_img = torch.squeeze(output.cpu().data)
print(out_img.size())

for i in range(2):
    plt.subplot(1,2,1)
    plt.imshow(torch.squeeze(image[i]).numpy(),cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(out_img[i].numpy(),cmap='gray')
    plt.show()        
'''



























'''

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    # for batch_idx, (data, _) in enumerate(train_loader):
    batch_idx = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
        batch_idx += 1

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss /(len(train_loader) * len(data))))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        # for i, (data, _) in enumerate(test_loader):
        i = 0
        for data in test_loader:
            # import ipdb; ipdb.set_trace()
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 3, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
            i += 1

    test_loss /=(len(test_loader) * len(data))
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
'''