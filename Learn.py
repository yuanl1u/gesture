import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import cv2
def loadFile(name, path='DATA'):
    p = os.path.join(path, name)
    datas = []
    for f in os.listdir(p):
        # data = np.load(os.path.join(path, name, f))
        data = cv2.imread(os.path.join(path, name, f))
        datas.append(data)
    datas = np.array(datas)
    return datas

source = 'g'


class GestureDataset(Dataset):
    def __init__(self):
        g1 = loadFile(source + '1')
        g2 = loadFile(source + '2')
        g3 = loadFile(source + '3')
        g4 = loadFile(source + '4')
        g5 = loadFile(source + '5')
        label1 = np.ones([len(g1), 1]) * 0
        label2 = np.ones([len(g2), 1]) * 1
        label3 = np.ones([len(g3), 1]) * 2
        label4 = np.ones([len(g4), 1]) * 3
        label5 = np.ones([len(g5), 1]) * 4
        data = np.concatenate((g1, g2, g3, g4, g5), axis=0).astype(np.float)
        data = data[:, np.newaxis, :, :]
        label = np.concatenate((label1, label2, label3, label4, label5), axis=0).astype(np.long)



        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label).view(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


import torchvision
net = torchvision.models.resnet18(pretrained=True)
net.fc = nn.Linear(512, 5)
net.cuda()

batchsize = 32
Epoch = 5
lr = 1e-3

optimizer = optim.Adam(net.parameters(), lr=lr)

if __name__ == '__main__':
    dataset = GestureDataset()
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

    for epoch in range(Epoch):
        net.train()
        correct = 0
        sum = 0

        for idx, (data, label) in enumerate(loader):
            data = data.permute(0, 4, 2, 3, 1)
            data = data.view(-1, 3, 480, 640)
            optimizer.zero_grad()
            label = label.long()
            data = data.cuda().float()
            label = label.cuda()
            data = F.adaptive_max_pool2d(data, 224)
            out = net(data)
            loss = F.nll_loss(out, label)
            loss.backward()
            optimizer.step()
            pred_choice = out.data.max(1)[1]
            correct += pred_choice.eq(label.data).cpu().sum().float()

            sum += len(label)

            print(epoch, loss.item(), 'correct rate:',correct/sum, correct.data.cpu(), sum)

    print('complete 10 epochs, saving model!!!')
    torch.save(net.state_dict(), 'model_parameters.zip')
    torch.save(net, 'model.zip')