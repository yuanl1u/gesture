import numpy as np
import numpy as np
import cv2
import sys
import os

def saveFile(data, name, path='DATA' ):
    p = os.listdir(os.path.join(path, name))
    num = len(p) + 1
    num = str(num)
    address = os.path.join(path, name, num + '.jpg')
    # np.save(address, data)
    cv2.imwrite(address, data)
    print('save file : ', address)


def findLocalmaximum(L):
    L[0:70] = 0
    L[130:] = 0
    out = np.array([])
    for i in range(5, len(L) - 5, 1):
        if L[i] > L[i-5] and L[i] > L[i+5]:
            out = np.append(out, i)

    ave = np.floor(np.mean(out))
    for i in range(int(ave), len(L) - 5, 1):
        if L[i] <= L[i-5] and L[i] <= L[i+5]:
            minimum2 = i
            break
    for i in range(int(ave), 5, -1):
        if L[i] <= L[i - 5] and L[i] <= L[i + 5]:
            minimum1 = i
            break

    return ave, minimum1, minimum2

def newMask(mask, trans):
    new_mask = np.zeros_like(mask)
    for j in range(new_mask.shape[1]):
        if (j + trans < new_mask.shape[1]) and (j + trans >= 0):
            new_mask[:, j + trans] = mask[:, j]
    return new_mask

import torch
import torch.nn as nn
import torch.nn.functional as F
import Learn

def loadModel():
    net = Learn.Net()
    parameter = torch.load('model_parameters.zip')
    net.load_state_dict(parameter)
    net.eval()
    return net

def eval(net, data):
    data = data[np.newaxis, :, :, :]
    data = torch.from_numpy(data)
    data = data.permute(0, 3, 2, 1)
    out = net(data)
    pred_choice = out.data.max(1)[1]
    return pred_choice
