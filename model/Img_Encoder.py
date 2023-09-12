import torch
import torch.nn as nn
import numpy as np
import sys
import os

ROOT_DIR = os.path.abspath('/data1/zhangliyuan/code/IMFNet_exp')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import model.resnet as resnet

# C:\Users\lenovo/.cache\torch\hub\checkpoints\resnet34-333f7ec4.pth
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()

        self.backbone = resnet.resnet34(in_channels=3, pretrained=True, progress=True)

    def forward(self, x):
        result = self.backbone(x)
        # return resnet_out[2],resnet_out[3],resnet_out[4], resnet_out[5]
        return result
if __name__ == '__main__':

    data = torch.zeros(size=(32,3,160,120))
    ie = ImageEncoder()
    result = ie(data)
    I1,I2,I3 = result[0],result[1],result[2]
    print(I1.shape)
    print(I2.shape)
    print(I3.shape)

