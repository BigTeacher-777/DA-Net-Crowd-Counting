# import torch.nn as nn
import network
import torch
import torch.nn as nn
from deform_conv import DeformConv2D
# from network import Conv2d
# from models import MCNN


class Fine(nn.Module):
    def __init__(self):
        super(Fine, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        #conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)

        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.conv6_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.relu6_2 = nn.ReLU(inplace=True)
        self.conv6_3 = nn.Conv2d(128, 64, 3, padding=1)
        self.relu6_3 = nn.ReLU(inplace=True)

        self.conv7_1 = nn.Conv2d(64, 32, 3, padding=1)
        self.relu7_1 = nn.ReLU(inplace=True)
        self.conv7_2 = nn.Conv2d(32, 16, 3, padding=1)
        self.relu7_2 = nn.ReLU(inplace=True)
        self.conv7_3 = nn.Conv2d(16, 8, 3, padding=1)
        self.relu7_3 = nn.ReLU(inplace=True)

        self.conv8_1 = nn.Conv2d(8, 4, 3, padding=1)
        self.relu8_1 = nn.ReLU(inplace=True)
        self.conv8_2 = nn.Conv2d(4, 2, 3, padding=1)
        self.relu8_2 = nn.ReLU(inplace=True)
        self.conv8_3 = nn.Conv2d(2, 1, 3, padding=1)
        self.relu8_3 = nn.ReLU(inplace=True)

        self.offsets4 = nn.Conv2d(512, 18, kernel_size=3, padding=1)
        self.conv4 = DeformConv2D(512, 1, 3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.offsets5 = nn.Conv2d(512, 18, kernel_size=3, padding=1)
        self.conv5 = DeformConv2D(512, 1, 3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)

        self.offsets6 = nn.Conv2d(64, 18, kernel_size=3, padding=1)
        self.conv6 = DeformConv2D(64, 1, 3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)

        self.offsets7 = nn.Conv2d(8, 18, kernel_size=3, padding=1)
        self.conv7 = DeformConv2D(8, 1, 3, padding=1)
        self.relu7 = nn.ReLU(inplace=True)       
               

        self.w4 = nn.Parameter(torch.ones(1))
        self.w5 = nn.Parameter(torch.ones(1))
        self.w6 = nn.Parameter(torch.ones(1))
        self.w7 = nn.Parameter(torch.ones(1))
        self.w8 = nn.Parameter(torch.ones(1))


    
    def forward(self, im_data):        
        h = im_data
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h) #1/4
        
         

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h) # 1/8
         
        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h) # 1/16

        deform4 = h
        offsets4 = self.offsets4(deform4)
        deform4 = self.relu4(self.conv4(deform4,offsets4))


        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))

        deform5 = h
        offsets5 = self.offsets5(deform5)
        deform5 = self.relu5(self.conv5(deform5,offsets5))

        h = self.relu6_1(self.conv6_1(h))
        h = self.relu6_2(self.conv6_2(h))
        h = self.relu6_3(self.conv6_3(h))

        deform6 = h
        offsets6 = self.offsets6(deform6)
        deform6 = self.relu6(self.conv6(deform6,offsets6))

        h = self.relu7_1(self.conv7_1(h))
        h = self.relu7_2(self.conv7_2(h))
        h = self.relu7_3(self.conv7_3(h))

        deform7 = h
        offsets7 = self.offsets7(deform7)
        deform7 = self.relu7(self.conv7(deform7,offsets7))
 
        h = self.relu8_1(self.conv8_1(h))
        h = self.relu8_2(self.conv8_2(h))
        h = self.relu8_3(self.conv8_3(h))

        pool8 = h

        # density_map = deform6
        # print self.w4,self.w5,self.w6,self.w7,self.w8
        density_map = self.w4*deform4 + self.w5*deform5 + self.w6*deform6 + self.w7*deform7 + self.w8*pool8


            
        return density_map

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if l2 == self.conv1_1:
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    # print l1.weight.data.size(), l1.bias.data.size()
                    # print l2.weight.data.size(), l2.bias.data.size()
                    l2.weight.data.copy_(l1.weight.data[:, 0:1, :, :])
                    l2.bias.data.copy_(l1.bias.data)
                continue
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        
