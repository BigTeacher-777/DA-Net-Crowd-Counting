import torch
import torch.nn as nn
import network
from src import utils
from torch.autograd import Variable, grad
from model import Fine
import cv2
import torchvision.models as models

class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()
        self.fine = Fine()
        self.loss_fn = nn.MSELoss()

        

    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self,im_data,gt_data):
        im_data = network.np_to_variable(im_data,is_cuda=True,is_training=self.training)
        final = self.DA_Net(im_data)

        if self.training:
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)

            self.loss_mse = self.build_loss(final,gt_data)
            return final
        else:
            return final
    
    def build_loss(self, density_map, gt_data):
        loss = self.loss_fn(density_map, gt_data)
        return loss


