# _*_ coding: UTF-8 _*_
import os
import torch
import numpy as np
import sys
import cv2
import scipy.io as scio
import torchvision.models as models
from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model
from collections import OrderedDict

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)



method = 'DA-Net'
dataset_name = 'shtechA'
output_dir = './saved_models/'


train_path = '../data/shanghaiA_100patches/train/'
train_gt_path = '../data/shanghaiA_100patches/train_den/'
val_path = '../data/shanghaiA_100patches/test_data/images/'
val_gt_path = '../data/shanghaiA_100patches/test_data/ground_truth_csv/'


ine_tune_model = './saved_models/DA-Net_shtechA_70.h5'

#training configuration
start_step = 0
end_step = 1000
lr = 0.00001
momentum = 0.9
disp_interval = 5000
log_interval = 250
pretrained_vgg16 = True
fine_tune = False

#Tensorboard  config
use_tensorboard = False
save_exp_name = method + '_' + dataset_name + '_' + 'v1'
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------
rand_seed = 64678  
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


# load net
net = CrowdCounter()

network.weights_normal_init(net, dev=0.01)
if pretrained_vgg16:
    vgg16_model = models.vgg16(pretrained=True)
    # vgg16_model.cuda()
    net.DA_Net.copy_params_from_vgg16(vgg16_model)

net.cuda()
net.train()

# pirnt network:
# params_dict = OrderedDict(net.named_parameters())
# for key, value in params_dict.items():
#     print(key,value)

params = list(net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:    
        exp_name = save_exp_name 
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

### ROI
# img_shape  = (158, 238, 3)
# data = scio.loadmat('../data/UCSD_mcnn/mask.mat')
# mask = np.zeros(img_shape,dtype=np.uint8)
# mask = cv2.fillPoly(mask,data['density'].astype(int),(1,1,1))

data_loader = ImageDataLoader(train_path, train_gt_path,shuffle=True, gt_downsample=True, pre_load=False)
data_loader_val = ImageDataLoader(val_path, val_gt_path,shuffle=False, gt_downsample=True, pre_load=False)
best_mae = sys.maxint

for epoch in range(start_step, end_step+1):    
    step = -1
    train_loss = 0
    for blob in data_loader:                
        step = step + 1        
        im_data = blob['data']
        gt_data = blob['gt_density']
        # roi = blob['roi']
        density_map = net(im_data, gt_data)
        zzk_loss = net.loss
        train_loss += zzk_loss.data[0]
        step_cnt += 1
        optimizer.zero_grad()
        zzk_loss.backward()
        optimizer.step()
        
        if step % disp_interval == 0:            
            duration = t.toc(average=False)
            fps = step_cnt / duration
            gt_count = np.sum(gt_data)    
            density_map = density_map.data.cpu().numpy()
            et_count = np.sum(density_map)
            utils.save_results(im_data,gt_data,density_map, output_dir)
            log_text = 'epoch: %4d, step %4d, Time: %.4fs, gt_cnt: %4.1f, et_cnt: %4.1f,loss: %e' % (epoch,
                step, 1./fps, gt_count,et_count,zzk_loss.data[0])
            log_print(log_text, color='green', attrs=['bold'])
            re_cnt = True    
    
       
        if re_cnt:                                
            t.tic()
            re_cnt = False

    if (epoch % 10 == 0):
        save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method,dataset_name,epoch))
        network.save_net(save_name, net)     
        #calculate error on the validation dataset 
        mae,mse = evaluate_model(save_name, data_loader_val)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_model = '{}_{}_{}.h5'.format(method,dataset_name,epoch)
        log_text = 'EPOCH: %d, MAE: %.1f, MSE: %0.1f' % (epoch,mae,mse)
        log_print(log_text, color='green', attrs=['bold'])
        log_text = 'BEST MAE: %0.1f, BEST MSE: %0.1f, BEST MODEL: %s' % (best_mae,best_mse, best_model)
        log_print(log_text, color='green', attrs=['bold'])
        if use_tensorboard:
            exp.add_scalar_value('MAE', mae, step=epoch)
            exp.add_scalar_value('MSE', mse, step=epoch)
            exp.add_scalar_value('train_loss', train_loss/data_loader.get_num_samples(), step=epoch)
        
    

