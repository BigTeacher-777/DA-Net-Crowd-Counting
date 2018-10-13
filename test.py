# _*_ coding:UTF-8 _*_
import os
import torch
import numpy as np

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = False

data_path = '../ShanghaiTech/part_A_final/test_data/images/'
gt_path = '../ShanghaiTech/part_A_final/test_data/after_ground_truth/'
model_path = '../deformable/baseline/DA-Net_shtechA_80.h5'

output_dir = './output_after/'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
txt = open('1.txt','w')
# txt1 = open('gt.txt','w')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


net = CrowdCounter()
      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()
mae = 0.0
mse = 0.0

#load test data
data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=False)

for blob in data_loader:                        
    im_data = blob['data']
    gt_data = blob['gt_density']
    height = im_data.shape[2]
    width = im_data.shape[3]
    patches = 1
    height_tmp = 0
    et_count = 0
    # for i in range(patches):
    #     width_tmp = 0
    #     for j in range(patches):
    #         im_data_tmp = im_data[:,:,height_tmp:height_tmp+height/patches,width_tmp:width_tmp+width/patches]
    #         density_map_tmp = net(im_data_tmp, gt_data)
    #         density_map_tmp = density_map_tmp.data.cpu().numpy()
    #         et_count += np.sum(density_map_tmp)
    #         width_tmp = width_tmp+width/patches
    #     height_tmp = height_tmp+height/patches
    density_map = net(im_data, gt_data)
    density_map = density_map.data.cpu().numpy()
    et_count += np.sum(density_map)	
    gt_count = np.sum(gt_data)
    txt.write(str(et_count)+' ')
    # txt1.write(str(gt_count)+' ')
    mae += abs(gt_count-et_count)
    mse += ((gt_count-et_count)*(gt_count-et_count))
    if vis:
        utils.display_results(im_data, gt_data, density_map)
    if save_output:
        # print "kkk",density_map.shape
        utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')
        
mae = mae/data_loader.get_num_samples()
mse = np.sqrt(mse/data_loader.get_num_samples())
print '\nMAE: %0.2f, MSE: %0.2f' % (mae,mse)

f = open(file_results, 'w') 
f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
f.close()