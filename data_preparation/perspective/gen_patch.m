clear

% Configuration
dataset = 'dataset_01';
crop_size = 256;
max_crop_per_frame = 10;
density_scale = 10000;
seg_thresh = 0.0001;

addpath('../utils');
% Get the scene list in specified dataset
scene_list = dir(fullfile('..', 'data', dataset, 'frames'));
scene_list = {scene_list([scene_list(:).isdir]).name};
scene_list(1:2) = [];

symbols = ['a':'z' '0':'9'];
MAX_ST_LENGTH = 10;
rng(0, 'twister');

for index = 1:numel(scene_list)
  scene_id = scene_list{index};

  frame_dir = fullfile('..', 'data', dataset, 'frames', scene_id);
  label_dir = fullfile('..', 'data', dataset, 'labels', scene_id);
  file_list = dir([label_dir '/*.mat']);

  output_dir = fullfile('..', 'output', dataset);
  mkdir_if_not_exist([output_dir '/sub_frames']);
  mkdir_if_not_exist([output_dir '/sub_segs']);
  mkdir_if_not_exist([output_dir '/sub_dens']);

  for i = 1:numel(file_list)
    [~, basename, ~] = fileparts(file_list(i).name);
    fprintf('%s\n', basename);
    % Load image
    im = imread(fullfile(frame_dir, [basename '.jpg']));
    [height, width, channels] = size(im);
    % load density map
    load([output_dir '/density_map/' scene_id '/' basename '.mat']);
    % generate segment mask
    segment = (density_map > seg_thresh);
 
    for k = 1:max_crop_per_frame
      x = randi([1 (width - crop_size)]);
      y = randi([1 (height - crop_size)]);
      % crop
      crop_img = imcrop(im, [x y crop_size - 1 crop_size - 1]);
      crop_seg = imcrop(segment, [x y crop_size - 1 crop_size - 1]);
      crop_dens = imcrop(density_map, [x y crop_size - 1 crop_size - 1]);
      crop_dens = crop_dens * density_scale;
   
      % Flip with a probability of 0.5
      if rand > 0.5
        crop_img = flip(crop_img, 2);
        crop_seg = flip(crop_seg, 2);
        crop_dens = flip(crop_dens, 2);
      end
   
      nums = randi(numel(symbols), [1 MAX_ST_LENGTH]);
      filename = symbols(nums);
      imwrite(crop_img, [output_dir '/sub_frames/' filename '.jpg']);
      imwrite(crop_seg, [output_dir '/sub_segs/' filename '.png']);
      save([output_dir '/sub_dens/' filename '.mat'], 'filename', 'crop_dens');
    end
  end
end
