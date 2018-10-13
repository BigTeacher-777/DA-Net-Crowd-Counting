clear

% Configuration
dataset = 'WorldExpo10';
display = 0;
head_only = 0;
gauss_size = 120;
person_height = gauss_size * 4;
gauss_std_low_thresh = 0.8;

addpath('../utils');
% Get the scene list in specified dataset
scene_list = dir(fullfile('..', '..', dataset, 'train','label'));
scene_list = {scene_list([scene_list(:).isdir]).name};
scene_list(1:2) = [];

for index = 1:numel(scene_list)
  scene_id = scene_list{index};
  % Loading the perspective map of the current scene
  pers_map_file = fullfile('..', 'data', dataset, 'perspective', ...
    [scene_id '.mat']);
  fprintf('Loading %s\n', pers_map_file);
  load(pers_map_file);

  % The following code fix a historical problem
  if exist('pMap_out', 'var')
    pMap = pMap_out;
    clear pMap_out
  end

  [height, width, channels] = size(pMap);

  frame_dir = fullfile('..', 'data', dataset, 'frames', scene_id);
  label_dir = fullfile('..', 'data', dataset, 'labels', scene_id);

  file_list = dir([label_dir '/*.mat']);

  output_dir = fullfile('..', 'output', dataset, 'density_map', scene_id);
  mkdir_if_not_exist(output_dir);

  for i = 1:numel(file_list)
    [~, basename, ~] = fileparts(file_list(i).name);
    fprintf('%s\n', basename);
 
    load(fullfile(label_dir, [basename '.mat']));
 
    % Skip images with no person
    if point_num <= 0
      continue
    end
 
    density_out = zeros(height + person_height, width + gauss_size);
    for k = 1:point_num
      x = point_position(k, 1);
      y = point_position(k, 2);
   
      % Skip error points
      if y > height || x > width || y < 1 || x < 1
        continue
      end
   
      pers_value = ceil(pMap(y, x));

      lamda = pers_value * 0.15;
      % Discard too small people
      if lamda < gauss_std_low_thresh
        continue
      end
      if lamda > 0.20 * gauss_size
        lamda = 0.20 * gauss_size;
        warning([basename ': lambda is greater than 0.20 * gauss_size']);
      end
      % Set head
      head = fspecial('gaussian', gauss_size, lamda);
   
      % set body
      temp_a = pdf('norm', 1:person_height, gauss_size / 2 + pers_value, lamda * 4);
      temp_b = pdf('norm', 1:gauss_size, gauss_size / 2, lamda);
      body = temp_a * temp_b * 4;
   
      if head_only
        body(:) = 0.0; %#ok<UNRCH>
      end
      % combine body with head
      body(1:gauss_size, 1:gauss_size) = body(1:gauss_size, 1:gauss_size) + head;
   
      % Normalization
      body(body < 0.000001) = 0.0;
      person = body ./ sum(body(:));
   
      density_out(y:(y+person_height-1), x:(x+gauss_size-1)) = ...
        density_out(y:(y+person_height-1), x:(x+gauss_size-1)) + person;
    end
 
    density_map = density_out(gauss_size / 2:(gauss_size / 2 + height - 1), gauss_size / 2:(gauss_size / 2 + width - 1));
    save(fullfile(output_dir, [basename '.mat']), 'density_map');
 
    if display
      im = imread(fullfile(frame_dir, [basename '.jpg'])); %#ok<UNRCH>
      
      figure(1)
      subplot(2, 1, 1);
      imshow(im);      
      hold on
      % Plot ground truth
      plot(point_position(:, 1), point_position(:, 2), 'r.');
      tl = title(sprintf('%s: %s', scene_id, basename));
      set(tl, 'Interpreter', 'none')

      subplot(2, 1, 2);
      title(sprintf('sum(density)=%.1f', sum(density_map(:))))
      imagesc(density_map);
      hold on
      % Plot ground truth again for double-check
      plot(point_position(:, 1), point_position(:, 2), 'r.');

      waitforbuttonpress
    end
  end
end

if display
  close %#ok<UNRCH>
end
