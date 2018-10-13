%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File to create training and validation set       %
% for ShanghaiTech Dataset Part A and B. 10% of    %
% the training set is set aside for validation     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc; clear all;
seed = 95461354;
rng(seed);
N = 100;


path = ('E:\copy\CVPR2018_CrowdCounting\data\original\ShanghaiTech\part_A_final\train_data\images\');
gt_path = ('E:\copy\CVPR2018_CrowdCounting\data\original\ShanghaiTech\part_A_final\train_data\ground_truth\');

train_path_img = ['E:\copy\CVPR2018_CrowdCounting\data\original\ShanghaiTech\part_A_final\train_data\after_images\'];
train_path_den = ['E:\copy\CVPR2018_CrowdCounting\data\original\ShanghaiTech\part_A_final\train_data\after_ground_truth\'];


mkdir(train_path_img);
mkdir(train_path_den);


num_images = 300

%indices = randperm(num_images);

for idx = 1:num_images
    i = idx;
    if (mod(idx,10)==0)
        fprintf(1,'Processing %3d/%d files\n', idx, num_images);
    end
    load(strcat(gt_path,'GT_IMG_',num2str(i),'.mat')) ;
    input_img_name = strcat(path,'IMG_',num2str(i),'.jpg');
    im = imread(input_img_name);
    [h, w, c] = size(im);
    if (c == 3)
        im = rgb2gray(im);
    end
    
    wn2 = w/8; hn2 = h/8;
    wn2 =8 * floor(wn2/8);
    hn2 =8 * floor(hn2/8);
    
    annPoints = image_info{1}.location;
    if( w <= 2*wn2 )
        im = imresize(im,[ h,2*wn2+1]);
        annPoints(:,1) = annPoints(:,1)*2*wn2/w;
    end
    if( h <= 2*hn2)
        im = imresize(im,[2*hn2+1,w]);
        annPoints(:,2) = annPoints(:,2)*2*hn2/h;
    end
    [h, w, c] = size(im);
    a_w = wn2+1; b_w = w - wn2;
    a_h = hn2+1; b_h = h - hn2;
    
    im_density = create_density(annPoints,h,w);
    for j = 1:N
        
        x = floor((b_w - a_w) * rand + a_w);
        y = floor((b_h - a_h) * rand + a_h);
        x1 = x - wn2; y1 = y - hn2;
        x2 = x + wn2-1; y2 = y + hn2-1;
        
        
        im_sampled = im(y1:y2, x1:x2,:);
        im_density_sampled = im_density(y1:y2,x1:x2);
        
        annPoints_sampled = annPoints(annPoints(:,1)>x1 & ...
            annPoints(:,1) < x2 & ...
            annPoints(:,2) > y1 & ...
            annPoints(:,2) < y2,:);
        annPoints_sampled(:,1) = annPoints_sampled(:,1) - x1;
        annPoints_sampled(:,2) = annPoints_sampled(:,2) - y1;
        img_idx = strcat(num2str(i), '_',num2str(j));        


        imwrite(im_sampled, [train_path_img num2str(img_idx) '.jpg']);
        csvwrite([train_path_den num2str(img_idx) '.csv'], im_density_sampled);

        
    end
    
end

