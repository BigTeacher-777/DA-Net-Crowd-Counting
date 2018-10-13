% Density Map with Geometry-Adaptive Kernels (from MCNN code:
% Single-Image Crowd Counting via Multi-Column Convolutional
% Neural Network; CVPR 2016)

% Create density maps from head annotation
function d_map = create_density(gt, img_h, img_w)

K = 7;
m = img_h;
n = img_w;

d_map = zeros(m, n);
gt = gt(gt(:, 1) < n, :);
gt = gt(gt(:, 2) < m, :);

for j = 1: size(gt, 1)
    [~, D] = knnsearch(gt, gt(j, :), 'K', K + 1);
    ksize = round(mean(D(2:end)));
    ksize = max(ksize,7);
    ksize = min(ksize,25);
    radius = ceil(ksize/2);
    sigma = ksize * 0.3;
    h = fspecial('gaussian',ksize,sigma);
    x_ = max(1,floor(gt(j,1)));
    y_ = max(1,floor(gt(j,2)));
       
       if (x_-radius+1<1)
              for ra = 0:radius-x_-1
                   h(:,end-ra) = h(:,end-ra)+h(:,1);
                   h(:,1)=[];
              end
       end  
       if (y_-radius+1<1)
           for ra = 0:radius-y_-1
               h(end-ra,:) = h(end-ra,:)+h(1,:);
               h(1,:)=[];
           end
       end
       if (x_+ksize-radius>n)   
           for ra = 0:x_+ksize-radius-n-1
               h (:,1+ra) = h(:,1+ra)+h(:,end);
               h(:,end) = [];
           end
       end
       if(y_+ksize-radius>m)    
            for ra = 0:y_+ksize-radius-m-1
                h (1+ra,:) = h(1+ra,:)+h(end,:);
                h(end,:) = [];
            end
       end             
          d_map(max(y_-radius+1,1):min(y_+ksize-radius,m),max(x_-radius+1,1):min(x_+ksize-radius,n))...
             = d_map(max(y_-radius+1,1):min(y_+ksize-radius,m),max(x_-radius+1,1):min(x_+ksize-radius,n))...
              + h;
end

end

