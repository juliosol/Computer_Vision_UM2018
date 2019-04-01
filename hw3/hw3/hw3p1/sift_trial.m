function [F,Whalf] = sift(im,x,y)
%
% W18 EECS 504 HW3p1 Matching balloons with SIFT
%
% Compute the SIFT on an image for a given list of feature points (x1,y1; x2,y2;...)
%  
% Input     
%   im: RGB image
%   x,y: coordinates of feature points (x downwards, y rightwards)
% Output    
%   F: lxn matrix of which each column is a feature vector. 
%      l is the length of feature vectors and n is the number of points
%   Whalf: nx1 column vector of which the ith element is the half size of the ith feature region.
%          A feature region is a square of (2Whalf+1)x(2Whalf+1) centered at the feature point.
%

%% Scale selection
% Use DoG to determine the characteristic scale of every feature point.
% You may use j_DoGScaleSpace.p; it is a solution to hw2p3 with two extra
% inputs: [L,sa] = DoGScaleSpace(im,levels,k,s1). Meaning of arguments
% is the same as what appeared in hw2 code.


% Solution: In this part of the code we will find the Scale space and
% using this scale space, we will figure out what is the characteristic
% scale by considering the value at the pixel (x,y) in different scales.
% For this first try, we are only taking the maximum value attained at the
% corresponding pixels throughout the scale space.
gray_im = rgb2gray(im);
levels = 25;
k = 1.2;
s1 = 3*k;
[L,Sa] = DoGScaleSpace(gray_im,levels,k,s1);

num_points = length(x);
max_coord_sspace = [];
max_scale_vector = [];

for j = 1:num_points
    L(x(j),y(j),:)
    max_val = 0;
    max_scale = 0;
    for i = 1:levels
        %Sa
        %S = 'Value of point is';
        %disp(S)
        %L(x(j),y(j),i)
        if abs(L(x(j),y(j),i)) >= max_val
            max_val = L(x(j),y(j),i)
            max_scale = Sa(i)
        end
    end
    max_coord_sspace = [max_coord_sspace; [x(j),y(j),max_scale]]
    max_scale_vector = [max_scale_vector; max_scale]
end



%% Orientation assignment (optional)
% Find the dominant orientation of the feature region and rotate it
% accordingly. Hint: equivalently, you could rotate the image 
% in the inverse direction.

x_grad = [-1,0,1];
y_grad = [-1;0;1];

% This part computes the x and y gradient of every level in the scale
% space.

%im_grad_x_scale_space_

im_grad_x = conv2(gray_im,x_grad,'same');
im_grad_y = conv2(gray_im,y_grad,'same');


im_magnitude = sqrt(im_grad_x.^2 + im_grad_y.^2);
im_orientation = atan(im_grad_y./im_grad_x);


%% Compute hog descriptors
% Compute the feature vectors for RGB channels separately and then
% concatenate them into a single vector as the descriptor. 
% For example, if the feature vector for R channel is 1xlen, descriptor of 
% of feature point should be 1x(3len). Note that the output feature vectors
% for all feature points should have the SAME length.
% 
% We follow the settings from David Lowe's paper:
% 1) Compute the histograms of gradient from each 4x4 subregions and concatnate
% them together. Normalize them before concatenation.
% 2) Use 8 bins for histogram.
% 3) Gradients are Gaussian weighted.
% 4) Use trillinear interpolation when distributing the gradient into bins.
% 5) Threshold the values in the feature vector to be no larger than 0.2.

% In this part we compute the histograms of gradient from each 4x4
% subregion and concatenate them together. 

% First we compute the magnitude and gradient for each of the RGB channels

% This first part of the code is for the red channel
red_channel = im(:,:,1);
Whalf = [];

number_keypoints = length(max_coord_sspace);

list_feature_vectors_red = [];

for i = 1:number_keypoints
    curr_keypoint = max_coord_sspace(i,:);
    keypoint_x = curr_keypoint(1);
    keypoint_y = curr_keypoint(2);
    keypoint_sigma = curr_keypoint(3);
    Whalf = [Whalf;keypoint_sigma];
    
    keypoint_x - ceil(keypoint_sigma)/2
    keypoint_x + ceil(keypoint_sigma)/2
    keypoint_y - ceil(keypoint_sigma)/2
    keypoint_y + ceil(keypoint_sigma)/2
    ceil(keypoint_sigma)/2
    
    sigma_window= red_channel(keypoint_x - ceil(keypoint_sigma)/2:keypoint_x + ceil(keypoint_sigma)/2,keypoint_y - ceil(keypoint_sigma)/2:keypoint_y + ceil(keypoint_sigma)/2); 
    sigma_window_16_16 = imresize(sigma_window,[16,16]);
    sigma_window_grad_x = conv2(sigma_window_16_16,x_grad,'same');
    sigma_window_grad_y = conv2(sigma_window_16_16,y_grad,'same');
    
    window_r_magnitude = sqrt(sigma_window_grad_x.^2 + sigma_window_grad_x.^2);
    window_r_orientation = atan(sigma_window_grad_y./sigma_window_grad_x);
    
    weighted_r_magnitude = imgaussfilt(window_r_magnitude,keypoint_sigma/2);
        
    sub_windows_mag = [];
    sub_windows_orient = [];
    
    for k = 1:16
        if k == 1
            l = 1;
        else
            l = ceil(k/4);
        end
        for t = 1:4
            if mod(k,4) == 0
                s = 4;
            else
                s = mod(k,4);
            end
            sub_windows_mag(t,:,k) = weighted_r_magnitude(4*(4^2*(l-1) + (s-1) + 4*(t-1))+1 : 4*(4^2*(l-1) + (s-1) + 4*(t-1)) + 1 + 3);
            sub_windows_orient(t,:,k) = window_r_orientation(4*(4^2*(l-1) + (s-1) + 4*(t-1)) + 1 : 4*(4^2*(l-1) + (s-1) + 4*(t-1)) + 1 + 3);
        end
    end
            
    feature_vector = [];
    
    for z = 1:16
        curr_window_mag = sub_windows_mag(:,:,z);
        curr_window_orient = sub_windows_orient(:,:,z);
        hist0 = 0;
        hist45 = 0;
        hist90 = 0;
        hist135 = 0;
        hist180 = 0;
        hist225 = 0;
        hist270 = 0;
        hist315 = 0;
        for y = 1:4
            for x = 1:4
                orientation = curr_window_orient(x,y);
                magnitude = curr_window_mag(x,y);
                if orientation < 45 && orientation >= 0
                    hist0 = hist0 + magnitude;
                elseif orientation < 90 && orientation >= 45
                    hist45 = hist45 + magnitude;
                elseif orientation < 135 && orientation >= 90
                    hist90 = hist90 + magnitude;
                elseif orientation < 180 && orientation >= 135
                    hist135 = hist135 + magnitude;
                elseif orientation < 225 && orientation >= 180
                    hist180 = hist180 + magnitude;
                elseif orientation < 270 && orientation >= 225
                    hist225 = hist225 + magnitude;
                elseif orientation < 315 && orientation >= 270
                    hist270 = hist270 + magnitude;
                elseif orientation < 360 && orientation >= 315
                    hist315 = hist315 + magnitude;
                end
            end
        end
        feature_vector = [feature_vector; hist0; hist45; hist90; hist135; hist180; hist225; hist270; hist315];
    end
    list_feature_vectors_red = [list_feature_vectors_red, feature_vector];
end

%This section is for the green channel 

green_channel = im(:,:,2);

number_keypoints = length(max_coord_sspace);

list_feature_vectors_green = [];

for i = 1:number_keypoints
    curr_keypoint = max_coord_sspace(i,:);
    keypoint_x = curr_keypoint(1);
    keypoint_y = curr_keypoint(2);
    keypoint_sigma = curr_keypoint(3);
    
    sigma_window= green_channel(keypoint_x - ceil(keypoint_sigma)/2:keypoint_x + ceil(keypoint_sigma)/2,keypoint_y - ceil(keypoint_sigma)/2:keypoint_y + ceil(keypoint_sigma)/2); 
    sigma_window_16_16 = imresize(sigma_window,[16,16]);
    sigma_window_grad_x = conv2(sigma_window_16_16,x_grad,'same');
    sigma_window_grad_y = conv2(sigma_window_16_16,y_grad,'same');
    
    window_g_magnitude = sqrt(sigma_window_grad_x.^2 + sigma_window_grad_x.^2);
    window_g_orientation = atan(sigma_window_grad_y./sigma_window_grad_x);
    
    weighted_g_magnitude = imgaussfilt(window_g_magnitude,keypoint_sigma/2);
    
    sub_windows_mag = [];
    sub_windows_orient = [];
    
    for k = 1:16
        if k == 1
            l = 1;
        else
            l = ceil(k/4);
        end
        for t = 1:4
            if mod(k,4) == 0
                s = 4;
            else
                s = mod(k,4);
            end
            sub_windows_mag(t,:,k) = weighted_g_magnitude(4*(4^2*(l-1) + (s-1) + 4*(t-1))+1 : 4*(4^2*(l-1) + (s-1) + 4*(t-1)) + 1 + 3);
            sub_windows_orient(t,:,k) = window_g_orientation(4*(4^2*(l-1) + (s-1) + 4*(t-1)) + 1 : 4*(4^2*(l-1) + (s-1) + 4*(t-1)) + 1 + 3);
        end
     end
     
    feature_vector = [];
    
    for z = 1:16
        curr_window_mag = sub_windows_mag(:,:,z);
        curr_window_orient = sub_windows_orient(:,:,z);
        hist0 = 0;
        hist45 = 0;
        hist90 = 0;
        hist135 = 0;
        hist180 = 0;
        hist225 = 0;
        hist270 = 0;
        hist315 = 0;
        for y = 1:4
            for x = 1:4
                orientation = curr_window_orient(x,y);
                magnitude = curr_window_mag(x,y);
                if orientation < 45 && orientation >= 0
                    hist0 = hist0 + magnitude;
                elseif orientation < 90 && orientation >= 45
                    hist45 = hist45 + magnitude;
                elseif orientation < 135 && orientation >= 90
                    hist90 = hist90 + magnitude;
                elseif orientation < 180 && orientation >= 135
                    hist135 = hist135 + magnitude;
                elseif orientation < 225 && orientation >= 180
                    hist180 = hist180 + magnitude;
                elseif orientation < 270 && orientation >= 225
                    hist225 = hist225 + magnitude;
                elseif orientation < 315 && orientation >= 270
                    hist270 = hist270 + magnitude;
                elseif orientation < 360 && orientation >= 315
                    hist315 = hist315 + magnitude;
                end
            end
        end
        feature_vector = [feature_vector; hist0; hist45; hist90; hist135; hist180; hist225; hist270; hist315];
    end
    list_feature_vectors_green = [list_feature_vectors_green, feature_vector];
end

blue_channel = im(:,:,3);

number_keypoints = length(max_coord_sspace);

list_feature_vectors_blue = [];

for i = 1:number_keypoints
    curr_keypoint = max_coord_sspace(i,:);
    keypoint_x = curr_keypoint(1);
    keypoint_y = curr_keypoint(2);
    keypoint_sigma = curr_keypoint(3);
    
    sigma_window= blue_channel(keypoint_x - ceil(keypoint_sigma)/2:keypoint_x + ceil(keypoint_sigma)/2,keypoint_y - ceil(keypoint_sigma)/2:keypoint_y + ceil(keypoint_sigma)/2); 
    sigma_window_16_16 = imresize(sigma_window,[16,16]);
    sigma_window_grad_x = conv2(sigma_window_16_16,x_grad,'same');
    sigma_window_grad_y = conv2(sigma_window_16_16,y_grad,'same');
    
    window_b_magnitude = sqrt(sigma_window_grad_x.^2 + sigma_window_grad_x.^2);
    window_b_orientation = atan(sigma_window_grad_y./sigma_window_grad_x);
    
    weighted_b_magnitude = imgaussfilt(window_b_magnitude,keypoint_sigma/2);
        
    sub_windows_mag = [];
    sub_windows_orient = [];
    
    for k = 1:16
        if k == 1
            l = 1;
        else
            l = ceil(k/4);
        end
        for t = 1:4
            if mod(k,4) == 0
                s = 4;
            else
                s = mod(k,4);
            end
            sub_windows_mag(t,:,k) = weighted_b_magnitude(4*(4^2*(l-1) + (s-1) + 4*(t-1))+1 : 4*(4^2*(l-1) + (s-1) + 4*(t-1)) + 1 + 3);
            sub_windows_orient(t,:,k) = window_b_orientation(4*(4^2*(l-1) + (s-1) + 4*(t-1)) + 1 : 4*(4^2*(l-1) + (s-1) + 4*(t-1)) + 1 + 3);
        end
     end
         
    feature_vector = [];
    
    for z = 1:16
        curr_window_mag = sub_windows_mag(:,:,z);
        curr_window_orient = sub_windows_orient(:,:,z);
        hist0 = 0;
        hist45 = 0;
        hist90 = 0;
        hist135 = 0;
        hist180 = 0;
        hist225 = 0;
        hist270 = 0;
        hist315 = 0;
        for y = 1:4
            for x = 1:4
                orientation = curr_window_orient(x,y);
                magnitude = curr_window_mag(x,y);
                if orientation < 45 && orientation >= 0
                    hist0 = hist0 + magnitude;
                elseif orientation < 90 && orientation >= 45
                    hist45 = hist45 + magnitude;
                elseif orientation < 135 && orientation >= 90
                    hist90 = hist90 + magnitude;
                elseif orientation < 180 && orientation >= 135
                    hist135 = hist135 + magnitude;
                elseif orientation < 225 && orientation >= 180
                    hist180 = hist180 + magnitude;
                elseif orientation < 270 && orientation >= 225
                    hist225 = hist225 + magnitude;
                elseif orientation < 315 && orientation >= 270
                    hist270 = hist270 + magnitude;
                elseif orientation < 360 && orientation >= 315
                    hist315 = hist315 + magnitude;
                end
            end
        end
        feature_vector = [feature_vector; hist0; hist45; hist90; hist135; hist180; hist225; hist270; hist315];
    end
    list_feature_vectors_blue = [list_feature_vectors_blue, feature_vector];
end


F = [list_feature_vectors_red;list_feature_vectors_green;list_feature_vectors_blue];
Whalf;