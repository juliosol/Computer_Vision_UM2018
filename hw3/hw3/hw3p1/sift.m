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
k = 1.1;
s1 = 4*k;
[L,Sa] = DoGScaleSpace(gray_im,levels,k,s1);

max_val = 0;
max_scale = 0;
num_points = length(x);
max_coord_sspace = [];
max_scale_vector = [];

for j = 1:num_points
    for i = 1:levels
        if abs(L(x(j),y(j),i)) > max_val
            max_val = L(x(j),y(j),i);
            max_scale = Sa(i);
        end
    end
    max_coord_sspace = [max_coord_sspace; [x(j),y(j),max_scale]];
    max_scale_vector = [max_scale_vector; max_scale];
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
im_r_grad_x = conv2(im(:,:,1),x_grad,'same');
im_r_grad_y = conv2(im(:,:,1),y_grad,'same');

im_r_magnitude = sqrt(im_r_grad_x.^2 + im_r_grad_y.^2);
im_r_orientation = atan(im_r_grad_y./im_r_grad_x);

number_keypoints = length(max_coord_sspace);

list_feature_vectors_red = [];

for i = 1:number_keypoints
    curr_keypoint = max_coord_sspace(i,:);
    keypoint_x = curr_keypoint(1);
    keypoint_y = curr_keypoint(2);
    keypoint_sigma = curr_keypoint(3);
    
    sigma_window_magnitude = im_r_magnitude(keypoint_x - keypoint_sigma:keypoint_x + keypoint_sigma,keypoint_y - keypoint_sigma:keypoint_y +keypoint_sigma); 
    weighted_magnitude = imgaussfilt(sigma_window_magnitude,keypoint_sigma/2);
    
    window_magnitude_16_16 = weighted_magnitude(keypoint_x - 4:keypoint_x + 4,keypoint_y - 4:keypoint_y +4);
    window_orientation_16_16 = im_r_orientation(keypoint_x - 4:keypoint_x + 4,keypoint_y - 4:keypoint_y +4);
    sub_window_mag_1 = window_magnitude_16_16(5 - 4:5,5 - 4:5);
    sub_window_orient_1 = window_orientation_16_16(5 - 4:5,5 - 4:5);
    sub_window_mag_2 = window_magnitude_16_16(5:5 + 4,5 -4:5);
    sub_window_orient_2 = window_orientation_16_16(5:5 + 4,5 - 4:5);
    sub_window_mag_3 = window_magnitude_16_16(5 - 4:5,5:5 +4);
    sub_window_orient_3 = window_orientation_16_16(5 - 4:5,5:5 +4);
    sub_window_mag_4 = window_magnitude_16_16(5:5 + 4,5:5 +4);
    sub_window_orient_4 = window_orientation_16_16(5:5 + 4,5:5 +4);
  
    sub_windows_mag = sub_window_mag_1;
    sub_windows_orient = sub_window_orient_1;
    sub_windows_mag(:,:,2) = sub_window_mag_2;
    sub_windows_orient(:,:,2) = sub_window_orient_2;
    sub_windows_mag(:,:,3) = sub_window_mag_3;
    sub_windows_orient(:,:,3) = sub_window_orient_3;
    sub_windows_mag(:,:,4) = sub_window_mag_4;
    sub_windows_orient(:,:,4) = sub_window_orient_4;
    
    feature_vector = [];
    
    for z = 1:4
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
    %i
    %length(feature_vector)
    list_feature_vectors_red = [list_feature_vectors_red, feature_vector];
end
%size(list_feature_vectors_red)

% This second part of the code is for the green channel.
im_g_grad_x = conv2(im(:,:,2),x_grad,'same');
im_g_grad_y = conv2(im(:,:,2),y_grad,'same');

im_g_magnitude = sqrt(im_g_grad_x.^2 + im_g_grad_y.^2);
im_g_orientation = atan(im_g_grad_y./im_g_grad_x);

number_keypoints = length(max_coord_sspace);

list_feature_vectors_green = [];

for i = 1:number_keypoints
    curr_keypoint = max_coord_sspace(i,:);
    keypoint_x = curr_keypoint(1);
    keypoint_y = curr_keypoint(2);
    keypoint_sigma = curr_keypoint(3);
    weighted_magnitude = imgaussfilt(im_g_magnitude,keypoint_sigma);
    
    window_sigma = weighted_magnitude
    window_magnitude_16_16 = weighted_magnitude(keypoint_x - 4:keypoint_x + 4,keypoint_y - 4:keypoint_y +4);
    window_orientation_16_16 = im_g_orientation(keypoint_x - 4:keypoint_x + 4,keypoint_y - 4:keypoint_y +4);
    sub_window_mag_1 = window_magnitude_16_16(5 - 4:5,5 - 4:5);
    sub_window_orient_1 = window_orientation_16_16(5 - 4:5,5 - 4:5);
    sub_window_mag_2 = window_magnitude_16_16(5:5 + 4,5 -4:5);
    sub_window_orient_2 = window_orientation_16_16(5:5 + 4,5 - 4:5);
    sub_window_mag_3 = window_magnitude_16_16(5 - 4:5,5:5 +4);
    sub_window_orient_3 = window_orientation_16_16(5 - 4:5,5:5 +4);
    sub_window_mag_4 = window_magnitude_16_16(5:5 + 4,5:5 +4);
    sub_window_orient_4 = window_orientation_16_16(5:5 + 4,5:5 +4);
  
    sub_windows_mag = sub_window_mag_1;
    sub_windows_orient = sub_window_orient_1;
    sub_windows_mag(:,:,2) = sub_window_mag_2;
    sub_windows_orient(:,:,2) = sub_window_orient_2;
    sub_windows_mag(:,:,3) = sub_window_mag_3;
    sub_windows_orient(:,:,3) = sub_window_orient_3;
    sub_windows_mag(:,:,4) = sub_window_mag_4;
    sub_windows_orient(:,:,4) = sub_window_orient_4;
    
    feature_vector = [];
    
    for z = 1:4
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
%size(list_feature_vectors_green)

% This second part of the code is for the blue channel.
im_b_grad_x = conv2(im(:,:,3),x_grad,'same');
im_b_grad_y = conv2(im(:,:,3),y_grad,'same');

im_b_magnitude = sqrt(im_b_grad_x.^2 + im_b_grad_y.^2);
im_b_orientation = atan(im_b_grad_y./im_b_grad_x);

list_feature_vectors_blue = [];

for i = 1:number_keypoints
    curr_keypoint = max_coord_sspace(i,:);
    keypoint_x = curr_keypoint(1);
    keypoint_y = curr_keypoint(2);
    keypoint_sigma = curr_keypoint(3);
    weighted_magnitude = imgaussfilt(im_b_magnitude,keypoint_sigma);
    
    window_magnitude_16_16 = weighted_magnitude(keypoint_x - 4:keypoint_x + 4,keypoint_y - 4:keypoint_y +4);
    window_orientation_16_16 = im_b_orientation(keypoint_x - 4:keypoint_x + 4,keypoint_y - 4:keypoint_y +4);
    sub_window_mag_1 = window_magnitude_16_16(5 - 4:5,5 - 4:5);
    sub_window_orient_1 = window_orientation_16_16(5 - 4:5,5 - 4:5);
    sub_window_mag_2 = window_magnitude_16_16(5:5 + 4,5 -4:5);
    sub_window_orient_2 = window_orientation_16_16(5:5 + 4,5 - 4:5);
    sub_window_mag_3 = window_magnitude_16_16(5 - 4:5,5:5 +4);
    sub_window_orient_3 = window_orientation_16_16(5 - 4:5,5:5 +4);
    sub_window_mag_4 = window_magnitude_16_16(5:5 + 4,5:5 +4);
    sub_window_orient_4 = window_orientation_16_16(5:5 + 4,5:5 +4);
  
    sub_windows_mag = sub_window_mag_1;
    sub_windows_orient = sub_window_orient_1;
    sub_windows_mag(:,:,2) = sub_window_mag_2;
    sub_windows_orient(:,:,2) = sub_window_orient_2;
    sub_windows_mag(:,:,3) = sub_window_mag_3;
    sub_windows_orient(:,:,3) = sub_window_orient_3;
    sub_windows_mag(:,:,4) = sub_window_mag_4;
    sub_windows_orient(:,:,4) = sub_window_orient_4;
    
    feature_vector = [];
    
    for z = 1:4
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
%size(list_feature_vectors_blue)

F = [list_feature_vectors_red;list_feature_vectors_green;list_feature_vectors_blue];
size(F)


