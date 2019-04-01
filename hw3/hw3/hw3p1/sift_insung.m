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
% Use DoG to determin the characteristic scale of every feature point.
% You may use j_DoGScaleSpace.p; it is a solution to hw2p3 with two extra
% inputs: [L,sa] = DoGScaleSpace(im,levels,k,s1). Meaning of arguments
% is the same as what appeared in hw2 code.
k=1.1; s1=4*k;
[L,sa] = j_DoGScaleSpace(double(rgb2gray(im)),25,k,s1);



%% Orientation assignment (optional)
% Find the dominant orientation of the feature region and rotate it
% accordingly. Hint: equivalently, you could rotate the image 
% in the inverse direction.


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
F = [];
Whalf = [];
bins = -180:45:180;

for i = 1:size(x)
%     pt_descriptor = [];
    point = [x(i), y(i)];
    %pt_ref = [x(i), y(i)];
    [value, index] = max(L(point(1),point(2),:));
    Whalf(i) = ceil(1.6*(sa(index)));
    
    for j = 1:3 % For channel in image.
        channel = im(:,:,j);
               
        gauss_filt = fspecial('gaussian',[2*Whalf(i)+1,2*Whalf(i)+1],sa(index));
        %size(channel);
        channel = conv2(double(channel),gauss_filt,'same');
        channel = padarray(channel,[2*Whalf(i),2*Whalf(i)]);
        [channel_mag, channel_dir] = imgradient(channel);
      
        slice_mag = channel_mag(point(1):point(1)+2*Whalf,point(2):point(2)+2*Whalf);
        slice_dir = channel_dir(point(1):point(1)+2*Whalf,point(2):point(2)+2*Whalf);
        slice_mag = imresize(slice_mag,[16,16]);
        slice_dir = imresize(slice_dir,[16,16]);
        H = zeros(128,1); kk = 0;
        for m = 1:4:16
            for n = 1:4:16
                kk = kk + 1;
                HH = zeros(8,1);
                subregion_mag = slice_mag(m:m+3,n:n+3);
                subregion_dir = slice_dir(m:m+3,n:n+3);
                subregion_mag = subregion_mag(:);
                subregion_dir = subregion_dir(:);
                for k = 1:size(subregion_dir)
                   if subregion_dir(k) < -180
                       subregion_dir(k) = subregion_dir(k) + 360;
                   elseif subregion_dir(k) > 180
                       subregion_dir(k) = subregion_dir(k) - 360;
                   else
                       subregion_dir(k) = subregion_dir(k);
                   end
                   main_bin = discretize(subregion_dir,bins);
                   d = mod((subregion_dir(k)+180),45);
                   if d > 25
                       d_in = 1 - (d-22.5)/45;
                       d_out = 1-d_in;
                       HH(main_bin(k)) = HH(main_bin(k)) + subregion_mag(k)*d_in;
                       if main_bin(k) == 8
                            HH(1) = HH(1) + subregion_mag(k)*d_out;
                       else
                            HH(main_bin(k)+1) = HH(main_bin(k)+1) + subregion_mag(k)*d_out;
                       end
                   elseif d < 25
                       d_in = 1 - (22.5-d)/45;
                       d_out = 1 - d_in;
                       HH(main_bin(k)) = HH(main_bin(k)) + subregion_mag(k)*d_in;
                       if main_bin(k) == 1
                            HH(8) = HH(8) + subregion_mag(k)*d_out;
                       else
                            HH(main_bin(k)-1) = HH(main_bin(k)-1) + subregion_mag(k)*d_out;
                       end
                   else
                       d_in = 1;
                       d_out = 0;
                       HH(main_bin(k)) = HH(main_bin(k)) + subregion_mag(k)*d_in;
                   end

                end
%                 HH = HH / norm(HH);
%                 H = [H, transpose(HH)];
                H((kk-1)*8+1:kk*8)= HH;     
                
                
                
            end
            H = H/norm(H);
            H(H>0.2) = 0.2;
            H = H/norm(H);
            F((j-1)*128+1:j*128,i) = H;
        end
%     pt_descriptor = [pt_descriptor, H];
    end
%     F = [F; pt_descriptor];
end
% F = transpose(F);

