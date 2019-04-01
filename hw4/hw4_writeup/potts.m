% W18 EECS 504 HW4p1 Mumford-Shah Piecewise Constant
% Siyuan Chen

function E = potts(I,beta)

if nargin==1
    beta=1;
end

%%% FILL IN YOUR CODE HERE

double_im = im2double(I);
E = 0;  

% In this part of the code we are forming the 3d kernel, where the vertical
% kernel has entries 1 and -1 in column form, so this kernel is 3x1x3 of size 
% and the horizontal kernel has entries 1 and -1 in verticla form, so this
% kernel has 1x3x3 of size. 

hor_mask = [1,-1];
ver_mask = [1;-1];
hor_mask_3d(:,:,1) = hor_mask;
hor_mask_3d(:,:,2) = hor_mask;
hor_mask_3d(:,:,3) = hor_mask;
ver_mask_3d(:,:,1) = ver_mask;
ver_mask_3d(:,:,2) = ver_mask;
ver_mask_3d(:,:,3) = ver_mask;

hor_conv = convn(double_im, hor_mask_3d,'valid');
ver_conv = convn(double_im,ver_mask_3d,'valid');
hor_sum = sum(hor_conv(:) ~= 0);
ver_sum = sum(ver_conv(:) ~= 0);
E = hor_sum + ver_sum;
end