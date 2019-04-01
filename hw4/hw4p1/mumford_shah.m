function Epc = mumford_shah(original_function,estimated_function)

% This function will compute the discrete piecewise Mumford Shah model. The
% INPUT: an original funcction and an approximation function. 
% Output: The energy of the image with respect to the approximation and
% certain boundary term.

alpha = 1;
gamma = 0.1;

double_orig_fxn = im2double(original_function);
double_estimated_fxn = im2double(estimated_function);

squared_difference = (double_estimated_fxn - double_orig_fxn).*(double_estimated_fxn - double_orig_fxn);

integral = sum(squared_difference(:));

hor_mask = [1,-1];
ver_mask = [1; -1];
hor_conv = conv2(double_estimated_fxn,hor_mask,'valid');
ver_conv = conv2(double_estimated_fxn,ver_mask,'valid');
hor_sum = sum(hor_conv(:) ~= 0);
ver_sum = sum(ver_conv(:) ~= 0);
boundary = hor_sum + ver_sum;

Epc = alpha*integral + gamma * boundary;

end