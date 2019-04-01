function [L,sa] = DoGScaleSpace(im,levels,k,s1)
%
%  W18 EECS 504 HW2p3 Blob detection
%
%  Regarding different parameters, the paper gives some empirical data which 
%  can be summarized as, number of octaves = 4, number of scale levels = 5, 
%  initial sigma=1.6*2^0.5, k=2^0.5 etc as optimal values.
%
%  Contrary to the Lowe DoG Scale Space, this does not separate the scales
%  into octaves, for simplicity.
%
%  im is a grayscale double image
%  levels are the number of levels you want in the scale-space
% 
%  L is the r x c x levels response DoG scale space
%  sa is a levels x 1 vector of sigma's corresponding to each layer in the scale space.


% initial conditions are given to you.  
%  s1 is the sigma of the first Gaussian
%  k is the multiplier per level
%  sa is the vector of each sigma per level
%k  = sqrt(2);
%s1 = 1.6*k;
sa = cumprod( [s1 ones(1,levels)*k] );

% you need to populate this array.  Each i-level (:,:,i) corresponds to a level of the scale space, 
% which is the difference of two Guassian-filtered images (sigma and k*sigma)

L = zeros([size(im) levels]);

%%%% YOU NEED TO FILL IN THE CODE BELOW 

for i = 1:levels
    Gauss1= imgaussfilt(im, sa(i),'FilterSize', 2*ceil(3*sa(i))+1);
    Gauss2 = imgaussfilt(im,sa(i+1),'FilterSize', 2*ceil(3*sa(i+1))+1);
    %size(L)
    %size(Gauss1)
    %size(Gauss2)
    L(:,:,i) = Gauss2 - Gauss1;
end



%%%% YOU NEED TO STOP HERE


end
