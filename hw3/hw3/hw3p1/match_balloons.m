% 
% W18 EECS 504 HW3p1 Matching balloons with SIFT
%
% Use SIFT to find matchings between feature points in two images.
% Feature points in balloon1.jpg and balloon2.jpg are given in 
% points1.mat and points2.mat respectively.
%
% Siyuan Chen, chsiyuan@umich.edu
%

clear
close all

I1 = imread('balloons1.jpg');
I2 = imread('balloons2.jpg');

% point (x,y)
%  +------>  y
%  |
%  |
%  v
%  x
load points1.mat;
load points2.mat;

%%%% Try matching balloons1 with balloons3 if you did the optional task.
% I3 = imread('balloons3.jpg');
% load points3.mat;

%%%% YOU NEED TO IMPLEMENT THE SIFT FUNCTION
%[F1,Whalf1] = sift_example(I1,points1(:,1),points1(:,2));
%[F2,Whalf2] = sift_example(I2,points2(:,1),points2(:,2));

[F1,Whalf1] = sift_trial(I1,points1(:,1),points1(:,2));
[F2,Whalf2] = sift_trial(I2,points2(:,1),points2(:,2));
Whalf1
Whalf2

%%%%%

% Match the points based on feature vectors
k = size(points1,1);
M = match(F1,F2,k);

% Display
color = zeros(k,3);
figure; imshow(I1); hold on;
for i = 1:k
    index = M(i,1);
    color(index,:) = rand(1,3);
    colorcircle([points1(index,2),points1(index,1)],Whalf1(index),color(index,:),16);
end
hold off;

figure; imshow(I2); hold on;
for i = 1:k
    index = M(i,2);
    colorcircle([points2(index,2),points2(index,1)],Whalf2(index),color(M(i,1),:),16);
end
hold off;
