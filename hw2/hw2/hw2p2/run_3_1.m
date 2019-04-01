% EECS 504 HW2p2
% run_3_1.m

% checkerboard part
% this loads the checkerboard image and generates a corner response
im = double(imread('checkerboard.png'))/255;
re = j_harris(im,[],1);
figure; imagesc(re);
imwrite(re,'response_checkerboard.png');

% this code will call harris, it does post-processing to extract corners
% from the corner response image
X = detect(im,2,'harris',2,100,1,true);
show(im,X)
fig = gcf;
fig.PaperUnits = 'points';
fig.PaperPosition = [0 0 150 150];
print('detect_checkerboard.png','-dpng','-r0');

