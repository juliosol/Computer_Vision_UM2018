% EECS 504 HW2p2
% run_3_3.m

% rings part b
% this code will call harris, it does post-processing to extract corners
% from the corner response image
X = detect(im,3,'harris',1,0.1,5,true);
show(im,X)
fig = gcf;
fig.PaperUnits = 'points';
fig.PaperPosition = [0 0 150 150];
print('detect_rings.png','-dpng','-r0');
