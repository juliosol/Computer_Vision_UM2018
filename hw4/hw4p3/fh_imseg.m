%
% W18 EECS 504 HW4p3 Image Segmentation with Minimum Spanning Forest
% Siyuan Chen, chsiyuan@umich.edu
%
% Segment an image using Felzenszwalb-Huttenlocher algorithm.
% This program reuses the following files in graph cut problem:
%   hisvec.m, reduce.m, RGB2Lab.m, segNeighbors.m, slic.m
% You need to implement F-H algorithm in fh.m
%

close all
im = im2double(imread('flower1.jpg'));
%im = im2double(imread('granos.jpg'));

% Generate superpixels
[seg_img,segments] = slic(im,144);

% Display
subplot 131
imagesc(im);
axis('off');
title('Original Image');

subplot 132;
cmap = 0.6*rand(length(segments),3)+0.4;
lambda=0.5;
h = imagesc(ind2rgb(seg_img,cmap)*lambda+im*(1-lambda));
axis('off');
title('Superpixels')

% Compute histogram for each superpixel
segments = reduce(im,segments,seg_img);

% Build adjacency matrix
adjacency = j_segNeighbors(seg_img);

% Normalization for distance calculation based on the image size
% For points (x1,y1) and (x2,y2), normalized distance D =
% ||(x1,y1)-(x2,y2)||^2/dnorm.
dnorm = sum(size(seg_img).^2);

% Construct graph
% Turn the adjacency matrix into an adjacency list G
% G is kx3 matrix, of which each row is an edge (node1, node2, weights). k
% is the number of edges. You should avoid duplicated edges.
%
% The edge weight is defined as the product of (1) difference of histograms
% (or equivalently, 1-intersection of histograms), and (2) inverse
% proximity exp(D(a,b)) with D specified as above.
% Notice the difference between the weights here and those in graph cuts.

%%%% FILL IN YOUR CODE HERE 
adj_size = size(adjacency);
ver_size = adj_size(1);
hor_size = adj_size(2);

%half_adjacency = zeros(hor_size,ver_size);
%for l = 1 : hor_size
%    if l == hor_size
%        half_adjacency(l,l) = adjacency(l,l);
%    end
%    half_adjacency(l,l+1:end) = adjacency(l,l+1:end);
%end

%k = find(half_adjacency > 0);
G = [];
G2 = [];
hist_vector = [];

for a = 1 : ver_size
    for b = 1 : hor_size
        if a <= b && adjacency(a,b) == 1
            weight = 0;
            %histintersect(segments(a).fv/norm(segments(a).fv),segments(b).fv/norm(segments(b).fv));
            histogram_difference = 1 - histintersect(segments(a).fv/sum(segments(a).fv),segments(b).fv/sum(segments(b).fv));
            %histogram_difference = 1 - sum(min(norm_seg_vector_a,norm_seg_vector_b)/norm(min(norm_seg_vector_a,norm_seg_vector_b)));
            %histogram_difference2 = 3 - sum(min(segments(a).fv,segments(b).fv));
            %histogram_difference = 3 - histintersect(segments(a).fv,segments(b).fv);
            point_1 = [segments(a).x; segments(a).y];
            point_2 = [segments(b).x; segments(b).y];
            spatial_proximity = exp((norm(point_1 - point_2))^2/dnorm);
            weight = histogram_difference * spatial_proximity;
            G = [G; [a,b,weight]];
        end
    end
end



%%%%


% Find minimum spanning forest from the graph
% Use a hyperparameter k = 4 for flower image.
%%%% YOU NEED TO IMPLEMENT THIS FUNCTION
seg_id = fh(G,1, seg_img);

%%%

% Display
for i = 1:length(segments)
    %i;
    seg_img(seg_img==i) = seg_id(i);
end
subplot 133;
imagesc(seg_img);
axis('off');
% cmap = 0.6*rand(length(unique(seg_id)),3)+0.4;
% imagesc(ind2rgb(seg_img,cmap)*lambda+im*(1-lambda));
title('Segmentation')

function c = histintersect(a,b)
    c = sum(min(a,b));
end