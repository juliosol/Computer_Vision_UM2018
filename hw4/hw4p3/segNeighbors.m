function Bmap = segNeighbors(svMap)
%  
%  W18 EECS 504 HW4p2 Fg-bg Graph-cut
%  Jason Corso, jjcorso@umich.edu
%
%  Implement the code to compute the adjacency matrix for the superpixel graph
%  captured by svMap
%
%  INPUT:  svMap is an integer image with the value of each pixel being
%           the id of the superpixel with which it is associated
%  OUTPUT: Bmap is a binary adjacency matrix NxN (N being the number of superpixels
%           in svMap).  Bmap has a 1 in cell i,j if superpixel i and j are neighbors.
%           Otherwise, it has a 0.  Superpixels are neighbors if any of their
%           pixels are neighbors.

segmentList = unique(svMap);
segmentNum = length(segmentList);

matrix_im_dim = size(svMap);
ver_pixels = matrix_im_dim(1);
hor_pixels = matrix_im_dim(2);

ver_zero_vector = zeros(ver_pixels,1);
pre_padded_svMap = [ver_zero_vector, svMap, ver_zero_vector];
hor_zero_vector = zeros(1,hor_pixels + 2);
zero_padded_svMap = [hor_zero_vector; pre_padded_svMap; hor_zero_vector];

matrix_padded_im_dim = size(zero_padded_svMap);
ver_padded_pixels = matrix_padded_im_dim(1);
hor_padded_pixels = matrix_padded_im_dim(2);

%%% FILL IN THE CODE HERE to calculate the adjacency

Bmap = zeros(segmentNum);
for i = 1 : segmentNum
    %i
    pre_neighbors_i = [];
    for x = 2 : ver_padded_pixels - 1
        for y = 2 : hor_padded_pixels - 1
            if zero_padded_svMap(x,y) == i
               submatrix_i = zero_padded_svMap(x-1:x+1,y-1:y+1);
               pre_neighbors_i = [pre_neighbors_i; unique(submatrix_i)];
            end
        end
    end
    %i
    neighbors_i = unique(pre_neighbors_i);
    if neighbors_i(1) == 0
        trimmed_neighbors_i = neighbors_i(2:end);
    else
        trimmed_neighbors_i = neighbors_i;
    end
    %trimmed_neighbors_i
    for k = 1 : length(trimmed_neighbors_i)
        if i ~= trimmed_neighbors_i(k)
            Bmap(i,trimmed_neighbors_i(k)) = 1;
            %Bmap(trimmed_neighbors_i(k),i) = 1;
        end
    end
    Bmap(i,:);
end


%degree = node_degree(Bmap)

%%%
end

